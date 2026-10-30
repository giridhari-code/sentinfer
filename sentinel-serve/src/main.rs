//! **kLLM** — production-oriented full-Rust Seek-V server.
//!
//!   GET  /health  /ready  /version  /models
//!   POST /generate  /generate/stream
//!
//! Flags: --api-key · --max-concurrent · --log-requests · --log-stats

use std::io::{Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use sentinel_model::{EngineArgs, Llm, SamplingParams};

const NAME: &str = "kLLM";
const VERSION: &str = env!("CARGO_PKG_VERSION");
const MAX_BODY: usize = 4 * 1024 * 1024;
const MAX_PROMPTS: usize = 32;
const MAX_PROMPT_BYTES: usize = 256 * 1024;

const USAGE: &str = "\
kLLM — full-Rust Seek-V inference server

usage: kllm --checkpoint FILE.skv [options]

  --host ADDR              (default 0.0.0.0)
  --port N                 (default 8080)
  --api-key SECRET         require Authorization: Bearer SECRET
  --max-concurrent N       in-flight HTTP handlers (default 32)
  --log-requests           log method path status ms
  --log-stats              engine step stats
  --vocab FILE             PieceTokenizer vocab (match model vocab_size)
  --quant int8|int4         load .skv as quantized runtime (or load .skq)
";

struct App {
    llm: Mutex<Llm>,
    model_id: String,
    api_key: Option<String>,
    max_concurrent: usize,
    in_flight: AtomicUsize,
    log_requests: bool,
    shutting_down: AtomicBool,
    requests_total: AtomicUsize,
    generate_total: AtomicUsize,
    stream_total: AtomicUsize,
    errors_total: AtomicUsize,
    prompt_tokens_total: AtomicUsize,
    completion_tokens_total: AtomicUsize,
}

fn now_ts() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

struct Cli {
    checkpoint: PathBuf,
    vocab: Option<PathBuf>,
    quant: Option<String>,
    host: String,
    port: u16,
    api_key: Option<String>,
    max_concurrent: usize,
    log_requests: bool,
    log_stats: bool,
}

fn parse_args() -> Result<Cli, String> {
    let mut checkpoint = None;
    let mut vocab = None;
    let mut quant = None;
    let mut host = "0.0.0.0".to_string();
    let mut port = 8080u16;
    let mut api_key = None;
    let mut max_concurrent = 32usize;
    let mut log_requests = false;
    let mut log_stats = false;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--checkpoint" => checkpoint = Some(it.next().ok_or("--checkpoint needs a value")?.into()),
            "--vocab" => vocab = Some(it.next().ok_or("--vocab needs a value")?.into()),
            "--quant" => quant = Some(it.next().ok_or("--quant needs int8|int4")?),
            "--host" => host = it.next().ok_or("--host needs a value")?,
            "--port" => port = it.next().ok_or("--port needs a value")?.parse().map_err(|_| "bad --port")?,
            "--api-key" => api_key = Some(it.next().ok_or("--api-key needs a value")?),
            "--max-concurrent" => {
                max_concurrent = it
                    .next()
                    .ok_or("--max-concurrent needs a value")?
                    .parse()
                    .map_err(|_| "bad --max-concurrent")?;
                if max_concurrent == 0 {
                    return Err("--max-concurrent must be >= 1".into());
                }
            }
            "--log-requests" => log_requests = true,
            "--log-stats" => log_stats = true,
            "-h" | "--help" => return Err(USAGE.into()),
            other => return Err(format!("unknown `{other}`\n\n{USAGE}")),
        }
    }
    let mut api_key = api_key;
    if api_key.is_none() {
        if let Ok(v) = std::env::var("KLLM_API_KEY") {
            if !v.is_empty() {
                api_key = Some(v);
            }
        }
    }
    Ok(Cli {
        checkpoint: checkpoint.ok_or_else(|| format!("--checkpoint required\n\n{USAGE}"))?,
        vocab,
        quant,
        host,
        port,
        api_key,
        max_concurrent,
        log_requests,
        log_stats,
    })
}

fn find_headers_end(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n").map(|i| i + 4)
}

struct HttpRequest {
    method: String,
    path: String,
    body: String,
    authorization: Option<String>,
}

fn read_request(stream: &mut TcpStream) -> Result<HttpRequest, String> {
    stream.set_read_timeout(Some(std::time::Duration::from_secs(120))).ok();
    stream.set_write_timeout(Some(std::time::Duration::from_secs(120))).ok();
    let mut buf = Vec::new();
    let mut tmp = [0u8; 4096];
    loop {
        let n = stream.read(&mut tmp).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&tmp[..n]);
        if let Some(pos) = find_headers_end(&buf) {
            let header = String::from_utf8_lossy(&buf[..pos]).into_owned();
            let mut content_length = 0usize;
            let mut authorization = None;
            for line in header.lines().skip(1) {
                let lower = line.to_ascii_lowercase();
                if let Some(v) = lower.strip_prefix("content-length:") {
                    content_length = v.trim().parse().unwrap_or(0);
                }
                if lower.starts_with("authorization:") {
                    authorization = Some(line[line.find(':').unwrap_or(0) + 1..].trim().to_string());
                }
            }
            if content_length > MAX_BODY {
                return Err(format!("body too large (max {MAX_BODY} bytes)"));
            }
            while buf.len() < pos + content_length {
                let n = stream.read(&mut tmp).map_err(|e| e.to_string())?;
                if n == 0 {
                    break;
                }
                buf.extend_from_slice(&tmp[..n]);
                if buf.len() > MAX_BODY + 64 * 1024 {
                    return Err("request too large".into());
                }
            }
            let body = String::from_utf8_lossy(&buf[pos..pos + content_length.min(buf.len() - pos)]).into_owned();
            let req_line = header.lines().next().unwrap_or("");
            let mut parts = req_line.split_whitespace();
            let method = parts.next().unwrap_or("").to_string();
            let path = parts.next().unwrap_or("/").split('?').next().unwrap_or("/").to_string();
            return Ok(HttpRequest { method, path, body, authorization });
        }
        if buf.len() > MAX_BODY {
            return Err("request too large".into());
        }
    }
    Err("incomplete request".into())
}

fn respond(stream: &mut TcpStream, status: u16, reason: &str, body: &str, content_type: &str) {
    let header = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\nServer: {NAME}/{VERSION}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    let _ = stream.write_all(header.as_bytes());
    let _ = stream.write_all(body.as_bytes());
}

fn respond_json(stream: &mut TcpStream, status: u16, body: &str) {
    let reason = match status {
        200 => "OK",
        401 => "Unauthorized",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        429 => "Too Many Requests",
        503 => "Service Unavailable",
        500 => "Internal Server Error",
        _ => "Error",
    };
    respond(stream, status, reason, body, "application/json; charset=utf-8");
}

fn json_escape(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if c.is_control() => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o
}

fn extract_string(body: &str, key: &str) -> Option<String> {
    let pat = format!("\"{key}\"");
    let i = body.find(&pat)?;
    let rest = body[i + pat.len()..].trim_start().trim_start_matches(':').trim_start();
    if !rest.starts_with('"') {
        return None;
    }
    let mut out = String::new();
    let mut chars = rest[1..].chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            if let Some(n) = chars.next() {
                out.push(match n {
                    'n' => '\n',
                    't' => '\t',
                    'r' => '\r',
                    '"' => '"',
                    '\\' => '\\',
                    other => other,
                });
            }
        } else if c == '"' {
            return Some(out);
        } else {
            out.push(c);
        }
    }
    None
}

fn extract_number(body: &str, key: &str) -> Option<f64> {
    let pat = format!("\"{key}\"");
    let i = body.find(&pat)?;
    let rest = body[i + pat.len()..].trim_start().trim_start_matches(':').trim_start();
    let num: String = rest
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == '-' || *c == 'e' || *c == 'E' || *c == '+')
        .collect();
    num.parse().ok()
}

fn extract_bool(body: &str, key: &str) -> Option<bool> {
    let pat = format!("\"{key}\"");
    let i = body.find(&pat)?;
    let rest = body[i + pat.len()..].trim_start().trim_start_matches(':').trim_start();
    if rest.starts_with("true") {
        Some(true)
    } else if rest.starts_with("false") {
        Some(false)
    } else {
        None
    }
}

fn extract_prompts(body: &str) -> Result<Vec<String>, String> {
    if let Some(idx) = body.find("\"prompt\"") {
        let rest = body[idx + 8..].trim_start().trim_start_matches(':').trim_start();
        if rest.starts_with('[') {
            let mut prompts = Vec::new();
            let mut s = &rest[1..];
            while let Some(q) = s.find('"') {
                s = &s[q..];
                let chunk = format!("\"p\":{s}");
                if let Some(val) = extract_string(&chunk, "p") {
                    prompts.push(val);
                    let bytes = s.as_bytes();
                    let mut i = 1;
                    while i < bytes.len() {
                        if bytes[i] == b'\\' {
                            i += 2;
                            continue;
                        }
                        if bytes[i] == b'"' {
                            s = &s[i + 1..];
                            break;
                        }
                        i += 1;
                    }
                    if prompts.len() > MAX_PROMPTS {
                        return Err(format!("too many prompts (max {MAX_PROMPTS})"));
                    }
                    continue;
                }
                break;
            }
            if prompts.is_empty() {
                return Err("prompt array is empty".into());
            }
            for p in &prompts {
                if p.len() > MAX_PROMPT_BYTES {
                    return Err(format!("prompt exceeds {MAX_PROMPT_BYTES} bytes"));
                }
            }
            return Ok(prompts);
        }
    }
    let one = extract_string(body, "prompt").ok_or_else(|| "field `prompt` (string or array) is required".to_string())?;
    if one.is_empty() {
        return Err("prompt must not be empty".into());
    }
    if one.len() > MAX_PROMPT_BYTES {
        return Err(format!("prompt exceeds {MAX_PROMPT_BYTES} bytes"));
    }
    Ok(vec![one])
}

fn sampling_from_body(body: &str) -> SamplingParams {
    SamplingParams {
        max_tokens: extract_number(body, "max_tokens").map(|n| n as usize).unwrap_or(64).max(1).min(4096),
        temperature: extract_number(body, "temperature").unwrap_or(0.8),
        top_p: extract_number(body, "top_p").unwrap_or(0.95),
        top_k: extract_number(body, "top_k").map(|n| n as i32).unwrap_or(40),
        seed: extract_number(body, "seed").map(|n| n as u64).unwrap_or(1),
        ignore_eos: extract_bool(body, "ignore_eos").unwrap_or(false),
        ..Default::default()
    }
}

fn err_json(msg: &str) -> String {
    format!(r#"{{"ok":false,"error":"{}"}}"#, json_escape(msg))
}

fn check_auth(app: &App, authorization: &Option<String>) -> Result<(), (u16, String)> {
    let Some(secret) = app.api_key.as_ref() else {
        return Ok(());
    };
    let Some(header) = authorization.as_ref() else {
        return Err((401, "missing Authorization header (Bearer token required)".into()));
    };
    let token = header
        .strip_prefix("Bearer ")
        .or_else(|| header.strip_prefix("bearer "))
        .unwrap_or(header.as_str())
        .trim();
    if token != secret {
        return Err((401, "invalid API key".into()));
    }
    Ok(())
}

fn result_json(app: &App, prompts: &[String], outs: &[sentinel_model::RequestOutput], steps: u64) -> String {
    if outs.len() == 1 {
        let o = &outs[0];
        return format!(
            r#"{{"ok":true,"id":"kllm-{}","model":"{}","prompt":"{}","text":"{}","output_token_ids":[{}],"finish_reason":"{}","usage":{{"prompt_tokens":{},"completion_tokens":{},"total_tokens":{}}},"engine":{{"steps":{},"name":"{NAME}","version":"{VERSION}"}}}}"#,
            now_ts(),
            json_escape(&app.model_id),
            json_escape(&prompts[0]),
            json_escape(&o.text),
            o.output_token_ids.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(","),
            o.finish_reason,
            o.prompt_tokens,
            o.completion_tokens,
            o.prompt_tokens + o.completion_tokens,
            steps
        );
    }
    let mut parts = Vec::new();
    for (i, o) in outs.iter().enumerate() {
        parts.push(format!(
            r#"{{"index":{},"prompt":"{}","text":"{}","output_token_ids":[{}],"finish_reason":"{}","usage":{{"prompt_tokens":{},"completion_tokens":{},"total_tokens":{}}}}}"#,
            i,
            json_escape(prompts.get(i).map(|s| s.as_str()).unwrap_or("")),
            json_escape(&o.text),
            o.output_token_ids.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(","),
            o.finish_reason,
            o.prompt_tokens,
            o.completion_tokens,
            o.prompt_tokens + o.completion_tokens
        ));
    }
    format!(
        r#"{{"ok":true,"id":"kllm-{}","model":"{}","results":[{}],"engine":{{"steps":{},"name":"{NAME}","version":"{VERSION}"}}}}"#,
        now_ts(),
        json_escape(&app.model_id),
        parts.join(","),
        steps
    )
}

fn handle_generate(app: &App, body: &str) -> Result<String, (u16, String)> {
    let prompts = extract_prompts(body).map_err(|e| (400u16, e))?;
    let params = sampling_from_body(body);
    let mut llm = app.llm.lock().map_err(|_| (500u16, "engine lock poisoned".into()))?;
    let outs = llm.generate(prompts.clone(), params).map_err(|e| (400u16, e.to_string()))?;
    for o in &outs {
        app.prompt_tokens_total.fetch_add(o.prompt_tokens, Ordering::Relaxed);
        app.completion_tokens_total.fetch_add(o.completion_tokens, Ordering::Relaxed);
    }
    let steps = llm.stats().steps;
    Ok(result_json(app, &prompts, &outs, steps))
}

fn write_sse(stream: &mut TcpStream, event: &str, data: &str) -> Result<(), String> {
    let mut msg = String::new();
    if !event.is_empty() {
        msg.push_str("event: ");
        msg.push_str(event);
        msg.push('\n');
    }
    for line in data.lines() {
        msg.push_str("data: ");
        msg.push_str(line);
        msg.push('\n');
    }
    msg.push('\n');
    stream.write_all(msg.as_bytes()).map_err(|e| e.to_string())?;
    stream.flush().map_err(|e| e.to_string())
}

fn handle_generate_stream(app: &App, stream: &mut TcpStream, body: &str) {
    let prompts = match extract_prompts(body) {
        Ok(p) => p,
        Err(e) => {
            respond_json(stream, 400, &err_json(&e));
            return;
        }
    };
    let params = sampling_from_body(body);
    let header = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\nCache-Control: no-cache\r\nAccess-Control-Allow-Origin: *\r\nServer: {NAME}/{VERSION}\r\nConnection: close\r\n\r\n"
    );
    if stream.write_all(header.as_bytes()).is_err() {
        return;
    }
    let _ = stream.flush();

    let n_prompts = prompts.len();
    let _ = write_sse(
        stream,
        "start",
        &format!(
            r#"{{"ok":true,"n":{},"model":"{}"}}"#,
            n_prompts,
            json_escape(&app.model_id)
        ),
    );

    let mut llm = match app.llm.lock() {
        Ok(g) => g,
        Err(_) => {
            let _ = write_sse(stream, "error", r#"{"ok":false,"error":"engine lock"}"#);
            return;
        }
    };

    let gen = llm.generate_stream(prompts, params, 1_000_000, |d| {
        let payload = if d.finished {
            format!(
                r#"{{"index":{},"request_id":"{}","token_ids":[{}],"text":"{}","text_so_far":"{}","finished":true,"finish_reason":"{}"}}"#,
                d.index,
                json_escape(&d.request_id),
                d.token_ids.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(","),
                json_escape(&d.text),
                json_escape(&d.text_so_far),
                d.finish_reason.as_deref().unwrap_or("")
            )
        } else {
            format!(
                r#"{{"index":{},"request_id":"{}","token_ids":[{}],"text":"{}","text_so_far":"{}","finished":false}}"#,
                d.index,
                json_escape(&d.request_id),
                d.token_ids.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(","),
                json_escape(&d.text),
                json_escape(&d.text_so_far)
            )
        };
        let _ = write_sse(stream, "token", &payload);
    });

    match gen {
        Ok(outs) => {
            let mut parts = Vec::new();
            for (i, o) in outs.iter().enumerate() {
                parts.push(format!(
                    r#"{{"index":{},"finish_reason":"{}","prompt_tokens":{},"completion_tokens":{},"text":"{}"}}"#,
                    i,
                    o.finish_reason,
                    o.prompt_tokens,
                    o.completion_tokens,
                    json_escape(&o.text)
                ));
            }
            let summary = format!(
                r#"{{"ok":true,"n":{},"engine_steps":{},"results":[{}]}}"#,
                outs.len(),
                llm.stats().steps,
                parts.join(",")
            );
            let _ = write_sse(stream, "done", &summary);
        }
        Err(e) => {
            let _ = write_sse(stream, "error", &format!(r#"{{"ok":false,"error":"{}"}}"#, json_escape(&e.to_string())));
        }
    }
}

fn handle(app: &App, stream: &mut TcpStream) {
    let started = Instant::now();
    app.requests_total.fetch_add(1, Ordering::Relaxed);
    if app.shutting_down.load(Ordering::Relaxed) {
        respond_json(stream, 503, &err_json("server shutting down"));
        return;
    }

    let req = match read_request(stream) {
        Ok(v) => v,
        Err(e) => {
            respond_json(stream, 400, &err_json(&e));
            return;
        }
    };

    // Public endpoints (no auth): health, ready
    let public = matches!(req.path.as_str(), "/health" | "/ready" | "/gpu" | "/metrics");
    if !public {
        if let Err((code, msg)) = check_auth(app, &req.authorization) {
            respond_json(stream, code, &err_json(&msg));
            if app.log_requests {
                eprintln!(
                    "[req] {} {} -> {} ({:.1}ms)",
                    req.method,
                    req.path,
                    code,
                    started.elapsed().as_secs_f64() * 1000.0
                );
            }
            return;
        }
    }

    if req.method == "OPTIONS" {
        respond(stream, 200, "OK", "{}", "application/json");
        return;
    }

    let mut status = 200u16;
    match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/") => {
            respond_json(
                stream,
                200,
                &format!(
                    r#"{{"name":"{NAME}","version":"{VERSION}","api":["/health","/ready","/version","/models","/gpu","/metrics","/generate","/generate/stream"],"auth":{},"max_concurrent":{}}}"#,
                    if app.api_key.is_some() { "true" } else { "false" },
                    app.max_concurrent
                ),
            );
        }
        ("GET", "/metrics") => {
            let body = format!(
                "# HELP kllm_requests_total Total HTTP requests\n# TYPE kllm_requests_total counter\nkllm_requests_total {}\n\
# HELP kllm_generate_total Generate calls\n# TYPE kllm_generate_total counter\nkllm_generate_total {}\n\
# HELP kllm_stream_total Stream calls\n# TYPE kllm_stream_total counter\nkllm_stream_total {}\n\
# HELP kllm_errors_total Error responses\n# TYPE kllm_errors_total counter\nkllm_errors_total {}\n\
# HELP kllm_prompt_tokens_total Prompt tokens\n# TYPE kllm_prompt_tokens_total counter\nkllm_prompt_tokens_total {}\n\
# HELP kllm_completion_tokens_total Completion tokens\n# TYPE kllm_completion_tokens_total counter\nkllm_completion_tokens_total {}\n\
# HELP kllm_in_flight In-flight handlers\n# TYPE kllm_in_flight gauge\nkllm_in_flight {}\n",
                app.requests_total.load(Ordering::Relaxed),
                app.generate_total.load(Ordering::Relaxed),
                app.stream_total.load(Ordering::Relaxed),
                app.errors_total.load(Ordering::Relaxed),
                app.prompt_tokens_total.load(Ordering::Relaxed),
                app.completion_tokens_total.load(Ordering::Relaxed),
                app.in_flight.load(Ordering::Relaxed),
            );
            respond(stream, 200, "OK", &body, "text/plain; version=0.0.4");
        }

        ("GET", "/health") => respond_json(stream, 200, r#"{"ok":true,"status":"healthy"}"#),
        ("GET", "/ready") => {
            if app.shutting_down.load(Ordering::Relaxed) {
                status = 503;
                respond_json(stream, 503, &err_json("not ready: shutting down"));
            } else {
                let inflight = app.in_flight.load(Ordering::Relaxed);
                respond_json(
                    stream,
                    200,
                    &format!(
                        r#"{{"ok":true,"ready":true,"in_flight":{},"max_concurrent":{}}}"#,
                        inflight, app.max_concurrent
                    ),
                );
            }
        }
        ("GET", "/version") => {
            respond_json(
                stream,
                200,
                &format!(
                    r#"{{"name":"{NAME}","version":"{VERSION}","engine":"sentinel","production":true,"language":"rust"}}"#
                ),
            );
        }
        ("GET", "/gpu") => {
            use sentinel_kernels::{Backend, GpuBackend};
            let g = GpuBackend::new();
            let accel = g.is_device_accelerated();
            let name = g.name();
            let device = g.device_name().unwrap_or("");
            let x = vec![1.0f32, 0.0, 0.0, 1.0];
            let w = vec![1.0f32, 0.0, 0.0, 1.0];
            let verify = match g.matmul_nt(&x, &w, 2, 2, 2) {
                Ok(out) => format!(
                    r#"{{"ok":true,"matmul_2x2":[{:.4},{:.4},{:.4},{:.4}]}}"#,
                    out[0], out[1], out[2], out[3]
                ),
                Err(e) => format!(r#"{{"ok":false,"error":"{}"}}"#, json_escape(&e.to_string())),
            };
            respond_json(
                stream,
                200,
                &format!(
                    r#"{{"ok":true,"backend":"{}","accelerated":{},"device":"{}","verify":{}}}"#,
                    json_escape(name),
                    if accel { "true" } else { "false" },
                    json_escape(device),
                    verify
                ),
            );
        }

        ("GET", "/models") => {
            respond_json(
                stream,
                200,
                &format!(
                    r#"{{"ok":true,"models":[{{"id":"{}","format":"skv","arch":"seek-v"}}]}}"#,
                    json_escape(&app.model_id)
                ),
            );
        }
        ("POST", "/generate") => {
            if extract_bool(&req.body, "stream").unwrap_or(false) {
                app.stream_total.fetch_add(1, Ordering::Relaxed);
                handle_generate_stream(app, stream, &req.body);
            } else {
                app.generate_total.fetch_add(1, Ordering::Relaxed);
                match handle_generate(app, &req.body) {
                    Ok(j) => respond_json(stream, 200, &j),
                    Err((code, e)) => {
                        status = code;
                        respond_json(stream, code, &err_json(&e));
                    }
                }
            }
        }
        ("POST", "/generate/stream") => {
            app.stream_total.fetch_add(1, Ordering::Relaxed);
            handle_generate_stream(app, stream, &req.body);
        }
        _ if req.method != "GET" && req.method != "POST" => {
            status = 405;
            respond_json(stream, 405, &err_json("method not allowed"));
        }
        _ => {
            status = 404;
            respond_json(stream, 404, &err_json(&format!("unknown path {}", req.path)));
        }
    }

    if app.log_requests {
        eprintln!(
            "[req] {} {} -> {} ({:.1}ms)",
            req.method,
            req.path,
            status,
            started.elapsed().as_secs_f64() * 1000.0
        );
    }
}

fn main() {
    let cli = match parse_args() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };
    let mut args = EngineArgs::default();
    args.log_stats = cli.log_stats;
    let llm = {
        let path = &cli.checkpoint;
        let is_skq = path.extension().and_then(|e| e.to_str()) == Some("skq");
        let result = if let Some(ref q) = cli.quant {
            let scheme = match q.as_str() {
                "int8" | "q8" => sentinel_kernels::QuantScheme::Int8,
                "int4" | "q4" => sentinel_kernels::QuantScheme::Int4,
                other => {
                    eprintln!("kLLM: unknown --quant {other} (use int8|int4)");
                    std::process::exit(2);
                }
            };
            Llm::from_checkpoint_quant(path, scheme, cli.vocab.as_deref(), args)
        } else if is_skq {
            Llm::from_checkpoint_quant(path, sentinel_kernels::QuantScheme::Int8, cli.vocab.as_deref(), args)
        } else {
            Llm::from_checkpoint_with_vocab(path, cli.vocab.as_deref(), args)
        };
        result.unwrap_or_else(|e| {
            eprintln!("kLLM: failed to load checkpoint: {e}");
            std::process::exit(1);
        })
    };
    let model_id = cli.checkpoint.file_stem().and_then(|s| s.to_str()).unwrap_or("seekv").to_string();
    let app = Arc::new(App {
        llm: Mutex::new(llm),
        model_id: model_id.clone(),
        api_key: cli.api_key.clone(),
        max_concurrent: cli.max_concurrent,
        in_flight: AtomicUsize::new(0),
        log_requests: cli.log_requests,
        shutting_down: AtomicBool::new(false),
        requests_total: AtomicUsize::new(0),
        generate_total: AtomicUsize::new(0),
        stream_total: AtomicUsize::new(0),
        errors_total: AtomicUsize::new(0),
        prompt_tokens_total: AtomicUsize::new(0),
        completion_tokens_total: AtomicUsize::new(0),
    });

    // Global stop flag for SIGINT / SIGTERM (Unix).
    static STOP: AtomicBool = AtomicBool::new(false);
    #[cfg(unix)]
    {
        extern "C" fn on_signal(_: i32) {
            STOP.store(true, Ordering::SeqCst);
        }
        unsafe {
            libc_signal(15, on_signal as usize); // SIGTERM
            libc_signal(2, on_signal as usize); // SIGINT
        }
    }

    let addr = format!("{}:{}", cli.host, cli.port);
    let listener = TcpListener::bind(&addr).unwrap_or_else(|e| {
        eprintln!("kLLM: bind {addr}: {e}");
        std::process::exit(1);
    });
    listener.set_nonblocking(true).ok();

    eprintln!("{NAME} v{VERSION}  production");
    eprintln!("  listen         http://{addr}");
    eprintln!("  model          {model_id}");
    eprintln!("  auth           {}", if cli.api_key.is_some() { "API key required" } else { "off" });
    eprintln!("  max_concurrent {}", cli.max_concurrent);
    eprintln!("  POST           /generate  /generate/stream");

    loop {
        if STOP.load(Ordering::SeqCst) {
            eprintln!("kLLM: shutdown signal — draining in-flight…");
            app.shutting_down.store(true, Ordering::SeqCst);
            // wait up to 30s for in-flight
            for _ in 0..300 {
                if app.in_flight.load(Ordering::SeqCst) == 0 {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
            eprintln!("kLLM: bye");
            break;
        }
        match listener.accept() {
            Ok((mut stream, _)) => {
                let inflight = app.in_flight.load(Ordering::SeqCst);
                if inflight >= app.max_concurrent {
                    respond_json(&mut stream, 429, &err_json("too many concurrent requests"));
                    let _ = stream.shutdown(Shutdown::Both);
                    continue;
                }
                if app.shutting_down.load(Ordering::SeqCst) {
                    respond_json(&mut stream, 503, &err_json("server shutting down"));
                    continue;
                }
                app.in_flight.fetch_add(1, Ordering::SeqCst);
                let app = Arc::clone(&app);
                std::thread::spawn(move || {
                    handle(&app, &mut stream);
                    app.in_flight.fetch_sub(1, Ordering::SeqCst);
                });
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            Err(e) => eprintln!("accept: {e}"),
        }
    }
}

#[cfg(unix)]
extern "C" {
    #[link_name = "signal"]
    fn libc_signal(sig: i32, handler: usize) -> usize;
}
