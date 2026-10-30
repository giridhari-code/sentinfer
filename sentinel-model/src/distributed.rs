//! Multi-node / multi-worker inference (coordinator + workers over TCP).
//!
//! Protocol (line-oriented JSON-ish, length-prefixed UTF-8 frames):
//! * Worker registers: `{"type":"hello","worker_id":"..."}`
//! * Coordinator assigns: `{"type":"generate","id":N,"prompt":"...","max_tokens":...}`
//! * Worker replies: `{"type":"result","id":N,"text":"...","finish_reason":"..."}`
//!
//! Tensor-parallel weight sharding is out of scope; this is **data-parallel request routing**.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use sentinel_core::{Result, SentinelError};

use crate::config::EngineArgs;
use crate::llm::Llm;
use crate::request::SamplingParams;

fn frame_write(stream: &mut TcpStream, msg: &str) -> std::io::Result<()> {
    let bytes = msg.as_bytes();
    let len = (bytes.len() as u32).to_le_bytes();
    stream.write_all(&len)?;
    stream.write_all(bytes)?;
    stream.flush()
}

fn frame_read(stream: &mut TcpStream) -> std::io::Result<String> {
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf)?;
    let len = u32::from_le_bytes(len_buf) as usize;
    if len > 16 * 1024 * 1024 {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "frame too large"));
    }
    let mut buf = vec![0u8; len];
    stream.read_exact(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

fn json_get_str(body: &str, key: &str) -> Option<String> {
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
                out.push(n);
            }
        } else if c == '"' {
            return Some(out);
        } else {
            out.push(c);
        }
    }
    None
}

fn json_get_u64(body: &str, key: &str) -> Option<u64> {
    let pat = format!("\"{key}\"");
    let i = body.find(&pat)?;
    let rest = body[i + pat.len()..].trim_start().trim_start_matches(':').trim_start();
    let num: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    num.parse().ok()
}

fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n")
}

/// Worker process: load model, connect or listen, handle generate jobs.
pub struct Worker {
    llm: Mutex<Llm>,
    worker_id: String,
}

impl Worker {
    pub fn from_checkpoint(path: impl AsRef<std::path::Path>, worker_id: impl Into<String>) -> Result<Self> {
        let llm = Llm::from_checkpoint(path.as_ref(), EngineArgs::default())?;
        Ok(Self { llm: Mutex::new(llm), worker_id: worker_id.into() })
    }

    fn handle_job(&self, body: &str) -> String {
        let id = json_get_u64(body, "id").unwrap_or(0);
        let prompt = json_get_str(body, "prompt").unwrap_or_default();
        let max_tokens = json_get_u64(body, "max_tokens").unwrap_or(32) as usize;
        let temperature = {
            let pat = "\"temperature\"";
            body.find(pat)
                .and_then(|i| {
                    let rest = body[i + pat.len()..].trim_start().trim_start_matches(':').trim_start();
                    let num: String = rest
                        .chars()
                        .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == '-')
                        .collect();
                    num.parse().ok()
                })
                .unwrap_or(0.8)
        };
        let params = SamplingParams {
            max_tokens: max_tokens.max(1),
            temperature,
            ..Default::default()
        };
        match self.llm.lock() {
            Ok(mut llm) => match llm.generate(vec![prompt], params) {
                Ok(outs) => {
                    let o = &outs[0];
                    format!(
                        r#"{{"type":"result","id":{id},"text":"{}","finish_reason":"{}","prompt_tokens":{},"completion_tokens":{}}}"#,
                        json_escape(&o.text),
                        o.finish_reason,
                        o.prompt_tokens,
                        o.completion_tokens
                    )
                }
                Err(e) => format!(r#"{{"type":"error","id":{id},"error":"{}"}}"#, json_escape(&e.to_string())),
            },
            Err(_) => format!(r#"{{"type":"error","id":{id},"error":"lock"}}"#),
        }
    }

    /// Serve as TCP worker on `bind` (coordinator dials in).
    pub fn serve(&self, bind: &str) -> Result<()> {
        let listener = TcpListener::bind(bind).map_err(|e| SentinelError::InvalidState(e.to_string()))?;
        eprintln!("kLLM worker {} listening on {bind}", self.worker_id);
        for stream in listener.incoming() {
            let mut stream = stream.map_err(|e| SentinelError::InvalidState(e.to_string()))?;
            let hello = format!(r#"{{"type":"hello","worker_id":"{}"}}"#, json_escape(&self.worker_id));
            frame_write(&mut stream, &hello).map_err(|e| SentinelError::InvalidState(e.to_string()))?;
            loop {
                match frame_read(&mut stream) {
                    Ok(msg) => {
                        if msg.contains("\"type\":\"generate\"") {
                            let resp = self.handle_job(&msg);
                            if frame_write(&mut stream, &resp).is_err() {
                                break;
                            }
                        } else if msg.contains("\"type\":\"ping\"") {
                            let _ = frame_write(&mut stream, r#"{"type":"pong"}"#);
                        } else if msg.contains("\"type\":\"bye\"") {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
        }
        Ok(())
    }
}

/// Coordinator: fans out prompts across worker addresses (round-robin).
pub struct Coordinator {
    workers: Vec<String>,
    next: Mutex<usize>,
}

impl Coordinator {
    pub fn new(workers: Vec<String>) -> Self {
        Self { workers, next: Mutex::new(0) }
    }

    fn next_worker(&self) -> Result<String> {
        if self.workers.is_empty() {
            return Err(SentinelError::InvalidState("no workers configured".into()));
        }
        let mut n = self.next.lock().map_err(|_| SentinelError::InvalidState("lock".into()))?;
        let addr = self.workers[*n % self.workers.len()].clone();
        *n += 1;
        Ok(addr)
    }

    /// Send one generate job to the next worker; returns result JSON body.
    pub fn generate_remote(&self, id: u64, prompt: &str, max_tokens: usize, temperature: f64) -> Result<String> {
        let addr = self.next_worker()?;
        let mut stream = TcpStream::connect(&addr).map_err(|e| SentinelError::InvalidState(format!("connect {addr}: {e}")))?;
        stream
            .set_read_timeout(Some(Duration::from_secs(600)))
            .ok();
        // read hello
        let _ = frame_read(&mut stream).map_err(|e| SentinelError::InvalidState(e.to_string()))?;
        let job = format!(
            r#"{{"type":"generate","id":{id},"prompt":"{}","max_tokens":{max_tokens},"temperature":{temperature}}}"#,
            json_escape(prompt)
        );
        frame_write(&mut stream, &job).map_err(|e| SentinelError::InvalidState(e.to_string()))?;
        let resp = frame_read(&mut stream).map_err(|e| SentinelError::InvalidState(e.to_string()))?;
        let _ = frame_write(&mut stream, r#"{"type":"bye"}"#);
        Ok(resp)
    }

    /// Batch: one request per prompt, parallel across workers when multiple.
    pub fn generate_batch(&self, prompts: Vec<String>, max_tokens: usize, temperature: f64) -> Result<Vec<String>> {
        let workers = Arc::new(self.workers.clone());
        if workers.is_empty() {
            return Err(SentinelError::InvalidState("no workers".into()));
        }
        let mut handles = Vec::new();
        for (i, prompt) in prompts.into_iter().enumerate() {
            let addr = workers[i % workers.len()].clone();
            handles.push(thread::spawn(move || {
                let coord = Coordinator::new(vec![addr]);
                coord.generate_remote(i as u64, &prompt, max_tokens, temperature)
            }));
        }
        let mut outs = Vec::with_capacity(handles.len());
        for h in handles {
            let r = h.join().map_err(|_| SentinelError::InvalidState("worker thread panicked".into()))?;
            outs.push(r?);
        }
        Ok(outs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::seekv::SeekVConfig;
    use std::net::SocketAddr;

    #[test]
    fn coordinator_round_robin_empty_errors() {
        let c = Coordinator::new(vec![]);
        assert!(c.generate_remote(1, "hi", 4, 0.0).is_err());
    }

    #[test]
    fn worker_job_json_smoke() {
        let llm = Llm::from_random(SeekVConfig::tiny(), 1, EngineArgs::default()).unwrap();
        let w = Worker { llm: Mutex::new(llm), worker_id: "w0".into() };
        let resp = w.handle_job(r#"{"type":"generate","id":7,"prompt":"hi","max_tokens":2,"temperature":0.0}"#);
        assert!(resp.contains("\"type\":\"result\"") || resp.contains("\"type\":\"error\""));
        assert!(resp.contains("\"id\":7"));
    }
}
