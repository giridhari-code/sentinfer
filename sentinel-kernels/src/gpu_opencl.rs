//! Real GPU kernels via OpenCL (runtime-loaded `libOpenCL.so.1`).
//!
//! When a GPU/accelerator platform is present the heavy ops (`matmul_nt`, `rms_norm`,
//! `silu`, `softmax_rows`, `rope`) run as OpenCL C kernels on the device. If no platform
//! is available the backend reports `is_device_accelerated() == false` and the caller
//! should use [`crate::ParallelCpuBackend`].

use std::ffi::CString;
use std::os::raw::c_void;
use std::sync::Arc;

use crate::error::{KernelError, Result as KResult};
use crate::opencl_ffi::*;

const KERNEL_SRC: &str = r#"
__kernel void matmul_nt(
    __global const float* x,
    __global const float* w,
    __global float* out,
    const int m, const int k, const int n)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    if (i >= m || j >= n) return;
    float acc = 0.0f;
    int c = 0;
    for (; c + 3 < k; c += 4) {
        acc += x[i*k+c]*w[j*k+c]
             + x[i*k+c+1]*w[j*k+c+1]
             + x[i*k+c+2]*w[j*k+c+2]
             + x[i*k+c+3]*w[j*k+c+3];
    }
    for (; c < k; ++c) acc += x[i*k+c]*w[j*k+c];
    out[i*n+j] = acc;
}

__kernel void rms_norm(
    __global const float* x,
    __global const float* weight,
    __global float* out,
    const int rows, const int hidden, const float eps)
{
    int r = get_global_id(0);
    if (r >= rows) return;
    float ss = 0.0f;
    for (int c = 0; c < hidden; ++c) {
        float v = x[r*hidden+c];
        ss += v*v;
    }
    float inv = 1.0f / sqrt(ss / (float)hidden + eps);
    for (int c = 0; c < hidden; ++c)
        out[r*hidden+c] = weight[c] * x[r*hidden+c] * inv;
}

__kernel void silu_inplace(__global float* x, const int n) {
    int i = get_global_id(0);
    if (i >= n) return;
    float v = x[i];
    x[i] = v / (1.0f + exp(-v));
}

__kernel void softmax_rows(
    __global float* v, const int rows, const int cols)
{
    int r = get_global_id(0);
    if (r >= rows) return;
    __global float* row = v + r*cols;
    float mx = row[0];
    for (int c = 1; c < cols; ++c) mx = fmax(mx, row[c]);
    float sum = 0.0f;
    for (int c = 0; c < cols; ++c) {
        row[c] = exp(row[c] - mx);
        sum += row[c];
    }
    float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    for (int c = 0; c < cols; ++c) row[c] *= inv;
}


__kernel void attention_causal(
    __global const float* q,
    __global const float* k_cache,
    __global const float* v_cache,
    __global float* attn,
    const int tokens, const int start_pos,
    const int num_heads, const int num_kv_heads, const int head_dim)
{
    int ti = get_global_id(0);
    int head = get_global_id(1);
    if (ti >= tokens || head >= num_heads) return;
    int group = num_heads / num_kv_heads;
    int kvh = head / group;
    int kv_stride = num_kv_heads * head_dim;
    int pos = start_pos + ti;
    float scale = 1.0f / sqrt((float)head_dim);
    __global const float* qv = q + (ti * num_heads + head) * head_dim;
    // scores on private memory limited — use iterative softmax in two passes via local max
    float mx = -INFINITY;
    for (int j = 0; j <= pos; ++j) {
        __global const float* kj = k_cache + j * kv_stride + kvh * head_dim;
        float dot = 0.0f;
        for (int c = 0; c < head_dim; ++c) dot += qv[c] * kj[c];
        float s = dot * scale;
        if (s > mx) mx = s;
    }
    float sum = 0.0f;
    // accumulate weighted V
    __global float* out = attn + (ti * num_heads + head) * head_dim;
    for (int c = 0; c < head_dim; ++c) out[c] = 0.0f;
    for (int j = 0; j <= pos; ++j) {
        __global const float* kj = k_cache + j * kv_stride + kvh * head_dim;
        float dot = 0.0f;
        for (int c = 0; c < head_dim; ++c) dot += qv[c] * kj[c];
        float wgt = exp(dot * scale - mx);
        sum += wgt;
        __global const float* vj = v_cache + j * kv_stride + kvh * head_dim;
        for (int c = 0; c < head_dim; ++c) out[c] += wgt * vj[c];
    }
    float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    for (int c = 0; c < head_dim; ++c) out[c] *= inv;
}

__kernel void rope_inplace(
    __global float* x,
    const int tokens, const int heads, const int head_dim,
    const int start_pos, const float theta)
{
    int t = get_global_id(0);
    int h = get_global_id(1);
    if (t >= tokens || h >= heads) return;
    int half = head_dim / 2;
    int base = (t * heads + h) * head_dim;
    float pos = (float)(start_pos + t);
    for (int i = 0; i < half; ++i) {
        float freq = pow(theta, -((float)(2*i)) / (float)head_dim);
        float angle = pos * freq;
        float s = sin(angle);
        float c = cos(angle);
        float x0 = x[base + i];
        float x1 = x[base + i + half];
        x[base + i] = x0 * c - x1 * s;
        x[base + i + half] = x1 * c + x0 * s;
    }
}
"#;

struct ClDevice {
    api: OpenClApi,
    context: cl_context,
    queue: cl_command_queue,
    device: cl_device_id,
    program: cl_program,
    device_name: String,
}

unsafe impl Send for ClDevice {}
unsafe impl Sync for ClDevice {}

impl Drop for ClDevice {
    fn drop(&mut self) {
        unsafe {
            (self.api.clReleaseProgram)(self.program);
            (self.api.clReleaseCommandQueue)(self.queue);
            (self.api.clReleaseContext)(self.context);
        }
    }
}

impl ClDevice {
    fn try_open() -> std::result::Result<Arc<Self>, String> {
        let api = OpenClApi::try_load()?;
        unsafe {
            let mut nplat: cl_uint = 0;
            let err = (api.clGetPlatformIDs)(0, std::ptr::null_mut(), &mut nplat);
            if err != CL_SUCCESS || nplat == 0 {
                return Err(format!("no OpenCL platforms (err={err})"));
            }
            let mut plats = vec![std::ptr::null_mut(); nplat as usize];
            (api.clGetPlatformIDs)(nplat, plats.as_mut_ptr(), std::ptr::null_mut());

            let mut chosen_dev: cl_device_id = std::ptr::null_mut();
            let mut chosen_plat: cl_platform_id = std::ptr::null_mut();
            // Prefer GPU devices, else any device.
            for &want in &[CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_ALL] {
                for &p in &plats {
                    let mut nd: cl_uint = 0;
                    if (api.clGetDeviceIDs)(p, want, 0, std::ptr::null_mut(), &mut nd) != CL_SUCCESS || nd == 0 {
                        continue;
                    }
                    let mut devs = vec![std::ptr::null_mut(); nd as usize];
                    (api.clGetDeviceIDs)(p, want, nd, devs.as_mut_ptr(), std::ptr::null_mut());
                    chosen_dev = devs[0];
                    chosen_plat = p;
                    break;
                }
                if !chosen_dev.is_null() {
                    break;
                }
            }
            if chosen_dev.is_null() {
                return Err("no OpenCL devices".into());
            }

            let mut err = CL_SUCCESS;
            let ctx = (api.clCreateContext)(
                std::ptr::null(),
                1,
                &chosen_dev,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &mut err,
            );
            if err != CL_SUCCESS || ctx.is_null() {
                return Err(format!("clCreateContext failed: {err}"));
            }
            let queue = (api.clCreateCommandQueue)(ctx, chosen_dev, 0, &mut err);
            if err != CL_SUCCESS || queue.is_null() {
                (api.clReleaseContext)(ctx);
                return Err(format!("clCreateCommandQueue failed: {err}"));
            }

            let src = CString::new(KERNEL_SRC).unwrap();
            let mut src_ptr = src.as_ptr();
            let program = (api.clCreateProgramWithSource)(ctx, 1, &mut src_ptr, std::ptr::null(), &mut err);
            if err != CL_SUCCESS || program.is_null() {
                (api.clReleaseCommandQueue)(queue);
                (api.clReleaseContext)(ctx);
                return Err(format!("clCreateProgramWithSource failed: {err}"));
            }
            let build = (api.clBuildProgram)(program, 1, &chosen_dev, std::ptr::null(), std::ptr::null_mut(), std::ptr::null_mut());
            if build != CL_SUCCESS {
                let mut log_size = 0usize;
                (api.clGetProgramBuildInfo)(program, chosen_dev, CL_PROGRAM_BUILD_LOG, 0, std::ptr::null_mut(), &mut log_size);
                let mut log = vec![0u8; log_size.max(1)];
                (api.clGetProgramBuildInfo)(
                    program,
                    chosen_dev,
                    CL_PROGRAM_BUILD_LOG,
                    log.len(),
                    log.as_mut_ptr() as *mut c_void,
                    std::ptr::null_mut(),
                );
                let msg = String::from_utf8_lossy(&log).to_string();
                (api.clReleaseProgram)(program);
                (api.clReleaseCommandQueue)(queue);
                (api.clReleaseContext)(ctx);
                return Err(format!("clBuildProgram failed: {build}: {msg}"));
            }

            let mut name = vec![0u8; 256];
            (api.clGetDeviceInfo)(chosen_dev, CL_DEVICE_NAME, name.len(), name.as_mut_ptr() as *mut c_void, std::ptr::null_mut());
            let device_name = String::from_utf8_lossy(&name).trim_end_matches('\0').trim().to_string();
            let _ = chosen_plat;

            Ok(Arc::new(Self { api, context: ctx, queue, device: chosen_dev, program, device_name }))
        }
    }

    fn kernel(&self, name: &str) -> std::result::Result<cl_kernel, String> {
        let cname = CString::new(name).unwrap();
        let mut err = CL_SUCCESS;
        let k = unsafe { (self.api.clCreateKernel)(self.program, cname.as_ptr(), &mut err) };
        if err != CL_SUCCESS || k.is_null() {
            return Err(format!("clCreateKernel({name}): {err}"));
        }
        Ok(k)
    }

    fn buffer_from(&self, data: &[f32], flags: cl_mem_flags) -> std::result::Result<cl_mem, String> {
        let mut err = CL_SUCCESS;
        let bytes = data.len() * 4;
        let ptr = data.as_ptr() as *mut c_void;
        let mem = unsafe {
            (self.api.clCreateBuffer)(self.context, flags | CL_MEM_COPY_HOST_PTR, bytes, ptr, &mut err)
        };
        if err != CL_SUCCESS || mem.is_null() {
            return Err(format!("clCreateBuffer: {err}"));
        }
        Ok(mem)
    }

    fn buffer_empty(&self, n: usize, flags: cl_mem_flags) -> std::result::Result<cl_mem, String> {
        let mut err = CL_SUCCESS;
        let mem = unsafe {
            (self.api.clCreateBuffer)(self.context, flags, n * 4, std::ptr::null_mut(), &mut err)
        };
        if err != CL_SUCCESS || mem.is_null() {
            return Err(format!("clCreateBuffer empty: {err}"));
        }
        Ok(mem)
    }

    fn read(&self, mem: cl_mem, n: usize) -> std::result::Result<Vec<f32>, String> {
        let mut out = vec![0.0f32; n];
        let err = unsafe {
            (self.api.clEnqueueReadBuffer)(
                self.queue,
                mem,
                CL_TRUE,
                0,
                n * 4,
                out.as_mut_ptr() as *mut c_void,
                0,
                std::ptr::null(),
                std::ptr::null_mut(),
            )
        };
        if err != CL_SUCCESS {
            return Err(format!("clEnqueueReadBuffer: {err}"));
        }
        Ok(out)
    }

    fn release_mem(&self, mem: cl_mem) {
        unsafe { (self.api.clReleaseMemObject)(mem); }
    }
    fn release_kernel(&self, k: cl_kernel) {
        unsafe { (self.api.clReleaseKernel)(k); }
    }
}

/// GPU context holder shared by [`super::GpuBackend`].
pub struct GpuDevice {
    inner: Arc<ClDevice>,
}

impl GpuDevice {
    pub fn try_new() -> std::result::Result<Self, String> {
        Ok(Self { inner: ClDevice::try_open()? })
    }

    pub fn device_name(&self) -> &str {
        &self.inner.device_name
    }

    pub fn matmul_nt(&self, x: &[f32], w: &[f32], m: usize, k: usize, n: usize) -> KResult<Vec<f32>> {
        if x.len() != m * k || w.len() != n * k {
            return Err(KernelError::Shape(format!("matmul_nt gpu: bad lens")));
        }
        let d = &self.inner;
        let kx = d.buffer_from(x, CL_MEM_READ_ONLY).map_err(|e| KernelError::Unsupported(e))?;
        let kw = d.buffer_from(w, CL_MEM_READ_ONLY).map_err(|e| KernelError::Unsupported(e))?;
        let ko = d.buffer_empty(m * n, CL_MEM_WRITE_ONLY).map_err(|e| KernelError::Unsupported(e))?;
        let kernel = d.kernel("matmul_nt").map_err(|e| KernelError::Unsupported(e))?;
        let (mi, ki, ni) = (m as i32, k as i32, n as i32);
        unsafe {
            (d.api.clSetKernelArg)(kernel, 0, std::mem::size_of::<cl_mem>(), &kx as *const _ as *const c_void);
            (d.api.clSetKernelArg)(kernel, 1, std::mem::size_of::<cl_mem>(), &kw as *const _ as *const c_void);
            (d.api.clSetKernelArg)(kernel, 2, std::mem::size_of::<cl_mem>(), &ko as *const _ as *const c_void);
            (d.api.clSetKernelArg)(kernel, 3, 4, &mi as *const _ as *const c_void);
            (d.api.clSetKernelArg)(kernel, 4, 4, &ki as *const _ as *const c_void);
            (d.api.clSetKernelArg)(kernel, 5, 4, &ni as *const _ as *const c_void);
            let global = [m.max(1), n.max(1)];
            let err = (d.api.clEnqueueNDRangeKernel)(
                d.queue, kernel, 2, std::ptr::null(), global.as_ptr(), std::ptr::null(), 0, std::ptr::null(), std::ptr::null_mut(),
            );
            if err != CL_SUCCESS {
                d.release_kernel(kernel);
                d.release_mem(kx); d.release_mem(kw); d.release_mem(ko);
                return Err(KernelError::Unsupported(format!("enqueue matmul: {err}")));
            }
            (d.api.clFinish)(d.queue);
        }
        let out = d.read(ko, m * n).map_err(|e| KernelError::Unsupported(e))?;
        d.release_kernel(kernel);
        d.release_mem(kx); d.release_mem(kw); d.release_mem(ko);
        Ok(out)
    }

    pub fn rms_norm(&self, x: &[f32], weight: &[f32], rows: usize, hidden: usize, eps: f32) -> KResult<Vec<f32>> {
        let d = &self.inner;
        let kx = d.buffer_from(x, CL_MEM_READ_ONLY).map_err(|e| KernelError::Unsupported(e))?;
        let kw = d.buffer_from(weight, CL_MEM_READ_ONLY).map_err(|e| KernelError::Unsupported(e))?;
        let ko = d.buffer_empty(rows * hidden, CL_MEM_WRITE_ONLY).map_err(|e| KernelError::Unsupported(e))?;
        let kernel = d.kernel("rms_norm").map_err(|e| KernelError::Unsupported(e))?;
        let (r, h) = (rows as i32, hidden as i32);
        unsafe {
            (d.api.clSetKernelArg)(kernel, 0, std::mem::size_of::<cl_mem>(), &kx as *const _ as *const c_void);
            (d.api.clSetKernelArg)(kernel, 1, std::mem::size_of::<cl_mem>(), &kw as *const _ as *const c_void);
            (d.api.clSetKernelArg)(kernel, 2, std::mem::size_of::<cl_mem>(), &ko as *const _ as *const c_void);
            (d.api.clSetKernelArg)(kernel, 3, 4, &r as *const _ as *const c_void);
            (d.api.clSetKernelArg)(kernel, 4, 4, &h as *const _ as *const c_void);
            (d.api.clSetKernelArg)(kernel, 5, 4, &eps as *const _ as *const c_void);
            let global = [rows.max(1)];
            let err = (d.api.clEnqueueNDRangeKernel)(
                d.queue, kernel, 1, std::ptr::null(), global.as_ptr(), std::ptr::null(), 0, std::ptr::null(), std::ptr::null_mut(),
            );
            if err != CL_SUCCESS {
                d.release_kernel(kernel);
                d.release_mem(kx); d.release_mem(kw); d.release_mem(ko);
                return Err(KernelError::Unsupported(format!("enqueue rms: {err}")));
            }
            (d.api.clFinish)(d.queue);
        }
        let out = d.read(ko, rows * hidden).map_err(|e| KernelError::Unsupported(e))?;
        d.release_kernel(kernel);
        d.release_mem(kx); d.release_mem(kw); d.release_mem(ko);
        Ok(out)
    }

    pub fn silu_inplace(&self, x: &mut [f32]) -> KResult<()> {
        let d = &self.inner;
        let n = x.len();
        let kx = d.buffer_from(x, CL_MEM_READ_WRITE).map_err(|e| KernelError::Unsupported(e))?;
        let kernel = d.kernel("silu_inplace").map_err(|e| KernelError::Unsupported(e))?;
        let ni = n as i32;
        unsafe {
            (d.api.clSetKernelArg)(kernel, 0, std::mem::size_of::<cl_mem>(), &kx as *const _ as *const c_void);
            (d.api.clSetKernelArg)(kernel, 1, 4, &ni as *const _ as *const c_void);
            let global = [n.max(1)];
            let err = (d.api.clEnqueueNDRangeKernel)(
                d.queue, kernel, 1, std::ptr::null(), global.as_ptr(), std::ptr::null(), 0, std::ptr::null(), std::ptr::null_mut(),
            );
            if err != CL_SUCCESS {
                d.release_kernel(kernel);
                d.release_mem(kx);
                return Err(KernelError::Unsupported(format!("enqueue silu: {err}")));
            }
            (d.api.clFinish)(d.queue);
        }
        let out = d.read(kx, n).map_err(|e| KernelError::Unsupported(e))?;
        x.copy_from_slice(&out);
        d.release_kernel(kernel);
        d.release_mem(kx);
        Ok(())
    }

    pub fn softmax_rows_inplace(&self, v: &mut [f32], rows: usize, cols: usize) -> KResult<()> {
        let d = &self.inner;
        let kx = d.buffer_from(v, CL_MEM_READ_WRITE).map_err(|e| KernelError::Unsupported(e))?;
        let kernel = d.kernel("softmax_rows").map_err(|e| KernelError::Unsupported(e))?;
        let (r, c) = (rows as i32, cols as i32);
        unsafe {
            (d.api.clSetKernelArg)(kernel, 0, std::mem::size_of::<cl_mem>(), &kx as *const _ as *const c_void);
            (d.api.clSetKernelArg)(kernel, 1, 4, &r as *const _ as *const c_void);
            (d.api.clSetKernelArg)(kernel, 2, 4, &c as *const _ as *const c_void);
            let global = [rows.max(1)];
            let err = (d.api.clEnqueueNDRangeKernel)(
                d.queue, kernel, 1, std::ptr::null(), global.as_ptr(), std::ptr::null(), 0, std::ptr::null(), std::ptr::null_mut(),
            );
            if err != CL_SUCCESS {
                d.release_kernel(kernel);
                d.release_mem(kx);
                return Err(KernelError::Unsupported(format!("enqueue softmax: {err}")));
            }
            (d.api.clFinish)(d.queue);
        }
        let out = d.read(kx, rows * cols).map_err(|e| KernelError::Unsupported(e))?;
        v.copy_from_slice(&out);
        d.release_kernel(kernel);
        d.release_mem(kx);
        Ok(())
    }

    pub fn rope_inplace(
        &self,
        x: &mut [f32],
        tokens: usize,
        heads: usize,
        head_dim: usize,
        start_pos: usize,
        theta: f32,
    ) -> KResult<()> {
        let d = &self.inner;
        let kx = d.buffer_from(x, CL_MEM_READ_WRITE).map_err(|e| KernelError::Unsupported(e))?;
        let kernel = d.kernel("rope_inplace").map_err(|e| KernelError::Unsupported(e))?;
        let (t, h, hd, sp) = (tokens as i32, heads as i32, head_dim as i32, start_pos as i32);
        unsafe {
            (d.api.clSetKernelArg)(kernel, 0, std::mem::size_of::<cl_mem>(), &kx as *const _ as *const c_void);
            (d.api.clSetKernelArg)(kernel, 1, 4, &t as *const _ as *const c_void);
            (d.api.clSetKernelArg)(kernel, 2, 4, &h as *const _ as *const c_void);
            (d.api.clSetKernelArg)(kernel, 3, 4, &hd as *const _ as *const c_void);
            (d.api.clSetKernelArg)(kernel, 4, 4, &sp as *const _ as *const c_void);
            (d.api.clSetKernelArg)(kernel, 5, 4, &theta as *const _ as *const c_void);
            let global = [tokens.max(1), heads.max(1)];
            let err = (d.api.clEnqueueNDRangeKernel)(
                d.queue, kernel, 2, std::ptr::null(), global.as_ptr(), std::ptr::null(), 0, std::ptr::null(), std::ptr::null_mut(),
            );
            if err != CL_SUCCESS {
                d.release_kernel(kernel);
                d.release_mem(kx);
                return Err(KernelError::Unsupported(format!("enqueue rope: {err}")));
            }
            (d.api.clFinish)(d.queue);
        }
        let out = d.read(kx, tokens * heads * head_dim).map_err(|e| KernelError::Unsupported(e))?;
        x.copy_from_slice(&out);
        d.release_kernel(kernel);
        d.release_mem(kx);
        Ok(())
    }

    pub fn attention_causal(
        &self,
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        tokens: usize,
        start_pos: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> KResult<Vec<f32>> {
        let d = &self.inner;
        let total = start_pos + tokens;
        let kv_stride = num_kv_heads * head_dim;
        let kq = d.buffer_from(q, CL_MEM_READ_ONLY).map_err(|e| KernelError::Unsupported(e))?;
        let kk = d.buffer_from(k_cache, CL_MEM_READ_ONLY).map_err(|e| KernelError::Unsupported(e))?;
        let kv = d.buffer_from(v_cache, CL_MEM_READ_ONLY).map_err(|e| KernelError::Unsupported(e))?;
        let ko = d.buffer_empty(tokens * num_heads * head_dim, CL_MEM_WRITE_ONLY).map_err(|e| KernelError::Unsupported(e))?;
        let kernel = d.kernel("attention_causal").map_err(|e| KernelError::Unsupported(e))?;
        let (t, sp, nh, nkv, hd) = (
            tokens as i32,
            start_pos as i32,
            num_heads as i32,
            num_kv_heads as i32,
            head_dim as i32,
        );
        unsafe {
            (d.api.clSetKernelArg)(kernel, 0, std::mem::size_of::<cl_mem>(), &kq as *const _ as *const c_void);
            (d.api.clSetKernelArg)(kernel, 1, std::mem::size_of::<cl_mem>(), &kk as *const _ as *const c_void);
            (d.api.clSetKernelArg)(kernel, 2, std::mem::size_of::<cl_mem>(), &kv as *const _ as *const c_void);
            (d.api.clSetKernelArg)(kernel, 3, std::mem::size_of::<cl_mem>(), &ko as *const _ as *const c_void);
            (d.api.clSetKernelArg)(kernel, 4, 4, &t as *const _ as *const c_void);
            (d.api.clSetKernelArg)(kernel, 5, 4, &sp as *const _ as *const c_void);
            (d.api.clSetKernelArg)(kernel, 6, 4, &nh as *const _ as *const c_void);
            (d.api.clSetKernelArg)(kernel, 7, 4, &nkv as *const _ as *const c_void);
            (d.api.clSetKernelArg)(kernel, 8, 4, &hd as *const _ as *const c_void);
            let global = [tokens.max(1), num_heads.max(1)];
            let err = (d.api.clEnqueueNDRangeKernel)(
                d.queue, kernel, 2, std::ptr::null(), global.as_ptr(), std::ptr::null(), 0, std::ptr::null(), std::ptr::null_mut(),
            );
            if err != CL_SUCCESS {
                d.release_kernel(kernel);
                d.release_mem(kq); d.release_mem(kk); d.release_mem(kv); d.release_mem(ko);
                return Err(KernelError::Unsupported(format!("enqueue attention: {err}")));
            }
            (d.api.clFinish)(d.queue);
        }
        let out = d.read(ko, tokens * num_heads * head_dim).map_err(|e| KernelError::Unsupported(e))?;
        d.release_kernel(kernel);
        d.release_mem(kq); d.release_mem(kk); d.release_mem(kv); d.release_mem(ko);
        let _ = (total, kv_stride);
        Ok(out)
    }
}

/// Global lazy device open attempt.
pub fn try_default_device() -> Option<GpuDevice> {
    GpuDevice::try_new().ok()
}
