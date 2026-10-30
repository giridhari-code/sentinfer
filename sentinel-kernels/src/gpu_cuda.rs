//! CUDA backend via Driver API (`libcuda.so.1`) and embedded PTX.
//!
//! Elementwise kernels (`silu`, `add`) ship as PTX. GEMM uses a simple tiled PTX
//! kernel when the module loads; otherwise callers should fall back to OpenCL/CPU.
//! No CUDA toolkit required at **compile** time — only a working NVIDIA driver at runtime.

use std::ffi::{c_void, CString};
use std::sync::Arc;

use crate::cuda_ffi::*;
use crate::error::{KernelError, Result as KResult};

/// Minimal PTX: silu, add_inplace, and a naive matmul (one thread per output element).
const PTX: &str = r#"
.version 7.0
.target sm_50
.address_size 64

.visible .entry silu_inplace(
    .param .u64 silu_x,
    .param .u32 silu_n
)
{
    .reg .pred %p;
    .reg .u32 %r<4>;
    .reg .u64 %rd<4>;
    .reg .f32 %f<6>;

    ld.param.u64 %rd1, [silu_x];
    ld.param.u32 %r1, [silu_n];
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.u32 %r2, %r2, %r3, %tid.x;
    setp.ge.u32 %p, %r2, %r1;
    @%p bra $Lsilu_done;

    mul.wide.u32 %rd2, %r2, 4;
    add.s64 %rd3, %rd1, %rd2;
    ld.global.f32 %f1, [%rd3];
    neg.f32 %f2, %f1;
    ex2.approx.f32 %f3, %f2;
    mov.f32 %f4, 0f3F800000;
    add.f32 %f3, %f3, %f4;
    div.approx.f32 %f5, %f1, %f3;
    st.global.f32 [%rd3], %f5;
$Lsilu_done:
    ret;
}

.visible .entry add_inplace(
    .param .u64 add_x,
    .param .u64 add_r,
    .param .u32 add_n
)
{
    .reg .pred %p;
    .reg .u32 %r<4>;
    .reg .u64 %rd<6>;
    .reg .f32 %f<3>;

    ld.param.u64 %rd1, [add_x];
    ld.param.u64 %rd2, [add_r];
    ld.param.u32 %r1, [add_n];
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.u32 %r2, %r2, %r3, %tid.x;
    setp.ge.u32 %p, %r2, %r1;
    @%p bra $Ladd_done;

    mul.wide.u32 %rd3, %r2, 4;
    add.s64 %rd4, %rd1, %rd3;
    add.s64 %rd5, %rd2, %rd3;
    ld.global.f32 %f1, [%rd4];
    ld.global.f32 %f2, [%rd5];
    add.f32 %f1, %f1, %f2;
    st.global.f32 [%rd4], %f1;
$Ladd_done:
    ret;
}

// Naive matmul: out[m,n] = x[m,k] @ w[n,k]^T  (same layout as OpenCL matmul_nt)
.visible .entry matmul_nt(
    .param .u64 mat_x,
    .param .u64 mat_w,
    .param .u64 mat_o,
    .param .u32 mat_m,
    .param .u32 mat_k,
    .param .u32 mat_n
)
{
    .reg .pred %p;
    .reg .u32 %r<12>;
    .reg .u64 %rd<10>;
    .reg .f32 %f<6>;

    ld.param.u64 %rd1, [mat_x];
    ld.param.u64 %rd2, [mat_w];
    ld.param.u64 %rd3, [mat_o];
    ld.param.u32 %r4, [mat_m];
    ld.param.u32 %r5, [mat_k];
    ld.param.u32 %r6, [mat_n];

    mov.u32 %r7, %ctaid.x;
    mov.u32 %r8, %ntid.x;
    mad.lo.u32 %r7, %r7, %r8, %tid.x;   // row
    mov.u32 %r8, %ctaid.y;
    mov.u32 %r9, %ntid.y;
    mad.lo.u32 %r8, %r8, %r9, %tid.y;   // col

    setp.ge.u32 %p, %r7, %r4;
    @%p bra $Lmm_done;
    setp.ge.u32 %p, %r8, %r6;
    @%p bra $Lmm_done;

    mov.f32 %f1, 0f00000000;
    mov.u32 %r10, 0;
$Lmm_loop:
    setp.ge.u32 %p, %r10, %r5;
    @%p bra $Lmm_store;
    // x[row, t] = x[row*k + t]
    mad.lo.u32 %r11, %r7, %r5, %r10;
    mul.wide.u32 %rd4, %r11, 4;
    add.s64 %rd5, %rd1, %rd4;
    ld.global.f32 %f2, [%rd5];
    // w[col, t] = w[col*k + t]
    mad.lo.u32 %r11, %r8, %r5, %r10;
    mul.wide.u32 %rd4, %r11, 4;
    add.s64 %rd5, %rd2, %rd4;
    ld.global.f32 %f3, [%rd5];
    fma.rn.f32 %f1, %f2, %f3, %f1;
    add.u32 %r10, %r10, 1;
    bra $Lmm_loop;
$Lmm_store:
    mad.lo.u32 %r11, %r7, %r6, %r8;
    mul.wide.u32 %rd4, %r11, 4;
    add.s64 %rd5, %rd3, %rd4;
    st.global.f32 [%rd5], %f1;
$Lmm_done:
    ret;
}
"#;

struct CudaInner {
    api: CudaApi,
    ctx: CUcontext,
    module: CUmodule,
    device_name: String,
    silu: CUfunction,
    add: CUfunction,
    matmul: CUfunction,
}

unsafe impl Send for CudaInner {}
unsafe impl Sync for CudaInner {}

impl Drop for CudaInner {
    fn drop(&mut self) {
        unsafe {
            let _ = (self.api.cuModuleUnload)(self.module);
            let _ = (self.api.cuCtxDestroy_v2)(self.ctx);
        }
    }
}

impl CudaInner {
    fn try_open() -> std::result::Result<Arc<Self>, String> {
        let api = CudaApi::try_load()?;
        unsafe {
            check("cuInit", (api.cuInit)(0))?;
            let mut count: i32 = 0;
            check("cuDeviceGetCount", (api.cuDeviceGetCount)(&mut count))?;
            if count < 1 {
                return Err("no CUDA devices".into());
            }
            let mut dev: CUdevice = 0;
            check("cuDeviceGet", (api.cuDeviceGet)(&mut dev, 0))?;
            let mut name_buf = vec![0i8; 256];
            check(
                "cuDeviceGetName",
                (api.cuDeviceGetName)(name_buf.as_mut_ptr(), 255, dev),
            )?;
            let device_name = std::ffi::CStr::from_ptr(name_buf.as_ptr())
                .to_string_lossy()
                .into_owned();

            let mut ctx: CUcontext = std::ptr::null_mut();
            check("cuCtxCreate", (api.cuCtxCreate_v2)(&mut ctx, CU_CTX_SCHED_AUTO, dev))?;
            check("cuCtxSetCurrent", (api.cuCtxSetCurrent)(ctx))?;

            // PTX must be null-terminated for cuModuleLoadData
            let mut ptx_bytes = PTX.as_bytes().to_vec();
            ptx_bytes.push(0);
            let mut module: CUmodule = std::ptr::null_mut();
            check(
                "cuModuleLoadData",
                (api.cuModuleLoadData)(&mut module, ptx_bytes.as_ptr() as *const c_void),
            )?;

            let mut silu: CUfunction = std::ptr::null_mut();
            let mut add: CUfunction = std::ptr::null_mut();
            let mut matmul: CUfunction = std::ptr::null_mut();
            let silu_name = CString::new("silu_inplace").unwrap();
            let add_name = CString::new("add_inplace").unwrap();
            let mm_name = CString::new("matmul_nt").unwrap();
            check(
                "cuModuleGetFunction(silu)",
                (api.cuModuleGetFunction)(&mut silu, module, silu_name.as_ptr()),
            )?;
            check(
                "cuModuleGetFunction(add)",
                (api.cuModuleGetFunction)(&mut add, module, add_name.as_ptr()),
            )?;
            check(
                "cuModuleGetFunction(matmul)",
                (api.cuModuleGetFunction)(&mut matmul, module, mm_name.as_ptr()),
            )?;

            Ok(Arc::new(Self {
                api,
                ctx,
                module,
                device_name,
                silu,
                add,
                matmul,
            }))
        }
    }

    fn alloc(&self, n_bytes: usize) -> std::result::Result<CUdeviceptr, String> {
        let mut ptr: CUdeviceptr = 0;
        unsafe {
            check("cuMemAlloc", (self.api.cuMemAlloc_v2)(&mut ptr, n_bytes))?;
        }
        Ok(ptr)
    }

    fn free(&self, ptr: CUdeviceptr) {
        unsafe {
            let _ = (self.api.cuMemFree_v2)(ptr);
        }
    }

    fn h2d(&self, dst: CUdeviceptr, src: &[f32]) -> std::result::Result<(), String> {
        unsafe {
            check(
                "cuMemcpyHtoD",
                (self.api.cuMemcpyHtoD_v2)(dst, src.as_ptr() as *const c_void, src.len() * 4),
            )
        }
    }

    fn d2h(&self, dst: &mut [f32], src: CUdeviceptr) -> std::result::Result<(), String> {
        unsafe {
            check(
                "cuMemcpyDtoH",
                (self.api.cuMemcpyDtoH_v2)(dst.as_mut_ptr() as *mut c_void, src, dst.len() * 4),
            )
        }
    }

    fn sync(&self) -> std::result::Result<(), String> {
        unsafe { check("cuCtxSynchronize", (self.api.cuCtxSynchronize)()) }
    }
}

/// CUDA device handle used by [`super::GpuBackend`] when NVIDIA GPU is present.
pub struct CudaDevice {
    inner: Arc<CudaInner>,
}

impl CudaDevice {
    pub fn try_new() -> std::result::Result<Self, String> {
        Ok(Self {
            inner: CudaInner::try_open()?,
        })
    }

    pub fn device_name(&self) -> &str {
        &self.inner.device_name
    }

    pub fn matmul_nt(&self, x: &[f32], w: &[f32], m: usize, k: usize, n: usize) -> KResult<Vec<f32>> {
        if x.len() != m * k || w.len() != n * k {
            return Err(KernelError::Shape("cuda matmul_nt: bad lens".into()));
        }
        let d = &self.inner;
        unsafe {
            check("cuCtxSetCurrent", (d.api.cuCtxSetCurrent)(d.ctx))
                .map_err(KernelError::Unsupported)?;
        }
        let dx = d.alloc(x.len() * 4).map_err(KernelError::Unsupported)?;
        let dw = d.alloc(w.len() * 4).map_err(KernelError::Unsupported)?;
        let dout = d.alloc(m * n * 4).map_err(KernelError::Unsupported)?;
        d.h2d(dx, x).map_err(KernelError::Unsupported)?;
        d.h2d(dw, w).map_err(KernelError::Unsupported)?;

        let (mi, ki, ni) = (m as u32, k as u32, n as u32);
        let mut dx_p = dx;
        let mut dw_p = dw;
        let mut dout_p = dout;
        let mut mi_p = mi;
        let mut ki_p = ki;
        let mut ni_p = ni;
        let mut args: [*mut c_void; 6] = [
            &mut dx_p as *mut _ as *mut c_void,
            &mut dw_p as *mut _ as *mut c_void,
            &mut dout_p as *mut _ as *mut c_void,
            &mut mi_p as *mut _ as *mut c_void,
            &mut ki_p as *mut _ as *mut c_void,
            &mut ni_p as *mut _ as *mut c_void,
        ];

        let block_x = 16u32;
        let block_y = 16u32;
        let grid_x = (m as u32 + block_x - 1) / block_x;
        let grid_y = (n as u32 + block_y - 1) / block_y;

        unsafe {
            check(
                "cuLaunchKernel(matmul)",
                (d.api.cuLaunchKernel)(
                    d.matmul,
                    grid_x,
                    grid_y,
                    1,
                    block_x,
                    block_y,
                    1,
                    0,
                    std::ptr::null_mut(),
                    args.as_mut_ptr(),
                    std::ptr::null_mut(),
                ),
            )
            .map_err(KernelError::Unsupported)?;
        }
        d.sync().map_err(KernelError::Unsupported)?;
        let mut out = vec![0.0f32; m * n];
        d.d2h(&mut out, dout).map_err(KernelError::Unsupported)?;
        d.free(dx);
        d.free(dw);
        d.free(dout);
        Ok(out)
    }

    pub fn silu_inplace(&self, x: &mut [f32]) -> KResult<()> {
        let d = &self.inner;
        let n = x.len();
        unsafe {
            check("cuCtxSetCurrent", (d.api.cuCtxSetCurrent)(d.ctx))
                .map_err(KernelError::Unsupported)?;
        }
        let dx = d.alloc(n * 4).map_err(KernelError::Unsupported)?;
        d.h2d(dx, x).map_err(KernelError::Unsupported)?;
        let mut dx_p = dx;
        let mut n_p = n as u32;
        let mut args: [*mut c_void; 2] = [
            &mut dx_p as *mut _ as *mut c_void,
            &mut n_p as *mut _ as *mut c_void,
        ];
        let block = 256u32;
        let grid = (n as u32 + block - 1) / block;
        unsafe {
            check(
                "cuLaunchKernel(silu)",
                (d.api.cuLaunchKernel)(
                    d.silu,
                    grid,
                    1,
                    1,
                    block,
                    1,
                    1,
                    0,
                    std::ptr::null_mut(),
                    args.as_mut_ptr(),
                    std::ptr::null_mut(),
                ),
            )
            .map_err(KernelError::Unsupported)?;
        }
        d.sync().map_err(KernelError::Unsupported)?;
        d.d2h(x, dx).map_err(KernelError::Unsupported)?;
        d.free(dx);
        Ok(())
    }

    pub fn add_inplace(&self, x: &mut [f32], residual: &[f32]) -> KResult<()> {
        if x.len() != residual.len() {
            return Err(KernelError::Shape("cuda add_inplace: len mismatch".into()));
        }
        let d = &self.inner;
        let n = x.len();
        unsafe {
            check("cuCtxSetCurrent", (d.api.cuCtxSetCurrent)(d.ctx))
                .map_err(KernelError::Unsupported)?;
        }
        let dx = d.alloc(n * 4).map_err(KernelError::Unsupported)?;
        let dr = d.alloc(n * 4).map_err(KernelError::Unsupported)?;
        d.h2d(dx, x).map_err(KernelError::Unsupported)?;
        d.h2d(dr, residual).map_err(KernelError::Unsupported)?;
        let mut dx_p = dx;
        let mut dr_p = dr;
        let mut n_p = n as u32;
        let mut args: [*mut c_void; 3] = [
            &mut dx_p as *mut _ as *mut c_void,
            &mut dr_p as *mut _ as *mut c_void,
            &mut n_p as *mut _ as *mut c_void,
        ];
        let block = 256u32;
        let grid = (n as u32 + block - 1) / block;
        unsafe {
            check(
                "cuLaunchKernel(add)",
                (d.api.cuLaunchKernel)(
                    d.add,
                    grid,
                    1,
                    1,
                    block,
                    1,
                    1,
                    0,
                    std::ptr::null_mut(),
                    args.as_mut_ptr(),
                    std::ptr::null_mut(),
                ),
            )
            .map_err(KernelError::Unsupported)?;
        }
        d.sync().map_err(KernelError::Unsupported)?;
        d.d2h(x, dx).map_err(KernelError::Unsupported)?;
        d.free(dx);
        d.free(dr);
        Ok(())
    }
}

pub fn try_default_cuda() -> Option<CudaDevice> {
    match CudaDevice::try_new() {
        Ok(d) => Some(d),
        Err(e) => {
            eprintln!("[sentinel-kernels] CUDA unavailable: {e}");
            None
        }
    }
}
