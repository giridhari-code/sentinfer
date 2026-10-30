//! Rust bridge to DeepSeek vendor CUDA trees (`vendor/FlashMLA`, `DeepGEMM`, `DeepEP`).
//!
//! ## How this is wired
//!
//! 1. **Default (always on):** pure-Rust references — [`flash_mla_ref`], [`deep_gemm_nt`], [`moe_forward`].
//! 2. **Optional `.so`:** if env `SENTINEL_VENDOR_SO` points at a shared library that exports the
//!    C ABI in `vendor/rust_abi/sentinel_vendor_abi.h`, those symbols are used for FlashMLA / DeepGEMM.
//!
//! Upstream sources are **PyTorch C++ extensions**, not a stable C ABI. Shipping a thin CUDA
//! wrapper `.so` that implements `sentinel_*` is the supported way to attach real kernels without
//! linking libtorch into this crate.
//!
//! Feature `cuda-vendor` only gates the dlopen path; references always compile.

use std::sync::OnceLock;

use crate::error::{KernelError, Result};
use crate::flash_mla::{flash_mla_ref, MlaDims};
use crate::deep_gemm::deep_gemm_nt;
use crate::moe::{moe_forward, MoeWeights};

#[cfg(unix)]
mod dl {
    use std::ffi::CString;
    use std::os::raw::{c_char, c_int, c_void};

    pub type DeepGemmFn = unsafe extern "C" fn(
        *const f32,
        *const f32,
        *mut f32,
        i32,
        i32,
        i32,
    ) -> c_int;
    pub type FlashMlaFn = unsafe extern "C" fn(
        *const f32,
        *const f32,
        *const f32,
        *const f32,
        *mut f32,
        i32,
        i32,
        i32,
        i32,
        i32,
    ) -> c_int;

    #[repr(C)]
    pub struct Lib {
        handle: *mut c_void,
        pub deep_gemm: Option<DeepGemmFn>,
        pub flash_mla: Option<FlashMlaFn>,
    }

    unsafe impl Send for Lib {}
    unsafe impl Sync for Lib {}

    extern "C" {
        fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
        fn dlclose(handle: *mut c_void) -> c_int;
    }

    const RTLD_NOW: c_int = 2;

    pub unsafe fn try_load(path: &str) -> Option<Lib> {
        let c = CString::new(path).ok()?;
        let handle = dlopen(c.as_ptr(), RTLD_NOW);
        if handle.is_null() {
            return None;
        }
        let mut lib = Lib { handle, deep_gemm: None, flash_mla: None };
        let g = CString::new("sentinel_deep_gemm_nt_f32").ok()?;
        let f = CString::new("sentinel_flash_mla_f32").ok()?;
        let pg = dlsym(handle, g.as_ptr());
        let pf = dlsym(handle, f.as_ptr());
        if !pg.is_null() {
            lib.deep_gemm = Some(std::mem::transmute(pg));
        }
        if !pf.is_null() {
            lib.flash_mla = Some(std::mem::transmute(pf));
        }
        if lib.deep_gemm.is_none() && lib.flash_mla.is_none() {
            dlclose(handle);
            return None;
        }
        Some(lib)
    }

    impl Drop for Lib {
        fn drop(&mut self) {
            if !self.handle.is_null() {
                unsafe {
                    dlclose(self.handle);
                }
            }
        }
    }
}

struct VendorState {
    #[cfg(unix)]
    lib: Option<dl::Lib>,
    #[cfg(not(unix))]
    _pad: (),
}

fn state() -> &'static VendorState {
    static S: OnceLock<VendorState> = OnceLock::new();
    S.get_or_init(|| {
        #[cfg(unix)]
        {
            let lib = std::env::var("SENTINEL_VENDOR_SO")
                .ok()
                .and_then(|p| unsafe { dl::try_load(&p) });
            if lib.is_some() {
                eprintln!("sentinel-kernels: loaded CUDA vendor SO from SENTINEL_VENDOR_SO");
            }
            VendorState { lib }
        }
        #[cfg(not(unix))]
        {
            VendorState { _pad: () }
        }
    })
}

/// Report whether a vendor shared library is active.
pub fn vendor_so_loaded() -> bool {
    #[cfg(unix)]
    {
        state().lib.is_some()
    }
    #[cfg(not(unix))]
    {
        false
    }
}

/// DeepGEMM path: vendor SO if present, else [`deep_gemm_nt`].
pub fn vendor_deep_gemm_nt(x: &[f32], w: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>> {
    #[cfg(unix)]
    if let Some(lib) = state().lib.as_ref() {
        if let Some(f) = lib.deep_gemm {
            let mut out = vec![0.0f32; m * n];
            let rc = unsafe {
                f(
                    x.as_ptr(),
                    w.as_ptr(),
                    out.as_mut_ptr(),
                    m as i32,
                    k as i32,
                    n as i32,
                )
            };
            if rc == 0 {
                return Ok(out);
            }
            return Err(KernelError::Backend(format!("vendor deep_gemm rc={rc}")));
        }
    }
    deep_gemm_nt(x, w, m, k, n)
}

/// FlashMLA path: vendor SO if present, else [`flash_mla_ref`].
pub fn vendor_flash_mla(
    q: &[f32],
    c_kv: &[f32],
    w_uk: &[f32],
    w_uv: &[f32],
    dims: MlaDims,
) -> Result<Vec<f32>> {
    #[cfg(unix)]
    if let Some(lib) = state().lib.as_ref() {
        if let Some(f) = lib.flash_mla {
            let mut out = vec![0.0f32; dims.tokens * dims.num_heads * dims.head_dim];
            let rc = unsafe {
                f(
                    q.as_ptr(),
                    c_kv.as_ptr(),
                    w_uk.as_ptr(),
                    w_uv.as_ptr(),
                    out.as_mut_ptr(),
                    dims.tokens as i32,
                    dims.start_pos as i32,
                    dims.num_heads as i32,
                    dims.head_dim as i32,
                    dims.kv_lora_rank as i32,
                )
            };
            if rc == 0 {
                return Ok(out);
            }
            return Err(KernelError::Backend(format!("vendor flash_mla rc={rc}")));
        }
    }
    flash_mla_ref(q, c_kv, w_uk, w_uv, dims)
}

/// MoE: always Rust reference today (DeepEP is distributed/torch-heavy).
pub fn vendor_moe_forward(x: &[f32], tokens: usize, w: &MoeWeights) -> Result<Vec<f32>> {
    moe_forward(x, tokens, w)
}

/// Human-readable backend selection for logs.
pub fn vendor_backend_name() -> &'static str {
    if vendor_so_loaded() {
        "cuda-vendor-so"
    } else {
        "rust-reference"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemm_falls_back_to_reference() {
        let m = 4;
        let k = 8;
        let n = 3;
        let x: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.01).collect();
        let w: Vec<f32> = (0..n * k).map(|i| i as f32 * 0.02).collect();
        let a = vendor_deep_gemm_nt(&x, &w, m, k, n).unwrap();
        let b = deep_gemm_nt(&x, &w, m, k, n).unwrap();
        assert_eq!(a, b);
        assert_eq!(vendor_backend_name(), "rust-reference");
    }
}
