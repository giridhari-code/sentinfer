//! Minimal CUDA Driver API bindings via `dlopen("libcuda.so.1")`.
//! No CUDA SDK at compile time — same pattern as OpenCL.

use std::ffi::{c_char, c_int, c_uint, c_void, CString};
use std::ptr;

pub type CUresult = c_int;
pub type CUdevice = c_int;
pub type CUcontext = *mut c_void;
pub type CUmodule = *mut c_void;
pub type CUfunction = *mut c_void;
pub type CUstream = *mut c_void;
pub type CUdeviceptr = u64;

pub const CUDA_SUCCESS: CUresult = 0;
pub const CU_CTX_SCHED_AUTO: c_uint = 0;

#[allow(non_snake_case)]
pub struct CudaApi {
    _lib: LibHolder,
    pub cuInit: unsafe extern "C" fn(c_uint) -> CUresult,
    pub cuDeviceGetCount: unsafe extern "C" fn(*mut c_int) -> CUresult,
    pub cuDeviceGet: unsafe extern "C" fn(*mut CUdevice, c_int) -> CUresult,
    pub cuDeviceGetName: unsafe extern "C" fn(*mut c_char, c_int, CUdevice) -> CUresult,
    pub cuCtxCreate_v2: unsafe extern "C" fn(*mut CUcontext, c_uint, CUdevice) -> CUresult,
    pub cuCtxDestroy_v2: unsafe extern "C" fn(CUcontext) -> CUresult,
    pub cuCtxSetCurrent: unsafe extern "C" fn(CUcontext) -> CUresult,
    pub cuMemAlloc_v2: unsafe extern "C" fn(*mut CUdeviceptr, usize) -> CUresult,
    pub cuMemFree_v2: unsafe extern "C" fn(CUdeviceptr) -> CUresult,
    pub cuMemcpyHtoD_v2: unsafe extern "C" fn(CUdeviceptr, *const c_void, usize) -> CUresult,
    pub cuMemcpyDtoH_v2: unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize) -> CUresult,
    pub cuModuleLoadData: unsafe extern "C" fn(*mut CUmodule, *const c_void) -> CUresult,
    pub cuModuleUnload: unsafe extern "C" fn(CUmodule) -> CUresult,
    pub cuModuleGetFunction: unsafe extern "C" fn(*mut CUfunction, CUmodule, *const c_char) -> CUresult,
    pub cuLaunchKernel: unsafe extern "C" fn(
        CUfunction,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        CUstream,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> CUresult,
    pub cuCtxSynchronize: unsafe extern "C" fn() -> CUresult,
}

struct LibHolder(*mut c_void);
unsafe impl Send for LibHolder {}
unsafe impl Sync for LibHolder {}
impl Drop for LibHolder {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { libc_dlclose(self.0) };
        }
    }
}

extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
}

const RTLD_NOW: c_int = 2;

unsafe fn libc_dlclose(h: *mut c_void) {
    let _ = dlclose(h);
}

unsafe fn load_sym(lib: *mut c_void, name: &str) -> Result<*mut c_void, String> {
    let c = CString::new(name).unwrap();
    let p = dlsym(lib, c.as_ptr());
    if p.is_null() {
        Err(format!("dlsym({name}) failed"))
    } else {
        Ok(p)
    }
}

macro_rules! load_fn {
    ($lib:expr, $name:ident) => {{
        let p = load_sym($lib, stringify!($name))?;
        std::mem::transmute(p)
    }};
}

impl CudaApi {
    pub fn try_load() -> Result<Self, String> {
        unsafe {
            let path = CString::new("libcuda.so.1").unwrap();
            let mut lib = dlopen(path.as_ptr(), RTLD_NOW);
            if lib.is_null() {
                let path2 = CString::new("libcuda.so").unwrap();
                lib = dlopen(path2.as_ptr(), RTLD_NOW);
            }
            if lib.is_null() {
                return Err("libcuda.so.1 / libcuda.so not found (install NVIDIA driver)".into());
            }
            Ok(Self {
                cuInit: load_fn!(lib, cuInit),
                cuDeviceGetCount: load_fn!(lib, cuDeviceGetCount),
                cuDeviceGet: load_fn!(lib, cuDeviceGet),
                cuDeviceGetName: load_fn!(lib, cuDeviceGetName),
                cuCtxCreate_v2: load_fn!(lib, cuCtxCreate_v2),
                cuCtxDestroy_v2: load_fn!(lib, cuCtxDestroy_v2),
                cuCtxSetCurrent: load_fn!(lib, cuCtxSetCurrent),
                cuMemAlloc_v2: load_fn!(lib, cuMemAlloc_v2),
                cuMemFree_v2: load_fn!(lib, cuMemFree_v2),
                cuMemcpyHtoD_v2: load_fn!(lib, cuMemcpyHtoD_v2),
                cuMemcpyDtoH_v2: load_fn!(lib, cuMemcpyDtoH_v2),
                cuModuleLoadData: load_fn!(lib, cuModuleLoadData),
                cuModuleUnload: load_fn!(lib, cuModuleUnload),
                cuModuleGetFunction: load_fn!(lib, cuModuleGetFunction),
                cuLaunchKernel: load_fn!(lib, cuLaunchKernel),
                cuCtxSynchronize: load_fn!(lib, cuCtxSynchronize),
                _lib: LibHolder(lib),
            })
        }
    }
}

pub fn check(api_name: &str, r: CUresult) -> Result<(), String> {
    if r == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(format!("{api_name} failed with CUDA error {r}"))
    }
}

/// Null pointer helper for optional args.
pub fn null_mut_void() -> *mut c_void {
    ptr::null_mut()
}
