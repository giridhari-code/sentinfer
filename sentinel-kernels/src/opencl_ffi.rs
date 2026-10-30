#![allow(non_snake_case)]
//! Minimal OpenCL 1.2 FFI loaded at runtime via `libOpenCL.so.1` (no system headers required).

#![allow(non_camel_case_types, dead_code)]

use std::os::raw::{c_char, c_void};

pub type cl_int = i32;
pub type cl_uint = u32;
pub type cl_ulong = u64;
pub type cl_bool = cl_uint;
pub type cl_bitfield = cl_ulong;
pub type cl_device_type = cl_bitfield;
pub type cl_mem_flags = cl_bitfield;
pub type cl_command_queue_properties = cl_bitfield;
pub type cl_program_build_info = cl_uint;
pub type cl_platform_info = cl_uint;
pub type cl_device_info = cl_uint;

pub type cl_platform_id = *mut c_void;
pub type cl_device_id = *mut c_void;
pub type cl_context = *mut c_void;
pub type cl_command_queue = *mut c_void;
pub type cl_mem = *mut c_void;
pub type cl_program = *mut c_void;
pub type cl_kernel = *mut c_void;
pub type cl_event = *mut c_void;

pub const CL_SUCCESS: cl_int = 0;
pub const CL_DEVICE_TYPE_GPU: cl_device_type = 1 << 2;
pub const CL_DEVICE_TYPE_ALL: cl_device_type = 0xFFFFFFFF;
pub const CL_MEM_READ_ONLY: cl_mem_flags = 1 << 2;
pub const CL_MEM_WRITE_ONLY: cl_mem_flags = 1 << 1;
pub const CL_MEM_READ_WRITE: cl_mem_flags = 1 << 0;
pub const CL_MEM_COPY_HOST_PTR: cl_mem_flags = 1 << 5;
pub const CL_PROGRAM_BUILD_LOG: cl_program_build_info = 0x1183;
pub const CL_PLATFORM_NAME: cl_platform_info = 0x0902;
pub const CL_DEVICE_NAME: cl_device_info = 0x102B;
pub const CL_TRUE: cl_bool = 1;

#[derive(Clone)]
pub struct OpenClApi {
    _lib: std::sync::Arc<LibHolder>,
    pub clGetPlatformIDs: unsafe extern "C" fn(cl_uint, *mut cl_platform_id, *mut cl_uint) -> cl_int,
    pub clGetPlatformInfo: unsafe extern "C" fn(cl_platform_id, cl_platform_info, usize, *mut c_void, *mut usize) -> cl_int,
    pub clGetDeviceIDs: unsafe extern "C" fn(cl_platform_id, cl_device_type, cl_uint, *mut cl_device_id, *mut cl_uint) -> cl_int,
    pub clGetDeviceInfo: unsafe extern "C" fn(cl_device_id, cl_device_info, usize, *mut c_void, *mut usize) -> cl_int,
    pub clCreateContext: unsafe extern "C" fn(*const isize, cl_uint, *const cl_device_id, *mut c_void, *mut c_void, *mut cl_int) -> cl_context,
    pub clCreateCommandQueue: unsafe extern "C" fn(cl_context, cl_device_id, cl_command_queue_properties, *mut cl_int) -> cl_command_queue,
    pub clCreateBuffer: unsafe extern "C" fn(cl_context, cl_mem_flags, usize, *mut c_void, *mut cl_int) -> cl_mem,
    pub clCreateProgramWithSource: unsafe extern "C" fn(cl_context, cl_uint, *const *const c_char, *const usize, *mut cl_int) -> cl_program,
    pub clBuildProgram: unsafe extern "C" fn(cl_program, cl_uint, *const cl_device_id, *const c_char, *mut c_void, *mut c_void) -> cl_int,
    pub clGetProgramBuildInfo: unsafe extern "C" fn(cl_program, cl_device_id, cl_program_build_info, usize, *mut c_void, *mut usize) -> cl_int,
    pub clCreateKernel: unsafe extern "C" fn(cl_program, *const c_char, *mut cl_int) -> cl_kernel,
    pub clSetKernelArg: unsafe extern "C" fn(cl_kernel, cl_uint, usize, *const c_void) -> cl_int,
    pub clEnqueueWriteBuffer: unsafe extern "C" fn(cl_command_queue, cl_mem, cl_bool, usize, usize, *const c_void, cl_uint, *const cl_event, *mut cl_event) -> cl_int,
    pub clEnqueueReadBuffer: unsafe extern "C" fn(cl_command_queue, cl_mem, cl_bool, usize, usize, *mut c_void, cl_uint, *const cl_event, *mut cl_event) -> cl_int,
    pub clEnqueueNDRangeKernel: unsafe extern "C" fn(cl_command_queue, cl_kernel, cl_uint, *const usize, *const usize, *const usize, cl_uint, *const cl_event, *mut cl_event) -> cl_int,
    pub clFinish: unsafe extern "C" fn(cl_command_queue) -> cl_int,
    pub clReleaseMemObject: unsafe extern "C" fn(cl_mem) -> cl_int,
    pub clReleaseKernel: unsafe extern "C" fn(cl_kernel) -> cl_int,
    pub clReleaseProgram: unsafe extern "C" fn(cl_program) -> cl_int,
    pub clReleaseCommandQueue: unsafe extern "C" fn(cl_command_queue) -> cl_int,
    pub clReleaseContext: unsafe extern "C" fn(cl_context) -> cl_int,
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
    fn dlopen(filename: *const c_char, flag: i32) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> i32;
}
fn libc_dlclose(h: *mut c_void) {
    unsafe { dlclose(h); }
}

const RTLD_NOW: i32 = 2;

macro_rules! load {
    ($h:expr, $name:ident) => {{
        let s = concat!(stringify!($name), "\0");
        let p = unsafe { dlsym($h, s.as_ptr() as *const c_char) };
        if p.is_null() {
            return Err(format!("dlsym {} failed", stringify!($name)));
        }
        unsafe { std::mem::transmute(p) }
    }};
}

impl OpenClApi {
    pub fn try_load() -> Result<Self, String> {
        let names = ["libOpenCL.so.1\0", "libOpenCL.so\0"];
        let mut handle = std::ptr::null_mut();
        for n in names {
            handle = unsafe { dlopen(n.as_ptr() as *const c_char, RTLD_NOW) };
            if !handle.is_null() {
                break;
            }
        }
        if handle.is_null() {
            return Err("libOpenCL not found".into());
        }
        Ok(Self {
            _lib: std::sync::Arc::new(LibHolder(handle)),
            clGetPlatformIDs: load!(handle, clGetPlatformIDs),
            clGetPlatformInfo: load!(handle, clGetPlatformInfo),
            clGetDeviceIDs: load!(handle, clGetDeviceIDs),
            clGetDeviceInfo: load!(handle, clGetDeviceInfo),
            clCreateContext: load!(handle, clCreateContext),
            clCreateCommandQueue: load!(handle, clCreateCommandQueue),
            clCreateBuffer: load!(handle, clCreateBuffer),
            clCreateProgramWithSource: load!(handle, clCreateProgramWithSource),
            clBuildProgram: load!(handle, clBuildProgram),
            clGetProgramBuildInfo: load!(handle, clGetProgramBuildInfo),
            clCreateKernel: load!(handle, clCreateKernel),
            clSetKernelArg: load!(handle, clSetKernelArg),
            clEnqueueWriteBuffer: load!(handle, clEnqueueWriteBuffer),
            clEnqueueReadBuffer: load!(handle, clEnqueueReadBuffer),
            clEnqueueNDRangeKernel: load!(handle, clEnqueueNDRangeKernel),
            clFinish: load!(handle, clFinish),
            clReleaseMemObject: load!(handle, clReleaseMemObject),
            clReleaseKernel: load!(handle, clReleaseKernel),
            clReleaseProgram: load!(handle, clReleaseProgram),
            clReleaseCommandQueue: load!(handle, clReleaseCommandQueue),
            clReleaseContext: load!(handle, clReleaseContext),
        })
    }
}
