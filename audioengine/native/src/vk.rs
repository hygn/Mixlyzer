//! VOLK's SIMD kernels (LGPL-3.0-or-later; the machine picked at run time: SSE / AVX / AVX-512 / NEON):
//! the ones volk-rs wraps, plus the few it does not, declared here the way it declares its own.

use num_complex::Complex32;

mod ffi {
    use num_complex::Complex32;
    use std::os::raw::c_uint;

    #[link(name = "volk")]
    unsafe extern "C" {
        pub static volk_32f_x2_multiply_32f: extern "C" fn(out: *mut f32, a: *const f32, b: *const f32, n: c_uint);
        pub static volk_32f_x2_add_32f: extern "C" fn(out: *mut f32, a: *const f32, b: *const f32, n: c_uint);
        pub static volk_32f_s32f_multiply_32f: extern "C" fn(out: *mut f32, a: *const f32, s: f32, n: c_uint);
        pub static volk_32fc_magnitude_32f: extern "C" fn(out: *mut f32, a: *const Complex32, n: c_uint);
        pub static volk_32fc_magnitude_squared_32f: extern "C" fn(out: *mut f32, a: *const Complex32, n: c_uint);
        pub static volk_32f_sin_32f: extern "C" fn(out: *mut f32, a: *const f32, n: c_uint);
        pub static volk_32f_cos_32f: extern "C" fn(out: *mut f32, a: *const f32, n: c_uint);
    }
}

/// out = a * b
pub fn mul(out: &mut [f32], a: &[f32], b: &[f32]) {
    assert!(out.len() == a.len() && a.len() == b.len());
    unsafe { ffi::volk_32f_x2_multiply_32f(out.as_mut_ptr(), a.as_ptr(), b.as_ptr(), a.len() as u32) }
}

/// out *= b
pub fn mul_in(out: &mut [f32], b: &[f32]) {
    assert!(out.len() == b.len());
    let p = out.as_mut_ptr();
    unsafe { ffi::volk_32f_x2_multiply_32f(p, p, b.as_ptr(), b.len() as u32) }
}

/// out += b
pub fn add_in(out: &mut [f32], b: &[f32]) {
    assert!(out.len() == b.len());
    let p = out.as_mut_ptr();
    unsafe { ffi::volk_32f_x2_add_32f(p, p, b.as_ptr(), b.len() as u32) }
}

/// out *= s
pub fn scale_in(out: &mut [f32], s: f32) {
    let p = out.as_mut_ptr();
    unsafe { ffi::volk_32f_s32f_multiply_32f(p, p, s, out.len() as u32) }
}

/// out = |a|
pub fn magnitude(out: &mut [f32], a: &[Complex32]) {
    assert!(out.len() == a.len());
    unsafe { ffi::volk_32fc_magnitude_32f(out.as_mut_ptr(), a.as_ptr(), a.len() as u32) }
}

/// out = |a|^2
pub fn magnitude_squared(out: &mut [f32], a: &[Complex32]) {
    assert!(out.len() == a.len());
    unsafe { ffi::volk_32fc_magnitude_squared_32f(out.as_mut_ptr(), a.as_ptr(), a.len() as u32) }
}

/// out = sin(a)
pub fn sin(out: &mut [f32], a: &[f32]) {
    assert!(out.len() == a.len());
    unsafe { ffi::volk_32f_sin_32f(out.as_mut_ptr(), a.as_ptr(), a.len() as u32) }
}

/// out = cos(a)
pub fn cos(out: &mut [f32], a: &[f32]) {
    assert!(out.len() == a.len());
    unsafe { ffi::volk_32f_cos_32f(out.as_mut_ptr(), a.as_ptr(), a.len() as u32) }
}

/// out = arg(a)
pub fn phase(out: &mut [f32], a: &[Complex32]) {
    volk::volk_32fc_s32f_atan2_32f(out, a, 1.0);
}

/// out *= b (complex)
pub fn cmul_in(out: &mut [Complex32], b: &[Complex32]) {
    volk::volk_32fc_x2_multiply_32fc_inplace(out, b);
}

/// out = sqrt(a)
pub fn sqrt(out: &mut [f32], a: &[f32]) {
    volk::volk_32f_sqrt_32f(out, a);
}

/// The machine VOLK runs (and its first call's dispatch done now, not in the audio path).
pub fn warm() -> String {
    let a = [Complex32::new(1.0, 1.0); 16];
    let mut m = [0f32; 16];
    magnitude(&mut m, &a);
    magnitude_squared(&mut m, &a);
    phase(&mut m, &a);
    let mut c = a;
    cmul_in(&mut c, &a);
    let b = m;
    mul_in(&mut m, &b);
    add_in(&mut m, &b);
    scale_in(&mut m, 0.5);
    let mut s = [0f32; 16];
    sqrt(&mut s, &b);
    sin(&mut s, &b);
    cos(&mut s, &b);
    let mut o = [0f32; 16];
    mul(&mut o, &s, &b);
    volk::volk_get_machine()
}
