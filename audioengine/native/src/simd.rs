//! Four f32 lanes, for the work that runs sample by sample (recursive filters, delay networks, modulation) and so
//! cannot go through VOLK's block kernels (vk.rs): stereo pairs and parallel filters each take a lane. SSE2 on
//! x86_64 and NEON on aarch64 (both always there: no run-time detection), plain arrays elsewhere.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct F4(
    #[cfg(target_arch = "x86_64")] __m128,
    #[cfg(target_arch = "aarch64")] float32x4_t,
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))] [f32; 4],
);

impl Default for F4 {
    fn default() -> F4 {
        F4::splat(0.0)
    }
}

impl std::fmt::Debug for F4 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.to_array().fmt(f)
    }
}

#[cfg(target_arch = "x86_64")]
impl F4 {
    #[inline(always)]
    pub fn splat(v: f32) -> F4 {
        unsafe { F4(_mm_set1_ps(v)) }
    }
    #[inline(always)]
    pub fn new(a: f32, b: f32, c: f32, d: f32) -> F4 {
        unsafe { F4(_mm_setr_ps(a, b, c, d)) }
    }
    #[inline(always)]
    pub fn from_array(a: [f32; 4]) -> F4 {
        unsafe { F4(_mm_loadu_ps(a.as_ptr())) }
    }
    #[inline(always)]
    pub fn to_array(self) -> [f32; 4] {
        let mut a = [0f32; 4];
        unsafe { _mm_storeu_ps(a.as_mut_ptr(), self.0) };
        a
    }
    /// Four f32 from ``s[i..i + 4]``.
    #[inline(always)]
    pub fn load(s: &[f32], i: usize) -> F4 {
        assert!(i + 4 <= s.len());
        unsafe { F4(_mm_loadu_ps(s.as_ptr().add(i))) }
    }
    #[inline(always)]
    pub fn store(self, s: &mut [f32], i: usize) {
        assert!(i + 4 <= s.len());
        unsafe { _mm_storeu_ps(s.as_mut_ptr().add(i), self.0) }
    }
    #[inline(always)]
    pub fn add(self, o: F4) -> F4 {
        unsafe { F4(_mm_add_ps(self.0, o.0)) }
    }
    #[inline(always)]
    pub fn sub(self, o: F4) -> F4 {
        unsafe { F4(_mm_sub_ps(self.0, o.0)) }
    }
    #[inline(always)]
    pub fn mul(self, o: F4) -> F4 {
        unsafe { F4(_mm_mul_ps(self.0, o.0)) }
    }
    #[inline(always)]
    pub fn div(self, o: F4) -> F4 {
        unsafe { F4(_mm_div_ps(self.0, o.0)) }
    }
    #[inline(always)]
    pub fn min(self, o: F4) -> F4 {
        unsafe { F4(_mm_min_ps(self.0, o.0)) }
    }
    #[inline(always)]
    pub fn max(self, o: F4) -> F4 {
        unsafe { F4(_mm_max_ps(self.0, o.0)) }
    }
    #[inline(always)]
    pub fn abs(self) -> F4 {
        unsafe { F4(_mm_andnot_ps(_mm_set1_ps(-0.0), self.0)) }
    }
    /// The lanes summed.
    #[inline(always)]
    pub fn sum(self) -> f32 {
        let a = self.to_array();
        (a[0] + a[1]) + (a[2] + a[3])
    }
}

#[cfg(target_arch = "aarch64")]
impl F4 {
    #[inline(always)]
    pub fn splat(v: f32) -> F4 {
        unsafe { F4(vdupq_n_f32(v)) }
    }
    #[inline(always)]
    pub fn new(a: f32, b: f32, c: f32, d: f32) -> F4 {
        F4::from_array([a, b, c, d])
    }
    #[inline(always)]
    pub fn from_array(a: [f32; 4]) -> F4 {
        unsafe { F4(vld1q_f32(a.as_ptr())) }
    }
    #[inline(always)]
    pub fn to_array(self) -> [f32; 4] {
        let mut a = [0f32; 4];
        unsafe { vst1q_f32(a.as_mut_ptr(), self.0) };
        a
    }
    #[inline(always)]
    pub fn load(s: &[f32], i: usize) -> F4 {
        assert!(i + 4 <= s.len());
        unsafe { F4(vld1q_f32(s.as_ptr().add(i))) }
    }
    #[inline(always)]
    pub fn store(self, s: &mut [f32], i: usize) {
        assert!(i + 4 <= s.len());
        unsafe { vst1q_f32(s.as_mut_ptr().add(i), self.0) }
    }
    #[inline(always)]
    pub fn add(self, o: F4) -> F4 {
        unsafe { F4(vaddq_f32(self.0, o.0)) }
    }
    #[inline(always)]
    pub fn sub(self, o: F4) -> F4 {
        unsafe { F4(vsubq_f32(self.0, o.0)) }
    }
    #[inline(always)]
    pub fn mul(self, o: F4) -> F4 {
        unsafe { F4(vmulq_f32(self.0, o.0)) }
    }
    #[inline(always)]
    pub fn div(self, o: F4) -> F4 {
        unsafe { F4(vdivq_f32(self.0, o.0)) }
    }
    #[inline(always)]
    pub fn min(self, o: F4) -> F4 {
        unsafe { F4(vminq_f32(self.0, o.0)) }
    }
    #[inline(always)]
    pub fn max(self, o: F4) -> F4 {
        unsafe { F4(vmaxq_f32(self.0, o.0)) }
    }
    #[inline(always)]
    pub fn abs(self) -> F4 {
        unsafe { F4(vabsq_f32(self.0)) }
    }
    #[inline(always)]
    pub fn sum(self) -> f32 {
        let a = self.to_array();
        (a[0] + a[1]) + (a[2] + a[3])
    }
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
impl F4 {
    #[inline(always)]
    pub fn splat(v: f32) -> F4 {
        F4([v; 4])
    }
    #[inline(always)]
    pub fn new(a: f32, b: f32, c: f32, d: f32) -> F4 {
        F4([a, b, c, d])
    }
    #[inline(always)]
    pub fn from_array(a: [f32; 4]) -> F4 {
        F4(a)
    }
    #[inline(always)]
    pub fn to_array(self) -> [f32; 4] {
        self.0
    }
    #[inline(always)]
    pub fn load(s: &[f32], i: usize) -> F4 {
        F4([s[i], s[i + 1], s[i + 2], s[i + 3]])
    }
    #[inline(always)]
    pub fn store(self, s: &mut [f32], i: usize) {
        s[i..i + 4].copy_from_slice(&self.0);
    }
    #[inline(always)]
    fn zip(self, o: F4, f: impl Fn(f32, f32) -> f32) -> F4 {
        F4(std::array::from_fn(|k| f(self.0[k], o.0[k])))
    }
    #[inline(always)]
    pub fn add(self, o: F4) -> F4 {
        self.zip(o, |a, b| a + b)
    }
    #[inline(always)]
    pub fn sub(self, o: F4) -> F4 {
        self.zip(o, |a, b| a - b)
    }
    #[inline(always)]
    pub fn mul(self, o: F4) -> F4 {
        self.zip(o, |a, b| a * b)
    }
    #[inline(always)]
    pub fn div(self, o: F4) -> F4 {
        self.zip(o, |a, b| a / b)
    }
    #[inline(always)]
    pub fn min(self, o: F4) -> F4 {
        self.zip(o, f32::min)
    }
    #[inline(always)]
    pub fn max(self, o: F4) -> F4 {
        self.zip(o, f32::max)
    }
    #[inline(always)]
    pub fn abs(self) -> F4 {
        F4(self.0.map(f32::abs))
    }
    #[inline(always)]
    pub fn sum(self) -> f32 {
        (self.0[0] + self.0[1]) + (self.0[2] + self.0[3])
    }
}

impl F4 {
    /// self * a + b
    #[inline(always)]
    pub fn mul_add(self, a: F4, b: F4) -> F4 {
        self.mul(a).add(b)
    }
    #[inline(always)]
    pub fn clamp(self, lo: F4, hi: F4) -> F4 {
        self.max(lo).min(hi)
    }
    /// A soft saturation, x / (1 + |x|) shaped to stay near linear below ~0.5 (cheap: no tanh).
    #[inline(always)]
    pub fn soft_sat(self) -> F4 {
        let x = self.clamp(F4::splat(-3.0), F4::splat(3.0));
        // Pade approximant of tanh, exact enough inside +-3
        let x2 = x.mul(x);
        let num = x.mul(x2.add(F4::splat(27.0)));
        let den = x2.mul(F4::splat(9.0)).add(F4::splat(27.0));
        num.div(den)
    }
}

impl std::ops::Add for F4 {
    type Output = F4;
    #[inline(always)]
    fn add(self, o: F4) -> F4 {
        F4::add(self, o)
    }
}

impl std::ops::Sub for F4 {
    type Output = F4;
    #[inline(always)]
    fn sub(self, o: F4) -> F4 {
        F4::sub(self, o)
    }
}

impl std::ops::Mul for F4 {
    type Output = F4;
    #[inline(always)]
    fn mul(self, o: F4) -> F4 {
        F4::mul(self, o)
    }
}

/// out += a * s, four at a time (VOLK as built has no multiply-add kernel).
pub fn axpy(out: &mut [f32], a: &[f32], s: f32) {
    assert!(out.len() == a.len());
    let k = F4::splat(s);
    let whole = out.len() / 4 * 4;
    let mut i = 0;
    while i < whole {
        F4::load(a, i).mul_add(k, F4::load(out, i)).store(out, i);
        i += 4;
    }
    for j in whole..out.len() {
        out[j] += a[j] * s;
    }
}

/// The largest |x| of ``x``, four at a time.
pub fn peak_abs(x: &[f32]) -> f32 {
    let mut m = F4::default();
    let whole = x.len() / 4 * 4;
    let mut i = 0;
    while i < whole {
        m = m.max(F4::load(x, i).abs());
        i += 4;
    }
    let a = m.to_array();
    x[whole..].iter().fold(a[0].max(a[1]).max(a[2]).max(a[3]), |m, v| m.max(v.abs()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lanes() {
        let a = F4::new(1.0, -2.0, 3.0, -4.0);
        let b = F4::splat(2.0);
        assert_eq!((a * b + b).to_array(), [4.0, -2.0, 8.0, -6.0]);
        assert_eq!(a.abs().to_array(), [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(a.sum(), -2.0);
        let s = F4::splat(0.25).soft_sat().to_array()[0];
        assert!((s - 0.25f32.tanh()).abs() < 1e-3, "{s}");
        assert_eq!(peak_abs(&[0.1, -0.9, 0.3, 0.2, 0.5, -0.95, 0.0]), 0.95);
        let mut o = [1.0f32; 5];
        axpy(&mut o, &[1.0, 2.0, 3.0, 4.0, 5.0], 0.5);
        assert_eq!(o, [1.5, 2.0, 2.5, 3.0, 3.5]);
        let mut buf = [0f32; 6];
        a.store(&mut buf, 2);
        assert_eq!(F4::load(&buf, 2).to_array(), a.to_array());
    }
}
