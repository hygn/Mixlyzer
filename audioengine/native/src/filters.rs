//! Recursive filters four at a time (simd::F4: a lane each): biquads (RBJ's cookbook, transposed direct form II)
//! and a state-variable filter (Cytomic's trapezoidal SVF: its cutoff moved every block without zipper noise). A
//! lane is usually one side of a stereo pair, so a pair of filters (the isolator's low and rest, a shelf pair) runs in
//! one pass.

use crate::simd::F4;

/// A biquad's coefficients, normalised (a0 = 1).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Coeffs {
    pub b0: f64,
    pub b1: f64,
    pub b2: f64,
    pub a1: f64,
    pub a2: f64,
}

impl Coeffs {
    pub const IDENTITY: Coeffs = Coeffs { b0: 1.0, b1: 0.0, b2: 0.0, a1: 0.0, a2: 0.0 };

    fn norm(b0: f64, b1: f64, b2: f64, a0: f64, a1: f64, a2: f64) -> Coeffs {
        Coeffs { b0: b0 / a0, b1: b1 / a0, b2: b2 / a0, a1: a1 / a0, a2: a2 / a0 }
    }

    fn w(f: f64, rate: f64) -> (f64, f64) {
        let w = 2.0 * std::f64::consts::PI * (f / rate).clamp(1e-6, 0.4999);
        w.sin_cos()
    }

    pub fn lowpass(f: f64, q: f64, rate: f64) -> Coeffs {
        let (sin, cos) = Self::w(f, rate);
        let alpha = sin / (2.0 * q);
        Self::norm((1.0 - cos) / 2.0, 1.0 - cos, (1.0 - cos) / 2.0, 1.0 + alpha, -2.0 * cos, 1.0 - alpha)
    }

    pub fn highpass(f: f64, q: f64, rate: f64) -> Coeffs {
        let (sin, cos) = Self::w(f, rate);
        let alpha = sin / (2.0 * q);
        Self::norm((1.0 + cos) / 2.0, -(1.0 + cos), (1.0 + cos) / 2.0, 1.0 + alpha, -2.0 * cos, 1.0 - alpha)
    }

    pub fn allpass(f: f64, q: f64, rate: f64) -> Coeffs {
        let (sin, cos) = Self::w(f, rate);
        let alpha = sin / (2.0 * q);
        Self::norm(1.0 - alpha, -2.0 * cos, 1.0 + alpha, 1.0 + alpha, -2.0 * cos, 1.0 - alpha)
    }

    /// A peaking band of ``db`` at ``f`` (its width ``q``).
    pub fn peak(f: f64, q: f64, db: f64, rate: f64) -> Coeffs {
        let (sin, cos) = Self::w(f, rate);
        let a = 10f64.powf(db / 40.0);
        let alpha = sin / (2.0 * q);
        Self::norm(1.0 + alpha * a, -2.0 * cos, 1.0 - alpha * a, 1.0 + alpha / a, -2.0 * cos, 1.0 - alpha / a)
    }

    /// A shelf of ``db`` below ``f`` (slope 1: its gentlest without overshoot).
    pub fn low_shelf(f: f64, db: f64, rate: f64) -> Coeffs {
        let (sin, cos) = Self::w(f, rate);
        let a = 10f64.powf(db / 40.0);
        let alpha = sin / 2.0 * std::f64::consts::SQRT_2;
        let s = 2.0 * a.sqrt() * alpha;
        Self::norm(
            a * ((a + 1.0) - (a - 1.0) * cos + s),
            2.0 * a * ((a - 1.0) - (a + 1.0) * cos),
            a * ((a + 1.0) - (a - 1.0) * cos - s),
            (a + 1.0) + (a - 1.0) * cos + s,
            -2.0 * ((a - 1.0) + (a + 1.0) * cos),
            (a + 1.0) + (a - 1.0) * cos - s,
        )
    }

    /// A shelf of ``db`` above ``f``.
    pub fn high_shelf(f: f64, db: f64, rate: f64) -> Coeffs {
        let (sin, cos) = Self::w(f, rate);
        let a = 10f64.powf(db / 40.0);
        let alpha = sin / 2.0 * std::f64::consts::SQRT_2;
        let s = 2.0 * a.sqrt() * alpha;
        Self::norm(
            a * ((a + 1.0) + (a - 1.0) * cos + s),
            -2.0 * a * ((a - 1.0) + (a + 1.0) * cos),
            a * ((a + 1.0) + (a - 1.0) * cos - s),
            (a + 1.0) - (a - 1.0) * cos + s,
            2.0 * ((a - 1.0) - (a + 1.0) * cos),
            (a + 1.0) - (a - 1.0) * cos - s,
        )
    }
}

/// Four biquads, a lane each (each its own coefficients), on four inputs at once.
#[derive(Clone, Copy, Default)]
pub struct Biquad4 {
    b0: F4,
    b1: F4,
    b2: F4,
    a1: F4,
    a2: F4,
    z1: F4,
    z2: F4,
}

impl Biquad4 {
    pub fn new(c: [Coeffs; 4]) -> Biquad4 {
        let mut b = Biquad4::default();
        b.set(c);
        b
    }

    pub fn set(&mut self, c: [Coeffs; 4]) {
        let lane = |f: fn(&Coeffs) -> f64| F4::new(f(&c[0]) as f32, f(&c[1]) as f32, f(&c[2]) as f32, f(&c[3]) as f32);
        self.b0 = lane(|c| c.b0);
        self.b1 = lane(|c| c.b1);
        self.b2 = lane(|c| c.b2);
        self.a1 = lane(|c| c.a1);
        self.a2 = lane(|c| c.a2);
    }

    #[inline(always)]
    pub fn run(&mut self, x: F4) -> F4 {
        let y = self.b0.mul_add(x, self.z1);
        self.z1 = self.b1.mul_add(x, self.z2) - self.a1 * y;
        self.z2 = self.b2 * x - self.a2 * y;
        y
    }

    pub fn reset(&mut self) {
        self.z1 = F4::default();
        self.z2 = F4::default();
    }
}

/// Four state-variable filters (Cytomic's trapezoidal SVF), a lane each: low-pass, band-pass and high-pass at once;
/// the cutoff and resonance set per block (set), smoothly.
#[derive(Clone, Copy, Default)]
pub struct Svf4 {
    k: F4,  // 1 / Q
    a1: F4,
    a2: F4,
    a3: F4,
    ic1: F4,
    ic2: F4,
}

/// One step's outputs.
pub struct SvfOut {
    pub lp: F4,
    pub bp: F4,
    pub hp: F4,
}

impl Svf4 {
    /// Cutoffs (Hz) and Qs per lane.
    pub fn set(&mut self, f: [f64; 4], q: [f64; 4], rate: f64) {
        let mut k = [0f32; 4];
        let (mut a1, mut a2, mut a3) = ([0f32; 4], [0f32; 4], [0f32; 4]);
        for i in 0..4 {
            let g = (std::f64::consts::PI * (f[i] / rate).clamp(1e-5, 0.49)).tan();
            let kk = 1.0 / q[i].max(0.05);
            let b1 = 1.0 / (1.0 + g * (g + kk));
            k[i] = kk as f32;
            a1[i] = b1 as f32;
            a2[i] = (g * b1) as f32;
            a3[i] = (g * g * b1) as f32;
        }
        self.k = F4::from_array(k);
        self.a1 = F4::from_array(a1);
        self.a2 = F4::from_array(a2);
        self.a3 = F4::from_array(a3);
    }

    #[inline(always)]
    pub fn run(&mut self, x: F4) -> SvfOut {
        let v3 = x - self.ic2;
        let v1 = self.a1.mul_add(self.ic1, self.a2 * v3);
        let v2 = self.ic2 + self.a2.mul_add(self.ic1, self.a3 * v3);
        let two = F4::splat(2.0);
        self.ic1 = two.mul_add(v1, F4::default()) - self.ic1;
        self.ic2 = two.mul_add(v2, F4::default()) - self.ic2;
        SvfOut { lp: v2, bp: v1, hp: x - self.k * v1 - v2 }
    }

    pub fn reset(&mut self) {
        self.ic1 = F4::default();
        self.ic2 = F4::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The level (dB) a sine at ``f`` comes out at, by RMS over its settled half.
    fn gain_at(c: Coeffs, f: f64, rate: f64) -> f64 {
        let mut b = Biquad4::new([c; 4]);
        let n = (rate as usize) / 2;
        let (mut out, mut inp) = (0f64, 0f64);
        for i in 0..n {
            let x = (2.0 * std::f64::consts::PI * f * i as f64 / rate).sin() as f32;
            let y = b.run(F4::splat(x)).to_array()[0];
            if i > n / 2 {
                out += (y as f64).powi(2);
                inp += (x as f64).powi(2);
            }
        }
        10.0 * (out / inp).log10()
    }

    #[test]
    fn shapes() {
        let r = 48000.0;
        assert!(gain_at(Coeffs::lowpass(1000.0, 0.707, r), 100.0, r).abs() < 0.1);
        assert!(gain_at(Coeffs::lowpass(1000.0, 0.707, r), 10000.0, r) < -35.0);
        assert!((gain_at(Coeffs::low_shelf(70.0, -26.0, r), 20.0, r) + 26.0).abs() < 1.0);
        assert!(gain_at(Coeffs::low_shelf(70.0, -26.0, r), 5000.0, r).abs() < 0.2);
        assert!((gain_at(Coeffs::high_shelf(13000.0, 6.0, r), 20000.0, r) - 6.0).abs() < 1.0);
        assert!((gain_at(Coeffs::peak(1000.0, 0.7, -12.0, r), 1000.0, r) + 12.0).abs() < 0.2);
        assert!(gain_at(Coeffs::allpass(1000.0, 0.707, r), 3000.0, r).abs() < 0.1);
    }

    #[test]
    fn svf_bands() {
        let r = 48000.0;
        let mut s = Svf4::default();
        s.set([1000.0; 4], [0.707; 4], r);
        let (mut lp, mut hp, mut inp) = (0f64, 0f64, 0f64);
        for i in 0..24000 {
            let x = (2.0 * std::f64::consts::PI * 8000.0 * i as f64 / r).sin() as f32;
            let o = s.run(F4::splat(x));
            if i > 12000 {
                lp += (o.lp.to_array()[0] as f64).powi(2);
                hp += (o.hp.to_array()[0] as f64).powi(2);
                inp += (x as f64).powi(2);
            }
        }
        let (lp, hp) = ((lp / inp).sqrt(), (hp / inp).sqrt());
        assert!(lp < 0.03 && hp > 0.95, "lp {lp} hp {hp}");
    }
}
