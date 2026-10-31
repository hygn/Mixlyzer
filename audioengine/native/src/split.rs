//! The time stretch's low band (stretch.rs, ``Stretcher``): the source decimated by ``d`` for the low band's
//! vocoder (a Kaiser-windowed lowpass, -90 dB from the low rate's Nyquist frequency down, its passband up to the
//! vocoders' crossover, its width and a margin: 1.65 kHz, 8 at 44.1 / 48 kHz), and the vocoder's synthesis at the
//! low rate brought back to the full rate by the same lowpass (``upsample``). Computed where it is read, around the
//! stream's position, into a cache of a few windows: nothing is kept per track.

use std::f64::consts::PI;

const ATTENUATION_DB: f64 = 90.0;
/// The passband's edge at most this of the low rate's Nyquist frequency (the transition the rest).
const PASS_MAX: f64 = 0.6;

/// sum a[i] * b[i], eight lanes (vectorised by the compiler; a float sum is not reordered on its own).
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let mut acc = [0f32; 8];
    let chunks = n / 8;
    for k in 0..chunks {
        let (x, y) = (&a[8 * k..8 * k + 8], &b[8 * k..8 * k + 8]);
        for l in 0..8 {
            acc[l] += x[l] * y[l];
        }
    }
    let mut s = acc.iter().sum::<f32>();
    for i in 8 * chunks..n {
        s += a[i] * b[i];
    }
    s
}

fn bessel_i0(x: f64) -> f64 {
    let (mut sum, mut term, mut k) = (1.0, 1.0, 1.0);
    while term > 1e-12 * sum {
        term *= (x / (2.0 * k)) * (x / (2.0 * k));
        sum += term;
        k += 1.0;
    }
    sum
}

/// A lowpass, cut at ``cutoff`` (of the rate), ``transition`` wide, ``ATTENUATION_DB`` down past it; odd length,
/// its sum 1.
fn kaiser_lowpass(cutoff: f64, transition: f64) -> Vec<f64> {
    let a = ATTENUATION_DB;
    let beta = 0.1102 * (a - 8.7);
    let n = ((a - 8.0) / (2.285 * 2.0 * PI * transition)).ceil() as usize + 1;
    let len = n | 1;
    let m = (len / 2) as f64;
    let i0b = bessel_i0(beta);
    let mut h: Vec<f64> = (0..len)
        .map(|i| {
            let t = i as f64 - m;
            let sinc = if t == 0.0 { 2.0 * cutoff } else { (2.0 * PI * cutoff * t).sin() / (PI * t) };
            let r = t / m;
            sinc * bessel_i0(beta * (1.0 - r * r).max(0.0).sqrt()) / i0b
        })
        .collect();
    let s: f64 = h.iter().sum();
    h.iter_mut().for_each(|v| *v /= s);
    h
}

/// A run of one band's samples, per channel: ``[base, base + len)``.
struct Run {
    base: i64,
    data: Vec<Vec<f32>>,
}

impl Run {
    fn new(ch: usize) -> Run {
        Run { base: 0, data: vec![Vec::new(); ch] }
    }

    fn end(&self) -> i64 {
        self.base + self.data[0].len() as i64
    }

    /// Keeps what overlaps ``[a, ..)`` and is contiguous with it; returns where to compute from.
    fn keep_from(&mut self, a: i64, b: i64, slack: i64) -> i64 {
        let (base, end) = (self.base, self.end());
        if self.data[0].is_empty() || a < base || a > end || b <= a {
            for d in &mut self.data {
                d.clear();
            }
            self.base = a;
            return a;
        }
        // drop what lies well behind the request
        let drop = (a - slack - base).max(0) as usize;
        if drop > 0 && drop >= self.data[0].len() / 2 {
            for d in &mut self.data {
                d.drain(..drop);
            }
            self.base += drop as i64;
        }
        end.max(a)
    }
}

pub struct BandSplit {
    pub d: usize,
    ch: usize,
    pub half: i64,
    fir: Vec<f32>,
    /// the interpolation's phases, each in reading order (oldest low sample first), times ``d``
    phases: Vec<Vec<f32>>,
    pub taps_per_phase: usize,
    low: Run,
    scratch: Vec<f32>,
}

impl BandSplit {
    /// The low band up to ``pass_hz``: decimated by the most that keeps it in ``PASS_MAX`` of the low Nyquist
    /// frequency (the transition from it, cut -90 dB by a Kaiser lowpass). None where the rate leaves nothing to
    /// decimate.
    pub fn new(ch: usize, rate: u32, pass_hz: f64) -> Option<BandSplit> {
        let d = (rate as f64 * PASS_MAX / (2.0 * pass_hz)).floor() as usize;
        if d < 2 {
            return None;
        }
        let nyquist_low = 0.5 / d as f64; // of the full rate
        let pass = pass_hz / rate as f64;
        let h = kaiser_lowpass(0.5 * (pass + nyquist_low), nyquist_low - pass);
        let len = h.len();
        let taps_per_phase = len.div_ceil(d);
        let phases = (0..d)
            .map(|r| {
                (0..taps_per_phase)
                    .map(|i| {
                        let j = taps_per_phase - 1 - i;
                        let k = r + j * d;
                        if k < len { (d as f64 * h[k]) as f32 } else { 0.0 }
                    })
                    .collect()
            })
            .collect();
        Some(BandSplit {
            d,
            ch,
            half: (len / 2) as i64,
            fir: h.iter().map(|&v| v as f32).collect(),
            phases,
            taps_per_phase,
            low: Run::new(ch),
            scratch: Vec::new(),
        })
    }

    /// Full-rate sample ``n`` of a low-rate signal (``low``: its samples from ``low_base`` on), interpolated by the
    /// same lowpass (polyphase). The low samples read: from (n + half) div d - taps + 1, ``taps_per_phase`` of them.
    pub fn upsample(&self, low: &[f32], low_base: i64, n: i64) -> f32 {
        let d = self.d as i64;
        let t = self.taps_per_phase as i64;
        let k_max = (n + self.half).div_euclid(d);
        let r = (n + self.half).rem_euclid(d) as usize;
        let s = k_max - t + 1 - low_base;
        if s < 0 || s + t > low.len() as i64 {
            // partly outside the run: the missing samples are silence
            let mut acc = 0f32;
            for i in 0..t {
                let k = s + i;
                if k >= 0 && k < low.len() as i64 {
                    acc += self.phases[r][i as usize] * low[k as usize];
                }
            }
            return acc;
        }
        dot(&self.phases[r], &low[s as usize..(s + t) as usize])
    }

    pub fn reset(&mut self) {
        self.low = Run::new(self.ch);
    }

    /// One channel of the source, ``[a, b)`` (zeros outside it), into ``out``.
    fn gather(source: &[f32], ch: usize, c: usize, a: i64, b: i64, out: &mut Vec<f32>) {
        let frames = (source.len() / ch) as i64;
        out.clear();
        out.resize((b - a).max(0) as usize, 0.0);
        let (lo, hi) = (a.max(0), b.min(frames));
        for i in lo..hi {
            out[(i - a) as usize] = source[i as usize * ch + c];
        }
    }

    /// The low band's samples ``[a, b)`` (low rate) computed; ``low(c, a, b)`` reads them.
    pub fn ensure_low(&mut self, source: &[f32], a: i64, b: i64) {
        let from = self.low.keep_from(a, b, 4096);
        if from >= b {
            return;
        }
        let d = self.d as i64;
        // low[k] = sum_j fir[j] x[k d - half + j]
        let (xa, xb) = (from * d - self.half, (b - 1) * d - self.half + self.fir.len() as i64);
        for c in 0..self.ch {
            Self::gather(source, self.ch, c, xa, xb, &mut self.scratch);
            let run = &mut self.low.data[c];
            for k in from..b {
                let at = ((k - from) * d) as usize;
                run.push(dot(&self.fir, &self.scratch[at..at + self.fir.len()]));
            }
        }
    }

    pub fn low(&self, c: usize, a: i64, b: i64) -> &[f32] {
        let s = (a - self.low.base) as usize;
        &self.low.data[c][s..s + (b - a) as usize]
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_low_band_keeps_its_passband_and_drops_the_rest() {
        let rate = 48_000u32;
        let mut split = BandSplit::new(2, rate, 1100.0).unwrap();
        assert_eq!(split.d, 13);
        let n = 48_000usize;
        let tone = |hz: f64, i: f64| (2.0 * PI * hz * i / rate as f64).sin() as f32;
        // left: 1 kHz + 3 kHz (past the low rate's Nyquist frequency, 1.85 kHz), right: 400 Hz
        let source: Vec<f32> =
            (0..n).flat_map(|i| [tone(1000.0, i as f64) + tone(3000.0, i as f64), tone(400.0, i as f64)]).collect();
        let (a, b) = (1_000i64, 2_500i64);
        split.ensure_low(&source, a, b);
        // low sample k is the source at k * d, band-limited: the 1 kHz and the 400 Hz tones there, the 3 kHz gone
        let (mut err_left, mut err_right) = (0f64, 0f64);
        for (j, k) in (a..b).enumerate() {
            let t = (k * 13) as f64;
            err_left = err_left.max((split.low(0, a, b)[j] - tone(1000.0, t)).abs() as f64);
            err_right = err_right.max((split.low(1, a, b)[j] - tone(400.0, t)).abs() as f64);
        }
        assert!(err_left < 1e-3, "the low band off the 1 kHz tone by {err_left} (the 3 kHz left in?)");
        assert!(err_right < 1e-3, "the low band off the 400 Hz tone by {err_right}");
    }

    #[test]
    fn the_low_band_comes_back_to_the_full_rate() {
        let rate = 48_000u32;
        let mut split = BandSplit::new(1, rate, 1100.0).unwrap();
        let tone = |hz: f64, i: f64| (2.0 * PI * hz * i / rate as f64).sin() as f32;
        let source: Vec<f32> = (0..48_000).map(|i| tone(1000.0, i as f64) + tone(250.0, i as f64)).collect();
        let (a, b) = (500i64, 3_000i64);
        split.ensure_low(&source, a, b);
        let low = split.low(0, a, b).to_vec();
        // inside the run, away from its ends: the source itself at every full-rate sample
        let mut err = 0f64;
        for t in (a * 13 + 2_000)..(b * 13 - 2_000) {
            err = err.max((split.upsample(&low, a, t) - source[t as usize]).abs() as f64);
        }
        assert!(err < 2e-3, "the interpolated low band off the source by {err}");
    }
}
