//! The band split of the light time stretch (stretch.rs, ``Stretcher::light``): the source's low band
//! decimated by ``d`` (a Kaiser-windowed lowpass, -90 dB from the low rate's Nyquist frequency down, its passband
//! to seven eighths of it: 7 kHz at 48 kHz; the transition, where both bands carry a tone and their stretched
//! parts add with unrelated phases, kept narrow) and the high band the residual, the source less the low band
//! interpolated back (the same filter, polyphase), so the two always sum to the source. Both are computed where
//! they are read, around the stream's position, into caches of a few windows: nothing is kept per track.

use std::f64::consts::PI;

const ATTENUATION_DB: f64 = 90.0;
/// The low band's rate is about this or more: 44.1 / 48 kHz split by 3, 88.2 / 96 kHz by 6.
const LOW_RATE_MIN: f64 = 14_000.0;
/// The passband's edge, of the low rate's Nyquist frequency.
const PASS: f64 = 0.875;

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
    half: i64,
    fir: Vec<f32>,
    /// the interpolation's phases, each in reading order (oldest low sample first), times ``d``
    phases: Vec<Vec<f32>>,
    taps_per_phase: usize,
    low: Run,
    high: Run,
    scratch: Vec<f32>,
}

impl BandSplit {
    /// None where the rate leaves nothing to split (a low band under 2x decimation).
    pub fn new(ch: usize, rate: u32) -> Option<BandSplit> {
        let d = (rate as f64 / LOW_RATE_MIN).floor() as usize;
        if d < 2 {
            return None;
        }
        let nyquist_low = 0.5 / d as f64; // of the full rate
        let pass = PASS * nyquist_low;
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
            high: Run::new(ch),
            scratch: Vec::new(),
        })
    }

    pub fn reset(&mut self) {
        self.low = Run::new(self.ch);
        self.high = Run::new(self.ch);
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

    /// The high band's samples ``[a, b)`` (full rate) computed; ``high(c, a, b)`` reads them.
    pub fn ensure_high(&mut self, source: &[f32], a: i64, b: i64) {
        let from = self.high.keep_from(a, b, 8192);
        if from >= b {
            return;
        }
        let d = self.d as i64;
        let t = self.taps_per_phase as i64;
        // the low samples the interpolation reads: k_max(n) = (n + half) div d, back t - 1 of them
        let k_lo = (from + self.half).div_euclid(d) - t + 1;
        let k_hi = (b - 1 + self.half).div_euclid(d) + 1;
        self.ensure_low(source, k_lo, k_hi);
        let frames = (source.len() / self.ch) as i64;
        for c in 0..self.ch {
            let s0 = (k_lo - self.low.base) as usize;
            let low = &self.low.data[c][s0..s0 + (k_hi - k_lo) as usize];
            let out = &mut self.high.data[c];
            for n in from..b {
                let k_max = (n + self.half).div_euclid(d);
                let r = (n + self.half).rem_euclid(d) as usize;
                let s = (k_max - t + 1 - k_lo) as usize;
                let up = dot(&self.phases[r], &low[s..s + t as usize]);
                let x = if n >= 0 && n < frames { source[n as usize * self.ch + c] } else { 0.0 };
                out.push(x - up);
            }
        }
    }

    pub fn high(&self, c: usize, a: i64, b: i64) -> &[f32] {
        let s = (a - self.high.base) as usize;
        &self.high.data[c][s..s + (b - a) as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bands_sum_to_the_source_and_split_where_they_should() {
        let rate = 48_000u32;
        let mut split = BandSplit::new(2, rate).unwrap();
        assert_eq!(split.d, 3);
        let n = 48_000usize;
        let tone = |hz: f64, i: usize| (2.0 * PI * hz * i as f64 / rate as f64).sin() as f32;
        // left: 1 kHz + 12 kHz, right: 4 kHz
        let source: Vec<f32> = (0..n).flat_map(|i| [tone(1000.0, i) + tone(12_000.0, i), tone(4000.0, i)]).collect();
        let (a, b) = (10_000i64, 20_000i64);
        split.ensure_high(&source, a, b);
        // the low band interpolated: the source less the high band
        let high_l = split.high(0, a, b).to_vec();
        let high_r = split.high(1, a, b).to_vec();
        // left: the high band is the 12 kHz tone alone (1 kHz kept by the low band); right: nothing (4 kHz)
        let mut err_left = 0f64;
        let mut leak_right = 0f64;
        for (k, i) in (a..b).enumerate() {
            let i = i as usize;
            err_left = err_left.max((high_l[k] - tone(12_000.0, i)).abs() as f64);
            leak_right = leak_right.max(high_r[k].abs() as f64);
        }
        assert!(err_left < 1e-3, "the high band off the 12 kHz tone by {err_left}");
        assert!(leak_right < 1e-3, "4 kHz leaks {leak_right} into the high band");
    }
}
