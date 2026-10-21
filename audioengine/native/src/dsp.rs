//! Small DSP pieces of the player (audioengine/python/dsp.py): fades, the soft clip, the varispeed reader, the
//! scratch reader and the platter under the hand.

/// Multiplies interleaved ``out`` by a linear ramp over ``fade`` frames (0 -> 1 when rising), held after.
pub fn fade(out: &mut [f32], ch: usize, fade: usize, rising: bool) {
    for (i, frame) in out.chunks_exact_mut(ch).enumerate() {
        let r = ((i as f32 + 0.5) / fade.max(1) as f32).clamp(0.0, 1.0);
        let g = if rising { r } else { 1.0 - r };
        for v in frame {
            *v *= g;
        }
    }
}

/// Linear up to |x| = knee, then a tanh curve to the ceiling 1.0 (continuous slope at the knee).
pub fn soft_clip(out: &mut [f32], knee: f32) {
    let span = 1.0 - knee;
    for v in out.iter_mut() {
        let a = v.abs();
        if a > knee {
            *v = v.signum() * (knee + span * ((a - knee) / span).tanh());
        }
    }
}

/// Peak of interleaved frames as dBFS within [floor, 0].
pub fn peak_dbfs(out: &[f32], floor: f64) -> f64 {
    let peak = out.iter().fold(0f32, |m, v| m.max(v.abs())) as f64;
    (20.0 * peak.clamp(1e-12, 1.0).log10()).clamp(floor, 0.0)
}

/// 5-tap Lagrange varispeed: output frame k is the input at ``pos + factor * k`` (taps i0-1 .. i0+3, the
/// edges held), so consecutive calls are seamless.
pub fn varispeed(pcm: &[f32], ch: usize, pos: f64, factor: f64, max_out: usize) -> Vec<f32> {
    let n = pcm.len() / ch;
    if max_out == 0 || pos >= n as f64 {
        return Vec::new();
    }
    let m = max_out.min(((n as f64 - pos) / factor).ceil() as usize);
    let mut out = vec![0f32; m * ch];
    if factor == 1.0 && pos.fract() == 0.0 {
        let i = pos as usize;
        out.copy_from_slice(&pcm[i * ch..(i + m) * ch]);
        return out;
    }
    let top = n as i64 - 1;
    for k in 0..m {
        let idx = pos + factor * k as f64;
        let i0 = idx.floor();
        let mu = idx - i0;
        let i0 = i0 as i64;
        let c = [
            (1.0 / 24.0) * mu * (mu - 1.0) * (mu - 2.0) * (mu - 3.0),
            (-1.0 / 6.0) * (mu + 1.0) * (mu - 1.0) * (mu - 2.0) * (mu - 3.0),
            (1.0 / 4.0) * (mu + 1.0) * mu * (mu - 2.0) * (mu - 3.0),
            (-1.0 / 6.0) * (mu + 1.0) * mu * (mu - 1.0) * (mu - 3.0),
            (1.0 / 24.0) * (mu + 1.0) * mu * (mu - 1.0) * (mu - 2.0),
        ];
        for c_i in 0..ch {
            let mut acc = 0f64;
            for (t, coeff) in c.iter().enumerate() {
                let j = (i0 - 1 + t as i64).clamp(0, top) as usize;
                acc += coeff * pcm[j * ch + c_i] as f64;
            }
            out[k * ch + c_i] = acc as f32;
        }
    }
    out
}

// Scratch reader: Kaiser-windowed sinc, cutoff READ_CUTOFF of the Nyquist divided by the speed (see
// audioengine/python/dsp.py's READ_* for the design).
const HALF: usize = 16;
const REACH: usize = 32;
const STRIDE_STRETCH: f64 = 4.0;
const BETA: f64 = 8.0;
const CUTOFF: f64 = 0.9;
const SINC_RES: usize = 512;
const WIN_RES: usize = 4096;

fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let q = x * x / 4.0;
    for k in 1..200 {
        term *= q / (k as f64 * k as f64);
        sum += term;
        if term < sum * 1e-17 {
            break;
        }
    }
    sum
}

fn sinc(x: f64) -> f64 {
    if x == 0.0 { 1.0 } else { (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x) }
}

pub struct ReadTables {
    unit: Vec<f64>,
    sinc: Vec<f64>,
    win: Vec<f64>,
    stride_w: Vec<f64>,
}

impl ReadTables {
    pub fn new() -> ReadTables {
        let i0b = bessel_i0(BETA);
        let kaiser = |x: f64| bessel_i0(BETA * (1.0 - x * x).max(0.0).sqrt()) / i0b;
        let sinc_t: Vec<f64> =
            (0..HALF * SINC_RES + 2).map(|i| CUTOFF * sinc(CUTOFF * i as f64 / SINC_RES as f64)).collect();
        let win: Vec<f64> = (0..WIN_RES + 2)
            .map(|i| {
                let x = i as f64 / WIN_RES as f64;
                if x >= 1.0 { 0.0 } else { kaiser(x) }
            })
            .collect();
        let unit: Vec<f64> = (0..HALF * SINC_RES + 2)
            .map(|i| {
                let u = i as f64 / SINC_RES as f64;
                if u >= HALF as f64 { 0.0 } else { sinc_t[i] * kaiser(u / HALF as f64) }
            })
            .collect();
        let mut stride_w: Vec<f64> = (1 - REACH as i64..REACH as i64)
            .map(|m| {
                let m = m as f64;
                CUTOFF * sinc(CUTOFF * m / STRIDE_STRETCH) * kaiser(m / REACH as f64)
            })
            .collect();
        let s: f64 = stride_w.iter().sum();
        stride_w.iter_mut().for_each(|w| *w /= s);
        ReadTables { unit, sinc: sinc_t, win, stride_w }
    }
}

fn lerp(table: &[f64], a: f64) -> f64 {
    let i = a as usize;
    if i >= table.len() - 1 {
        return table[table.len() - 1];
    }
    table[i] + (a - i as f64) * (table[i + 1] - table[i])
}

/// ``pcm`` read at fractional input ``positions`` (any order, either direction), anti-aliased by the speed
/// (each frame's distance from the one before; ``previous``: the position before the first), edges held.
pub fn read_at(t: &ReadTables, pcm: &[f32], ch: usize, positions: &[f64], previous: Option<f64>) -> Vec<f32> {
    let n = pcm.len() / ch;
    let mut out = vec![0f32; positions.len() * ch];
    if n == 0 || positions.is_empty() {
        return out;
    }
    let top = (n - 1) as f64;
    let clip = |x: f64| x.clamp(0.0, top);
    let at = |k: i64, c: usize| pcm[(k.clamp(0, n as i64 - 1) as usize) * ch + c] as f64;
    let mut acc = vec![0f64; ch];
    let (mut lo_f, mut hi_f, mut d_max) = (f64::INFINITY, f64::NEG_INFINITY, 0f64);
    let first = clip(previous.unwrap_or(positions[0]));
    let mut before = first;
    for (m, &p) in positions.iter().enumerate() {
        let pm = clip(p);
        let rm = (pm - before).abs().max(1.0);
        before = pm;
        if rm > STRIDE_STRETCH {
            lo_f = lo_f.min(pm);
            hi_f = hi_f.max(pm);
            d_max = d_max.max(rm / STRIDE_STRETCH);
            continue;
        }
        let reach = (HALF as f64 * rm).min(REACH as f64);
        let sinc_scale = SINC_RES as f64 / rm;
        let win_scale = WIN_RES as f64 / reach;
        acc.iter_mut().for_each(|a| *a = 0.0);
        let mut wsum = 0.0;
        for j in (pm - reach).floor() as i64 + 1..(pm + reach).floor() as i64 + 1 {
            let tt = (j as f64 - pm).abs();
            if tt >= reach {
                continue;
            }
            let w = if rm <= 2.0 {
                lerp(&t.unit, tt * sinc_scale)
            } else {
                lerp(&t.sinc, tt * sinc_scale) * lerp(&t.win, tt * win_scale)
            };
            wsum += w;
            for c in 0..ch {
                acc[c] += w * at(j, c);
            }
        }
        for c in 0..ch {
            out[m * ch + c] = (acc[c] / wsum) as f32;
        }
    }
    if d_max == 0.0 {
        return out;
    }
    // strided taps: c2 is the second running integral of the input over the span
    let span = REACH as f64 * d_max + 2.0;
    let lo = (lo_f - span).floor() as i64;
    let size = ((hi_f + span).ceil() as i64 + 2 - lo) as usize;
    let mut c2 = vec![0f64; (size + 1) * ch];
    let mut c1 = vec![0f64; ch];
    for i in 0..size {
        for c in 0..ch {
            let v = at(lo + i as i64, c);
            c2[(i + 1) * ch + c] = c2[i * ch + c] + c1[c] + 0.5 * v;
            c1[c] += v;
        }
    }
    let taps = t.stride_w.len();
    let mut g0 = vec![0f64; ch];
    let mut g1 = vec![0f64; ch];
    let mut before = first;
    for (m, &p) in positions.iter().enumerate() {
        let pm = clip(p);
        let rm = (pm - before).abs().max(1.0);
        before = pm;
        if rm <= STRIDE_STRETCH {
            continue;
        }
        let d = rm / STRIDE_STRETCH;
        let base = pm - lo as f64 + 0.5;
        let inv = 1.0 / (d * d);
        acc.iter_mut().for_each(|a| *a = 0.0);
        for e in 0..taps + 2 {
            let q = base + (e as f64 - REACH as f64) * d;
            let k = q as usize;
            let f = q - k as f64;
            for c in 0..ch {
                let v = at(lo + k as i64, c);
                let c2k = c2[k * ch + c];
                let g = c2k + f * (c2[(k + 1) * ch + c] - c2k - 0.5 * v + 0.5 * f * v);
                if e >= 2 {
                    acc[c] += t.stride_w[e - 2] * (g - 2.0 * g1[c] + g0[c]) * inv;
                }
                g0[c] = g1[c];
                g1[c] = g;
            }
        }
        for c in 0..ch {
            out[m * ch + c] = acc[c] as f32;
        }
    }
    out
}

/// The record under the hand while scratching (audioengine/python/dsp.py's ScratchPlatter): the hand's samples
/// followed DELAY late (a jitter buffer), carried on past the last one briefly, smoothed twice.
pub struct Platter {
    rate: f64,
    pole: f64,
    times: Vec<f64>,
    path: Vec<f64>,
    rest: Vec<bool>,
    pixel: f64,
    clock: f64,
    zi: [f64; 2],
    pub position: f64,
}

impl Platter {
    const DELAY: f64 = 0.06;
    const EXTRAPOLATE: f64 = 0.02;
    const RATE_SEC: f64 = 0.012;
    const SMOOTH: f64 = 0.008;
    const HISTORY: f64 = 1.0;
    const CLOCK_GAIN: f64 = 0.02;
    const CLOCK_RESYNC: f64 = 0.05;

    pub fn new(rate: f64) -> Platter {
        let mut p = Platter {
            rate,
            pole: (-1.0 / (Self::SMOOTH * rate)).exp(),
            times: vec![],
            path: vec![],
            rest: vec![],
            pixel: 0.0,
            clock: 0.0,
            zi: [0.0; 2],
            position: 0.0,
        };
        p.reset(0.0, 0.0, 0.0);
        p
    }

    pub fn reset(&mut self, pos: f64, now: f64, pixel: f64) {
        self.times = vec![now];
        self.path = vec![pos];
        self.rest = vec![true];
        self.pixel = pixel.abs();
        self.clock = now;
        self.zi = [self.pole * pos; 2];
        self.position = pos;
    }

    pub fn set_target(&mut self, pos: f64, at: f64, rest: bool) {
        let at = at.max(*self.times.last().unwrap());
        self.times.push(at);
        self.path.push(pos);
        self.rest.push(rest);
        let mut cut = 0;
        while cut + 2 < self.times.len() && self.times[cut] < at - Self::HISTORY {
            cut += 1;
        }
        if cut > 0 {
            self.times.drain(..cut);
            self.path.drain(..cut);
            self.rest.drain(..cut);
        }
    }

    fn interp(&self, x: f64) -> f64 {
        let t = &self.times;
        if x <= t[0] {
            return self.path[0];
        }
        let last = t.len() - 1;
        if x >= t[last] {
            return self.path[last];
        }
        let j = t.partition_point(|&v| v <= x); // t[j-1] <= x < t[j]
        let (t0, t1) = (t[j - 1], t[j]);
        if t1 <= t0 {
            return self.path[j];
        }
        self.path[j - 1] + (x - t0) / (t1 - t0) * (self.path[j] - self.path[j - 1])
    }

    /// Input positions of the next ``n`` output frames, the first heard at ``heard_at``.
    pub fn positions(&mut self, n: usize, heard_at: f64) -> Vec<f64> {
        let error = heard_at - self.clock;
        if error.abs() > Self::CLOCK_RESYNC {
            self.clock = heard_at;
        } else {
            self.clock += Self::CLOCK_GAIN * error;
        }
        let start = self.clock;
        self.clock += n as f64 / self.rate;
        let t_last = *self.times.last().unwrap();
        let mut speed = None;
        if self.times.len() >= 2 && !*self.rest.last().unwrap() {
            let mut j = self.times.len() - 2;
            while j > 0 && self.times[j] > t_last - Self::RATE_SEC {
                j -= 1;
            }
            let span = t_last - self.times[j];
            if span > 1e-4 {
                speed = Some((self.path[self.path.len() - 1] - self.path[j]) / span);
            }
        }
        let b = 1.0 - self.pole;
        let mut out = Vec::with_capacity(n);
        for k in 0..n {
            let lookup = start + k as f64 / self.rate - Self::DELAY;
            let mut hand = self.interp(lookup);
            if lookup > t_last {
                if let Some(speed) = speed {
                    let mut ahead = speed * (lookup - t_last).min(Self::EXTRAPOLATE);
                    if self.pixel > 0.0 {
                        ahead = ahead.clamp(-self.pixel, self.pixel);
                    }
                    hand += ahead;
                }
            }
            let y0 = b * hand + self.zi[0];
            self.zi[0] = self.pole * y0;
            let y1 = b * y0 + self.zi[1];
            self.zi[1] = self.pole * y1;
            out.push(y1);
        }
        if let Some(&last) = out.last() {
            self.position = last;
        }
        out
    }
}
