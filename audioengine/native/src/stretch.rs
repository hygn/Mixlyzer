//! Streaming, percussion-aware, pitch-preserving time scaling: audioengine/python/timestretch.py's
//! StreamingTimeStretcher (its docs hold the design), with the FFTs on realfft / rustfft and the vector work
//! on VOLK. ``factor`` is input frames per output frame. The phase vocoder runs on the low band alone, the high
//! band by WSOLA on its frames (see ``Light``); the event detector, the window lengths, the time map and the
//! stream's handovers are the Python one's.

use std::collections::{HashMap, VecDeque};
use std::f64::consts::PI;
use std::rc::Rc;
use std::sync::Arc;

use num_complex::Complex32;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};

use crate::split::{dot, BandSplit};
use crate::vk;

const EPS: f64 = 1e-12;
const TWO_PI: f64 = 2.0 * PI;

#[inline(always)]
fn wrap(v: f64) -> f64 {
    const INV_TWO_PI: f64 = 1.0 / TWO_PI;
    v - TWO_PI * (v * INV_TWO_PI).round_ties_even()
}

fn periodic_hann(n: usize) -> Vec<f32> {
    if n <= 1 {
        return vec![1.0; n];
    }
    (0..n).map(|i| (0.5 - 0.5 * (TWO_PI * i as f64 / n as f64).cos()) as f32).collect()
}

/// np.interp: ``x`` on the polyline (xs, ys), held past its ends.
fn interp(x: f64, xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len();
    if n == 0 {
        return 0.0;
    }
    if x <= xs[0] {
        return ys[0];
    }
    if x >= xs[n - 1] {
        return ys[n - 1];
    }
    let j = xs.partition_point(|&v| v <= x);
    let (x0, x1) = (xs[j - 1], xs[j]);
    if x1 <= x0 {
        return ys[j];
    }
    ys[j - 1] + (x - x0) / (x1 - x0) * (ys[j] - ys[j - 1])
}

#[derive(Clone, Copy)]
pub struct Config {
    pub window_ms: f64,
    pub synthesis_hop_ms: f64,
    pub frequency_oversampling: f64,
    pub minimum_window_ratio: f64,
    pub squeeze_strength: f64,
    pub squeeze_gamma: f64,
    pub kick_profile_stretch: f64,
    pub window_quantum: i64,
    pub detector_magnitude_floor: f64,
    pub mpd_lower: f64,
    pub mpd_upper: f64,
    pub detector_median_frames: i64,
    pub peak_prominence: f64,
    pub prominence_window_ms: f64,
    pub kick_low_hz: f64,
    pub phase_lock_strength: f64,
    pub stereo_coherence: bool,
    pub rtpghi_tolerance: f64,
    pub dry_transient_threshold: f64,
    pub dry_transient_full_strength: f64,
    pub dry_transient_pre_ms: f64,
    pub dry_transient_post_ms: f64,
    pub kick_transient_pre_ms: f64,
    pub kick_transient_post_ms: f64,
    pub attack_filter_pad_ms: f64,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            window_ms: 93.0,
            synthesis_hop_ms: 10.67,
            frequency_oversampling: 1.5,
            minimum_window_ratio: 0.70,
            squeeze_strength: 1.50,
            squeeze_gamma: 1.50,
            kick_profile_stretch: 2.0,
            window_quantum: 16,
            detector_magnitude_floor: 0.01,
            mpd_lower: 0.50,
            mpd_upper: 0.75,
            detector_median_frames: 5,
            peak_prominence: 0.10,
            prominence_window_ms: 90.0,
            kick_low_hz: 180.0,
            phase_lock_strength: 1.0,
            stereo_coherence: true,
            rtpghi_tolerance: 1e-6,
            dry_transient_threshold: 0.40,
            dry_transient_full_strength: 0.68,
            dry_transient_pre_ms: 4.0,
            dry_transient_post_ms: 24.0,
            kick_transient_pre_ms: 10.0,
            kick_transient_post_ms: 24.0,
            attack_filter_pad_ms: 10.0,
        }
    }
}

// --- the 4th-order Butterworth highpass, run forward and backward (scipy's butter + sosfiltfilt) ---

type Sos = [[f64; 6]; 2];

/// Two biquads of a 4th-order Butterworth highpass at ``hz`` (the bilinear transform, prewarped).
fn butter_highpass4(hz: f64, fs: f64) -> Sos {
    let w0 = TWO_PI * hz / fs;
    let (s, c) = w0.sin_cos();
    let mut sos = [[0.0; 6]; 2];
    for (k, q) in [1.0 / (2.0 * (PI / 8.0).cos()), 1.0 / (2.0 * (3.0 * PI / 8.0).cos())].iter().enumerate() {
        let alpha = s / (2.0 * q);
        let a0 = 1.0 + alpha;
        sos[k] = [(1.0 + c) / 2.0 / a0, -(1.0 + c) / a0, (1.0 + c) / 2.0 / a0, 1.0, -2.0 * c / a0, (1.0 - alpha) / a0];
    }
    sos
}

/// lfilter_zi of one section: the state of a unit step's steady state.
fn section_zi(s: &[f64; 6]) -> [f64; 2] {
    let (b0, b1, b2, a1, a2) = (s[0], s[1], s[2], s[4], s[5]);
    // [[1 + a1, -1], [a2, 1]] zi = [b1 - a1 b0, b2 - a2 b0]
    let (r0, r1) = (b1 - a1 * b0, b2 - a2 * b0);
    let det = (1.0 + a1) + a2;
    let z0 = (r0 + r1) / det;
    let z1 = r1 - a2 * z0;
    [z0, z1]
}

fn sosfilt(sos: &Sos, x: &mut [f64], zi: &[[f64; 2]; 2], scale: f64) {
    for (k, s) in sos.iter().enumerate() {
        let (mut z0, mut z1) = (zi[k][0] * scale, zi[k][1] * scale);
        for v in x.iter_mut() {
            let xin = *v;
            let y = s[0] * xin + z0;
            z0 = s[1] * xin - s[4] * y + z1;
            z1 = s[2] * xin - s[5] * y;
            *v = y;
        }
    }
}

/// scipy.signal.sosfiltfilt (odd extension of 3 * (2 * sections + 1) frames) of one channel.
fn sosfiltfilt(sos: &Sos, x: &[f64]) -> Vec<f64> {
    let pad = 15usize;
    let n = x.len();
    let mut zi = [[0.0; 2]; 2];
    let mut scale = 1.0;
    for (k, s) in sos.iter().enumerate() {
        let z = section_zi(s);
        zi[k] = [z[0] * scale, z[1] * scale];
        scale *= (s[0] + s[1] + s[2]) / (s[3] + s[4] + s[5]);
    }
    let mut ext = Vec::with_capacity(n + 2 * pad);
    for i in (1..=pad).rev() {
        ext.push(2.0 * x[0] - x[i]);
    }
    ext.extend_from_slice(x);
    for i in 0..pad {
        ext.push(2.0 * x[n - 1] - x[n - 2 - i]);
    }
    let x0 = ext[0];
    sosfilt(sos, &mut ext, &zi, x0);
    ext.reverse();
    let y0 = ext[0];
    sosfilt(sos, &mut ext, &zi, y0);
    ext.reverse();
    ext[pad..pad + n].to_vec()
}

// --- the percussive-event detector (SELEBI), on a fixed input grid: frame j centred on j * hop ---

#[derive(Clone, Copy)]
struct Event {
    strength: f64,
    kick: f64,
}

struct Detector {
    hop: i64,
    window: usize,
    n_fft: usize,
    bins: usize,
    hann: Vec<f32>,
    ramp: Vec<Complex32>,
    kick_bin: usize,
    wide_bin: usize,
    half_median: i64,
    prominence_frames: i64,
    level_decay: f64,
    floor: f64,
    mpd_lower: f64,
    mpd_upper: f64,
    peak_prominence: f64,
    first: i64,
    next_raw: i64,
    checked: i64,
    raw: HashMap<i64, f64>,
    low: HashMap<i64, f64>,
    median: HashMap<i64, f64>,
    previous_phase: Option<Vec<f64>>,
    level: f64,
    events: HashMap<i64, Event>,
    fft: Arc<dyn RealToComplex<f32>>,
    input: Vec<f32>,
    spectrum: Vec<Complex32>,
    scratch: Vec<Complex32>,
    mag: Vec<f32>,
    ph: Vec<f32>,
}

impl Detector {
    fn new(rate: f64, window: usize, hop: i64, n_fft: usize, cfg: &Config, planner: &mut RealFftPlanner<f32>) -> Self {
        let bins = n_fft / 2 + 1;
        let ramp = (0..bins)
            .map(|k| {
                let a = TWO_PI * k as f64 / n_fft as f64 * (window / 2) as f64;
                Complex32::new(a.cos() as f32, a.sin() as f32)
            })
            .collect();
        let kick_bin = ((cfg.kick_low_hz * n_fft as f64 / rate).round_ties_even() as i64).clamp(2, bins as i64 - 1) as usize;
        let wide_hz = 12000f64.min(0.5 * rate);
        let wide_bin = bins.min((kick_bin + 1).max((wide_hz * n_fft as f64 / rate).round_ties_even() as usize));
        let fft = planner.plan_fft_forward(n_fft);
        let scratch = fft.make_scratch_vec();
        let mut d = Detector {
            hop,
            window,
            n_fft,
            bins,
            hann: periodic_hann(window),
            ramp,
            kick_bin,
            wide_bin,
            half_median: cfg.detector_median_frames / 2,
            prominence_frames: ((cfg.prominence_window_ms * rate / 1000.0 / hop as f64).round_ties_even() as i64).max(1),
            level_decay: 0.5f64.powf(hop as f64 / (4.0 * rate)),
            floor: cfg.detector_magnitude_floor,
            mpd_lower: cfg.mpd_lower,
            mpd_upper: cfg.mpd_upper,
            peak_prominence: cfg.peak_prominence,
            first: 0,
            next_raw: 0,
            checked: 1,
            raw: HashMap::new(),
            low: HashMap::new(),
            median: HashMap::new(),
            previous_phase: None,
            level: 0.0,
            events: HashMap::new(),
            input: vec![0.0; n_fft],
            spectrum: vec![Complex32::new(0.0, 0.0); bins],
            scratch,
            mag: vec![0.0; bins],
            ph: vec![0.0; bins],
            fft,
        };
        d.start(0);
        d
    }

    fn start(&mut self, first: i64) {
        self.first = first;
        self.next_raw = first;
        self.checked = first + 1;
        self.raw.clear();
        self.low.clear();
        self.median.clear();
        self.previous_phase = None;
        self.level = 0.0;
        self.events.clear();
    }

    /// The raw frames ``ensure(frame)`` reads, at most ``budget`` of them now; true once all are there.
    fn prepare(&mut self, source: &[f32], ch: usize, frame: i64, mut budget: usize) -> bool {
        let last = frame + self.prominence_frames + self.half_median + 1;
        while self.next_raw <= last && budget > 0 {
            self.compute_raw(source, ch, self.next_raw);
            self.next_raw += 1;
            budget -= 1;
        }
        self.next_raw > last
    }

    /// Confirms or rejects the peaks at every frame up to ``frame``.
    fn ensure(&mut self, source: &[f32], ch: usize, frame: i64) {
        let lookahead = self.prominence_frames + self.half_median + 1;
        while self.checked <= frame {
            while self.next_raw <= self.checked + lookahead {
                self.compute_raw(source, ch, self.next_raw);
                self.next_raw += 1;
            }
            self.check_peak(self.checked);
            self.checked += 1;
        }
        self.prune(frame);
    }

    fn compute_raw(&mut self, source: &[f32], ch: usize, j: i64) {
        let window = self.window as i64;
        let frames = (source.len() / ch) as i64;
        let left = j * self.hop - window / 2;
        self.input.iter_mut().for_each(|v| *v = 0.0);
        let (lo, hi) = (left.max(0), frames.min(left + window));
        for i in lo..hi {
            let s = &source[i as usize * ch..(i as usize + 1) * ch];
            self.input[(i - left) as usize] = s.iter().sum::<f32>() / ch as f32;
        }
        vk::mul_in(&mut self.input[..self.window], &self.hann);
        self.fft.process_with_scratch(&mut self.input, &mut self.spectrum, &mut self.scratch).ok();
        vk::cmul_in(&mut self.spectrum, &self.ramp);
        vk::magnitude(&mut self.mag, &self.spectrum);
        vk::phase(&mut self.ph, &self.spectrum);

        let bins = self.bins;
        let peak = self.mag.iter().fold(0f32, |m, &v| m.max(v)) as f64;
        self.level = peak.max(self.level * self.level_decay);
        let mut ratio = 0.0;
        if let Some(prev) = &self.previous_phase {
            if bins > 1 {
                let bin_width = TWO_PI / self.n_fft as f64;
                let floor = self.floor * self.level;
                let (lower, upper) = (1.0 - self.mpd_lower, 1.0 + self.mpd_upper);
                let hop = self.hop as f64;
                let (mut total, mut percussive) = (0.0, 0.0);
                let mut dt = wrap(self.ph[0] as f64 - prev[0]) / hop;
                let mut mixed = 0.0;
                for m in 0..bins {
                    if m + 1 < bins {
                        let dt_next = wrap(self.ph[m + 1] as f64 - prev[m + 1]) / hop;
                        mixed = wrap(dt_next - dt) / bin_width;
                        dt = dt_next;
                    }
                    let mag = self.mag[m] as f64;
                    total += mag;
                    if mag > floor && mixed > lower && mixed < upper {
                        percussive += mag;
                    }
                }
                ratio = percussive / (total + EPS);
            }
        }
        let prev = self.previous_phase.get_or_insert_with(|| vec![0.0; bins]);
        for m in 0..bins {
            prev[m] = self.ph[m] as f64;
        }
        let (mut low, mut wide) = (0.0, 0.0);
        for m in 1..self.wide_bin {
            let p = self.mag[m] as f64 * self.mag[m] as f64;
            wide += p;
            if m < self.kick_bin {
                low += p;
            }
        }
        self.raw.insert(j, ratio);
        self.low.insert(j, low / (wide + EPS));
    }

    fn median_at(&mut self, j: i64) -> f64 {
        if let Some(&v) = self.median.get(&j) {
            return v;
        }
        let h = self.half_median;
        let mut w: Vec<f64> =
            (j - h..=j + h).map(|k| self.raw[&k.max(self.first).min(self.next_raw - 1)]).collect();
        w.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = w.len();
        let v = if n % 2 == 1 { w[n / 2] } else { 0.5 * (w[n / 2 - 1] + w[n / 2]) };
        self.median.insert(j, v);
        v
    }

    fn check_peak(&mut self, j: i64) {
        let value = self.median_at(j);
        if !(value > self.median_at(j - 1) && value >= self.median_at(j + 1)) {
            return;
        }
        let span = self.prominence_frames;
        let mut left_min = value;
        let mut k = j - 1;
        while k >= (self.first).max(j - span) {
            let other = self.median_at(k);
            if other > value {
                break;
            }
            left_min = left_min.min(other);
            k -= 1;
        }
        let mut right_min = value;
        for k in j + 1..=j + span {
            let other = self.median_at(k);
            if other > value {
                break;
            }
            right_min = right_min.min(other);
        }
        if value - left_min.max(right_min) < self.peak_prominence {
            return;
        }
        let low_score = ((self.low[&j] - 0.25) / 0.35).clamp(0.0, 1.0);
        let percussion_score = ((value - 0.30) / 0.30).clamp(0.0, 1.0);
        self.events.insert(j, Event { strength: value, kick: (low_score * percussion_score).sqrt() });
    }

    fn prune(&mut self, frame: i64) {
        let keep_from = frame - 8 * (self.prominence_frames + self.half_median + 8);
        if self.raw.len() < 4096 || keep_from <= self.first {
            return;
        }
        self.raw.retain(|&k, _| k >= keep_from);
        self.low.retain(|&k, _| k >= keep_from);
        self.median.retain(|&k, _| k >= keep_from);
        self.events.retain(|&k, _| k >= keep_from);
        self.first = keep_from;
    }
}

// --- the phase kernels ---

/// Identity phase locking around the joint peaks, then the phases wrapped.
fn lock_and_wrap(
    out: &mut [f64],
    analysis: &[f64],
    joint: &[f32],
    channels: usize,
    bins: usize,
    lock_strength: f64,
    work: &mut PhaseWork,
) {
    if lock_strength > 0.0 && bins >= 3 {
        let mut peak_at = 0;
        let mut peak_value = EPS;
        for m in 0..bins {
            if joint[m] as f64 > peak_value {
                peak_value = joint[m] as f64;
                peak_at = m;
            }
        }
        let threshold = 1e-3 * peak_value;
        let peaks = &mut work.peaks;
        peaks.clear();
        for m in 1..bins - 1 {
            if joint[m] >= joint[m - 1] && joint[m] > joint[m + 1] && joint[m] as f64 > threshold {
                peaks.push(m);
            }
        }
        if peaks.is_empty() {
            peaks.push(peak_at);
        }
        let mut start = 0;
        for index in 0..peaks.len() {
            let peak = peaks[index];
            let stop = if index + 1 < peaks.len() { (peak + peaks[index + 1]) / 2 + 1 } else { bins };
            for m in start..stop {
                for c in 0..channels {
                    let b = c * bins;
                    let locked = out[b + peak] + wrap(analysis[b + m] - analysis[b + peak]);
                    out[b + m] += lock_strength * wrap(locked - out[b + m]);
                }
            }
            start = stop;
        }
    }
    for v in out.iter_mut() {
        *v = wrap(*v);
    }
}

/// RTPGHI: the synthesis phases of one frame (bins integrated in order of magnitude, in time from the previous
/// frame or in frequency from a neighbour), then identity phase locking around the joint peaks. The order is kept
/// by a bucket queue in place of the Python one's heap and sorted seeds: magnitudes ordered to 1/8 octave (0.75
/// dB; the f32's exponent and three mantissa bits), each push and pop O(1). Bins within one bucket go in last-in
/// first-out order: the order of integration is RTPGHI's heuristic, not its result's definition.
#[allow(clippy::too_many_arguments)]
fn rtpghi_frame(
    out: &mut [f64],
    analysis: &[f64],
    joint: &[f32],
    previous_analysis: &[f64],
    previous_synthesis: &[f64],
    previous_joint: &[f32],
    channels: usize,
    bins: usize,
    analysis_hop: i64,
    synthesis_hop: i64,
    n_fft: usize,
    tolerance: f64,
    lock_strength: f64,
    work: &mut PhaseWork,
) {
    const SHIFT: u32 = 20;
    out.copy_from_slice(analysis);
    let ha = analysis_hop.max(1) as f64;
    let hs = synthesis_hop.max(1) as f64;
    let local = hs / ha;
    let mut max_mag = EPS;
    for m in 0..bins {
        max_mag = max_mag.max(previous_joint[m] as f64).max(joint[m] as f64);
    }
    let floor = tolerance * max_mag;
    let floor_key = (floor as f32).to_bits() >> SHIFT;
    let count = (((max_mag as f32).to_bits() >> SHIFT).saturating_sub(floor_key) + 2) as usize;
    let key = |v: f32| -> usize { ((v.to_bits() >> SHIFT).saturating_sub(floor_key) as usize).min(count - 1) };
    let buckets = &mut work.buckets;
    if buckets.len() < count {
        buckets.resize_with(count, Vec::new);
    }
    for b in buckets.iter_mut().take(count) {
        b.clear();
    }
    let remaining = &mut work.remaining;
    remaining.clear();
    remaining.resize(bins, false);
    let mut remaining_count = 0usize;
    let mut top = 0usize;
    let mut queued = 0usize;
    // the seeds: a bin's integration in time, keyed by its previous magnitude (item 2m)
    for m in 0..bins {
        if joint[m] as f64 > floor {
            remaining[m] = true;
            remaining_count += 1;
            let k = key(previous_joint[m]);
            buckets[k].push(2 * m as u32);
            top = top.max(k);
            queued += 1;
        }
    }
    let time_step = |out: &mut [f64], m: usize| {
        let omega = TWO_PI * m as f64 / n_fft as f64;
        for c in 0..channels {
            let i = c * bins + m;
            let residual = wrap(analysis[i] - previous_analysis[i] - omega * ha);
            out[i] = previous_synthesis[i] + omega * hs + local * residual;
        }
    };
    while remaining_count > 0 {
        if queued == 0 {
            let mut m = 0;
            while m < bins && !remaining[m] {
                m += 1;
            }
            time_step(out, m);
            remaining[m] = false;
            remaining_count -= 1;
            let k = key(joint[m]);
            buckets[k].push(2 * m as u32 + 1);
            top = top.max(k);
            queued += 1;
        }
        while buckets[top].is_empty() {
            top -= 1;
        }
        let item = buckets[top].pop().unwrap() as usize;
        queued -= 1;
        let m = item >> 1;
        if item & 1 == 0 {
            if remaining[m] {
                time_step(out, m);
                remaining[m] = false;
                remaining_count -= 1;
                let k = key(joint[m]);
                buckets[k].push(2 * m as u32 + 1);
                top = top.max(k);
                queued += 1;
            }
            continue;
        }
        let up = m + 1;
        if up < bins && remaining[up] {
            for c in 0..channels {
                let b = c * bins;
                out[b + up] = out[b + m] + local * wrap(analysis[b + up] - analysis[b + m]);
            }
            remaining[up] = false;
            remaining_count -= 1;
            let k = key(joint[up]);
            buckets[k].push(2 * up as u32 + 1);
            top = top.max(k);
            queued += 1;
        }
        if m >= 1 && remaining[m - 1] {
            let down = m - 1;
            for c in 0..channels {
                let b = c * bins;
                out[b + down] = out[b + m] - local * wrap(analysis[b + m] - analysis[b + down]);
            }
            remaining[down] = false;
            remaining_count -= 1;
            let k = key(joint[down]);
            buckets[k].push(2 * down as u32 + 1);
            top = top.max(k);
            queued += 1;
        }
    }
    lock_and_wrap(out, analysis, joint, channels, bins, lock_strength, work);
}

/// One rotation per bin shared by all channels (the input's L/R phase relation kept): the synthesis phases
/// and the analysis coefficients turned by that rotation. Vectorised on VOLK: each channel's rotation from its
/// analysis phase (f32: its sine and cosine to 1e-7), their magnitude-weighted sum, its angle and norm.
fn coherent_synthesis(
    phase: &mut [f64],
    analysis: &[f64],
    magnitude: &[f32],
    coeff: &mut [Complex32],
    channels: usize,
    bins: usize,
    w: &mut CoherentWork,
) {
    w.resize(bins);
    w.sum.iter_mut().for_each(|v| *v = Complex32::new(0.0, 0.0));
    for c in 0..channels {
        let b = c * bins;
        for m in 0..bins {
            w.rot[m] = (phase[b + m] - analysis[b + m]) as f32;
        }
        vk::sin(&mut w.sin, &w.rot);
        vk::cos(&mut w.cos, &w.rot);
        let mag = &magnitude[b..b + bins];
        for m in 0..bins {
            w.sum[m].re += mag[m] * w.cos[m];
            w.sum[m].im += mag[m] * w.sin[m];
        }
    }
    vk::phase(&mut w.common, &w.sum);
    vk::magnitude(&mut w.norm, &w.sum);
    for m in 0..bins {
        let n = w.norm[m];
        w.turn[m] = if n > 0.0 { w.sum[m] / n } else { Complex32::new(1.0, 0.0) };
    }
    for c in 0..channels {
        let b = c * bins;
        for m in 0..bins {
            phase[b + m] = wrap(analysis[b + m] + w.common[m] as f64);
        }
        vk::cmul_in(&mut coeff[b..b + bins], &w.turn);
    }
}

#[derive(Default)]
struct CoherentWork {
    rot: Vec<f32>,
    sin: Vec<f32>,
    cos: Vec<f32>,
    sum: Vec<Complex32>,
    common: Vec<f32>,
    norm: Vec<f32>,
    turn: Vec<Complex32>,
}

impl CoherentWork {
    fn resize(&mut self, bins: usize) {
        if self.rot.len() != bins {
            let z = Complex32::new(0.0, 0.0);
            *self = CoherentWork {
                rot: vec![0.0; bins],
                sin: vec![0.0; bins],
                cos: vec![0.0; bins],
                sum: vec![z; bins],
                common: vec![0.0; bins],
                norm: vec![0.0; bins],
                turn: vec![z; bins],
            };
        }
    }
}

#[derive(Default)]
struct PhaseWork {
    buckets: Vec<Vec<u32>>,
    remaining: Vec<bool>,
    peaks: Vec<usize>,
}

/// An attack's reach in the output around its destination (before, after).
#[derive(Clone, Copy)]
struct Attack {
    pre: i64,
    post: i64,
}

/// The analysis and synthesis windows for one low-rate window length ``wl``: the analysis at the low rate, the
/// synthesis the same window at the full rate (``d * wl``).
struct LightWindow {
    analysis: Vec<f32>,
    center_ramp: Vec<Complex32>,
    inverse_ramp: Vec<Complex32>,
    synthesis: Vec<f32>,
    synthesis_sq: Vec<f32>,
}

/// The band split's part of the stretcher (a full-band vocoder's work at about 40 %): the phase vocoder on the low
/// band only (below 7 - 8 kHz, decimated by ``d``: a third of the bins, the RTPGHI's queue with them, a bucket
/// queue: ``rtpghi_frame``), synthesised at the full rate from the low band's bins (a zero-padded
/// inverse FFT: the band-limited interpolation itself); the high band (the residual: the bands sum to the
/// source, split.rs) by WSOLA on the vocoder's own frames: each frame's input and output centres are the
/// vocoder's, so both bands follow one time map. WSOLA looks within 5 ms for the segment continuing the
/// previous one (the high band's partials, a synth's or a voice's harmonics, align only over their
/// fundamental's period: within 0.5 ms they warbled), its splices' level kept whatever their correlation.
/// The attack is the source's own: at each event (its onset found in the source, ``onset``, and carried to
/// its destination on the time map) the high band, and the vocoder's band above 200 Hz where the event is strong
/// (a dry or a kick attack), are crossfaded to the source's (``restore_light``), so every band's attack lands at
/// the same time and none is overlap-added; the bass below keeps its own phase.
struct Light {
    split: BandSplit,
    rate: f64,
    hop_low: i64,
    fft: Arc<dyn RealToComplex<f32>>,
    ifft: Arc<dyn ComplexToReal<f32>>,
    fft_scratch: Vec<Complex32>,
    ifft_scratch: Vec<Complex32>,
    fft_in: Vec<f32>,
    fft_out: Vec<Complex32>,
    synth_spec: Vec<Complex32>,
    synth_out: Vec<f32>,
    windows: HashMap<usize, Rc<LightWindow>>,
    previous_center: Option<i64>,
    // WSOLA: segments of two hops (a periodic Hann: at one hop apart they sum to 1), placed within
    // ``tolerance`` of the vocoder's input centre where they best continue the previous one
    segment_window: Vec<f32>,
    tolerance: i64,
    previous_start: Option<i64>,
    energy: Vec<f64>,
    // the candidates' cross-correlation with the continuation, by FFT (a length of at least 2 tolerance + 2 hops)
    xc_fft: Arc<dyn RealToComplex<f32>>,
    xc_ifft: Arc<dyn ComplexToReal<f32>>,
    xc_in: Vec<f32>,
    xc_spec: Vec<Complex32>,
    xc_ref: Vec<Complex32>,
    xc_sum: Vec<Complex32>,
    xc_out: Vec<f32>,
    xc_scratch: Vec<Complex32>,
}

/// The smallest even length of the factors 2, 3 and 5 from ``n`` up (a fast size for rustfft).
fn fast_even_length(n: usize) -> usize {
    let mut m = n.max(2) + (n & 1);
    loop {
        let mut r = m;
        for f in [2, 3, 5] {
            while r % f == 0 {
                r /= f;
            }
        }
        if r == 1 {
            return m;
        }
        m += 2;
    }
}

pub struct Stretcher {
    ch: usize,
    cfg: Config,
    max_window: usize,
    hop: i64,
    n_fft: usize,
    bins: usize,
    min_window: usize,
    detector: Detector,
    event_reach: i64,
    handover_timeout: i64,
    dry: Attack,
    kick: Attack,
    filter_pad: i64,
    envelope_window: usize,
    attack_ahead: i64,

    source_token: Option<(usize, usize)>,
    factor: Option<f64>,
    expected_pos: Option<f64>,
    next_analysis_center: f64,
    next_synthesis_center: i64,
    previous_analysis_phase: Vec<f64>,
    previous_synthesis_phase: Vec<f64>,
    previous_joint: Vec<f32>,
    has_previous: bool,

    output_read: i64,
    finalized_until: i64,
    ola_base: i64,
    ola: Vec<f32>, // [len, ch]
    weight: Vec<f32>,
    correction: Vec<f32>, // [len, ch]

    frame_map: VecDeque<(i64, i64)>,
    next_event_frame: i64,
    pending_events: VecDeque<(i64, i64, Event)>, // (destination, the input sample aligned to it, the event)

    streaming: bool,
    stream_gain: f64,
    gain_target: f64,
    unity_frames: i64,
    handover_at: Option<i64>,
    anchor_in: f64,
    anchor_out: i64,
    warming: bool,

    // per-frame work
    frame_buf: Vec<f32>,
    coeff: Vec<Complex32>,       // [ch, bins]
    analysis_phase: Vec<f64>,    // [ch, bins]
    synthesis_phase: Vec<f64>,   // [ch, bins]
    magnitude: Vec<f32>,         // [ch, bins]
    joint: Vec<f32>,             // [bins]
    phase32: Vec<f32>,
    phase_work: PhaseWork,
    coherent_work: CoherentWork,
    light: Option<Box<Light>>, // always there: taken out while a frame works on it beside the rest
    ola_high: Vec<f32>,        // [len, ch]: the high band
}

impl Stretcher {
    const UNITY_EPS: f64 = 1e-8;
    const POSITION_EPS: f64 = 1e-4;
    const COMPACT_AFTER: i64 = 32_768;
    const XFADE_FRAMES: f64 = 256.0;
    const CATCHUP_RATIO: f64 = 0.02;
    const ONSET_FLUX: f64 = 0.2;
    const HANDOVER_TIMEOUT_SEC: f64 = 1.0;
    const KICK_CONFIDENCE_FROM: f64 = 0.70;
    const KICK_CONFIDENCE_SPAN: f64 = 0.25;
    const WARM_EXTRA_FRAMES: i64 = 1;
    const WARM_DETECTOR_FRAMES: usize = 8;

    /// The stretcher at ``sample_rate``, 44.1 kHz or more (ipc::MIN_RATE: the engine plays at no less, and below
    /// 28 kHz no band is left to split).
    pub fn new(channels: usize, sample_rate: u32) -> Stretcher {
        let cfg = Config::default();
        let rate = sample_rate as f64;
        let split = BandSplit::new(channels, sample_rate).expect("the time stretch at 44.1 kHz or more");
        let d = split.d;
        // the detector's grid and the window lengths as the Python stretcher's at this rate
        let requested = 128.max((rate * cfg.window_ms / 1000.0).round_ties_even() as usize);
        let full_window = requested + (requested & 1);
        let full_hop = 1.max((rate * cfg.synthesis_hop_ms / 1000.0).round_ties_even() as i64);
        let detector_fft = ((full_window as f64 * cfg.frequency_oversampling).ceil() as usize).next_power_of_two();
        let min_window = 16.max((cfg.minimum_window_ratio * full_window as f64).ceil() as usize);
        let mut planner = RealFftPlanner::<f32>::new();
        let detector = Detector::new(rate, full_window, full_hop, detector_fft, &cfg, &mut planner);
        let event_reach = (full_window as f64 / (2.0 * full_hop as f64)).ceil() as i64;
        let handover_timeout = 1.max((rate * Self::HANDOVER_TIMEOUT_SEC / full_hop as f64).round_ties_even() as i64);
        let ms = |v: f64| 1.max((v * rate / 1000.0).round_ties_even() as i64);
        let dry = Attack { pre: ms(cfg.dry_transient_pre_ms), post: ms(cfg.dry_transient_post_ms) };
        let kick = Attack { pre: ms(cfg.kick_transient_pre_ms), post: ms(cfg.kick_transient_post_ms) };
        let filter_pad = ms(cfg.attack_filter_pad_ms);
        let envelope_window = 3.max(ms(1.0) as usize);
        let attack_ahead = dry.pre.max(kick.pre) + dry.post.max(kick.post) + filter_pad + 1;
        // the vocoder at the low band's rate (1 / d), its hop a whole number of low-rate samples
        let hop_low = 1.max((full_hop as f64 / d as f64).round_ties_even() as i64);
        let hop = d as i64 * hop_low;
        let wl_max = Self::low_length(d, full_window);
        let max_window = full_window.max(d * wl_max);
        let n_fft = fast_even_length((wl_max as f64 * cfg.frequency_oversampling).ceil() as usize);
        let n_fft_full = d * n_fft;
        let bins = n_fft / 2 + 1;
        let zc = Complex32::new(0.0, 0.0);
        let fft = planner.plan_fft_forward(n_fft);
        let ifft = planner.plan_fft_inverse(n_fft_full);
        let segment = 2 * hop as usize;
        let tolerance = 1.max((rate * Self::WSOLA_TOLERANCE_MS / 1000.0).round_ties_even() as i64);
        let xc_n = fast_even_length(2 * tolerance as usize + 2 * hop as usize);
        let xc_fft = planner.plan_fft_forward(xc_n);
        let xc_ifft = planner.plan_fft_inverse(xc_n);
        let xc_scratch = vec![zc; xc_fft.get_scratch_len().max(xc_ifft.get_scratch_len())];
        let light = Light {
            hop_low,
            fft_scratch: fft.make_scratch_vec(),
            ifft_scratch: ifft.make_scratch_vec(),
            fft_in: vec![0.0; n_fft],
            fft_out: vec![zc; bins],
            synth_spec: vec![zc; n_fft_full / 2 + 1],
            synth_out: vec![0.0; n_fft_full],
            fft,
            ifft,
            windows: HashMap::new(),
            previous_center: None,
            segment_window: periodic_hann(segment),
            tolerance,
            rate,
            previous_start: None,
            energy: Vec::new(),
            xc_in: vec![0.0; xc_n],
            xc_spec: vec![zc; xc_n / 2 + 1],
            xc_ref: vec![zc; xc_n / 2 + 1],
            xc_sum: vec![zc; xc_n / 2 + 1],
            xc_out: vec![0.0; xc_n],
            xc_scratch,
            xc_fft,
            xc_ifft,
            split,
        };
        let mut s = Stretcher {
            ch: channels,
            cfg,
            max_window,
            hop,
            n_fft,
            bins,
            min_window,
            detector,
            event_reach,
            handover_timeout,
            dry,
            kick,
            filter_pad,
            envelope_window,
            attack_ahead,
            source_token: None,
            factor: None,
            expected_pos: None,
            next_analysis_center: 0.0,
            next_synthesis_center: 0,
            previous_analysis_phase: vec![0.0; channels * bins],
            previous_synthesis_phase: vec![0.0; channels * bins],
            previous_joint: vec![0.0; bins],
            has_previous: false,
            output_read: 0,
            finalized_until: 0,
            ola_base: 0,
            ola: vec![],
            weight: vec![],
            correction: vec![],
            frame_map: VecDeque::with_capacity(512),
            next_event_frame: 0,
            pending_events: VecDeque::new(),
            streaming: false,
            stream_gain: 0.0,
            gain_target: 0.0,
            unity_frames: 0,
            handover_at: None,
            anchor_in: 0.0,
            anchor_out: 0,
            warming: false,
            frame_buf: vec![0.0; detector_fft],
            coeff: vec![zc; channels * bins],
            analysis_phase: vec![0.0; channels * bins],
            synthesis_phase: vec![0.0; channels * bins],
            magnitude: vec![0.0; channels * bins],
            joint: vec![0.0; bins],
            phase32: vec![0.0; bins],
            phase_work: PhaseWork::default(),
            coherent_work: CoherentWork::default(),
            light: Some(Box::new(light)),
            ola_high: vec![],
        };
        s.reset();
        s
    }

    const WSOLA_TOLERANCE_MS: f64 = 5.0;
    /// The attack: the vocoder's band crossfaded to the source's above this.
    const ATTACK_LOW_CUTOFF_HZ: f64 = 200.0;

    /// A full-rate window length's low-rate one (even).
    fn low_length(d: usize, length: usize) -> usize {
        let wl = 16.max((length as f64 / d as f64).round_ties_even() as usize);
        wl + (wl & 1)
    }

    pub fn expected_input_position(&self) -> Option<f64> {
        self.expected_pos
    }

    /// Input frames per output frame the next call advances: ``factor``, or 1 while a new stream warms up.
    pub fn input_step(&self, factor: f64) -> f64 {
        if self.warming { 1.0 } else { factor }
    }

    pub fn reset(&mut self) {
        self.source_token = None;
        self.factor = None;
        self.expected_pos = None;
        self.has_previous = false;
        self.output_read = 0;
        self.finalized_until = 0;
        self.ola_base = 0;
        self.ola.clear();
        self.weight.clear();
        self.correction.clear();
        self.ola_high.clear();
        if let Some(light) = &mut self.light {
            light.previous_center = None;
            light.previous_start = None;
            light.split.reset();
        }
        self.frame_map.clear();
        self.pending_events.clear();
        self.streaming = false;
        self.stream_gain = 0.0;
        self.gain_target = 0.0;
        self.unity_frames = 0;
        self.handover_at = None;
        self.warming = false;
    }

    fn token(&self, source: &[f32]) -> (usize, usize) {
        (source.as_ptr() as usize, source.len())
    }

    /// Up to ``max_out`` frames from input ``pos`` (the next sequential call reads from
    /// ``expected_input_position``).
    pub fn render(&mut self, source: &[f32], pos: f64, factor: f64, max_out: usize) -> Vec<f32> {
        let frames = source.len() / self.ch;
        if max_out == 0 || pos >= frames as f64 || !(factor > 0.0) || !pos.is_finite() {
            return Vec::new();
        }
        let available = (frames as f64 - pos) / factor;
        let snap = 32.0 * f64::EPSILON * 1f64.max(frames as f64).max(available.abs());
        let count = (max_out as f64).min((available - snap).ceil());
        if count <= 0.0 {
            return Vec::new();
        }
        let count = count as usize;
        let token = self.token(source);
        let unity = (factor - 1.0).abs() <= Self::UNITY_EPS;
        let sequential = self.continues(token, pos, factor);
        self.source_token = Some(token);
        if !sequential {
            self.streaming = false;
            self.warming = false;
            self.stream_gain = if unity { 0.0 } else { 1.0 };
        }
        if self.warming && unity {
            self.streaming = false;
            self.warming = false;
            self.stream_gain = 0.0;
        }
        if !self.streaming {
            if unity {
                let result = self.read_direct(source, pos, count);
                self.factor = Some(factor);
                self.expected_pos = Some(pos + factor * count as f64);
                return result;
            }
            self.streaming = true;
            if !sequential {
                self.start_stream(token, pos, factor);
            } else {
                self.start_stream(token, pos, 1.0);
                self.warming = true;
                self.stream_gain = 0.0;
                self.gain_target = 0.0;
            }
        }
        if self.warming {
            if let Some(r) = self.render_warming(source, pos, count) {
                return r;
            }
        }
        let current = self.factor.unwrap_or(factor);
        if (factor - current).abs() > (1e-9f64).max(1e-7 * factor.abs().max(current.abs())) {
            self.retime(pos, factor);
        }
        if !unity {
            self.unity_frames = 0;
            self.handover_at = None;
            self.gain_target = 1.0;
        }
        let start = self.output_read;
        while self.finalized_until < start + count as i64 + self.attack_ahead {
            self.process_frame(source);
        }
        self.apply_due_events(source);
        if unity && self.handover_at.is_some() {
            self.gain_target = 0.0;
        }
        let mut result = self.read_output(count);
        self.compact_output();
        if self.stream_gain < 1.0 || self.gain_target < 1.0 {
            result = self.mix_direct(source, pos, start, result);
            if self.stream_gain <= 0.0 && self.gain_target <= 0.0 {
                self.streaming = false;
            }
        }
        self.expected_pos = Some(pos + factor * count as f64);
        result
    }

    /// Starts a stream at input ``pos`` to be read later: ``prepare_step`` does its work a little at a time.
    pub fn prepare(&mut self, source: &[f32], pos: f64, factor: f64) {
        self.warming = false;
        self.streaming = true;
        let token = self.token(source);
        self.start_stream(token, pos, factor);
        self.stream_gain = 1.0;
        self.gain_target = 1.0;
    }

    /// Some of a prepared stream's work; true once its first ``count`` output frames can be read without more.
    pub fn prepare_step(&mut self, source: &[f32], count: usize) -> bool {
        let need = self.output_read + count as i64 + self.attack_ahead;
        if self.finalized_until >= need {
            return true;
        }
        let next_center = ((self.next_analysis_center + 0.5).floor() / self.detector.hop as f64).round_ties_even() as i64;
        if self.detector.prepare(source, self.ch, next_center + self.event_reach + 1, Self::WARM_DETECTOR_FRAMES) {
            self.process_frame(source);
        }
        self.unity_frames = 0;
        self.handover_at = None;
        self.finalized_until >= need
    }

    fn render_warming(&mut self, source: &[f32], pos: f64, count: usize) -> Option<Vec<f32>> {
        let need = self.output_read + count as i64 + self.attack_ahead;
        let mut budget = (count as f64 / self.hop as f64).ceil() as i64 + Self::WARM_EXTRA_FRAMES;
        let next_center = ((self.next_analysis_center + 0.5).floor() / self.detector.hop as f64).round_ties_even() as i64;
        if !self.detector.prepare(source, self.ch, next_center + self.event_reach + 1, Self::WARM_DETECTOR_FRAMES) {
            budget = 0;
        }
        while self.finalized_until < need && budget > 0 {
            self.process_frame(source);
            budget -= 1;
        }
        self.unity_frames = 0;
        self.handover_at = None;
        if self.finalized_until < need {
            let r = self.read_direct(source, pos, count);
            self.output_read += count as i64;
            self.expected_pos = Some(pos + count as f64);
            return Some(r);
        }
        self.warming = false;
        self.gain_target = 1.0;
        None
    }

    fn continues(&self, token: (usize, usize), pos: f64, factor: f64) -> bool {
        match (self.source_token, self.expected_pos) {
            (Some(t), Some(e)) if t == token => {
                (pos - e).abs() <= Self::POSITION_EPS.max(factor * Self::POSITION_EPS)
            }
            _ => false,
        }
    }

    fn retime(&mut self, pos: f64, factor: f64) {
        self.factor = Some(factor);
        self.anchor_in = pos;
        self.anchor_out = self.output_read;
    }

    fn mix_direct(&mut self, source: &[f32], pos: f64, start: i64, stream: Vec<f32>) -> Vec<f32> {
        let ch = self.ch;
        let count = stream.len() / ch;
        let step = 1.0 / Self::XFADE_FRAMES;
        let direction = if self.gain_target > self.stream_gain { 1.0 } else { -1.0 };
        let mut first = start;
        if direction < 0.0 {
            if let Some(h) = self.handover_at {
                first = start.max(h);
            }
        }
        let direct = self.read_direct(source, pos, count);
        let mut out = vec![0f32; count * ch];
        let mut gain = self.stream_gain;
        for k in 0..count {
            let moved = (start + k as i64 - first + 1).max(0) as f64;
            gain = (self.stream_gain + direction * step * moved).clamp(0.0, 1.0);
            let g = gain as f32;
            for c in 0..ch {
                let i = k * ch + c;
                out[i] = direct[i] + g * (stream[i] - direct[i]);
            }
        }
        if count > 0 {
            self.stream_gain = gain;
        }
        out
    }

    fn start_stream(&mut self, token: (usize, usize), pos: f64, factor: f64) {
        self.source_token = Some(token);
        self.factor = Some(factor);
        self.expected_pos = Some(pos);
        self.has_previous = false;
        self.unity_frames = 0;
        self.handover_at = None;
        self.gain_target = 1.0;
        let pre_roll = ((self.max_window / 2) as f64 / self.hop as f64).ceil() as i64;
        self.next_synthesis_center = -pre_roll * self.hop;
        self.next_analysis_center = pos + factor * self.next_synthesis_center as f64;
        self.anchor_in = pos;
        self.anchor_out = 0;
        let first_center = self.next_analysis_center;
        let dh = self.detector.hop as f64;
        let first = (first_center / dh).floor() as i64 - self.event_reach - self.detector.prominence_frames - 2;
        self.detector.start(first);
        self.next_event_frame = (first_center / dh).floor() as i64;
        self.frame_map.clear();
        self.pending_events.clear();
        self.output_read = 0;
        self.finalized_until = 0;
        self.ola_base = 0;
        let initial = self.max_window + 2 * self.hop as usize + self.attack_ahead as usize;
        self.ola = vec![0.0; initial * self.ch];
        self.weight = vec![0.0; initial];
        self.correction = vec![0.0; initial * self.ch];
        if let Some(light) = &mut self.light {
            self.ola_high = vec![0.0; initial * self.ch];
            light.previous_center = None;
            light.previous_start = None;
            light.split.reset();
        }
    }

    fn process_frame(&mut self, source: &[f32]) {
        let factor = self.factor.unwrap_or(1.0);
        let input_center = (self.next_analysis_center + 0.5).floor() as i64;
        let output_center = self.next_synthesis_center;
        let (window_length, lock_strength) = self.frame_window(source, input_center);
        let mut light = self.light.take().expect("the stretcher's bands");
        // the vocoder's frame at the low band's rate
        let d = light.split.d as i64;
        let center = (input_center as f64 / d as f64).round_ties_even() as i64;
        let analysis_hop = match light.previous_center {
            Some(p) => 1.max(center - p),
            None => 1.max((factor * light.hop_low as f64).round_ties_even() as i64),
        };
        let synthesis_hop = light.hop_low;
        let window_length = Self::low_length(light.split.d, window_length);
        self.analyze_light(&mut light, source, center, window_length);
        light.previous_center = Some(center);
        // polar: phases (f64 from VOLK's f32 atan2), magnitudes, the joint magnitude over the channels
        let (ch, bins) = (self.ch, self.bins);
        for c in 0..ch {
            let spec = &self.coeff[c * bins..(c + 1) * bins];
            vk::magnitude(&mut self.magnitude[c * bins..(c + 1) * bins], spec);
            vk::phase(&mut self.phase32, spec);
            for m in 0..bins {
                self.analysis_phase[c * bins + m] = self.phase32[m] as f64;
            }
        }
        for m in 0..bins {
            let mut p = 0f64;
            for c in 0..ch {
                let v = self.magnitude[c * bins + m] as f64;
                p += v * v;
            }
            self.joint[m] = p.sqrt() as f32;
        }
        // the synthesis spectrum, into coeff
        if self.has_previous {
            rtpghi_frame(
                &mut self.synthesis_phase,
                &self.analysis_phase,
                &self.joint,
                &self.previous_analysis_phase,
                &self.previous_synthesis_phase,
                &self.previous_joint,
                ch,
                bins,
                analysis_hop,
                synthesis_hop,
                self.n_fft,
                self.cfg.rtpghi_tolerance,
                lock_strength,
                &mut self.phase_work,
            );
            if self.cfg.stereo_coherence && ch > 1 {
                coherent_synthesis(
                    &mut self.synthesis_phase,
                    &self.analysis_phase,
                    &self.magnitude,
                    &mut self.coeff,
                    ch,
                    bins,
                    &mut self.coherent_work,
                );
            } else {
                for i in 0..ch * bins {
                    let (s, c) = self.synthesis_phase[i].sin_cos();
                    let mag = self.magnitude[i] as f64;
                    self.coeff[i] = Complex32::new((mag * c) as f32, (mag * s) as f32);
                }
            }
        } else {
            // the first frame keeps its analysis phases: the coefficients as they are
            self.synthesis_phase.copy_from_slice(&self.analysis_phase);
        }
        if (factor - 1.0).abs() <= Self::UNITY_EPS && self.handover_at.is_none() {
            self.unity_frames += 1;
            let mut flux = 0.0;
            if self.has_previous {
                let (mut up, mut total) = (0f64, 0f64);
                for m in 0..bins {
                    up += (self.joint[m] - self.previous_joint[m]).max(0.0) as f64;
                    total += self.joint[m] as f64;
                }
                flux = up / (total + EPS);
            }
            if flux > Self::ONSET_FLUX || self.unity_frames >= self.handover_timeout {
                self.handover_at = Some(output_center.max(self.output_read));
            }
        }
        self.overlap_add_light(&mut light, output_center, window_length);
        self.wsola_frame(&mut light, source, input_center, output_center);
        self.light = Some(light);
        if self.frame_map.len() == 512 {
            self.frame_map.pop_front();
        }
        self.frame_map.push_back((input_center, output_center));
        self.queue_events(source, input_center);
        std::mem::swap(&mut self.previous_analysis_phase, &mut self.analysis_phase);
        std::mem::swap(&mut self.previous_synthesis_phase, &mut self.synthesis_phase);
        self.previous_joint.copy_from_slice(&self.joint);
        self.has_previous = true;
        self.next_synthesis_center += self.hop;
        let nominal = self.next_analysis_center + factor * self.hop as f64;
        let mapped = self.anchor_in + factor * (self.next_synthesis_center - self.anchor_out) as f64;
        let limit = Self::CATCHUP_RATIO * factor * self.hop as f64;
        self.next_analysis_center = nominal + (mapped - nominal).clamp(-limit, limit);
        self.finalized_until = self.finalized_until.max(self.next_synthesis_center - (self.max_window / 2) as i64);
    }

    /// SELEBI window length at ``input_center`` and the identity-lock strength.
    fn frame_window(&mut self, source: &[f32], input_center: i64) -> (usize, f64) {
        let dh = self.detector.hop;
        let center_frame = (input_center as f64 / dh as f64).round_ties_even() as i64;
        self.detector.ensure(source, self.ch, center_frame + self.event_reach + 1);
        let mut window = self.max_window;
        let factor = self.factor.unwrap_or(1.0);
        let alpha = 1.0 / factor;
        let mut strength = self.cfg.phase_lock_strength;
        if alpha <= 1.0 + Self::UNITY_EPS {
            return (window, strength);
        }
        let cfg = &self.cfg;
        let max_reduction = 1.0 - cfg.minimum_window_ratio;
        let kick_profile = 1.0 - 1.0 / cfg.kick_profile_stretch;
        let ramp_offset = self.max_window as f64 - 2.0 * dh as f64 * self.event_reach as f64;
        let mut shortest = window as f64;
        for frame in center_frame - self.event_reach - 1..center_frame + self.event_reach + 2 {
            let Some(event) = self.detector.events.get(&frame) else { continue };
            let compression = max_reduction.min(
                (cfg.squeeze_strength * event.strength.powf(cfg.squeeze_gamma) * (1.0 - 1.0 / alpha))
                    .max(event.kick * event.kick * kick_profile),
            );
            let target = (self.max_window as f64 * (1.0 - compression)).floor();
            let target = target.clamp(self.min_window as f64, self.max_window as f64);
            let distance = (input_center - frame * dh).abs() as f64;
            shortest = shortest.min(target.max(2.0 * distance + ramp_offset));
        }
        if shortest < window as f64 {
            let q = cfg.window_quantum as f64;
            let mut w = ((shortest / q).round_ties_even() * q) as usize;
            w = w.clamp(self.min_window, self.max_window);
            w += w & 1;
            window = w;
        }
        let reduction_span = self.max_window as f64 * (1.0 - 1.0 / alpha);
        if reduction_span > 0.5 {
            let percussiveness = ((self.max_window - window) as f64 / reduction_span).clamp(0.0, 1.0);
            strength *= 1.0 - percussiveness;
        }
        (window, strength)
    }

    fn light_window(&self, light: &mut Light, wl: usize) -> Rc<LightWindow> {
        if let Some(w) = light.windows.get(&wl) {
            return w.clone();
        }
        let d = light.split.d;
        let half = (wl / 2) as f64;
        let (mut center_ramp, mut inverse_ramp) = (Vec::with_capacity(self.bins), Vec::with_capacity(self.bins));
        for k in 0..self.bins {
            let a = TWO_PI * k as f64 / self.n_fft as f64 * half;
            center_ramp.push(Complex32::new(a.cos() as f32, a.sin() as f32));
            inverse_ramp.push(Complex32::new(a.cos() as f32, -a.sin() as f32));
        }
        let synthesis = periodic_hann(d * wl);
        let synthesis_sq = synthesis.iter().map(|w| w * w).collect();
        let w = Rc::new(LightWindow { analysis: periodic_hann(wl), center_ramp, inverse_ramp, synthesis, synthesis_sq });
        light.windows.insert(wl, w.clone());
        w
    }

    /// The low band's windowed frame (``wl`` low-rate samples about ``center``) per channel, centre-referenced,
    /// into coeff ([ch, bins]).
    fn analyze_light(&mut self, light: &mut Light, source: &[f32], center: i64, wl: usize) {
        let wd = self.light_window(light, wl);
        let left = center - (wl / 2) as i64;
        light.split.ensure_low(source, left, left + wl as i64);
        for c in 0..self.ch {
            vk::mul(&mut light.fft_in[..wl], light.split.low(c, left, left + wl as i64), &wd.analysis);
            light.fft_in[wl..].iter_mut().for_each(|v| *v = 0.0);
            light.fft.process_with_scratch(&mut light.fft_in, &mut light.fft_out, &mut light.fft_scratch).ok();
            vk::cmul_in(&mut light.fft_out, &wd.center_ramp);
            self.coeff[c * self.bins..(c + 1) * self.bins].copy_from_slice(&light.fft_out);
        }
    }

    /// The low band's synthesis frame at the full rate: its bins zero-padded to the full rate's inverse FFT,
    /// the full-rate window, overlap-added as ``overlap_add`` does.
    fn overlap_add_light(&mut self, light: &mut Light, center: i64, wl: usize) {
        let wd = self.light_window(light, wl);
        let ch = self.ch;
        let length = light.split.d * wl;
        let mut left = center - (length / 2) as i64;
        let right = left + length as i64;
        let mut src_start = 0usize;
        if left < self.ola_base {
            src_start = (self.ola_base - left) as usize;
            left = self.ola_base;
        }
        if right <= left || src_start >= length {
            return;
        }
        self.ensure_capacity(right);
        let start = (left - self.ola_base) as usize;
        let stop = (right - self.ola_base) as usize;
        let span = stop - start;
        // the low band's spectrum was taken at 1 / d of the rate: 1 / n_fft (not the full rate's) keeps its level
        let scale = 1.0 / self.n_fft as f32;
        let bins = self.bins;
        for c in 0..ch {
            let spec = &mut light.synth_spec;
            spec[..bins].copy_from_slice(&self.coeff[c * bins..(c + 1) * bins]);
            vk::cmul_in(&mut spec[..bins], &wd.inverse_ramp);
            spec[0].im = 0.0;
            spec[bins - 1] = Complex32::new(0.0, 0.0); // the low rate's Nyquist bin: the band stops below it
            spec[bins..].iter_mut().for_each(|v| *v = Complex32::new(0.0, 0.0));
            light.ifft.process_with_scratch(spec, &mut light.synth_out, &mut light.ifft_scratch).ok();
            let frame = &mut self.frame_buf[..span];
            vk::mul(frame, &light.synth_out[src_start..src_start + span], &wd.synthesis[src_start..src_start + span]);
            for k in 0..span {
                self.ola[(start + k) * ch + c] += frame[k] * scale;
            }
        }
        vk::add_in(&mut self.weight[start..stop], &wd.synthesis_sq[src_start..src_start + span]);
    }

    /// The high band's segment for one vocoder frame: two hops of the source's high band from within
    /// ``tolerance`` of the frame's input start, where it best continues the previous segment, overlap-added at
    /// the frame's output start.
    fn wsola_frame(&mut self, light: &mut Light, source: &[f32], input_center: i64, output_center: i64) {
        let ch = self.ch;
        let hop = self.hop;
        let segment = 2 * hop;
        let tol = light.tolerance;
        let nominal = input_center - hop;
        let mut start = nominal;
        // the splice's correlation: the previous segment's second half and this one's first
        let mut correlation = 1f32;
        if let Some(previous) = light.previous_start {
            // compared over the overlap: this segment's first half against the previous one's continuation
            let natural = previous + hop;
            let search = (natural - nominal).abs() <= 4 * hop;
            let (a, b) = if search { (nominal - tol, nominal + tol + hop) } else { (nominal, nominal + hop) };
            light.split.ensure_high(source, a.min(natural), b.max(natural + hop));
            // each candidate's energy from running sums
            let span = (b - a) as usize;
            let energy = &mut light.energy;
            energy.clear();
            energy.resize(span + 1, 0.0);
            for c in 0..ch {
                let x = light.split.high(c, a, b);
                let mut acc = 0f64;
                for k in 0..span {
                    acc += (x[k] * x[k]) as f64;
                    energy[k + 1] += acc;
                }
            }
            let hop_u = hop as usize;
            let energy_at = |at: i64| -> f32 {
                let k = (at - a) as usize;
                (light.energy[k + hop_u] - light.energy[k]).max(0.0) as f32
            };
            let cross = |at: i64| -> f32 {
                let mut num = 0f32;
                for c in 0..ch {
                    num += dot(light.split.high(c, at, at + hop), light.split.high(c, natural, natural + hop));
                }
                num
            };
            let mut num = cross(nominal);
            let mut best = num / (energy_at(nominal) + 1e-12).sqrt();
            if search {
                // every candidate's correlation at once: sum over channels of X conj(Y), one inverse FFT
                let n = light.xc_in.len();
                light.xc_sum.iter_mut().for_each(|v| *v = Complex32::new(0.0, 0.0));
                for c in 0..ch {
                    light.xc_in[..span].copy_from_slice(light.split.high(c, a, b));
                    light.xc_in[span..].iter_mut().for_each(|v| *v = 0.0);
                    light.xc_fft.process_with_scratch(&mut light.xc_in, &mut light.xc_spec, &mut light.xc_scratch).ok();
                    light.xc_in[..hop_u].copy_from_slice(light.split.high(c, natural, natural + hop));
                    light.xc_in[hop_u..].iter_mut().for_each(|v| *v = 0.0);
                    light.xc_fft.process_with_scratch(&mut light.xc_in, &mut light.xc_ref, &mut light.xc_scratch).ok();
                    for (sum, (x, y)) in light.xc_sum.iter_mut().zip(light.xc_spec.iter().zip(light.xc_ref.iter())) {
                        *sum += x * y.conj();
                    }
                }
                light.xc_sum[0].im = 0.0;
                let last = light.xc_sum.len() - 1;
                light.xc_sum[last].im = 0.0;
                light.xc_ifft.process_with_scratch(&mut light.xc_sum, &mut light.xc_out, &mut light.xc_scratch).ok();
                let scale = 1.0 / n as f32;
                for k in 0..=(2 * tol) as usize {
                    let at = a + k as i64;
                    if at == nominal {
                        continue;
                    }
                    let c = light.xc_out[k] * scale;
                    let score = c / (energy_at(at) + 1e-12).sqrt();
                    if score > best {
                        best = score;
                        num = c;
                        start = at;
                    }
                }
            }
            let mut e0 = 0f32;
            for c in 0..ch {
                let x0 = light.split.high(c, natural, natural + hop);
                e0 += dot(x0, x0);
            }
            let e1 = energy_at(start);
            correlation = if e0 > 0.0 && e1 > 0.0 { (num / (e0 * e1).sqrt()).clamp(0.0, 1.0) } else { 1.0 };
        }
        light.previous_start = Some(start);
        let mut left = output_center - hop;
        let right = left + segment;
        let mut offset = 0i64;
        if left < self.ola_base {
            offset = self.ola_base - left;
            left = self.ola_base;
        }
        if right <= left {
            return;
        }
        self.ensure_capacity(right);
        light.split.ensure_high(source, start + offset, start + segment);
        let base = (left - self.ola_base) as usize;
        let n = (right - left) as usize;
        let window = &light.segment_window[offset as usize..];
        for c in 0..ch {
            let x = light.split.high(c, start + offset, start + segment);
            for k in 0..n {
                self.ola_high[(base + k) * ch + c] += window[k] * x[k];
            }
        }
        // the crossfade's level kept whatever the splice's correlation r: the two halves' sum divided by
        // sqrt(a^2 + b^2 + 2 r a b) (a + b = 1): Hann's own at r = 1, an equal-power crossfade at r = 0
        if correlation < 1.0 {
            let r = correlation as f64;
            for k in 0..(hop - offset).max(0) as usize {
                let a = window[k] as f64;
                let b = 1.0 - a;
                let g = (1.0 / (a * a + b * b + 2.0 * r * a * b).sqrt()) as f32;
                for c in 0..ch {
                    self.ola_high[(base + k) * ch + c] *= g;
                }
            }
        }
    }

    /// The attack, one band of it (``low``: the vocoder's, else WSOLA's): the output crossfaded to
    /// the source's own band, its onset at the event's destination on the time map, so the attack is the
    /// source's and never overlap-added. The crossfades adapt to what they join: the rise (1 ms) ends before
    /// the attack by as much as WSOLA may have placed a copy of it early (the stretch's drift over a hop plus
    /// its tolerance); the plateau ends where the attack has decayed by 9 dB and the source and the stretched
    /// output agree best (their correlation r over 4 ms, 6 - 30 ms after the onset); the fall lasts 2 ms when
    /// they agree (r = 1) to 8 ms when they do not (r = 0), and both crossfades keep the level whatever r
    /// (the sum divided by sqrt(s^2 + (1 - s)^2 + 2 r s (1 - s))).
    fn restore_light(&mut self, light: &mut Light, source: &[f32], at: i64, destination: i64, low: bool) {
        let ch = self.ch;
        let ms = |v: f64| 1.max((v * light.rate / 1000.0).round_ties_even() as i64);
        let factor = self.factor.unwrap_or(1.0);
        let drift = ((1.0 - factor).abs() * self.hop as f64).ceil() as i64 + light.tolerance;
        let rise = ms(1.0);
        let first = destination - drift - rise - ms(1.0);
        let (post_min, post_max) = (ms(6.0), ms(30.0));
        let fall_max = ms(8.0);
        let end = (destination + post_max + fall_max).min(self.finalized_until);
        let lo = first.max(self.output_read).max(self.ola_base);
        if end <= lo.max(destination + post_min) {
            return;
        }
        // the vocoder's band is crossfaded above a cutoff only: the bass below keeps its own phase (the
        // source's and the stretched one's phases are unrelated there, and a crossfade of the two cancels)
        let head = if low { ms(10.0) } else { 0 };
        let first = first - head;
        let len = (end - first) as usize;
        // the source's band (x) and the output's (y) over [first, end), per channel
        let x_first = at - (destination - first);
        light.split.ensure_high(source, x_first, x_first + len as i64);
        let frames = (source.len() / ch) as i64;
        let mut x = vec![vec![0f32; len]; ch];
        let mut y = vec![vec![0f32; len]; ch];
        for c in 0..ch {
            let high = light.split.high(c, x_first, x_first + len as i64);
            for k in 0..len {
                x[c][k] = if low {
                    let i = x_first + k as i64;
                    let v = if i >= 0 && i < frames { source[i as usize * ch + c] } else { 0.0 };
                    v - high[k]
                } else {
                    high[k]
                };
            }
        }
        for k in 0..len {
            let t = first + k as i64;
            if t < self.ola_base {
                continue;
            }
            let i = (t - self.ola_base) as usize;
            let w = self.weight[i];
            for c in 0..ch {
                y[c][k] = if low {
                    (if w > 1e-10 { self.ola[i * ch + c] / w } else { 0.0 }) + self.correction[i * ch + c]
                } else {
                    self.ola_high[i * ch + c]
                };
            }
        }
        if low {
            let sos = butter_highpass4(Self::ATTACK_LOW_CUTOFF_HZ, light.rate);
            for c in 0..ch {
                let xf: Vec<f64> = x[c].iter().map(|&v| v as f64).collect();
                let yf: Vec<f64> = y[c].iter().map(|&v| v as f64).collect();
                x[c] = sosfiltfilt(&sos, &xf).iter().map(|&v| v as f32).collect();
                y[c] = sosfiltfilt(&sos, &yf).iter().map(|&v| v as f32).collect();
            }
        }
        let correlation = |a: usize, b: usize| -> f64 {
            let (mut xy, mut xx, mut yy) = (0f64, 0f64, 0f64);
            for c in 0..ch {
                xy += dot(&x[c][a..b], &y[c][a..b]) as f64;
                xx += dot(&x[c][a..b], &x[c][a..b]) as f64;
                yy += dot(&y[c][a..b], &y[c][a..b]) as f64;
            }
            if xx > 0.0 && yy > 0.0 { (xy / (xx * yy).sqrt()).clamp(0.0, 1.0) } else { 1.0 }
        };
        // the attack's envelope (1 ms power) and its peak over the first 10 ms
        let w = ms(1.0) as usize;
        let power: Vec<f64> = (0..len).map(|k| (0..ch).map(|c| (x[c][k] * x[c][k]) as f64).sum::<f64>()).collect();
        let mut envelope = vec![0f64; len];
        let mut acc = 0.0;
        for k in 0..len {
            acc += power[k];
            if k >= w {
                acc -= power[k - w];
            }
            envelope[k] = acc.max(0.0);
        }
        let d = (destination - first) as usize;
        let peak = envelope[d..(d + ms(10.0) as usize).min(len)].iter().cloned().fold(0.0, f64::max);
        let decayed = peak * 10f64.powf(-0.9);
        // the plateau's end: the decayed position of best agreement (else the quietest)
        let window = ms(4.0) as usize;
        let (mut best, mut fall_at, mut quietest) = (-1.0f64, usize::MAX, (f64::MAX, d + post_min as usize));
        let mut k = d + post_min as usize;
        while k <= d + post_max as usize && k + fall_max as usize <= len {
            let e = envelope[k];
            if e < quietest.0 {
                quietest = (e, k);
            }
            if e <= decayed {
                let r = correlation(k, (k + window).min(len));
                if r > best {
                    best = r;
                    fall_at = k;
                }
            }
            k += ms(0.25) as usize;
        }
        if fall_at == usize::MAX {
            fall_at = quietest.1.min(len.saturating_sub(fall_max as usize));
            best = correlation(fall_at, (fall_at + window).min(len));
        }
        let fall = ms(2.0 + 6.0 * (1.0 - best)) as usize;
        let (h, rise) = (head as usize, rise as usize);
        let rise_r = correlation(h, h + rise);
        let stop = (fall_at + fall).min(len);
        for k in ((lo - first) as usize).max(h)..stop {
            let (s, r) = if k < h + rise {
                ((0.5 * PI * (k - h) as f64 / rise as f64).sin().powi(2), rise_r)
            } else if k >= fall_at {
                ((0.5 * PI * (stop - k) as f64 / fall as f64).sin().powi(2), best)
            } else {
                (1.0, 1.0)
            };
            let norm = (s * s + (1.0 - s) * (1.0 - s) + 2.0 * r * s * (1.0 - s)).sqrt();
            let i = ((first + k as i64) - self.ola_base) as usize;
            for c in 0..ch {
                let out = ((s * x[c][k] as f64 + (1.0 - s) * y[c][k] as f64) / norm) as f32;
                if low {
                    self.correction[i * ch + c] += out - y[c][k];
                } else {
                    self.ola_high[i * ch + c] = out;
                }
            }
        }
    }

    fn ensure_capacity(&mut self, absolute_stop: i64) {
        let required = (absolute_stop - self.ola_base) as usize;
        if required <= self.weight.len() {
            return;
        }
        let growth = (required - self.weight.len()).max(self.max_window).max(4096);
        let len = self.weight.len() + growth;
        self.weight.resize(len, 0.0);
        self.ola.resize(len * self.ch, 0.0);
        self.correction.resize(len * self.ch, 0.0);
        self.ola_high.resize(len * self.ch, 0.0);
    }

    fn read_output(&mut self, count: usize) -> Vec<f32> {
        let ch = self.ch;
        let start = (self.output_read - self.ola_base) as usize;
        let mut out = vec![0f32; count * ch];
        for k in 0..count {
            let w = self.weight[start + k];
            for c in 0..ch {
                let i = (start + k) * ch + c;
                let v = if w > 1e-10 { self.ola[i] / w } else { 0.0 };
                out[k * ch + c] = v + self.ola_high[i] + self.correction[i];
            }
        }
        self.output_read += count as i64;
        out
    }

    fn compact_output(&mut self) {
        let keep = self.dry.pre.max(self.kick.pre) + self.filter_pad;
        let consumed = self.output_read - keep - self.ola_base;
        if consumed < Self::COMPACT_AFTER {
            return;
        }
        let c = consumed as usize;
        self.weight.drain(..c);
        self.ola.drain(..c * self.ch);
        self.correction.drain(..c * self.ch);
        self.ola_high.drain(..c * self.ch);
        self.ola_base += consumed;
    }

    fn read_direct(&self, source: &[f32], pos: f64, count: usize) -> Vec<f32> {
        let ch = self.ch;
        let frames = source.len() / ch;
        let start = (pos + 0.5).floor() as usize;
        let mut out = vec![0f32; count * ch];
        if start < frames {
            let n = count.min(frames - start);
            out[..n * ch].copy_from_slice(&source[start * ch..(start + n) * ch]);
        }
        out
    }

    // Attack preservation
    /// Queues the events the time map reaches, each with its destination: its onset (``onset``) aligned, so every
    /// band's attack lands where the map puts it.
    fn queue_events(&mut self, source: &[f32], input_center: i64) {
        let dh = self.detector.hop;
        // two frames on: an onset lies up to 1.5 frames after its frame's centre
        let last = ((input_center + 2 * dh) as f64 / dh as f64).floor() as i64;
        if last < self.next_event_frame {
            return;
        }
        let mut centers: Option<(Vec<f64>, Vec<f64>)> = None;
        for frame in self.next_event_frame..=last {
            if let Some(&event) = self.detector.events.get(&frame) {
                let at = self.onset(source, frame * dh);
                if at > input_center {
                    // the map does not reach it yet
                    self.next_event_frame = frame;
                    return;
                }
                let (xs, ys) = centers.get_or_insert_with(|| {
                    (self.frame_map.iter().map(|p| p.0 as f64).collect(), self.frame_map.iter().map(|p| p.1 as f64).collect())
                });
                let destination = interp(at as f64, xs, ys).round_ties_even() as i64;
                self.pending_events.push_back((destination, at, event));
            }
            self.next_event_frame = frame + 1;
        }
    }

    /// An event's onset in the source: the steepest rise of the 1 ms energy of the mono signal's first
    /// difference (its highs) within 1.5 detector frames of the frame's centre.
    fn onset(&self, source: &[f32], center: i64) -> i64 {
        let ch = self.ch;
        let frames = (source.len() / ch) as i64;
        let dh = self.detector.hop;
        let w = self.envelope_window as i64;
        let (a, b) = (center - 3 * dh / 2, center + 3 * dh / 2);
        let first = a - 2 * w - 1;
        let mono = |i: i64| -> f32 {
            if i < 0 || i >= frames {
                return 0.0;
            }
            source[i as usize * ch..(i as usize + 1) * ch].iter().sum::<f32>()
        };
        let energy: Vec<f64> = (first..b).map(|i| ((mono(i) - mono(i - 1)) as f64).powi(2)).collect();
        // running 1 ms sums: sum[k] over energy[k - w + 1 ..= k]
        let mut sum = vec![0f64; energy.len()];
        let mut acc = 0.0;
        for k in 0..energy.len() {
            acc += energy[k];
            if k as i64 >= w {
                acc -= energy[k - w as usize];
            }
            sum[k] = acc;
        }
        let (mut best, mut at) = (f64::MIN, center);
        for n in a..b {
            let k = (n - first) as usize;
            let rise = sum[k] - sum[k - w as usize];
            if rise > best {
                best = rise;
                at = n - w / 2;
            }
        }
        at
    }

    fn apply_due_events(&mut self, source: &[f32]) {
        while let Some(&(destination, at, event)) = self.pending_events.front() {
            if destination + self.attack_ahead - self.dry.pre.max(self.kick.pre) > self.finalized_until {
                return;
            }
            self.pending_events.pop_front();
            let cfg = self.cfg;
            let center = at;
            let span = cfg.dry_transient_full_strength - cfg.dry_transient_threshold;
            let dry = ((event.strength - cfg.dry_transient_threshold) / span).clamp(0.0, 1.0);
            let kick = ((event.kick - Self::KICK_CONFIDENCE_FROM) / Self::KICK_CONFIDENCE_SPAN).clamp(0.0, 1.0);
            let mut light = self.light.take().expect("the stretcher's bands");
            // the high band's attack at every event; the vocoder's where it is a dry or a kick attack
            self.restore_light(&mut light, source, center, destination, false);
            if dry > 0.0 || kick > 0.0 {
                self.restore_light(&mut light, source, center, destination, true);
            }
            self.light = Some(light);
        }
    }
}
