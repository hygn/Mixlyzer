//! Streaming, percussion-aware, pitch-preserving time scaling: audioengine/python/timestretch.py's
//! StreamingTimeStretcher (its docs hold the design), with the FFTs on realfft / rustfft and the vector work
//! on VOLK. ``factor`` is input frames per output frame.

use std::collections::{HashMap, VecDeque};
use std::f64::consts::PI;
use std::rc::Rc;
use std::sync::Arc;

use num_complex::Complex32;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};

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
    pub dry_transient_amount: f64,
    pub dry_transient_threshold: f64,
    pub dry_transient_full_strength: f64,
    pub dry_transient_pre_ms: f64,
    pub dry_transient_post_ms: f64,
    pub dry_transient_highpass_hz: f64,
    pub kick_transient_bypass: f64,
    pub kick_transient_pre_ms: f64,
    pub kick_transient_post_ms: f64,
    pub kick_transient_highpass_hz: f64,
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
            dry_transient_amount: 0.70,
            dry_transient_threshold: 0.40,
            dry_transient_full_strength: 0.68,
            dry_transient_pre_ms: 4.0,
            dry_transient_post_ms: 24.0,
            dry_transient_highpass_hz: 500.0,
            kick_transient_bypass: 1.0,
            kick_transient_pre_ms: 10.0,
            kick_transient_post_ms: 24.0,
            kick_transient_highpass_hz: 220.0,
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
    frame: i64,
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
        self.events.insert(j, Event { frame: j, strength: value, kick: (low_score * percussion_score).sqrt() });
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

const ITEM_BITS: u32 = 15;

#[inline(always)]
fn entry(bits: u32, item: i64) -> i64 {
    ((0xFFFF_FFFFi64 - bits as i64) << ITEM_BITS) | item
}

fn sift_down(heap: &mut [i64], start: usize, size: usize) {
    let e = heap[start];
    let mut parent = start;
    loop {
        let mut child = 2 * parent + 1;
        if child >= size {
            break;
        }
        let right = child + 1;
        if right < size && heap[right] < heap[child] {
            child = right;
        }
        if heap[child] < e {
            heap[parent] = heap[child];
            parent = child;
        } else {
            break;
        }
    }
    heap[parent] = e;
}

fn push(heap: &mut [i64], size: usize, e: i64) -> usize {
    let mut child = size;
    while child > 0 {
        let parent = (child - 1) / 2;
        if e < heap[parent] {
            heap[child] = heap[parent];
            child = parent;
        } else {
            break;
        }
    }
    heap[child] = e;
    size + 1
}

/// RTPGHI: the synthesis phases of one frame (bins integrated in order of magnitude, in time from the
/// previous frame or in frequency from a neighbour), then identity phase locking around the joint peaks.
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
    out.copy_from_slice(analysis);
    let ha = analysis_hop.max(1) as f64;
    let hs = synthesis_hop.max(1) as f64;
    let local = hs / ha;
    let mut max_mag = EPS;
    for m in 0..bins {
        max_mag = max_mag.max(previous_joint[m] as f64).max(joint[m] as f64);
    }
    let floor = tolerance * max_mag;
    let item_mask = (1i64 << ITEM_BITS) - 1;
    let remaining = &mut work.remaining;
    remaining.clear();
    remaining.resize(bins, false);
    let mut remaining_count = 0usize;
    let seeds = &mut work.seeds;
    seeds.clear();
    for m in 0..bins {
        if joint[m] as f64 > floor {
            remaining[m] = true;
            remaining_count += 1;
            seeds.push(entry(previous_joint[m].to_bits(), 2 * m as i64));
        }
    }
    seeds.sort_unstable();
    let seed_count = seeds.len();
    let mut next_seed = 0;
    let heap = &mut work.heap;
    heap.clear();
    heap.resize(2 * bins + 4, 0);
    let mut size = 0usize;
    let time_step = |out: &mut [f64], m: usize| {
        let omega = TWO_PI * m as f64 / n_fft as f64;
        for c in 0..channels {
            let i = c * bins + m;
            let residual = wrap(analysis[i] - previous_analysis[i] - omega * ha);
            out[i] = previous_synthesis[i] + omega * hs + local * residual;
        }
    };
    while remaining_count > 0 {
        if size == 0 && next_seed >= seed_count {
            let mut m = 0;
            while m < bins && !remaining[m] {
                m += 1;
            }
            time_step(out, m);
            remaining[m] = false;
            remaining_count -= 1;
            size = push(heap, size, entry(joint[m].to_bits(), 2 * m as i64 + 1));
        }
        let item;
        if size > 0 && (next_seed >= seed_count || heap[0] < seeds[next_seed]) {
            item = heap[0] & item_mask;
            size -= 1;
            if size > 0 {
                heap[0] = heap[size];
                sift_down(heap, 0, size);
            }
        } else {
            item = seeds[next_seed] & item_mask;
            next_seed += 1;
        }
        let m = (item >> 1) as usize;
        if item & 1 == 0 {
            if remaining[m] {
                time_step(out, m);
                remaining[m] = false;
                remaining_count -= 1;
                size = push(heap, size, entry(joint[m].to_bits(), 2 * m as i64 + 1));
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
            size = push(heap, size, entry(joint[up].to_bits(), 2 * up as i64 + 1));
        }
        if m >= 1 && remaining[m - 1] {
            let down = m - 1;
            for c in 0..channels {
                let b = c * bins;
                out[b + down] = out[b + m] - local * wrap(analysis[b + m] - analysis[b + down]);
            }
            remaining[down] = false;
            remaining_count -= 1;
            size = push(heap, size, entry(joint[down].to_bits(), 2 * down as i64 + 1));
        }
    }

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
    remaining: Vec<bool>,
    seeds: Vec<i64>,
    heap: Vec<i64>,
    peaks: Vec<usize>,
}

struct WindowData {
    window: Vec<f32>,
    window_sq: Vec<f32>,
    center_ramp: Vec<Complex32>,
    inverse_ramp: Vec<Complex32>,
}

#[derive(Clone, Copy)]
struct Attack {
    pre: i64,
    post: i64,
    fade: usize,
    sos: Option<Sos>,
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
    windows: HashMap<usize, Rc<WindowData>>,
    fft: Arc<dyn RealToComplex<f32>>,
    ifft: Arc<dyn ComplexToReal<f32>>,

    source_token: Option<(usize, usize)>,
    factor: Option<f64>,
    expected_pos: Option<f64>,
    next_analysis_center: f64,
    next_synthesis_center: i64,
    previous_analysis_center: Option<i64>,
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
    pending_events: VecDeque<(i64, Event)>,

    streaming: bool,
    stream_gain: f64,
    gain_target: f64,
    unity_frames: i64,
    handover_at: Option<i64>,
    anchor_in: f64,
    anchor_out: i64,
    warming: bool,

    // per-frame work
    fft_in: Vec<f32>,
    fft_out: Vec<Complex32>,
    fft_scratch: Vec<Complex32>,
    ifft_scratch: Vec<Complex32>,
    frame_buf: Vec<f32>,
    coeff: Vec<Complex32>,       // [ch, bins]
    analysis_phase: Vec<f64>,    // [ch, bins]
    synthesis_phase: Vec<f64>,   // [ch, bins]
    magnitude: Vec<f32>,         // [ch, bins]
    joint: Vec<f32>,             // [bins]
    phase32: Vec<f32>,
    phase_work: PhaseWork,
    coherent_work: CoherentWork,
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

    pub fn new(channels: usize, sample_rate: u32) -> Stretcher {
        let cfg = Config::default();
        let rate = sample_rate as f64;
        let requested = 128.max((rate * cfg.window_ms / 1000.0).round_ties_even() as usize);
        let max_window = requested + (requested & 1);
        let hop = 1.max((rate * cfg.synthesis_hop_ms / 1000.0).round_ties_even() as i64);
        let n_fft = ((max_window as f64 * cfg.frequency_oversampling).ceil() as usize).next_power_of_two();
        let bins = n_fft / 2 + 1;
        let min_window = 16.max((cfg.minimum_window_ratio * max_window as f64).ceil() as usize);
        let mut planner = RealFftPlanner::<f32>::new();
        let detector = Detector::new(rate, max_window, hop, n_fft, &cfg, &mut planner);
        let event_reach = (max_window as f64 / (2.0 * hop as f64)).ceil() as i64;
        let handover_timeout = 1.max((rate * Self::HANDOVER_TIMEOUT_SEC / hop as f64).round_ties_even() as i64);
        let ms = |v: f64| 1.max((v * rate / 1000.0).round_ties_even() as i64);
        let hp = |hz: f64| if hz <= 0.0 { None } else { Some(butter_highpass4(hz, rate)) };
        let dry = Attack {
            pre: ms(cfg.dry_transient_pre_ms),
            post: ms(cfg.dry_transient_post_ms),
            fade: ms(3.0) as usize,
            sos: hp(cfg.dry_transient_highpass_hz),
        };
        let kick = Attack {
            pre: ms(cfg.kick_transient_pre_ms),
            post: ms(cfg.kick_transient_post_ms),
            fade: ms(4.0) as usize,
            sos: hp(cfg.kick_transient_highpass_hz),
        };
        let filter_pad = ms(cfg.attack_filter_pad_ms);
        let envelope_window = 3.max(ms(1.0) as usize);
        let attack_ahead = dry.pre.max(kick.pre) + dry.post.max(kick.post) + filter_pad + 1;
        let fft = planner.plan_fft_forward(n_fft);
        let ifft = planner.plan_fft_inverse(n_fft);
        let fft_scratch = fft.make_scratch_vec();
        let ifft_scratch = ifft.make_scratch_vec();
        let zc = Complex32::new(0.0, 0.0);
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
            windows: HashMap::new(),
            fft,
            ifft,
            source_token: None,
            factor: None,
            expected_pos: None,
            next_analysis_center: 0.0,
            next_synthesis_center: 0,
            previous_analysis_center: None,
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
            fft_in: vec![0.0; n_fft],
            fft_out: vec![zc; bins],
            fft_scratch,
            ifft_scratch,
            frame_buf: vec![0.0; n_fft],
            coeff: vec![zc; channels * bins],
            analysis_phase: vec![0.0; channels * bins],
            synthesis_phase: vec![0.0; channels * bins],
            magnitude: vec![0.0; channels * bins],
            joint: vec![0.0; bins],
            phase32: vec![0.0; bins],
            phase_work: PhaseWork::default(),
            coherent_work: CoherentWork::default(),
        };
        s.reset();
        s
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
        self.previous_analysis_center = None;
        self.has_previous = false;
        self.output_read = 0;
        self.finalized_until = 0;
        self.ola_base = 0;
        self.ola.clear();
        self.weight.clear();
        self.correction.clear();
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
        self.previous_analysis_center = None;
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
    }

    fn window_data(&mut self, length: usize) -> Rc<WindowData> {
        if let Some(w) = self.windows.get(&length) {
            return w.clone();
        }
        let window = periodic_hann(length);
        let window_sq = window.iter().map(|w| w * w).collect();
        let half = (length / 2) as f64;
        let (mut center_ramp, mut inverse_ramp) = (Vec::with_capacity(self.bins), Vec::with_capacity(self.bins));
        for k in 0..self.bins {
            let a = TWO_PI * k as f64 / self.n_fft as f64 * half;
            center_ramp.push(Complex32::new(a.cos() as f32, a.sin() as f32));
            inverse_ramp.push(Complex32::new(a.cos() as f32, -a.sin() as f32));
        }
        let w = Rc::new(WindowData { window, window_sq, center_ramp, inverse_ramp });
        self.windows.insert(length, w.clone());
        w
    }

    fn process_frame(&mut self, source: &[f32]) {
        let factor = self.factor.unwrap_or(1.0);
        let input_center = (self.next_analysis_center + 0.5).floor() as i64;
        let output_center = self.next_synthesis_center;
        let analysis_hop = match self.previous_analysis_center {
            None => 1.max((factor * self.hop as f64).round_ties_even() as i64),
            Some(p) => 1.max(input_center - p),
        };
        let (window_length, lock_strength) = self.frame_window(source, input_center);
        self.analyze(source, input_center, window_length);
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
                self.hop,
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
        self.overlap_add(output_center, window_length);
        if self.frame_map.len() == 512 {
            self.frame_map.pop_front();
        }
        self.frame_map.push_back((input_center, output_center));
        self.queue_events(input_center);
        self.previous_analysis_center = Some(input_center);
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

    /// The windowed frame's spectrum per channel, centre-referenced, into coeff ([ch, bins]).
    fn analyze(&mut self, source: &[f32], center: i64, length: usize) {
        let wd = self.window_data(length);
        let ch = self.ch;
        let frames = (source.len() / ch) as i64;
        let left = center - (length / 2) as i64;
        let (lo, hi) = (left.max(0), frames.min(left + length as i64));
        for c in 0..ch {
            self.frame_buf[..length].iter_mut().for_each(|v| *v = 0.0);
            for i in lo..hi {
                self.frame_buf[(i - left) as usize] = source[i as usize * ch + c];
            }
            vk::mul(&mut self.fft_in[..length], &self.frame_buf[..length], &wd.window);
            self.fft_in[length..].iter_mut().for_each(|v| *v = 0.0);
            self.fft.process_with_scratch(&mut self.fft_in, &mut self.fft_out, &mut self.fft_scratch).ok();
            vk::cmul_in(&mut self.fft_out, &wd.center_ramp);
            self.coeff[c * self.bins..(c + 1) * self.bins].copy_from_slice(&self.fft_out);
        }
    }

    fn overlap_add(&mut self, center: i64, length: usize) {
        let wd = self.window_data(length);
        let ch = self.ch;
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
        let scale = 1.0 / self.n_fft as f32;
        for c in 0..ch {
            self.fft_out.copy_from_slice(&self.coeff[c * self.bins..(c + 1) * self.bins]);
            vk::cmul_in(&mut self.fft_out, &wd.inverse_ramp);
            // irfft takes the DC and Nyquist bins as real
            self.fft_out[0].im = 0.0;
            let last = self.bins - 1;
            self.fft_out[last].im = 0.0;
            self.ifft.process_with_scratch(&mut self.fft_out, &mut self.fft_in, &mut self.ifft_scratch).ok();
            let frame = &mut self.frame_buf[..span];
            vk::mul(frame, &self.fft_in[src_start..src_start + span], &wd.window[src_start..src_start + span]);
            for k in 0..span {
                self.ola[(start + k) * ch + c] += frame[k] * scale;
            }
        }
        vk::add_in(&mut self.weight[start..stop], &wd.window_sq[src_start..src_start + span]);
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
                out[k * ch + c] = v + self.correction[i];
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
    fn queue_events(&mut self, input_center: i64) {
        let dh = self.detector.hop;
        let last = (input_center as f64 / dh as f64).floor() as i64;
        if last < self.next_event_frame {
            return;
        }
        let mut centers: Option<(Vec<f64>, Vec<f64>)> = None;
        for frame in self.next_event_frame..=last {
            let Some(&event) = self.detector.events.get(&frame) else { continue };
            let (xs, ys) = centers.get_or_insert_with(|| {
                (self.frame_map.iter().map(|p| p.0 as f64).collect(), self.frame_map.iter().map(|p| p.1 as f64).collect())
            });
            let destination = interp((frame * dh) as f64, xs, ys).round_ties_even() as i64;
            self.pending_events.push_back((destination, event));
        }
        self.next_event_frame = last + 1;
    }

    fn apply_due_events(&mut self, source: &[f32]) {
        while let Some(&(destination, event)) = self.pending_events.front() {
            if destination + self.attack_ahead - self.dry.pre.max(self.kick.pre) > self.finalized_until {
                return;
            }
            self.pending_events.pop_front();
            let cfg = self.cfg;
            let center = event.frame * self.detector.hop;
            let span = cfg.dry_transient_full_strength - cfg.dry_transient_threshold;
            let confidence = ((event.strength - cfg.dry_transient_threshold) / span).clamp(0.0, 1.0);
            if cfg.dry_transient_amount > 0.0 && confidence > 0.0 {
                self.restore_attack(source, center, destination, self.dry, cfg.dry_transient_amount * confidence, true);
            }
            let confidence = ((event.kick - Self::KICK_CONFIDENCE_FROM) / Self::KICK_CONFIDENCE_SPAN).clamp(0.0, 1.0);
            if cfg.kick_transient_bypass > 0.0 && confidence > 0.0 {
                self.restore_attack(source, center, destination, self.kick, cfg.kick_transient_bypass * confidence, false);
            }
        }
    }

    /// Pulls the output's high band around one attack toward the unstretched source (``match_envelope``:
    /// the output's own band scaled toward the source's 1 ms envelope; else the source's band crossfaded in).
    fn restore_attack(&mut self, source: &[f32], input_center: i64, output_center: i64, spec: Attack, amount: f64, match_envelope: bool) {
        let ch = self.ch;
        let pad = self.filter_pad;
        let lo = (output_center - spec.pre).max(self.output_read).max(self.ola_base);
        let hi = (output_center + spec.post + 1).min(self.finalized_until);
        if hi <= lo {
            return;
        }
        let y_lo = self.ola_base.max(lo - pad);
        let y_hi = self.finalized_until.min(hi + pad);
        let len = (y_hi - y_lo) as usize;
        if spec.sos.is_some() && len <= 15 {
            return;
        }
        // per channel: the output span (normalised, with corrections) and the source under it
        let frames = (source.len() / ch) as i64;
        let x_lo = input_center - (output_center - y_lo);
        let a = (y_lo - self.ola_base) as usize;
        let mut y_band = vec![vec![0f64; len]; ch];
        let mut x_band = vec![vec![0f64; len]; ch];
        for k in 0..len {
            let w = self.weight[a + k] as f64;
            for c in 0..ch {
                let i = (a + k) * ch + c;
                let v = if w > 1e-10 { self.ola[i] as f64 / w } else { 0.0 };
                y_band[c][k] = v + self.correction[i] as f64;
            }
            let s = x_lo + k as i64;
            if s >= 0 && s < frames {
                for c in 0..ch {
                    x_band[c][k] = source[s as usize * ch + c] as f64;
                }
            }
        }
        if let Some(sos) = &spec.sos {
            for c in 0..ch {
                y_band[c] = sosfiltfilt(sos, &y_band[c]);
                x_band[c] = sosfiltfilt(sos, &x_band[c]);
            }
        }
        let n = (hi - lo) as usize;
        let fade = spec.fade;
        let ramp: Vec<f64> = (0..fade).map(|i| (0.5 * PI * i as f64 / fade as f64).sin().powi(2)).collect();
        let mut shape = vec![1f64; n];
        for (k, s) in shape.iter_mut().enumerate() {
            let offset = lo + k as i64 - output_center; // -pre .. post
            let from_start = (offset + spec.pre) as usize;
            let to_end = (spec.post - offset) as usize;
            if from_start < fade {
                *s = ramp[from_start];
            }
            if to_end < fade {
                *s = s.min(ramp[to_end]);
            }
        }
        let base = (lo - y_lo) as usize;
        let start = ((lo - self.ola_base) as usize) * ch;
        if match_envelope {
            let envelope = |band: &Vec<Vec<f64>>| -> Vec<f64> {
                let power: Vec<f64> = (0..len).map(|k| band.iter().map(|b| b[k] * b[k]).sum::<f64>() / ch as f64).collect();
                let w = self.envelope_window as i64;
                let half = w / 2;
                (0..len as i64)
                    .map(|i| {
                        let mut s = 0.0;
                        for j in i - half..i - half + w {
                            s += power[j.clamp(0, len as i64 - 1) as usize];
                        }
                        (s / w as f64 + EPS).sqrt()
                    })
                    .collect()
            };
            let (ex, ey) = (envelope(&x_band), envelope(&y_band));
            for k in 0..n {
                let ratio = (ex[base + k] / (ey[base + k] + EPS)).clamp(0.60, 1.80);
                let g = amount * shape[k] * (ratio - 1.0);
                for c in 0..ch {
                    self.correction[start + k * ch + c] += (g * y_band[c][base + k]) as f32;
                }
            }
        } else {
            for k in 0..n {
                let g = amount * shape[k];
                for c in 0..ch {
                    self.correction[start + k * ch + c] += (g * (x_band[c][base + k] - y_band[c][base + k])) as f32;
                }
            }
        }
    }
}
