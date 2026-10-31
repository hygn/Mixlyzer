//! Streaming, percussion-aware, pitch-preserving time scaling, grown from audioengine/python/timestretch.py's
//! StreamingTimeStretcher (its event detector, RTPGHI with identity phase locking, the time map and the stream's
//! handovers are that one's), with the FFTs on realfft / rustfft and the vector work on VOLK. ``factor`` is input
//! frames per output frame. Two phase vocoders on the same frames and time map, split in the spectrum (see
//! ``Light``), and each attack made the source's by the spectrum (``restore_attack``).

use std::collections::{HashMap, VecDeque};
use std::f64::consts::PI;
use std::rc::Rc;
use std::sync::Arc;

use num_complex::Complex32;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};

use crate::split::BandSplit;
use crate::vk;

const EPS: f64 = 1e-12;
const TWO_PI: f64 = 2.0 * PI;

#[inline(always)]
fn wrap(v: f64) -> f64 {
    const INV_TWO_PI: f64 = 1.0 / TWO_PI;
    v - TWO_PI * round_even(v * INV_TWO_PI)
}

/// ``x.round_ties_even()`` (a zero's sign aside) without a library call: x86-64 without SSE4.1 has no rounding
/// instruction, and ``rint`` was called per bin (RTPGHI, the phase locking, the detector). Under 2^51, the addition
/// of 1.5 * 2^52 rounds x to an integer itself (the f64's step is 1 there, ties to even).
#[inline(always)]
fn round_even(x: f64) -> f64 {
    const MAGIC: f64 = 6_755_399_441_055_744.0;
    if x.abs() < 2_251_799_813_685_248.0 { (x + MAGIC) - MAGIC } else { x.round_ties_even() }
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

/// ln x within about 0.06, from the float's bits (Schraudolph): a threshold's background needs no more.
#[inline(always)]
fn fast_ln(x: f32) -> f32 {
    (x.to_bits() as f32 - 1_064_866_805.0) * 8.262_958e-8
}

/// A spectrum's partials (``power``: a frame's power per bin): each bin within ``lobe`` of a peak, from bin ``from``
/// on, ``ratio`` times over the background of the bins within ``reach`` either side, marked in ``partial`` (the
/// others left as they are). The background is a noise's median power: the bins' geometric mean over 0.81, from
/// running sums of the log power (``logsum``), compared in logs: O(bins), where a median per peak was the band's
/// dearest work.
fn mark_partials(power: &[f64], from: usize, ratio: f64, reach: usize, lobe: usize, logsum: &mut Vec<f64>,
                 partial: &mut [bool]) {
    let bins = power.len();
    logsum.clear();
    logsum.push(0.0);
    let mut sum = 0f64;
    for &p in power {
        sum += fast_ln(p as f32 + 1e-30) as f64;
        logsum.push(sum);
    }
    let over = (ratio * 1.23).ln();
    for m in from.max(1)..bins - 1 {
        let p = power[m];
        if !(p >= power[m - 1] && p >= power[m + 1]) {
            continue;
        }
        let (a, b) = (m.saturating_sub(reach), (m + reach + 1).min(bins));
        let background = (logsum[b] - logsum[a]) / (b - a) as f64;
        if logsum[m + 1] - logsum[m] > over + background {
            let (a, b) = (m.saturating_sub(lobe), (m + lobe + 1).min(bins));
            partial[a..b].iter_mut().for_each(|t| *t = true);
        }
    }
}

#[derive(Clone, Copy)]
pub struct Config {
    /// the event detector's window (the Python stretcher's)
    pub window_ms: f64,
    pub synthesis_hop_ms: f64,
    /// the zero padding of the detector's and the low band's FFTs, and of the upper band's: none there (its bins a
    /// third fewer, a fifth of the work less, chosen by listening; a chord's partials' contrast kept, its noise's
    /// level kept by ``NoiseBins::UNRELATED`` measured again)
    pub frequency_oversampling: f64,
    pub upper_oversampling: f64,
    pub detector_magnitude_floor: f64,
    pub mpd_lower: f64,
    pub mpd_upper: f64,
    pub detector_median_frames: i64,
    pub peak_prominence: f64,
    pub prominence_window_ms: f64,
    pub kick_low_hz: f64,
    pub phase_lock_strength: f64,
    pub stereo_coherence: bool,
    /// the bins RTPGHI integrates: over this of the frame's loudest (-60 dB); those under it, masked by the louder
    /// ones, keep their analysis phase (-120 dB integrated bins nobody hears)
    pub rtpghi_tolerance: f64,
    pub dry_transient_threshold: f64,
    pub dry_transient_full_strength: f64,
    /// the low band's vocoder's window: a held note's partials resolved (a shorter one rang: a noise vocoded on a
    /// coarser grid is a sum of steadier sinusoids), its smear before an attack gated. With the crossover at 1.5 kHz,
    /// 80 ms (1 kHz and 110 ms before, chosen by listening): a vibrato's partials move with their frequency, their
    /// smear in the window with its length, so the higher crossover takes the shorter window
    pub low_window_ms: f64,
    /// the upper band's: from ``mid_hz`` up, a window of ``mid_window_hops`` synthesis hops (every frame). Above a
    /// voice's resolved harmonics (about its eighth) the ear hears the pitch in the waveform's period, not in the
    /// partials, so the time structure matters there: and a vibrato or a glide moves its partials by tens of Hz
    /// within a long window, the bins' phases then at odds (a level beating at twice the vibrato's rate: the "LFO"
    /// of a slowed voice). Short, its noise's bins are made noise again (``noise_*``)
    pub mid_hz: f64,
    pub mid_window_hops: usize,
    /// the crossover's width: each bin is one vocoder's, so no tone is the sum of two unrelated phases; the
    /// crossover moved each frame to the spectrum's valley nearest it, within ``crossover_range_hz`` of ``mid_hz``,
    /// by ``crossover_slew_hz`` a frame at most (``valley``)
    pub crossover_width_hz: f64,
    pub crossover_range_hz: f64,
    pub crossover_slew_hz: f64,
    /// the upper band's noise: a bin within ``noise_lobe_hz`` of a peak ``noise_peak_db`` over the background (a
    /// noise's median power) of the bins within ``noise_median_hz`` is a partial's (a voice's harmonics, between
    /// them nothing: far over it, a vibrato's smeared ones too); so is one within ``partial_lobe_hz`` of a peak
    /// ``partial_peak_db`` over it in the event detector's spectrum (its window two of the band's: the partials of a
    /// chord, too close for the band's short window, resolved there; from ``partial_from_hz``, below the
    /// crossover's range). The others are noise's, their synthesis phase turned at random each frame, by up to a
    /// whole turn from half speed down (the
    /// slower, the longer a bin's phase advance holds its noise still as a sinusoid: a metallic ring), none at unity,
    /// their level kept (a turned frame overlap-adds partly unrelated). A partial turned as noise is heard blurred: a
    /// chord's notes lost their pitch slowed down
    pub noise_peak_db: f64,
    pub noise_median_hz: f64,
    pub noise_lobe_hz: f64,
    pub partial_peak_db: f64,
    pub partial_lobe_hz: f64,
    pub partial_from_hz: f64,
    /// the attack's spectra: their window, how long after the attack its frames reach, how wide in Hz a bin's
    /// power is averaged when compared (a noise's single bins are as random as its phase), the bass left alone, and
    /// above ``attack_whole_hz`` the attack added whole whatever the event (a cymbal's, a hat's)
    pub attack_window_ms: f64,
    pub attack_after_ms: f64,
    pub attack_smoothing_hz: f64,
    pub attack_bass_hz: f64,
    pub attack_whole_hz: f64,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            window_ms: 93.0,
            synthesis_hop_ms: 10.67,
            frequency_oversampling: 1.5,
            upper_oversampling: 1.0,
            detector_magnitude_floor: 0.01,
            mpd_lower: 0.50,
            mpd_upper: 0.75,
            detector_median_frames: 5,
            peak_prominence: 0.10,
            prominence_window_ms: 90.0,
            kick_low_hz: 180.0,
            phase_lock_strength: 1.0,
            stereo_coherence: true,
            rtpghi_tolerance: 1e-3,
            dry_transient_threshold: 0.40,
            dry_transient_full_strength: 0.68,
            low_window_ms: 80.0,
            mid_hz: 1500.0,
            mid_window_hops: 4,
            crossover_width_hz: 100.0,
            crossover_range_hz: 300.0,
            crossover_slew_hz: 50.0,
            noise_peak_db: 18.0,
            noise_median_hz: 180.0,
            noise_lobe_hz: 45.0,
            partial_peak_db: 8.0,
            partial_lobe_hz: 50.0,
            partial_from_hz: 1100.0,
            attack_window_ms: 20.0,
            attack_after_ms: 20.0,
            attack_smoothing_hz: 150.0,
            attack_bass_hz: 200.0,
            attack_whole_hz: 6500.0,
        }
    }
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
    // per frame (oldest first), its partials (``Config::partial_peak_db``), which the upper band's frames read
    // (``partials_at``): the detector runs ahead of them
    partials: VecDeque<(i64, Vec<bool>)>,
    partial_ratio: f64,
    partial_reach: usize,
    partial_lobe: usize,
    partial_from: usize,
    power: Vec<f64>,
    logsum: Vec<f64>,
}

impl Detector {
    /// The frames of partials kept: 1.7 s of the source at 48 kHz, past the detector's lead on the vocoder's frames.
    const PARTIAL_FRAMES: usize = 160;

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
        let bin_hz = rate / n_fft as f64;
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
            partials: VecDeque::with_capacity(Self::PARTIAL_FRAMES),
            partial_ratio: 10f64.powf(cfg.partial_peak_db / 10.0),
            partial_reach: 1.max((cfg.noise_median_hz / bin_hz).round() as usize),
            partial_lobe: (cfg.partial_lobe_hz / bin_hz).round() as usize,
            partial_from: (cfg.partial_from_hz / bin_hz) as usize,
            power: vec![0.0; bins],
            logsum: Vec::with_capacity(bins + 1),
            fft,
        };
        d.start(0);
        d
    }

    /// The partials of the frame nearest at or before ``j`` (none before the stream's first).
    fn partials_at(&self, j: i64) -> Option<&[bool]> {
        let k = self.partials.partition_point(|f| f.0 <= j);
        (k > 0).then(|| self.partials[k - 1].1.as_slice())
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
        self.partials.clear();
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
        // the frame's partials, in the oldest frame's vector once all are kept
        let mut mask = if self.partials.len() >= Self::PARTIAL_FRAMES {
            self.partials.pop_front().map(|f| f.1).unwrap_or_default()
        } else {
            vec![false; self.bins]
        };
        mask.iter_mut().for_each(|t| *t = false);
        for (p, m) in self.power.iter_mut().zip(self.mag.iter()) {
            *p = (*m as f64) * (*m as f64);
        }
        mark_partials(&self.power, self.partial_from, self.partial_ratio, self.partial_reach, self.partial_lobe,
                      &mut self.logsum, &mut mask);
        self.partials.push_back((j, mask));

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
/// first-out order: the order of integration is RTPGHI's heuristic, not its result's definition. With ``only``, those
/// bins alone are integrated (the others keep their analysis phases).
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
    only: Option<&[bool]>,
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
        if joint[m] as f64 > floor && only.is_none_or(|o| o[m]) {
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

/// The analysis and synthesis windows for one low-rate window length ``wl`` (both at the low rate).
struct LightWindow {
    analysis: Vec<f32>,
    center_ramp: Vec<Complex32>,
    inverse_ramp: Vec<Complex32>,
    synthesis: Vec<f32>,
    synthesis_sq: Vec<f32>,
}

/// The STFT ``restore_attack`` works in (a periodic Hann, a quarter overlap) and its buffers.
struct AttackSpectra {
    n: usize,
    hop: usize,
    window: Vec<f32>,
    fft: Arc<dyn RealToComplex<f32>>,
    ifft: Arc<dyn ComplexToReal<f32>>,
    scratch: Vec<Complex32>,
    input: Vec<f32>,
    out: Vec<f32>,
    x: Vec<Vec<Complex32>>, // [ch][bins]
    y: Vec<Complex32>,
}

impl AttackSpectra {
    fn new(ch: usize, n: usize, planner: &mut RealFftPlanner<f32>) -> AttackSpectra {
        let fft = planner.plan_fft_forward(n);
        let ifft = planner.plan_fft_inverse(n);
        let bins = n / 2 + 1;
        let scratch = vec![Complex32::new(0.0, 0.0); fft.get_scratch_len().max(ifft.get_scratch_len())];
        AttackSpectra {
            n,
            hop: n / 4,
            window: periodic_hann(n),
            fft,
            ifft,
            scratch,
            input: vec![0.0; n],
            out: vec![0.0; n],
            x: vec![vec![Complex32::new(0.0, 0.0); bins]; ch],
            y: vec![Complex32::new(0.0, 0.0); bins],
        }
    }
}

/// The upper band's phase vocoder: the source at the full rate, on the low band's frames (their centres: one time
/// map), a periodic Hann of four of its hops (analysis and synthesis: the overlap sums to 1.5), only its band's bins
/// (``gain``), RTPGHI and the stereo rotation as the low band's; its noise's bins turned at random (``NoiseBins``).
struct UpperVocoder {
    window: Vec<f32>,
    n_fft: usize,
    bins: usize,
    fft: Arc<dyn RealToComplex<f32>>,
    ifft: Arc<dyn ComplexToReal<f32>>,
    fft_scratch: Vec<Complex32>,
    ifft_scratch: Vec<Complex32>,
    input: Vec<f32>,
    spec: Vec<Complex32>,
    output: Vec<f32>,
    center_ramp: Vec<Complex32>,
    inverse_ramp: Vec<Complex32>,
    coeff: Vec<Complex32>,
    analysis_phase: Vec<f64>,
    synthesis_phase: Vec<f64>,
    previous_analysis_phase: Vec<f64>,
    previous_synthesis_phase: Vec<f64>,
    magnitude: Vec<f32>,
    joint: Vec<f32>,
    previous_joint: Vec<f32>,
    phase32: Vec<f32>,
    phase_work: PhaseWork,
    coherent_work: CoherentWork,
    previous_center: Option<i64>,
    frames: Vec<f32>,    // [ch, window]: the last frame's synthesis, windowed and scaled
    frames_in: Vec<f32>, // [ch, window]: the frame's input
    gain: Vec<f32>,      // per bin: its share of the crossover
    noise: NoiseBins,
}

/// The noise's bins of the upper band (``Config::noise_peak_db``): what tells them from a partial's, and the frame's
/// work.
struct NoiseBins {
    peak_ratio: f64,          // a partial's peak over its neighbourhood's background
    reach: usize,             // the neighbourhood, either side
    lobe: usize,              // a partial's bins, either side of its peak
    to_detector: Vec<usize>,  // per bin: the event detector's bin at its frequency
    rng: u64,                 // xorshift64
    level: Vec<f32>,          // per bin: 1, or a noise's bin's level kept
    power: Vec<f64>,
    partial: Vec<bool>,
    logsum: Vec<f64>,
}

impl NoiseBins {
    /// The share of a frame's level a fully turned (random) bin keeps overlap-added with its neighbours, measured on
    /// pink noise: four hops of a periodic Hann, not zero-padded (a turned spectrum spreads past the window, cut by
    /// it; 0.19 at 1.5 times zero padding).
    const UNRELATED: f64 = 0.26;

    /// A frame's partials (``joint``: its joint magnitude): those of the event detector's frame (``detected``) and
    /// those this frame's own spectrum shows.
    fn classify(&mut self, joint: &[f32], detected: Option<&[bool]>) {
        for (p, j) in self.power.iter_mut().zip(joint.iter()) {
            *p = (*j as f64) * (*j as f64);
        }
        match detected {
            Some(d) => {
                for (t, &k) in self.partial.iter_mut().zip(self.to_detector.iter()) {
                    *t = d[k];
                }
            }
            None => self.partial.iter_mut().for_each(|t| *t = false),
        }
        mark_partials(&self.power, 0, self.peak_ratio, self.reach, self.lobe, &mut self.logsum, &mut self.partial);
    }

    /// The noise's bins of a frame (``joint``: its joint magnitude) turned by up to ``jitter`` of a turn at random,
    /// every channel alike (the stereo image kept), and their level kept: a turn of up to j turns keeps sinc(j) of
    /// a bin's relation to the next frame (rho), and the level is 1 / sqrt(rho^2 + (1 - rho^2) UNRELATED). The
    /// partials are ``classify``'s.
    fn turn(&mut self, synthesis_phase: &mut [f64], ch: usize, jitter: f64) {
        let bins = self.partial.len();
        let rho = (PI * jitter).sin() / (PI * jitter);
        let level = (1.0 / (rho * rho + (1.0 - rho * rho) * Self::UNRELATED).sqrt()) as f32;
        self.level.iter_mut().for_each(|g| *g = 1.0);
        for m in 0..bins {
            if self.partial[m] {
                continue;
            }
            self.rng ^= self.rng << 13;
            self.rng ^= self.rng >> 7;
            self.rng ^= self.rng << 17;
            let u = (self.rng >> 11) as f64 / (1u64 << 53) as f64;
            let turn = jitter * (u - 0.5) * TWO_PI;
            for c in 0..ch {
                synthesis_phase[c * bins + m] = wrap(synthesis_phase[c * bins + m] + turn);
            }
            self.level[m] = level;
        }
    }
}

/// A crossover between two vocoders, per bin of an ``n_fft`` grid at ``rate``: 0 below ``hz - width / 2``,
/// 1 above ``hz + width / 2``, sin^2 between (``high``; else its complement).
fn crossover_gain(n_fft: usize, rate: f64, hz: f64, width: f64, high: bool) -> Vec<f32> {
    (0..n_fft / 2 + 1).map(|k| crossover_at(k as f64 * rate / n_fft as f64, hz, width, high)).collect()
}

fn crossover_at(f: f64, hz: f64, width: f64, high: bool) -> f32 {
    let t = ((f - (hz - 0.5 * width)) / width).clamp(0.0, 1.0);
    let g = (0.5 * PI * t).sin().powi(2);
    (if high { g } else { 1.0 - g }) as f32
}

/// ``gain`` (``crossover_gain``'s, its bins ``bin_hz`` apart) moved from ``from_hz`` to ``to_hz``: only the bins
/// either ramp covers set again.
fn move_crossover(gain: &mut [f32], bin_hz: f64, from_hz: f64, to_hz: f64, width: f64, high: bool) {
    let a = ((from_hz.min(to_hz) - 0.5 * width) / bin_hz).floor().max(0.0) as usize;
    let b = (((from_hz.max(to_hz) + 0.5 * width) / bin_hz).ceil() as usize + 1).min(gain.len());
    for (k, g) in gain.iter_mut().enumerate().take(b).skip(a) {
        *g = crossover_at(k as f64 * bin_hz, to_hz, width, high);
    }
}

/// The crossover's frequency this frame: from ``hz`` (the last frame's), downhill on
/// ``joint`` (the low band's joint magnitude, bins ``bin_hz`` apart, summed over 3 bins) by ``slew_hz`` at most, to
/// the valley nearest it, inside ``default_hz`` +- ``range_hz``. A voice's partial gliding across a fixed crossover
/// was the sum of two vocoders' unrelated phases, its level beating (a vibrato's harmonic there trembled at -50 %);
/// between the partials a valley is moved through by none.
fn valley(joint: &[f32], bin_hz: f64, hz: f64, default_hz: f64, range_hz: f64, slew_hz: f64) -> f64 {
    let lo = (((default_hz - range_hz) / bin_hz).ceil() as usize).max(2);
    let hi = (((default_hz + range_hz) / bin_hz).floor() as usize).min(joint.len().saturating_sub(3));
    if hi <= lo {
        return default_hz;
    }
    let level = |m: usize| joint[m - 1] + joint[m] + joint[m + 1];
    let mut k = ((hz / bin_hz).round() as usize).clamp(lo, hi);
    for _ in 0..(slew_hz / bin_hz).round() as usize {
        let (down, here, up) = (level(k - 1), level(k), level(k + 1));
        if down < here && down <= up && k > lo {
            k -= 1;
        } else if up < here && k < hi {
            k += 1;
        } else {
            break;
        }
    }
    k as f64 * bin_hz
}

impl UpperVocoder {
    /// The band from ``Config::mid_hz`` up, a window of ``length``; the event detector's bins ``detector_bin_hz``
    /// wide.
    fn new(ch: usize, length: usize, rate: f64, cfg: &Config, detector_bin_hz: f64,
           planner: &mut RealFftPlanner<f32>) -> UpperVocoder {
        let oversampling = cfg.upper_oversampling;
        let n_fft = fast_even_length((length as f64 * oversampling).ceil() as usize);
        let bins = n_fft / 2 + 1;
        let zc = Complex32::new(0.0, 0.0);
        let fft = planner.plan_fft_forward(n_fft);
        let ifft = planner.plan_fft_inverse(n_fft);
        let half = (length / 2) as f64;
        let ramp = |sign: f64| -> Vec<Complex32> {
            (0..bins)
                .map(|k| {
                    let a = TWO_PI * k as f64 / n_fft as f64 * half;
                    Complex32::new(a.cos() as f32, (sign * a.sin()) as f32)
                })
                .collect()
        };
        let gain = crossover_gain(n_fft, rate, cfg.mid_hz, cfg.crossover_width_hz, true);
        let bin_hz = rate / n_fft as f64;
        let noise = NoiseBins {
            peak_ratio: 10f64.powf(cfg.noise_peak_db / 10.0),
            reach: 1.max((cfg.noise_median_hz / bin_hz).round() as usize),
            lobe: (cfg.noise_lobe_hz / bin_hz).round() as usize,
            to_detector: (0..bins).map(|m| (m as f64 * bin_hz / detector_bin_hz).round() as usize).collect(),
            rng: 0x9E37_79B9_7F4A_7C15,
            level: vec![1.0; bins],
            power: vec![0.0; bins],
            partial: vec![false; bins],
            logsum: Vec::with_capacity(bins + 1),
        };
        UpperVocoder {
            window: periodic_hann(length),
            n_fft,
            bins,
            fft_scratch: fft.make_scratch_vec(),
            ifft_scratch: ifft.make_scratch_vec(),
            fft,
            ifft,
            input: vec![0.0; n_fft],
            spec: vec![zc; bins],
            output: vec![0.0; n_fft],
            center_ramp: ramp(1.0),
            inverse_ramp: ramp(-1.0),
            coeff: vec![zc; ch * bins],
            analysis_phase: vec![0.0; ch * bins],
            synthesis_phase: vec![0.0; ch * bins],
            previous_analysis_phase: vec![0.0; ch * bins],
            previous_synthesis_phase: vec![0.0; ch * bins],
            magnitude: vec![0.0; ch * bins],
            joint: vec![0.0; bins],
            previous_joint: vec![0.0; bins],
            phase32: vec![0.0; bins],
            phase_work: PhaseWork::default(),
            coherent_work: CoherentWork::default(),
            previous_center: None,
            frames: vec![0.0; ch * length],
            frames_in: vec![0.0; ch * length],
            gain,
            noise,
        }
    }

    /// The frame about input ``center`` of the source, synthesised for an output hop of ``synthesis_hop`` into
    /// ``frames``; its noise's bins turned by up to ``jitter`` of a turn, the event detector's partials (``detected``)
    /// kept.
    #[allow(clippy::too_many_arguments)]
    fn frame(&mut self, source: &[f32], ch: usize, center: i64, synthesis_hop: i64, tolerance: f64,
             lock_strength: f64, stereo: bool, jitter: f64, detected: Option<&[bool]>) {
        let wl = self.window.len();
        let bins = self.bins;
        let left = center - (wl / 2) as i64;
        let frames = (source.len() / ch) as i64;
        for k in 0..wl {
            let i = left + k as i64;
            for c in 0..ch {
                self.frames_in[c * wl + k] = if i >= 0 && i < frames { source[i as usize * ch + c] } else { 0.0 };
            }
        }
        for c in 0..ch {
            vk::mul(&mut self.input[..wl], &self.frames_in[c * wl..(c + 1) * wl], &self.window);
            self.input[wl..].iter_mut().for_each(|v| *v = 0.0);
            self.fft.process_with_scratch(&mut self.input, &mut self.spec, &mut self.fft_scratch).ok();
            vk::cmul_in(&mut self.spec, &self.center_ramp);
            for (v, g) in self.spec.iter_mut().zip(self.gain.iter()) {
                *v *= *g;
            }
            self.coeff[c * bins..(c + 1) * bins].copy_from_slice(&self.spec);
            vk::magnitude(&mut self.magnitude[c * bins..(c + 1) * bins], &self.spec);
            vk::phase(&mut self.phase32, &self.spec);
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
        if let Some(previous) = self.previous_center {
            if jitter > 0.0 {
                self.noise.classify(&self.joint, detected);
            }
            // turned a whole turn at random, the noise's phases are random whatever RTPGHI gives them (the phase
            // locking sets every other bin's from its peak's): only the partials' integrated
            let only = (jitter >= 1.0).then_some(self.noise.partial.as_slice());
            rtpghi_frame(
                &mut self.synthesis_phase,
                &self.analysis_phase,
                &self.joint,
                &self.previous_analysis_phase,
                &self.previous_synthesis_phase,
                &self.previous_joint,
                ch,
                bins,
                1.max(center - previous),
                synthesis_hop,
                self.n_fft,
                tolerance,
                lock_strength,
                only,
                &mut self.phase_work,
            );
            if jitter > 0.0 {
                self.noise.turn(&mut self.synthesis_phase, ch, jitter);
            }
            if stereo && ch > 1 {
                coherent_synthesis(&mut self.synthesis_phase, &self.analysis_phase, &self.magnitude, &mut self.coeff, ch,
                                   bins, &mut self.coherent_work);
            } else {
                for i in 0..ch * bins {
                    let (s, c) = self.synthesis_phase[i].sin_cos();
                    let mag = self.magnitude[i] as f64;
                    self.coeff[i] = Complex32::new((mag * c) as f32, (mag * s) as f32);
                }
            }
            if jitter > 0.0 {
                for c in 0..ch {
                    for (v, g) in self.coeff[c * bins..(c + 1) * bins].iter_mut().zip(self.noise.level.iter()) {
                        *v *= *g;
                    }
                }
            }
        } else {
            self.synthesis_phase.copy_from_slice(&self.analysis_phase);
        }
        // the inverse unscaled (1 / n_fft) and the Hann^2 overlap (3/8 of the window over the hop)
        let scale = 1.0 / (self.n_fft as f32 * 0.375 * wl as f32 / synthesis_hop as f32);
        for c in 0..ch {
            self.spec.copy_from_slice(&self.coeff[c * bins..(c + 1) * bins]);
            vk::cmul_in(&mut self.spec, &self.inverse_ramp);
            self.spec[0].im = 0.0;
            self.spec[bins - 1].im = 0.0;
            self.ifft.process_with_scratch(&mut self.spec, &mut self.output, &mut self.ifft_scratch).ok();
            let frame = &mut self.frames[c * wl..(c + 1) * wl];
            vk::mul(frame, &self.output[..wl], &self.window);
            frame.iter_mut().for_each(|v| *v *= scale);
        }
        std::mem::swap(&mut self.previous_analysis_phase, &mut self.analysis_phase);
        std::mem::swap(&mut self.previous_synthesis_phase, &mut self.synthesis_phase);
        self.previous_joint.copy_from_slice(&self.joint);
        self.previous_center = Some(center);
    }
}

/// The two vocoders, each bin one's (the crossover 100 Hz wide, each frame at a spectral valley within 300 Hz of
/// 1.5 kHz: ``valley``), on the same frames: one time map. The low band's below it (``Config::mid_hz``), on the
/// source decimated by ``d`` (split.rs: 8 at 48 kHz, its RTPGHI's queue a bucket queue: ``rtpghi_frame``), a long window (a voice's resolved harmonics, the bass), synthesised at its own rate
/// (its bins above the crossover dropped: ``low_gain``) and brought to the full rate by the split's lowpass where no
/// later frame reaches (``upsample_low``). The upper band's (``UpperVocoder``) on the source itself, a short window
/// every frame (a voice's unresolved harmonics: the period's shape, a vibrato followed), its noise's bins turned at
/// random as the stretch slows. A third vocoder above 6.5 kHz with a window three times as long, every third frame,
/// kept the air a little noisier, at a quarter more of the work. The windows' smear before an attack is gated, bin by
/// bin, and the attack is the source's (``restore_attack``).
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
    low_gain: Vec<f32>, // per bin of the low band's: its share of the crossover
    upper: UpperVocoder,
    attack: AttackSpectra,
    // the low band overlap-added at its rate (from ``low_base``), and how far it is brought to the full rate
    ola_low: Vec<f32>, // [len, ch]
    weight_low: Vec<f32>,
    low_base: i64,
    up_done: i64,
    lows: Vec<Vec<f32>>, // per channel: the normalised low samples an interpolation reads
    cross_hz: f64,       // the crossover this frame (``valley``), and where a new stream starts it
    cross_default_hz: f64,
    cross_width_hz: f64,
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
    low_window: usize, // the low band's vocoder's window (full-rate samples)
    reach: usize,      // the longest window of either band's
    hop: i64,
    n_fft: usize,
    bins: usize,
    detector: Detector,
    event_reach: i64,
    handover_timeout: i64,
    envelope_window: usize,
    // an attack's restoration reaches this far after its destination (and, before it, the vocoders' smear and the
    // spectra's whole window): the output is finalized the two ahead of its reading
    attack_after: i64,
    attack_ahead: i64,
    attack_lead: i64,

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
    coeff: Vec<Complex32>,       // [ch, bins]
    analysis_phase: Vec<f64>,    // [ch, bins]
    synthesis_phase: Vec<f64>,   // [ch, bins]
    magnitude: Vec<f32>,         // [ch, bins]
    joint: Vec<f32>,             // [bins]
    phase32: Vec<f32>,
    phase_work: PhaseWork,
    coherent_work: CoherentWork,
    light: Option<Box<Light>>, // always there: taken out while a frame works on it beside the rest
    ola_upper: Vec<f32>,       // [len, ch]: the upper band
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

    /// The stretcher at ``sample_rate``, 44.1 kHz or more (ipc::MIN_RATE: the engine plays at no less).
    pub fn new(channels: usize, sample_rate: u32) -> Stretcher {
        let cfg = Config::default();
        let rate = sample_rate as f64;
        // the low band decimated to keep the crossover and its transition (and a margin) in its passband
        let split = BandSplit::new(channels, sample_rate, cfg.mid_hz + cfg.crossover_width_hz + 50.0)
            .expect("the time stretch at 44.1 kHz or more");
        let d = split.d;
        // the detector's grid and the window lengths as the Python stretcher's at this rate
        let requested = 128.max((rate * cfg.window_ms / 1000.0).round_ties_even() as usize);
        let full_window = requested + (requested & 1);
        let full_hop = 1.max((rate * cfg.synthesis_hop_ms / 1000.0).round_ties_even() as i64);
        let detector_fft = ((full_window as f64 * cfg.frequency_oversampling).ceil() as usize).next_power_of_two();
        let mut planner = RealFftPlanner::<f32>::new();
        let detector = Detector::new(rate, full_window, full_hop, detector_fft, &cfg, &mut planner);
        let event_reach = (full_window as f64 / (2.0 * full_hop as f64)).ceil() as i64;
        let handover_timeout = 1.max((rate * Self::HANDOVER_TIMEOUT_SEC / full_hop as f64).round_ties_even() as i64);
        let ms = |v: f64| 1.max((v * rate / 1000.0).round_ties_even() as i64);
        let envelope_window = 3.max(ms(1.0) as usize);
        // the gate and the attack's crossfade end this far before the attack's onset
        let attack_lead = ms(1.0);
        // the vocoders' frames: the low band's rate (1 / d), its hop a whole number of low-rate samples
        let hop_low = 1.max((full_hop as f64 / d as f64).round_ties_even() as i64);
        let hop = d as i64 * hop_low;
        let wl_low = Self::low_length(d, (rate * cfg.low_window_ms / 1000.0).round() as usize);
        let low_window = d * wl_low;
        let upper_window = cfg.mid_window_hops * hop as usize;
        let reach = low_window.max(upper_window);
        let n_fft = fast_even_length((wl_low as f64 * cfg.frequency_oversampling).ceil() as usize);
        let bins = n_fft / 2 + 1;
        let zc = Complex32::new(0.0, 0.0);
        let fft = planner.plan_fft_forward(n_fft);
        let ifft = planner.plan_fft_inverse(n_fft);
        let attack_n = ((rate * cfg.attack_window_ms / 1000.0) as usize).next_power_of_two();
        let attack_half = (attack_n / 2) as i64;
        let attack_before = (reach / 2) as i64 + 2 * attack_half;
        let attack_after = ms(cfg.attack_after_ms) + attack_half + 1;
        let attack_ahead = attack_before + attack_after;
        let light = Light {
            hop_low,
            fft_scratch: fft.make_scratch_vec(),
            ifft_scratch: ifft.make_scratch_vec(),
            fft_in: vec![0.0; n_fft],
            fft_out: vec![zc; bins],
            synth_spec: vec![zc; bins],
            synth_out: vec![0.0; n_fft],
            fft,
            ifft,
            windows: HashMap::new(),
            previous_center: None,
            rate,
            split,
            low_gain: crossover_gain(n_fft, rate / d as f64, cfg.mid_hz, cfg.crossover_width_hz, false),
            upper: UpperVocoder::new(channels, upper_window, rate, &cfg, rate / detector_fft as f64, &mut planner),
            attack: AttackSpectra::new(channels, attack_n, &mut planner),
            ola_low: Vec::new(),
            weight_low: Vec::new(),
            low_base: 0,
            up_done: 0,
            lows: Vec::new(),
            cross_hz: cfg.mid_hz,
            cross_default_hz: cfg.mid_hz,
            cross_width_hz: cfg.crossover_width_hz,
        };
        let mut s = Stretcher {
            ch: channels,
            cfg,
            low_window,
            reach,
            hop,
            n_fft,
            bins,
            detector,
            event_reach,
            handover_timeout,
            envelope_window,
            attack_after,
            attack_ahead,
            attack_lead,
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
            coeff: vec![zc; channels * bins],
            analysis_phase: vec![0.0; channels * bins],
            synthesis_phase: vec![0.0; channels * bins],
            magnitude: vec![0.0; channels * bins],
            joint: vec![0.0; bins],
            phase32: vec![0.0; bins],
            phase_work: PhaseWork::default(),
            coherent_work: CoherentWork::default(),
            light: Some(Box::new(light)),
            ola_upper: vec![],
        };
        s.reset();
        s
    }

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
        self.ola_upper.clear();
        if let Some(light) = &mut self.light {
            light.restart();
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
        let pre_roll = ((self.reach / 2) as f64 / self.hop as f64).ceil() as i64;
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
        let initial = self.reach + 2 * self.hop as usize + self.attack_ahead as usize;
        self.ola = vec![0.0; initial * self.ch];
        self.weight = vec![0.0; initial];
        self.correction = vec![0.0; initial * self.ch];
        self.ola_upper = vec![0.0; initial * self.ch];
        if let Some(light) = &mut self.light {
            light.restart();
        }
    }

    fn process_frame(&mut self, source: &[f32]) {
        let factor = self.factor.unwrap_or(1.0);
        let input_center = (self.next_analysis_center + 0.5).floor() as i64;
        let output_center = self.next_synthesis_center;
        // the events confirmed ahead of the frame (``queue_events``, the attacks); both vocoders' windows fixed: a
        // window shortened about each event (the Python stretcher's SELEBI) moved held notes by 1 - 2 dB at every hit
        let dh = self.detector.hop;
        let center_frame = (input_center as f64 / dh as f64).round_ties_even() as i64;
        self.detector.ensure(source, self.ch, center_frame + self.event_reach + 1);
        let (window_length, lock_strength) = (self.low_window, self.cfg.phase_lock_strength);
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
                None,
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
        // the crossover at the valley nearest it, both bands' gains with it
        let low_bin_hz = light.rate / d as f64 / self.n_fft as f64;
        let cfg = &self.cfg;
        let cross = valley(&self.joint, low_bin_hz, light.cross_hz, cfg.mid_hz, cfg.crossover_range_hz,
                           cfg.crossover_slew_hz);
        light.set_crossover(cross, low_bin_hz);
        self.overlap_add_light(&mut light, output_center, window_length);
        // the upper band's noise turned by up to a turn per octave slowed (a whole one from half speed down)
        let jitter = (1.0 / factor).log2().clamp(0.0, 1.0);
        self.vocode_upper(&mut light.upper, source, input_center, output_center, lock_strength, jitter);
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
        // finished where no later frame reaches, nor the low band's interpolation from a sample still to be finished
        let margin = light.split.half + 2 * light.split.d as i64;
        self.finalized_until = self.finalized_until.max(self.next_synthesis_center - (self.reach / 2) as i64 - margin);
        self.upsample_low(&mut light, self.finalized_until);
        self.light = Some(light);
    }

    fn light_window(&self, light: &mut Light, wl: usize) -> Rc<LightWindow> {
        if let Some(w) = light.windows.get(&wl) {
            return w.clone();
        }
        let half = (wl / 2) as f64;
        let (mut center_ramp, mut inverse_ramp) = (Vec::with_capacity(self.bins), Vec::with_capacity(self.bins));
        for k in 0..self.bins {
            let a = TWO_PI * k as f64 / self.n_fft as f64 * half;
            center_ramp.push(Complex32::new(a.cos() as f32, a.sin() as f32));
            inverse_ramp.push(Complex32::new(a.cos() as f32, -a.sin() as f32));
        }
        let synthesis = periodic_hann(wl);
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

    /// The low band's synthesis frame (``center`` at the full rate) at the low rate, overlap-added into the low
    /// band's buffer; ``upsample_low`` brings what is finished to the full rate.
    fn overlap_add_light(&mut self, light: &mut Light, center: i64, wl: usize) {
        let wd = self.light_window(light, wl);
        let ch = self.ch;
        let center = center.div_euclid(light.split.d as i64);
        let mut left = center - (wl / 2) as i64;
        let right = left + wl as i64;
        let mut src_start = 0usize;
        if left < light.low_base {
            src_start = (light.low_base - left) as usize;
            left = light.low_base;
        }
        if right <= left || src_start >= wl {
            return;
        }
        let need = (right - light.low_base) as usize;
        if light.weight_low.len() < need {
            light.weight_low.resize(need + 1024, 0.0);
            light.ola_low.resize((need + 1024) * ch, 0.0);
        }
        let start = (left - light.low_base) as usize;
        let span = (right - left) as usize;
        let scale = 1.0 / self.n_fft as f32;
        let bins = self.bins;
        for c in 0..ch {
            let spec = &mut light.synth_spec;
            spec.copy_from_slice(&self.coeff[c * bins..(c + 1) * bins]);
            // the bins above the crossover are the upper band's
            for (v, g) in spec.iter_mut().zip(light.low_gain.iter()) {
                *v *= *g;
            }
            vk::cmul_in(spec, &wd.inverse_ramp);
            spec[0].im = 0.0;
            spec[bins - 1] = Complex32::new(0.0, 0.0); // the low rate's Nyquist bin: the band stops below it
            light.ifft.process_with_scratch(spec, &mut light.synth_out, &mut light.ifft_scratch).ok();
            for k in 0..span {
                let j = src_start + k;
                light.ola_low[(start + k) * ch + c] += light.synth_out[j] * wd.synthesis[j] * scale;
            }
        }
        vk::add_in(&mut light.weight_low[start..start + span], &wd.synthesis_sq[src_start..src_start + span]);
    }

    /// The low band's finished part, ``[light.up_done, until)``, interpolated to the full rate into ``ola`` (its
    /// weight 1): the split's lowpass, polyphase, over the low band normalised by its overlap.
    fn upsample_low(&mut self, light: &mut Light, until: i64) {
        if until <= light.up_done {
            return;
        }
        let ch = self.ch;
        self.ensure_capacity(until);
        let from = light.up_done.max(self.ola_base);
        let d = light.split.d as i64;
        // the low samples the range reads
        let k0 = (from + light.split.half).div_euclid(d) - light.split.taps_per_phase as i64 + 1;
        let k1 = (until - 1 + light.split.half).div_euclid(d) + 1;
        let lows = &mut light.lows;
        lows.resize_with(ch, Vec::new);
        for l in lows.iter_mut() {
            l.clear();
        }
        for k in k0..k1 {
            let i = k - light.low_base;
            let w = if i >= 0 && (i as usize) < light.weight_low.len() { light.weight_low[i as usize] } else { 0.0 };
            for (c, l) in lows.iter_mut().enumerate() {
                l.push(if w > 1e-10 { light.ola_low[i as usize * ch + c] / w } else { 0.0 });
            }
        }
        for t in from..until {
            let i = (t - self.ola_base) as usize;
            for c in 0..ch {
                self.ola[i * ch + c] = light.split.upsample(&lows[c], k0, t);
            }
            self.weight[i] = 1.0;
        }
        light.up_done = until;
    }

    /// The upper band's frame about this frame's centres, overlap-added at its output centre.
    #[allow(clippy::too_many_arguments)]
    fn vocode_upper(&mut self, v: &mut UpperVocoder, source: &[f32], input_center: i64, output_center: i64, lock: f64,
                    jitter: f64) {
        let ch = self.ch;
        let synthesis_hop = self.hop;
        let wl = v.window.len() as i64;
        let cfg = &self.cfg;
        // the event detector's partials at the frame's centre
        let detected = self.detector.partials_at((input_center as f64 / self.detector.hop as f64).round() as i64);
        v.frame(source, ch, input_center, synthesis_hop, cfg.rtpghi_tolerance, lock, cfg.stereo_coherence, jitter,
                detected);
        let mut left = output_center - wl / 2;
        let right = left + wl;
        let mut offset = 0i64;
        if left < self.ola_base {
            offset = self.ola_base - left;
            left = self.ola_base;
        }
        if right <= left {
            return;
        }
        self.ensure_capacity(right);
        let base = (left - self.ola_base) as usize;
        let n = (right - left) as usize;
        for c in 0..ch {
            let frame = &v.frames[c * wl as usize + offset as usize..(c + 1) * wl as usize];
            for k in 0..n {
                self.ola_upper[(base + k) * ch + c] += frame[k];
            }
        }
    }

    fn ensure_capacity(&mut self, absolute_stop: i64) {
        let required = (absolute_stop - self.ola_base) as usize;
        if required <= self.weight.len() {
            return;
        }
        let growth = (required - self.weight.len()).max(self.reach).max(4096);
        let len = self.weight.len() + growth;
        self.weight.resize(len, 0.0);
        self.ola.resize(len * self.ch, 0.0);
        self.correction.resize(len * self.ch, 0.0);
        self.ola_upper.resize(len * self.ch, 0.0);
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
                out[k * ch + c] = v + self.ola_upper[i] + self.correction[i];
            }
        }
        self.output_read += count as i64;
        out
    }

    fn compact_output(&mut self) {
        // the attacks' frames read the output from the frame being read on (``restore_attack``)
        let keep = self.attack_lead;
        let consumed = self.output_read - keep - self.ola_base;
        if consumed < Self::COMPACT_AFTER {
            return;
        }
        let c = consumed as usize;
        self.weight.drain(..c);
        self.ola.drain(..c * self.ch);
        self.correction.drain(..c * self.ch);
        self.ola_upper.drain(..c * self.ch);
        self.ola_base += consumed;
        // the low band's, up to what the interpolation still reads
        if let Some(light) = &mut self.light {
            let d = light.split.d as i64;
            let keep_from = (light.up_done + light.split.half).div_euclid(d) - light.split.taps_per_phase as i64 - 2;
            let drop = (keep_from - light.low_base).clamp(0, light.weight_low.len() as i64) as usize;
            if drop > 0 {
                light.weight_low.drain(..drop);
                light.ola_low.drain(..drop * self.ch);
                light.low_base += drop as i64;
            }
        }
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

    /// The attack, by the spectrum: STFT frames (``AttackSpectra``) of the output about the attack's destination,
    /// beside the source's.
    /// - From the attack (the frames over its onset) on, the source's energy the output lacks, bin by bin, is added:
    ///   the source's frame about its attack (at its own speed, its onset at the destination) times
    ///   sqrt(1 - |Y|^2 / |X|^2) (none where the output is as loud). A held note's bins are as loud in both: they
    ///   keep their stretched phase (a crossfade to the source's waveform, unrelated, cancelled them at every hit:
    ///   the held notes trembled at the beat's rate); the hit's own energy is the source's, sharp, on the time map.
    /// - Before it, over the vocoders' smear (half the longer window), each bin louder than the source's there is
    ///   brought down to it: the source's frame where that music is (the time map), at its own speed, both frames
    ///   cut 1 ms before the attack (a frame over the attack still gates what precedes it, to the millisecond), the
    ///   cut kept before it in time too (a bin's gain spreads over its frame). A gate over the whole band ducked the
    ///   held notes with the smear.
    /// - The bins' powers compared averaged over ``Config::attack_smoothing_hz``: a noise's single bins are as
    ///   random as its phase, and compared one by one, half a room's or a reverb's bins were cut before each hit.
    /// - Below the bass cutoff nothing (the bass keeps its phase); below ``Config::attack_whole_hz`` the added attack by
    ///   ``amount_low`` (how sure the event is a dry or a kick attack: a sung syllable keeps its stretched voice),
    ///   above it whole. The gains of all the frames are taken before any is added.
    fn restore_attack(&mut self, light: &mut Light, source: &[f32], at: i64, destination: i64, amount_low: f64) {
        let ch = self.ch;
        let rate = light.rate;
        let factor = self.factor.unwrap_or(1.0);
        let cfg = self.cfg;
        let sr = &mut light.attack;
        let (n, hop) = (sr.n, sr.hop);
        let bins = n / 2 + 1;
        let half = (n / 2) as i64;
        let ms = |v: f64| 1.max((v * rate / 1000.0).round_ties_even() as i64);
        let ramp = |f: f64, at: f64, width: f64| ((f - (at - 0.5 * width)) / width).clamp(0.0, 1.0);
        let bass: Vec<f32> = (0..bins).map(|k| ramp(k as f64 * rate / n as f64, cfg.attack_bass_hz, 100.0) as f32).collect();
        let band: Vec<f32> = (0..bins)
            .map(|k| {
                let high = ramp(k as f64 * rate / n as f64, cfg.attack_whole_hz, cfg.crossover_width_hz);
                bass[k] * (amount_low + (1.0 - amount_low) * high) as f32
            })
            .collect();
        let spread = (cfg.attack_smoothing_hz / (rate / n as f64)).round() as usize;
        let smooth = |v: &[f32]| -> Vec<f32> {
            (0..v.len())
                .map(|m| {
                    let (a, b) = (m.saturating_sub(spread), (m + spread + 1).min(v.len()));
                    v[a..b].iter().sum::<f32>() / (b - a) as f32
                })
                .collect()
        };
        let fade = ms(2.0);
        // 1 up to ``fade`` before ``edge``, a sin^2 fade to 0 at it
        let before = |t: i64, edge: i64| -> f32 {
            if t >= edge {
                0.0
            } else if t + fade <= edge {
                1.0
            } else {
                (0.5 * std::f32::consts::PI * (edge - t) as f32 / fade as f32).sin().powi(2)
            }
        };
        let boundary = destination - self.attack_lead;
        let src_boundary = at - self.attack_lead;
        let add_from = destination - half - ms(1.0);
        let lo = self.output_read.max(self.ola_base);
        let frames_src = (source.len() / ch) as i64;
        let src = |i: i64, k: usize| if i >= 0 && i < frames_src { source[i as usize * ch + k] } else { 0.0 };
        // per frame: its left edge, the attack's gains and the source's spectra, the cut's gains and the output's
        let mut frames: Vec<(i64, Vec<f32>, Vec<Vec<Complex32>>, Vec<f32>, Vec<Vec<Complex32>>)> = Vec::new();
        let mut c = (destination - (self.reach / 2) as i64 - half).min(add_from);
        while c <= destination + ms(cfg.attack_after_ms) {
            let left = c - half;
            c += hop as i64;
            if left < lo || left + n as i64 > self.finalized_until {
                continue;
            }
            let (mut py, mut px) = (vec![0f32; bins], vec![0f32; bins]);
            let (mut py_pre, mut px_pre) = (vec![0f32; bins], vec![0f32; bins]);
            let (mut xs, mut ys_pre) = (Vec::with_capacity(ch), Vec::with_capacity(ch));
            let keep = (boundary - left).clamp(0, n as i64) as usize;
            for k in 0..ch {
                let output = |t: i64| -> f32 {
                    let i = (t - self.ola_base) as usize;
                    let w = self.weight[i];
                    let low = if w > 1e-10 { self.ola[i * ch + k] / w } else { 0.0 };
                    low + self.ola_upper[i * ch + k] + self.correction[i * ch + k]
                };
                // the output, and the source about its attack at its own speed
                for (j, v) in sr.input.iter_mut().enumerate() {
                    *v = output(left + j as i64) * sr.window[j];
                }
                sr.fft.process_with_scratch(&mut sr.input, &mut sr.y, &mut sr.scratch).ok();
                for (j, v) in sr.input.iter_mut().enumerate() {
                    *v = src(at + left + j as i64 - destination, k) * sr.window[j];
                }
                sr.fft.process_with_scratch(&mut sr.input, &mut sr.x[k], &mut sr.scratch).ok();
                for m in 0..bins {
                    py[m] += sr.y[m].norm_sqr();
                    px[m] += sr.x[k][m].norm_sqr();
                }
                xs.push(sr.x[k].clone());
                if keep == 0 {
                    continue;
                }
                // before the attack: the output's frame cut there, the source's where that music is, cut where its
                // attack begins, each by its window's energy (the two kept spans differ)
                let mut ey = 0f32;
                for (j, v) in sr.input.iter_mut().enumerate() {
                    let t = left + j as i64;
                    let a = if keep == n { 1.0 } else { before(t, boundary) } * sr.window[j];
                    ey += a * a;
                    *v = a * output(t);
                }
                let mut yp = vec![Complex32::new(0.0, 0.0); bins];
                sr.fft.process_with_scratch(&mut sr.input, &mut yp, &mut sr.scratch).ok();
                let mapped = at - ((destination - (left + half)) as f64 * factor).round() as i64;
                let mut ex = 0f32;
                for (j, v) in sr.input.iter_mut().enumerate() {
                    let i = mapped + j as i64 - half;
                    let a = before(i, src_boundary) * sr.window[j];
                    ex += a * a;
                    *v = a * src(i, k);
                }
                let mut xp = vec![Complex32::new(0.0, 0.0); bins];
                sr.fft.process_with_scratch(&mut sr.input, &mut xp, &mut sr.scratch).ok();
                if ey > 0.0 && ex > 0.0 {
                    for m in 0..bins {
                        py_pre[m] += yp[m].norm_sqr() / ey;
                        px_pre[m] += xp[m].norm_sqr() / ex;
                    }
                }
                ys_pre.push(yp);
            }
            let (py, px, py_pre, px_pre) = (smooth(&py), smooth(&px), smooth(&py_pre), smooth(&px_pre));
            let adds = left + half >= add_from;
            let g: Vec<f32> = (0..bins)
                .map(|m| if adds && px[m] > 0.0 { (1.0 - py[m] / px[m]).max(0.0).sqrt() * band[m] } else { 0.0 })
                .collect();
            let cut: Vec<f32> = (0..bins)
                .map(|m| {
                    if keep > 0 && py_pre[m] > px_pre[m] {
                        ((px_pre[m] / py_pre[m]).sqrt() - 1.0) * bass[m]
                    } else {
                        0.0
                    }
                })
                .collect();
            frames.push((left, g, xs, cut, ys_pre));
        }
        let Some(first) = frames.first().map(|f| f.0) else { return };
        let last = frames.last().unwrap().0 + n as i64;
        let mut added = vec![0f32; (last - first) as usize * ch];
        // the inverse unscaled (1 / n), Hann^2 at a quarter overlap sums to 1.5
        let scale = 1.0 / (n as f32 * 1.5);
        let mut spec = vec![Complex32::new(0.0, 0.0); bins];
        for (left, g, xs, cut, ys_pre) in &frames {
            let base = (left - first) as usize;
            for k in 0..ch {
                for m in 0..bins {
                    spec[m] = xs[k][m] * g[m];
                }
                spec[0].im = 0.0;
                spec[bins - 1].im = 0.0;
                sr.ifft.process_with_scratch(&mut spec, &mut sr.out, &mut sr.scratch).ok();
                for j in 0..n {
                    added[(base + j) * ch + k] += sr.out[j] * sr.window[j] * scale;
                }
                if ys_pre.is_empty() || cut.iter().all(|v| *v == 0.0) {
                    continue;
                }
                for m in 0..bins {
                    spec[m] = ys_pre[k][m] * cut[m];
                }
                spec[0].im = 0.0;
                spec[bins - 1].im = 0.0;
                sr.ifft.process_with_scratch(&mut spec, &mut sr.out, &mut sr.scratch).ok();
                for j in 0..n {
                    added[(base + j) * ch + k] += sr.out[j] * sr.window[j] * scale * before(left + j as i64, boundary);
                }
            }
        }
        let start = (first - self.ola_base) as usize;
        for (j, v) in added.chunks(ch).enumerate() {
            for k in 0..ch {
                self.correction[(start + j) * ch + k] += v[k];
            }
        }
    }

    fn apply_due_events(&mut self, source: &[f32]) {
        while let Some(&(destination, at, event)) = self.pending_events.front() {
            if destination + self.attack_after > self.finalized_until {
                return;
            }
            self.pending_events.pop_front();
            let cfg = self.cfg;
            let span = cfg.dry_transient_full_strength - cfg.dry_transient_threshold;
            let dry = ((event.strength - cfg.dry_transient_threshold) / span).clamp(0.0, 1.0);
            let kick = ((event.kick - Self::KICK_CONFIDENCE_FROM) / Self::KICK_CONFIDENCE_SPAN).clamp(0.0, 1.0);
            let mut light = self.light.take().expect("the stretcher's bands");
            self.restore_attack(&mut light, source, at, destination, dry.max(kick));
            self.light = Some(light);
        }
    }
}

impl Light {
    /// A new stream: nothing of the last one's frames kept.
    fn restart(&mut self) {
        self.previous_center = None;
        self.upper.previous_center = None;
        self.split.reset();
        self.ola_low.clear();
        self.weight_low.clear();
        self.low_base = 0;
        self.up_done = 0;
        let low_bin_hz = self.rate / self.split.d as f64 / (2 * (self.low_gain.len() - 1)) as f64;
        self.set_crossover(self.cross_default_hz, low_bin_hz);
    }

    /// The crossover at ``hz``: the low band's gain (its bins ``low_bin_hz`` apart) and the upper band's moved.
    fn set_crossover(&mut self, hz: f64, low_bin_hz: f64) {
        if hz == self.cross_hz {
            return;
        }
        let w = self.cross_width_hz;
        move_crossover(&mut self.low_gain, low_bin_hz, self.cross_hz, hz, w, false);
        let upper_bin_hz = self.rate / self.upper.n_fft as f64;
        move_crossover(&mut self.upper.gain, upper_bin_hz, self.cross_hz, hz, w, true);
        self.cross_hz = hz;
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_even_is_round_ties_even() {
        let mut x = 0x2545_F491_4F6C_DD1Du64;
        let mut values = vec![0.0, -0.0, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 1e-300, 2.0f64.powi(51) - 0.5, 2.0f64.powi(51),
                              2.0f64.powi(51) + 0.5, -(2.0f64.powi(52) - 0.5), 1e300, f64::INFINITY];
        for _ in 0..100_000 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let u = (x >> 11) as f64 / (1u64 << 53) as f64;
            values.push((u - 0.5) * 2.0f64.powi((x % 60) as i32));
        }
        // equal (a zero's sign may differ: -0.4 gives +0, which ``wrap`` subtracts alike)
        for v in values {
            assert_eq!(round_even(v), v.round_ties_even(), "{v}");
        }
    }
}
