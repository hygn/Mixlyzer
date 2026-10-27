//! The master's spectrum for the GUI (Settings > View's waveform bands): the master bus after its level, as mono,
//! tapped by the main output's callback (`Tap`: a lock-free ring of the newest samples, written only while the GUI
//! asks, op::SET_SPECTRUM), and its power spectral density made on the engine thread about FRAMES_PER_SEC times a
//! second (`Analyzer`: a Hann-windowed FFT of the last WINDOW_SEC, one-sided, per Hz), averaged into
//! ipc::SPECTRUM_BINS log-spaced bins and sent as dB (ipc::Control::put_spectrum).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

use num_complex::Complex32;
use realfft::{RealFftPlanner, RealToComplex};

use crate::ipc::{SPECTRUM_BINS, SPECTRUM_BINS_PER_OCTAVE, SPECTRUM_DB_FLOOR, SPECTRUM_DB_STEP, SPECTRUM_F_LO};

/// The tap's ring: the newest samples (a power of two, more than the longest window).
const TAP_LEN: usize = 1 << 14;
pub const FRAMES_PER_SEC: f64 = 30.0;
/// The FFT's window: 85 ms (4096 samples at 48 kHz), ~12 Hz apart.
const WINDOW_SEC: f64 = 0.085;

/// The master's newest samples, mono: the callback writes, the engine thread reads the last ones.
pub struct Tap {
    on: AtomicBool,
    samples: Box<[AtomicU32]>, // f32 bits
    written: AtomicU64,
}

impl Default for Tap {
    fn default() -> Self {
        Tap {
            on: AtomicBool::new(false),
            samples: (0..TAP_LEN).map(|_| AtomicU32::new(0)).collect(),
            written: AtomicU64::new(0),
        }
    }
}

impl Tap {
    pub fn set_on(&self, on: bool) {
        self.on.store(on, Ordering::Relaxed);
    }

    pub fn is_on(&self) -> bool {
        self.on.load(Ordering::Relaxed)
    }

    /// The callback's: ``n`` stereo frames of the master (nothing while off).
    pub fn push(&self, stereo: &[f32], n: usize) {
        if !self.is_on() {
            return;
        }
        let w = self.written.load(Ordering::Relaxed);
        for f in 0..n {
            let v = 0.5 * (stereo[2 * f] + stereo[2 * f + 1]);
            self.samples[(w as usize + f) % TAP_LEN].store(v.to_bits(), Ordering::Relaxed);
        }
        self.written.store(w + n as u64, Ordering::Release);
    }

    /// The last ``out.len()`` samples (zeros before the first) and the count written so far.
    fn last(&self, out: &mut [f32]) -> u64 {
        let w = self.written.load(Ordering::Acquire);
        let n = out.len() as u64;
        for (i, o) in out.iter_mut().enumerate() {
            let k = w as i64 - n as i64 + i as i64;
            *o = if k < 0 { 0.0 } else { f32::from_bits(self.samples[k as usize % TAP_LEN].load(Ordering::Relaxed)) };
        }
        w
    }
}

/// The spectrum made from the tap on the engine thread.
pub struct Analyzer {
    fft: Arc<dyn RealToComplex<f32>>,
    window: Vec<f32>,
    input: Vec<f32>,
    output: Vec<Complex32>,
    scratch: Vec<Complex32>,
    density: Vec<f64>,
    /// Per log bin: the FFT bins it averages ([first, last)), or the one holding its centre when it is narrower.
    bins: Vec<(usize, usize)>,
    scale: f64, // |X|^2 -> one-sided power per Hz
    hop: u64,
    last: u64,
}

impl Analyzer {
    pub fn new(rate: f64) -> Analyzer {
        let n = ((rate * WINDOW_SEC) as usize).next_power_of_two();
        let fft = RealFftPlanner::<f32>::new().plan_fft_forward(n);
        let window: Vec<f32> = (0..n)
            .map(|i| (0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / n as f64).cos()) as f32)
            .collect();
        let power: f64 = window.iter().map(|&w| (w as f64) * (w as f64)).sum();
        let spacing = rate / n as f64;
        let top = n / 2;
        let bins = (0..SPECTRUM_BINS)
            .map(|b| {
                let edge = |k: usize| SPECTRUM_F_LO * 2f64.powf(k as f64 / SPECTRUM_BINS_PER_OCTAVE as f64);
                let (lo, hi) = (edge(b), edge(b + 1));
                let first = ((lo / spacing).ceil() as usize).min(top);
                let last = ((hi / spacing).ceil() as usize).min(top + 1);
                if last > first {
                    (first, last)
                } else {
                    let centre = (((lo * hi).sqrt() / spacing).round() as usize).min(top);
                    (centre, centre + 1)
                }
            })
            .collect();
        Analyzer {
            input: vec![0.0; n],
            output: fft.make_output_vec(),
            scratch: fft.make_scratch_vec(),
            density: vec![0.0; top + 1],
            fft,
            window,
            bins,
            scale: 2.0 / (rate * power),
            hop: (rate / FRAMES_PER_SEC).round().max(1.0) as u64,
            last: 0,
        }
    }

    /// A frame when a hop's samples came since the last one (the tap on), else None.
    pub fn tick(&mut self, tap: &Tap) -> Option<[u8; SPECTRUM_BINS]> {
        if !tap.is_on() {
            return None;
        }
        let written = tap.written.load(Ordering::Acquire);
        if written < self.last + self.hop && written >= self.last {
            return None;
        }
        // a hop on from the last frame (its blocks' sizes do not slow the frame rate); far behind (off a while), now
        let latest = tap.last(&mut self.input);
        self.last = if latest > self.last + 2 * self.hop || latest < self.last { latest } else { self.last + self.hop };
        Some(self.frame())
    }

    /// The spectrum of ``self.input``, as sent.
    fn frame(&mut self) -> [u8; SPECTRUM_BINS] {
        for (x, w) in self.input.iter_mut().zip(&self.window) {
            *x *= w;
        }
        self.fft.process_with_scratch(&mut self.input, &mut self.output, &mut self.scratch).ok();
        for (d, c) in self.density.iter_mut().zip(&self.output) {
            *d = c.norm_sqr() as f64 * self.scale;
        }
        let mut out = [0u8; SPECTRUM_BINS];
        for (o, &(first, last)) in out.iter_mut().zip(&self.bins) {
            let mean = self.density[first..last].iter().sum::<f64>() / (last - first) as f64;
            let db = 10.0 * (mean + 1e-30).log10();
            *o = ((db - SPECTRUM_DB_FLOOR) / SPECTRUM_DB_STEP).round().clamp(0.0, 255.0) as u8;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bin_of(hz: f64) -> usize {
        ((hz / SPECTRUM_F_LO).log2() * SPECTRUM_BINS_PER_OCTAVE as f64) as usize
    }

    fn db(v: u8) -> f64 {
        SPECTRUM_DB_FLOOR + v as f64 * SPECTRUM_DB_STEP
    }

    fn played(rate: f64, signal: impl Fn(f64) -> f32, seconds: f64) -> [u8; SPECTRUM_BINS] {
        let tap = Tap::default();
        tap.set_on(true);
        let mut a = Analyzer::new(rate);
        let n = (rate * seconds) as usize;
        let stereo: Vec<f32> = (0..n).flat_map(|i| { let v = signal(i as f64 / rate); [v, v] }).collect();
        let mut got = None;
        for block in stereo.chunks(2 * 256) {
            tap.push(block, block.len() / 2);
            if let Some(f) = a.tick(&tap) {
                got = Some(f);
            }
        }
        got.expect("a frame")
    }

    #[test]
    fn a_sine_peaks_in_its_bin_and_the_bins_hold_its_power() {
        let edge = |k: usize| SPECTRUM_F_LO * 2f64.powf(k as f64 / SPECTRUM_BINS_PER_OCTAVE as f64);
        for rate in [44_100.0, 48_000.0, 96_000.0] {
            let f = played(rate, |t| (std::f64::consts::TAU * 1000.0 * t).sin() as f32, 0.5);
            let peak = (0..SPECTRUM_BINS).max_by_key(|&b| f[b]).unwrap();
            assert!((peak as i64 - bin_of(1000.0) as i64).abs() <= 1, "{rate}: bin {peak}");
            // the density times each bin's width, summed: a full-scale sine's power (0.5), as the GUI integrates a
            // band (within the u8 steps and the bins narrower than the FFT's down low)
            let power: f64 = (0..SPECTRUM_BINS).map(|b| 10f64.powf(db(f[b]) / 10.0) * (edge(b + 1) - edge(b))).sum();
            assert!((10.0 * (power / 0.5).log10()).abs() < 1.0, "{rate}: {power}");
            // far from it, the window's sidelobes are far down
            assert!(db(f[bin_of(100.0)]) < db(f[peak]) - 60.0);
        }
    }

    #[test]
    fn silence_is_the_floor_and_off_makes_nothing() {
        let f = played(48_000.0, |_| 0.0, 0.2);
        assert!(f.iter().all(|&v| v == 0));
        let tap = Tap::default();
        let mut a = Analyzer::new(48_000.0);
        tap.push(&[0.5; 2 * 4096], 4096);
        assert!(a.tick(&tap).is_none());
        assert_eq!(tap.written.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn a_frame_about_every_hop() {
        let rate = 48_000.0;
        let tap = Tap::default();
        tap.set_on(true);
        let mut a = Analyzer::new(rate);
        let block = vec![0.1f32; 2 * 240];
        let mut frames = 0;
        for _ in 0..200 {   // one second
            tap.push(&block, 240);
            frames += a.tick(&tap).is_some() as usize;
        }
        assert!((frames as f64 - FRAMES_PER_SEC).abs() <= 1.0, "{frames}");
    }
}
