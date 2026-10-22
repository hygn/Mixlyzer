//! The mixer, as a DJ mixer's, in the main device's callback (device.rs): a channel strip per deck (trim, a 3-band
//! isolator EQ, the meter and the pre-fader CUE tap, the channel fader: the deck's volume, deck.rs), the crossfader
//! with its curve and each channel's assign (A / THRU / B), the master, the headphones (CUE, the master mixed in or
//! not, their level), the booth, soft clipping. The players (engine.rs) render the decks' music before all of it,
//! so a fader or a knob is heard after the device's latency alone, not the engine's buffer. A strip at its neutral
//! settings (trim 0 dB, every EQ band 0 dB, the fader at the top, THRU) leaves the samples as they are, bit for bit.
//!
//! The EQ splits the signal with Linkwitz-Riley 4th-order crossovers (two Butterworth biquads in a row; low and high
//! sum to a 2nd-order allpass), at LOW_HZ and HIGH_HZ (Mixxx's isolator's): low = LP(f1), the rest = HP(f1), mid =
//! LP(f2) of the rest, high = HP(f2) of the rest; the low band goes through the f2 allpass too, so the three bands
//! at 0 dB sum to an allpass (flat). Each band from a kill (silence) to +6 dB; gains move smoothly (no zipper).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicU64, Ordering};

pub const LOW_HZ: f64 = 246.0;
pub const HIGH_HZ: f64 = 2484.0;
pub const EQ_MAX_DB: f64 = 6.0; // a band's knob at the top
pub const EQ_MIN_DB: f64 = -26.0; // just above its bottom (the bottom: a kill)
pub const TRIM_MAX_DB: f64 = 9.0; // the trim knob's top, as a DJM's
pub const TRIM_MIN_DB: f64 = -24.0;
const SMOOTH_SEC: f64 = 0.01; // a gain change eased over this
const Q_BUTTERWORTH: f64 = std::f64::consts::FRAC_1_SQRT_2;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Assign {
    A = 0,
    Thru = 1,
    B = 2,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Curve {
    Smooth, // constant power: the two sides cross at -3 dB (blends)
    Sharp,  // both sides full but for the edges (cuts, scratching)
}

/// A crossfader position (0 the A side, 1 the B side) as a channel's gain.
pub fn crossfader_gain(position: f64, assign: Assign, curve: Curve) -> f64 {
    let x = position.clamp(0.0, 1.0);
    let toward = match assign {
        Assign::Thru => return 1.0,
        Assign::A => 1.0 - x, // how far the fader is toward this side's end
        Assign::B => x,
    };
    match curve {
        Curve::Smooth => (toward * std::f64::consts::FRAC_PI_2).sin(),
        Curve::Sharp => (toward / 0.05).min(1.0), // full until the last 5 % toward the other side
    }
}

/// A knob's position (0 - 1, the middle 0 dB) as an EQ band's gain: the bottom a kill, else EQ_MIN_DB .. 0 ..
/// EQ_MAX_DB.
pub fn eq_knob_gain(knob: f64) -> f64 {
    knob_gain(knob, EQ_MIN_DB, EQ_MAX_DB, true)
}

/// The trim knob's gain: TRIM_MIN_DB .. 0 (the middle) .. TRIM_MAX_DB.
pub fn trim_knob_gain(knob: f64) -> f64 {
    knob_gain(knob, TRIM_MIN_DB, TRIM_MAX_DB, false)
}

fn knob_gain(knob: f64, min_db: f64, max_db: f64, kill: bool) -> f64 {
    let k = if knob.is_finite() { knob.clamp(0.0, 1.0) } else { 0.5 };
    if kill && k <= 0.0 {
        return 0.0;
    }
    let db = if k < 0.5 { min_db * (1.0 - 2.0 * k) } else { max_db * (2.0 * k - 1.0) };
    10f64.powf(db / 20.0)
}

/// A biquad (RBJ's cookbook), transposed direct form II, one state per channel.
#[derive(Clone)]
struct Biquad {
    b: [f64; 3],
    a: [f64; 2],
    z: Vec<[f64; 2]>,
}

impl Biquad {
    fn new(kind: u8, f: f64, rate: f64, ch: usize) -> Biquad {
        let w = 2.0 * std::f64::consts::PI * f / rate;
        let (sin, cos) = w.sin_cos();
        let alpha = sin / (2.0 * Q_BUTTERWORTH);
        let (b0, b1, b2) = match kind {
            0 => ((1.0 - cos) / 2.0, 1.0 - cos, (1.0 - cos) / 2.0), // low-pass
            1 => ((1.0 + cos) / 2.0, -(1.0 + cos), (1.0 + cos) / 2.0), // high-pass
            _ => (1.0 - alpha, -2.0 * cos, 1.0 + alpha),             // allpass
        };
        let a0 = 1.0 + alpha;
        Biquad { b: [b0 / a0, b1 / a0, b2 / a0], a: [-2.0 * cos / a0, (1.0 - alpha) / a0], z: vec![[0.0; 2]; ch] }
    }

    #[inline]
    fn run(&mut self, c: usize, x: f64) -> f64 {
        let z = &mut self.z[c];
        let y = self.b[0] * x + z[0];
        z[0] = self.b[1] * x - self.a[0] * y + z[1];
        z[1] = self.b[2] * x - self.a[1] * y;
        y
    }

    fn reset(&mut self) {
        self.z.iter_mut().for_each(|z| *z = [0.0; 2]);
    }
}

/// Linkwitz-Riley 4th order: a Butterworth biquad twice.
#[derive(Clone)]
struct Lr4 {
    first: Biquad,
    second: Biquad,
}

impl Lr4 {
    fn new(kind: u8, f: f64, rate: f64, ch: usize) -> Lr4 {
        Lr4 { first: Biquad::new(kind, f, rate, ch), second: Biquad::new(kind, f, rate, ch) }
    }

    #[inline]
    fn run(&mut self, c: usize, x: f64) -> f64 {
        let y = self.first.run(c, x);
        self.second.run(c, y)
    }

    fn reset(&mut self) {
        self.first.reset();
        self.second.reset();
    }
}

/// The 3-band isolator.
pub struct Eq3 {
    low_lp: Lr4,
    rest_hp: Lr4,
    mid_lp: Lr4,
    high_hp: Lr4,
    low_ap: Biquad,
    gains: [f64; 3],  // wanted (linear): low, mid, high
    smooth: [f64; 3], // as applied now
    pole: f64,
    ch: usize,
    active: bool,
}

impl Eq3 {
    pub fn new(rate: f64, ch: usize) -> Eq3 {
        Eq3 {
            low_lp: Lr4::new(0, LOW_HZ, rate, ch),
            rest_hp: Lr4::new(1, LOW_HZ, rate, ch),
            mid_lp: Lr4::new(0, HIGH_HZ, rate, ch),
            high_hp: Lr4::new(1, HIGH_HZ, rate, ch),
            low_ap: Biquad::new(2, HIGH_HZ, rate, ch),
            gains: [1.0; 3],
            smooth: [1.0; 3],
            pole: (-1.0 / (SMOOTH_SEC * rate)).exp(),
            ch,
            active: false,
        }
    }

    pub fn set_gains(&mut self, low: f64, mid: f64, high: f64) {
        self.gains = [low, mid, high].map(|g| if g.is_finite() { g.max(0.0) } else { 1.0 });
    }

    /// Every band at 0 dB, and settled there: the signal passes untouched.
    pub fn neutral(&self) -> bool {
        self.gains == [1.0; 3] && self.smooth.iter().all(|&g| (g - 1.0).abs() < 1e-6)
    }

    /// ``out`` (interleaved) through the bands; neutral, untouched (the filters' state reset, so the next use starts
    /// clean).
    pub fn process(&mut self, out: &mut [f32]) {
        if self.neutral() {
            if self.active {
                self.active = false;
                self.smooth = [1.0; 3];
                for f in [&mut self.low_lp, &mut self.rest_hp, &mut self.mid_lp, &mut self.high_hp] {
                    f.reset();
                }
                self.low_ap.reset();
            }
            return;
        }
        self.active = true;
        let ch = self.ch;
        for frame in out.chunks_exact_mut(ch) {
            for k in 0..3 {
                self.smooth[k] = self.gains[k] + self.pole * (self.smooth[k] - self.gains[k]);
            }
            for (c, s) in frame.iter_mut().enumerate() {
                let x = *s as f64;
                let low = self.low_ap.run(c, self.low_lp.run(c, x));
                let rest = self.rest_hp.run(c, x);
                let mid = self.mid_lp.run(c, rest);
                let high = self.high_hp.run(c, rest);
                *s = (low * self.smooth[0] + mid * self.smooth[1] + high * self.smooth[2]) as f32;
            }
        }
    }
}

/// A channel strip's settings, as its deck (deck.rs: its buttons, keys and a controller's knobs) sets them and the
/// main device's callback reads them: trim, EQ bands, the channel fader's gain (the deck's volume), its headphones
/// CUE and crossfader assign; and its meter, which the callback writes (the peak before the fader since it was last
/// taken).
pub struct StripParams {
    trim: AtomicU64,    // f64 bits, linear
    eq: [AtomicU64; 3], // f64 bits, linear: low, mid, high
    fader: AtomicU64,   // f64 bits, linear
    cue: AtomicBool,
    assign: AtomicU8,
    peak: AtomicU32, // f32 bits (>= 0: ordered as integers)
}

impl StripParams {
    fn new() -> StripParams {
        StripParams {
            trim: AtomicU64::new(1f64.to_bits()),
            eq: std::array::from_fn(|_| AtomicU64::new(1f64.to_bits())),
            fader: AtomicU64::new(1f64.to_bits()),
            cue: AtomicBool::new(false),
            assign: AtomicU8::new(Assign::Thru as u8),
            peak: AtomicU32::new(0),
        }
    }

    fn gain(g: f64) -> u64 {
        (if g.is_finite() { g.max(0.0) } else { 1.0 }).to_bits()
    }

    pub fn set_trim(&self, g: f64) {
        self.trim.store(Self::gain(g), Ordering::Relaxed);
    }

    pub fn set_eq(&self, gains: [f64; 3]) {
        for (a, g) in self.eq.iter().zip(gains) {
            a.store(Self::gain(g), Ordering::Relaxed);
        }
    }

    pub fn set_fader(&self, g: f64) {
        self.fader.store(Self::gain(g), Ordering::Relaxed);
    }

    pub fn set_cue(&self, on: bool) {
        self.cue.store(on, Ordering::Relaxed);
    }

    pub fn toggle_cue(&self) {
        self.cue.fetch_xor(true, Ordering::Relaxed);
    }

    pub fn cue_on(&self) -> bool {
        self.cue.load(Ordering::Relaxed)
    }

    pub fn set_assign(&self, assign: Assign) {
        self.assign.store(assign as u8, Ordering::Relaxed);
    }

    fn assign(&self) -> Assign {
        match self.assign.load(Ordering::Relaxed) {
            a if a == Assign::A as u8 => Assign::A,
            a if a == Assign::B as u8 => Assign::B,
            _ => Assign::Thru,
        }
    }

    fn load(a: &AtomicU64) -> f64 {
        f64::from_bits(a.load(Ordering::Relaxed))
    }

    /// The peak before the fader (linear) since the last call.
    pub fn take_peak(&self) -> f32 {
        f32::from_bits(self.peak.swap(0, Ordering::Relaxed))
    }

    fn add_peak(&self, peak: f32) {
        if peak.is_finite() && peak > 0.0 {
            self.peak.fetch_max(peak.to_bits(), Ordering::Relaxed);
        }
    }
}

/// A channel strip's processing, in the main device's callback: the player's music through trim and EQ, its meter
/// and its pre-fader tap (the headphones' CUE), then the channel fader; the player's listening aids (the metronome,
/// reference tones) added after the fader, as they were (their level their own). At its neutral settings (trim 0 dB,
/// EQ flat, the fader at the top) the music passes bit for bit.
pub struct Strip {
    eq: Eq3,
    trim: f32,  // as applied (a change moved to over a block)
    fader: f32, // as applied
}

impl Strip {
    pub fn new(rate: f64, ch: usize) -> Strip {
        Strip { eq: Eq3::new(rate, ch), trim: 1.0, fader: 1.0 }
    }

    /// ``lane``: ``n`` frames of the player's [music (``ch``), aids (``ch``)]: the music before the fader into ``pfl``,
    /// the strip's output into ``post`` (``ch`` a frame each).
    pub fn process(&mut self, lane: &[f32], n: usize, ch: usize, p: &StripParams, pfl: &mut [f32], post: &mut [f32]) {
        let lch = 2 * ch;
        for f in 0..n {
            pfl[f * ch..(f + 1) * ch].copy_from_slice(&lane[f * lch..f * lch + ch]);
        }
        let pfl = &mut pfl[..n * ch];
        let trim = StripParams::load(&p.trim) as f32;
        if trim != 1.0 || self.trim != 1.0 {
            ramp(pfl, ch, self.trim, trim);
            self.trim = trim;
        }
        let [low, mid, high] = p.eq.each_ref().map(StripParams::load);
        self.eq.set_gains(low, mid, high);
        self.eq.process(pfl);
        p.add_peak(pfl.iter().fold(0f32, |m, v| m.max(v.abs())));
        let fader = StripParams::load(&p.fader) as f32;
        let step = (fader - self.fader) / n.max(1) as f32;
        for f in 0..n {
            let g = if step == 0.0 { fader } else { self.fader + step * (f + 1) as f32 };
            for c in 0..ch {
                let v = pfl[f * ch + c];
                post[f * ch + c] = (if g == 1.0 { v } else { v * g }) + lane[f * lch + ch + c];
            }
        }
        self.fader = fader;
    }
}

/// ``x`` (``ch`` a frame) scaled from ``from`` to ``to`` over its frames.
fn ramp(x: &mut [f32], ch: usize, from: f32, to: f32) {
    let n = (x.len() / ch).max(1);
    let step = (to - from) / n as f32;
    for (f, frame) in x.chunks_exact_mut(ch).enumerate() {
        let g = from + step * (f + 1) as f32;
        frame.iter_mut().for_each(|v| *v *= g);
    }
}

/// No output channel (a bus not routed).
pub const NO_CHANNEL: u8 = 0xFF;

/// The mixer's settings, changed from the engine thread and read once per callback: each channel's strip
/// (StripParams), the crossfader; the headphones: the CUE'd channels, the master mixed in by the CUE / MASTER mix or
/// not (a controller that mixes the master into its headphones itself, as the DDJ-400 does in analog), their level;
/// soft clipping of the outputs; where each bus goes on the main device: MAIN (the master), SUB (the booth: the master
/// again), CUE (the headphones), as a pair of its channels (NO_CHANNEL: not there).
pub struct MixParams {
    pub strips: [Arc<StripParams>; crate::ipc::MAX_DECKS],
    crossfader: AtomicU64, // f64 bits, 0 the A side .. 1 the B side
    curve: AtomicU8,       // 0 smooth, 1 sharp
    cue_mix: AtomicU64,    // f64 bits, 0 CUE only .. 1 MASTER only
    cue_level: AtomicU64,  // f64 bits, linear
    master_cue: AtomicBool, // the master mixed into the headphones
    soft_clip: AtomicBool,
    routes: [AtomicU8; 6], // main L / R, cue L / R, sub L / R
}

/// One reading of MixParams (the callback's).
#[derive(Clone, Copy)]
pub struct Mix {
    pub gains: [f32; crate::ipc::MAX_DECKS], // each channel's crossfader gain
    pub cue: [bool; crate::ipc::MAX_DECKS],
    pub cue_mix: f32,
    pub cue_level: f32,
    pub master_cue: bool,
    pub soft_clip: bool,
    pub routes: [u8; 6],
}

impl MixParams {
    pub fn new(out_channels: usize) -> MixParams {
        let p = MixParams {
            strips: std::array::from_fn(|_| Arc::new(StripParams::new())),
            crossfader: AtomicU64::new(0.5f64.to_bits()),
            curve: AtomicU8::new(0),
            cue_mix: AtomicU64::new(0.5f64.to_bits()),
            cue_level: AtomicU64::new(1.0f64.to_bits()),
            master_cue: AtomicBool::new(true),
            soft_clip: AtomicBool::new(false),
            routes: std::array::from_fn(|_| AtomicU8::new(NO_CHANNEL)),
        };
        // the master on the device's first two channels (one: its only one), nothing else routed
        p.set_routes([0, if out_channels > 1 { 1 } else { NO_CHANNEL }, NO_CHANNEL, NO_CHANNEL, NO_CHANNEL, NO_CHANNEL]);
        p
    }

    pub fn set_assign(&self, deck: usize, assign: Assign) {
        if let Some(s) = self.strips.get(deck) {
            s.set_assign(assign);
        }
    }

    pub fn set_cue(&self, deck: usize, on: bool) {
        if let Some(s) = self.strips.get(deck) {
            s.set_cue(on);
        }
    }

    pub fn toggle_cue(&self, deck: usize) {
        if let Some(s) = self.strips.get(deck) {
            s.toggle_cue();
        }
    }

    pub fn cue_on(&self, deck: usize) -> bool {
        self.strips.get(deck).is_some_and(|s| s.cue_on())
    }

    pub fn set_crossfader(&self, position: f64) {
        let x = if position.is_finite() { position.clamp(0.0, 1.0) } else { 0.5 };
        self.crossfader.store(x.to_bits(), Ordering::Relaxed);
    }

    pub fn set_curve(&self, curve: Curve) {
        self.curve.store((curve == Curve::Sharp) as u8, Ordering::Relaxed);
    }

    pub fn set_cue_mix(&self, mix: f64) {
        let m = if mix.is_finite() { mix.clamp(0.0, 1.0) } else { 0.5 };
        self.cue_mix.store(m.to_bits(), Ordering::Relaxed);
    }

    pub fn set_cue_level(&self, level: f64) {
        let l = if level.is_finite() { level.clamp(0.0, 4.0) } else { 1.0 };
        self.cue_level.store(l.to_bits(), Ordering::Relaxed);
    }

    pub fn set_master_cue(&self, on: bool) {
        self.master_cue.store(on, Ordering::Relaxed);
    }

    pub fn set_soft_clip(&self, on: bool) {
        self.soft_clip.store(on, Ordering::Relaxed);
    }

    /// main L / R, cue L / R, sub L / R (NO_CHANNEL: not routed).
    pub fn set_routes(&self, routes: [u8; 6]) {
        for (a, r) in self.routes.iter().zip(routes) {
            a.store(r, Ordering::Relaxed);
        }
    }

    pub fn read(&self) -> Mix {
        let x = f64::from_bits(self.crossfader.load(Ordering::Relaxed));
        let curve = if self.curve.load(Ordering::Relaxed) == 1 { Curve::Sharp } else { Curve::Smooth };
        Mix {
            gains: std::array::from_fn(|k| crossfader_gain(x, self.strips[k].assign(), curve) as f32),
            cue: std::array::from_fn(|k| self.strips[k].cue_on()),
            cue_mix: f64::from_bits(self.cue_mix.load(Ordering::Relaxed)) as f32,
            cue_level: f64::from_bits(self.cue_level.load(Ordering::Relaxed)) as f32,
            master_cue: self.master_cue.load(Ordering::Relaxed),
            soft_clip: self.soft_clip.load(Ordering::Relaxed),
            routes: std::array::from_fn(|k| self.routes[k].load(Ordering::Relaxed)),
        }
    }
}

/// The mixer, in the main device's callback: each channel's strip, the master (the strips through the crossfader),
/// the headphones (the CUE'd channels' pre-fader sound, and the master by the CUE / MASTER mix when it is mixed in,
/// at their level), soft clipping; the outputs as stereo buses (a mono deck's on both sides). The booth is the
/// master. A channel whose lane gave nothing is left out: deck 1 alone at its neutral settings comes out bit for
/// bit as the player rendered it.
pub struct Mixer {
    strips: Vec<Strip>,
    ch: usize,
    pfl: Vec<f32>,
    post: Vec<f32>,
    cue: Vec<f32>,
    pub master: Vec<f32>,     // stereo, a piece's frames
    pub headphones: Vec<f32>, // stereo
}

impl Mixer {
    pub fn new(rate: f64, ch: usize, frames: usize) -> Mixer {
        let ch = ch.clamp(1, 2);
        Mixer {
            strips: (0..crate::ipc::MAX_DECKS).map(|_| Strip::new(rate, ch)).collect(),
            ch,
            pfl: vec![0.0; frames * ch],
            post: vec![0.0; frames * ch],
            cue: vec![0.0; frames * 2],
            master: vec![0.0; frames * 2],
            headphones: vec![0.0; frames * 2],
        }
    }

    /// ``n`` frames (at most the mixer's): ``lanes[k]`` channel k's lane block where it gave frames; the master into
    /// ``master``, and with ``headphones`` the headphones into ``headphones``.
    pub fn process(&mut self, lanes: &[Option<&[f32]>], n: usize, mix: &Mix, params: &MixParams, headphones: bool) {
        let ch = self.ch;
        let mut first = true;
        let mut cueing = false;
        for (k, lane) in lanes.iter().enumerate() {
            let Some(lane) = lane else { continue };
            self.strips[k].process(lane, n, ch, &params.strips[k], &mut self.pfl, &mut self.post);
            let g = mix.gains[k];
            for f in 0..n {
                for side in 0..2 {
                    let v = self.post[f * ch + side.min(ch - 1)];
                    let v = if g == 1.0 { v } else { v * g };
                    let m = &mut self.master[f * 2 + side];
                    *m = if first { v } else { *m + v };
                }
            }
            first = false;
            if headphones && mix.cue[k] {
                for f in 0..n {
                    for side in 0..2 {
                        let v = self.pfl[f * ch + side.min(ch - 1)];
                        let c = &mut self.cue[f * 2 + side];
                        *c = if cueing { *c + v } else { v };
                    }
                }
                cueing = true;
            }
        }
        if first {
            self.master[..n * 2].iter_mut().for_each(|v| *v = 0.0);
        }
        if headphones {
            let (m, level) = (if mix.master_cue { mix.cue_mix } else { 0.0 }, mix.cue_level);
            for i in 0..n * 2 {
                let cue = if cueing { self.cue[i] } else { 0.0 };
                self.headphones[i] = ((1.0 - m) * cue + m * self.master[i]) * level;
            }
            if mix.soft_clip {
                crate::dsp::soft_clip(&mut self.headphones[..n * 2], SOFT_CLIP_KNEE);
            }
        }
        if mix.soft_clip {
            crate::dsp::soft_clip(&mut self.master[..n * 2], SOFT_CLIP_KNEE);
        }
    }
}

/// tanh soft clipping above this level (-1.9 dBFS) up to 0 dBFS.
const SOFT_CLIP_KNEE: f32 = 0.8;

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(f: f64, rate: f64, n: usize, ch: usize) -> Vec<f32> {
        (0..n).flat_map(|i| std::iter::repeat((2.0 * std::f64::consts::PI * f * i as f64 / rate).sin() as f32).take(ch)).collect()
    }

    fn rms(x: &[f32]) -> f64 {
        (x.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / x.len() as f64).sqrt()
    }

    #[test]
    fn neutral_is_untouched() {
        let mut eq = Eq3::new(48000.0, 2);
        let x = sine(440.0, 48000.0, 4800, 2);
        let mut y = x.clone();
        eq.process(&mut y);
        assert_eq!(x, y);
        // a strip at its neutral settings: the music bit for bit, the aids added after the fader
        let mut strip = Strip::new(48000.0, 2);
        let lane: Vec<f32> = x.chunks_exact(2).flat_map(|f| [f[0], f[1], 0.0, 0.0]).collect();
        let (mut pfl, mut post) = (vec![0f32; x.len()], vec![0f32; x.len()]);
        strip.process(&lane, 4800, 2, &StripParams::new(), &mut pfl, &mut post);
        assert_eq!(x, pfl);
        assert_eq!(x, post);
    }

    #[test]
    fn bands_at_0_db_sum_flat() {
        // forced through the filters (a band a hair off 0 dB): every frequency comes out at its level
        for f in [60.0, 246.0, 1000.0, 2484.0, 8000.0] {
            let mut eq = Eq3::new(48000.0, 1);
            eq.set_gains(1.0, 1.0, 1.0 + 1e-9);
            let mut y = sine(f, 48000.0, 48000, 1);
            eq.process(&mut y);
            let level = 20.0 * (rms(&y[24000..]) / rms(&sine(f, 48000.0, 24000, 1))).log10();
            assert!(level.abs() < 0.05, "{f} Hz: {level} dB");
        }
    }

    #[test]
    fn kills_isolate() {
        let mut eq = Eq3::new(48000.0, 1);
        eq.set_gains(0.0, 1.0, 1.0); // low killed
        let mut y = sine(50.0, 48000.0, 48000, 1);
        eq.process(&mut y);
        let level = 20.0 * (rms(&y[24000..]) / rms(&sine(50.0, 48000.0, 24000, 1))).log10();
        assert!(level < -30.0, "50 Hz with the low killed: {level} dB");
        let mut eq = Eq3::new(48000.0, 1);
        eq.set_gains(1.0, 1.0, 0.0); // high killed
        let mut y = sine(10000.0, 48000.0, 48000, 1);
        eq.process(&mut y);
        let level = 20.0 * (rms(&y[24000..]) / rms(&sine(10000.0, 48000.0, 24000, 1))).log10();
        assert!(level < -30.0, "10 kHz with the high killed: {level} dB");
    }

    /// The headphones: silent with no CUE on and the master not mixed in; the CUE'd channel before its fader; the
    /// master by the CUE / MASTER mix when it is mixed in.
    #[test]
    fn headphones_cue() {
        let params = MixParams::new(2);
        let mut mixer = Mixer::new(48000.0, 2, 64);
        let lane: Vec<f32> = (0..64).flat_map(|_| [0.5f32, 0.5, 0.0, 0.0]).collect();
        let lanes: Vec<Option<&[f32]>> = (0..crate::ipc::MAX_DECKS).map(|k| (k == 0).then_some(&lane[..])).collect();
        params.strips[0].set_fader(0.5);
        params.set_master_cue(false);
        let run = |mixer: &mut Mixer| {
            for _ in 0..4 {
                mixer.process(&lanes, 64, &params.read(), &params, true);
            }
            (mixer.master[0], mixer.headphones[0])
        };
        let (master, hp) = run(&mut mixer);
        assert!((master - 0.25).abs() < 1e-6 && hp == 0.0, "{master} {hp}");
        params.toggle_cue(0);
        let (_, hp) = run(&mut mixer);
        assert!((hp - 0.5).abs() < 1e-6, "pre-fader: {hp}");
        params.set_master_cue(true);
        params.set_cue_mix(1.0);
        let (_, hp) = run(&mut mixer);
        assert!((hp - 0.25).abs() < 1e-6, "the master alone: {hp}");
        params.set_cue_mix(0.5);
        let (_, hp) = run(&mut mixer);
        assert!((hp - 0.375).abs() < 1e-6, "half and half: {hp}");
    }

    #[test]
    fn knobs_and_crossfader() {
        assert_eq!(eq_knob_gain(0.0), 0.0);
        assert!((eq_knob_gain(0.5) - 1.0).abs() < 1e-12);
        assert!((20.0 * eq_knob_gain(1.0).log10() - EQ_MAX_DB).abs() < 1e-9);
        assert!((trim_knob_gain(0.5) - 1.0).abs() < 1e-12);
        assert_eq!(crossfader_gain(0.0, Assign::Thru, Curve::Smooth), 1.0);
        assert!((crossfader_gain(0.0, Assign::A, Curve::Smooth) - 1.0).abs() < 1e-12);
        assert!(crossfader_gain(1.0, Assign::A, Curve::Smooth).abs() < 1e-12);
        assert!((crossfader_gain(0.5, Assign::B, Curve::Smooth) - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-12);
        assert_eq!(crossfader_gain(0.9, Assign::A, Curve::Sharp), 1.0);
        assert_eq!(crossfader_gain(1.0, Assign::A, Curve::Sharp), 0.0);
    }
}
