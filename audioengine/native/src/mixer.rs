//! The mixer, as a DJM's channel and its master section, in the main device's callback (device.rs). The deck's
//! channel strip, as the DJM-A9's block diagram (page 93): trim, a 3-band EQ (an isolator down to a kill, or an EQ
//! down to -26 dB: ipc::mix::EQ_MODE), the meter and the pre-fader CUE tap (the headphones), the channel fader (its
//! curve the mixer's), with three places for the effects (colorfx::Slot): after the trim, after the EQ (before the
//! tap), after the fader; Sound Color FX (colorfx.rs: the effect the mixer picks, the channel's COLOR knob) in its
//! effect's place (ColorType::slot). Then the master (its level, its meter), the booth (its level and HI / LOW EQ,
//! -12 .. +6 dB), the headphones (the CUE'd channel before the fader, the master by their CUE / MASTER mix when it is
//! mixed in, or MONO SPLIT: the CUE left, the master right; their level), soft clipping. The player (engine.rs)
//! renders the deck's music before all of it, so a fader or a knob is heard after the device's latency alone, not
//! the engine's buffer. The strip at its neutral settings (trim 0 dB, every EQ band 0 dB, COLOR off, the fader at
//! the top) leaves the samples as they are, bit for bit, and so does the master at 0 dB.
//!
//! SIMD: the buses' sums and gains go through VOLK's kernels (vk.rs: the machine's widest), the filters four lanes
//! at a time (simd::F4 through filters.rs: a stereo pair and a second pair of filters in one pass).
//!
//! The isolator splits the signal with Linkwitz-Riley 4th-order crossovers (two Butterworth biquads in a row; low and
//! high sum to a 2nd-order allpass), at LOW_HZ and HIGH_HZ (Mixxx's isolator's): low = LP(f1), the rest = HP(f1),
//! mid = LP(f2) of the rest, high = HP(f2) of the rest; the low band goes through the f2 allpass too, so the three
//! bands at 0 dB sum to an allpass (flat). The EQ is a low shelf (EQ_LOW_HZ), a peak (EQ_MID_HZ) and a high shelf
//! (EQ_HIGH_HZ), a DJM's. Gains move smoothly (no zipper).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicU64, Ordering};

use crate::colorfx::{ColorFx, ColorType, Slot};
use crate::filters::{Biquad4, Coeffs};
use crate::fxkit::Smooth;
use crate::ipc::mix;
use crate::simd::{F4, peak_abs};
use crate::vk;

pub const LOW_HZ: f64 = 246.0;
pub const HIGH_HZ: f64 = 2484.0;
pub const EQ_LOW_HZ: f64 = 70.0;
pub const EQ_MID_HZ: f64 = 1000.0;
pub const EQ_HIGH_HZ: f64 = 13000.0;
const EQ_MID_Q: f64 = 0.7;
pub const EQ_MAX_DB: f64 = 6.0; // a band's knob at the top
pub const EQ_MIN_DB: f64 = -26.0; // its bottom (the isolator's: just above its kill)
pub const BOOTH_EQ_MIN_DB: f64 = -12.0;
pub const TRIM_MAX_DB: f64 = 9.0; // the trim knob's top, as a DJM's
pub const TRIM_MIN_DB: f64 = -24.0;
const SMOOTH_SEC: f64 = 0.01; // a gain change eased over this
const Q_BUTTERWORTH: f64 = std::f64::consts::FRAC_1_SQRT_2;

/// A channel fader's position (0 .. 1) as its gain, on the mixer's curve (ipc::mix::FADER_CURVE): 0 the GUI's
/// (utils/volume.py: slow at first, full at the top), 1 steep (up fast), 2 slow (a cube: most of the rise at the top).
pub fn fader_gain(position: f64, curve: u8) -> f64 {
    let p = if position.is_finite() { position.clamp(0.0, 1.0) } else { 1.0 };
    match curve {
        1 => 1.0 - (1.0 - p).powi(3),
        2 => p.powi(3),
        _ => {
            let v = p.powf(0.4);
            if v > 0.99 { 1.0 } else { -(1.0 - v).ln() / 100f64.ln() }
        }
    }
}

/// A level knob (master, booth, headphones) as a gain: silence at the bottom, 0 dB at mix::LEVEL_UNITY,
/// about +4 dB at the top.
pub fn level_gain(knob: f64) -> f64 {
    let k = if knob.is_finite() { knob.clamp(0.0, 1.0) } else { mix::LEVEL_UNITY };
    let g = (k / mix::LEVEL_UNITY).powi(2);
    if (g - 1.0).abs() < 1e-9 { 1.0 } else { g }
}

/// The booth EQ's knob in dB: BOOTH_EQ_MIN_DB at the bottom, 0 at the middle, EQ_MAX_DB at the top.
pub fn booth_eq_db(knob: f64) -> f64 {
    let k = if knob.is_finite() { knob.clamp(0.0, 1.0) } else { 0.5 };
    if k < 0.5 { BOOTH_EQ_MIN_DB * (1.0 - 2.0 * k) } else { EQ_MAX_DB * (2.0 * k - 1.0) }
}

/// An EQ knob's band level in dB (the middle 0, the top EQ_MAX_DB, the bottom EQ_MIN_DB); None: a kill (the
/// isolator's bottom).
pub fn eq_knob_db(knob: f64, kill: bool) -> Option<f64> {
    let k = if knob.is_finite() { knob.clamp(0.0, 1.0) } else { 0.5 };
    if kill && k <= 0.0 {
        return None;
    }
    Some(if k < 0.5 { EQ_MIN_DB * (1.0 - 2.0 * k) } else { EQ_MAX_DB * (2.0 * k - 1.0) })
}

/// A knob's position (0 - 1, the middle 0 dB) as an isolator band's gain: the bottom a kill.
pub fn eq_knob_gain(knob: f64) -> f64 {
    eq_knob_db(knob, true).map_or(0.0, |db| 10f64.powf(db / 20.0))
}

/// The trim knob's gain: TRIM_MIN_DB .. 0 (the middle) .. TRIM_MAX_DB.
pub fn trim_knob_gain(knob: f64) -> f64 {
    let k = if knob.is_finite() { knob.clamp(0.0, 1.0) } else { 0.5 };
    let db = if k < 0.5 { TRIM_MIN_DB * (1.0 - 2.0 * k) } else { TRIM_MAX_DB * (2.0 * k - 1.0) };
    10f64.powf(db / 20.0)
}

/// A frame of a strip's interleaved block (1 or 2 channels) as a stereo pair, twice: lanes left, right, left, right.
#[inline(always)]
fn pair2(x: &[f32], f: usize, ch: usize) -> F4 {
    if ch == 1 {
        F4::splat(x[f])
    } else {
        let (l, r) = (x[2 * f], x[2 * f + 1]);
        F4::new(l, r, l, r)
    }
}

/// The isolator, the four Linkwitz-Riley crossovers' stages two at a time (simd lanes: a band's left, right, the
/// other band's left, right).
pub struct Isolator {
    split1: [Biquad4; 2], // low LP | rest HP at LOW_HZ (two stages: LR4)
    split2: [Biquad4; 2], // mid LP | high HP at HIGH_HZ, on the rest
    low_ap: Biquad4,      // the low band through HIGH_HZ's allpass (lanes 0, 1)
    smooth: F4,           // the bands' gains as applied: low, mid, high
    pole: f32,
    active: bool,
}

impl Isolator {
    pub fn new(rate: f64) -> Isolator {
        let lp1 = Coeffs::lowpass(LOW_HZ, Q_BUTTERWORTH, rate);
        let hp1 = Coeffs::highpass(LOW_HZ, Q_BUTTERWORTH, rate);
        let lp2 = Coeffs::lowpass(HIGH_HZ, Q_BUTTERWORTH, rate);
        let hp2 = Coeffs::highpass(HIGH_HZ, Q_BUTTERWORTH, rate);
        let ap = Coeffs::allpass(HIGH_HZ, Q_BUTTERWORTH, rate);
        let s1 = Biquad4::new([lp1, lp1, hp1, hp1]);
        let s2 = Biquad4::new([lp2, lp2, hp2, hp2]);
        Isolator {
            split1: [s1, s1],
            split2: [s2, s2],
            low_ap: Biquad4::new([ap, ap, Coeffs::IDENTITY, Coeffs::IDENTITY]),
            smooth: F4::new(1.0, 1.0, 1.0, 0.0),
            pole: (-1.0 / (SMOOTH_SEC * rate)).exp() as f32,
            active: false,
        }
    }

    fn reset(&mut self) {
        for b in self.split1.iter_mut().chain(self.split2.iter_mut()) {
            b.reset();
        }
        self.low_ap.reset();
    }

    /// ``x``: ``n`` frames of ``ch``; ``gains`` low, mid, high (linear). Every band at 0 dB and settled: untouched
    /// (the state reset for a clean next use).
    pub fn process(&mut self, x: &mut [f32], n: usize, ch: usize, gains: [f64; 3]) {
        let target = F4::new(gains[0] as f32, gains[1] as f32, gains[2] as f32, 0.0);
        let settled = gains == [1.0; 3] && self.smooth.to_array()[..3].iter().all(|&g| (g - 1.0).abs() < 1e-6);
        if settled {
            if self.active {
                self.active = false;
                self.smooth = F4::new(1.0, 1.0, 1.0, 0.0);
                self.reset();
            }
            return;
        }
        self.active = true;
        let pole = F4::splat(self.pole);
        for f in 0..n {
            self.smooth = target + pole * (self.smooth - target);
            let x4 = pair2(x, f, ch);
            let a = self.split1[0].run(x4);
            let a = self.split1[1].run(a); // low L, R | rest L, R
            let [low_l, low_r, rest_l, rest_r] = a.to_array();
            let b = self.split2[0].run(F4::new(rest_l, rest_r, rest_l, rest_r));
            let b = self.split2[1].run(b); // mid L, R | high L, R
            let low = self.low_ap.run(F4::new(low_l, low_r, 0.0, 0.0)).to_array();
            let [mid_l, mid_r, high_l, high_r] = b.to_array();
            let [gl, gm, gh, _] = self.smooth.to_array();
            let l = low[0] * gl + mid_l * gm + high_l * gh;
            let r = low[1] * gl + mid_r * gm + high_r * gh;
            if ch == 1 {
                x[f] = l;
            } else {
                x[2 * f] = l;
                x[2 * f + 1] = r;
            }
        }
    }
}

/// The EQ: a low shelf, a mid peak, a high shelf in a row (a stereo pair a pass), their levels moved a block at a
/// time.
pub struct ClassicEq {
    rate: f64,
    bands: [Biquad4; 3],
    db: [Smooth; 3],
    set: [f64; 3], // the levels the coefficients were made for
    active: bool,
}

impl ClassicEq {
    pub fn new(rate: f64) -> ClassicEq {
        ClassicEq {
            rate,
            bands: [Biquad4::default(); 3],
            db: [Smooth::new(0.0, SMOOTH_SEC, rate); 3],
            set: [f64::NAN; 3],
            active: false,
        }
    }

    fn reset(&mut self) {
        self.bands.iter_mut().for_each(Biquad4::reset);
    }

    /// ``db``: low, mid, high levels. All 0 dB and settled: untouched.
    pub fn process(&mut self, x: &mut [f32], n: usize, ch: usize, db: [f64; 3]) {
        let now: [f64; 3] = std::array::from_fn(|k| self.db[k].step(db[k], n));
        if now == [0.0; 3] {
            if self.active {
                self.active = false;
                self.reset();
            }
            return;
        }
        self.active = true;
        for k in 0..3 {
            if now[k] != self.set[k] {
                self.set[k] = now[k];
                let c = match k {
                    0 => Coeffs::low_shelf(EQ_LOW_HZ, now[k], self.rate),
                    1 => Coeffs::peak(EQ_MID_HZ, EQ_MID_Q, now[k], self.rate),
                    _ => Coeffs::high_shelf(EQ_HIGH_HZ, now[k], self.rate),
                };
                self.bands[k].set([c; 4]);
            }
        }
        for f in 0..n {
            let mut v = pair2(x, f, ch);
            for b in self.bands.iter_mut() {
                v = b.run(v);
            }
            let y = v.to_array();
            if ch == 1 {
                x[f] = y[0];
            } else {
                x[2 * f] = y[0];
                x[2 * f + 1] = y[1];
            }
        }
    }
}

/// The channel strip's settings, as the deck (deck.rs: its buttons, keys and a controller's knobs) sets them and the
/// main device's callback reads them: trim (a gain), the EQ knobs, the COLOR knob, the channel fader's position
/// (the deck's volume), its headphones CUE, the deck's tempo (Sound Color FX's DUB ECHO); and its meter, which the
/// callback writes (the peak before the fader since it was last taken).
pub struct StripParams {
    trim: AtomicU64,    // f64 bits, linear
    eq: [AtomicU64; 3], // f64 bits, knobs 0 .. 1: low, mid, high
    color: AtomicU64,   // f64 bits, knob 0 .. 1 (0.5 off)
    fader: AtomicU64,   // f64 bits, position 0 .. 1
    cue: AtomicBool,
    bpm: AtomicU64,  // f64 bits (0: unknown)
    peak: AtomicU32, // f32 bits (>= 0: ordered as integers)
}

impl StripParams {
    fn new() -> StripParams {
        StripParams {
            trim: AtomicU64::new(1f64.to_bits()),
            eq: std::array::from_fn(|_| AtomicU64::new(0.5f64.to_bits())),
            color: AtomicU64::new(0.5f64.to_bits()),
            fader: AtomicU64::new(1f64.to_bits()),
            cue: AtomicBool::new(false),
            bpm: AtomicU64::new(0f64.to_bits()),
            peak: AtomicU32::new(0),
        }
    }

    fn unit(v: f64, default: f64) -> u64 {
        (if v.is_finite() { v.clamp(0.0, 1.0) } else { default }).to_bits()
    }

    pub fn set_trim(&self, g: f64) {
        self.trim.store((if g.is_finite() { g.max(0.0) } else { 1.0 }).to_bits(), Ordering::Relaxed);
    }

    /// The EQ knobs (0 .. 1, the middle 0 dB): low, mid, high.
    pub fn set_eq(&self, knobs: [f64; 3]) {
        for (a, k) in self.eq.iter().zip(knobs) {
            a.store(Self::unit(k, 0.5), Ordering::Relaxed);
        }
    }

    pub fn set_color(&self, knob: f64) {
        self.color.store(Self::unit(knob, 0.5), Ordering::Relaxed);
    }

    /// The channel fader's position (0 .. 1; its gain on the mixer's curve).
    pub fn set_fader(&self, position: f64) {
        self.fader.store(Self::unit(position, 1.0), Ordering::Relaxed);
    }

    pub fn set_bpm(&self, bpm: f64) {
        self.bpm.store((if bpm.is_finite() { bpm.max(0.0) } else { 0.0 }).to_bits(), Ordering::Relaxed);
    }

    pub fn bpm(&self) -> f64 {
        Self::load(&self.bpm)
    }

    /// The headphones' CUE of this channel.
    pub fn set_cue(&self, on: bool) {
        self.cue.store(on, Ordering::Relaxed);
    }

    pub fn toggle_cue(&self) {
        self.cue.fetch_xor(true, Ordering::Relaxed);
    }

    pub fn cue_on(&self) -> bool {
        self.cue.load(Ordering::Relaxed)
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

/// The channel strip's processing, in the main device's callback: the player's music through trim, EQ, its meter and
/// its pre-fader tap (the headphones' CUE), then the channel fader, Sound Color FX in its place (Slot); the player's
/// listening aids (the metronome, reference tones) added after it all, as they were (their level their own). At its
/// neutral settings the music passes bit for bit.
pub struct Strip {
    iso: Isolator,
    eq: ClassicEq,
    color: ColorFx,
    eq_mode: u8,
    trim: f32,  // as applied (a change moved to over a block)
    fader: f32, // as applied
}

impl Strip {
    pub fn new(rate: f64, seed: u32) -> Strip {
        Strip { iso: Isolator::new(rate), eq: ClassicEq::new(rate), color: ColorFx::new(rate, seed), eq_mode: 0, trim: 1.0,
                fader: 1.0 }
    }

    /// Its effects' tails left (a deck stopped: the strip still runs on silence until they are out).
    pub fn ringing(&self) -> bool {
        !self.color.idle()
    }

    /// The Color FX on ``x`` (``n`` frames of ``ch``) where its place is ``slot``.
    fn effects(&mut self, slot: Slot, x: &mut [f32], n: usize, ch: usize, p: &StripParams, m: &Mix) {
        let color = if m.color_type == ColorType::Off { self.color.kind() } else { m.color_type };
        if color.slot() == slot {
            self.color.process(x, n, ch, m.color_type, StripParams::load(&p.color), m.color_param, p.bpm());
        }
    }

    /// ``lane``: ``n`` frames of the player's [music (``ch``), aids (``ch``)]: the music before the fader into
    /// ``pfl``, the strip's output into ``post`` (``ch`` a frame each).
    #[allow(clippy::too_many_arguments)]
    pub fn process(&mut self, lane: &[f32], n: usize, ch: usize, p: &StripParams, m: &Mix, pfl: &mut [f32],
                   post: &mut [f32]) {
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
        self.effects(Slot::PreEq, pfl, n, ch, p, m);
        let knobs = p.eq.each_ref().map(StripParams::load);
        if m.eq_mode != self.eq_mode {
            // the other EQ: started clean (each fades in from 0 dB; the one left is reset when settled)
            self.eq_mode = m.eq_mode;
        }
        if self.eq_mode == 1 {
            self.iso.process(pfl, n, ch, [1.0; 3]);
            self.eq.process(pfl, n, ch, knobs.map(|k| eq_knob_db(k, false).unwrap_or(EQ_MIN_DB)));
        } else {
            self.eq.process(pfl, n, ch, [0.0; 3]);
            self.iso.process(pfl, n, ch, knobs.map(eq_knob_gain));
        }
        self.effects(Slot::PostEq, pfl, n, ch, p, m);
        p.add_peak(peak_abs(pfl));
        let fader = fader_gain(StripParams::load(&p.fader), m.fader_curve) as f32;
        let step = (fader - self.fader) / n.max(1) as f32;
        let post = &mut post[..n * ch];
        if step == 0.0 && fader == 1.0 {
            post.copy_from_slice(pfl);
        } else if step == 0.0 {
            post.copy_from_slice(pfl);
            vk::scale_in(post, fader);
        } else {
            for f in 0..n {
                let g = self.fader + step * (f + 1) as f32;
                for c in 0..ch {
                    post[f * ch + c] = pfl[f * ch + c] * g;
                }
            }
        }
        self.effects(Slot::PostFader, post, n, ch, p, m);
        // the aids after the fader and its effects
        for f in 0..n {
            for c in 0..ch {
                let a = lane[f * lch + ch + c];
                if a != 0.0 {
                    post[f * ch + c] += a;
                }
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
/// The routes: main L / R, headphones L / R, booth L / R.
pub const ROUTES: usize = 6;

/// The mixer's settings, changed from the engine thread and read once per callback: the channel's strip
/// (StripParams), the mixer's own (ipc::mix keys, f64 each), soft clipping, where each bus goes on the main device
/// (ROUTES: a pair of its channels, NO_CHANNEL: not there); and its meters, which the callback writes.
pub struct MixParams {
    pub strip: Arc<StripParams>,
    values: [AtomicU64; mix::KEYS],
    soft_clip: AtomicBool,
    routes: [AtomicU8; ROUTES],
    meters: [AtomicU32; mix::METER_COUNT], // f32 bits: peaks (linear) since taken
}

/// One reading of MixParams (the callback's).
#[derive(Clone, Copy)]
pub struct Mix {
    pub cue: bool,               // the channel's headphones CUE
    pub hp: (f32, f32, bool),    // the headphones: CUE / MASTER mix, level gain, the master mixed in
    pub master: f32,
    pub booth: f32,
    pub booth_eq: [f64; 2], // dB: low, high
    pub eq_mode: u8,
    pub fader_curve: u8,
    pub color_type: ColorType,
    pub color_param: f64,
    pub hp_split: bool, // the headphones' MONO SPLIT
    pub soft_clip: bool,
    pub routes: [u8; ROUTES],
}

impl MixParams {
    pub fn new(out_channels: usize) -> MixParams {
        let p = MixParams {
            strip: Arc::new(StripParams::new()),
            values: std::array::from_fn(|k| AtomicU64::new(mix::DEFAULTS[k].to_bits())),
            soft_clip: AtomicBool::new(false),
            routes: std::array::from_fn(|_| AtomicU8::new(NO_CHANNEL)),
            meters: std::array::from_fn(|_| AtomicU32::new(0)),
        };
        // the master on the device's first two channels (one: its only one), nothing else routed
        let mut routes = [NO_CHANNEL; ROUTES];
        routes[0] = 0;
        routes[1] = if out_channels > 1 { 1 } else { NO_CHANNEL };
        p.set_routes(&routes);
        p
    }

    /// Setting ``key`` (ipc::mix) to ``value`` (held to its range).
    pub fn set(&self, key: usize, value: f64) {
        let Some(a) = self.values.get(key) else { return };
        let v = if value.is_finite() { value } else { mix::DEFAULTS[key] };
        let v = match key {
            mix::EQ_MODE | mix::HP_MASTER | mix::HP_SPLIT => v.round().clamp(0.0, 1.0),
            mix::FADER_CURVE => v.round().clamp(0.0, 2.0),
            mix::COLOR_TYPE => v.round().clamp(0.0, ColorType::COUNT as f64),
            _ => v.clamp(0.0, 1.0),
        };
        a.store(v.to_bits(), Ordering::Relaxed);
    }

    pub fn get(&self, key: usize) -> f64 {
        self.values.get(key).map_or(0.0, |a| f64::from_bits(a.load(Ordering::Relaxed)))
    }

    /// A switch flipped (MONO SPLIT ...) or a choice stepped by ``by`` (COLOR_TYPE: around).
    pub fn step(&self, key: usize, by: i64) {
        let hi = if key == mix::COLOR_TYPE { ColorType::COUNT as i64 - 1 } else { 1 };
        let now = self.get(key).round() as i64;
        self.set(key, (now + by).rem_euclid(hi + 1) as f64);
    }

    pub fn set_cue_mix(&self, value: f64) {
        self.set(mix::HP_MIX, value);
    }

    pub fn set_cue_level(&self, knob: f64) {
        self.set(mix::HP_LEVEL, knob);
    }

    pub fn set_master_cue(&self, on: bool) {
        self.set(mix::HP_MASTER, on as u8 as f64);
    }

    pub fn set_soft_clip(&self, on: bool) {
        self.soft_clip.store(on, Ordering::Relaxed);
    }

    /// main L / R, headphones L / R, booth L / R (NO_CHANNEL: not routed); a shorter list leaves the rest as they
    /// are.
    pub fn set_routes(&self, routes: &[u8]) {
        for (a, &r) in self.routes.iter().zip(routes) {
            a.store(r, Ordering::Relaxed);
        }
    }

    fn meter(&self, k: usize, v: f32) {
        if v.is_finite() && v > 0.0 {
            self.meters[k].fetch_max(v.to_bits(), Ordering::Relaxed);
        }
    }

    /// The meters since the last call (linear peaks).
    pub fn take_meters(&self) -> [f32; mix::METER_COUNT] {
        std::array::from_fn(|k| f32::from_bits(self.meters[k].swap(0, Ordering::Relaxed)))
    }

    pub fn read(&self) -> Mix {
        Mix {
            cue: self.strip.cue_on(),
            hp: (self.get(mix::HP_MIX) as f32, level_gain(self.get(mix::HP_LEVEL)) as f32,
                 self.get(mix::HP_MASTER) != 0.0),
            master: level_gain(self.get(mix::MASTER_LEVEL)) as f32,
            booth: level_gain(self.get(mix::BOOTH_LEVEL)) as f32,
            booth_eq: [booth_eq_db(self.get(mix::BOOTH_EQ_LOW)), booth_eq_db(self.get(mix::BOOTH_EQ_HI))],
            eq_mode: self.get(mix::EQ_MODE) as u8,
            fader_curve: self.get(mix::FADER_CURVE) as u8,
            color_type: ColorType::from_code(self.get(mix::COLOR_TYPE) as u8),
            color_param: self.get(mix::COLOR_PARAM),
            hp_split: self.get(mix::HP_SPLIT) != 0.0,
            soft_clip: self.soft_clip.load(Ordering::Relaxed),
            routes: std::array::from_fn(|k| self.routes[k].load(Ordering::Relaxed)),
        }
    }
}

/// What the callback wants of the mixer: the headphones, the booth (each routed or bridged).
#[derive(Clone, Copy, Default)]
pub struct Wanted {
    pub headphones: bool,
    pub booth: bool,
}

/// The mixer, in the main device's callback. The outputs as stereo buses (a mono deck's on both sides). A lane that
/// gave nothing leaves the channel out: the deck at its neutral settings comes out bit for bit as the player
/// rendered it.
pub struct Mixer {
    rate: f64,
    strip: Strip,
    ch: usize,
    pfl: Vec<f32>,
    post: Vec<f32>,
    silence: Vec<f32>, // a lane's block of nothing: a stopped deck's strip run while its effect rings out
    cue: Vec<f32>,     // stereo
    pub master: Vec<f32>,     // stereo, a piece's frames
    pub headphones: Vec<f32>, // stereo
    pub booth: Vec<f32>,      // stereo
    booth_eq: [Biquad4; 2],
    booth_set: [f64; 2],
}

impl Mixer {
    pub fn new(rate: f64, ch: usize, frames: usize) -> Mixer {
        let ch = ch.clamp(1, 2);
        let stereo = || vec![0f32; frames * 2];
        Mixer {
            rate,
            strip: Strip::new(rate, 0x9E37_79B9 ^ 1),
            ch,
            pfl: vec![0.0; frames * ch],
            post: vec![0.0; frames * ch],
            silence: vec![0.0; frames * 2 * ch],
            cue: stereo(),
            master: stereo(),
            headphones: stereo(),
            booth: stereo(),
            booth_eq: [Biquad4::default(); 2],
            booth_set: [f64::NAN; 2],
        }
    }

    /// ``x`` (``ch`` a frame) added into the stereo ``bus`` (a mono deck on both sides).
    fn add_stereo(bus: &mut [f32], x: &[f32], n: usize, ch: usize) {
        if ch == 2 {
            vk::add_in(&mut bus[..n * 2], &x[..n * 2]);
        } else {
            for f in 0..n {
                bus[2 * f] += x[f];
                bus[2 * f + 1] += x[f];
            }
        }
    }

    /// ``n`` frames (at most the mixer's): ``lane`` the deck's lane block where it gave frames; the master into
    /// ``master`` and what ``wanted`` asks into the others.
    pub fn process(&mut self, lane: Option<&[f32]>, n: usize, m: &Mix, params: &MixParams, wanted: Wanted) {
        let ch = self.ch;
        let s2 = n * 2;
        self.cue[..s2].iter_mut().for_each(|v| *v = 0.0);
        self.master[..s2].iter_mut().for_each(|v| *v = 0.0);
        // a stopped deck (its lane gave nothing) still runs its strip on silence while its effect rings out
        let lane = match lane {
            Some(lane) => Some(lane),
            None if self.strip.ringing() => Some(&self.silence[..n * 2 * ch]),
            None => None,
        };
        if let Some(lane) = lane {
            self.strip.process(lane, n, ch, &params.strip, m, &mut self.pfl, &mut self.post);
            if wanted.headphones && m.cue {
                Self::add_stereo(&mut self.cue, &self.pfl, n, ch);
            }
            Self::add_stereo(&mut self.master, &self.post, n, ch);
        }
        // the booth: the master before its level, at the booth's level through its EQ
        if wanted.booth {
            let booth = &mut self.booth[..s2];
            booth.copy_from_slice(&self.master[..s2]);
            if m.booth != 1.0 {
                vk::scale_in(booth, m.booth);
            }
            if m.booth_eq != [0.0; 2] {
                for k in 0..2 {
                    if m.booth_eq[k] != self.booth_set[k] {
                        self.booth_set[k] = m.booth_eq[k];
                        let c = if k == 0 { Coeffs::low_shelf(EQ_LOW_HZ, m.booth_eq[0], self.rate) } else {
                            Coeffs::high_shelf(EQ_HIGH_HZ, m.booth_eq[1], self.rate)
                        };
                        self.booth_eq[k].set([c; 4]);
                    }
                }
                for f in 0..n {
                    let v = F4::new(booth[2 * f], booth[2 * f + 1], 0.0, 0.0);
                    let v = self.booth_eq[0].run(v);
                    let y = self.booth_eq[1].run(v).to_array();
                    booth[2 * f] = y[0];
                    booth[2 * f + 1] = y[1];
                }
            } else if self.booth_set != [0.0; 2] {
                self.booth_set = [0.0; 2];
                self.booth_eq.iter_mut().for_each(Biquad4::reset);
            }
            params.meter(mix::METER_BOOTH, peak_abs(booth));
            if m.soft_clip {
                crate::dsp::soft_clip(booth, SOFT_CLIP_KNEE);
            }
        }
        // the master's level
        let master = &mut self.master[..s2];
        if m.master != 1.0 {
            vk::scale_in(master, m.master);
        }
        params.meter(mix::METER_MASTER, peak_abs(master));
        // the headphones: the CUE'd channel, the master by their mix when it is mixed in, their level
        if wanted.headphones {
            let (mix, level, master_in) = m.hp;
            let mx = if master_in { mix } else { 0.0 };
            let hp = &mut self.headphones[..s2];
            if m.hp_split {
                // MONO SPLIT: the CUE (mono) left, the master (mono) right, the mix their balance
                let gc = (2.0 * (1.0 - mix)).min(1.0);
                let gm = if master_in { (2.0 * mix).min(1.0) } else { 0.0 };
                let (cue, master) = (&self.cue, &self.master);
                for f in 0..n {
                    hp[2 * f] = gc * 0.5 * (cue[2 * f] + cue[2 * f + 1]);
                    hp[2 * f + 1] = gm * 0.5 * (master[2 * f] + master[2 * f + 1]);
                }
            } else {
                hp.copy_from_slice(&self.cue[..s2]);
                if mx != 0.0 {
                    vk::scale_in(hp, 1.0 - mx);
                    crate::simd::axpy(hp, &self.master[..s2], mx);
                }
            }
            if level != 1.0 {
                vk::scale_in(hp, level);
            }
            if m.soft_clip {
                crate::dsp::soft_clip(hp, SOFT_CLIP_KNEE);
            }
        }
        if m.soft_clip {
            crate::dsp::soft_clip(&mut self.master[..s2], SOFT_CLIP_KNEE);
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

    fn level_after(eq: &mut dyn FnMut(&mut [f32]), f: f64) -> f64 {
        let mut y = sine(f, 48000.0, 48000, 1);
        eq(&mut y);
        20.0 * (rms(&y[24000..]) / rms(&sine(f, 48000.0, 24000, 1))).log10()
    }

    #[test]
    fn neutral_is_untouched() {
        let mut iso = Isolator::new(48000.0);
        let x = sine(440.0, 48000.0, 4800, 2);
        let mut y = x.clone();
        iso.process(&mut y, 4800, 2, [1.0; 3]);
        assert_eq!(x, y);
        // a strip at its neutral settings: the music bit for bit, the aids added after the fader
        let mut strip = Strip::new(48000.0, 1);
        let lane: Vec<f32> = x.chunks_exact(2).flat_map(|f| [f[0], f[1], 0.0, 0.0]).collect();
        let (mut pfl, mut post) = (vec![0f32; x.len()], vec![0f32; x.len()]);
        let params = MixParams::new(2);
        strip.process(&lane, 4800, 2, &StripParams::new(), &params.read(), &mut pfl, &mut post);
        assert_eq!(x, pfl);
        assert_eq!(x, post);
        // through the whole mixer, the master at 0 dB
        let mut mixer = Mixer::new(48000.0, 2, 4800);
        mixer.process(Some(&lane), 4800, &params.read(), &params, Wanted::default());
        assert_eq!(&mixer.master[..x.len()], &x[..]);
    }

    #[test]
    fn isolator_bands_at_0_db_sum_flat() {
        for f in [60.0, 246.0, 1000.0, 2484.0, 8000.0] {
            let mut iso = Isolator::new(48000.0);
            let level = level_after(&mut |y| iso.process(y, 48000, 1, [1.0, 1.0, 1.0 + 1e-6]), f);
            assert!(level.abs() < 0.05, "{f} Hz: {level} dB");
        }
    }

    #[test]
    fn isolator_kills() {
        let mut iso = Isolator::new(48000.0);
        assert!(level_after(&mut |y| iso.process(y, 48000, 1, [0.0, 1.0, 1.0]), 50.0) < -30.0);
        let mut iso = Isolator::new(48000.0);
        assert!(level_after(&mut |y| iso.process(y, 48000, 1, [1.0, 1.0, 0.0]), 10000.0) < -30.0);
    }

    #[test]
    fn classic_eq_bands() {
        let mut eq = ClassicEq::new(48000.0);
        let low = level_after(&mut |y| eq.process(y, 48000, 1, [EQ_MIN_DB, 0.0, 0.0]), 15.0);
        assert!((low - EQ_MIN_DB).abs() < 2.0, "low shelf at 15 Hz: {low}");
        let mut eq = ClassicEq::new(48000.0);
        let mid = level_after(&mut |y| eq.process(y, 48000, 1, [0.0, -12.0, 0.0]), 1000.0);
        assert!((mid + 12.0).abs() < 0.5, "mid at 1 kHz: {mid}");
        let mut eq = ClassicEq::new(48000.0);
        let flat = level_after(&mut |y| eq.process(y, 48000, 1, [0.0, 0.0, 0.0]), 1000.0);
        assert!(flat.abs() < 1e-6);
    }

    /// The headphones: silent with no CUE on and the master not mixed in; the CUE'd channel before its fader; the
    /// master by the mix when it is mixed in.
    #[test]
    fn headphones_cue() {
        let params = MixParams::new(2);
        let mut mixer = Mixer::new(48000.0, 2, 64);
        let lane: Vec<f32> = (0..64).flat_map(|_| [0.5f32, 0.5, 0.0, 0.0]).collect();
        params.strip.set_fader(fader_inverse(0.5));
        params.set_master_cue(false);
        let want = Wanted { headphones: true, booth: true };
        let run = |mixer: &mut Mixer| {
            for _ in 0..8 {
                mixer.process(Some(&lane), 64, &params.read(), &params, want);
            }
            (mixer.master[0], mixer.headphones[0])
        };
        let (master, hp) = run(&mut mixer);
        assert!((master - 0.25).abs() < 1e-4 && hp == 0.0, "{master} {hp}");
        params.strip.toggle_cue();
        let (_, hp) = run(&mut mixer);
        assert!((hp - 0.5).abs() < 1e-6, "pre-fader: {hp}");
        params.set_master_cue(true);
        params.set_cue_mix(1.0);
        let (_, hp) = run(&mut mixer);
        assert!((hp - 0.25).abs() < 1e-4, "the master alone: {hp}");
    }

    /// The effects in their places (the A9's block diagram): NOISE before the EQ (a kill takes its noise too), DUB
    /// ECHO after the fader (not in the CUE; its tail on with the fader down and the deck stopped).
    #[test]
    fn effects_in_their_places() {
        let x = sine(440.0, 48000.0, 512, 2).iter().map(|v| 0.4 * v).collect::<Vec<f32>>();
        let lane: Vec<f32> = x.chunks_exact(2).flat_map(|f| [f[0], f[1], 0.0, 0.0]).collect();
        let want = Wanted { headphones: true, booth: false };
        let setup = || {
            let params = MixParams::new(2);
            params.strip.toggle_cue();
            params.set(mix::HP_MASTER, 0.0); // the headphones: the CUE alone
            params
        };
        let run = |mixer: &mut Mixer, params: &MixParams, lane: Option<&[f32]>, blocks: usize| {
            let (mut master, mut cue) = (0f64, 0f64);
            for _ in 0..blocks {
                mixer.process(lane, 512, &params.read(), params, want);
                master = rms(&mixer.master[..1024]);
                cue = rms(&mixer.headphones[..1024]);
            }
            (master, cue)
        };
        // NOISE, before the EQ: every band killed, nothing comes out (after it, the noise would)
        let params = setup();
        let mut mixer = Mixer::new(48000.0, 2, 512);
        params.set(mix::COLOR_TYPE, ColorType::Noise as u8 as f64);
        params.set(mix::COLOR_PARAM, 1.0);
        params.strip.set_color(1.0);
        params.strip.set_eq([0.0; 3]);
        let (master, _) = run(&mut mixer, &params, Some(&lane), 40);
        assert!(master < 1e-3, "NOISE through a full kill: {master}");
        // DUB ECHO, after the fader: the CUE is the music as it was; the fader down and the deck stopped, its
        // repeats ring on
        let params = setup();
        let mut mixer = Mixer::new(48000.0, 2, 512);
        params.set(mix::COLOR_TYPE, ColorType::DubEcho as u8 as f64);
        params.set(mix::COLOR_PARAM, 0.6);
        params.strip.set_color(0.95);
        run(&mut mixer, &params, Some(&lane), 40);
        assert_eq!(&mixer.headphones[..1024], &x[..], "the CUE holds no echo");
        params.strip.set_fader(0.0);
        let (master, _) = run(&mut mixer, &params, None, 4);
        assert!(master > 1e-3, "the echo's tail with the fader down, the deck stopped: {master}");
    }

    /// The fader position giving gain ``g`` on the default curve.
    fn fader_inverse(g: f64) -> f64 {
        let (mut lo, mut hi) = (0.0, 1.0);
        for _ in 0..60 {
            let mid = 0.5 * (lo + hi);
            if fader_gain(mid, 0) < g { lo = mid } else { hi = mid }
        }
        0.5 * (lo + hi)
    }

    #[test]
    fn knobs() {
        assert_eq!(eq_knob_gain(0.0), 0.0);
        assert!((eq_knob_gain(0.5) - 1.0).abs() < 1e-12);
        assert!((20.0 * eq_knob_gain(1.0).log10() - EQ_MAX_DB).abs() < 1e-9);
        assert!((trim_knob_gain(0.5) - 1.0).abs() < 1e-12);
        assert_eq!(level_gain(mix::LEVEL_UNITY), 1.0);
        assert_eq!(fader_gain(1.0, 0), 1.0);
        assert!(fader_gain(0.5, 1) > fader_gain(0.5, 0) && fader_gain(0.5, 2) < fader_gain(0.5, 0));
    }

    #[test]
    fn color_type_steps_around() {
        let params = MixParams::new(2);
        params.set(mix::COLOR_TYPE, (ColorType::COUNT - 1) as f64);
        params.step(mix::COLOR_TYPE, 1);
        assert_eq!(params.get(mix::COLOR_TYPE), 0.0);
        params.step(mix::COLOR_TYPE, -1);
        assert_eq!(params.get(mix::COLOR_TYPE), (ColorType::COUNT - 1) as f64);
        params.step(mix::HP_SPLIT, 1);
        assert_eq!(params.get(mix::HP_SPLIT), 1.0);
    }
}
