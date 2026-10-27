//! Sound Color FX, as a DJM-A9's (its manual's pages 38 - 41), and JET (a DJM-800's): one effect picked for the
//! mixer (SPACE, DUB ECHO, SWEEP, NOISE, CRUSH, FILTER, JET, or none: its button pressed again), each channel's
//! COLOR knob turning it on that channel (the middle off; left and right two flavours, more the farther), the
//! PARAMETER knob its quantity (SPACE / DUB ECHO: feedback, SWEEP: the filter's centre frequency, NOISE: the noise's
//! volume, CRUSH: how crushed, FILTER: resonance, JET: feedback). Each effect has its place in the channel's strip
//! (``ColorType::slot``), as the A9's block diagram (page 93) has three: after the trim, before the EQ (NOISE: the
//! EQ shapes its noise, as the manual says), after the EQ, before the CUE and the fader (the filters, CRUSH, JET), and
//! after the fader (SPACE, DUB ECHO: their tail rings on with the fader down, as a DJM's echoes do). At the knob's
//! middle with no tail left the samples pass bit for bit.
//!
//! The filters run a stereo pair per SIMD pass (simd::F4 lanes: left, right), the reverb's four lines a lane each
//! (fxkit::Fdn4).

use crate::filters::Svf4;
use crate::fxkit::{Delay, Fdn4, Noise, Smooth};
use crate::simd::F4;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ColorType {
    Space = 0,
    DubEcho = 1,
    Sweep = 2,
    Noise = 3,
    Crush = 4,
    Filter = 5,
    Jet = 6,
    Off = 7,
}

/// A place in a channel's strip (the A9's: its block diagram, page 93).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Slot {
    PreEq,     // after the trim, before the EQ
    PostEq,    // after the EQ, before the CUE taps and the fader
    PostFader, // after the fader: a tail rings on with it down
}

impl ColorType {
    pub const COUNT: u8 = 7; // the effects (Off past them)

    /// Its place in the strip.
    pub fn slot(self) -> Slot {
        match self {
            ColorType::Noise => Slot::PreEq,
            ColorType::Space | ColorType::DubEcho => Slot::PostFader,
            _ => Slot::PostEq,
        }
    }

    pub fn from_code(c: u8) -> ColorType {
        match c {
            0 => ColorType::Space,
            1 => ColorType::DubEcho,
            2 => ColorType::Sweep,
            3 => ColorType::Noise,
            4 => ColorType::Crush,
            5 => ColorType::Filter,
            6 => ColorType::Jet,
            _ => ColorType::Off,
        }
    }
}

const ECHO_MAX_SEC: f64 = 2.0;
/// DUB ECHO's time: 3/4 of a beat (a dotted eighth, a dub delay's) at the channel's deck's BPM, else this.
const ECHO_SEC: f64 = 0.375;
/// A tail (echo, reverb) left after the knob went back to the middle is dropped once below this level.
const TAIL_FLOOR: f32 = 1e-5;
/// SPACE's lines hold a sound this long at most.
const SPACE_SPAN_SEC: f64 = 0.25;
/// NOISE's level at the PARAMETER's top.
const NOISE_MAX: f64 = 0.4;
/// JET's comb: its delay from the knob's middle to its end (ms; the comb's teeth rise in pitch as it turns), its
/// feedback at the PARAMETER's bottom and top.
const JET_MS: (f64, f64) = (6.0, 0.12);
const JET_FEEDBACK: (f64, f64) = (0.35, 0.92);

/// One channel's Color FX (stereo).
pub struct ColorFx {
    rate: f64,
    kind: ColorType,
    knob: Smooth,  // -1 (left) .. 0 .. 1 (right)
    param: Smooth, // 0 .. 1
    svf: Svf4,     // lanes: left, right
    fb_svf: Svf4,  // the echo's feedback filter
    echo: Delay,
    reverb: Fdn4,
    noise: Noise,
    crush_hold: (f32, f32),
    crush_count: f64,
    // the echo's / reverb's line: frames since anything above TAIL_FLOOR went in or came out, and how long the line
    // holds a sound (silent that long: empty)
    quiet: usize,
    span: usize,
}

impl ColorFx {
    pub fn new(rate: f64, seed: u32) -> ColorFx {
        ColorFx {
            rate,
            kind: ColorType::Filter,
            knob: Smooth::new(0.0, 0.02, rate),
            param: Smooth::new(0.5, 0.05, rate),
            svf: Svf4::default(),
            fb_svf: Svf4::default(),
            echo: Delay::new((ECHO_MAX_SEC * rate) as usize + 8),
            reverb: Fdn4::new(rate, 1.3),
            noise: Noise::new(seed),
            crush_hold: (0.0, 0.0),
            crush_count: 0.0,
            quiet: usize::MAX / 2,
            span: 0,
        }
    }

    /// Nothing to do: the knob at its middle (settled) and no tail.
    pub fn idle(&self) -> bool {
        self.knob.value == 0.0 && self.quiet >= self.span
    }

    /// The effect it plays now (the one picked; with none picked, the one still ringing out).
    pub fn kind(&self) -> ColorType {
        self.kind
    }

    fn switch(&mut self, kind: ColorType) {
        if kind != self.kind {
            self.kind = kind;
            self.svf.reset();
            self.fb_svf.reset();
            self.echo.clear();
            self.reverb.clear();
            self.quiet = usize::MAX / 2;
            self.span = 0;
        }
    }

    /// ``x``: ``n`` frames of ``ch`` (1 or 2) channels, in place. ``knob`` 0 .. 1 (0.5 off), ``param`` 0 .. 1,
    /// ``bpm`` the channel's deck's (0 unknown). None picked (Off): the effect in use goes as its knob would back to
    /// the middle (its tail rings out).
    #[allow(clippy::too_many_arguments)]
    pub fn process(&mut self, x: &mut [f32], n: usize, ch: usize, kind: ColorType, knob: f64, param: f64, bpm: f64) {
        let target = ((knob.clamp(0.0, 1.0) - 0.5) * 2.0).clamp(-1.0, 1.0);
        let target = if target.abs() < 0.02 || kind == ColorType::Off { 0.0 } else { target }; // a dead zone at the
                                                                                                  // middle
        if target == 0.0 && self.idle() {
            self.knob.value = 0.0;
            return;
        }
        if kind != ColorType::Off {
            self.switch(kind);
        }
        let k = self.knob.step(target, n);
        let p = self.param.step(param.clamp(0.0, 1.0), n);
        let amount = k.abs();
        let left = k < 0.0;
        match self.kind {
            ColorType::Filter => self.filter(x, n, ch, amount, left, p),
            ColorType::Sweep => self.sweep(x, n, ch, amount, left, p),
            ColorType::Noise => self.noise(x, n, ch, amount, left, p),
            ColorType::Crush => self.crush(x, n, ch, amount, left, p),
            ColorType::DubEcho => self.dub_echo(x, n, ch, amount, left, p, bpm),
            ColorType::Space => self.space(x, n, ch, amount, left, p),
            ColorType::Jet => self.jet(x, n, ch, amount, left, p),
            ColorType::Off => {}
        }
    }

    #[inline(always)]
    fn frame(x: &[f32], f: usize, ch: usize) -> (f32, f32) {
        if ch == 1 { (x[f], x[f]) } else { (x[f * 2], x[f * 2 + 1]) }
    }

    #[inline(always)]
    fn put(x: &mut [f32], f: usize, ch: usize, l: f32, r: f32) {
        if ch == 1 {
            x[f] = 0.5 * (l + r);
        } else {
            x[f * 2] = l;
            x[f * 2 + 1] = r;
        }
    }

    /// Left: a low-pass from 20 kHz down to 40 Hz; right: a high-pass from 20 Hz up to 12 kHz; the parameter its
    /// resonance.
    fn filter(&mut self, x: &mut [f32], n: usize, ch: usize, amount: f64, left: bool, p: f64) {
        let f = if left { 20000.0 * (40.0f64 / 20000.0).powf(amount) } else { 20.0 * (12000.0f64 / 20.0).powf(amount) };
        let q = 0.6 + p * 6.0;
        self.svf.set([f; 4], [q; 4], self.rate);
        for i in 0..n {
            let (l, r) = Self::frame(x, i, ch);
            let o = self.svf.run(F4::new(l, r, 0.0, 0.0));
            let y = if left { o.lp } else { o.hp }.to_array();
            Self::put(x, i, ch, y[0], y[1]);
        }
    }

    /// Left: a notch, wider the farther; right: a band-pass, narrower the farther; the parameter their centre
    /// frequency (100 Hz - 10 kHz, 1 kHz at the middle).
    fn sweep(&mut self, x: &mut [f32], n: usize, ch: usize, amount: f64, left: bool, p: f64) {
        let f = 100.0 * 100f64.powf(p);
        let q = if left { 4.0 * (0.15f64 / 4.0).powf(amount) } else { 0.5 * (10.0f64 / 0.5).powf(amount) };
        self.svf.set([f; 4], [q; 4], self.rate);
        let k = F4::splat((1.0 / q) as f32);
        let mix = (4.0 * amount).min(1.0) as f32; // eased in from the middle
        for i in 0..n {
            let (l, r) = Self::frame(x, i, ch);
            let v = F4::new(l, r, 0.0, 0.0);
            let bp = k * self.svf.run(v).bp; // unity at the centre
            let y = if left { v - bp } else { bp }.to_array();
            Self::put(x, i, ch, l + mix * (y[0] - l), r + mix * (y[1] - r));
        }
    }

    /// White noise added through a filter: left a low-pass coming down (16 kHz to 150 Hz), right a high-pass going
    /// up (20 Hz to 10 kHz); the parameter its volume.
    fn noise(&mut self, x: &mut [f32], n: usize, ch: usize, amount: f64, left: bool, p: f64) {
        let f = if left { 16000.0 * (150.0f64 / 16000.0).powf(amount) } else { 20.0 * (10000.0f64 / 20.0).powf(amount) };
        self.svf.set([f; 4], [0.707; 4], self.rate);
        let level = (NOISE_MAX * p * (5.0 * amount).min(1.0)) as f32;
        for i in 0..n {
            let (l, r) = Self::frame(x, i, ch);
            let w = F4::new(self.noise.next(), self.noise.next(), 0.0, 0.0);
            let o = self.svf.run(w);
            let y = if left { o.lp } else { o.hp }.to_array();
            Self::put(x, i, ch, l + level * y[0], r + level * y[1]);
        }
    }

    /// The sound crushed (fewer bits, a lower sample rate held), more the farther and the higher the parameter;
    /// right also through a high-pass going up (20 Hz to 2 kHz).
    fn crush(&mut self, x: &mut [f32], n: usize, ch: usize, amount: f64, left: bool, p: f64) {
        let depth = amount * (0.25 + 0.75 * p);
        let bits = 16.0 - depth * 12.0;
        let steps = 2f64.powf(bits) as f32;
        let hold = 1.0 + depth * 24.0;
        if !left {
            let f = 20.0 * (2000.0f64 / 20.0).powf(amount);
            self.svf.set([f; 4], [0.707; 4], self.rate);
        }
        let wet = (4.0 * amount).min(1.0) as f32;
        for i in 0..n {
            let (l, r) = Self::frame(x, i, ch);
            self.crush_count += 1.0;
            if self.crush_count >= hold {
                self.crush_count -= hold;
                self.crush_hold = ((l * steps).round() / steps, (r * steps).round() / steps);
            }
            let (cl, cr) = self.crush_hold;
            let (yl, yr) = if left { (cl, cr) } else {
                let h = self.svf.run(F4::new(cl, cr, 0.0, 0.0)).hp.to_array();
                (h[0], h[1])
            };
            Self::put(x, i, ch, l + wet * (yl - l), r + wet * (yr - r));
        }
    }

    /// An echo (3/4 of a beat) whose repeats go through a filter: left a band-pass on the middle range (1 kHz),
    /// right a high-pass going up (1.5 to 5 kHz); the knob its send, the parameter its feedback (0.3 - 0.95); its
    /// tail rings out after the knob went back.
    #[allow(clippy::too_many_arguments)]
    fn dub_echo(&mut self, x: &mut [f32], n: usize, ch: usize, amount: f64, left: bool, p: f64, bpm: f64) {
        let sec = if bpm.is_finite() && bpm > 20.0 { 45.0 / bpm } else { ECHO_SEC };
        let delay = (sec * self.rate).min((ECHO_MAX_SEC * self.rate) - 8.0);
        let feedback = (0.3 + p * 0.65) as f32;
        let send = amount as f32;
        let (f, q) = if left { (1000.0, 0.8) } else { (1500.0 * (5000.0f64 / 1500.0).powf(amount), 0.707) };
        self.fb_svf.set([f; 4], [q; 4], self.rate);
        let norm = F4::splat(if left { (1.0 / q) as f32 } else { 1.0 });
        let mut peak = 0f32;
        for i in 0..n {
            let (l, r) = Self::frame(x, i, ch);
            let (el, er) = self.echo.read(delay);
            let o = self.fb_svf.run(F4::new(send * l + feedback * el, send * r + feedback * er, 0.0, 0.0));
            let w = if left { norm * o.bp } else { o.hp }.to_array();
            self.echo.write(w[0], w[1]);
            peak = peak.max(el.abs()).max(er.abs()).max(w[0].abs()).max(w[1].abs());
            Self::put(x, i, ch, l + el, r + er);
        }
        self.heard(peak, n, delay as usize + 2);
    }

    /// A reverb, the knob its send; left its wet low-passed (the middle and low range), right high-passed (the
    /// middle and high); the parameter its feedback (decay); its tail rings out after the knob went back.
    fn space(&mut self, x: &mut [f32], n: usize, ch: usize, amount: f64, left: bool, p: f64) {
        self.reverb.feedback = (0.55 + p * 0.4) as f32;
        self.reverb.damp = 0.35;
        let f = if left { 8000.0 * (800.0f64 / 8000.0).powf(amount) } else { 80.0 * (1500.0f64 / 80.0).powf(amount) };
        self.svf.set([f; 4], [0.707; 4], self.rate);
        let send = amount as f32;
        let mut peak = 0f32;
        for i in 0..n {
            let (l, r) = Self::frame(x, i, ch);
            let (wl, wr) = self.reverb.run(send * l, send * r, 0.01 * self.rate);
            let o = self.svf.run(F4::new(wl, wr, 0.0, 0.0));
            let y = if left { o.lp } else { o.hp }.to_array();
            peak = peak.max(y[0].abs()).max(y[1].abs()).max(send * l.abs()).max(send * r.abs());
            Self::put(x, i, ch, l + y[0], r + y[1]);
        }
        self.heard(peak, n, (SPACE_SPAN_SEC * self.rate) as usize);
    }

    /// JET: a flanger's comb turned by hand. The knob moves its delay from 6 ms at the middle down to 0.12 ms at either
    /// end (evenly in pitch: the comb's teeth climb as it turns, a jet's whoosh); right with its repeats fed back in
    /// phase (a ringing jet), left inverted (hollow, the odd teeth); the parameter its feedback (saturated in the loop,
    /// so it never runs away), eased in from the middle.
    fn jet(&mut self, x: &mut [f32], n: usize, ch: usize, amount: f64, left: bool, p: f64) {
        let ms = JET_MS.0 * (JET_MS.1 / JET_MS.0).powf(amount);
        let delay = ms * 0.001 * self.rate;
        let fb = (JET_FEEDBACK.0 + p * (JET_FEEDBACK.1 - JET_FEEDBACK.0)) as f32;
        let g = F4::splat(if left { -fb } else { fb });
        let wet = (4.0 * amount).min(1.0) as f32;
        let mut peak = 0f32;
        for i in 0..n {
            let (l, r) = Self::frame(x, i, ch);
            let dry = F4::new(l, r, 0.0, 0.0);
            let (el, er) = self.echo.read(delay);
            let echo = F4::new(el, er, 0.0, 0.0);
            let into = (dry + g * echo.soft_sat()).to_array();
            self.echo.write(into[0], into[1]);
            let y = ((dry + echo) * F4::splat(0.6)).to_array();
            peak = peak.max(el.abs()).max(er.abs());
            Self::put(x, i, ch, l + wet * (y[0] - l), r + wet * (y[1] - r));
        }
        self.heard(peak, n, (JET_MS.0 * 0.001 * self.rate) as usize + 64);
    }

    /// An echo's / reverb's block: ``peak`` the most that went into or came out of its line, ``span`` the frames the
    /// line holds a sound.
    fn heard(&mut self, peak: f32, n: usize, span: usize) {
        self.quiet = if peak >= TAIL_FLOOR { 0 } else { self.quiet.saturating_add(n) };
        self.span = span;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(n: usize) -> Vec<f32> {
        (0..n).flat_map(|i| {
            let v = (2.0 * std::f64::consts::PI * 440.0 * i as f64 / 48000.0).sin() as f32 * 0.5;
            [v, v]
        }).collect()
    }

    #[test]
    fn middle_passes_untouched() {
        for kind in 0..=ColorType::COUNT {
            let mut fx = ColorFx::new(48000.0, 7);
            let x = tone(4800);
            let mut y = x.clone();
            fx.process(&mut y, 4800, 2, ColorType::from_code(kind), 0.5, 0.5, 120.0);
            assert_eq!(x, y, "kind {kind}");
        }
    }

    #[test]
    fn every_kind_changes_and_stays_finite() {
        for kind in 0..ColorType::COUNT {
            for knob in [0.05, 0.95] {
                let mut fx = ColorFx::new(48000.0, 7);
                let x = tone(48000);
                let mut y = x.clone();
                for block in y.chunks_mut(512) {
                    let frames = block.len() / 2;
                    fx.process(block, frames, 2, ColorType::from_code(kind), knob, 0.5, 120.0);
                }
                assert!(y.iter().all(|v| v.is_finite()), "kind {kind}");
                let diff: f32 = x.iter().zip(&y).map(|(a, b)| (a - b).abs()).sum();
                assert!(diff > 1.0, "kind {kind} knob {knob}: unchanged");
            }
        }
    }

    #[test]
    fn off_lets_the_tail_ring_out() {
        let mut fx = ColorFx::new(48000.0, 7);
        let mut y = tone(9600);
        fx.process(&mut y, 9600, 2, ColorType::DubEcho, 0.9, 0.2, 120.0);
        // its repeats 3/4 of a beat apart (375 ms at 120 BPM)
        let mut silence = vec![0f32; 2 * 24000];
        fx.process(&mut silence, 24000, 2, ColorType::Off, 0.9, 0.2, 120.0);
        assert!(silence.iter().any(|v| v.abs() > 1e-3), "no tail");
        for _ in 0..200 {
            let mut s = vec![0f32; 2 * 4800];
            fx.process(&mut s, 4800, 2, ColorType::Off, 0.9, 0.2, 120.0);
        }
        assert!(fx.idle());
    }
}
