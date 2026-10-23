//! Beat FX, as a DJM-A9's: one effect at a time on a channel, a crossfader side or the master (mixer.rs puts it
//! there), its time a number of beats at the BPM of that channel's deck (the master's: the first deck playing) or the
//! BPM tapped, or a time set on the TIME knob; its LEVEL / DEPTH knob, the X-PAD, the bands it works on (FX
//! FREQUENCY LOW / MID / HI: the others pass by it), on and off. DELAY, ECHO, PING PONG, SPIRAL, HELIX, REVERB,
//! FLANGER, PHASER, FILTER, TRIPLET FILTER, TRANS, ROLL, TRIPLET ROLL, MOBIUS (the A9 manual's pages 52 - 56: each
//! type's beat values, time range and what LEVEL / DEPTH does). Switched off, the echoes and the reverb ring out
//! (their send closed); the rest stop at once. Off with nothing ringing, the samples pass bit for bit.
//!
//! The filters run a stereo pair per SIMD pass (simd::F4), the reverb's four lines a lane each (fxkit::Fdn4),
//! MOBIUS's 48 oscillators four a pass (rotating phasors).

use crate::filters::{Biquad4, Coeffs, Svf4};
use crate::fxkit::{Delay, Fdn4, Lfo, Smooth};
use crate::mixer::Isolator;
use crate::simd::F4;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BeatType {
    Delay = 0,
    Echo = 1,
    PingPong = 2,
    Spiral = 3,
    Helix = 4,
    Reverb = 5,
    Flanger = 6,
    Phaser = 7,
    Filter = 8,
    TripletFilter = 9,
    Trans = 10,
    Roll = 11,
    TripletRoll = 12,
    Mobius = 13,
}

const STRAIGHT: [f64; 12] = [1.0 / 16.0, 1.0 / 8.0, 1.0 / 4.0, 1.0 / 3.0, 1.0 / 2.0, 2.0 / 3.0, 3.0 / 4.0, 1.0, 2.0, 4.0,
                             8.0, 16.0];
const LOOPS: [f64; 9] = [1.0 / 16.0, 1.0 / 8.0, 1.0 / 4.0, 1.0 / 2.0, 1.0, 2.0, 4.0, 8.0, 16.0];
const REVERB: [f64; 11] = [1.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0];
const CYCLES: [f64; 11] = [1.0 / 16.0, 1.0 / 8.0, 1.0 / 4.0, 1.0 / 2.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0];
const TRIPLETS: [f64; 7] = [1.0 / 12.0, 1.0 / 6.0, 1.0 / 3.0, 2.0 / 3.0, 4.0 / 3.0, 8.0 / 3.0, 12.0 / 3.0];
const MOBIUS: [f64; 22] = [1.0 / 16.0, 1.0 / 8.0, 1.0 / 4.0, 1.0 / 2.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, -64.0,
                           -32.0, -16.0, -8.0, -4.0, -2.0, -1.0, -1.0 / 2.0, -1.0 / 4.0, -1.0 / 8.0, -1.0 / 16.0];

impl BeatType {
    pub const COUNT: u8 = 14;

    /// Its place in a channel's strip, Beat FX put on a channel (the A9's block diagram has three, page 93): after
    /// the fader the echoes and reverbs (the manual's notes on DELAY, ECHO, PING PONG, SPIRAL, REVERB: not heard in
    /// the CUE, and their sound left with the fader down); before the EQ the ones that record and repeat (HELIX,
    /// ROLL, TRIPLET ROLL: the EQ then shapes what repeats); the others after the EQ, before the CUE and the fader.
    pub fn slot(self) -> crate::colorfx::Slot {
        use crate::colorfx::Slot;
        use BeatType::*;
        match self {
            Delay | Echo | PingPong | Spiral | Reverb => Slot::PostFader,
            Helix | Roll | TripletRoll => Slot::PreEq,
            _ => Slot::PostEq,
        }
    }

    pub fn from_code(c: u8) -> BeatType {
        use BeatType::*;
        [Delay, Echo, PingPong, Spiral, Helix, Reverb, Flanger, Phaser, Filter, TripletFilter, Trans, Roll, TripletRoll,
         Mobius][(c.min(Self::COUNT - 1)) as usize]
    }

    /// Its sound goes on after it is switched off (the repeats, the reverb's tail).
    fn rings(self) -> bool {
        matches!(self, BeatType::Echo | BeatType::PingPong | BeatType::Spiral | BeatType::Reverb | BeatType::Delay)
    }

    /// The values its BEAT buttons step through, in their order (REVERB's: its reverberation level, %).
    pub fn beats(self) -> &'static [f64] {
        use BeatType::*;
        match self {
            Delay | Echo | PingPong | Spiral => &STRAIGHT,
            Helix | Trans | Roll => &LOOPS,
            Reverb => &REVERB,
            Flanger | Phaser | Filter => &CYCLES,
            TripletFilter | TripletRoll => &TRIPLETS,
            Mobius => &MOBIUS,
        }
    }

    /// Its BEAT value when picked (the manual's defaults).
    pub fn default_beat(self) -> f64 {
        use BeatType::*;
        match self {
            Reverb => 50.0,
            Flanger | Phaser | Filter | Mobius => 4.0,
            TripletFilter => 4.0 / 3.0,
            TripletRoll => 1.0 / 3.0,
            _ => 1.0,
        }
    }

    /// The TIME knob's range (ms; MOBIUS's either sign).
    pub fn time_range(self) -> (f64, f64) {
        use BeatType::*;
        match self {
            Delay | Echo => (1.0, 4000.0),
            Flanger | Phaser | Filter | TripletFilter | Mobius => (10.0, 32000.0),
            Trans => (10.0, 16000.0),
            _ => (10.0, 4000.0),
        }
    }

    /// ``beats`` held to the nearest of its values (REVERB's: 1 .. 100 as they are).
    pub fn snap(self, beats: f64) -> f64 {
        if self == BeatType::Reverb {
            return beats.clamp(1.0, 100.0);
        }
        let list = self.beats();
        let near = |v: f64| (v.abs().ln() - beats.abs().max(1e-6).ln()).abs() + if v.signum() == beats.signum() { 0.0 } else { 100.0 };
        *list.iter().min_by(|a, b| near(**a).total_cmp(&near(**b))).unwrap()
    }
}

/// What Beat FX is set to (one reading of the mixer's settings).
#[derive(Clone, Copy, Debug)]
pub struct BeatSet {
    pub kind: BeatType,
    pub on: bool,
    pub beats: f64,   // BEAT (REVERB: its level %; MOBIUS: below 0 falling)
    pub time_ms: f64, // TIME: its time (ms) in place of the beats', 0 by the beats
    pub bpm: f64,     // the tempo it plays at (0: unknown, 120 assumed)
    pub level: f64,   // LEVEL / DEPTH 0 .. 1
    pub bands: u8,    // FX FREQUENCY: 1 LOW, 2 MID, 4 HI
    pub xpad: f64,    // the X-PAD where it does something of its own (MOBIUS's waveform, REVERB's filter): 0 .. 1,
                      // below 0 untouched
}

impl Default for BeatSet {
    fn default() -> BeatSet {
        BeatSet { kind: BeatType::Echo, on: false, beats: 1.0, time_ms: 0.0, bpm: 0.0, level: 0.5, bands: ALL_BANDS,
                  xpad: -1.0 }
    }
}

pub const ALL_BANDS: u8 = 7;
const MAX_SEC: f64 = 16.0;
const TAIL_FLOOR: f32 = 1e-5;
const PHASER_STAGES: usize = 4;
// SPIRAL: its diffusers (s, gain) and how fast its time glides to a new one
const SPIRAL_DIFFUSE: [(f64, f32); 2] = [(0.0047, 0.6), (0.0131, 0.55)];
const SPIRAL_GLIDE_SEC: f64 = 0.15;
// REVERB: its lines hold a sound this long at most (a quieter block than TAIL_FLOOR that long: the tail is gone)
const REVERB_SPAN_SEC: f64 = 0.25;
// FLANGER: the delay's sweep (ms, short .. long), its feedback, and the output's gain (the comb's peaks ring at up to
// 1 / (1 - feedback): brought back to about the dry loudness)
const FLANGER_MS: (f64, f64) = (0.12, 6.0);
const FLANGER_FEEDBACK: f32 = 0.88;
const FLANGER_GAIN: f32 = 0.4;
// MOBIUS: a Shepard tone, PARTIALS octaves from MOBIUS_LOW_HZ, each HARMONICS deep, under a bell over the octaves
const PARTIALS: usize = 8;
const HARMONICS: usize = 6;
const OSC: usize = PARTIALS * HARMONICS / 4; // F4s
const MOBIUS_LOW_HZ: f64 = 40.0;
const MOBIUS_BELL_OCTAVES: f64 = 1.4;
const MOBIUS_GAIN: f64 = 0.2;
// the band split's fade back to the whole sound when every band is picked again (frames: ~ 50 ms)
const SPLIT_FADE_SEC: f64 = 0.05;

pub struct BeatFx {
    rate: f64,
    kind: BeatType,
    was_on: bool,
    wet: Smooth,
    dry: Smooth,
    send: Smooth,
    delay: Delay,
    reverb: Fdn4,
    lfo: Lfo,
    svf: Svf4,
    wet_svf: Svf4, // REVERB's X-PAD filter
    phaser: [Biquad4; PHASER_STAGES], // lanes: left, right (2, 3 unused)
    fb: (f32, f32),
    // ROLL / HELIX: the loop captured when switched on (its length in frames, the place read in it)
    loop_len: usize,
    loop_pos: usize,
    loop_fill: usize,
    loop_buf: Vec<f32>,
    // MOBIUS: the oscillators' phasors (partial-major, harmonic-minor, four a F4), its sweep's place (0 .. 1 an
    // octave)
    osc_re: [F4; OSC],
    osc_im: [F4; OSC],
    sweep: f64,
    // SPIRAL: its delay time as it glides (a time changed bends the pitch), the diffusers in its loop
    glide: Smooth,
    diffuse: [Delay; 2],
    // a ringing type's line: frames since anything above TAIL_FLOOR went in or came out, and how long the line holds
    // a sound (silent that long: empty)
    quiet: usize,
    span: usize,
    // FX FREQUENCY: the picked bands (fed to the effect) and the rest (passed by), the rest's frames
    split_sel: Isolator,
    split_rest: Isolator,
    rest: Vec<f32>,
    split_left: usize, // frames the split keeps running after every band is picked again
}

impl BeatFx {
    /// At ``rate``, for blocks of ``frames`` at most.
    pub fn new(rate: f64, frames: usize) -> BeatFx {
        let len = (MAX_SEC * rate) as usize + 8;
        BeatFx {
            rate,
            kind: BeatType::Echo,
            was_on: false,
            wet: Smooth::new(0.0, 0.01, rate),
            dry: Smooth::new(1.0, 0.01, rate),
            send: Smooth::new(0.0, 0.005, rate),
            delay: Delay::new(len),
            reverb: Fdn4::new(rate, 1.6),
            lfo: Lfo::default(),
            svf: Svf4::default(),
            wet_svf: Svf4::default(),
            phaser: [Biquad4::default(); PHASER_STAGES],
            fb: (0.0, 0.0),
            loop_len: 0,
            loop_pos: 0,
            loop_fill: 0,
            loop_buf: vec![0.0; len * 2],
            osc_re: [F4::splat(1.0); OSC],
            osc_im: [F4::default(); OSC],
            sweep: 0.0,
            glide: Smooth::new(0.0, SPIRAL_GLIDE_SEC, rate),
            diffuse: SPIRAL_DIFFUSE.map(|(sec, _)| Delay::new((sec * rate) as usize + 4)),
            quiet: usize::MAX / 2,
            span: 0,
            split_sel: Isolator::new(rate),
            split_rest: Isolator::new(rate),
            rest: vec![0.0; frames * 2],
            split_left: 0,
        }
    }

    /// Nothing to do: off, settled, nothing ringing.
    pub fn idle(&self) -> bool {
        // a ringing type keeps its wet level after it is off (its tail heard): idle once its line is empty
        !self.was_on && self.send.value == 0.0 && self.quiet >= self.span && self.dry.value == 1.0
            && (self.kind.rings() || self.wet.value == 0.0)
    }

    fn reset(&mut self) {
        self.delay.clear();
        self.reverb.clear();
        self.svf.reset();
        self.wet_svf.reset();
        self.phaser.iter_mut().for_each(|b| b.reset());
        self.fb = (0.0, 0.0);
        self.loop_len = 0;
        self.osc_re = [F4::splat(1.0); OSC];
        self.osc_im = [F4::default(); OSC];
        self.glide.value = 0.0;
        self.diffuse.iter_mut().for_each(Delay::clear);
        self.quiet = usize::MAX / 2;
        self.span = 0;
    }

    /// A ringing type's block: ``peak`` the most that went into or came out of its line, ``span`` the frames the line
    /// holds a sound.
    fn heard(&mut self, peak: f32, n: usize, span: usize) {
        self.quiet = if peak >= TAIL_FLOOR { 0 } else { self.quiet.saturating_add(n) };
        self.span = span;
    }

    /// Its time in frames (MOBIUS's below 0 falling): the TIME knob's when set, else the beats' at the BPM.
    fn period(&self, s: &BeatSet) -> f64 {
        let (lo, hi) = s.kind.time_range();
        let ms = if s.time_ms != 0.0 && s.kind != BeatType::Reverb {
            s.time_ms.signum() * s.time_ms.abs().clamp(lo, hi)
        } else {
            let bpm = if s.bpm.is_finite() && s.bpm > 20.0 { s.bpm } else { 120.0 };
            60000.0 / bpm * s.kind.snap(s.beats)
        };
        let frames = ms * 0.001 * self.rate;
        frames.signum() * frames.abs().max(16.0)
    }

    /// ``x``: ``n`` stereo frames, in place.
    pub fn process(&mut self, x: &mut [f32], n: usize, s: &BeatSet) {
        if !s.on && self.idle() {
            self.split_left = 0;
            return;
        }
        if s.kind != self.kind {
            self.kind = s.kind;
            self.reset();
        }
        if s.on && !self.was_on {
            self.loop_len = 0; // ROLL / HELIX capture anew
            self.loop_fill = 0;
            self.loop_pos = 0;
        }
        self.was_on = s.on;
        // FX FREQUENCY: the bands not picked pass by the effect (the split's crossovers, so they sum back flat)
        let split = s.bands & ALL_BANDS != ALL_BANDS || self.split_left > 0;
        if split {
            let pick = [1u8, 2, 4].map(|b| if s.bands & b != 0 { 1.0 } else { 0.0 });
            let rest = &mut self.rest[..2 * n];
            rest.copy_from_slice(&x[..2 * n]);
            self.split_rest.process(rest, n, 2, pick.map(|p| 1.0 - p));
            self.split_sel.process(x, n, 2, pick);
            self.split_left = if s.bands & ALL_BANDS != ALL_BANDS {
                (SPLIT_FADE_SEC * self.rate) as usize
            } else {
                self.split_left.saturating_sub(n)
            };
        }
        let d = self.period(s);
        let level = s.level.clamp(0.0, 1.0);
        let rings = self.kind.rings();
        // LEVEL / DEPTH: a balance (the original down past the middle, the effect up to it) or the effect's depth
        let balance = !matches!(self.kind, BeatType::Flanger | BeatType::Phaser | BeatType::Filter
            | BeatType::TripletFilter | BeatType::Helix | BeatType::Mobius);
        let (dry_to, wet_to) = if balance {
            ((2.0 * (1.0 - level)).min(1.0), (2.0 * level).min(1.0))
        } else if self.kind == BeatType::Helix {
            (1.0, (4.0 * level).min(1.0))
        } else {
            (1.0, level)
        };
        // off: an echo / reverb's send closes (its tail rings out), the others' wet fades out; the original back up
        let wet = self.wet.step(if s.on || rings { wet_to } else { 0.0 }, n) as f32;
        let dry = self.dry.step(if s.on { dry_to } else { 1.0 }, n) as f32;
        let send = self.send.step(if s.on { 1.0 } else { 0.0 }, n) as f32;
        let dl = d.abs();
        match self.kind {
            BeatType::Delay => self.echo(x, n, dl, dry, wet, send, 0.0, false),
            BeatType::Echo => self.echo(x, n, dl, dry, wet, send, 0.55, false),
            BeatType::PingPong => self.echo(x, n, dl, dry, wet, send, 0.6, true),
            BeatType::Spiral => self.spiral(x, n, dl, dry, wet, send, 0.4 + 0.5 * level as f32),
            BeatType::Helix => self.helix(x, n, dl, wet, send, level),
            BeatType::Roll | BeatType::TripletRoll => self.roll(x, n, dl, dry, wet, send),
            BeatType::Reverb => self.reverb(x, n, s.kind.snap(s.beats), dry, wet, send, s.xpad),
            BeatType::Flanger => self.flanger(x, n, dl, wet),
            BeatType::Phaser => self.phaser(x, n, dl, wet),
            BeatType::Filter | BeatType::TripletFilter => self.filter(x, n, dl, wet),
            BeatType::Trans => self.trans(x, n, dl, dry, wet, level),
            BeatType::Mobius => self.mobius(x, n, d, wet, s.xpad),
        }
        if split {
            crate::vk::add_in(&mut x[..2 * n], &self.rest[..2 * n]);
        }
        if !s.on && !rings && self.wet.value == 0.0 {
            self.reset();
        }
    }

    /// DELAY (one repeat), ECHO (repeats fading), PING PONG (repeats alternating sides).
    #[allow(clippy::too_many_arguments)]
    fn echo(&mut self, x: &mut [f32], n: usize, d: f64, dry: f32, wet: f32, send: f32, feedback: f32, ping: bool) {
        let mut peak = 0f32;
        for i in 0..n {
            let (l, r) = (x[2 * i], x[2 * i + 1]);
            let (el, er) = self.delay.read(d);
            let (wl, wr) = if ping { (send * 0.5 * (l + r) + feedback * er, feedback * el) } else {
                (send * l + feedback * el, send * r + feedback * er)
            };
            self.delay.write(wl, wr);
            peak = peak.max(el.abs()).max(er.abs()).max(wl.abs()).max(wr.abs());
            x[2 * i] = dry * l + wet * el;
            x[2 * i + 1] = dry * r + wet * er;
        }
        self.heard(peak, n, d as usize + 2);
    }

    /// SPIRAL: reverberation from a delay (the manual's): its repeats fed back through two diffusers (Schroeder
    /// allpasses, SPIRAL_DIFFUSE), each repeat more smeared until they are a reverb's tail; its time glides to a new
    /// one (SPIRAL_GLIDE_SEC), so a time changed bends the pitch of what is in the loop. LEVEL / DEPTH its balance
    /// and feedback.
    #[allow(clippy::too_many_arguments)]
    fn spiral(&mut self, x: &mut [f32], n: usize, d: f64, dry: f32, wet: f32, send: f32, feedback: f32) {
        if self.glide.value == 0.0 {
            self.glide.value = d; // its first time taken at once
        }
        let start = self.glide.value;
        let end = self.glide.step(d, n);
        let lens = SPIRAL_DIFFUSE.map(|(sec, _)| sec * self.rate);
        let mut peak = 0f32;
        for i in 0..n {
            let (l, r) = (x[2 * i], x[2 * i + 1]);
            let t = start + (end - start) * (i + 1) as f64 / n as f64;
            let (el, er) = self.delay.read(t);
            // the loop: through the diffusers
            let (mut fl, mut fr) = (el, er);
            for (k, (_, g)) in SPIRAL_DIFFUSE.iter().enumerate() {
                let (dl, dr) = self.diffuse[k].read(lens[k]);
                let (yl, yr) = (dl - g * fl, dr - g * fr);
                self.diffuse[k].write(fl + g * yl, fr + g * yr);
                (fl, fr) = (yl, yr);
            }
            let (wl, wr) = (send * l + feedback * fl, send * r + feedback * fr);
            self.delay.write(wl, wr);
            peak = peak.max(el.abs()).max(er.abs()).max(wl.abs()).max(wr.abs());
            x[2 * i] = dry * l + wet * el;
            x[2 * i + 1] = dry * r + wet * er;
        }
        self.heard(peak, n, end as usize + (lens[0] + lens[1]) as usize + 2);
    }

    /// HELIX: the beat length from when it was switched on recorded (heard as it is), then played over the sound
    /// in a loop, the loop keeping more of itself the higher LEVEL / DEPTH (its top: the loop held, nothing new in).
    fn helix(&mut self, x: &mut [f32], n: usize, d: f64, wet: f32, send: f32, level: f64) {
        let len = (d as usize).min(self.loop_buf.len() / 2 - 1).max(1);
        let keep = (0.5 + 0.5 * level) as f32;
        let take = (1.0 - keep) * 2.0 * send; // a whole new sound at the bottom, none at the top
        if self.loop_fill >= len && self.loop_len != len {
            self.loop_len = len;
            self.loop_pos %= len;
        }
        for i in 0..n {
            let (l, r) = (x[2 * i], x[2 * i + 1]);
            if self.loop_fill < len {
                // recording the first beat length
                if send > 0.0 {
                    self.loop_buf[2 * self.loop_fill] = l;
                    self.loop_buf[2 * self.loop_fill + 1] = r;
                    self.loop_fill += 1;
                    if self.loop_fill == len {
                        self.loop_len = len;
                        self.loop_pos = 0;
                    }
                }
                continue;
            }
            let at = 2 * self.loop_pos;
            let (ll, lr) = (self.loop_buf[at], self.loop_buf[at + 1]);
            self.loop_buf[at] = keep * ll + take * l;
            self.loop_buf[at + 1] = keep * lr + take * r;
            self.loop_pos = (self.loop_pos + 1) % len;
            x[2 * i] = l + wet * ll;
            x[2 * i + 1] = r + wet * lr;
        }
    }

    /// ROLL / TRIPLET ROLL: the beat length from when it was switched on repeated, balanced with the sound.
    fn roll(&mut self, x: &mut [f32], n: usize, d: f64, dry: f32, wet: f32, send: f32) {
        let len = (d as usize).min(self.loop_buf.len() / 2 - 1).max(1);
        for i in 0..n {
            let (l, r) = (x[2 * i], x[2 * i + 1]);
            if send > 0.0 && self.loop_fill < len {
                // capturing the first beat length (heard as it is)
                self.loop_buf[2 * self.loop_fill] = l;
                self.loop_buf[2 * self.loop_fill + 1] = r;
                self.loop_fill += 1;
                self.loop_len = self.loop_fill;
                continue;
            }
            if self.loop_len == 0 {
                continue;
            }
            let at = 2 * (self.loop_pos % self.loop_len);
            let (ll, lr) = (self.loop_buf[at], self.loop_buf[at + 1]);
            self.loop_pos = (self.loop_pos + 1) % self.loop_len;
            x[2 * i] = dry * l + wet * ll;
            x[2 * i + 1] = dry * r + wet * lr;
        }
    }

    /// REVERB: its decay from the reverberation level (BEAT: 1 - 100 %); the X-PAD a high-pass on its wet (20 Hz -
    /// 8 kHz).
    #[allow(clippy::too_many_arguments)]
    fn reverb(&mut self, x: &mut [f32], n: usize, percent: f64, dry: f32, wet: f32, send: f32, xpad: f64) {
        self.reverb.feedback = (0.45 + 0.52 * percent / 100.0).min(0.97) as f32;
        self.reverb.damp = 0.3;
        let filtered = xpad >= 0.0;
        if filtered {
            let f = 20.0 * (8000.0f64 / 20.0).powf(xpad.clamp(0.0, 1.0));
            self.wet_svf.set([f; 4], [0.707; 4], self.rate);
        }
        let mut peak = 0f32;
        for i in 0..n {
            let (l, r) = (x[2 * i], x[2 * i + 1]);
            let (mut wl, mut wr) = self.reverb.run(send * l, send * r, 0.02 * self.rate);
            if filtered {
                let h = self.wet_svf.run(F4::new(wl, wr, 0.0, 0.0)).hp.to_array();
                (wl, wr) = (h[0], h[1]);
            }
            peak = peak.max(wl.abs()).max(wr.abs()).max(send * l.abs()).max(send * r.abs());
            x[2 * i] = dry * l + wet * wl;
            x[2 * i + 1] = dry * r + wet * wr;
        }
        let span = (REVERB_SPAN_SEC * self.rate) as usize;
        self.heard(peak, n, span);
    }

    /// FLANGER: a jet flanger. A short delay swept once per beat length from 0.12 to 6 ms, evenly in pitch (the
    /// delay exponential in the sweep), fed back strongly (FLANGER_FEEDBACK, saturated in the loop so it never runs
    /// away): the comb's peaks ring, and at the short end they climb past 4 kHz, a jet's whoosh.
    fn flanger(&mut self, x: &mut [f32], n: usize, d: f64, wet: f32) {
        let g = F4::splat(FLANGER_FEEDBACK);
        let gain = F4::splat(FLANGER_GAIN);
        for i in 0..n {
            let dry = F4::new(x[2 * i], x[2 * i + 1], 0.0, 0.0);
            self.lfo.advance(self.rate / d, self.rate);
            let ms = FLANGER_MS.0 * (FLANGER_MS.1 / FLANGER_MS.0).powf(0.5 * (1.0 - self.lfo.triangle()));
            let (el, er) = self.delay.read(ms * 0.001 * self.rate);
            let echo = F4::new(el, er, 0.0, 0.0);
            let into = (dry + g * echo.soft_sat()).to_array();
            self.delay.write(into[0], into[1]);
            let y = ((dry + echo) * gain).to_array();
            x[2 * i] += wet * (y[0] - x[2 * i]);
            x[2 * i + 1] += wet * (y[1] - x[2 * i + 1]);
        }
    }

    /// PHASER: four allpass stages swept once per beat length (a block at a time), the result mixed with the sound.
    fn phaser(&mut self, x: &mut [f32], n: usize, d: f64, wet: f32) {
        let ph = self.lfo.advance(self.rate / d * n as f64, self.rate);
        let f = 300.0 * (4000.0f64 / 300.0).powf(0.5 * (1.0 - (2.0 * std::f64::consts::PI * ph).cos()));
        for (k, stage) in self.phaser.iter_mut().enumerate() {
            let c = Coeffs::allpass(f * (1.0 + 0.3 * k as f64), 0.7, self.rate);
            stage.set([c; 4]);
        }
        for i in 0..n {
            let (l, r) = (x[2 * i], x[2 * i + 1]);
            let mut v = F4::new(l + 0.4 * self.fb.0, r + 0.4 * self.fb.1, 0.0, 0.0);
            for stage in self.phaser.iter_mut() {
                v = stage.run(v);
            }
            let y = v.to_array();
            self.fb = (y[0], y[1]);
            x[2 * i] = l + wet * (0.5 * (l + y[0]) - l);
            x[2 * i + 1] = r + wet * (0.5 * (r + y[1]) - r);
        }
    }

    /// FILTER / TRIPLET FILTER: a resonant low-pass swept down and back once per beat length (TRIPLET: its values
    /// thirds of beats).
    fn filter(&mut self, x: &mut [f32], n: usize, d: f64, wet: f32) {
        let ph = self.lfo.advance(self.rate / d * n as f64, self.rate);
        let f = 200.0 * (12000.0f64 / 200.0).powf(0.5 * (1.0 + (2.0 * std::f64::consts::PI * ph).cos()));
        self.svf.set([f; 4], [2.5; 4], self.rate);
        for i in 0..n {
            let (l, r) = (x[2 * i], x[2 * i + 1]);
            let y = self.svf.run(F4::new(l, r, 0.0, 0.0)).lp.to_array();
            x[2 * i] = l + wet * (y[0] - l);
            x[2 * i + 1] = r + wet * (y[1] - r);
        }
    }

    /// TRANS: the sound cut once per beat length (a 2 ms ramp at each edge); LEVEL / DEPTH its balance (how deep
    /// the cut) and duty (open 75 % of the beat length at the bottom, 25 % at the top).
    fn trans(&mut self, x: &mut [f32], n: usize, d: f64, dry: f32, wet: f32, level: f64) {
        let _ = dry;
        let open = 0.75 - 0.5 * level;
        let edge = (0.002 * self.rate) / d;
        for i in 0..n {
            let ph = self.lfo.advance(self.rate / d, self.rate);
            let g = if ph < open { ((open - ph) / edge).min(ph / edge).min(1.0) } else { 0.0 } as f32;
            let k = 1.0 - wet * (1.0 - g);
            x[2 * i] *= k;
            x[2 * i + 1] *= k;
        }
    }

    /// MOBIUS: an oscillator rising (``d`` above 0) or falling for ever, an octave per period: a Shepard tone,
    /// PARTIALS octaves under a bell over log frequency, so the top fades out as the bottom fades in. Its waveform
    /// the X-PAD's (square at the left, saw in the middle (untouched), triangle at the right: each partial's first
    /// HARMONICS harmonics, band-limited by making); LEVEL / DEPTH its volume, added to the sound.
    fn mobius(&mut self, x: &mut [f32], n: usize, d: f64, wet: f32, xpad: f64) {
        // the octave's place: whole octaves passed move the phasors a partial up (down)
        self.sweep += n as f64 / d;
        while self.sweep >= 1.0 {
            self.sweep -= 1.0;
            self.shift(true);
        }
        while self.sweep < 0.0 {
            self.sweep += 1.0;
            self.shift(false);
        }
        // the harmonics' weights: square (odd, 1 / h), saw (1 / h), triangle (odd, 1 / h^2), between them by the pad
        let u = if xpad < 0.0 { 0.5 } else { xpad.clamp(0.0, 1.0) };
        let weight = |h: usize| {
            let hf = h as f64;
            let odd = if h % 2 == 1 { 1.0 } else { 0.0 };
            let (sq, saw, tri) = (odd / hf, 1.0 / hf, odd / (hf * hf));
            if u < 0.5 { sq + (saw - sq) * 2.0 * u } else { saw + (tri - saw) * (2.0 * u - 1.0) }
        };
        let mut amp = [0f32; PARTIALS * HARMONICS];
        let mut cos = [0f32; PARTIALS * HARMONICS];
        let mut sin = [0f32; PARTIALS * HARMONICS];
        let mut power = 0.0;
        for k in 0..PARTIALS {
            let place = k as f64 + self.sweep;
            let bell = (-(place - PARTIALS as f64 / 2.0).powi(2) / (2.0 * MOBIUS_BELL_OCTAVES.powi(2))).exp();
            let f0 = MOBIUS_LOW_HZ * place.exp2();
            for h in 1..=HARMONICS {
                let f = f0 * h as f64;
                let a = if f < 0.4 * self.rate { bell * weight(h) } else { 0.0 };
                let w = 2.0 * std::f64::consts::PI * f / self.rate;
                let j = k * HARMONICS + h - 1;
                amp[j] = a as f32;
                cos[j] = w.cos() as f32;
                sin[j] = w.sin() as f32;
                power += a * a;
            }
        }
        let norm = (MOBIUS_GAIN / power.max(1e-9).sqrt()) as f32 * wet;
        let a: [F4; OSC] = std::array::from_fn(|j| F4::load(&amp, 4 * j) * F4::splat(norm));
        let c: [F4; OSC] = std::array::from_fn(|j| F4::load(&cos, 4 * j));
        let s: [F4; OSC] = std::array::from_fn(|j| F4::load(&sin, 4 * j));
        for i in 0..n {
            let mut acc = F4::default();
            for j in 0..OSC {
                let (re, im) = (self.osc_re[j], self.osc_im[j]);
                self.osc_re[j] = re * c[j] - im * s[j];
                self.osc_im[j] = re * s[j] + im * c[j];
                acc = a[j].mul_add(self.osc_im[j], acc);
            }
            let v = acc.sum();
            x[2 * i] += v;
            x[2 * i + 1] += v;
        }
        // the phasors back to unit length (rounding drifts them)
        for j in 0..OSC {
            let (re, im) = (self.osc_re[j].to_array(), self.osc_im[j].to_array());
            let m: [f32; 4] = std::array::from_fn(|l| 1.0 / (re[l] * re[l] + im[l] * im[l]).sqrt().max(1e-6));
            let m = F4::from_array(m);
            self.osc_re[j] = self.osc_re[j] * m;
            self.osc_im[j] = self.osc_im[j] * m;
        }
    }

    /// The phasors moved a partial up (an octave risen: partial k is now where k + 1 was) or down; the one coming in
    /// at the end starts fresh (its level ~ 0 under the bell).
    fn shift(&mut self, up: bool) {
        let mut re = [0f32; PARTIALS * HARMONICS];
        let mut im = [0f32; PARTIALS * HARMONICS];
        for j in 0..OSC {
            self.osc_re[j].store(&mut re, 4 * j);
            self.osc_im[j].store(&mut im, 4 * j);
        }
        if up {
            re.copy_within(..(PARTIALS - 1) * HARMONICS, HARMONICS);
            im.copy_within(..(PARTIALS - 1) * HARMONICS, HARMONICS);
            re[..HARMONICS].fill(1.0);
            im[..HARMONICS].fill(0.0);
        } else {
            re.copy_within(HARMONICS.., 0);
            im.copy_within(HARMONICS.., 0);
            re[(PARTIALS - 1) * HARMONICS..].fill(1.0);
            im[(PARTIALS - 1) * HARMONICS..].fill(0.0);
        }
        for j in 0..OSC {
            self.osc_re[j] = F4::load(&re, 4 * j);
            self.osc_im[j] = F4::load(&im, 4 * j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(n: usize) -> Vec<f32> {
        (0..n).flat_map(|i| {
            let v = (2.0 * std::f64::consts::PI * 330.0 * i as f64 / 48000.0).sin() as f32 * 0.5;
            [v, v]
        }).collect()
    }

    fn set(kind: BeatType, on: bool, beats: f64, level: f64) -> BeatSet {
        BeatSet { kind, on, beats, bpm: 128.0, level, ..BeatSet::default() }
    }

    #[test]
    fn off_passes_untouched() {
        let mut fx = BeatFx::new(48000.0, 4800);
        let x = tone(4800);
        let mut y = x.clone();
        fx.process(&mut y, 4800, &set(BeatType::Echo, false, 1.0, 0.5));
        assert_eq!(x, y);
    }

    #[test]
    fn every_type_changes_and_stays_finite() {
        for code in 0..BeatType::COUNT {
            let kind = BeatType::from_code(code);
            let mut fx = BeatFx::new(48000.0, 1024);
            let x = tone(96000);
            let mut y = x.clone();
            for block in y.chunks_mut(2048) {
                let n = block.len() / 2;
                fx.process(block, n, &set(kind, true, kind.default_beat(), 0.8));
            }
            assert!(y.iter().all(|v| v.is_finite()), "type {code}");
            let diff: f32 = x.iter().zip(&y).map(|(a, b)| (a - b).abs()).sum();
            assert!(diff > 1.0, "type {code}: unchanged");
        }
    }

    #[test]
    fn echo_out_rings_then_stops() {
        let mut fx = BeatFx::new(48000.0, 48000);
        let mut y = tone(48000);
        fx.process(&mut y, 48000, &set(BeatType::Echo, true, 0.5, 0.5));
        let mut s = vec![0f32; 2 * 4800];
        fx.process(&mut s, 4800, &set(BeatType::Echo, false, 0.5, 0.5));
        assert!(s.iter().any(|v| v.abs() > 1e-3), "no echo out");
        for _ in 0..400 {
            let mut s = vec![0f32; 2 * 4800];
            fx.process(&mut s, 4800, &set(BeatType::Echo, false, 0.5, 0.5));
        }
        assert!(fx.idle());
    }

    /// Switched off between repeats, in blocks shorter than its time: the repeats still in the line are heard.
    #[test]
    fn echo_rings_out_in_short_blocks() {
        for kind in [BeatType::Echo, BeatType::Delay, BeatType::Spiral] {
            let mut fx = BeatFx::new(48000.0, 256);
            let mut out = vec![];
            for b in 0..400 {
                let mut x = vec![0f32; 512];
                if b == 4 {
                    x[0] = 1.0;
                    x[1] = 1.0;
                }
                fx.process(&mut x, 256, &BeatSet { bpm: 120.0, ..set(kind, b < 20, 1.0, 0.5) });
                out.extend_from_slice(&x);
            }
            // the click at 1024 frames, the repeat a beat (24000 frames) on, off from 5120 frames
            let at = 2 * (1024 + 24000);
            let heard = out[at - 200..at + 2000].iter().fold(0f32, |m, v| m.max(v.abs()));
            assert!(heard > 0.1, "{kind:?}: {heard}");
        }
    }

    /// HELIX switched on at the top of LEVEL / DEPTH: its first beat recorded, then held.
    #[test]
    fn helix_at_the_top_holds_its_first_beat() {
        let mut fx = BeatFx::new(48000.0, 480);
        let x = tone(96000);
        let mut y = x.clone();
        for block in y.chunks_mut(960) {
            fx.process(block, 480, &BeatSet { bpm: 120.0, ..set(BeatType::Helix, true, 1.0, 1.0) });
        }
        let loop_at = |i: usize| y[2 * i] - x[2 * i];
        let (a, b) = (48000, 48000 + 24000);
        assert!((a..a + 1000).any(|i| loop_at(i).abs() > 0.1), "no loop");
        assert!((a..a + 1000).all(|i| (loop_at(i) - loop_at(i + 24000)).abs() < 1e-6) && b < 96000);
    }

    /// SPIRAL's time changed: it glides there (no jump in the read place).
    #[test]
    fn spiral_glides_to_a_new_time() {
        let mut fx = BeatFx::new(48000.0, 480);
        let mut x = tone(48000);
        fx.process(&mut x, 48000, &BeatSet { bpm: 120.0, ..set(BeatType::Spiral, true, 1.0, 0.5) });
        let before = fx.glide.value;
        let mut x = tone(480);
        fx.process(&mut x, 480, &BeatSet { bpm: 120.0, ..set(BeatType::Spiral, true, 2.0, 0.5) });
        let after = fx.glide.value;
        assert!(after > before && after < before + 0.1 * before, "{before} -> {after}");
    }

    #[test]
    fn beats_snap_to_the_types_values() {
        assert_eq!(BeatType::Delay.snap(0.7), 2.0 / 3.0);
        assert_eq!(BeatType::Roll.snap(0.7), 1.0 / 2.0);
        assert_eq!(BeatType::TripletRoll.snap(1.0), 4.0 / 3.0);
        assert_eq!(BeatType::Mobius.snap(-3.0), -4.0);
        assert_eq!(BeatType::Reverb.snap(250.0), 100.0);
    }

    #[test]
    fn unpicked_bands_pass_by() {
        // HI only: a low tone through ECHO comes out as it went in (the split's allpass phase aside: same level)
        let low: Vec<f32> = (0..96000).flat_map(|i| {
            let v = (2.0 * std::f64::consts::PI * 60.0 * i as f64 / 48000.0).sin() as f32 * 0.5;
            [v, v]
        }).collect();
        let mut fx = BeatFx::new(48000.0, 1024);
        let mut y = low.clone();
        for block in y.chunks_mut(2048) {
            let n = block.len() / 2;
            fx.process(block, n, &BeatSet { bands: 4, ..set(BeatType::Echo, true, 1.0, 1.0) });
        }
        let rms = |v: &[f32]| (v.iter().map(|s| (s * s) as f64).sum::<f64>() / v.len() as f64).sqrt();
        let (a, b) = (rms(&low[96000..]), rms(&y[96000..]));
        assert!((b / a - 1.0).abs() < 0.05, "{a} {b}");
    }
}
