//! The deck's controls: one place for the requests of every source (the GUI's buttons, keys and faders, a MIDI
//! controller), combined here and acted on in the engine thread; the GUI shows what the deck then is (its
//! state in ipc.rs st::DECK, the play state and playhead beside it), never a state of its own.
//!
//! CUE as rekordbox's: stopped off the cue point, it sets the cue point there (on the nearest beat or track
//! edge, as rekordbox's quantize) and moves the playhead to it; stopped on it, held, it plays and goes back
//! there when let go (PLAY while it is held: it plays on); playing, it goes back to the cue point and stops.
//! A CUE point's CALL (stopped) moves the cue point there too. The lamps: PLAY lit while playing, else
//! blinking; CUE lit while playing or on the cue point, else blinking (a new cue point can be set).
//!
//! A JumpCUE's JUMP (the GUI's former logic, ui/edit_panel/jumpcue.py): playing, the jump is armed on the next
//! beats (the player jumps on the first its render head has not passed); pressed again before it, it stays as it
//! is; after a jump, a new one is offered the beats after where it landed too (tapped in a row). The jump is
//! dropped when playback stops or the playhead leaves the beats it was offered. Stopped: the playhead moves there.
//!
//! A MIDI controller's messages reach a deck through controller.rs (the bindings, the lamps, the meters); its
//! mixer channel's knobs (trim, EQ) set the deck's channel strip (mixer.rs), rendered with it.
//!
//! The jog wheel as rekordbox's: touched, the record is held and follows the platter (both its rotation messages:
//! a DDJ sends the scratch CC or, out of its vinyl mode and once the hand is off, the bend CC); let go while the
//! platter still turns (a spin, a backspin), the record follows it on until it stops, then plays (or rests) as
//! before. Not held, a rotation bends the tempo. Its mode (JOG_MODE, set_jog_vinyl): VINYL as above, CDJ the top's
//! touch ignored and any rotation a bend (as a CDJ's jog without its VINYL mode).

use std::sync::Arc;

use crate::clock;
use crate::engine::Engine;

use crate::controller::{LAMP_CUE, LAMP_PLAY, LAMP_VINYL};
use crate::device::Ring;
use crate::ipc::{Reader, act};
use crate::mixer::{self, StripParams};

const CUE_AT_SEC: f64 = 0.001; // the playhead is on the cue point this close to it
const JUMP_CANDIDATE_BEATS: usize = 8;
const JUMP_LANDING_SEC: f64 = 0.5;
const BLINK_SEC: f64 = 0.5; // a lamp blinks this long on, this long off
// the tempo fader's ranges (+-%), as rekordbox's; WIDE: 0 - 200 % (0 stops the deck)
pub const TEMPO_RANGES: [u32; 4] = [6, 10, 16, 100];
// a jog wheel: a record at 33 1/3 rpm; its messages come steadier than the mouse's, so the platter follows them
// later by less; held still this long, the record rests
const JOG_TURN_SEC: f64 = 1.8;
const JOG_DELAY_SEC: f64 = 0.02;
const JOG_REST_SEC: f64 = 0.03;
// let go while turning: followed on until no message for this long (the platter under 5 % of 33 1/3 rpm)
const JOG_STOP_SEC: f64 = 0.05;
// the pitch bend: a turn a second +10 %, eased out over BEND_TAU_SEC
const BEND_PER_TURN: f64 = 0.1;
const BEND_TAU_SEC: f64 = 0.08;
const BEND_MAX: f64 = 0.5;
// a pitch bend button held: the tempo bent this much (a CDJ's PITCH BEND -/+)
const BUTTON_BEND: f64 = 0.04;
const SEARCH_SPEED: f64 = 16.0; // the jog's search: this many times the record's speed

pub struct Deck {
    rate: f64,
    volume: f64,
    cue: f64,
    previewing: bool,
    latched: bool,
    beats: Vec<f64>,
    downbeats: Vec<f64>,
    cue_points: Vec<f64>,
    jumpcues: [f64; 26],
    armed: Option<(f64, Vec<(f64, f64)>)>,
    landing: Option<(f64, f64)>,
    jumped_seen: u64,
    pub tempo_range: u32,
    pub jog_ticks: u32,
    jog_vinyl: bool, // VINYL mode: the top's touch holds the record (CDJ: a bend only)
    // the mixer channel's knobs (0 - 1, the middle 0 dB / off): trim, EQ low / mid / high, COLOR
    trim_knob: f64,
    eq_knobs: [f64; 3],
    color_knob: f64,
    jog_held: bool,     // the record held: the hand on the platter, or the platter let go of still turning
    jog_spinning: bool, // let go of while it turned: held until it stops
    jog_last: f64,
    jog_rested: bool,
    bend: f64,
    button_bend: f64,
    bend_ticks: f64,
    search_ticks: f64,
    last_tick: f64,
}

impl Deck {
    pub fn new(rate: f64) -> Deck {
        Deck {
            rate,
            volume: 1.0,
            cue: 0.0,
            previewing: false,
            latched: false,
            beats: vec![],
            downbeats: vec![],
            cue_points: vec![],
            jumpcues: [f64::NAN; 26],
            armed: None,
            landing: None,
            jumped_seen: 0,
            tempo_range: 100,
            jog_ticks: 720,
            jog_vinyl: true,
            trim_knob: 0.5,
            eq_knobs: [0.5; 3],
            color_knob: 0.5,
            jog_held: false,
            jog_spinning: false,
            jog_last: 0.0,
            jog_rested: true,
            bend: 0.0,
            button_bend: 0.0,
            bend_ticks: 0.0,
            search_ticks: 0.0,
            last_tick: clock::now(),
        }
    }

    // The GUI's requests
    /// A track loaded (or none): its cue point at the start, its marks to come.
    pub fn new_track(&mut self) {
        self.cue = 0.0;
        self.previewing = false;
        self.latched = false;
        self.beats.clear();
        self.downbeats.clear();
        self.cue_points.clear();
        self.jumpcues = [f64::NAN; 26];
        self.armed = None;
        self.landing = None;
        self.jog_held = false;
        self.jog_spinning = false;
    }

    pub fn set_marks(&mut self, payload: &[u8]) {
        let mut r = Reader::new(payload);
        let n = r.u32().unwrap_or(0) as u64;
        let n = r.count(n, 8 + 1);
        let beats: Vec<f64> = (0..n).filter_map(|_| r.f64()).collect();
        let down: Vec<bool> = (0..n).filter_map(|_| r.u8()).map(|v| v != 0).collect();
        let m = r.u32().unwrap_or(0) as u64;
        let m = r.count(m, 8);
        let mut cues: Vec<f64> = (0..m).filter_map(|_| r.f64()).filter(|v| v.is_finite()).collect();
        cues.sort_by(f64::total_cmp);
        let mut jumpcues = [f64::NAN; 26];
        for slot in jumpcues.iter_mut() {
            *slot = r.f64().unwrap_or(f64::NAN);
        }
        self.downbeats = beats.iter().zip(down.iter()).filter(|(_, d)| **d).map(|(b, _)| *b).collect();
        self.beats = beats;
        self.cue_points = cues;
        self.jumpcues = jumpcues;
    }

    pub fn set_volume(&mut self, e: &mut Engine, position: f64) {
        self.volume = if position.is_finite() { position.clamp(0.0, 1.0) } else { 1.0 };
        self.apply_volume(e);
    }

    /// The channel fader's position to the mixer (its gain on the mixer's fader curve, mixer::fader_gain; the
    /// channel's TRIM knob sets its level before it).
    fn apply_volume(&self, e: &mut Engine) {
        e.strip.set_fader(self.volume);
    }

    /// The tempo the deck plays at (BPM, the beats around the playhead at its speed; 0: no beats), for its channel's
    /// Sound Color FX (DUB ECHO's time).
    pub fn publish_bpm(&self, e: &Engine) {
        let at = e.playhead_frame();
        let i = self.beats.partition_point(|&b| b <= at).clamp(1, self.beats.len().max(1)) ;
        let bpm = if self.beats.len() >= 2 {
            let i = i.min(self.beats.len() - 1);
            let interval = self.beats[i] - self.beats[i - 1];
            if interval > 0.0 { 60.0 * self.rate / interval * e.factor() } else { 0.0 }
        } else {
            0.0
        };
        e.strip.set_bpm(bpm);
    }

    pub fn set_jog_ticks(&mut self, jog_ticks: u32) {
        self.jog_ticks = jog_ticks.max(1);
    }

    /// The jog wheel's mode: VINYL (``on``: the top's touch holds the record) or CDJ (a bend only; a record held
    /// let go).
    pub fn set_jog_vinyl(&mut self, e: &mut Engine, on: bool) {
        self.jog_vinyl = on;
        if !on && self.jog_held {
            self.release_jog(e);
        }
    }

    #[cfg(test)]
    pub fn jog_vinyl(&self) -> bool {
        self.jog_vinyl
    }

    #[cfg(test)]
    pub fn jog_held(&self) -> bool {
        self.jog_held
    }

    /// The tempo fader's range (TEMPO_RANGES): the tempo kept, inside the new range.
    pub fn set_tempo_range(&mut self, e: &mut Engine, range: u32) {
        self.tempo_range = if TEMPO_RANGES.contains(&range) { range } else { 100 };
        let f = self.clamp_tempo(e.factor());
        if f != e.factor() {
            e.set_tempo(f);
        }
    }

    fn clamp_tempo(&self, f: f64) -> f64 {
        let r = self.tempo_range as f64 / 100.0;
        if f.is_finite() { f.clamp((1.0 - r).max(0.0), 1.0 + r) } else { 1.0 }
    }

    /// The tempo fader moved (the GUI's): inside the range.
    pub fn set_tempo(&mut self, e: &mut Engine, f: f64) {
        e.set_tempo(self.clamp_tempo(f));
    }

    /// A deck action (act) pressed or let go.
    pub fn action(&mut self, e: &mut Engine, code: u32, pressed: bool) {
        if code == act::CUE {
            if pressed { self.cue_down(e) } else { self.cue_up(e) }
            return;
        }
        if code == act::BEND_DOWN || code == act::BEND_UP {
            // a pitch bend button: the tempo bent while it is held
            let bend = if code == act::BEND_UP { BUTTON_BEND } else { -BUTTON_BEND };
            self.button_bend = if pressed { bend } else { 0.0 };
            return;
        }
        if !pressed {
            return;
        }
        match code {
            act::PLAY_PAUSE => {
                if self.previewing {
                    self.latched = true; // played on when CUE is let go
                } else if e.playing {
                    e.pause();
                } else {
                    e.play();
                }
            }
            act::STOP => {
                self.previewing = false;
                e.pause();
                e.seek(0.0);
            }
            act::TRACK_START => e.seek(0.0), // playing on, as rekordbox's SHIFT + CUE
            c if (act::BEAT_JUMP..act::BEAT_JUMP + 2 * act::BEAT_JUMP_SIZES.len() as u32).contains(&c) => {
                let k = (c - act::BEAT_JUMP) as usize;
                let beats = act::BEAT_JUMP_SIZES[k / 2];
                self.step_beats(e, if k % 2 == 1 { beats } else { -beats }, false);
            }
            act::BEAT_BACK => self.step_beats(e, -1, false),
            act::BEAT_FORWARD => self.step_beats(e, 1, false),
            act::BAR_BACK => self.step_beats(e, -1, true),
            act::BAR_FORWARD => self.step_beats(e, 1, true),
            act::TEMPO_DOWN => self.set_tempo(e, ((e.factor() * 100.0).round() - 1.0) / 100.0),
            act::TEMPO_UP => self.set_tempo(e, ((e.factor() * 100.0).round() + 1.0) / 100.0),
            act::TEMPO_RESET => e.set_tempo(1.0),
            act::CALL_PREVIOUS | act::CALL_NEXT => {
                if e.playing {
                    return;
                }
                let t = e.playhead_frame();
                let near = 0.001 * self.rate;
                let target = if code == act::CALL_NEXT {
                    self.cue_points.iter().copied().find(|&p| p > t + near)
                } else {
                    self.cue_points.iter().rev().copied().find(|&p| p < t - near)
                };
                if let Some(p) = target {
                    self.call_at(e, p);
                }
            }
            act::TEMPO_RANGE => {
                let at = TEMPO_RANGES.iter().position(|&r| r == self.tempo_range).map(|i| i + 1).unwrap_or(0);
                self.set_tempo_range(e, TEMPO_RANGES[at % TEMPO_RANGES.len()]);
            }
            act::JOG_MODE => self.set_jog_vinyl(e, !self.jog_vinyl),
            c if (act::JUMP_A..act::JUMP_A + 8).contains(&c) => {
                let dest = self.jumpcues[(c - act::JUMP_A) as usize];
                if dest.is_finite() {
                    self.jump_to(e, dest);
                }
            }
            _ => {}
        }
    }

    fn cue_down(&mut self, e: &mut Engine) {
        if !e.has_track() || self.previewing {
            return;
        }
        if e.playing {
            e.pause(); // back cue
            e.seek(self.cue);
            return;
        }
        let pos = e.playhead_frame();
        if (pos - self.cue).abs() <= CUE_AT_SEC * self.rate {
            if e.play() {
                self.previewing = true;
                self.latched = false;
            }
        } else {
            self.cue = self.quantize(pos, e.len_frames() as f64);
            e.seek(self.cue);
        }
    }

    fn cue_up(&mut self, e: &mut Engine) {
        if !self.previewing {
            return;
        }
        self.previewing = false;
        if !self.latched {
            e.pause();
            e.seek(self.cue);
        }
    }

    /// The nearest beat or track edge (ui/edit_panel/quantize.py), the position itself without beats.
    fn quantize(&self, pos: f64, len: f64) -> f64 {
        if self.beats.is_empty() {
            return pos;
        }
        let mut best = 0.0;
        for &p in self.beats.iter().filter(|&&b| b >= 0.0 && b <= len).chain([0.0, len].iter()) {
            if (p - pos).abs() < (best - pos).abs() {
                best = p;
            }
        }
        best
    }

    /// A CUE point's CALL (stopped): the playhead and the cue point there.
    pub fn call_at(&mut self, e: &mut Engine, frame: f64) {
        if e.playing || !frame.is_finite() {
            return;
        }
        let frame = frame.clamp(0.0, e.len_frames() as f64);
        e.seek(frame);
        self.cue = frame;
    }

    /// The playhead ``step`` beats (bars) on (ui/pane.py's former _step_beats): playing, by as many beats from
    /// where it is, its place in the beat kept (a beat jump); stopped, to the beat (downbeat) itself.
    fn step_beats(&mut self, e: &mut Engine, step: i64, bars: bool) {
        let marks = if bars { &self.downbeats } else { &self.beats };
        if !e.has_track() || marks.is_empty() {
            return;
        }
        let t = e.playhead_frame();
        let near = 0.001 * self.rate;
        let i = marks.partition_point(|&m| m <= t) as i64 - 1;
        let target = if e.playing && i >= 0 {
            let j = (i + step).clamp(0, marks.len() as i64 - 1);
            t + marks[j as usize] - marks[i as usize]
        } else if step > 0 {
            let ahead: Vec<f64> = marks.iter().copied().filter(|&m| m > t + near).collect();
            match ahead.get((step as usize).min(ahead.len()).saturating_sub(1)) {
                Some(&m) => m,
                None => return,
            }
        } else {
            let behind: Vec<f64> = marks.iter().copied().filter(|&m| m < t - near).collect();
            if behind.is_empty() {
                return;
            }
            behind[(behind.len() as i64 + step).max(0) as usize]
        };
        e.seek(target.clamp(0.0, e.len_frames() as f64));
    }

    /// A JumpCUE's JUMP (see the module's docs).
    pub fn jump_to(&mut self, e: &mut Engine, dest: f64) {
        if !e.has_track() || !dest.is_finite() {
            return;
        }
        if !e.playing {
            e.seek(dest);
            return;
        }
        let near = 0.001 * self.rate;
        if self.armed.as_ref().is_some_and(|(d, _)| (d - dest).abs() < near) {
            return;
        }
        let mut bases = vec![e.playhead_frame()];
        if let Some((landed, at)) = self.landing {
            if clock::now() - at < JUMP_LANDING_SEC {
                bases.push(landed);
            }
        }
        let (mut ranges, mut starts) = (vec![], vec![]);
        for base in bases {
            let ahead: Vec<f64> = self.beats.iter().copied().filter(|&b| b > base).take(JUMP_CANDIDATE_BEATS).collect();
            if let Some(&last) = ahead.last() {
                ranges.push((base, last));
                starts.extend(ahead);
            }
        }
        if starts.is_empty() {
            return;
        }
        starts.sort_by(f64::total_cmp);
        starts.dedup();
        e.arm_jump(None, dest, true, starts);
        self.armed = Some((dest, ranges));
    }

    /// A bound message's value acted on: a fader's ``value`` of ``top``, a knob's relative ``steps``, a button
    /// pressed (``d2`` above 0) or let go.
    pub(crate) fn midi_value(&mut self, e: &mut Engine, action: u32, (value, top): (u32, f64), steps: i32, d2: u8,
                             at: f64) {
        let ticks = steps as f64;
        match action {
            act::TEMPO_FADER => {
                let half = (top + 1.0) / 2.0; // the middle: 64 (8192 with 14 bits)
                let norm = ((value as f64 - half) / half).clamp(-1.0, 1.0);
                self.set_tempo(e, 1.0 + norm * self.tempo_range as f64 / 100.0);
            }
            act::VOLUME_FADER => self.set_volume(e, value as f64 / top),
            act::TRIM => {
                self.trim_knob = value as f64 / top;
                e.strip.set_trim(mixer::trim_knob_gain(self.trim_knob));
            }
            act::COLOR_KNOB => {
                self.color_knob = value as f64 / top;
                e.strip.set_color(self.color_knob);
            }
            act::EQ_LOW | act::EQ_MID | act::EQ_HIGH => {
                self.eq_knobs[(action - act::EQ_LOW) as usize] = value as f64 / top;
                e.strip.set_eq(self.eq_knobs);
            }
            act::JOG_TOUCH if !self.jog_vinyl => {} // CDJ mode: the top is no platter
            act::JOG_TOUCH if d2 > 0 => {
                self.jog_spinning = false; // the hand back on a turning platter: held again
                if !self.jog_held {
                    self.jog_held = true;
                    e.grab(true, JOG_DELAY_SEC);
                    self.jog_last = at;
                    self.jog_rested = true;
                }
            }
            act::JOG_TOUCH => {
                if self.jog_held && !self.jog_spinning {
                    if self.jog_rested {
                        self.release_jog(e);
                    } else {
                        self.jog_spinning = true; // still turning: followed until it stops (tick)
                    }
                }
            }
            act::JOG_SCRATCH | act::JOG_BEND if self.jog_held => {
                let frames = ticks * JOG_TURN_SEC * self.rate / self.jog_ticks as f64;
                let target = (e.scratch_target() + frames).clamp(0.0, e.len_frames().saturating_sub(1) as f64);
                e.scratch_to(target, at);
                self.jog_last = at;
                self.jog_rested = false;
            }
            act::JOG_SCRATCH | act::JOG_BEND => self.bend_ticks += ticks,
            act::JOG_SEARCH => self.search_ticks += ticks,
            _ => self.action(e, action, d2 > 0),
        }
    }

    fn release_jog(&mut self, e: &mut Engine) {
        self.jog_held = false;
        self.jog_spinning = false;
        e.grab(false, JOG_DELAY_SEC);
    }

    // Once a tick
    pub fn tick(&mut self, e: &mut Engine) {
        let now = clock::now();
        let dt = (now - self.last_tick).max(0.0);
        self.last_tick = now;
        // the armed jump: played, dropped with playback, or moot once the playhead left the beats it was offered
        if e.jumped_count != self.jumped_seen {
            self.jumped_seen = e.jumped_count;
            if let Some((dest, _)) = self.armed.take() {
                self.landing = Some((dest, now));
            }
        }
        if let Some((_, ranges)) = &self.armed {
            let t = e.playhead_frame();
            let inside = ranges.iter().any(|&(lo, hi)| lo - 0.1 * self.rate <= t && t <= hi + 0.02 * self.rate);
            if !e.playing || !inside || e.jump_armed().is_none() {
                self.armed = None;
                e.disarm_jump();
            }
        }
        if self.previewing && !e.playing {
            self.previewing = false; // the track's end reached
        }
        // the jog: a platter let go of has stopped (let go of it); the record rests when the hand stops; the bend
        // eases out; the search seeks once a tick
        if self.jog_spinning && now - self.jog_last > JOG_STOP_SEC {
            self.release_jog(e);
        } else if self.jog_held && !self.jog_spinning && !self.jog_rested && now - self.jog_last > JOG_REST_SEC {
            e.scratch_rest(now);
            self.jog_rested = true;
        }
        let gain = BEND_PER_TURN / (self.jog_ticks as f64 * BEND_TAU_SEC);
        self.bend = (self.bend * (-dt / BEND_TAU_SEC).exp() + self.bend_ticks * gain).clamp(-BEND_MAX, BEND_MAX);
        self.bend_ticks = 0.0;
        if self.bend.abs() < 1e-4 {
            self.bend = 0.0;
        }
        e.set_bend(self.bend + self.button_bend);
        if self.search_ticks != 0.0 && e.has_track() {
            let frames = self.search_ticks * JOG_TURN_SEC * self.rate / self.jog_ticks as f64 * SEARCH_SPEED;
            if self.jog_held {
                // the record held: it is moved under the hand (a seek would play the record's sound under the scratch)
                let target = (e.scratch_target() + frames).clamp(0.0, e.len_frames().saturating_sub(1) as f64);
                e.scratch_to(target, now);
                self.jog_last = now;
                self.jog_rested = false;
            } else {
                e.seek((e.playhead_frame() + frames).clamp(0.0, e.len_frames() as f64));
            }
        }
        self.search_ticks = 0.0;
    }

    /// Every light the bindings may show: bit k (0 - 25) the JumpCUE A + k lit (set; blinking while its jump is
    /// armed), bit LAMP_PLAY / LAMP_CUE those buttons' (lamps).
    pub fn lights(&self, e: &Engine, now: f64) -> u64 {
        let lamps = self.lamps(e, now) as u64;
        let mut out = (lamps & 1) << LAMP_PLAY | (lamps >> 1 & 1) << LAMP_CUE | (self.jog_vinyl as u64) << LAMP_VINYL;
        if !e.has_track() {
            return out;
        }
        let blink = (now / BLINK_SEC).floor() as i64 % 2 == 0;
        let armed = e.jump_armed();
        for (k, &point) in self.jumpcues.iter().enumerate() {
            if !point.is_finite() {
                continue;
            }
            let is_armed = armed.is_some_and(|d| (d - point).abs() < 0.001 * self.rate);
            if !is_armed || blink {
                out |= 1 << k;
            }
        }
        out
    }

    /// Bit 0 PLAY, bit 1 CUE: lit now (a blinking one as it is at this moment); bit 2 the jog's VINYL mode.
    pub fn lamps(&self, e: &Engine, now: f64) -> u32 {
        let vinyl = (self.jog_vinyl as u32) << 2;
        if !e.has_track() {
            return vinyl;
        }
        let blink = (now / BLINK_SEC).floor() as i64 % 2 == 0;
        let play = e.playing || blink;
        let on_cue = (e.playhead_frame() - self.cue).abs() <= CUE_AT_SEC * self.rate;
        let cue = e.playing || self.previewing || on_cue || blink;
        play as u32 | (cue as u32) << 1 | vinyl
    }

    // Reported
    pub fn cue_frame(&self, e: &Engine) -> f64 {
        if e.has_track() { self.cue } else { f64::NAN }
    }

    pub fn volume(&self) -> f64 {
        self.volume
    }

    /// Its mixer channel's knobs: TRIM, EQ LOW, MID, HIGH (0 .. 1, the middle 0 dB).
    pub fn knobs(&self) -> [f64; 4] {
        [self.trim_knob, self.eq_knobs[0], self.eq_knobs[1], self.eq_knobs[2]]
    }

    /// Its mixer channel's COLOR knob (0 .. 1, 0.5 off).
    pub fn color_knob(&self) -> f64 {
        self.color_knob
    }

    /// A mixer channel knob turned (the GUI's; a controller's come as midi_value): 0 TRIM, 1 EQ LOW, 2 MID, 3 HIGH,
    /// 4 COLOR.
    pub fn set_knob(&mut self, e: &mut Engine, which: u32, position: f64) {
        let p = if position.is_finite() { position.clamp(0.0, 1.0) } else { 0.5 };
        match which {
            0 => {
                self.trim_knob = p;
                e.strip.set_trim(mixer::trim_knob_gain(p));
            }
            1..=3 => {
                self.eq_knobs[which as usize - 1] = p;
                e.strip.set_eq(self.eq_knobs);
            }
            4 => {
                self.color_knob = p;
                e.strip.set_color(p);
            }
            _ => {}
        }
    }

    /// JumpCUE ``k`` (A + k) is set.
    pub fn jumpcue_set(&self, k: usize) -> bool {
        self.jumpcues.get(k).is_some_and(|p| p.is_finite())
    }

    pub fn wants_fast_ticks(&self) -> bool {
        self.jog_held || self.bend != 0.0
    }
}

/// The deck: its renderer and its controls.
pub struct Slot {
    pub engine: Engine,
    pub deck: Deck,
}

impl Slot {
    /// The deck on the main device's ring (its lane), its mixer channel's strip ``strip``: the time stretch off,
    /// a 40 ms buffer, a jog wheel of 720 messages a turn in VINYL mode until the GUI or a controller says otherwise.
    pub fn new(rate: u32, ch: usize, ring: Arc<Ring>, strip: Arc<StripParams>) -> Slot {
        let mut engine = Engine::new(rate, ch, ring, strip);
        let mut deck = Deck::new(rate as f64);
        engine.set_timestretch(false);
        engine.set_buffer_ms(40.0);
        deck.set_jog_ticks(720);
        deck.set_jog_vinyl(&mut engine, true);
        Slot { engine, deck }
    }
}
