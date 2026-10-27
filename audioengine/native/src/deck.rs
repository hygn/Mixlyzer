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
// the jog's search (SHIFT): the playhead a bar on or back per this much of a turn, as One bar forward / back:
// a turn an 8-bar phrase
const SEARCH_BARS_PER_TURN: f64 = 8.0;
// a search's last step taken as where the deck is for this long (its crossfade not heard yet: the buffer and
// the device's queue), and a part of a bar turned kept as long
const SEARCH_AIM_SEC: f64 = 0.5;

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
    search_part: f64,               // ticks turned toward the next bar
    search_aim: Option<(f64, f64)>, // the last step's (frame, clock time)
    search_at: f64,                 // the last turn's clock time
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
            search_part: 0.0,
            search_aim: None,
            search_at: 0.0,
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
        self.search_aim = None;
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
        if let Some(target) = self.step_target(e, e.playhead_frame(), step, bars) {
            e.seek(target);
        }
    }

    /// Where ``step`` beats (bars) from ``t`` are: playing, as far on as the marks between (the phase kept);
    /// stopped, on the mark itself. None: no marks, or none that way.
    fn step_target(&self, e: &Engine, t: f64, step: i64, bars: bool) -> Option<f64> {
        let marks = if bars { &self.downbeats } else { &self.beats };
        if !e.has_track() || marks.is_empty() {
            return None;
        }
        let near = 0.001 * self.rate;
        let i = marks.partition_point(|&m| m <= t) as i64 - 1;
        let target = if e.playing && i >= 0 {
            let j = (i + step).clamp(0, marks.len() as i64 - 1);
            t + marks[j as usize] - marks[i as usize]
        } else if step > 0 {
            let ahead: Vec<f64> = marks.iter().copied().filter(|&m| m > t + near).collect();
            *ahead.get((step as usize).min(ahead.len()).saturating_sub(1))?
        } else {
            let behind: Vec<f64> = marks.iter().copied().filter(|&m| m < t - near).collect();
            if behind.is_empty() {
                return None;
            }
            behind[(behind.len() as i64 + step).max(0) as usize]
        };
        Some(target.clamp(0.0, e.len_frames() as f64))
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
        // eases out; the search steps by bars
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
            if now - self.search_at > SEARCH_AIM_SEC {
                self.search_part = 0.0; // a new turn: the last one's part of a bar dropped
            }
            self.search_at = now;
            self.search_part += self.search_ticks;
            let per_bar = self.jog_ticks as f64 / SEARCH_BARS_PER_TURN;
            let bars = (self.search_part / per_bar).trunc();
            if bars != 0.0 {
                self.search_part -= bars * per_bar;
                self.search_bars(e, bars as i64, now);
            }
        }
        self.search_ticks = 0.0;
    }

    /// The jog's search: ``bars`` on (back), as One bar forward / back. Playing, from the last step while it may
    /// not be heard yet (the playhead heard still before it: steps in a row would land on the same bar).
    fn search_bars(&mut self, e: &mut Engine, bars: i64, now: f64) {
        let from = match self.search_aim {
            Some((frame, at)) if e.playing && !e.scratching && now - at < SEARCH_AIM_SEC => {
                frame + (now - at) * self.rate * e.factor()
            }
            _ => e.playhead_frame(),
        };
        if let Some(target) = self.step_target(e, from, bars, true) {
            e.seek(target);
            self.search_aim = Some((target, now));
        }
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

/// The deck's rules (CUE / PLAY, CALL, steps, JumpCUE, the tempo range, the lamps) on a track of its own, its ring
/// read by the main device's callback as a device would (no device opened).
#[cfg(test)]
mod tests {
    use super::*;
    use crate::bridge::Bridge;
    use crate::device::{Bus, CallbackState, Port};
    use crate::mixer::MixParams;
    use crate::shm::Pcm;

    const RATE: u32 = 48000;
    const SEC: f64 = RATE as f64;
    const BEAT: f64 = 0.5 * SEC; // 120 BPM, a downbeat every 4 beats
    const BLINK_ON: f64 = 0.1; // a lamp's blink lit at this clock time, dark half a second on
    const BLINK_OFF: f64 = 0.6;

    struct Rig {
        s: Slot,
        ring: Arc<Ring>,
        cb: CallbackState,
        out: Vec<f32>,
    }

    impl Rig {
        /// A deck with a ``seconds``-long track (none: 0), its beats, ``cues`` (s) its CUE points, ``jumps`` (s)
        /// JumpCUE A, B, ...
        fn new(seconds: f64, beats: bool, cues: &[f64], jumps: &[f64]) -> Rig {
            let port = Arc::new(Port::default());
            let params = Arc::new(MixParams::new(2));
            let ring = Arc::new(Ring::new(RATE as u64 * 5 / 2, 4, SEC, port));
            ring.attach_bus(Bus { params: params.clone(), bridges: std::array::from_fn(|_| Arc::new(Bridge::new(RATE as u64, SEC))) });
            let mut s = Slot::new(RATE, 2, ring.clone(), params.strip.clone());
            if seconds > 0.0 {
                let n = (seconds * SEC) as usize;
                let pcm: Vec<f32> = (0..2 * n).map(|i| 0.1 * ((i / 2) as f32 * 0.05).sin()).collect();
                s.engine.set_track(Some(Pcm::owned(pcm, 2)), 1);
            }
            s.deck.new_track();
            let beats: Vec<f64> = if beats { (0..).map(|k| k as f64 * BEAT).take_while(|&b| b < seconds * SEC).collect() } else { vec![] };
            let mut payload = (beats.len() as u32).to_le_bytes().to_vec();
            beats.iter().for_each(|b| payload.extend(b.to_le_bytes()));
            (0..beats.len()).for_each(|k| payload.push((k % 4 == 0) as u8));
            payload.extend((cues.len() as u32).to_le_bytes());
            cues.iter().for_each(|c| payload.extend((c * SEC).to_le_bytes()));
            for k in 0..26 {
                payload.extend(jumps.get(k).map(|j| j * SEC).unwrap_or(f64::NAN).to_le_bytes());
            }
            s.deck.set_marks(&payload);
            Rig { s, ring, cb: CallbackState::new(2, SEC), out: vec![0.0; 480 * 2] }
        }

        fn press(&mut self, code: u32) {
            self.s.deck.action(&mut self.s.engine, code, true);
        }

        fn release(&mut self, code: u32) {
            self.s.deck.action(&mut self.s.engine, code, false);
        }

        fn tap(&mut self, code: u32) {
            self.press(code);
            self.release(code);
        }

        fn seek(&mut self, sec: f64) {
            self.s.engine.seek(sec * SEC);
        }

        /// ``blocks`` of 10 ms read by the device's callback, the engine and the deck ticking between.
        fn pump(&mut self, blocks: usize) {
            for _ in 0..blocks {
                self.s.engine.tick();
                self.ring.fill(&mut self.out, &mut self.cb, 0.0);
                self.s.deck.tick(&mut self.s.engine);
            }
        }

        fn at(&self) -> f64 {
            self.s.engine.playhead_frame() / SEC
        }

        fn cue(&self) -> f64 {
            self.s.deck.cue / SEC
        }

        fn playing(&self) -> bool {
            self.s.engine.playing
        }

        /// (PLAY lit, CUE lit) at clock time ``now``.
        fn lamps(&self, now: f64) -> (bool, bool) {
            let l = self.s.deck.lamps(&self.s.engine, now);
            (l & 1 != 0, l & 2 != 0)
        }
    }

    fn near(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-6
    }

    #[test]
    fn no_track_cue_and_play_do_nothing() {
        let mut r = Rig::new(0.0, false, &[], &[]);
        r.press(act::CUE);
        r.release(act::CUE);
        r.tap(act::PLAY_PAUSE);
        assert!(!r.playing());
        assert_eq!(r.s.deck.lamps(&r.s.engine, BLINK_ON), 4, "only the jog's VINYL lit");
        assert!(r.s.deck.cue_frame(&r.s.engine).is_nan());
    }

    #[test]
    fn cue_stopped_off_the_cue_point_sets_it_on_the_nearest_beat() {
        let mut r = Rig::new(10.0, true, &[], &[]);
        r.seek(1.6);
        r.press(act::CUE);
        assert!(!r.playing());
        assert!(near(r.cue(), 1.5) && near(r.at(), 1.5), "cue {} playhead {}", r.cue(), r.at());
        r.release(act::CUE); // let go: nothing more
        assert!(!r.playing() && near(r.at(), 1.5));
        // without beats the cue point is where the playhead stands; past the last beat, the track's end is nearer
        let mut r = Rig::new(10.0, false, &[], &[]);
        r.seek(1.6);
        r.tap(act::CUE);
        assert!(near(r.cue(), 1.6));
        let mut r = Rig::new(10.0, true, &[], &[]);
        r.seek(9.9);
        r.tap(act::CUE);
        assert!(near(r.cue(), 10.0));
    }

    #[test]
    fn cue_held_on_the_cue_point_previews_and_goes_back() {
        let mut r = Rig::new(10.0, true, &[], &[]);
        r.seek(2.0);
        r.tap(act::CUE); // the cue point at 2 s
        r.press(act::CUE);
        assert!(r.playing() && r.s.deck.previewing);
        r.pump(30);
        assert!(r.at() > 2.1, "previewing plays on: {}", r.at());
        r.press(act::CUE); // a second press while held (the GUI's and a controller's): ignored
        assert!(r.playing());
        r.release(act::CUE);
        assert!(!r.playing() && !r.s.deck.previewing);
        assert!(near(r.at(), 2.0));
    }

    #[test]
    fn play_while_cue_is_held_plays_on() {
        let mut r = Rig::new(10.0, true, &[], &[]);
        r.press(act::CUE); // on the cue point (the start): previewing
        r.tap(act::PLAY_PAUSE);
        assert!(r.s.deck.latched);
        r.tap(act::PLAY_PAUSE); // again: no change
        assert!(r.playing() && r.s.deck.latched);
        r.release(act::CUE);
        assert!(r.playing() && !r.s.deck.previewing, "latched: plays on");
    }

    #[test]
    fn cue_while_playing_goes_back_to_it_and_stops() {
        let mut r = Rig::new(10.0, true, &[], &[]);
        r.seek(1.0);
        r.tap(act::CUE);
        r.tap(act::PLAY_PAUSE);
        r.pump(50);
        assert!(r.at() > 1.3);
        r.press(act::CUE);
        assert!(!r.playing() && near(r.at(), 1.0));
        r.release(act::CUE);
        assert!(!r.playing() && near(r.at(), 1.0));
    }

    #[test]
    fn play_pause_stop_and_track_start() {
        let mut r = Rig::new(10.0, true, &[], &[]);
        r.seek(3.0);
        r.tap(act::CUE); // the cue point at 3 s
        r.tap(act::PLAY_PAUSE);
        assert!(r.playing());
        r.pump(50);
        r.tap(act::PLAY_PAUSE);
        let paused = r.at();
        assert!(!r.playing() && paused > 3.2, "paused where it was heard: {paused}");
        r.tap(act::PLAY_PAUSE);
        r.pump(10);
        r.tap(act::TRACK_START);
        assert!(r.playing(), "back to the start, playing on");
        r.tap(act::STOP);
        assert!(!r.playing() && near(r.at(), 0.0) && near(r.cue(), 3.0), "stopped at the start, the cue point kept");
    }

    #[test]
    fn call_moves_the_playhead_and_the_cue_point_while_stopped() {
        let mut r = Rig::new(10.0, true, &[2.0, 4.0], &[]);
        r.tap(act::CALL_NEXT);
        assert!(near(r.at(), 2.0) && near(r.cue(), 2.0));
        r.tap(act::CALL_NEXT);
        assert!(near(r.at(), 4.0) && near(r.cue(), 4.0));
        r.tap(act::CALL_NEXT); // none after: stays
        assert!(near(r.at(), 4.0));
        r.tap(act::CALL_PREVIOUS);
        assert!(near(r.at(), 2.0) && near(r.cue(), 2.0));
        r.s.deck.call_at(&mut r.s.engine, 20.0 * SEC); // past the end: the end
        assert!(near(r.cue(), 10.0));
        r.tap(act::CALL_PREVIOUS);
        r.tap(act::PLAY_PAUSE);
        assert!(r.playing());
        r.tap(act::CALL_NEXT); // playing: ignored
        r.s.deck.call_at(&mut r.s.engine, 2.0 * SEC);
        assert!(near(r.cue(), 4.0));
    }

    #[test]
    fn beat_and_bar_steps_while_stopped() {
        let mut r = Rig::new(10.0, true, &[], &[]);
        r.seek(1.2);
        r.tap(act::BEAT_FORWARD);
        assert!(near(r.at(), 1.5));
        r.tap(act::BEAT_BACK);
        assert!(near(r.at(), 1.0));
        r.tap(act::BAR_FORWARD);
        assert!(near(r.at(), 2.0));
        r.tap(act::BAR_BACK);
        assert!(near(r.at(), 0.0));
        r.tap(act::BAR_BACK); // none before: stays
        assert!(near(r.at(), 0.0));
    }

    #[test]
    fn lamps_follow_the_state() {
        let mut r = Rig::new(10.0, true, &[], &[]);
        // stopped on the cue point: PLAY blinks, CUE lit
        assert_eq!((r.lamps(BLINK_ON), r.lamps(BLINK_OFF)), ((true, true), (false, true)));
        // stopped off it: both blink
        r.seek(1.2);
        assert_eq!((r.lamps(BLINK_ON), r.lamps(BLINK_OFF)), ((true, true), (false, false)));
        // playing: both lit
        r.tap(act::PLAY_PAUSE);
        assert_eq!((r.lamps(BLINK_ON), r.lamps(BLINK_OFF)), ((true, true), (true, true)));
        r.tap(act::STOP);
        r.press(act::CUE); // previewing: both lit
        assert_eq!(r.lamps(BLINK_OFF), (true, true));
    }

    #[test]
    fn cue_at_the_tracks_end_does_not_preview() {
        let mut r = Rig::new(10.0, true, &[], &[]);
        r.seek(10.0);
        r.tap(act::CUE); // the cue point at the end
        r.press(act::CUE);
        assert!(!r.playing() && !r.s.deck.previewing);
    }

    #[test]
    fn a_new_track_clears_the_cue_point_and_the_preview() {
        let mut r = Rig::new(10.0, true, &[2.0], &[3.0]);
        r.seek(2.0);
        r.tap(act::CUE);
        r.press(act::CUE);
        r.s.deck.new_track();
        assert!(!r.s.deck.previewing && near(r.cue(), 0.0));
        assert!(!r.s.deck.jumpcue_set(0) && r.s.deck.cue_points.is_empty());
    }

    #[test]
    fn the_track_end_plays_on() {
        let mut r = Rig::new(1.0, true, &[], &[]);
        r.seek(0.8);
        r.tap(act::PLAY_PAUSE);
        r.pump(60);
        assert!(r.playing(), "silence after the end, still playing");
        assert_eq!(r.s.engine.finished_count, 1);
        r.seek(0.2); // moved back: plays again
        r.pump(10);
        assert!(r.playing() && r.at() > 0.2 && r.at() < 0.5, "{}", r.at());
    }

    #[test]
    fn jumpcue_stopped_moves_the_playhead() {
        let mut r = Rig::new(10.0, true, &[], &[6.0]);
        r.seek(1.0);
        r.tap(act::CUE);
        r.tap(act::JUMP_A);
        assert!(near(r.at(), 6.0) && near(r.cue(), 1.0), "the cue point kept");
        r.tap(act::JUMP_A + 1); // B not set: nothing
        assert!(near(r.at(), 6.0));
    }

    #[test]
    fn jumpcue_playing_is_armed_on_the_next_beats_and_jumps() {
        let mut r = Rig::new(10.0, true, &[], &[6.0]);
        r.tap(act::PLAY_PAUSE);
        r.tap(act::JUMP_A);
        assert!(r.s.engine.jump_armed().is_some_and(|d| near(d, 6.0 * SEC)));
        let lit = r.s.deck.lights(&r.s.engine, BLINK_OFF);
        assert_eq!(lit & 1, 0, "the armed pad blinks");
        r.tap(act::JUMP_A); // pressed again: kept as it is
        assert!(r.s.deck.armed.is_some());
        r.pump(80); // past the first beat (0.5 s) and its fade
        assert_eq!(r.s.engine.jumped_count, 1);
        assert!(r.s.deck.armed.is_none() && r.s.deck.landing.is_some());
        assert!(r.at() > 6.0 && r.at() < 6.5, "landed: {}", r.at());
    }

    #[test]
    fn jumpcue_armed_is_dropped_when_playback_stops() {
        let mut r = Rig::new(10.0, true, &[], &[6.0]);
        r.tap(act::PLAY_PAUSE);
        r.tap(act::JUMP_A);
        r.tap(act::PLAY_PAUSE);
        r.pump(1);
        assert!(r.s.deck.armed.is_none() && r.s.engine.jump_armed().is_none());
    }

    #[test]
    fn tempo_range_keeps_the_tempo_inside() {
        let mut r = Rig::new(10.0, true, &[], &[]);
        r.s.deck.set_tempo_range(&mut r.s.engine, 16);
        r.s.deck.set_tempo(&mut r.s.engine, 1.12);
        assert!(near(r.s.engine.factor(), 1.12));
        r.s.deck.set_tempo_range(&mut r.s.engine, 10);
        assert!(near(r.s.engine.factor(), 1.10));
        r.tap(act::TEMPO_RANGE); // 10 -> 16 (kept)
        assert_eq!(r.s.deck.tempo_range, 16);
        r.tap(act::TEMPO_RANGE); // -> WIDE
        r.s.deck.set_tempo(&mut r.s.engine, 0.0);
        assert!(near(r.s.engine.factor(), 0.0), "WIDE: down to 0");
        r.tap(act::TEMPO_RANGE); // -> 6
        assert_eq!(r.s.deck.tempo_range, 6);
        assert!(near(r.s.engine.factor(), 0.94));
        r.s.deck.set_tempo(&mut r.s.engine, f64::NAN);
        assert!(near(r.s.engine.factor(), 1.0));
        r.s.deck.set_tempo_range(&mut r.s.engine, 7); // not a range: WIDE
        assert_eq!(r.s.deck.tempo_range, 100);
    }

    impl Rig {
        /// The jog turned ``ticks`` with SHIFT (its search), then a tick.
        fn search(&mut self, ticks: i32) {
            self.s.deck.midi_value(&mut self.s.engine, act::JOG_SEARCH, (0, 127.0), ticks, 0, clock::now());
            self.s.deck.tick(&mut self.s.engine);
        }
    }

    #[test]
    fn jog_search_steps_by_bars_while_stopped() {
        let mut r = Rig::new(10.0, true, &[], &[]); // a bar: 2 s
        r.seek(1.2);
        r.search(90); // an eighth of a turn: a bar on, as One bar forward
        assert!(near(r.at(), 2.0), "{}", r.at());
        r.search(45);
        assert!(near(r.at(), 2.0), "half a bar's turn: not yet");
        r.search(45);
        assert!(near(r.at(), 4.0), "{}", r.at());
        r.search(-180);
        assert!(near(r.at(), 0.0), "two bars back: {}", r.at());
    }

    #[test]
    fn jog_search_while_playing_steps_bars_in_a_row() {
        let mut r = Rig::new(20.0, true, &[], &[]);
        r.seek(1.0);
        r.tap(act::PLAY_PAUSE);
        r.pump(2);
        let from = r.at();
        for _ in 0..4 {
            r.search(90); // faster than the buffer is heard: each from the last step, not the playhead heard
        }
        r.pump(20);
        let moved = r.at() - from;
        assert!(r.playing() && moved > 8.0 && moved < 8.5, "4 bars on (8 s), the phase kept: {moved}");
    }

    #[test]
    fn play_after_cue_shows_the_cue_point_not_where_it_played() {
        let mut r = Rig::new(10.0, true, &[], &[]);
        r.seek(1.0);
        r.tap(act::CUE); // the cue point at 1.0
        r.tap(act::PLAY_PAUSE);
        r.pump(100); // ~2.0
        r.tap(act::CUE); // back to the cue point, stopped
        r.pump(2);
        assert!(!r.playing() && near(r.at(), 1.0));
        r.tap(act::PLAY_PAUSE);
        for _ in 0..5 {
            // before the new audio is heard: still at the cue point (the flushed audio's ~2 s never shown)
            let (frame, _speed) = r.s.engine.display_frame(clock::now());
            assert!(r.at() >= 1.0 - 1e-6 && r.at() < 1.1 && frame / SEC < 1.1, "{} {}", r.at(), frame / SEC);
            r.pump(1);
        }
    }

    #[test]
    fn seeks_faster_than_the_ring_drains_never_overrun_it() {
        let mut r = Rig::new(20.0, true, &[], &[]);
        r.tap(act::PLAY_PAUSE);
        let most = r.s.engine.buffer_frames() as u64 + 2 * (SEC as u64 / 100);
        for k in 0..600 {
            r.seek(1.0 + 0.01 * k as f64); // a seek each tick, the device reading nothing between
            r.s.engine.tick();
            assert!(r.ring.level() <= most, "{} frames queued", r.ring.level());
        }
        r.pump(10);
        assert!(r.at() > 6.99 && r.at() < 7.2, "the last seek heard: {}", r.at());
    }
}
