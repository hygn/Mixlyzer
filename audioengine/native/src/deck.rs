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
//! The MIDI controller's bindings (op::MIDI_MAP) are messages to actions: a button (a note or a CC, pressed
//! above 0), a fader (a CC, 7 or 14 bits: MSB at data1, LSB at data1 + 32), a jog wheel (relative CC around 64)
//! and its touch (a note). The GUI's own actions (library, editor ...) are passed on to it (ipc.rs MIDI_EVENTS).
//!
//! The jog wheel as rekordbox's: touched, the record is held and follows the platter (both its rotation messages:
//! a DDJ sends the scratch CC or, out of its vinyl mode and once the hand is off, the bend CC); let go while the
//! platter still turns (a spin, a backspin), the record follows it on until it stops, then plays (or rests) as
//! before. Not held, a rotation bends the tempo.
//!
//! The controller's lights (each binding's lamp): PLAY and CUE as above, a JumpCUE's pad lit while it is set
//! (blinking while its jump is armed), as rekordbox's hot cue pads; the level meter (LEVEL_METER, a CC) as the
//! GUI's peak meter: the heard peak times its gain, -24 - 0 dBFS as 0 - 127, falling 24 dB/s.

use crate::clock;
use crate::engine::Engine;
use crate::ipc::{Control, MIDI_EVENT_GUI, Reader, act};
use crate::midi::Midi;

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
const SEARCH_SPEED: f64 = 16.0; // the jog's search: this many times the record's speed
const VOLUME_CURVE_GAMMA: f64 = 0.4; // utils/volume.py
// a binding's lamp (ipc.rs op::MIDI_MAP): 1 - 26 the JumpCUE A - Z
pub const LAMP_PLAY: u8 = 32;
pub const LAMP_CUE: u8 = 33;
// the level meter: ui/track_info_panel.py VerticalPeakMeter's scale and fall; sent at most this often
const METER_FLOOR_DBFS: f64 = -24.0;
const METER_FALL_DB_PER_SEC: f64 = 24.0;
const METER_INTERVAL_SEC: f64 = 0.02;

struct Binding {
    status: u8,
    data1: u8,
    hires: bool,
    lamp: u8,
    action: u32,
    msb: u8,
    lsb: u8,
}

pub struct Deck {
    rate: f64,
    volume: f64,
    trim_db: f64,
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
    jog_ticks: u32,
    bindings: Vec<Binding>,
    jog_held: bool,     // the record held: the hand on the platter, or the platter let go of still turning
    jog_spinning: bool, // let go of while it turned: held until it stops
    jog_last: f64,
    jog_rested: bool,
    bend: f64,
    bend_ticks: f64,
    search_ticks: f64,
    last_tick: f64,
    midi: Option<Midi>,
    pub midi_state: u32,
    lamps_sent: Option<u64>,
    meter_gain: f64,
    meter_db: f64,
    meter_sent: Option<u8>,
    meter_at: f64,
}

impl Deck {
    pub fn new(rate: f64) -> Deck {
        Deck {
            rate,
            volume: 1.0,
            trim_db: 0.0,
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
            bindings: vec![],
            jog_held: false,
            jog_spinning: false,
            jog_last: 0.0,
            jog_rested: true,
            bend: 0.0,
            bend_ticks: 0.0,
            search_ticks: 0.0,
            last_tick: clock::now(),
            midi: None,
            midi_state: 0,
            lamps_sent: None,
            meter_gain: 1.0,
            meter_db: METER_FLOOR_DBFS,
            meter_sent: None,
            meter_at: 0.0,
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
        let n = r.u32().unwrap_or(0) as usize;
        let beats: Vec<f64> = (0..n).filter_map(|_| r.f64()).collect();
        let down: Vec<bool> = (0..n).filter_map(|_| r.u8()).map(|v| v != 0).collect();
        let m = r.u32().unwrap_or(0) as usize;
        let mut cues: Vec<f64> = (0..m).filter_map(|_| r.f64()).filter(|v| v.is_finite()).collect();
        cues.sort_by(|a, b| a.partial_cmp(b).unwrap());
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

    pub fn set_volume_trim(&mut self, e: &mut Engine, db: f64) {
        self.trim_db = if db.is_finite() { db } else { 0.0 };
        self.apply_volume(e);
    }

    /// utils/volume.py's slider_percent_to_linear: the fader's position on a curve, Qt's logarithmic volume
    /// scale as linear, the trim on top.
    fn apply_volume(&self, e: &mut Engine) {
        let v = self.volume.powf(VOLUME_CURVE_GAMMA);
        let base = if v > 0.99 { 1.0 } else { -(1.0 - v).ln() / 100f64.ln() };
        let trim = 10f64.powf(self.trim_db / 20.0).clamp(0.0, 1.0);
        e.set_music_gain((base * trim).clamp(0.0, 1.0));
    }

    pub fn set_jog_ticks(&mut self, jog_ticks: u32) {
        self.jog_ticks = jog_ticks.max(1);
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
        starts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        starts.dedup();
        e.arm_jump(None, dest, true, starts);
        self.armed = Some((dest, ranges));
    }

    // MIDI
    pub fn open_midi(&mut self, name: &str, wake: Option<std::thread::Thread>) {
        self.close_midi();
        if name.is_empty() {
            return;
        }
        match Midi::open(name, wake) {
            Ok(m) => {
                eprintln!("[AudioEngine] MIDI input: {}", m.name);
                self.midi = Some(m);
                self.midi_state = 1;
            }
            Err(err) => {
                eprintln!("[AudioEngine] {err}");
                self.midi_state = 2;
            }
        }
    }

    pub fn close_midi(&mut self) {
        self.send_lamps(0, None); // the controller's lights out
        self.send_meter(0);
        self.midi = None;
        self.midi_state = 0;
        self.lamps_sent = None;
    }

    pub fn set_bindings(&mut self, payload: &[u8]) {
        let mut r = Reader::new(payload);
        let n = r.u32().unwrap_or(0);
        self.bindings.clear();
        for _ in 0..n {
            let (Some(status), Some(data1), Some(flags), Some(lamp), Some(action)) =
                (r.u8(), r.u8(), r.u8(), r.u8(), r.u32())
            else {
                break;
            };
            self.bindings.push(Binding { status, data1, hires: flags & 1 != 0, lamp, action, msb: 0, lsb: 0 });
        }
        self.lamps_sent = None;
        self.meter_sent = None;
    }

    pub fn set_meter_gain(&mut self, gain: f64) {
        self.meter_gain = if gain.is_finite() { gain.max(0.0) } else { 1.0 };
    }

    /// The messages come since the last call: each acted on, and into the GUI's ring.
    pub fn read_midi(&mut self, e: &mut Engine, ctl: &Control) {
        let Some(midi) = self.midi.as_ref() else { return };
        for (at, msg) in midi.drain() {
            let (flags, gui_action) = self.message(e, at, msg);
            ctl.put_midi_event(msg, flags, gui_action);
        }
    }

    /// One message acted on: (flags, the GUI's action) for the GUI's ring.
    fn message(&mut self, e: &mut Engine, at: f64, msg: [u8; 3]) -> (u8, u32) {
        let [status, d1, d2] = msg;
        // the response measured (play / pause / seek heard) from this message's arrival, as a GUI command's from
        // its sending
        e.command_at = at;
        let mut flags = 0u8;
        let mut gui_action = 0u32;
        for i in 0..self.bindings.len() {
            let b = &mut self.bindings[i];
            if b.status != status || b.action == act::LEVEL_METER {
                continue; // an output
            }
            let value = if b.data1 == d1 {
                b.msb = d2;
                if b.hires { ((d2 as u32) << 7 | b.lsb as u32, 16383.0) } else { (d2 as u32, 127.0) }
            } else if b.hires && d1 == b.data1.wrapping_add(32) && status & 0xF0 == 0xB0 {
                b.lsb = d2;
                ((b.msb as u32) << 7 | d2 as u32, 16383.0)
            } else {
                continue;
            };
            let action = b.action;
            if action >= act::GUI {
                flags |= MIDI_EVENT_GUI;
                gui_action = action;
                continue;
            }
            self.midi_value(e, action, value, d2, at);
        }
        (flags, gui_action)
    }

    fn midi_value(&mut self, e: &mut Engine, action: u32, (value, top): (u32, f64), d2: u8, at: f64) {
        let ticks = d2 as f64 - 64.0; // a relative CC: 65 one step on, 63 one back
        match action {
            act::TEMPO_FADER => {
                let half = (top + 1.0) / 2.0; // the middle: 64 (8192 with 14 bits)
                let norm = ((value as f64 - half) / half).clamp(-1.0, 1.0);
                self.set_tempo(e, 1.0 + norm * self.tempo_range as f64 / 100.0);
            }
            act::VOLUME_FADER => self.set_volume(e, value as f64 / top),
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
        e.set_bend(self.bend);
        if self.search_ticks != 0.0 && e.has_track() {
            let frames = self.search_ticks * JOG_TURN_SEC * self.rate / self.jog_ticks as f64 * SEARCH_SPEED;
            e.seek((e.playhead_frame() + frames).clamp(0.0, e.len_frames() as f64));
        }
        self.search_ticks = 0.0;
        let lights = self.lights(e, now);
        if self.lamps_sent != Some(lights) {
            let before = self.lamps_sent;
            self.send_lamps(lights, before);
            self.lamps_sent = Some(lights);
        }
        // the level meter: the GUI meter's reading
        let peak = e.heard_peak_dbfs() + 20.0 * self.meter_gain.max(1e-9).log10();
        self.meter_db = peak.max(self.meter_db - METER_FALL_DB_PER_SEC * dt).clamp(METER_FLOOR_DBFS, 0.0);
        if now - self.meter_at >= METER_INTERVAL_SEC {
            let value = meter_value(self.meter_db);
            if self.meter_sent != Some(value) {
                self.send_meter(value);
                self.meter_sent = Some(value);
                self.meter_at = now;
            }
        }
    }

    /// Every light the bindings may show: bit k (0 - 25) the JumpCUE A + k lit (set; blinking while its jump is
    /// armed), bit LAMP_PLAY / LAMP_CUE those buttons' (lamps).
    fn lights(&self, e: &Engine, now: f64) -> u64 {
        let lamps = self.lamps(e, now) as u64;
        let mut out = (lamps & 1) << LAMP_PLAY | (lamps >> 1 & 1) << LAMP_CUE;
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

    /// Bit 0 PLAY, bit 1 CUE: lit now (a blinking one as it is at this moment).
    pub fn lamps(&self, e: &Engine, now: f64) -> u32 {
        if !e.has_track() {
            return 0;
        }
        let blink = (now / BLINK_SEC).floor() as i64 % 2 == 0;
        let play = e.playing || blink;
        let on_cue = (e.playhead_frame() - self.cue).abs() <= CUE_AT_SEC * self.rate;
        let cue = e.playing || self.previewing || on_cue || blink;
        play as u32 | (cue as u32) << 1
    }

    /// The bindings' lights as ``lights`` (lights()), those that changed since ``before`` (None: all).
    fn send_lamps(&mut self, lights: u64, before: Option<u64>) {
        let Some(midi) = self.midi.as_mut() else { return };
        for b in &self.bindings {
            if b.lamp == 0 || !matches!(b.status & 0xF0, 0x90 | 0xB0) {
                continue;
            }
            let bit = if b.lamp <= 26 { b.lamp - 1 } else { b.lamp };
            let lit = lights >> bit & 1 != 0;
            if before.is_some_and(|was| (was >> bit & 1 != 0) == lit) {
                continue;
            }
            midi.send(&[b.status, b.data1, if lit { 0x7F } else { 0 }]);
        }
    }

    fn send_meter(&mut self, value: u8) {
        let Some(midi) = self.midi.as_mut() else { return };
        for b in self.bindings.iter().filter(|b| b.action == act::LEVEL_METER && b.status & 0xF0 == 0xB0) {
            midi.send(&[b.status, b.data1, value.min(127)]);
        }
    }

    // Reported
    pub fn cue_frame(&self, e: &Engine) -> f64 {
        if e.has_track() { self.cue } else { f64::NAN }
    }

    pub fn volume(&self) -> f64 {
        self.volume
    }

    pub fn wants_fast_ticks(&self) -> bool {
        self.jog_held || self.bend != 0.0
    }
}

/// A level (dBFS, METER_FLOOR_DBFS - 0) as the meter's CC value.
fn meter_value(db: f64) -> u8 {
    (127.0 * ((db - METER_FLOOR_DBFS) / -METER_FLOOR_DBFS).clamp(0.0, 1.0)).round() as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Ring;
    use std::sync::Arc;

    fn deck_with(bindings: &[(u8, u8, bool, u32)]) -> (Deck, Engine) {
        let ring = Arc::new(Ring::new(48000, 2, 48000.0));
        let e = Engine::new(48000, 2, ring);
        let mut d = Deck::new(48000.0);
        let mut payload = (bindings.len() as u32).to_le_bytes().to_vec();
        for &(st, d1, hires, action) in bindings {
            let lamp = if (act::JUMP_A..act::JUMP_A + 8).contains(&action) { (action - act::JUMP_A + 1) as u8 } else { 0 };
            payload.extend([st, d1, hires as u8, lamp]);
            payload.extend(action.to_le_bytes());
        }
        d.set_bindings(&payload);
        (d, e)
    }

    #[test]
    fn tempo_fader_14_bits_over_the_range() {
        let (mut d, mut e) = deck_with(&[(0xB0, 0x00, true, act::TEMPO_FADER)]);
        d.tempo_range = 10;
        d.message(&mut e, 0.0, [0xB0, 0x00, 0x40]); // MSB 64, LSB 0: the middle
        d.message(&mut e, 0.0, [0xB0, 0x20, 0x00]);
        assert!((e.factor() - 1.0).abs() < 1e-12, "{}", e.factor());
        d.message(&mut e, 0.0, [0xB0, 0x00, 0x7F]);
        d.message(&mut e, 0.0, [0xB0, 0x20, 0x7F]); // the top: +range
        assert!((e.factor() - 1.1).abs() < 1e-3, "{}", e.factor());
        d.message(&mut e, 0.0, [0xB0, 0x00, 0x00]);
        d.message(&mut e, 0.0, [0xB0, 0x20, 0x00]); // the bottom: -range
        assert!((e.factor() - 0.9).abs() < 1e-12, "{}", e.factor());
    }

    #[test]
    fn gui_actions_flagged_and_note_off_read_as_release() {
        let (mut d, mut e) = deck_with(&[(0x90, 0x46, false, act::GUI + 7)]);
        assert_eq!(d.message(&mut e, 0.0, [0x90, 0x46, 0x7F]), (MIDI_EVENT_GUI, act::GUI + 7));
        assert_eq!(d.message(&mut e, 0.0, [0x91, 0x46, 0x7F]), (0, 0)); // another channel: not bound
    }

    #[test]
    fn volume_fader_and_jog_bend() {
        let (mut d, mut e) = deck_with(&[(0xB0, 0x13, false, act::VOLUME_FADER), (0xB0, 0x21, false, act::JOG_BEND)]);
        d.message(&mut e, 0.0, [0xB0, 0x13, 127]);
        assert_eq!(d.volume(), 1.0);
        d.message(&mut e, 0.0, [0xB0, 0x13, 0]);
        assert_eq!(d.volume(), 0.0);
        for _ in 0..10 {
            d.message(&mut e, 0.0, [0xB0, 0x21, 0x42]); // two steps on, ten times
        }
        assert_eq!(d.bend_ticks, 20.0);
        d.tick(&mut e);
        assert!(d.bend > 0.0);
    }

    #[test]
    fn jog_let_go_while_turning_is_held_until_it_stops() {
        let (mut d, mut e) = deck_with(&[(0x90, 0x36, false, act::JOG_TOUCH), (0xB0, 0x22, false, act::JOG_SCRATCH),
                                         (0xB0, 0x23, false, act::JOG_BEND)]);
        d.message(&mut e, clock::now(), [0x90, 0x36, 0x7F]);
        assert!(e.scratching);
        d.message(&mut e, clock::now(), [0xB0, 0x22, 0x3C]); // turned back (a backspin)
        d.message(&mut e, clock::now(), [0x90, 0x36, 0x00]); // let go while turning
        assert!(e.scratching, "let go while turning: still held");
        d.message(&mut e, clock::now(), [0xB0, 0x23, 0x3E]); // its rotation, now the bend CC: still the record
        assert_eq!(d.bend_ticks, 0.0);
        d.tick(&mut e);
        assert!(e.scratching);
        std::thread::sleep(std::time::Duration::from_millis(80)); // stopped: no message
        d.tick(&mut e);
        assert!(!e.scratching, "stopped: let go");
        d.message(&mut e, clock::now(), [0xB0, 0x23, 0x42]); // not held: a bend
        assert_eq!(d.bend_ticks, 2.0);
    }

    #[test]
    fn level_meter_scale() {
        assert_eq!(meter_value(0.0), 127);
        assert_eq!(meter_value(-24.0), 0);
        assert_eq!(meter_value(-12.0), 64);
        assert_eq!(meter_value(-40.0), 0);
    }

    #[test]
    fn midi_actions_measured_from_their_arrival() {
        let (mut d, mut e) = deck_with(&[(0x90, 0x0B, false, act::PLAY_PAUSE)]);
        e.command_at = 1.0; // a GUI command long ago
        d.message(&mut e, 123.5, [0x90, 0x0B, 0x7F]);
        assert_eq!(e.command_at, 123.5);
    }

    #[test]
    fn meter_binding_is_an_output() {
        let (mut d, mut e) = deck_with(&[(0xB0, 0x02, false, act::LEVEL_METER)]);
        assert_eq!(d.message(&mut e, 0.0, [0xB0, 0x02, 0x40]), (0, 0));
        assert_eq!(e.factor(), 1.0);
    }

    #[test]
    fn jog_let_go_at_rest_is_let_go_at_once() {
        let (mut d, mut e) = deck_with(&[(0x90, 0x36, false, act::JOG_TOUCH)]);
        d.message(&mut e, clock::now(), [0x90, 0x36, 0x7F]);
        d.message(&mut e, clock::now(), [0x90, 0x36, 0x00]);
        assert!(!e.scratching);
    }
}
