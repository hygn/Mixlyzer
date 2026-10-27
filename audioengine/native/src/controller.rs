//! A MIDI controller: its bindings (op::MIDI_MAP: messages to actions), its messages acted on in the engine thread
//! as they come (never through the GUI's Python), its lights and level meter. The deck's actions go to the deck, the
//! mixer channel's trim and EQ to its channel strip, its CUE, the headphones' mix and level, the master, the booth
//! and Sound Color FX to the mix (MixParams); the GUI's own actions (library, editor ...) are passed on to it (ipc.rs
//! MIDI_EVENTS).
//!
//! A binding reads a button (a note or a CC, pressed above 0), a fader (a CC, 7 or 14 bits: MSB at data1, LSB at
//! data1 + 32, or a pitch bend; inverted on demand), a knob (a relative CC in its encoding; bound to one direction,
//! a step an action) and a touch (a note or a CC). A JumpCUE pad on an empty slot asks the GUI to set one (an edit).
//!
//! The lights (each binding's lamp): PLAY and CUE (deck.rs lamps), a JumpCUE's pad lit while it is set (blinking
//! while its jump is armed), as rekordbox's hot cue pads; the channel's level meter (LEVEL_METER, a CC) as the GUI's
//! peak meter: the deck's heard peak times its gain, the GUI meter's segments lit (a DJM-A9's, -30 to +12 dB: -42 to
//! 0 dBFS) as their share of 0 - 127, falling 24 dB/s.

use crate::clock;
use crate::deck::Slot;
use crate::ipc::{Control, MIDI_EVENT_GUI, MIDI_EVENT_HOTCUE_SET, MIDI_EVENT_RELATIVE, Reader, act, bind};
use crate::midi::Midi;
use crate::ipc::mix as keys;
use crate::colorfx::ColorType;
use crate::mixer::MixParams;

// a binding's lamp (ipc.rs op::MIDI_MAP): 1 - 26 the JumpCUE A - Z
pub const LAMP_PLAY: u8 = 32;
pub const LAMP_CUE: u8 = 33;
pub const LAMP_CHANNEL_CUE: u8 = 34; // the mixer channel's headphones CUE: lit while it is on
// the mixer's own, lit while on: MONO SPLIT, the Sound Color FX picked (LAMP_COLOR + its code)
pub const LAMP_HP_SPLIT: u8 = 40;
pub const LAMP_COLOR: u8 = 45;
pub const LAMP_VINYL: u8 = 52; // the jog in VINYL mode (deck.rs lights; LAMP_COLOR's seven before it)
// a knob's steps acted on from one message at most (a fast turn)
const RELATIVE_STEPS_MAX: u32 = 16;
// the level meter: ui/track_info_panel.py VerticalPeakMeter's scale (its 0 dB, its segments from -30 dB, 3 dB each,
// the top at 0 dBFS) and fall; sent at most this often
const METER_REFERENCE_DBFS: f64 = -12.0;
const METER_SEGMENTS: u32 = 15;
const METER_FLOOR_DBFS: f64 = METER_REFERENCE_DBFS - 33.0; // below the lowest segment
const METER_FALL_DB_PER_SEC: f64 = 24.0;
const METER_INTERVAL_SEC: f64 = 0.02;

struct Binding {
    status: u8,
    data1: u8,
    hires: bool,
    // a knob turned (a relative CC): +1 only clockwise, -1 only counterclockwise, each step an action; 0 any value
    dir: i8,
    invert: bool, // a fader read the other way up, a knob's turns the other way round
    encoding: u8, // a relative CC's (ipc.rs bind::ENC_*)
    lamp: u8,
    action: u32,
    msb: u8,
    lsb: u8,
}

#[derive(Clone, Copy)]
struct Meter {
    db: f64,
    sent: Option<u8>,
    at: f64,
}

pub struct Controller {
    midi: Option<Midi>,
    // a port being opened on a thread of its own (an open takes tens of ms: the engine thread never waits for it)
    opening: Option<std::sync::mpsc::Receiver<Result<Midi, String>>>,
    port: String, // the input open or being opened ("": none)
    closing: Option<std::thread::JoinHandle<()>>, // the last input let go, until it is (an open waits for it)
    pub midi_state: u32,
    bindings: Vec<Binding>,
    lamps_sent: Option<u64>,
    meter_gain: f64,
    meter: Meter,
    last_tick: f64,
}

impl Controller {
    pub fn new() -> Controller {
        Controller {
            midi: None,
            opening: None,
            port: String::new(),
            closing: None,
            midi_state: 0,
            bindings: vec![],
            lamps_sent: None,
            meter_gain: 1.0,
            meter: Meter { db: METER_FLOOR_DBFS, sent: None, at: 0.0 },
            last_tick: clock::now(),
        }
    }

    /// The input ``name`` opened ("": none), on a thread of its own: taken by the engine thread when it is
    /// (take_opened).
    pub fn open_midi(&mut self, name: &str, wake: Option<std::thread::Thread>) {
        // the input already open or being opened: kept (asked again, it is not let go and taken anew, which the
        // system may refuse while the old port is still being closed: MMSYSERR_ALLOCATED)
        if !name.is_empty() && name == self.port && (self.midi.is_some() || self.opening.is_some()) {
            return;
        }
        self.close_midi();
        self.opening = None;   // one asked for before: dropped when it comes
        if name.is_empty() {
            return;
        }
        let (tx, rx) = std::sync::mpsc::channel();
        self.port = name.to_string();
        let name = name.to_string();
        let closing = self.closing.take();
        std::thread::spawn(move || {
            if let Some(c) = closing {
                let _ = c.join(); // the port let go first
            }
            let _ = tx.send(Midi::open(&name, wake));
        });
        self.opening = Some(rx);
    }

    /// The input opened meanwhile, when it is.
    fn take_opened(&mut self) {
        let Some(rx) = self.opening.as_ref() else { return };
        let result = match rx.try_recv() {
            Ok(r) => r,
            Err(std::sync::mpsc::TryRecvError::Empty) => return,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => Err("the MIDI input's opening thread ended".into()),
        };
        self.opening = None;
        match result {
            Ok(m) => {
                log_line!("MIDI input: {}", m.name);
                self.midi = Some(m);
                self.midi_state = 1;
                self.lamps_sent = None;   // its lights sent whole at the next tick
            }
            Err(err) => {
                log_line!("{err}");
                self.midi_state = 2;
            }
        }
    }

    pub fn close_midi(&mut self) {
        self.send_lamps(0, None); // the controller's lights out
        self.send_meter(0);
        // let go on a thread of its own (closing a port takes as long as opening one); the next open waits for it
        if let Some(m) = self.midi.take() {
            let before = self.closing.take();
            self.closing = Some(std::thread::spawn(move || {
                if let Some(b) = before {
                    let _ = b.join();
                }
                drop(m)
            }));
        }
        self.port.clear();
        self.midi_state = 0;
        self.lamps_sent = None;
    }

    pub fn set_bindings(&mut self, payload: &[u8]) {
        let mut r = Reader::new(payload);
        let n = r.u32().unwrap_or(0) as u64;
        let n = r.count(n, 8);
        self.bindings.clear();
        for _ in 0..n {
            let (Some(status), Some(data1), Some(flags), Some(lamp), Some(action)) =
                (r.u8(), r.u8(), r.u8(), r.u8(), r.u32())
            else {
                break;
            };
            let dir = if flags & bind::CLOCKWISE != 0 { 1 } else if flags & bind::COUNTERCLOCKWISE != 0 { -1 } else { 0 };
            self.bindings.push(Binding {
                status,
                data1,
                hires: flags & bind::HIRES != 0,
                dir,
                invert: flags & bind::INVERT != 0,
                encoding: (flags & bind::ENCODING) >> bind::ENCODING_SHIFT,
                lamp,
                action,
                msb: 0,
                lsb: 0,
            });
        }
        self.lamps_sent = None;
        self.meter.sent = None;
    }

    pub fn set_meter_gain(&mut self, gain: f64) {
        self.meter_gain = if gain.is_finite() { gain.max(0.0) } else { 1.0 };
    }

    /// The messages come since the last call: each acted on, and into the GUI's ring.
    pub fn read_midi(&mut self, slot: &mut Slot, mix: &MixParams, ctl: &Control) {
        self.take_opened();
        let Some(midi) = self.midi.as_ref() else { return };
        for (at, msg) in midi.drain() {
            let (flags, gui_action, steps) = self.message(slot, mix, at, msg);
            ctl.put_midi_event(msg, flags, gui_action, steps);
        }
    }

    /// One message acted on: (flags, the action passed to the GUI, a knob's steps) for the GUI's ring.
    fn message(&mut self, slot: &mut Slot, mix: &MixParams, at: f64, msg: [u8; 3]) -> (u8, u32, i32) {
        let [status, d1, d2] = msg;
        let mut flags = 0u8;
        let mut gui_action = 0u32;
        let mut gui_steps = 0i32;
        for i in 0..self.bindings.len() {
            let b = &mut self.bindings[i];
            if b.status != status || b.action == act::LEVEL_METER {
                continue; // an output
            }
            // the value: a pitch bend's 14 bits (its data 1 the LSB), a 14-bit CC's MSB / LSB, else data 2
            let (mut value, top) = if status & 0xF0 == 0xE0 {
                ((d2 as u32) << 7 | d1 as u32, 16383.0)
            } else if b.data1 == d1 {
                b.msb = d2;
                if b.hires { ((d2 as u32) << 7 | b.lsb as u32, 16383.0) } else { (d2 as u32, 127.0) }
            } else if b.hires && d1 == b.data1.wrapping_add(32) && status & 0xF0 == 0xB0 {
                b.lsb = d2;
                ((b.msb as u32) << 7 | d2 as u32, 16383.0)
            } else {
                continue;
            };
            if b.invert {
                value = top as u32 - value.min(top as u32);
            }
            let mut steps = if status & 0xF0 == 0xB0 { relative_steps(d2, b.encoding) } else { 0 };
            if b.invert {
                steps = -steps;
            }
            let (code, dir) = (b.action, b.dir);
            if code >= act::GUI {
                if dir != 0 && (steps == 0 || steps.signum() != dir as i32) {
                    continue;
                }
                flags |= MIDI_EVENT_GUI | if dir != 0 { MIDI_EVENT_RELATIVE } else { 0 };
                gui_action = code;
                gui_steps = steps;
                continue;
            }
            // the mixer's: the headphones, the channel's CUE, the master, the booth, Sound Color FX
            let unit = value as f64 / top;
            let pressed = d2 > 0;
            match code {
                act::MASTER_LEVEL => {
                    mix.set(keys::MASTER_LEVEL, unit);
                    continue;
                }
                act::BOOTH_LEVEL => {
                    mix.set(keys::BOOTH_LEVEL, unit);
                    continue;
                }
                act::BOOTH_EQ_HI => {
                    mix.set(keys::BOOTH_EQ_HI, unit);
                    continue;
                }
                act::BOOTH_EQ_LOW => {
                    mix.set(keys::BOOTH_EQ_LOW, unit);
                    continue;
                }
                act::COLOR_PARAM => {
                    mix.set(keys::COLOR_PARAM, unit);
                    continue;
                }
                c if (act::COLOR_SELECT..act::COLOR_SELECT + ColorType::COUNT as u32).contains(&c) => {
                    // the effect's button; pressed again it turns the effect off
                    if pressed {
                        let code = (c - act::COLOR_SELECT) as f64;
                        let off = ColorType::Off as u8 as f64;
                        mix.set(keys::COLOR_TYPE, if mix.get(keys::COLOR_TYPE) == code { off } else { code });
                    }
                    continue;
                }
                act::HEADPHONES_SPLIT => {
                    if pressed {
                        mix.step(keys::HP_SPLIT, 1);
                    }
                    continue;
                }
                act::HEADPHONES_MIX => {
                    mix.set_cue_mix(value as f64 / top);
                    continue;
                }
                act::HEADPHONES_LEVEL => {
                    mix.set_cue_level(value as f64 / top);
                    continue;
                }
                act::CHANNEL_CUE => {
                    if d2 > 0 {
                        mix.strip.toggle_cue(); // as a DJM's CUE buttons: a press turns it on or off
                    }
                    continue;
                }
                _ => {}
            }
            // the response measured (play / pause / seek heard) from this message's arrival, as a GUI command's from
            // its sending
            slot.engine.command_at = at;
            if dir != 0 {
                // a knob turned one way: only that direction, a step of it an action
                if steps == 0 || steps.signum() != dir as i32 {
                    continue;
                }
                for _ in 0..steps.unsigned_abs().min(RELATIVE_STEPS_MAX) {
                    slot.deck.action(&mut slot.engine, code, true);
                }
                continue;
            }
            // a JumpCUE pad on an empty slot: the GUI sets the JumpCUE there (an edit), as a CDJ's pad sets a hot cue
            if (act::JUMP_A..act::JUMP_A + 8).contains(&code) && d2 > 0
                && !slot.deck.jumpcue_set((code - act::JUMP_A) as usize) && slot.engine.has_track()
            {
                flags |= MIDI_EVENT_HOTCUE_SET;
                gui_action = code;
                continue;
            }
            slot.deck.midi_value(&mut slot.engine, code, (value, top), steps, d2, at);
        }
        (flags, gui_action, gui_steps)
    }

    /// Once a tick: the lights and the level meter (where they changed).
    pub fn tick(&mut self, slot: &Slot, mix: &MixParams) {
        let now = clock::now();
        let dt = (now - self.last_tick).max(0.0);
        self.last_tick = now;
        if self.midi.is_none() {
            return;
        }
        let lights = slot.deck.lights(&slot.engine, now)
            | (mix.strip.cue_on() as u64) << LAMP_CHANNEL_CUE
            | Self::mixer_lights(mix);
        if self.lamps_sent != Some(lights) {
            let before = self.lamps_sent;
            self.send_lamps(lights, before);
            self.lamps_sent = Some(lights);
        }
        // the level meter: the GUI meter's reading
        let peak = slot.engine.heard_peak_dbfs() + 20.0 * self.meter_gain.max(1e-9).log10();
        let m = &mut self.meter;
        m.db = peak.max(m.db - METER_FALL_DB_PER_SEC * dt).clamp(METER_FLOOR_DBFS, 0.0);
        if now - m.at >= METER_INTERVAL_SEC {
            let value = meter_value(m.db);
            if m.sent != Some(value) {
                m.sent = Some(value);
                m.at = now;
                self.send_meter(value);
            }
        }
    }

    /// The mixer's own lights.
    fn mixer_lights(mix: &MixParams) -> u64 {
        let color = mix.get(keys::COLOR_TYPE) as u64;
        ((mix.get(keys::HP_SPLIT) != 0.0) as u64) << LAMP_HP_SPLIT
            | if color < ColorType::COUNT as u64 { 1u64 << (LAMP_COLOR as u64 + color) } else { 0 }
    }

    /// The bindings' lights as ``lights`` (Deck::lights and the mixer's), those that changed since ``before`` (None:
    /// all).
    fn send_lamps(&mut self, lights: u64, before: Option<u64>) {
        let Some(midi) = self.midi.as_mut() else { return };
        for b in self.bindings.iter().filter(|b| b.action < act::GUI) {
            if b.lamp == 0 || !matches!(b.status & 0xF0, 0x90 | 0xB0) {
                continue;
            }
            let bit = if b.lamp <= 26 { b.lamp - 1 } else { b.lamp };
            let lit = lights.checked_shr(bit as u32).unwrap_or(0) & 1 != 0;
            if before.is_some_and(|was| (was.checked_shr(bit as u32).unwrap_or(0) & 1 != 0) == lit) {
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
}

/// A relative CC's steps (+ clockwise) in its ``encoding`` (bind::ENC_*); 0: none.
fn relative_steps(d2: u8, encoding: u8) -> i32 {
    let v = d2 as i32;
    match encoding {
        bind::ENC_OFFSET => v - 64,
        bind::ENC_TWOS => if v < 64 { v } else { v - 128 },
        bind::ENC_SIGN => if v & 0x40 != 0 { -(v & 0x3F) } else { v & 0x3F },
        // ENC_AUTO: two's complement near 0 (1 .. 31 on, 97 .. 127 back), else 64 + steps (a turn under 32 steps a
        // message tells them apart; sign and magnitude must be named)
        _ => match v {
            1..=31 => v,
            97..=127 => v - 128,
            32..=96 => v - 64,
            _ => 0,
        },
    }
}

/// A level (dBFS) as the meter's CC value: the GUI meter's segments lit (each from -30 dB on, 3 dB apart, lit at
/// its level and above) as their share of 127.
fn meter_value(db: f64) -> u8 {
    let lit = (0..METER_SEGMENTS).filter(|&k| db - METER_REFERENCE_DBFS >= -30.0 + 3.0 * k as f64 - 1e-6).count();
    (127.0 * lit as f64 / METER_SEGMENTS as f64).round() as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Ring;
    use std::sync::Arc;

    fn rig(bindings: &[(u8, u8, u8, u32)]) -> (Controller, Slot, Arc<MixParams>) {
        let port = Arc::new(crate::device::Port::default());
        let params = Arc::new(MixParams::new(2));
        let slot = Slot::new(48000, 2, Arc::new(Ring::new(48000, 4, 48000.0, port)), params.strip.clone());
        let mut c = Controller::new();
        let mut payload = (bindings.len() as u32).to_le_bytes().to_vec();
        for &(st, d1, flags, action) in bindings {
            let lamp = if (act::JUMP_A..act::JUMP_A + 8).contains(&action) { (action - act::JUMP_A + 1) as u8 } else { 0 };
            payload.extend([st, d1, flags, lamp]);
            payload.extend(action.to_le_bytes());
        }
        c.set_bindings(&payload);
        (c, slot, params)
    }

    #[test]
    fn tempo_fader_14_bits_over_the_range() {
        let (mut c, mut d, m) = rig(&[(0xB0, 0x00, bind::HIRES, act::TEMPO_FADER)]);
        d.deck.tempo_range = 10;
        c.message(&mut d, &m, 0.0, [0xB0, 0x00, 0x40]); // MSB 64, LSB 0: the middle
        c.message(&mut d, &m, 0.0, [0xB0, 0x20, 0x00]);
        assert!((d.engine.factor() - 1.0).abs() < 1e-12);
        c.message(&mut d, &m, 0.0, [0xB0, 0x00, 0x7F]);
        c.message(&mut d, &m, 0.0, [0xB0, 0x20, 0x7F]); // the top: +range
        assert!((d.engine.factor() - 1.1).abs() < 1e-3);
        c.message(&mut d, &m, 0.0, [0xB0, 0x00, 0x00]);
        c.message(&mut d, &m, 0.0, [0xB0, 0x20, 0x00]); // the bottom: -range
        assert!((d.engine.factor() - 0.9).abs() < 1e-12);
    }

    #[test]
    fn gui_actions_flagged() {
        let (mut c, mut d, m) = rig(&[(0x90, 0x46, 0, act::GUI + 7)]);
        assert_eq!(c.message(&mut d, &m, 0.0, [0x90, 0x46, 0x7F]), (MIDI_EVENT_GUI, act::GUI + 7, 0));
        assert_eq!(c.message(&mut d, &m, 0.0, [0x91, 0x46, 0x7F]), (0, 0, 0)); // another channel: not bound
    }

    #[test]
    fn volume_fader_and_jog_bend() {
        let (mut c, mut d, m) = rig(&[(0xB0, 0x13, 0, act::VOLUME_FADER), (0xB0, 0x21, 0, act::JOG_BEND)]);
        c.message(&mut d, &m, 0.0, [0xB0, 0x13, 127]);
        assert_eq!(d.deck.volume(), 1.0);
        c.message(&mut d, &m, 0.0, [0xB0, 0x13, 0]);
        assert_eq!(d.deck.volume(), 0.0);
        for _ in 0..10 {
            c.message(&mut d, &m, 0.0, [0xB0, 0x21, 0x42]); // two steps on, ten times
        }
        let s = &mut d;
        s.deck.tick(&mut s.engine);
        assert!(s.deck.wants_fast_ticks(), "bent");
    }

    #[test]
    fn jog_let_go_while_turning_is_held_until_it_stops() {
        let (mut c, mut d, m) = rig(&[(0x90, 0x36, 0, act::JOG_TOUCH), (0xB0, 0x22, 0, act::JOG_SCRATCH),
                                      (0xB0, 0x23, 0, act::JOG_BEND)]);
        c.message(&mut d, &m, clock::now(), [0x90, 0x36, 0x7F]);
        assert!(d.engine.scratching);
        c.message(&mut d, &m, clock::now(), [0xB0, 0x22, 0x3C]); // turned back (a backspin)
        c.message(&mut d, &m, clock::now(), [0x90, 0x36, 0x00]); // let go while turning
        assert!(d.engine.scratching, "let go while turning: still held");
        c.message(&mut d, &m, clock::now(), [0xB0, 0x23, 0x3E]); // its rotation, now the bend CC: still the record
        let s = &mut d;
        s.deck.tick(&mut s.engine);
        assert!(s.engine.scratching);
        std::thread::sleep(std::time::Duration::from_millis(80)); // stopped: no message
        s.deck.tick(&mut s.engine);
        assert!(!s.engine.scratching, "stopped: let go");
    }

    #[test]
    fn jog_let_go_at_rest_is_let_go_at_once() {
        let (mut c, mut d, m) = rig(&[(0x90, 0x36, 0, act::JOG_TOUCH)]);
        c.message(&mut d, &m, clock::now(), [0x90, 0x36, 0x7F]);
        c.message(&mut d, &m, clock::now(), [0x90, 0x36, 0x00]);
        assert!(!d.engine.scratching);
    }

    #[test]
    fn level_meter_scale() {
        // as the GUI meter: 0 dBFS all 15 segments, -12 dBFS (0 dB) 11, -42 dBFS (-30 dB) the lowest, under it none
        assert_eq!(meter_value(0.0), 127);
        assert_eq!(meter_value(-12.0), 93);
        assert_eq!(meter_value(-13.0), 85);
        assert_eq!(meter_value(-42.0), 8);
        assert_eq!(meter_value(-43.0), 0);
        assert_eq!(meter_value(METER_FLOOR_DBFS), 0);
    }

    #[test]
    fn midi_actions_measured_from_their_arrival() {
        let (mut c, mut d, m) = rig(&[(0x90, 0x0B, 0, act::PLAY_PAUSE)]);
        d.engine.command_at = 1.0; // a GUI command long ago
        c.message(&mut d, &m, 123.5, [0x90, 0x0B, 0x7F]);
        assert_eq!(d.engine.command_at, 123.5);
    }

    #[test]
    fn meter_binding_is_an_output() {
        let (mut c, mut d, m) = rig(&[(0xB0, 0x02, 0, act::LEVEL_METER)]);
        assert_eq!(c.message(&mut d, &m, 0.0, [0xB0, 0x02, 0x40]), (0, 0, 0));
        assert_eq!(d.engine.factor(), 1.0);
    }

    #[test]
    fn relative_encodings() {
        assert_eq!(relative_steps(0x41, bind::ENC_AUTO), 1); // Pioneer jog: 64 + steps
        assert_eq!(relative_steps(0x3F, bind::ENC_AUTO), -1);
        assert_eq!(relative_steps(0x01, bind::ENC_AUTO), 1); // Pioneer BROWSE: two's complement
        assert_eq!(relative_steps(0x7F, bind::ENC_AUTO), -1);
        assert_eq!(relative_steps(0x62, bind::ENC_AUTO), -30);
        assert_eq!(relative_steps(0x41, bind::ENC_SIGN), -1); // sign and magnitude
        assert_eq!(relative_steps(0x03, bind::ENC_SIGN), 3);
        assert_eq!(relative_steps(0x40, bind::ENC_AUTO), 0);
        assert_eq!(relative_steps(0x7F, bind::ENC_TWOS), -1);
        assert_eq!(relative_steps(0x3F, bind::ENC_OFFSET), -1);
    }

    #[test]
    fn knob_bound_by_direction() {
        let next = act::GUI + 1;
        let previous = act::GUI + 2;
        let (mut c, mut d, m) =
            rig(&[(0xB6, 0x40, bind::CLOCKWISE, next), (0xB6, 0x40, bind::COUNTERCLOCKWISE, previous)]);
        assert_eq!(c.message(&mut d, &m, 0.0, [0xB6, 0x40, 0x01]), (MIDI_EVENT_GUI | MIDI_EVENT_RELATIVE, next, 1));
        assert_eq!(c.message(&mut d, &m, 0.0, [0xB6, 0x40, 0x7D]), (MIDI_EVENT_GUI | MIDI_EVENT_RELATIVE, previous, -3));
    }

    #[test]
    fn pitch_bend_fader_and_inverted() {
        let (mut c, mut d, m) = rig(&[(0xE0, 0x00, 0, act::TEMPO_FADER)]);
        d.deck.tempo_range = 10;
        c.message(&mut d, &m, 0.0, [0xE0, 0x7F, 0x7F]); // 16383: the top
        assert!((d.engine.factor() - 1.1).abs() < 1e-3);
        let (mut c, mut d, m) = rig(&[(0xB0, 0x00, bind::INVERT, act::TEMPO_FADER)]);
        d.deck.tempo_range = 10;
        c.message(&mut d, &m, 0.0, [0xB0, 0x00, 0x7F]); // inverted: the top is the bottom
        assert!((d.engine.factor() - 0.9).abs() < 1e-3);
    }

    #[test]
    fn jumpcue_pad_without_a_track_does_nothing() {
        let (mut c, mut d, m) = rig(&[(0x97, 0x00, 0, act::JUMP_A)]);
        assert_eq!(c.message(&mut d, &m, 0.0, [0x97, 0x00, 0x7F]), (0, 0, 0));
    }

    #[test]
    fn color_fx_buttons_and_jog_mode() {
        let (mut c, mut d, m) = rig(&[
            (0x94, 0x63, 0, act::COLOR_SELECT + ColorType::Crush as u32),
            (0x90, 0x10, 0, act::JOG_MODE),
            (0x90, 0x36, 0, act::JOG_TOUCH),
        ]);
        c.message(&mut d, &m, 0.0, [0x94, 0x63, 0x7F]); // CRUSH picked
        c.message(&mut d, &m, 0.0, [0x94, 0x63, 0x00]); // let go: kept
        assert_eq!(m.get(keys::COLOR_TYPE), ColorType::Crush as u8 as f64);
        c.message(&mut d, &m, 0.0, [0x94, 0x63, 0x7F]); // pressed again: off
        assert_eq!(m.get(keys::COLOR_TYPE), ColorType::Off as u8 as f64);

        assert!(d.deck.jog_vinyl(), "VINYL at the start");
        c.message(&mut d, &m, 0.0, [0x90, 0x36, 0x7F]); // the top touched: the record held
        assert!(d.deck.jog_held());
        c.message(&mut d, &m, 0.0, [0x90, 0x10, 0x7F]); // JOG MODE: CDJ, the record let go
        c.message(&mut d, &m, 0.0, [0x90, 0x10, 0x00]);
        assert!(!d.deck.jog_vinyl() && !d.deck.jog_held());
        c.message(&mut d, &m, 0.0, [0x90, 0x36, 0x00]);
        c.message(&mut d, &m, 0.0, [0x90, 0x36, 0x7F]); // CDJ: the top's touch ignored
        assert!(!d.deck.jog_held());
        c.message(&mut d, &m, 0.0, [0x90, 0x36, 0x00]);
        c.message(&mut d, &m, 0.0, [0x90, 0x10, 0x7F]); // VINYL again
        assert!(d.deck.jog_vinyl());
    }

    #[test]
    fn mixer_actions() {
        let (mut c, mut d, m) = rig(&[
            (0xB0, 0x0F, 0, act::EQ_LOW),
            (0x90, 0x54, 0, act::CHANNEL_CUE),
            (0xB6, 0x0C, 0, act::HEADPHONES_MIX),
            (0x96, 0x0D, 0, act::HEADPHONES_SPLIT),
        ]);
        c.message(&mut d, &m, 0.0, [0xB0, 0x0F, 0]); // EQ LOW to the bottom: a kill
        let x: Vec<f32> = (0..9600).map(|i| (2.0 * std::f64::consts::PI * 50.0 * (i / 2) as f64 / 48000.0).sin() as f32).collect();
        // through the mixer channel (its strip as the callback runs it)
        let lane: Vec<f32> = x.chunks_exact(2).flat_map(|f| [f[0], f[1], 0.0, 0.0]).collect();
        let (mut pfl, mut post) = (vec![0f32; x.len()], vec![0f32; x.len()]);
        crate::mixer::Strip::new(48000.0, 1).process(&lane, 4800, 2, &d.engine.strip, &m.read(), &mut pfl, &mut post);
        let rms = (post[4800..].iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / 4800.0).sqrt();
        assert!(rms < 0.05, "50 Hz with the low killed: {rms}");
        c.message(&mut d, &m, 0.0, [0x90, 0x54, 0x7F]);
        c.message(&mut d, &m, 0.0, [0x90, 0x54, 0x00]); // let go: still on
        assert!(m.strip.cue_on());
        c.message(&mut d, &m, 0.0, [0x90, 0x54, 0x7F]); // pressed again: off
        assert!(!m.strip.cue_on());
        c.message(&mut d, &m, 0.0, [0xB6, 0x0C, 127]);
        assert_eq!(m.read().hp.0, 1.0);
        c.message(&mut d, &m, 0.0, [0x96, 0x0D, 0x7F]); // MONO SPLIT on
        assert!(m.read().hp_split);
    }
}
