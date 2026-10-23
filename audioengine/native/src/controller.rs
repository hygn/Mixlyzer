//! A MIDI controller: its bindings (op::MIDI_MAP: messages to actions), its messages acted on in the engine thread
//! as they come (never through the GUI's Python), its lights and level meters. An action carries its deck (its
//! mixer channel) from act::DECK_SHIFT up: the deck's actions go to that deck (made when first used), a mixer
//! channel's trim and EQ to its deck's channel strip, its CUE and the crossfader, the headphones' mix and level to
//! the mix (MixParams); the GUI's own actions (library, editor ...) are passed on to it (ipc.rs MIDI_EVENTS).
//!
//! A binding reads a button (a note or a CC, pressed above 0), a fader (a CC, 7 or 14 bits: MSB at data1, LSB at
//! data1 + 32, or a pitch bend; inverted on demand), a knob (a relative CC in its encoding; bound to one direction,
//! a step an action) and a touch (a note or a CC). A JumpCUE pad on an empty slot asks the GUI to set one (an edit).
//!
//! The lights (each binding's lamp, of its deck): PLAY and CUE (deck.rs lamps), a JumpCUE's pad lit while it is set
//! (blinking while its jump is armed), as rekordbox's hot cue pads; a channel's level meter (LEVEL_METER, a CC) as
//! the GUI's peak meter: the deck's heard peak times its gain, -24 - 0 dBFS as 0 - 127, falling 24 dB/s.

use crate::clock;
use crate::deck::Decks;
use crate::ipc::{Control, MAX_DECKS, MIDI_EVENT_GUI, MIDI_EVENT_HOTCUE_SET, MIDI_EVENT_RELATIVE, Reader, act, bind};
use crate::midi::Midi;
use crate::ipc::mix as keys;
use crate::beatfx::BeatType;
use crate::colorfx::ColorType;
use crate::mixer::MixParams;

// a binding's lamp (ipc.rs op::MIDI_MAP): 1 - 26 the JumpCUE A - Z
pub const LAMP_PLAY: u8 = 32;
pub const LAMP_CUE: u8 = 33;
pub const LAMP_CHANNEL_CUE: u8 = 34; // the mixer channel's headphones CUE: lit while it is on
pub const LAMP_CHANNEL_CUE_B: u8 = 35; // its headphones B CUE
pub const LAMP_BEAT_FX: u8 = 36; // Beat FX on: lit (every deck's binding of it)
// the mixer's own, lit while on (every deck's binding of them): FX FREQUENCY LOW, MID, HI; MONO SPLIT A, B; the
// master's CUE A, B; TAP (a BPM tapped), the Sound Color FX picked (LAMP_COLOR + its code)
pub const LAMP_BEAT_BAND: u8 = 37;
pub const LAMP_HP_A_SPLIT: u8 = 40;
pub const LAMP_HP_B_SPLIT: u8 = 41;
pub const LAMP_MASTER_CUE_A: u8 = 42;
pub const LAMP_MASTER_CUE_B: u8 = 43;
pub const LAMP_BEAT_TAP: u8 = 44;
pub const LAMP_COLOR: u8 = 45;
pub const LAMP_VINYL: u8 = 52; // a deck's jog in VINYL mode (deck.rs lights; LAMP_COLOR's seven before it)
// TAP: taps further apart than this start anew; the BPM from the last TAP_KEEP of them (more than two)
const TAP_GAP_SEC: f64 = 2.0;
const TAP_KEEP: usize = 8;
// the TIME knob turned (a relative control): its time times this a step
const TIME_STEP: f64 = 1.03;
// a knob's steps acted on from one message at most (a fast turn)
const RELATIVE_STEPS_MAX: u32 = 16;
// the level meter: ui/track_info_panel.py VerticalPeakMeter's scale and fall; sent at most this often
const METER_FLOOR_DBFS: f64 = -24.0;
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
    action: u32, // its deck from act::DECK_SHIFT up
    msb: u8,
    lsb: u8,
}

impl Binding {
    /// Its deck (0: deck 1; the GUI's actions and the crossfader: 0).
    fn deck(&self) -> usize {
        (self.action >> act::DECK_SHIFT) as usize
    }

    /// Its action without its deck.
    fn code(&self) -> u32 {
        self.action & act::CODE_MASK
    }
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
    lamps_sent: [Option<u64>; MAX_DECKS],
    meter_gain: f64,
    meters: [Meter; MAX_DECKS],
    last_tick: f64,
    taps: Vec<f64>,        // TAP's times
    xpad_turned_on: bool,  // the X-PAD's touch turned Beat FX on (its release turns it off)
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
            lamps_sent: [None; MAX_DECKS],
            meter_gain: 1.0,
            meters: [Meter { db: METER_FLOOR_DBFS, sent: None, at: 0.0 }; MAX_DECKS],
            last_tick: clock::now(),
            taps: Vec::with_capacity(TAP_KEEP),
            xpad_turned_on: false,
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
                self.lamps_sent = [None; MAX_DECKS];   // its lights sent whole at the next tick
            }
            Err(err) => {
                log_line!("{err}");
                self.midi_state = 2;
            }
        }
    }

    pub fn close_midi(&mut self) {
        for k in 0..MAX_DECKS {
            self.send_lamps(k, 0, None); // the controller's lights out
            self.send_meter(k, 0);
        }
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
        self.lamps_sent = [None; MAX_DECKS];
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
        self.lamps_sent = [None; MAX_DECKS];
        for m in self.meters.iter_mut() {
            m.sent = None;
        }
    }

    pub fn set_meter_gain(&mut self, gain: f64) {
        self.meter_gain = if gain.is_finite() { gain.max(0.0) } else { 1.0 };
    }

    /// The messages come since the last call: each acted on, and into the GUI's ring.
    pub fn read_midi(&mut self, decks: &mut Decks, mix: &MixParams, ctl: &Control) {
        self.take_opened();
        let Some(midi) = self.midi.as_ref() else { return };
        for (at, msg) in midi.drain() {
            let (flags, gui_action, steps) = self.message(decks, mix, at, msg);
            ctl.put_midi_event(msg, flags, gui_action, steps);
        }
    }

    /// One message acted on: (flags, the action passed to the GUI, a knob's steps) for the GUI's ring.
    fn message(&mut self, decks: &mut Decks, mix: &MixParams, at: f64, msg: [u8; 3]) -> (u8, u32, i32) {
        let [status, d1, d2] = msg;
        let mut flags = 0u8;
        let mut gui_action = 0u32;
        let mut gui_steps = 0i32;
        for i in 0..self.bindings.len() {
            let b = &mut self.bindings[i];
            if b.status != status || b.code() == act::LEVEL_METER {
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
            let (code, k, dir) = (b.code(), b.deck(), b.dir);
            if code >= act::GUI {
                if dir != 0 && (steps == 0 || steps.signum() != dir as i32) {
                    continue;
                }
                flags |= MIDI_EVENT_GUI | if dir != 0 { MIDI_EVENT_RELATIVE } else { 0 };
                gui_action = code;
                gui_steps = steps;
                continue;
            }
            if k >= MAX_DECKS {
                continue;
            }
            // the mixer's: the crossfader, the headphones, a channel's CUE, its COLOR knob, the master, the booth, Sound
            // Color FX and Beat FX (a channel's deck need not exist)
            let unit = value as f64 / top;
            let pressed = d2 > 0;
            match code {
                act::CHANNEL_CUE_B => {
                    if d2 > 0 {
                        mix.toggle_cue_bus(k, 1);
                    }
                    continue;
                }
                act::HEADPHONES_B_MIX => {
                    mix.set(keys::HP_B_MIX, unit);
                    continue;
                }
                act::HEADPHONES_B_LEVEL => {
                    mix.set(keys::HP_B_LEVEL, unit);
                    continue;
                }
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
                act::BEAT_FX_TIME => {
                    self.beat_time(mix, unit, steps);
                    continue;
                }
                act::BEAT_FX_TAP => {
                    if pressed {
                        self.tap(mix, at);
                    }
                    continue;
                }
                act::BEAT_FX_AUTO => {
                    if pressed {
                        mix.set(keys::BEAT_BPM, 0.0);
                        self.taps.clear();
                    }
                    continue;
                }
                act::BEAT_FX_XPAD => {
                    // MOBIUS's waveform, REVERB's filter; the others' BEAT, the pad's width split among its values
                    let kind = BeatType::from_code(mix.get(keys::BEAT_TYPE) as u8);
                    if matches!(kind, BeatType::Mobius | BeatType::Reverb) {
                        mix.set(keys::BEAT_XPAD, unit);
                    } else {
                        let list = kind.beats();
                        mix.set(keys::BEAT_BEAT, list[((unit * list.len() as f64) as usize).min(list.len() - 1)]);
                        mix.set(keys::BEAT_TIME, 0.0);
                    }
                    continue;
                }
                act::BEAT_FX_XPAD_TOUCH => {
                    if pressed && mix.get(keys::BEAT_ON) == 0.0 {
                        mix.set(keys::BEAT_ON, 1.0);
                        self.xpad_turned_on = true;
                    } else if !pressed {
                        if std::mem::take(&mut self.xpad_turned_on) {
                            mix.set(keys::BEAT_ON, 0.0);
                        }
                        if BeatType::from_code(mix.get(keys::BEAT_TYPE) as u8) == BeatType::Reverb {
                            mix.set(keys::BEAT_XPAD, -1.0); // its filter off with the finger
                        }
                    }
                    continue;
                }
                c if (act::BEAT_FX_BAND..act::BEAT_FX_BAND + 3).contains(&c) => {
                    if pressed {
                        mix.toggle_band((c - act::BEAT_FX_BAND) as usize);
                    }
                    continue;
                }
                act::HEADPHONES_A_SPLIT | act::HEADPHONES_B_SPLIT | act::MASTER_CUE_A | act::MASTER_CUE_B => {
                    if pressed {
                        mix.step(match code {
                            act::HEADPHONES_A_SPLIT => keys::HP_A_SPLIT,
                            act::HEADPHONES_B_SPLIT => keys::HP_B_SPLIT,
                            act::MASTER_CUE_A => keys::MASTER_CUE_A,
                            _ => keys::MASTER_CUE_B,
                        }, 1);
                    }
                    continue;
                }
                act::BEAT_FX_ON => {
                    if pressed {
                        mix.step(keys::BEAT_ON, 1);
                    }
                    continue;
                }
                act::BEAT_FX_LEVEL => {
                    mix.set(keys::BEAT_LEVEL, unit);
                    continue;
                }
                act::BEAT_FX_BEAT_DOWN | act::BEAT_FX_BEAT_UP => {
                    if pressed {
                        mix.step(keys::BEAT_BEAT, if code == act::BEAT_FX_BEAT_UP { 1 } else { -1 });
                    }
                    continue;
                }
                act::BEAT_FX_SELECT_BACK => {
                    if pressed {
                        mix.step(keys::BEAT_TYPE, -1);
                    }
                    continue;
                }
                act::BEAT_FX_SELECT | act::BEAT_FX_CHANNEL => {
                    // the selector stepped (an effect picked anew: its BEAT at its default)
                    let key = if code == act::BEAT_FX_SELECT { keys::BEAT_TYPE } else { keys::BEAT_CHANNEL };
                    if steps != 0 {
                        mix.step(key, steps.signum() as i64);
                    } else if pressed {
                        mix.step(key, 1);
                    }
                    continue;
                }
                act::CROSSFADER => {
                    mix.set_crossfader(value as f64 / top);
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
                        mix.toggle_cue(k); // as a DJM's CUE buttons: a press turns it on or off
                    }
                    continue;
                }
                _ => {}
            }
            let slot = decks.get(k);
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
                gui_action = b_action(code, k);
                continue;
            }
            slot.deck.midi_value(&mut slot.engine, code, (value, top), steps, d2, at);
        }
        (flags, gui_action, gui_steps)
    }

    /// Once a tick: the lights and the level meters (where they changed).
    pub fn tick(&mut self, decks: &Decks, mix: &MixParams) {
        let now = clock::now();
        let dt = (now - self.last_tick).max(0.0);
        self.last_tick = now;
        if self.midi.is_none() {
            return;
        }
        for k in 0..MAX_DECKS {
            let lights = decks.slot(k).map(|s| s.deck.lights(&s.engine, now)).unwrap_or(0)
                | (mix.cue_on(k) as u64) << LAMP_CHANNEL_CUE
                | (mix.cue_on_bus(k, 1) as u64) << LAMP_CHANNEL_CUE_B
                | ((mix.get(keys::BEAT_ON) != 0.0) as u64) << LAMP_BEAT_FX
                | Self::mixer_lights(mix);
            if self.lamps_sent[k] != Some(lights) {
                let before = self.lamps_sent[k];
                self.send_lamps(k, lights, before);
                self.lamps_sent[k] = Some(lights);
            }
            // the level meter: the GUI meter's reading
            let peak = decks.slot(k).map(|s| s.engine.heard_peak_dbfs()).unwrap_or(METER_FLOOR_DBFS)
                + 20.0 * self.meter_gain.max(1e-9).log10();
            let m = &mut self.meters[k];
            m.db = peak.max(m.db - METER_FALL_DB_PER_SEC * dt).clamp(METER_FLOOR_DBFS, 0.0);
            if now - m.at >= METER_INTERVAL_SEC {
                let value = meter_value(m.db);
                if m.sent != Some(value) {
                    m.sent = Some(value);
                    m.at = now;
                    self.send_meter(k, value);
                }
            }
        }
    }

    /// The mixer's own lights (every deck's bindings of them).
    fn mixer_lights(mix: &MixParams) -> u64 {
        let on = |key: usize| (mix.get(key) != 0.0) as u64;
        let bands = mix.get(keys::BEAT_BANDS) as u64 & 7;
        let color = mix.get(keys::COLOR_TYPE) as u64;
        bands << LAMP_BEAT_BAND
            | on(keys::HP_A_SPLIT) << LAMP_HP_A_SPLIT
            | on(keys::HP_B_SPLIT) << LAMP_HP_B_SPLIT
            | on(keys::MASTER_CUE_A) << LAMP_MASTER_CUE_A
            | on(keys::MASTER_CUE_B) << LAMP_MASTER_CUE_B
            | on(keys::BEAT_BPM) << LAMP_BEAT_TAP
            | if color < ColorType::COUNT as u64 { 1u64 << (LAMP_COLOR as u64 + color) } else { 0 }
    }

    /// TAP pressed at ``at``: the BPM from the taps so far (more than two, none further apart than TAP_GAP_SEC), its
    /// BEAT then 1 (the manual's), the TIME knob's time let go.
    fn tap(&mut self, mix: &MixParams, at: f64) {
        if self.taps.last().is_some_and(|&t| at - t > TAP_GAP_SEC || at < t) {
            self.taps.clear();
        }
        if self.taps.len() == TAP_KEEP {
            self.taps.remove(0);
        }
        self.taps.push(at);
        if self.taps.len() >= 3 {
            let span = self.taps[self.taps.len() - 1] - self.taps[0];
            let bpm = 60.0 * (self.taps.len() - 1) as f64 / span.max(1e-3);
            mix.set(keys::BEAT_BPM, bpm);
            if BeatType::from_code(mix.get(keys::BEAT_TYPE) as u8) != BeatType::Reverb {
                mix.set(keys::BEAT_BEAT, 1.0);
            }
            mix.set(keys::BEAT_TIME, 0.0);
        }
    }

    /// The TIME knob: a fader's ``unit`` over the effect's time range (log; MOBIUS's falling below the middle,
    /// REVERB's its level 1 - 100 %), or a knob's ``steps`` moving the time it plays at by TIME_STEP each.
    fn beat_time(&self, mix: &MixParams, unit: f64, steps: i32) {
        let kind = BeatType::from_code(mix.get(keys::BEAT_TYPE) as u8);
        if kind == BeatType::Reverb {
            let v = if steps != 0 { kind.snap(mix.get(keys::BEAT_BEAT)) + steps as f64 } else { 1.0 + 99.0 * unit };
            mix.set(keys::BEAT_BEAT, v.clamp(1.0, 100.0));
            return;
        }
        let (lo, hi) = kind.time_range();
        let ms = if steps != 0 {
            let now = match mix.get(keys::BEAT_TIME) {
                0.0 => {
                    let bpm = match mix.get(keys::BEAT_BPM) {
                        0.0 => mix.beat_bpm(),
                        b => b,
                    };
                    60000.0 / if bpm > 20.0 { bpm } else { 120.0 } * kind.snap(mix.get(keys::BEAT_BEAT))
                }
                t => t,
            };
            let m = (now.abs() * TIME_STEP.powi(steps)).clamp(lo, hi);
            if now < 0.0 { -m } else { m }
        } else if kind == BeatType::Mobius {
            let u = (2.0 * unit - 1.0).abs();
            let m = lo * (hi / lo).powf(1.0 - u);
            if unit < 0.5 { -m } else { m }
        } else {
            lo * (hi / lo).powf(unit)
        };
        mix.set(keys::BEAT_TIME, ms);
    }

    /// Deck ``k``'s bindings' lights as ``lights`` (Deck::lights), those that changed since ``before`` (None: all).
    fn send_lamps(&mut self, k: usize, lights: u64, before: Option<u64>) {
        let Some(midi) = self.midi.as_mut() else { return };
        for b in self.bindings.iter().filter(|b| b.deck() == k && b.code() < act::GUI) {
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

    fn send_meter(&mut self, k: usize, value: u8) {
        let Some(midi) = self.midi.as_mut() else { return };
        for b in self.bindings.iter().filter(|b| b.code() == act::LEVEL_METER && b.deck() == k && b.status & 0xF0 == 0xB0) {
            midi.send(&[b.status, b.data1, value.min(127)]);
        }
    }
}

/// An action's code with its deck (from act::DECK_SHIFT up).
fn b_action(code: u32, deck: usize) -> u32 {
    code | (deck as u32) << act::DECK_SHIFT
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

/// A level (dBFS, METER_FLOOR_DBFS - 0) as the meter's CC value.
fn meter_value(db: f64) -> u8 {
    (127.0 * ((db - METER_FLOOR_DBFS) / -METER_FLOOR_DBFS).clamp(0.0, 1.0)).round() as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Ring;
    use std::sync::Arc;

    fn rig(bindings: &[(u8, u8, u8, u32)]) -> (Controller, Decks, Arc<MixParams>) {
        let port = Arc::new(crate::device::Port::default());
        let lanes = |_| Arc::new(Ring::new(48000, 4, 48000.0, port.clone()));
        let params = Arc::new(MixParams::new(2));
        let decks = Decks::new(48000, 2, (0..MAX_DECKS).map(lanes).collect(), params.clone());
        let mut c = Controller::new();
        let mut payload = (bindings.len() as u32).to_le_bytes().to_vec();
        for &(st, d1, flags, action) in bindings {
            let code = action & act::CODE_MASK;
            let lamp = if (act::JUMP_A..act::JUMP_A + 8).contains(&code) { (code - act::JUMP_A + 1) as u8 } else { 0 };
            payload.extend([st, d1, flags, lamp]);
            payload.extend(action.to_le_bytes());
        }
        c.set_bindings(&payload);
        (c, decks, params)
    }

    fn deck2(code: u32) -> u32 {
        code | 1 << act::DECK_SHIFT
    }

    #[test]
    fn tempo_fader_14_bits_over_the_range() {
        let (mut c, mut d, m) = rig(&[(0xB0, 0x00, bind::HIRES, act::TEMPO_FADER)]);
        d.first().deck.tempo_range = 10;
        c.message(&mut d, &m, 0.0, [0xB0, 0x00, 0x40]); // MSB 64, LSB 0: the middle
        c.message(&mut d, &m, 0.0, [0xB0, 0x20, 0x00]);
        assert!((d.first().engine.factor() - 1.0).abs() < 1e-12);
        c.message(&mut d, &m, 0.0, [0xB0, 0x00, 0x7F]);
        c.message(&mut d, &m, 0.0, [0xB0, 0x20, 0x7F]); // the top: +range
        assert!((d.first().engine.factor() - 1.1).abs() < 1e-3);
        c.message(&mut d, &m, 0.0, [0xB0, 0x00, 0x00]);
        c.message(&mut d, &m, 0.0, [0xB0, 0x20, 0x00]); // the bottom: -range
        assert!((d.first().engine.factor() - 0.9).abs() < 1e-12);
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
        assert_eq!(d.first().deck.volume(), 1.0);
        c.message(&mut d, &m, 0.0, [0xB0, 0x13, 0]);
        assert_eq!(d.first().deck.volume(), 0.0);
        for _ in 0..10 {
            c.message(&mut d, &m, 0.0, [0xB0, 0x21, 0x42]); // two steps on, ten times
        }
        let s = d.first();
        s.deck.tick(&mut s.engine);
        assert!(s.deck.wants_fast_ticks(), "bent");
    }

    #[test]
    fn jog_let_go_while_turning_is_held_until_it_stops() {
        let (mut c, mut d, m) = rig(&[(0x90, 0x36, 0, act::JOG_TOUCH), (0xB0, 0x22, 0, act::JOG_SCRATCH),
                                      (0xB0, 0x23, 0, act::JOG_BEND)]);
        c.message(&mut d, &m, clock::now(), [0x90, 0x36, 0x7F]);
        assert!(d.first().engine.scratching);
        c.message(&mut d, &m, clock::now(), [0xB0, 0x22, 0x3C]); // turned back (a backspin)
        c.message(&mut d, &m, clock::now(), [0x90, 0x36, 0x00]); // let go while turning
        assert!(d.first().engine.scratching, "let go while turning: still held");
        c.message(&mut d, &m, clock::now(), [0xB0, 0x23, 0x3E]); // its rotation, now the bend CC: still the record
        let s = d.first();
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
        assert!(!d.first().engine.scratching);
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
        let (mut c, mut d, m) = rig(&[(0x90, 0x0B, 0, act::PLAY_PAUSE)]);
        d.first().engine.command_at = 1.0; // a GUI command long ago
        c.message(&mut d, &m, 123.5, [0x90, 0x0B, 0x7F]);
        assert_eq!(d.first().engine.command_at, 123.5);
    }

    #[test]
    fn meter_binding_is_an_output() {
        let (mut c, mut d, m) = rig(&[(0xB0, 0x02, 0, act::LEVEL_METER)]);
        assert_eq!(c.message(&mut d, &m, 0.0, [0xB0, 0x02, 0x40]), (0, 0, 0));
        assert_eq!(d.first().engine.factor(), 1.0);
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
        d.first().deck.tempo_range = 10;
        c.message(&mut d, &m, 0.0, [0xE0, 0x7F, 0x7F]); // 16383: the top
        assert!((d.first().engine.factor() - 1.1).abs() < 1e-3);
        let (mut c, mut d, m) = rig(&[(0xB0, 0x00, bind::INVERT, act::TEMPO_FADER)]);
        d.first().deck.tempo_range = 10;
        c.message(&mut d, &m, 0.0, [0xB0, 0x00, 0x7F]); // inverted: the top is the bottom
        assert!((d.first().engine.factor() - 0.9).abs() < 1e-3);
    }

    #[test]
    fn jumpcue_pad_without_a_track_does_nothing() {
        let (mut c, mut d, m) = rig(&[(0x97, 0x00, 0, act::JUMP_A)]);
        assert_eq!(c.message(&mut d, &m, 0.0, [0x97, 0x00, 0x7F]), (0, 0, 0));
    }

    #[test]
    fn another_deck_made_and_driven() {
        let (mut c, mut d, m) = rig(&[(0xB1, 0x00, 0, deck2(act::TEMPO_FADER))]);
        assert!(d.slot(1).is_none());
        d.get(1).deck.tempo_range = 10;
        c.message(&mut d, &m, 0.0, [0xB1, 0x00, 0x7F]);
        assert!((d.slot(1).unwrap().engine.factor() - 1.0 - 0.1 * 63.0 / 64.0).abs() < 1e-9);
        assert_eq!(d.first().engine.factor(), 1.0, "deck 1 untouched");
    }

    #[test]
    fn effect_selector_and_jog_mode() {
        let (mut c, mut d, m) = rig(&[
            (0x94, 0x63, 0, act::BEAT_FX_SELECT),
            (0x94, 0x64, 0, act::BEAT_FX_SELECT_BACK),
            (0x90, 0x10, 0, act::JOG_MODE),
            (0x90, 0x36, 0, act::JOG_TOUCH),
        ]);
        let first = m.get(keys::BEAT_TYPE);
        c.message(&mut d, &m, 0.0, [0x94, 0x63, 0x7F]); // the next
        c.message(&mut d, &m, 0.0, [0x94, 0x63, 0x00]);
        assert_ne!(m.get(keys::BEAT_TYPE), first);
        c.message(&mut d, &m, 0.0, [0x94, 0x64, 0x7F]); // the previous: back
        c.message(&mut d, &m, 0.0, [0x94, 0x64, 0x00]); // let go: nothing
        assert_eq!(m.get(keys::BEAT_TYPE), first);
        c.message(&mut d, &m, 0.0, [0x94, 0x64, 0x7F]); // from the first, the last
        assert_ne!(m.get(keys::BEAT_TYPE), first);

        assert!(d.first().deck.jog_vinyl(), "VINYL at the start");
        c.message(&mut d, &m, 0.0, [0x90, 0x36, 0x7F]); // the top touched: the record held
        assert!(d.first().deck.jog_held());
        c.message(&mut d, &m, 0.0, [0x90, 0x10, 0x7F]); // JOG MODE: CDJ, the record let go
        c.message(&mut d, &m, 0.0, [0x90, 0x10, 0x00]);
        assert!(!d.first().deck.jog_vinyl() && !d.first().deck.jog_held());
        c.message(&mut d, &m, 0.0, [0x90, 0x36, 0x00]);
        c.message(&mut d, &m, 0.0, [0x90, 0x36, 0x7F]); // CDJ: the top's touch ignored
        assert!(!d.first().deck.jog_held());
        c.message(&mut d, &m, 0.0, [0x90, 0x36, 0x00]);
        c.message(&mut d, &m, 0.0, [0x90, 0x10, 0x7F]); // VINYL again, a deck made later too
        assert!(d.first().deck.jog_vinyl());
        assert!(d.get(1).deck.jog_vinyl());
    }

    #[test]
    fn mixer_actions() {
        let (mut c, mut d, m) = rig(&[
            (0xB0, 0x0F, 0, act::EQ_LOW),
            (0xB6, 0x1F, 0, act::CROSSFADER),
            (0x91, 0x54, 0, deck2(act::CHANNEL_CUE)),
            (0xB6, 0x0C, 0, act::HEADPHONES_MIX),
        ]);
        c.message(&mut d, &m, 0.0, [0xB0, 0x0F, 0]); // EQ LOW to the bottom: a kill
        let x: Vec<f32> = (0..9600).map(|i| (2.0 * std::f64::consts::PI * 50.0 * (i / 2) as f64 / 48000.0).sin() as f32).collect();
        // through deck 1's mixer channel (its strip as the callback runs it)
        let lane: Vec<f32> = x.chunks_exact(2).flat_map(|f| [f[0], f[1], 0.0, 0.0]).collect();
        let (mut pfl, mut post) = (vec![0f32; x.len()], vec![0f32; x.len()]);
        crate::mixer::Strip::new(48000.0, 1).process(&lane, 4800, 2, &d.first().engine.strip, &m.read(), &mut pfl,
                                                     &mut post, None);
        let rms = (post[4800..].iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / 4800.0).sqrt();
        assert!(rms < 0.05, "50 Hz with the low killed: {rms}");
        c.message(&mut d, &m, 0.0, [0xB6, 0x1F, 0]);
        c.message(&mut d, &m, 0.0, [0x91, 0x54, 0x7F]);
        c.message(&mut d, &m, 0.0, [0x91, 0x54, 0x00]); // let go: still on
        assert!(m.cue_on(1));
        c.message(&mut d, &m, 0.0, [0x91, 0x54, 0x7F]); // pressed again: off
        assert!(!m.cue_on(1));
        c.message(&mut d, &m, 0.0, [0xB6, 0x0C, 127]);
        let mix = m.read();
        assert_eq!(mix.hp[0].0, 1.0);
        assert_eq!(mix.assign[0], crate::mixer::Assign::Thru, "deck 1 THRU: the crossfader passes it");
    }
}
