//! The control block shared with the GUI process (audioengine/native/client.py holds the same layout).
//!
//! [0, STATE_OFFSET): header: magic, protocol version.
//! [STATE_OFFSET, RING_OFFSET): the engine's state, written by the engine under a sequence lock (odd while
//!   it is being written), read by the GUI at its own pace.
//! [RING_OFFSET, end): the commands, a byte ring the GUI writes (one writer) and the engine reads: the
//!   ring's write position (bytes ever written), its read position (bytes ever read, written by the engine),
//!   then the records. A record is [size u32 (header included, 8-aligned), op u32, seq u64, checksum u32, payload
//!   length u32, sent at f64 (the GUI's perf_counter: the response measurement)] + payload (+ padding); a record never wraps (the GUI pads to the ring's end with a size-only record of op 0). The
//!   checksum (CRC-32 over op, seq and payload: Python's zlib.crc32) is checked before a record is taken: a weakly ordered CPU may
//!   show the write position before the bytes it covers, which are then read again on the next tick.
//! [MIDI_PORTS_OFFSET, MIDI_EVENTS_OFFSET): the MIDI input ports, written when asked (op::MIDI_LIST), as the devices.
//! [MIDI_EVENTS_OFFSET, DEVICES_OFFSET): the MIDI messages received, a ring the engine writes (deck.rs: the GUI's
//!   own actions bound to them, and every message for the settings' MIDI learn).
//! [DEVICES_OFFSET, OUTPUTS_OFFSET): the output devices, written when asked (op::LIST_DEVICES).
//! [OUTPUTS_OFFSET, MIX_OFFSET): the state of the buses on devices of their own (the headphones, the booth;
//!   out::*), written with the state under its sequence lock.
//! [MIX_OFFSET, SPECTRUM_OFFSET): the mixer's settings as it has them (mix::*: f64 each by key) and its meters
//!   (mix::METERS), written with the state under its sequence lock.
//! [SPECTRUM_OFFSET, RING_OFFSET): the master's spectrum (spectrum.rs), a ring of frames the engine writes while
//!   asked (op::SET_SPECTRUM).

use std::sync::atomic::{AtomicU64, Ordering, fence};

pub const MAGIC: u64 = u64::from_le_bytes(*b"MXAUDIC1");
pub const VERSION: u32 = 9;
pub const STATE_OFFSET: usize = 64;
pub const RING_OFFSET: usize = 172032;
pub const RING_HEADER: usize = 64;
pub const HIST_MAX: usize = 4096;
/// The lowest rate anything plays at, every output's and the offline renders' (controls.MIN_RATE; the time
/// stretch's band split needs it): a device at less is not opened.
pub const MIN_RATE: u32 = 44_100;
/// The output devices, written when asked (op::LIST_DEVICES): [generation u64, length u32, pad u32] + UTF-8 text:
/// the main output's device in use on the first line, the system's default as "name<TAB>channels" on the second
/// (empty: none), then one listed device a line, "name<TAB>channels". The generation is written last.
pub const DEVICES_OFFSET: usize = 40960;
pub const DEVICES_MAX: usize = OUTPUTS_OFFSET - DEVICES_OFFSET - 16;
/// The buses that may play on a device of their own (device.rs BRIDGED_BUSES: 0 the headphones, 1 the booth), each
/// a block of out::SIZE from here.
pub const OUTPUTS_OFFSET: usize = 65536;
/// The mixer's settings and meters (mix::*), after the buses' blocks.
pub const MIX_OFFSET: usize = 165888;
/// The master's spectrum (spectrum.rs): [written u64 (frames ever), pad u64] + SPECTRUM_FRAMES slots of
/// SPECTRUM_SLOT bytes, frame k at k % SPECTRUM_FRAMES: [seq u32 (k + 1, written last), pad 12 bytes, SPECTRUM_BINS
/// u8: the power spectral density of SPECTRUM_BINS log-spaced bins from SPECTRUM_F_LO Hz, SPECTRUM_BINS_PER_OCTAVE
/// an octave, each SPECTRUM_DB_FLOOR + v * SPECTRUM_DB_STEP dB (re full scale, per Hz)].
pub const SPECTRUM_OFFSET: usize = 166912;
pub const SPECTRUM_FRAMES: usize = 16;
pub const SPECTRUM_SLOT: usize = 256;
pub const SPECTRUM_BINS: usize = 240;
pub const SPECTRUM_F_LO: f64 = 20.0;
pub const SPECTRUM_BINS_PER_OCTAVE: usize = 24;
pub const SPECTRUM_DB_FLOOR: f64 = -160.0;
pub const SPECTRUM_DB_STEP: f64 = 0.625;
const _: () = assert!(SPECTRUM_OFFSET >= MIX_OFFSET + 8 * (mix::METERS + mix::METER_COUNT));
const _: () = assert!(SPECTRUM_OFFSET + 16 + SPECTRUM_FRAMES * SPECTRUM_SLOT <= RING_OFFSET);
const _: () = assert!(16 + SPECTRUM_BINS <= SPECTRUM_SLOT);
/// The MIDI input ports (op::MIDI_LIST): [generation u64, length u32, pad u32] + UTF-8 text, one port a line.
pub const MIDI_PORTS_OFFSET: usize = 34816;
pub const MIDI_PORTS_MAX: usize = MIDI_EVENTS_OFFSET - MIDI_PORTS_OFFSET - 16;
/// The MIDI messages received: [written u64 (messages ever), pad u64] + MIDI_EVENTS slots of 16 bytes, message k
/// at k % MIDI_EVENTS: [seq u32 (k + 1, written last), status u8, data1 u8, data2 u8, flags u8 (MIDI_EVENT_*),
/// action u32 (the binding's, 0: none), steps i32 (a CC read as a knob's turn: + clockwise)].
pub const MIDI_EVENTS_OFFSET: usize = 36864;
pub const MIDI_EVENTS: usize = 240;
/// The message is bound to one of the GUI's actions (its action code: 0x1000 and up, the GUI's numbering).
pub const MIDI_EVENT_GUI: u8 = 1;
/// ... through a binding of one turning direction: the action done once per step.
pub const MIDI_EVENT_RELATIVE: u8 = 2;
/// A hot cue pad (act::JUMP_A + k) pressed on an empty slot: the GUI sets the JumpCUE (an edit).
pub const MIDI_EVENT_HOTCUE_SET: u8 = 4;

/// A binding's flags (op::MIDI_MAP).
pub mod bind {
    pub const HIRES: u8 = 1; // a 14-bit CC: its LSB at data1 + 32
    pub const CLOCKWISE: u8 = 2; // a knob's turns clockwise only, a step an action
    pub const COUNTERCLOCKWISE: u8 = 4;
    pub const INVERT: u8 = 8; // a fader read the other way up, a knob's turns the other way round
    pub const ENCODING: u8 = 0x30; // a relative CC's: ENC_* << ENCODING_SHIFT
    pub const ENCODING_SHIFT: u8 = 4;
    #[allow(dead_code)] // the default: what the other values are not
    pub const ENC_AUTO: u8 = 0; // offset or two's complement, by the value's range (steps under 32)
    pub const ENC_OFFSET: u8 = 1; // 64 + steps (Pioneer's jog wheels)
    pub const ENC_TWOS: u8 = 2; // two's complement (Pioneer's BROWSE: 1 .. 30 / 127 .. 98)
    pub const ENC_SIGN: u8 = 3; // sign and magnitude: bit 6 the sign (Native Instruments, Behringer)
}

// State fields (byte offsets inside the state area)
pub mod st {
    pub const SEQ: usize = 0; // u64
    pub const READY: usize = 8; // u32: 0 starting, 1 running, 2 failed
    pub const RATE: usize = 12; // u32
    pub const CHANNELS: usize = 16; // u32
    pub const PLAYING: usize = 20; // u32
    pub const TRACK_GEN: usize = 24; // u32: the load the engine holds
    pub const AT_END: usize = 28; // u32: the render head is at the track's end
    pub const CMD_DONE: usize = 32; // u64: seq of the last command taken
    pub const PLAYHEAD: usize = 40; // f64 input frame being heard
    pub const DISPLAY: usize = 48; // f64 input frame to draw, at DISPLAY_AT
    pub const DISPLAY_AT: usize = 56; // f64 perf_counter s
    pub const DISPLAY_RATE: usize = 64; // f64 input frames per second it moves at (0: held)
    pub const PEAK: usize = 72; // f64 dBFS of the block being heard (before the volume)
    pub const QUEUED: usize = 80; // f64 frames in this engine's buffer (ahead of the device's latency)
    pub const CAPACITY: usize = 88; // f64 frames: the buffer setting
    pub const JUMPED: usize = 96; // u64 count of jumps played
    pub const FINISHED: usize = 104; // u64 count of track ends heard
    pub const BUFFER_GEN: usize = 112; // u64: the histogram restarted (its size changed)
    pub const OBS_SECONDS: usize = 120; // f64 cumulative
    pub const OBS_LEVEL: usize = 128; // f64 cumulative integral of level (ms) over time (s)
    pub const UNDERRUNS: usize = 136; // u64 frames played as silence while sound was due
    pub const HIST_LEN: usize = 144; // u32
    pub const DEVICE_ERRORS: usize = 148; // u32
    pub const STRETCH_MS: usize = 152; // f64 cumulative ms spent rendering (diagnostics)
    pub const LATENCY_MS: usize = 160; // f64 the device's own latency (fixed by its driver / mode)
    pub const OUTPUT_MODE: usize = 168; // u32 the output in use: API (0 WASAPI, 1 cpal, 2 JACK) | 0x100 exclusive
    pub const PERIOD_MS: usize = 176; // f64 the device's period (0: not known)
    // the last measured response (ms) from the GUI's command to its sound being heard (0: none yet)
    pub const RESPONSE_PLAY: usize = 184; // f64 play: its first frame heard
    pub const RESPONSE_PAUSE: usize = 192; // f64 pause: its fade-out over
    pub const RESPONSE_SEEK: usize = 200; // f64 seek: the new position's first frame heard
    pub const HIST_BUCKET_MS: usize = 208; // f64 the histogram's bucket width (ms)
    pub const LOWEST_MS: usize = 216; // f64 the buffer's lowest level over the last 10 s of playing (-1: none)
    pub const OUTPUT_OPEN: usize = 224; // u32 1: the device is open; 0: it could not be (held: tried each second)
    pub const DEVICE_LATE: usize = 232; // u64 the output thread's late wake-ups (the engine's own WASAPI output)
    pub const MAX_GAP_MS: usize = 240; // f64 its longest gap between wake-ups since it opened
    // the threads' scheduling (sched.rs): u32 the engine thread's state | the output thread's << 8; u32 power
    pub const THREADS: usize = 248;
    pub const POWER: usize = 252;
    pub const HIST: usize = 256; // f64[HIST_LEN] cumulative seconds per 1 ms level
    // the deck (deck.rs), after the histogram: what the GUI shows, as the engine has it
    pub const DECK: usize = 256 + 8 * super::HIST_MAX;
    pub const FACTOR: usize = DECK; // f64 the tempo (1: 100 %)
    pub const VOLUME: usize = DECK + 8; // f64 the volume fader (0 .. 1)
    pub const CUE: usize = DECK + 16; // f64 the cue point (input frame; NaN: no track)
    pub const JUMP_DEST: usize = DECK + 24; // f64 the JumpCUE armed: its destination (input frame; NaN: none)
    pub const LEDS: usize = DECK + 32; // u32 bit 0 PLAY lit, bit 1 CUE lit (blinking: as they are now)
    pub const TEMPO_RANGE: usize = DECK + 36; // u32 the tempo fader's range (+-%; 100: WIDE, 0 - 200 %)
    pub const MIDI_STATE: usize = DECK + 40; // u32 0 closed, 1 open, 2 the port did not open
    pub const OUT_CHANNELS: usize = DECK + 44; // u32 the output device's channels (the deck plays stereo)
    // its mixer channel's (the meter is PEAK above)
    pub const CHANNEL_CUE: usize = DECK + 48; // u32 its headphones CUE on
    pub const KNOBS: usize = DECK + 56; // f64 x 4: its TRIM, EQ LOW, MID, HIGH knobs (0 .. 1, the middle 0 dB)
    pub const COLOR: usize = DECK + 88; // f64 its COLOR knob (0 .. 1, 0.5 off)
}

/// A bus's output on a device of its own (OUTPUTS_OFFSET + k * SIZE), its fields as the main output's above.
pub mod out {
    pub const STATE: usize = 0; // u32 STATE_*
    pub const STATE_OFF: u32 = 0;
    pub const STATE_OPEN: u32 = 1;
    pub const STATE_FAILED: u32 = 2; // not opened: tried again each second
    pub const STATE_OPENING: u32 = 3;
    pub const MODE: usize = 4; // u32 as st::OUTPUT_MODE
    pub const PERIOD_MS: usize = 8; // f64 its device's period (0: not known)
    pub const LATENCY_MS: usize = 16; // f64 its device's latency (with the interpolator's)
    pub const QUEUED: usize = 24; // f64 frames in its bridge, ahead of its device's latency
    pub const CAPACITY: usize = 32; // f64 frames: the buffer it keeps
    pub const UNDERRUNS: usize = 40; // u64 frames played as silence while sound was due
    pub const LATE: usize = 48; // u64 its output thread's late wake-ups
    pub const MAX_GAP_MS: usize = 56; // f64
    pub const THREAD: usize = 64; // u32 its output thread's state (sched.rs)
    pub const CHANNELS: usize = 68; // u32 its device's channels
    pub const DRIFT_PPM: usize = 72; // f64 its clock against the main device's (the ratio it reads at - 1)
    pub const BUFFER_GEN: usize = 80; // u64 as st::BUFFER_GEN
    pub const OBS_SECONDS: usize = 88; // f64
    pub const OBS_LEVEL: usize = 96; // f64
    pub const HIST_BUCKET_MS: usize = 104; // f64
    pub const LOWEST_MS: usize = 112; // f64
    pub const HIST_LEN: usize = 120; // u32
    pub const ERRORS: usize = 124; // u32 its device's open failures
    pub const NAME: usize = 128; // u32 length + UTF-8: the device opened
    pub const NAME_MAX: usize = 240;
    pub const RATE: usize = 376; // u32 its device's rate (the bus converted to it)
    pub const HIST: usize = 384; // f64[HIST_LEN] cumulative seconds per bucket
    pub const SIZE: usize = HIST + 8 * super::HIST_MAX;
    /// Field ``f`` of bus ``k``'s block, as a state field (Control::put_*).
    pub const fn at(k: usize, f: usize) -> usize {
        super::OUTPUTS_OFFSET - super::STATE_OFFSET + k * SIZE + f
    }
}

/// The mixer's settings (op::SET_MIXER: u32 key, f64 value; each published at MIX_OFFSET + 8 * key) and meters.
/// Knobs are positions 0 .. 1 (EQ and the like: the middle 0 dB; levels: LEVEL_UNITY 0 dB), switches 0 / 1, choices
/// their code.
pub mod mix {
    pub const EQ_MODE: usize = 0; // 0 isolator (a band down to a kill), 1 EQ (-26 dB at the bottom)
    pub const FADER_CURVE: usize = 1; // the channel fader's: 0 smooth (the GUI's), 1 steep (up fast), 2 slow
    pub const COLOR_TYPE: usize = 2; // Sound Color FX: colorfx::ColorType's code (ColorType::COUNT: none)
    pub const COLOR_PARAM: usize = 3; // its PARAMETER knob
    pub const HP_MIX: usize = 4; // the headphones: CUE / MASTER mix (0 CUE .. 1 MASTER)
    pub const HP_LEVEL: usize = 5; // their level
    pub const HP_MASTER: usize = 6; // the master mixed in (0: a device mixing it in itself)
    pub const HP_SPLIT: usize = 7; // MONO SPLIT: the CUE left, the master right
    pub const MASTER_LEVEL: usize = 8;
    pub const BOOTH_LEVEL: usize = 9;
    pub const BOOTH_EQ_HI: usize = 10;
    pub const BOOTH_EQ_LOW: usize = 11;
    pub const KEYS: usize = 12;
    /// The keys' names (client.py MIX_KEYS; mixlyzer-audio --mix-file's --set name=value).
    pub const NAMES: [&str; KEYS] = [
        "eq_mode", "fader_curve", "color_type", "color_param", "hp_mix", "hp_level", "hp_master", "hp_split",
        "master_level", "booth_level", "booth_eq_hi", "booth_eq_low",
    ];
    /// A level knob's position at 0 dB (mixer::level_gain).
    pub const LEVEL_UNITY: f64 = 0.8;
    /// Each key's value at the start.
    pub const DEFAULTS: [f64; KEYS] = [
        0.0, 0.0, 5.0, 0.5, //
        0.5, LEVEL_UNITY, 1.0, 0.0, //
        LEVEL_UNITY, LEVEL_UNITY, 0.5, 0.5,
    ];
    /// The meters, after the keys' values (f64 linear peaks since the last publish): MIX_OFFSET + 8 * (METERS + k).
    pub const METERS: usize = 64;
    pub const METER_MASTER: usize = 0;
    pub const METER_BOOTH: usize = 1;
    pub const METER_COUNT: usize = 2;
    /// Value / meter ``k``'s place, as a state field (Control::put_*).
    pub const fn at(k: usize) -> usize {
        super::MIX_OFFSET - super::STATE_OFFSET + 8 * k
    }
}

pub mod op {
    pub const PAD: u32 = 0;
    pub const LOAD: u32 = 1; // gen u32, channels u32, frames u64, name (u32 len + utf8)
    pub const UNLOAD: u32 = 2; // gen u32
    pub const PLAY: u32 = 3;
    pub const PAUSE: u32 = 4;
    pub const STOP: u32 = 5;
    pub const SEEK: u32 = 6; // f64 frame
    pub const SCRUB: u32 = 7; // u32 on
    pub const SCRATCH: u32 = 8; // u32 on, f64 frames per px
    pub const SCRATCH_TO: u32 = 9; // f64 frame, f64 at
    pub const ARM_JUMP: u32 = 10; // f64 start (NaN: none), f64 dest, u32 once, u32 n, f64[n] starts
    pub const DISARM_JUMP: u32 = 11;
    pub const SET_MODE: u32 = 12; // u32 1 speed / 0 none
    pub const SET_FACTOR: u32 = 13; // f64
    pub const SET_TIMESTRETCH: u32 = 14; // u32
    pub const SET_CLIP: u32 = 15; // u32 x 3: each output's last stage (mixer::Clip), the master's, the headphones', the booth's
    pub const SET_BUFFER_MS: u32 = 16; // f64
    pub const TONE_HOLD: u32 = 17; // u32 n, i32[n]
    // 18: none (the deck's volume is its mixer channel's fader: op::SET_VOLUME)
    pub const CLICK_SAMPLE: u32 = 19; // u32 frames, u32 channels, f32[frames * channels]
    pub const CLICK_PITCHES: u32 = 20; // f64 beat, f64 accent (semitones)
    pub const CLICK_BEATS: u32 = 21; // u32 n, f64[n] seconds, f32[n] gains, u8[n] accents
    pub const CLICK_ENABLED: u32 = 22; // u32
    pub const CLICK_OFFSET: u32 = 23; // f64 s
    pub const CLICK_DUCKING: u32 = 24; // u32
    pub const CLICK_CLEAR: u32 = 25;
    pub const SHUTDOWN: u32 = 26;
    pub const WAKE: u32 = 27;
    pub const SET_DEVICE: u32 = 28; // name (u32 len + utf8; empty: the system's default)
    pub const LIST_DEVICES: u32 = 29; // [u32 API: the devices of that one; none: the main output's]
    pub const SET_OUTPUT_MODE: u32 = 30; // u32 API (0 WASAPI, 1 cpal, 2 JACK), u32 exclusive (WASAPI)
    // u32 priority (0 normal, 1 above normal, 2 high), u32 power throttling allowed (sched.rs: per system)
    pub const SET_PROCESS: u32 = 31;
    // the deck (deck.rs): the GUI's requests, combined there with the MIDI controller's
    pub const DECK_ACTION: u32 = 32; // u32 action (act), u32 pressed (1) / released (0)
    pub const SET_TEMPO: u32 = 33; // f64 factor (the tempo fader's position)
    pub const SET_VOLUME: u32 = 34; // f64 the volume fader (0 .. 1)
    // u32 n, f64[n] beats (input frames), u8[n] downbeat, u32 m, f64[m] CUE points, f64[26] JumpCUEs A-Z (NaN: none)
    pub const DECK_MARKS: u32 = 36;
    pub const JUMP_TO: u32 = 37; // f64 frame: a JumpCUE's JUMP
    pub const CALL_AT: u32 = 38; // f64 frame: a CUE point's CALL
    pub const MIDI_OPEN: u32 = 39; // name (u32 len + utf8; empty: closed)
    // u32 n, n x [status u8, data1 u8 (a pitch bend's: not read), flags u8 (bind::*), lamp u8, action u32]; lamp:
    // the controller's light at that message (deck.rs LAMP_*: 1 - 26 the JumpCUE A - Z set, PLAY, CUE; 0 none)
    pub const MIDI_MAP: u32 = 40;
    pub const MIDI_LIST: u32 = 41;
    pub const MIDI_OPTIONS: u32 = 42; // u32 a jog wheel's messages (ticks) per turn; u32 its mode (1 VINYL, 0 CDJ)
    pub const SET_TEMPO_RANGE: u32 = 43; // u32 the tempo fader's range (+-%: 6, 10, 16, 100 WIDE)
    pub const SET_METER_GAIN: u32 = 44; // f64 linear: the level meter's input, as the GUI's peak meter
    // u8 x 6 the device's channels: main L / R, headphones L / R, booth L / R (255: not routed); the headphones'
    // CUE / MASTER mix and level are mix::HP_*
    pub const SET_ROUTING: u32 = 47;
    pub const SET_CHANNEL_CUE: u32 = 49; // u32 the mixer channel's headphones CUE on
    // a bus on a device of its own: u32 bus (0 the headphones, 1 the booth), u32 on (0: not: SET_ROUTING's channels
    // of the main device, if any), u32 API, u32 exclusive, f64 buffer ms, u8 left, u8 right (its device's channels),
    // u16 pad, u32 its rate (0: the device's mix format's), the device's name (u32 len + utf8; empty: the system's
    // default)
    pub const SET_AUX_OUTPUT: u32 = 50;
    // the headphones: u32 the master mixed in (0: a device mixing it in itself), f64 the CUE / MASTER mix (0 CUE ..
    // 1 MASTER), f64 their level (linear); a controller's knobs (act::HEADPHONES_MIX / LEVEL) move them later
    pub const SET_HEADPHONES: u32 = 51;
    // the deck's mixer channel knob: u32 which (0 TRIM, 1 EQ LOW, 2 MID, 3 HIGH, 4 COLOR), f64 its position (0 .. 1,
    // the middle 0 dB / off)
    pub const SET_MIXER_KNOB: u32 = 52;
    pub const SET_MIXER: u32 = 53; // u32 key (mix::*), f64 value
    pub const SET_SPECTRUM: u32 = 54; // u32 the master's spectrum made (1) or not (SPECTRUM_OFFSET)
}

/// The deck's actions (deck.rs; client.py DECK_ACTIONS): the GUI's buttons and keys and the MIDI controller's.
pub mod act {
    pub const PLAY_PAUSE: u32 = 1;
    pub const CUE: u32 = 2;
    pub const STOP: u32 = 3;
    pub const BEAT_BACK: u32 = 4;
    pub const BEAT_FORWARD: u32 = 5;
    pub const BAR_BACK: u32 = 6;
    pub const BAR_FORWARD: u32 = 7;
    pub const TEMPO_DOWN: u32 = 8;
    pub const TEMPO_UP: u32 = 9;
    pub const TEMPO_RESET: u32 = 10;
    pub const CALL_PREVIOUS: u32 = 11;
    pub const CALL_NEXT: u32 = 12;
    pub const TEMPO_RANGE: u32 = 13;
    pub const TRACK_START: u32 = 14; // to the track's start, playing on (rekordbox's SHIFT + CUE)
    pub const JUMP_A: u32 = 16; // .. JUMP_A + 7: H
    pub const BEND_DOWN: u32 = 24; // held: the tempo bent down (a pitch bend button)
    pub const BEND_UP: u32 = 25;
    pub const JOG_MODE: u32 = 26; // a press: the jog wheel's VINYL / CDJ mode switched (DeckRow lamps bit 2: VINYL)
    /// Beat jumps: BEAT_JUMP + 2 k back, + 2 k + 1 forward, by BEAT_JUMP_SIZES[k] beats.
    pub const BEAT_JUMP: u32 = 64;
    pub const BEAT_JUMP_SIZES: [i64; 6] = [1, 2, 4, 8, 16, 32];
    // a MIDI controller's alone (values, not presses)
    pub const TEMPO_FADER: u32 = 32;
    pub const VOLUME_FADER: u32 = 33;
    pub const JOG_TOUCH: u32 = 34;
    pub const JOG_SCRATCH: u32 = 35;
    pub const JOG_BEND: u32 = 36;
    pub const JOG_SEARCH: u32 = 37;
    /// An output: the controller's level meter (a CC, 0 - 127 over the GUI meter's -24 - 0 dBFS).
    pub const LEVEL_METER: u32 = 38;
    // the mixer (mixer.rs): the channel's trim and EQ bands (knobs: the middle 0 dB), its headphones CUE (a press
    // turns it on or off), the headphones' CUE / MASTER mix and level
    pub const TRIM: u32 = 39;
    pub const EQ_LOW: u32 = 40;
    pub const EQ_MID: u32 = 41;
    pub const EQ_HIGH: u32 = 42;
    pub const CHANNEL_CUE: u32 = 44;
    pub const HEADPHONES_MIX: u32 = 45;
    pub const HEADPHONES_LEVEL: u32 = 46;
    // DJM's: the channel's COLOR knob, the others the mixer's
    pub const COLOR_KNOB: u32 = 47;
    pub const MASTER_LEVEL: u32 = 51;
    pub const BOOTH_LEVEL: u32 = 52;
    pub const BOOTH_EQ_HI: u32 = 53;
    pub const BOOTH_EQ_LOW: u32 = 54;
    pub const COLOR_PARAM: u32 = 55;
    /// COLOR_SELECT + k: Sound Color FX k picked (colorfx::ColorType's code: 0 - 6, up to 62)
    pub const COLOR_SELECT: u32 = 56;
    pub const HEADPHONES_SPLIT: u32 = 94; // MONO SPLIT on or off
    /// The GUI's own actions (library, editor ...) from here: passed on to it (MIDI_EVENT_GUI).
    pub const GUI: u32 = 0x1000;
}

pub struct Control {
    base: *mut u8,
    len: usize,
}

unsafe impl Send for Control {}

pub struct Command {
    pub op: u32,
    pub seq: u64,
    pub sent_at: f64,
    pub payload: Vec<u8>,
}

const RECORD_HEADER: usize = 32;

fn checksum(op: u32, seq: u64, sent_at: f64, payload: &[u8]) -> u64 {
    let mut h = crc32fast::Hasher::new();
    h.update(&op.to_le_bytes());
    h.update(&seq.to_le_bytes());
    h.update(&sent_at.to_le_bytes());
    h.update(payload);
    h.finalize() as u64
}

impl Control {
    pub fn new(base: *mut u8, len: usize) -> Result<Control, String> {
        if len <= RING_OFFSET + RING_HEADER + 4096 {
            return Err("control block too small".into());
        }
        let c = Control { base, len };
        if c.read_u64(0) != MAGIC || c.read_u32(8) != VERSION {
            return Err("control block: wrong magic or protocol version".into());
        }
        Ok(c)
    }

    fn atomic(&self, offset: usize) -> &AtomicU64 {
        debug_assert!(offset % 8 == 0 && offset + 8 <= self.len);
        unsafe { &*(self.base.add(offset) as *const AtomicU64) }
    }

    fn read_u64(&self, offset: usize) -> u64 {
        unsafe { std::ptr::read_volatile(self.base.add(offset) as *const u64) }
    }

    fn read_u32(&self, offset: usize) -> u32 {
        unsafe { std::ptr::read_volatile(self.base.add(offset) as *const u32) }
    }

    // Commands
    fn ring_capacity(&self) -> u64 {
        (self.len - RING_OFFSET - RING_HEADER) as u64
    }

    /// The next whole command, if one is there (and its bytes check out).
    pub fn next_command(&self) -> Option<Command> {
        let cap = self.ring_capacity();
        loop {
            let write = self.atomic(RING_OFFSET).load(Ordering::Acquire);
            let read = self.atomic(RING_OFFSET + 8).load(Ordering::Relaxed);
            if write <= read {
                return None;
            }
            let at = (read % cap) as usize;
            let rec = unsafe { self.base.add(RING_OFFSET + RING_HEADER + at) };
            fence(Ordering::Acquire);
            let size = unsafe { std::ptr::read_volatile(rec as *const u32) } as u64;
            if size < 8 || size % 8 != 0 || at as u64 + size > cap || read + size > write {
                return None; // not all there yet (or torn): next tick
            }
            let op = unsafe { std::ptr::read_volatile(rec.add(4) as *const u32) };
            if op == op::PAD {
                self.atomic(RING_OFFSET + 8).store(read + size, Ordering::Release);
                continue;
            }
            if (size as usize) < RECORD_HEADER {
                return None;
            }
            let seq = unsafe { std::ptr::read_volatile(rec.add(8) as *const u64) };
            let sum = unsafe { std::ptr::read_volatile(rec.add(16) as *const u32) } as u64;
            let length = unsafe { std::ptr::read_volatile(rec.add(20) as *const u32) } as usize;
            let sent_at = unsafe { std::ptr::read_volatile(rec.add(24) as *const f64) };
            if length > size as usize - RECORD_HEADER {
                return None;
            }
            let payload = unsafe { std::slice::from_raw_parts(rec.add(RECORD_HEADER), length) }.to_vec();
            if checksum(op, seq, sent_at, &payload) != sum {
                return None;
            }
            self.atomic(RING_OFFSET + 8).store(read + size, Ordering::Release);
            return Some(Command { op, seq, sent_at, payload });
        }
    }

    // State
    pub fn state_begin(&self) {
        let seq = self.atomic(STATE_OFFSET + st::SEQ);
        seq.store(seq.load(Ordering::Relaxed) | 1, Ordering::Relaxed);
        fence(Ordering::Release);
    }

    pub fn state_end(&self) {
        let seq = self.atomic(STATE_OFFSET + st::SEQ);
        let v = seq.load(Ordering::Relaxed);
        seq.store((v | 1) + 1, Ordering::Release);
    }

    pub fn put_u32(&self, field: usize, v: u32) {
        unsafe { std::ptr::write_volatile(self.base.add(STATE_OFFSET + field) as *mut u32, v) }
    }

    pub fn put_u64(&self, field: usize, v: u64) {
        unsafe { std::ptr::write_volatile(self.base.add(STATE_OFFSET + field) as *mut u64, v) }
    }

    pub fn put_f64(&self, field: usize, v: f64) {
        unsafe { std::ptr::write_volatile(self.base.add(STATE_OFFSET + field) as *mut f64, v) }
    }

    /// The main output's device in use, the system's default and the devices listed, with their channels (see
    /// DEVICES_OFFSET).
    pub fn put_devices(&self, current: &str, default: Option<&(String, usize)>, listed: &[(String, usize)]) {
        let mut text = String::from(current);
        text.push('\n');
        if let Some((name, ch)) = default {
            text.push_str(&format!("{name}\t{ch}"));
        }
        for (name, ch) in listed {
            text.push_str(&format!("\n{name}\t{ch}"));
        }
        let mut bytes = text.into_bytes();
        bytes.truncate(DEVICES_MAX);
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.base.add(DEVICES_OFFSET + 16), bytes.len());
            std::ptr::write_volatile(self.base.add(DEVICES_OFFSET + 8) as *mut u32, bytes.len() as u32);
        }
        let generation = self.atomic(DEVICES_OFFSET);
        generation.store(generation.load(Ordering::Relaxed) + 1, Ordering::Release);
    }

    /// The MIDI input ports (see MIDI_PORTS_OFFSET).
    pub fn put_midi_ports(&self, ports: &[String]) {
        let mut bytes = ports.join("\n").into_bytes();
        bytes.truncate(MIDI_PORTS_MAX);
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.base.add(MIDI_PORTS_OFFSET + 16), bytes.len());
            std::ptr::write_volatile(self.base.add(MIDI_PORTS_OFFSET + 8) as *mut u32, bytes.len() as u32);
        }
        let generation = self.atomic(MIDI_PORTS_OFFSET);
        generation.store(generation.load(Ordering::Relaxed) + 1, Ordering::Release);
    }

    /// A MIDI message received, into the ring (see MIDI_EVENTS_OFFSET).
    pub fn put_midi_event(&self, msg: [u8; 3], flags: u8, action: u32, steps: i32) {
        let written = self.atomic(MIDI_EVENTS_OFFSET);
        let k = written.load(Ordering::Relaxed);
        let slot = MIDI_EVENTS_OFFSET + 16 + (k as usize % MIDI_EVENTS) * 16;
        unsafe {
            let p = self.base.add(slot);
            std::ptr::write_volatile(p as *mut u32, 0);
            fence(Ordering::Release);
            std::ptr::write_volatile(p.add(4), msg[0]);
            std::ptr::write_volatile(p.add(5), msg[1]);
            std::ptr::write_volatile(p.add(6), msg[2]);
            std::ptr::write_volatile(p.add(7), flags);
            std::ptr::write_volatile(p.add(8) as *mut u32, action);
            std::ptr::write_volatile(p.add(12) as *mut i32, steps);
            fence(Ordering::Release);
            std::ptr::write_volatile(p as *mut u32, (k + 1) as u32);
        }
        written.store(k + 1, Ordering::Release);
    }

    /// A frame of the master's spectrum, into the ring (see SPECTRUM_OFFSET). Its seq is 0 while it is written:
    /// the reader takes a frame whose seq is the frame's before and after it copies it.
    pub fn put_spectrum(&self, bins: &[u8; SPECTRUM_BINS]) {
        let written = self.atomic(SPECTRUM_OFFSET);
        let k = written.load(Ordering::Relaxed);
        let slot = SPECTRUM_OFFSET + 16 + (k as usize % SPECTRUM_FRAMES) * SPECTRUM_SLOT;
        unsafe {
            let p = self.base.add(slot);
            std::ptr::write_volatile(p as *mut u32, 0);
            fence(Ordering::Release);
            for (i, &v) in bins.iter().enumerate() {
                std::ptr::write_volatile(p.add(16 + i), v);
            }
            fence(Ordering::Release);
            std::ptr::write_volatile(p as *mut u32, (k + 1) as u32);
        }
        written.store(k + 1, Ordering::Release);
    }

    /// A histogram (st::HIST, or a bus's out::HIST) and its length.
    pub fn put_hist_at(&self, len_field: usize, field: usize, hist: &[f64]) {
        let n = hist.len().min(HIST_MAX);
        self.put_u32(len_field, n as u32);
        unsafe {
            std::ptr::copy_nonoverlapping(hist.as_ptr(), self.base.add(STATE_OFFSET + field) as *mut f64, n);
        }
    }

    pub fn put_hist(&self, hist: &[f64]) {
        self.put_hist_at(st::HIST_LEN, st::HIST, hist);
    }

    /// A short text: [u32 length + UTF-8], at most ``max`` bytes (cut at a character's start).
    pub fn put_str(&self, field: usize, text: &str, max: usize) {
        let mut n = text.len().min(max);
        while !text.is_char_boundary(n) {
            n -= 1;
        }
        self.put_u32(field, n as u32);
        unsafe {
            std::ptr::copy_nonoverlapping(text.as_ptr(), self.base.add(STATE_OFFSET + field + 4), n);
        }
    }
}

/// A command's payload read in order.
pub struct Reader<'a> {
    data: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Reader { data, at: 0 }
    }

    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        if n > self.data.len() - self.at {
            return None;
        }
        let s = &self.data[self.at..self.at + n];
        self.at += n;
        Some(s)
    }

    /// How many of ``n`` items of ``each`` bytes the rest of the payload holds: a count read from a command is
    /// never trusted for more (a loop over it ends with the payload).
    pub fn count(&self, n: u64, each: usize) -> usize {
        let room = (self.data.len() - self.at) / each.max(1);
        usize::try_from(n).map_or(room, |n| n.min(room))
    }

    pub fn u8(&mut self) -> Option<u8> {
        self.take(1).map(|b| b[0])
    }

    pub fn u32(&mut self) -> Option<u32> {
        self.take(4).map(|b| u32::from_le_bytes(b.try_into().unwrap()))
    }

    pub fn i32(&mut self) -> Option<i32> {
        self.take(4).map(|b| i32::from_le_bytes(b.try_into().unwrap()))
    }

    pub fn u64(&mut self) -> Option<u64> {
        self.take(8).map(|b| u64::from_le_bytes(b.try_into().unwrap()))
    }

    pub fn f32(&mut self) -> Option<f32> {
        self.take(4).map(|b| f32::from_le_bytes(b.try_into().unwrap()))
    }

    pub fn f64(&mut self) -> Option<f64> {
        self.take(8).map(|b| f64::from_le_bytes(b.try_into().unwrap()))
    }

    pub fn str(&mut self) -> Option<String> {
        let n = self.u32()? as usize;
        self.take(n).map(|b| String::from_utf8_lossy(b).into_owned())
    }
}
