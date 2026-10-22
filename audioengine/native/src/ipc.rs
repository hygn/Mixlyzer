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

use std::sync::atomic::{AtomicU64, Ordering, fence};

pub const MAGIC: u64 = u64::from_le_bytes(*b"MXAUDIC1");
pub const VERSION: u32 = 3;
pub const STATE_OFFSET: usize = 64;
pub const RING_OFFSET: usize = 65536;
pub const RING_HEADER: usize = 64;
pub const HIST_MAX: usize = 4096;
/// The output devices, written when asked (op::LIST_DEVICES): [generation u64, length u32, pad u32] + UTF-8 text,
/// the device in use on its first line, then one listed device a line. The generation is written last.
pub const DEVICES_OFFSET: usize = 40960;
pub const DEVICES_MAX: usize = RING_OFFSET - DEVICES_OFFSET - 16;
/// The MIDI input ports (op::MIDI_LIST): [generation u64, length u32, pad u32] + UTF-8 text, one port a line.
pub const MIDI_PORTS_OFFSET: usize = 34816;
pub const MIDI_PORTS_MAX: usize = MIDI_EVENTS_OFFSET - MIDI_PORTS_OFFSET - 16;
/// The MIDI messages received: [written u64 (messages ever), pad u64] + MIDI_EVENTS slots of 16 bytes, message k
/// at k % MIDI_EVENTS: [seq u32 (k + 1, written last), status u8, data1 u8, data2 u8, flags u8 (MIDI_EVENT_*),
/// action u32 (the binding's, 0: none), pad u32].
pub const MIDI_EVENTS_OFFSET: usize = 36864;
pub const MIDI_EVENTS: usize = 240;
/// The message is bound to one of the GUI's actions (its action code: 0x1000 and up, the GUI's numbering).
pub const MIDI_EVENT_GUI: u8 = 1;

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
    pub const SET_SOFT_CLIP: u32 = 15; // u32
    pub const SET_BUFFER_MS: u32 = 16; // f64
    pub const TONE_HOLD: u32 = 17; // u32 n, i32[n]
    pub const SET_MUSIC_GAIN: u32 = 18; // f64
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
    pub const LIST_DEVICES: u32 = 29;
    pub const SET_OUTPUT_MODE: u32 = 30; // u32 API (0 WASAPI, 1 cpal, 2 JACK), u32 exclusive (WASAPI)
    // u32 priority (0 normal, 1 above normal, 2 high), u32 power throttling allowed (sched.rs: per system)
    pub const SET_PROCESS: u32 = 31;
    // the deck (deck.rs): the GUI's requests, combined there with the MIDI controller's
    pub const DECK_ACTION: u32 = 32; // u32 action (act), u32 pressed (1) / released (0)
    pub const SET_TEMPO: u32 = 33; // f64 factor (the tempo fader's position)
    pub const SET_VOLUME: u32 = 34; // f64 the volume fader (0 .. 1)
    pub const SET_VOLUME_TRIM: u32 = 35; // f64 dB
    // u32 n, f64[n] beats (input frames), u8[n] downbeat, u32 m, f64[m] CUE points, f64[26] JumpCUEs A-Z (NaN: none)
    pub const DECK_MARKS: u32 = 36;
    pub const JUMP_TO: u32 = 37; // f64 frame: a JumpCUE's JUMP
    pub const CALL_AT: u32 = 38; // f64 frame: a CUE point's CALL
    pub const MIDI_OPEN: u32 = 39; // name (u32 len + utf8; empty: closed)
    // u32 n, n x [status u8, data1 u8, flags u8 (1: 14-bit, LSB at data1 + 32), lamp u8, action u32]; lamp: the
    // controller's light at that message (deck.rs LAMP_*: 1 - 26 the JumpCUE A - Z set, PLAY, CUE; 0 none)
    pub const MIDI_MAP: u32 = 40;
    pub const MIDI_LIST: u32 = 41;
    pub const MIDI_OPTIONS: u32 = 42; // u32 a jog wheel's messages (ticks) per turn
    pub const SET_TEMPO_RANGE: u32 = 43; // u32 the tempo fader's range (+-%: 6, 10, 16, 100 WIDE)
    pub const SET_METER_GAIN: u32 = 44; // f64 linear: the level meter's input, as the GUI's peak meter
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
    pub const JUMP_A: u32 = 16; // .. JUMP_A + 7: H
    // a MIDI controller's alone (values, not presses)
    pub const TEMPO_FADER: u32 = 32;
    pub const VOLUME_FADER: u32 = 33;
    pub const JOG_TOUCH: u32 = 34;
    pub const JOG_SCRATCH: u32 = 35;
    pub const JOG_BEND: u32 = 36;
    pub const JOG_SEARCH: u32 = 37;
    /// An output: the controller's level meter (a CC, 0 - 127 over the GUI meter's -24 - 0 dBFS).
    pub const LEVEL_METER: u32 = 38;
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

    /// The device in use and the devices listed (see DEVICES_OFFSET).
    pub fn put_devices(&self, current: &str, listed: &[String]) {
        let mut text = String::from(current);
        for name in listed {
            text.push('\n');
            text.push_str(name);
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
        let mut bytes = ports.join("
").into_bytes();
        bytes.truncate(MIDI_PORTS_MAX);
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.base.add(MIDI_PORTS_OFFSET + 16), bytes.len());
            std::ptr::write_volatile(self.base.add(MIDI_PORTS_OFFSET + 8) as *mut u32, bytes.len() as u32);
        }
        let generation = self.atomic(MIDI_PORTS_OFFSET);
        generation.store(generation.load(Ordering::Relaxed) + 1, Ordering::Release);
    }

    /// A MIDI message received, into the ring (see MIDI_EVENTS_OFFSET).
    pub fn put_midi_event(&self, msg: [u8; 3], flags: u8, action: u32) {
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
            fence(Ordering::Release);
            std::ptr::write_volatile(p as *mut u32, (k + 1) as u32);
        }
        written.store(k + 1, Ordering::Release);
    }

    pub fn put_hist(&self, hist: &[f64]) {
        let n = hist.len().min(HIST_MAX);
        self.put_u32(st::HIST_LEN, n as u32);
        unsafe {
            std::ptr::copy_nonoverlapping(hist.as_ptr(), self.base.add(STATE_OFFSET + st::HIST) as *mut f64, n);
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
