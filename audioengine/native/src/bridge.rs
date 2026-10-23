//! A bus played on a device of its own (the headphones', the booth's): the main device's callback, the clock the
//! decks play to, writes the bus's frames into a bridge; the other device's callback reads them at its own clock,
//! and at its own rate (opened at its mix format: a DJ controller's 44.1 kHz beside a 48 kHz main device).
//!
//! Two devices' clocks never run quite together (crystals a few tens to a few hundred ppm apart), so the reader
//! follows the writer: it plays the bus at a ratio of input frames per output frame (the rates' ratio, times the
//! clocks' drift) that keeps the bridge at its buffer, set by a slow PI loop on the bridge's level (a time constant
//! of seconds; the drift at most 1000 ppm, under 2 cents). The level is read as zita-ajbridge reads it, from where each device is playing: each side's position is
//! its device's play position (the frames it has given or taken, less those still queued in the device, stamped
//! with their time) and its mean queue (the writer's before its writes, the reader's after its reads: the level
//! held is the bridge's lowest), so the level does not saw with the devices' periods nor with how much each
//! callback takes (a shared stream tops its device up to full: more after a late wake-up). The level the fill left
//! is held at first (so the loop's integral starts on the drift alone) and moved to the buffer at SLEW, that speed
//! fed forward; another buffer setting is reached the same way. The frames are read between samples by a
//! band-limited interpolator (64-tap Kaiser-windowed sinc, its error under -85 dB up to 20 kHz at 44.1 and 48 kHz;
//! to a lower rate its cutoff the output's Nyquist frequency: what folds back lands above 20 kHz), so nothing is
//! repeated or dropped: the booth sounds as the master does. The bridge fills before the reader starts (and again
//! after it ran dry).
//!
//! A device takes one stream (Sink): the buses on it (the headphones on a controller's 3-4, the booth on its 1-2)
//! are each read from their bridge into their channels, summed where they share them.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use cpal::{FromSample, SizedSample};

use crate::clock;
use crate::device::{BRIDGED_BUSES, CallbackState, HeardClock, Port};

/// The interpolator's taps (its half on each side of the point read) and the phases of its kernel table.
const TAPS: usize = 64;
const HALF: usize = TAPS / 2;
const PHASES: usize = 2048;
const KAISER_BETA: f64 = 9.0; // the error under -85 dB up to 20 kHz at 44.1 and 48 kHz (-98 dB at 1 kHz)
/// The level is averaged over this much time before each step of the loop.
const CONTROL_SEC: f64 = 0.25;
/// The level held moves to the buffer at this speed (s per s: 500 ppm, under a cent).
const SLEW: f64 = 0.0005;
/// A device's mean queue (its frames queued about a callback) is followed over this time.
const QUEUE_TAU_SEC: f64 = 0.5;
/// After the bridge fills, the loop waits this long (the queues' means settled; the ratio held: even 300 ppm moves
/// the level under 1 ms meanwhile) before it takes the level to hold.
const SETTLE_SEC: f64 = 3.0;
/// The loop: natural frequency 0.2 rad/s, critically damped (KP = 2 w, KI = w^2), on the level's error in s.
const KP: f64 = 0.4;
const KI: f64 = 0.04;
/// The drift's room around 1 (the clocks' drift is well inside it).
const MAX_DRIFT: f64 = 0.001;
/// A surplus this far over the buffer (the reader stalled, the writer not; the buffer set this much lower) is
/// skipped at once.
const RESYNC_SEC: f64 = 0.1;
/// A reader held up longer than this (or three of its callbacks) between its device's callbacks (its thread late,
/// its device run dry meanwhile: already heard as a gap) starts again as at a restart: what was written meanwhile
/// dropped, the bridge filled to its buffer, the loop settled anew. Followed on instead, the catch-up's surplus and
/// its queue's jump were taken for drift: a 60 ms hold drove it to its 1000 ppm limit, drained at 1 ms/s for 50 s.
const GAP_SEC: f64 = 0.02;

pub struct Bridge {
    buf: Box<[UnsafeCell<f32>]>, // stereo frames
    cap: u64,
    rate: f64,
    write: AtomicU64,
    read: AtomicU64,
    pub sound_due: AtomicBool, // underruns count only while sound is expected (the engine sets it each tick)
    active: AtomicBool,     // the main callback writes while the output is open
    restart: AtomicBool,    // the reader drops what is there and fills again (the output opened or resumed)
    pair: AtomicU32,        // its device's channels: left | right << 8
    buffer: AtomicU64,      // frames: the buffer setting
    writer_max: AtomicU64,  // the most frames the main callback wrote at once
    // the device's rate against the bus's: input frames per output frame (f64 bits), and the kernel for it (the
    // reader takes both when it starts)
    nominal: AtomicU64,
    table: Mutex<Arc<Vec<f32>>>,
    // the mean frames the writer gives and the reader takes at once (f64 bits, bus frames): the buffer monitor's
    // capacity
    w_chunk: AtomicU64,
    r_chunk: AtomicU64,
    // the writer's smooth position at its last write (f64 bits: its device's play position and its mean queue
    // before a write) and the time then (f64 bits), under a sequence lock; that mean queue (the writer's own)
    w_seq: AtomicU64,
    w_base: AtomicU64,
    w_at: AtomicU64,
    w_queue_mean: UnsafeCell<f64>,
    w_chunk_mean: UnsafeCell<f64>,
    pub underruns: AtomicU64,
    pub resyncs: AtomicU64, // surpluses skipped (RESYNC_SEC): the outputs' thread reopens a device that keeps needing them
    drift: AtomicU64,       // f64 bits: the reader's ratio - 1
    heard: HeardClock,
}

unsafe impl Sync for Bridge {}
unsafe impl Send for Bridge {}

impl Bridge {
    pub fn new(cap_frames: u64, rate: f64) -> Bridge {
        Bridge {
            buf: (0..cap_frames as usize * 2).map(|_| UnsafeCell::new(0.0)).collect(),
            cap: cap_frames,
            rate,
            write: AtomicU64::new(0),
            read: AtomicU64::new(0),
            sound_due: AtomicBool::new(false),
            active: AtomicBool::new(false),
            restart: AtomicBool::new(false),
            pair: AtomicU32::new(1 << 8),
            buffer: AtomicU64::new(0),
            writer_max: AtomicU64::new(0),
            nominal: AtomicU64::new(1f64.to_bits()),
            table: Mutex::new(kernel(1.0)),
            w_chunk: AtomicU64::new(0f64.to_bits()),
            r_chunk: AtomicU64::new(0f64.to_bits()),
            w_seq: AtomicU64::new(0),
            w_base: AtomicU64::new(0f64.to_bits()),
            w_at: AtomicU64::new(0f64.to_bits()),
            w_queue_mean: UnsafeCell::new(-1.0),
            w_chunk_mean: UnsafeCell::new(-1.0),
            underruns: AtomicU64::new(0),
            resyncs: AtomicU64::new(0),
            drift: AtomicU64::new(0f64.to_bits()),
            heard: HeardClock::default(),
        }
    }

    // Engine side
    /// The bus's rate (the session's).
    pub fn rate(&self) -> f64 {
        self.rate
    }

    /// The device's rate (before it is opened): the reader's ratio and its kernel (made here, not in a callback).
    pub fn prepare(&self, device_rate: f64) {
        let nominal = if device_rate > 0.0 { self.rate / device_rate } else { 1.0 };
        // to a lower rate the kernel's cutoff is the output's Nyquist frequency
        let table = kernel((1.0 / nominal).min(1.0));
        *self.table.lock().unwrap_or_else(|e| e.into_inner()) = table;
        self.nominal.store(nominal.to_bits(), Ordering::Release);
    }

    /// The output opened (or resumed): the bus is written from now on, and the reader fills it to ``buffer_frames``
    /// before it plays, on its device's channels ``left`` / ``right``.
    pub fn start(&self, buffer_frames: u64, left: u8, right: u8) {
        self.set(buffer_frames, left, right);
        self.restart.store(true, Ordering::Release);
        self.active.store(true, Ordering::Release);
    }

    pub fn set(&self, buffer_frames: u64, left: u8, right: u8) {
        self.buffer.store(buffer_frames.min(self.cap / 2), Ordering::Relaxed);
        self.pair.store(left as u32 | (right as u32) << 8, Ordering::Relaxed);
    }

    pub fn stop(&self) {
        self.active.store(false, Ordering::Release);
    }

    pub fn active(&self) -> bool {
        self.active.load(Ordering::Acquire)
    }

    /// The level the reader keeps (frames): the setting, as the writer's position (writer_position) is ahead of
    /// what was read.
    fn target(&self) -> f64 {
        self.buffer.load(Ordering::Relaxed) as f64
    }

    /// The buffer kept (the setting) and what the writer's and the reader's chunks add to it (as they usually are),
    /// bus frames.
    pub fn capacity_frames(&self) -> f64 {
        self.target() + f64::from_bits(self.w_chunk.load(Ordering::Relaxed))
            + f64::from_bits(self.r_chunk.load(Ordering::Relaxed))
    }

    /// Frames written and not yet heard, ahead of the device's latency (the buffer monitor's level).
    pub fn buffered(&self, now: f64) -> f64 {
        let written = self.write.load(Ordering::Acquire) as f64;
        (written - self.heard.heard(now, self.rate) - self.heard.latency_frames(self.rate)).max(0.0)
    }

    pub fn latency_ms(&self) -> f64 {
        self.heard.latency_frames(self.rate) / self.rate * 1000.0
    }

    /// The reader's clock against the writer's (the loop's integral: its estimate, without the corrections of the
    /// level), in ppm.
    pub fn drift_ppm(&self) -> f64 {
        f64::from_bits(self.drift.load(Ordering::Relaxed)) * 1e6
    }

    // The writer: the main device's callback
    /// Appends stereo frames (those beyond its room are dropped: a reader that does not read).
    pub(crate) fn push(&self, data: &[f32]) {
        let w = self.write.load(Ordering::Relaxed);
        let free = self.cap - (w - self.read.load(Ordering::Acquire)).min(self.cap);
        let n = ((data.len() / 2) as u64).min(free);
        for k in 0..n {
            let at = ((w + k) % self.cap) as usize * 2;
            unsafe {
                *self.buf[at].get() = data[2 * k as usize];
                *self.buf[at + 1].get() = data[2 * k as usize + 1];
            }
        }
        self.write.store(w + n, Ordering::Release);
    }

    /// The callback wrote ``frames`` at once (its period), its device holding ``latency`` (s) before them.
    pub(crate) fn wrote(&self, frames: u64, latency: f64) {
        self.wrote_at(frames, latency, clock::now());
    }

    fn wrote_at(&self, frames: u64, latency: f64, now: f64) {
        self.writer_max.fetch_max(frames, Ordering::Relaxed);
        // the writer's device: its play position (what was written before these frames, less its queue) and its
        // mean queue before a write (the writer alone touches it): the line under the steps its writes make, so the
        // level held is the bridge's lowest
        let queue = latency.max(0.0) * self.rate;
        let dt = frames as f64 / self.rate;
        let mean = unsafe { &mut *self.w_queue_mean.get() };
        *mean = follow_mean(*mean, queue, dt);
        let chunk = unsafe { &mut *self.w_chunk_mean.get() };
        *chunk = follow_mean(*chunk, frames as f64, dt);
        self.w_chunk.store(chunk.to_bits(), Ordering::Relaxed);
        let played = (self.write.load(Ordering::Relaxed) - frames) as f64 - queue;
        let s = self.w_seq.load(Ordering::Relaxed);
        self.w_seq.store(s + 1, Ordering::Relaxed);
        std::sync::atomic::fence(Ordering::Release);
        self.w_base.store((played + *mean).to_bits(), Ordering::Relaxed);
        self.w_at.store(now.to_bits(), Ordering::Relaxed);
        self.w_seq.store(s + 2, Ordering::Release);
    }

    /// Where the writer is at ``now``, smoothly: its device's play position then and its mean queue (as far as
    /// a writer's period past its last write: one that stopped writing stops there).
    fn writer_position(&self, now: f64) -> f64 {
        let (base, at) = loop {
            let s1 = self.w_seq.load(Ordering::Acquire);
            if s1 & 1 == 1 {
                std::hint::spin_loop();
                continue;
            }
            let v = (f64::from_bits(self.w_base.load(Ordering::Relaxed)), f64::from_bits(self.w_at.load(Ordering::Relaxed)));
            std::sync::atomic::fence(Ordering::Acquire);
            if self.w_seq.load(Ordering::Relaxed) == s1 {
                break v;
            }
        };
        let most = 2.0 * self.writer_max.load(Ordering::Relaxed) as f64;
        base + ((now - at) * self.rate).clamp(0.0, most)
    }

    // The reader: its device's callback (Sink)
    /// The bus read into ``out`` (``ch_out`` channels a frame: its pair added to what is there), ``latency`` (s)
    /// before its device plays it.
    fn fill_at(&self, out: &mut [f32], ch_out: usize, rd: &mut BridgeReader, latency: f64, now: f64) {
        let frames = out.len() / ch_out;
        // a gap: longer than three of its callbacks (GAP_SEC at least): started again
        let usual = if rd.chunk_mean > 0.0 { rd.chunk_mean / self.rate } else { 0.0 };
        if rd.last_at > 0.0 && now - rd.last_at > GAP_SEC.max(3.0 * usual) {
            self.restart.store(true, Ordering::Release);
        }
        let write = self.write.load(Ordering::Acquire);
        let mut read = self.read.load(Ordering::Relaxed);
        if self.restart.swap(false, Ordering::AcqRel) {
            read = write;
            rd.reset();
            // the device's rate and kernel (set before the restart; only the engine's thread takes the lock, and
            // only to set them)
            if let Ok(table) = self.table.try_lock() {
                rd.table = table.clone();
            }
            rd.nominal = f64::from_bits(self.nominal.load(Ordering::Acquire));
            rd.ratio = rd.nominal;
        }
        rd.last_at = now;
        let pair = self.pair.load(Ordering::Relaxed);
        let (left, right) = ((pair & 0xFF) as usize, (pair >> 8 & 0xFF) as usize);
        let target = self.target();
        let start_read = read;
        let mut missing = 0u64;
        // filled: the buffer, this read, the interpolator's half, and a write of the writer's (the bridge then lasts
        // till its next one, whatever the two periods' phase)
        let chunk = self.writer_max.load(Ordering::Relaxed) as f64;
        if rd.filling && (write - read) as f64 >= target + frames as f64 * rd.ratio + HALF as f64 + chunk {
            rd.filling = false;
        }
        if !rd.filling {
            for f in 0..frames {
                while rd.frac >= 1.0 && read < write {
                    let at = (read % self.cap) as usize * 2;
                    rd.push(unsafe { [*self.buf[at].get(), *self.buf[at + 1].get()] });
                    read += 1;
                    rd.frac -= 1.0;
                }
                if rd.frac >= 1.0 {
                    // ran dry: silence until the bridge holds its buffer again
                    rd.filling = true;
                    if self.sound_due.load(Ordering::Relaxed) {
                        missing = (frames - f) as u64;
                    }
                    break;
                }
                let [l, r] = rd.interpolate();
                let o = &mut out[f * ch_out..(f + 1) * ch_out];
                if left < ch_out {
                    o[left] += l;
                }
                if right < ch_out {
                    o[right] += r;
                }
                rd.frac += rd.ratio;
            }
            if !rd.filling {
                // this side as the writer's: its device's play position (what was read before, less its queue) and
                // its mean queue after a callback, in bus frames
                let queue = latency.max(0.0) * self.rate;
                let dt = frames as f64 * rd.nominal / self.rate;
                rd.queue_mean = follow_mean(rd.queue_mean, queue + frames as f64 * rd.nominal, dt);
                rd.chunk_mean = follow_mean(rd.chunk_mean, frames as f64 * rd.nominal, dt);
                self.r_chunk.store(rd.chunk_mean.to_bits(), Ordering::Relaxed);
                let position = start_read as f64 - queue + rd.queue_mean;
                let level = self.writer_position(now) - position;
                if let Some(skip) = rd.follow(level, dt, target, self.rate) {
                    read = (read + skip).min(write);
                    rd.reset_history();
                    self.resyncs.fetch_add(1, Ordering::Relaxed);
                }
                self.drift.store((KI * rd.integral).to_bits(), Ordering::Relaxed);
            }
        }
        self.read.store(read, Ordering::Release);
        if missing > 0 {
            self.underruns.fetch_add(missing, Ordering::Relaxed);
        }
        // the frame read now is heard after the interpolator's half and the device's latency
        let delay = HALF as f64 / self.rate;
        self.heard.report(start_read.saturating_sub(HALF as u64), read - start_read, now, latency + delay);
    }
}

/// A device's stream: the buses on it (bus index, its bridge), read into their channels.
pub struct Sink {
    pub port: Arc<Port>,
    rate: f64, // the device's
    bridges: Vec<(usize, Arc<Bridge>)>,
}

impl Sink {
    pub fn new(port: Arc<Port>, rate: f64, bridges: Vec<(usize, Arc<Bridge>)>) -> Sink {
        Sink { port, rate, bridges }
    }

    /// Its readers fill again from what is written now (Source::restarted).
    pub fn restart_bridges(&self) {
        for (_, b) in &self.bridges {
            b.restart.store(true, Ordering::Release);
        }
    }

    /// The device's callback: each bus read into its channels (summed where they share them), in pieces of the
    /// callback's mix buffer.
    pub(crate) fn fill<T: SizedSample + FromSample<f32>>(&self, out: &mut [T], state: &mut CallbackState, latency: f64) {
        #[cfg(target_os = "linux")]
        crate::sched::output_thread(&self.port);
        let ch = state.channels.max(1);
        let frames = out.len() / ch;
        self.port.max_frames.fetch_max(frames as u64, Ordering::Relaxed);
        let now = clock::now();
        let piece = (state.mix.len() / ch).max(1);
        let mut done = 0;
        while done < frames {
            let n = (frames - done).min(piece);
            let mix = &mut state.mix[..n * ch];
            mix.iter_mut().for_each(|v| *v = 0.0);
            // a later piece is heard as much later
            let lat = latency + done as f64 / self.rate;
            for (k, b) in &self.bridges {
                if let Some(rd) = state.readers.get_mut(*k) {
                    b.fill_at(mix, ch, rd, lat, now);
                }
            }
            for (o, v) in out[done * ch..(done + n) * ch].iter_mut().zip(mix.iter()) {
                *o = T::from_sample(*v);
            }
            done += n;
        }
    }
}

/// Every bus's reader, as a device's callback keeps them.
pub(crate) fn readers() -> Vec<Box<BridgeReader>> {
    (0..BRIDGED_BUSES).map(|_| Box::new(BridgeReader::new())).collect()
}

/// The reader's state (in its callback's).
pub(crate) struct BridgeReader {
    window: [[f32; 2]; 2 * TAPS], // the last TAPS frames, twice over: window[pos .. pos + TAPS] in order
    pos: usize,
    frac: f64,     // the point read, past window[HALF - 1] (input frames)
    ratio: f64,    // input frames per output frame: the rates' ratio times the clocks'
    nominal: f64,  // the rates' ratio
    table: Arc<Vec<f32>>, // the kernel
    integral: f64, // the loop's integral of the level's error (s * s)
    filling: bool, // waiting for the bridge to hold its buffer
    level_sum: f64,
    level_time: f64,
    queue_mean: f64, // its device's mean queue after a callback (bus frames; < 0: not yet)
    chunk_mean: f64, // the frames it takes at once (bus frames; < 0: not yet)
    // the level held (frames; None: not yet): at first the one the fill left, so the loop's integral starts on the
    // drift alone, not on the fill's surplus; then moved to the buffer at SLEW
    hold: Option<f64>,
    settled: f64, // seconds played since the bridge filled (up to SETTLE_SEC)
    last_at: f64, // its last callback's time (0: none yet)
}

impl BridgeReader {
    pub(crate) fn new() -> BridgeReader {
        BridgeReader {
            window: [[0.0; 2]; 2 * TAPS],
            pos: 0,
            frac: 0.0,
            ratio: 1.0,
            nominal: 1.0,
            table: kernel(1.0),
            integral: 0.0,
            filling: true,
            level_sum: 0.0,
            level_time: 0.0,
            queue_mean: -1.0,
            chunk_mean: -1.0,
            hold: None,
            settled: 0.0,
            last_at: 0.0,
        }
    }

    /// A new start (another device, or the same one resumed): the clocks' estimate goes too (the kernel is kept
    /// till the bridge's is taken).
    fn reset(&mut self) {
        let table = std::mem::replace(&mut self.table, kernel(1.0));
        *self = BridgeReader { table, ..BridgeReader::new() };
    }

    fn reset_history(&mut self) {
        self.window = [[0.0; 2]; 2 * TAPS];
        self.level_sum = 0.0;
        self.level_time = 0.0;
        self.hold = None;
        self.settled = 0.0;
    }

    fn push(&mut self, frame: [f32; 2]) {
        self.window[self.pos] = frame;
        self.window[self.pos + TAPS] = frame;
        self.pos = (self.pos + 1) % TAPS;
    }

    /// The bus at the point read (between window[HALF - 1] and window[HALF], ``frac`` past the first).
    fn interpolate(&self) -> [f32; 2] {
        let table = &self.table[..];
        let p = self.frac * PHASES as f64;
        let k = (p as usize).min(PHASES - 1);
        let a = (p - k as f64) as f32;
        let (h0, h1) = (&table[k * TAPS..(k + 1) * TAPS], &table[(k + 1) * TAPS..(k + 2) * TAPS]);
        let w = &self.window[self.pos..self.pos + TAPS];
        let (mut l, mut r) = (0f32, 0f32);
        for j in 0..TAPS {
            let h = h0[j] + a * (h1[j] - h0[j]);
            l += h * w[j][0];
            r += h * w[j][1];
        }
        [l, r]
    }

    /// The loop, after a read that left the writer ``level`` bus frames ahead, ``dt`` s of output: every
    /// CONTROL_SEC, the mean level's error against the level held sets the ratio; the level held moves to
    /// ``target`` (the buffer) at SLEW. Returns the frames to skip when far over the buffer.
    fn follow(&mut self, level: f64, dt: f64, target: f64, rate: f64) -> Option<u64> {
        if self.settled < SETTLE_SEC {
            self.settled += dt;
            return None;
        }
        self.level_sum += level * dt;
        self.level_time += dt;
        if self.level_time < CONTROL_SEC {
            return None;
        }
        let mean = self.level_sum / self.level_time;
        let step = self.level_time;
        self.level_sum = 0.0;
        self.level_time = 0.0;
        // at first the level the fill left (the bridge held the buffer and a read then), moved to the buffer at
        // SLEW: the reader's speed for that move fed forward, so the integral stays the clocks' drift
        let hold = self.hold.get_or_insert(mean);
        let room = SLEW * rate * step;
        let moved = (target - *hold).clamp(-room, room);
        *hold += moved;
        let feed = -moved / (rate * step);
        let error = (mean - *hold) / rate;
        // far over the buffer itself (not the level held: a reader stalled while the bridge settled would have its
        // surplus held, then moved off at SLEW only): skipped at once
        if (mean - target) / rate > RESYNC_SEC {
            return Some((mean - target) as u64);
        }
        self.integral = (self.integral + error * step).clamp(-MAX_DRIFT / KI, MAX_DRIFT / KI);
        let room = MAX_DRIFT + SLEW;
        self.ratio = self.nominal * (1.0 + (KP * error + KI * self.integral + feed).clamp(-room, room));
        None
    }
}

/// A mean followed over QUEUE_TAU_SEC: ``mean`` (< 0: none yet) after ``value``, ``dt`` s later.
fn follow_mean(mean: f64, value: f64, dt: f64) -> f64 {
    if mean < 0.0 {
        return value;
    }
    mean + (1.0 - (-dt / QUEUE_TAU_SEC).exp()) * (value - mean)
}

/// The kernel at the same rates made (main.rs, at the start: not in a device's callback).
pub fn warm() {
    kernel(1.0);
}

/// The interpolator's kernel with its cutoff at ``cutoff`` (of the input's Nyquist frequency): PHASES + 1 rows of
/// TAPS, row p for the point p / PHASES past window[HALF - 1] (each row summing to 1: a constant passes as it is).
/// Each one made is kept (a reader's never freed in its callback).
fn kernel(cutoff: f64) -> Arc<Vec<f32>> {
    static TABLES: Mutex<Vec<(u64, Arc<Vec<f32>>)>> = Mutex::new(Vec::new());
    let fc = cutoff.clamp(0.1, 1.0);
    let mut tables = TABLES.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((_, t)) = tables.iter().find(|(key, _)| *key == fc.to_bits()) {
        return t.clone();
    }
    let table = Arc::new({
        let i0 = |x: f64| {
            // the modified Bessel function of the first kind, order 0 (its series)
            let (mut sum, mut term) = (1.0, 1.0);
            for k in 1..50 {
                term *= (x / 2.0) / k as f64;
                sum += term * term;
            }
            sum
        };
        let norm = i0(KAISER_BETA);
        let mut table = Vec::with_capacity((PHASES + 1) * TAPS);
        for p in 0..=PHASES {
            let frac = p as f64 / PHASES as f64;
            let row: Vec<f64> = (0..TAPS)
                .map(|j| {
                    let t = frac + (HALF - 1) as f64 - j as f64; // the point's distance from tap j
                    let x = t / HALF as f64;
                    if x.abs() >= 1.0 {
                        return 0.0;
                    }
                    let u = fc * t;
                    let sinc = if u == 0.0 { 1.0 } else { (std::f64::consts::PI * u).sin() / (std::f64::consts::PI * u) };
                    sinc * i0(KAISER_BETA * (1.0 - x * x).sqrt()) / norm
                })
                .collect();
            let sum: f64 = row.iter().sum();
            table.extend(row.iter().map(|v| (v / sum) as f32));
        }
        table
    });
    tables.push((fc.to_bits(), table.clone()));
    table
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The interpolator read along a sine (``rate``) at ``ratio``, its kernel cut at ``cutoff``: its error against
    /// the sine itself.
    fn interpolation_error_db(freq: f64, ratio: f64) -> f64 {
        resampled_error_db(freq, 48000.0, ratio, 1.0)
    }

    fn resampled_error_db(freq: f64, rate: f64, ratio: f64, cutoff: f64) -> f64 {
        let mut rd = BridgeReader::new();
        rd.ratio = ratio;
        rd.table = kernel(cutoff);
        let x = |n: f64| (2.0 * std::f64::consts::PI * freq * n / rate).sin();
        let mut next = 0u64; // input frames pushed
        let (mut err, mut sig) = (0.0, 0.0);
        for f in 0..20000 {
            while rd.frac >= 1.0 || next < TAPS as u64 {
                rd.push([x(next as f64) as f32; 2]);
                next += 1;
                if rd.frac >= 1.0 {
                    rd.frac -= 1.0;
                }
            }
            let y = rd.interpolate()[0] as f64;
            // the point read: window[HALF - 1] is the frame pushed TAPS - HALF + 1 before the newest
            let at = (next - 1 - HALF as u64) as f64 + rd.frac;
            if f > 1000 {
                err += (y - x(at)).powi(2);
                sig += x(at).powi(2);
            }
            rd.frac += rd.ratio;
        }
        10.0 * (err / sig).log10()
    }

    #[test]
    fn interpolator_is_clean_to_20k() {
        for (freq, limit) in [(1000.0, -95.0), (10000.0, -95.0), (18000.0, -95.0), (20000.0, -95.0)] {
            let db = interpolation_error_db(freq, 1.0003);
            assert!(db < limit, "{freq} Hz: {db:.1} dB");
        }
    }

    #[test]
    fn to_a_lower_rate_clean_where_heard() {
        // 48 kHz to 44.1 kHz: the audible band as clean as at the same rate
        let ratio = 48000.0 / 44100.0;
        for (freq, limit) in [(1000.0, -95.0), (10000.0, -95.0), (18000.0, -80.0)] {
            let db = resampled_error_db(freq, 48000.0, ratio, 1.0 / ratio);
            assert!(db < limit, "{freq} Hz: {db:.1} dB");
        }
        // 44.1 kHz to 48 kHz: the input's whole band
        for (freq, limit) in [(1000.0, -95.0), (18000.0, -90.0)] {
            let db = resampled_error_db(freq, 44100.0, 44100.0 / 48000.0, 1.0);
            assert!(db < limit, "up {freq} Hz: {db:.1} dB");
        }
    }

    #[test]
    fn constant_passes_unchanged() {
        let mut rd = BridgeReader::new();
        for _ in 0..TAPS {
            rd.push([0.5, -0.25]);
        }
        for p in 0..64 {
            rd.frac = p as f64 / 64.0;
            let [l, r] = rd.interpolate();
            assert!((l - 0.5).abs() < 1e-6 && (r + 0.25).abs() < 1e-6);
        }
    }

    /// The writer at the main device's clock (``drift`` off the reader's), in its periods, the reader in its own at
    /// ``out_rate``, both woken late now and then by up to ``jitter`` s and topping their device up to full (a shared
    /// stream): the bridge is followed without running dry, the ratio settles on the drift, the level on the buffer.
    /// ``stall``: the reader not called from a time for so long (its device's callbacks held up, the writer's not).
    fn follow_drift(drift: f64, jitter: f64, out_rate: f64, stall: Option<(f64, f64)>) {
        let rate = 48000.0;
        let bridge = Bridge::new(48000, rate);
        let (w_period, r_period) = (480usize, (out_rate / 100.0) as usize - 7);
        let buffer = 240u64; // 5 ms
        bridge.prepare(out_rate);
        bridge.start(buffer, 0, 1);
        let mut rd = BridgeReader::new();
        // each device's buffer (two periods), its play position at its own clock; a wake-up's frames: what is free
        let (w_rate, r_rate) = (rate * (1.0 + drift), out_rate);
        let (w_buf, r_buf) = (2.0 * w_period as f64, 2.0 * r_period as f64);
        let (mut w_given, mut r_taken) = (0.0f64, 0.0f64);
        let (mut tw, mut tr) = (0.0f64, 0.0f64);
        let mut seed = 12345u64;
        let mut late = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            jitter * ((seed >> 33) as f64 / (1u64 << 31) as f64)
        };
        let mut drifts = Vec::new();
        let mut early = f64::NAN;
        let mut most_after_stall = 0u64;
        while tr < 200.0 {
            if tw <= tr {
                let queue = (w_given - tw * w_rate).max(0.0);
                let n = (w_buf - queue).round().max(0.0) as usize;
                bridge.push(&vec![0.1f32; n * 2]);
                bridge.wrote_at(n as u64, queue / w_rate, tw);
                w_given += n as f64;
                tw += w_period as f64 / w_rate + late();
            } else {
                if let Some((at, len)) = stall {
                    if tr >= at && tr < at + len {
                        tr = at + len; // its device's clock went on: its queue played out meanwhile
                        continue;
                    }
                    if tr > at + len + SETTLE_SEC + 2.0 {
                        let level = bridge.write.load(Ordering::Relaxed) - bridge.read.load(Ordering::Relaxed);
                        most_after_stall = most_after_stall.max(level);
                    }
                }
                let queue = (r_taken - tr * r_rate).max(0.0);
                let n = (r_buf - queue).round().max(0.0) as usize;
                let mut out = vec![0f32; n * 2];
                bridge.fill_at(&mut out, 2, &mut rd, queue / r_rate, tr);
                r_taken += n as f64;
                if tr > 150.0 {
                    drifts.push(bridge.drift_ppm());
                }
                if (tr - 20.0).abs() < 0.005 {
                    early = bridge.drift_ppm();
                }
                tr += r_period as f64 / r_rate + late();
            }
        }
        assert!(!rd.filling, "ran dry");
        assert_eq!(bridge.underruns.load(Ordering::Relaxed), 0);
        // the level held reached the buffer
        assert!((rd.hold.unwrap() - bridge.target()).abs() < 1.0, "held {:?} for {}", rd.hold, bridge.target());
        let mean = drifts.iter().sum::<f64>() / drifts.len() as f64;
        let spread = drifts.iter().fold(0f64, |m, &v| m.max((v - mean).abs()));
        eprintln!("drift {:+.0} ppm, jitter {:.1} ms, {out_rate} Hz: read {mean:+.2} +- {spread:.2} ppm (at 20 s {early:+.1})",
                  drift * 1e6, jitter * 1e3);
        if stall.is_some() {
            // back to the buffer, not holding the stall's surplus
            let most_ms = most_after_stall as f64 / rate * 1000.0;
            eprintln!("  after the stall {stall:?}: the bridge held at most {most_ms:.1} ms");
            assert!(most_ms < (bridge.target() / rate + RESYNC_SEC) * 1000.0, "held {most_ms:.1} ms after the stall");
        } else {
            // on its way there from 0, not past it (the fill's surplus is no drift)
            assert!(early * drift >= 0.0 && early.abs() <= 1.05 * drift.abs() * 1e6 + 15.0, "at 20 s {early:.1} ppm");
        }
        assert!((mean - drift * 1e6).abs() < 2.0 && spread < 8.0, "drift {mean:.2} +- {spread:.2} ppm for {}", drift * 1e6);
    }

    /// Two buses on one device: each into its pair, summed where they share one.
    #[test]
    fn buses_share_a_device() {
        let rate = 48000.0;
        let (a, b) = (Arc::new(Bridge::new(48000, rate)), Arc::new(Bridge::new(48000, rate)));
        for (bridge, pair) in [(&a, (2u8, 3u8)), (&b, (0, 1))] {
            bridge.prepare(rate);
            bridge.start(0, pair.0, pair.1);
        }
        let sink = Sink::new(Arc::new(Port::default()), rate, vec![(0, a.clone()), (1, b.clone())]);
        let mut state = CallbackState::new(4, rate);
        let mut out = vec![0f32; 480 * 4];
        // the restart taken (what was there dropped), then the writer's frames
        sink.fill(&mut out, &mut state, 0.0);
        a.push(&vec![0.25f32; 48000]);
        b.push(&vec![0.5f32; 48000]);
        for _ in 0..10 {
            sink.fill(&mut out, &mut state, 0.0);
        }
        let last = &out[out.len() - 4..];
        assert!((last[0] - 0.5).abs() < 1e-5 && (last[1] - 0.5).abs() < 1e-5, "{last:?}");
        assert!((last[2] - 0.25).abs() < 1e-5 && (last[3] - 0.25).abs() < 1e-5, "{last:?}");
        // the same pair: summed
        a.start(0, 0, 1);
        sink.fill(&mut out, &mut state, 0.0);
        a.push(&vec![0.25f32; 48000]);
        for _ in 0..10 {
            sink.fill(&mut out, &mut state, 0.0);
        }
        let last = &out[out.len() - 4..];
        assert!((last[0] - 0.75).abs() < 1e-5 && last[2].abs() < 1e-6, "{last:?}");
    }

    #[test]
    fn follows_a_faster_and_a_slower_clock() {
        for jitter in [0.0, 0.002] {
            for out_rate in [48000.0, 44100.0] {
                follow_drift(250e-6, jitter, out_rate, None);
                follow_drift(-180e-6, jitter, out_rate, None);
                follow_drift(0.0, jitter, out_rate, None);
            }
        }
    }

    /// The reader held up for 1.5 s (the bridge full), while it settles and later: back to its buffer at once.
    #[test]
    fn a_stalled_reader_comes_back_to_its_buffer() {
        for at in [1.0, 60.0] {
            follow_drift(120e-6, 0.002, 44100.0, Some((at, 1.5)));
        }
    }
}
