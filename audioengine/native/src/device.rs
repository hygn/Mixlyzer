//! The output devices (cpal: WASAPI, ALSA / PulseAudio / JACK, CoreAudio; wasapi.rs) and the rings the engine
//! renders into ahead of them.
//!
//! The main device's callback only copies from the ring: rendering (the time stretch's tens of ms for a new
//! stream) happens on the engine's thread, which keeps the ring filled to the output buffer. Frames are
//! numbered by their ring index (counted since the engine started, never wrapping); the callback reports the
//! index it starts at and when that frame is heard (its time + the device's latency), so the engine knows the
//! frame being heard at any moment (`heard`). A flush (pause, a grab of the record) drops what the ring
//! holds from the read point on, faded out over a few ms in the callback itself (no click; the device's own
//! buffer, ~10 ms, still plays out).
//!
//! Each deck's player renders into a ring of its own (its lane: [music, aids] a frame, before its mixer channel).
//! The main device's ring is deck 1's lane; the others hang on it (Bus), and the callback reads every lane as it
//! reads its own (a flush fades one deck out, the others play on), then runs the mixer (mixer.rs: each channel's
//! strip, the crossfader, the master, the headphones) and writes the device's channels as routed. Deck 1 alone at
//! its mixer channel's neutral settings, on the first two channels, comes out as its player rendered it. The
//! outputs run from their opening to their end: silence while nothing plays.
//!
//! The headphones and the booth may go to devices of their own (bridge.rs): the main callback, the clock the
//! decks play to, writes those buses into a bridge each, which the other device's callback reads at its own clock.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SizedSample};

use crate::bridge::{Bridge, BridgeReader, Sink};
use crate::clock;
use crate::mixer::{Mixer, NO_CHANNEL};

/// An open device's side of things: what its output thread reports and is told. The main device's lanes share
/// one; each bridge has its own.
#[derive(Default)]
pub struct Port {
    pub sound_due: AtomicBool, // late wake-ups count only while sound is expected (the engine sets it each tick)
    pub max_frames: AtomicU64, // the most frames the device took at once since it was opened (its period or so)
    // the output thread woken late (WASAPI: more than 1.8 periods after the last wake-up: the device may have
    // played a gap although the ring held audio), and the longest gap between wake-ups (µs)
    pub late: AtomicU64,
    pub max_gap_us: AtomicU64,
    pub failed: AtomicBool,
    // the engine thread, woken after each read of the main device (it tops the ring up at once, not at its next
    // tick)
    pub engine_thread: OnceLock<std::thread::Thread>,
    // the output thread's scheduling (sched.rs): its state where its backend set it (the engine's own WASAPI
    // thread: MMCSS), else the engine's; Linux: its thread id (written by the callback) and the timer slack it
    // should take (it can set only its own: ns + 1, 0 = not asked)
    pub output_state: AtomicU32,
    pub output_tid: AtomicU64,
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    pub output_slack: AtomicU64,
}

impl Port {
    /// A device opened again: its period and thread are seen anew.
    pub fn reset(&self) {
        self.max_frames.store(0, Ordering::Relaxed);
        self.output_state.store(0, Ordering::Relaxed);
        self.output_tid.store(0, Ordering::Relaxed);
    }
}

/// When a reader's frames are heard: its callback's report (the frame it started at, heard after the device's
/// latency, and the frames it took), under a sequence lock, and the frame being heard at any moment from it.
#[derive(Default)]
pub struct HeardClock {
    seq: AtomicU64,
    read: AtomicU64,
    at: AtomicU64, // f64 bits: clock::now() when `read` is heard
    frames: AtomicU64,
    latency: AtomicU64, // f64 bits, s
}

impl HeardClock {
    pub fn latency_frames(&self, rate: f64) -> f64 {
        f64::from_bits(self.latency.load(Ordering::Relaxed)) * rate
    }

    /// The frame being heard at clock time ``now`` (fractional).
    pub fn heard(&self, now: f64, rate: f64) -> f64 {
        let (r0, at, n0) = loop {
            let s1 = self.seq.load(Ordering::Acquire);
            if s1 & 1 == 1 {
                std::hint::spin_loop();
                continue;
            }
            let r0 = self.read.load(Ordering::Relaxed);
            let at = f64::from_bits(self.at.load(Ordering::Relaxed));
            let n0 = self.frames.load(Ordering::Relaxed);
            std::sync::atomic::fence(Ordering::Acquire);
            if self.seq.load(Ordering::Relaxed) == s1 {
                break (r0, at, n0);
            }
        };
        if at == 0.0 {
            return r0 as f64;
        }
        let elapsed = (now - at) * rate;
        (r0 as f64 + elapsed.clamp(-self.latency_frames(rate) - n0 as f64, n0 as f64)).max(0.0)
    }

    /// Frame ``start`` is heard after ``latency``, and the frames after it up to those taken. A callback that took
    /// nothing (silence alone) leaves the last report: the frames it took go on being heard after it, up to the
    /// last one (reported again, the frame heard would stay behind it).
    pub fn report(&self, start: u64, taken: u64, now: f64, latency: f64) {
        if taken == 0 && f64::from_bits(self.at.load(Ordering::Relaxed)) != 0.0 {
            self.latency.store(latency.to_bits(), Ordering::Relaxed);
            return;
        }
        let s = self.seq.load(Ordering::Relaxed);
        self.seq.store(s + 1, Ordering::Relaxed);
        std::sync::atomic::fence(Ordering::Release);
        self.read.store(start, Ordering::Relaxed);
        self.at.store((now + latency).to_bits(), Ordering::Relaxed);
        self.frames.store(taken, Ordering::Relaxed);
        self.latency.store(latency.to_bits(), Ordering::Relaxed);
        self.seq.store(s + 2, Ordering::Release);
    }
}

pub struct Ring {
    buf: Box<[UnsafeCell<f32>]>,
    cap: u64, // frames
    ch: usize,
    rate: f64,
    fade: u64,
    write: AtomicU64,
    read: AtomicU64,
    flush_gen: AtomicU64,
    flush_cut: AtomicU64,
    heard: HeardClock,
    pub sound_due: AtomicBool, // underruns count only while sound is expected
    pub underruns: AtomicU64,
    pub port: Arc<Port>, // the main device's (every lane's)
    // the device ring's (deck 1's main lane): the other lanes and the mix (attach_bus)
    bus: OnceLock<Bus>,
}

unsafe impl Sync for Ring {}
unsafe impl Send for Ring {}

impl Ring {
    pub fn new(cap_frames: u64, ch: usize, rate: f64, port: Arc<Port>) -> Ring {
        let buf: Vec<UnsafeCell<f32>> = (0..cap_frames as usize * ch).map(|_| UnsafeCell::new(0.0)).collect();
        Ring {
            buf: buf.into_boxed_slice(),
            cap: cap_frames,
            ch,
            rate,
            fade: (rate / 250.0).max(1.0) as u64, // 4 ms
            write: AtomicU64::new(0),
            read: AtomicU64::new(0),
            flush_gen: AtomicU64::new(0),
            flush_cut: AtomicU64::new(0),
            heard: HeardClock::default(),
            sound_due: AtomicBool::new(false),
            underruns: AtomicU64::new(0),
            port,
            bus: OnceLock::new(),
        }
    }

    // Engine side
    /// Frames rendered so far: the next frame's index.
    pub fn written(&self) -> u64 {
        self.write.load(Ordering::Relaxed)
    }

    pub fn free(&self) -> u64 {
        self.cap - (self.write.load(Ordering::Relaxed) - self.read.load(Ordering::Acquire))
    }

    /// The next frame the callback takes.
    pub fn read_index(&self) -> u64 {
        self.read.load(Ordering::Acquire)
    }

    /// Frames written and not yet taken by the callback.
    pub fn level(&self) -> u64 {
        self.write.load(Ordering::Relaxed).saturating_sub(self.read.load(Ordering::Acquire))
    }

    /// Appends interleaved frames (the caller keeps within `free`).
    pub fn push(&self, data: &[f32]) {
        let n = (data.len() / self.ch) as u64;
        debug_assert!(n <= self.free());
        let w = self.write.load(Ordering::Relaxed);
        for (k, frame) in data.chunks_exact(self.ch).enumerate() {
            let at = ((w + k as u64) % self.cap) as usize * self.ch;
            for c in 0..self.ch {
                unsafe { *self.buf[at + c].get() = frame[c] };
            }
        }
        self.write.store(w + n, Ordering::Release);
    }

    /// Drops what the ring holds from the frame being read on (faded out by the callback).
    pub fn flush(&self) {
        self.flush_cut.store(self.write.load(Ordering::Relaxed), Ordering::Relaxed);
        self.flush_gen.fetch_add(1, Ordering::Release);
    }

    pub fn latency_frames(&self) -> f64 {
        self.heard.latency_frames(self.rate)
    }

    /// The frame being heard at clock time ``now`` (fractional).
    pub fn heard(&self, now: f64) -> f64 {
        self.heard.heard(now, self.rate)
    }

    /// The other decks' lanes and the mix (main.rs sets it once, before the device is opened): this ring is deck
    /// 1's main lane and the main device's.
    pub fn attach_bus(&self, bus: Bus) {
        let _ = self.bus.set(bus);
    }

    // Callback side
    /// Reads up to ``frames`` frames of this lane into ``dst`` (``self.ch`` a frame; silence where the lane has none),
    /// a flush's fade applied; returns the frames taken (played) and those missing while sound was due.
    fn read_lane(&self, dst: &mut [f32], frames: usize, ls: &mut LaneState) -> (u64, u64) {
        let generation = self.flush_gen.load(Ordering::Acquire);
        if generation != ls.flush_seen {
            ls.flush_seen = generation;
            ls.cut = self.flush_cut.load(Ordering::Relaxed);
            ls.fade_left = self.fade;
        }
        let mut read = self.read.load(Ordering::Relaxed);
        let write = self.write.load(Ordering::Acquire);
        let due = self.sound_due.load(Ordering::Relaxed);
        let (mut taken, mut missing) = (0u64, 0u64);
        for f in 0..frames {
            let o = &mut dst[f * self.ch..(f + 1) * self.ch];
            if ls.fade_left > 0 && read >= ls.cut {
                ls.fade_left = 0; // the old audio ran out: the new goes on as it is
            }
            if read < write {
                let at = (read % self.cap) as usize * self.ch;
                let gain = if ls.fade_left > 0 {
                    let g = ls.fade_left as f32 / self.fade as f32;
                    ls.fade_left -= 1;
                    if ls.fade_left == 0 {
                        read = read.max(ls.cut).saturating_sub(1); // +1 below
                    }
                    g
                } else {
                    1.0
                };
                for (c, v) in o.iter_mut().enumerate() {
                    *v = unsafe { *self.buf[at + c].get() } * gain;
                }
                read += 1;
                taken += 1;
            } else {
                if ls.fade_left > 0 {
                    ls.fade_left = 0;
                    read = read.max(ls.cut);
                }
                o.iter_mut().for_each(|v| *v = 0.0);
                if due {
                    missing += 1;
                }
            }
        }
        self.read.store(read, Ordering::Release);
        (taken, missing)
    }

    /// The lane's report (HeardClock::report) and its underruns.
    fn report(&self, start_read: u64, taken: u64, missing: u64, now: f64, latency: f64) {
        if missing > 0 {
            self.underruns.fetch_add(missing, Ordering::Relaxed);
        }
        self.heard.report(start_read, taken, now, latency);
    }

    /// The main device's callback: every lane read (deck 1's this ring, the others' from the bus), mixed (mixer.rs),
    /// the buses written to the device's channels as routed and into the bridges of those on devices of their own.
    pub(crate) fn fill<T: SizedSample + FromSample<f32>>(&self, out: &mut [T], state: &mut CallbackState, latency: f64) {
        #[cfg(target_os = "linux")]
        crate::sched::output_thread(&self.port);
        let ch_out = state.channels;
        let frames = out.len() / ch_out;
        let now = clock::now();
        self.port.max_frames.fetch_max(frames as u64, Ordering::Relaxed);
        let Some(bus) = self.bus.get() else {
            out.iter_mut().for_each(|v| *v = T::EQUILIBRIUM);
            return;
        };
        let lanes = 1 + bus.decks.len();
        let mix = bus.params.read();
        let bridges: [Option<&Bridge>; BRIDGED_BUSES] = std::array::from_fn(|k| Some(&*bus.bridges[k]).filter(|b| b.active()));
        let lch = self.ch; // a lane's frame: the deck's music and aids
        let lane_len = SCRATCH_FRAMES * lch;
        let [_, _, cue_l, cue_r, _, _] = mix.routes;
        let headphones = cue_l != NO_CHANNEL || cue_r != NO_CHANNEL || bridges[0].is_some();
        // each lane's report covers the whole callback: where it started and what it took
        let mut start_read = [0u64; MAX_DECKS];
        let mut taken = [0u64; MAX_DECKS];
        let mut missing = [0u64; MAX_DECKS];
        for k in 0..lanes {
            start_read[k] = self.lane(k).read.load(Ordering::Relaxed);
        }
        let mut done = 0;
        while done < frames {
            let n = (frames - done).min(SCRATCH_FRAMES);
            let mut gave = [false; MAX_DECKS];
            for k in 0..lanes {
                let (t, m) = self.lane(k).read_lane(&mut state.scratch[k * lane_len..], n, &mut state.lanes[k]);
                taken[k] += t;
                missing[k] += m;
                gave[k] = t > 0;
            }
            let blocks: [Option<&[f32]>; MAX_DECKS] =
                std::array::from_fn(|k| gave[k].then(|| &state.scratch[k * lane_len..k * lane_len + n * lch]));
            state.mixer.process(&blocks, n, &mix, &bus.params, headphones);
            let out = &mut out[done * ch_out..(done + n) * ch_out];
            route(out, ch_out, n, &mix, &state.mixer, headphones);
            if let Some(b) = bridges[0] {
                b.push(&state.mixer.headphones[..n * 2]);
            }
            if let Some(b) = bridges[1] {
                b.push(&state.mixer.master[..n * 2]);
            }
            done += n;
        }
        for b in bridges.iter().flatten() {
            b.wrote(frames as u64, latency);
        }
        let mut any = false;
        for k in 0..lanes {
            self.lane(k).report(start_read[k], taken[k], missing[k], now, latency);
            any |= taken[k] > 0;
        }
        if any {
            if let Some(t) = self.port.engine_thread.get() {
                t.unpark();
            }
        }
    }

    /// Deck ``k``'s main lane (0: this ring).
    fn lane(&self, k: usize) -> &Ring {
        if k == 0 { self } else { &self.bus.get().expect("bus").decks[k - 1] }
    }
}

/// The frames a callback reads at once per lane (a longer callback is read in pieces).
const SCRATCH_FRAMES: usize = 2048;
/// The most channels a deck plays (stereo; a mono device's decks: one); its lane holds twice as many (its music and
/// its listening aids).
pub const DECK_CH_MAX: usize = 2;
const LANE_CH_MAX: usize = 2 * DECK_CH_MAX;
pub use crate::ipc::MAX_DECKS;
/// The buses that may go to a device of their own: the headphones, the booth.
pub const BRIDGED_BUSES: usize = 2;

/// The other decks' lanes, the mixer's settings, and the bridges to the devices of the headphones and the booth
/// (Ring::attach_bus).
pub struct Bus {
    pub decks: Vec<Arc<Ring>>, // decks 2 .. MAX_DECKS' lanes
    pub params: Arc<crate::mixer::MixParams>,
    pub bridges: [Arc<Bridge>; BRIDGED_BUSES],
}

/// The mixer's buses into ``out``'s channels as routed: the master (MAIN), the booth (SUB: the master again), the
/// headphones (CUE).
fn route<T: SizedSample + FromSample<f32>>(out: &mut [T], ch_out: usize, n: usize, mix: &crate::mixer::Mix,
                                          mixer: &Mixer, headphones: bool) {
    let [main_l, main_r, cue_l, cue_r, sub_l, sub_r] = mix.routes;
    for f in 0..n {
        let o = &mut out[f * ch_out..(f + 1) * ch_out];
        o.iter_mut().for_each(|v| *v = T::EQUILIBRIUM);
        let mut put = |channel: u8, v: f32| {
            if (channel as usize) < ch_out {
                o[channel as usize] = T::from_sample(v);
            }
        };
        let (left, right) = (mixer.master[2 * f], mixer.master[2 * f + 1]);
        if headphones {
            put(cue_l, mixer.headphones[2 * f]);
            put(cue_r, mixer.headphones[2 * f + 1]);
        }
        put(sub_l, left);
        put(sub_r, right);
        put(main_l, left);
        put(main_r, right);
    }
}

/// A lane's flush, as the callback follows it.
#[derive(Clone, Copy, Default)]
struct LaneState {
    flush_seen: u64,
    cut: u64,
    fade_left: u64,
}

pub(crate) struct CallbackState {
    pub(crate) channels: usize,
    lanes: [LaneState; MAX_DECKS],
    scratch: Vec<f32>, // every lane's frames, SCRATCH_FRAMES each
    mixer: Mixer,
    pub(crate) readers: Vec<Box<BridgeReader>>, // a Sink's: each bus's reader
    pub(crate) mix: Vec<f32>,                   // a Sink's: its buses mixed (a piece of the callback)
}

impl CallbackState {
    /// A device's (``channels``, ``rate``): the main one's mixer (its decks stereo, mono on a mono device), or a
    /// sink's readers.
    pub(crate) fn new(channels: usize, rate: f64) -> CallbackState {
        CallbackState {
            channels,
            lanes: [LaneState::default(); MAX_DECKS],
            scratch: vec![0.0; MAX_DECKS * SCRATCH_FRAMES * LANE_CH_MAX],
            mixer: Mixer::new(rate, channels.clamp(1, DECK_CH_MAX), SCRATCH_FRAMES),
            readers: crate::bridge::readers(),
            mix: vec![0.0; SCRATCH_FRAMES * channels.max(1)],
        }
    }
}

/// What a device's output reads: the main ring (the decks' lanes, mixed) or a sink (buses of the main mix through
/// their bridges, followed at this device's clock).
#[derive(Clone)]
pub enum Source {
    Main(Arc<Ring>),
    Sink(Arc<Sink>),
}

impl Source {
    pub fn port(&self) -> &Port {
        match self {
            Source::Main(r) => &r.port,
            Source::Sink(s) => &s.port,
        }
    }

    pub(crate) fn fill<T: SizedSample + FromSample<f32>>(&self, out: &mut [T], state: &mut CallbackState, latency: f64) {
        match self {
            Source::Main(r) => r.fill(out, state, latency),
            Source::Sink(s) => s.fill(out, state, latency),
        }
    }
}

/// The audio API the output goes through (Settings > Playback's Audio API, native engine).
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Api {
    /// Windows: the engine's own WASAPI output (wasapi.rs): the low-latency shared stream, or exclusive.
    Wasapi,
    /// cpal's output for the system: WASAPI's usual shared stream (Windows), ALSA (Linux), CoreAudio (macOS).
    Cpal,
    /// Linux: a JACK client (JACK's or PipeWire's server).
    Jack,
}

/// How the output reaches the device: the API and, for WASAPI, whether the device is taken exclusively.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Mode {
    pub api: Api,
    pub exclusive: bool,
}

impl Mode {
    /// From the codes the GUI sends (api: 0 WASAPI, 1 cpal, 2 JACK); an API this system has not is cpal's.
    pub fn from_codes(api: u32, exclusive: u32) -> Mode {
        let api = match api {
            0 if cfg!(windows) => Api::Wasapi,
            2 if cfg!(target_os = "linux") => Api::Jack,
            _ => Api::Cpal,
        };
        Mode { api, exclusive: exclusive != 0 && api == Api::Wasapi }
    }

    /// The code published: the API's, and 0x100 when exclusive.
    pub fn code(self) -> u32 {
        let api = match self.api {
            Api::Wasapi => 0,
            Api::Cpal => 1,
            Api::Jack => 2,
        };
        api | ((self.exclusive as u32) << 8)
    }
}

/// What keeps a device's stream: dropped, the stream ends and the device is let go.
#[allow(dead_code)] // held, not read
enum Backend {
    Cpal(cpal::Stream),
    #[cfg(windows)]
    Wasapi(crate::wasapi::WasapiOutput),
}

/// An open device, its stream running from its opening to its end.
pub struct Output {
    _backend: Backend,
    pub mode: Mode,      // the one in use (one that could not be opened falls back to cpal's)
    pub period_ms: f64,  // the device's period, 0 = not known
}

/// cpal's host for ``mode`` (JACK's on Linux, else the system's).
fn host(mode: Mode) -> cpal::Host {
    #[cfg(target_os = "linux")]
    if mode.api == Api::Jack {
        if let Ok(h) = cpal::host_from_id(cpal::HostId::Jack) {
            return h;
        }
    }
    let _ = mode;
    cpal::default_host()
}

/// A device's name as listed (its description's).
pub fn device_name(dev: &cpal::Device) -> String {
    dev.description().map(|d| d.name().to_string()).unwrap_or_default()
}

/// The output devices of ``mode``'s host: (name, channels) each, and the system's default's.
pub fn list(mode: Mode) -> (Option<(String, usize)>, Vec<(String, usize)>) {
    let host = host(mode);
    let channels = |d: &cpal::Device| d.default_output_config().map(|c| c.channels() as usize).unwrap_or(2);
    let default = host.default_output_device().map(|d| (device_name(&d), channels(&d)));
    let devices = match host.output_devices() {
        Ok(devices) => devices.map(|d| (device_name(&d), channels(&d))).filter(|(n, _)| !n.is_empty()).collect(),
        Err(_) => Vec::new(),
    };
    (default, devices)
}

/// The output device named ``name`` (None or "": the default; one not found: the default too).
fn find(name: Option<&str>, mode: Mode) -> Result<cpal::Device, String> {
    let host = host(mode);
    if let Some(wanted) = name.filter(|n| !n.is_empty()) {
        if let Ok(mut devices) = host.output_devices() {
            if let Some(dev) = devices.find(|d| device_name(d) == wanted) {
                return Ok(dev);
            }
        }
    }
    host.default_output_device().ok_or_else(|| "no output device".to_string())
}

/// The preferred format of the device named ``name`` (see find): (sample rate, channels).
pub fn default_format(name: Option<&str>, mode: Mode) -> Result<(u32, usize), String> {
    let dev = find(name, mode)?;
    let cfg = dev.default_output_config().map_err(|e| e.to_string())?;
    Ok((cfg.sample_rate(), cfg.channels() as usize))
}

/// Opens the device named ``name`` (see find) in ``mode`` at ``rate`` / ``channels`` (the session's: the track
/// was decoded for them) and starts it reading ``source``; a mode that cannot be opened falls back to Shared.
/// Returns the output and the name of the device opened.
pub fn open(source: Source, rate: u32, channels: usize, name: Option<&str>, mode: Mode)
    -> Result<(Output, String), String> {
    source.port().reset();
    #[cfg(windows)]
    if mode.api == Api::Wasapi {
        match crate::wasapi::WasapiOutput::open(source.clone(), rate, channels, name, mode.exclusive) {
            Ok((w, device, period_ms)) => {
                return Ok((Output { _backend: Backend::Wasapi(w), mode, period_ms }, device));
            }
            Err(e) => eprintln!("[AudioEngine] {mode:?} output not opened ({e}); cpal's WASAPI shared stream instead"),
        }
    }
    let cpal_mode = Mode { api: if mode.api == Api::Jack { Api::Jack } else { Api::Cpal }, exclusive: false };
    let dev = find(name, cpal_mode)?;
    let def = dev.default_output_config().map_err(|e| e.to_string())?;
    // a small device buffer where the system lets one be asked for (ALSA, CoreAudio: ~5 ms; WASAPI's shared
    // stream takes its own)
    let buffer_size = match def.buffer_size() {
        cpal::SupportedBufferSize::Range { min, max } if !cfg!(windows) && cpal_mode.api != Api::Jack => {
            cpal::BufferSize::Fixed((rate / 200).clamp(*min, *max))
        }
        _ => cpal::BufferSize::Default,
    };
    let config = cpal::StreamConfig { channels: channels as u16, sample_rate: rate, buffer_size };
    let build_any = |config: &cpal::StreamConfig| match def.sample_format() {
        cpal::SampleFormat::F32 => build::<f32>(&dev, config, source.clone()),
        cpal::SampleFormat::I16 => build::<i16>(&dev, config, source.clone()),
        cpal::SampleFormat::I32 => build::<i32>(&dev, config, source.clone()),
        cpal::SampleFormat::U16 => build::<u16>(&dev, config, source.clone()),
        other => Err(format!("unsupported sample format {other:?}")),
    };
    let stream = match build_any(&config) {
        Ok(s) => s,
        Err(_) if config.buffer_size != cpal::BufferSize::Default => {
            build_any(&cpal::StreamConfig { buffer_size: cpal::BufferSize::Default, ..config })?
        }
        Err(e) => return Err(e),
    };
    stream.play().map_err(|e| e.to_string())?;
    let period_ms = match config.buffer_size {
        cpal::BufferSize::Fixed(f) => f as f64 * 1000.0 / rate as f64,
        cpal::BufferSize::Default => 0.0,
    };
    Ok((Output { _backend: Backend::Cpal(stream), mode: cpal_mode, period_ms }, device_name(&dev)))
}

fn build<T: SizedSample + FromSample<f32> + 'static>(
    dev: &cpal::Device,
    config: &cpal::StreamConfig,
    source: Source,
) -> Result<cpal::Stream, String> {
    let mut state = CallbackState::new(config.channels as usize, config.sample_rate as f64);
    let err_source = source.clone();
    dev.build_output_stream(
        *config,
        move |data: &mut [T], info: &cpal::OutputCallbackInfo| {
            let ts = info.timestamp();
            let latency = ts.playback.duration_since(ts.callback).as_secs_f64();
            source.fill(data, &mut state, latency);
        },
        move |err| {
            eprintln!("[AudioEngine] device error: {err}");
            err_source.port().failed.store(true, Ordering::Relaxed);
        },
        None,
    )
    .map_err(|e| e.to_string())
}
