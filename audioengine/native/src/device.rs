//! The output device (cpal: WASAPI, ALSA / PulseAudio / JACK, CoreAudio) and the ring the engine renders
//! into ahead of it.
//!
//! The device's callback only copies from the ring: rendering (the time stretch's tens of ms for a new
//! stream) happens on the engine's thread, which keeps the ring filled to the output buffer. Frames are
//! numbered by their ring index (counted since the engine started, never wrapping); the callback reports the
//! index it starts at and when that frame is heard (its time + the device's latency), so the engine knows the
//! frame being heard at any moment (`heard`). A flush (pause, a grab of the record) drops what the ring
//! holds from the read point on, faded out over a few ms in the callback itself (no click; the device's own
//! buffer, ~10 ms, still plays out).

use std::cell::UnsafeCell;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SizedSample};

use crate::clock;

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
    // the callback's report, under a sequence lock
    cb_seq: AtomicU64,
    cb_read: AtomicU64,
    cb_at: AtomicU64, // f64 bits: clock::now() when cb_read is heard
    cb_frames: AtomicU64,
    latency: AtomicU64, // f64 bits, s
    pub sound_due: AtomicBool, // underruns count only while sound is expected
    pub max_frames: AtomicU64, // the most frames the device took at once since it was opened (its period or so)
    pub underruns: AtomicU64,
    // the output thread woken late (WASAPI: more than 1.8 periods after the last wake-up: the device may have
    // played a gap although the ring held audio), and the longest gap between wake-ups (µs)
    pub late: AtomicU64,
    pub max_gap_us: AtomicU64,
    pub failed: AtomicBool,
    // the engine thread, woken after each read of the device (it tops the ring up at once, not at its next tick)
    pub engine_thread: std::sync::OnceLock<std::thread::Thread>,
    // the output thread's scheduling (sched.rs): its state where its backend set it (the engine's own WASAPI
    // thread: MMCSS), else the engine's; Linux: its thread id (written by the callback) and the timer slack it
    // should take (it can set only its own: ns + 1, 0 = not asked)
    pub output_state: AtomicU32,
    pub output_tid: AtomicU64,
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    pub output_slack: AtomicU64,
}

unsafe impl Sync for Ring {}
unsafe impl Send for Ring {}

impl Ring {
    pub fn new(cap_frames: u64, ch: usize, rate: f64) -> Ring {
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
            cb_seq: AtomicU64::new(0),
            cb_read: AtomicU64::new(0),
            cb_at: AtomicU64::new(0f64.to_bits()),
            cb_frames: AtomicU64::new(0),
            latency: AtomicU64::new(0f64.to_bits()),
            sound_due: AtomicBool::new(false),
            max_frames: AtomicU64::new(0),
            underruns: AtomicU64::new(0),
            late: AtomicU64::new(0),
            max_gap_us: AtomicU64::new(0),
            failed: AtomicBool::new(false),
            engine_thread: std::sync::OnceLock::new(),
            output_state: AtomicU32::new(0),
            output_tid: AtomicU64::new(0),
            output_slack: AtomicU64::new(0),
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
        f64::from_bits(self.latency.load(Ordering::Relaxed)) * self.rate
    }

    /// The frame being heard at clock time ``now`` (fractional).
    pub fn heard(&self, now: f64) -> f64 {
        let (r0, at, n0) = loop {
            let s1 = self.cb_seq.load(Ordering::Acquire);
            if s1 & 1 == 1 {
                std::hint::spin_loop();
                continue;
            }
            let r0 = self.cb_read.load(Ordering::Relaxed);
            let at = f64::from_bits(self.cb_at.load(Ordering::Relaxed));
            let n0 = self.cb_frames.load(Ordering::Relaxed);
            std::sync::atomic::fence(Ordering::Acquire);
            if self.cb_seq.load(Ordering::Relaxed) == s1 {
                break (r0, at, n0);
            }
        };
        if at == 0.0 {
            return r0 as f64;
        }
        let elapsed = (now - at) * self.rate;
        (r0 as f64 + elapsed.clamp(-self.latency_frames() - n0 as f64, n0 as f64)).max(0.0)
    }

    // Callback side
    pub(crate) fn fill<T: SizedSample + FromSample<f32>>(&self, out: &mut [T], state: &mut CallbackState, latency: f64) {
        #[cfg(target_os = "linux")]
        crate::sched::output_thread(self);
        let ch_out = state.channels;
        let frames = out.len() / ch_out;
        let generation = self.flush_gen.load(Ordering::Acquire);
        if generation != state.flush_seen {
            state.flush_seen = generation;
            state.cut = self.flush_cut.load(Ordering::Relaxed);
            state.fade_left = self.fade;
        }
        let mut read = self.read.load(Ordering::Relaxed);
        let start_read = read;
        let write = self.write.load(Ordering::Acquire);
        let now = clock::now();
        let mut taken = 0u64; // frames of the ring played (the rest silence: no track frame is heard there)
        self.max_frames.fetch_max(frames as u64, Ordering::Relaxed);
        let due = self.sound_due.load(Ordering::Relaxed);
        let mut missing = 0u64;
        for f in 0..frames {
            let o = &mut out[f * ch_out..(f + 1) * ch_out];
            if state.fade_left > 0 && read >= state.cut {
                state.fade_left = 0; // the old audio ran out: the new goes on as it is
            }
            if read < write {
                let at = (read % self.cap) as usize * self.ch;
                let gain = if state.fade_left > 0 {
                    let g = state.fade_left as f32 / self.fade as f32;
                    state.fade_left -= 1;
                    if state.fade_left == 0 {
                        read = read.max(state.cut).saturating_sub(1); // +1 below
                    }
                    g
                } else {
                    1.0
                };
                for c in 0..ch_out {
                    let v = if c < self.ch { unsafe { *self.buf[at + c].get() } } else { 0.0 };
                    o[c] = T::from_sample(v * gain);
                }
                read += 1;
                taken += 1;
            } else {
                if state.fade_left > 0 {
                    state.fade_left = 0;
                    read = read.max(state.cut);
                }
                for v in o.iter_mut() {
                    *v = T::EQUILIBRIUM;
                }
                if due {
                    missing += 1;
                }
            }
        }
        self.read.store(read, Ordering::Release);
        if taken > 0 {
            if let Some(t) = self.engine_thread.get() {
                t.unpark();
            }
        }
        if missing > 0 {
            self.underruns.fetch_add(missing, Ordering::Relaxed);
        }
        // the report: frame `start_read` is heard after the device's latency, and the frames after it up to those
        // taken. A callback of silence alone (the ring empty) leaves the last report: the frames it took go on
        // being heard after it, up to the last one (reported again, the frame heard would stay behind it)
        if taken == 0 && f64::from_bits(self.cb_at.load(Ordering::Relaxed)) != 0.0 {
            self.latency.store(latency.to_bits(), Ordering::Relaxed);
            return;
        }
        let s = self.cb_seq.load(Ordering::Relaxed);
        self.cb_seq.store(s + 1, Ordering::Relaxed);
        std::sync::atomic::fence(Ordering::Release);
        self.cb_read.store(start_read, Ordering::Relaxed);
        self.cb_at.store((now + latency).to_bits(), Ordering::Relaxed);
        self.cb_frames.store(taken, Ordering::Relaxed);
        self.latency.store(latency.to_bits(), Ordering::Relaxed);
        self.cb_seq.store(s + 2, Ordering::Release);
    }
}

pub(crate) struct CallbackState {
    channels: usize,
    flush_seen: u64,
    cut: u64,
    fade_left: u64,
}

impl CallbackState {
    pub(crate) fn new(channels: usize) -> CallbackState {
        CallbackState { channels, flush_seen: 0, cut: 0, fade_left: 0 }
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

enum Backend {
    Cpal(cpal::Stream),
    #[cfg(windows)]
    Wasapi(crate::wasapi::WasapiOutput),
}

pub struct Output {
    backend: Backend,
    pub running: bool,
    pub mode: Mode,      // the one in use (one that could not be opened falls back to cpal's)
    pub period_ms: f64,  // the device's period, 0 = not known
}

impl Output {
    pub fn play(&mut self) {
        match &self.backend {
            Backend::Cpal(s) => {
                let _ = s.play();
            }
            #[cfg(windows)]
            Backend::Wasapi(w) => w.play(),
        }
        self.running = true;
    }

    pub fn pause(&mut self) {
        match &self.backend {
            Backend::Cpal(s) => {
                let _ = s.pause();
            }
            #[cfg(windows)]
            Backend::Wasapi(w) => w.pause(),
        }
        self.running = false;
    }
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

/// The output devices' names (of ``mode``'s host).
pub fn list(mode: Mode) -> Vec<String> {
    match host(mode).output_devices() {
        Ok(devices) => devices.map(|d| device_name(&d)).filter(|n| !n.is_empty()).collect(),
        Err(_) => Vec::new(),
    }
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
/// was decoded for them) and starts it reading ``ring``; a mode that cannot be opened falls back to Shared.
/// Returns the output and the name of the device opened.
pub fn open(ring: Arc<Ring>, rate: u32, channels: usize, name: Option<&str>, mode: Mode)
    -> Result<(Output, String), String> {
    ring.max_frames.store(0, Ordering::Relaxed);
    ring.output_state.store(0, Ordering::Relaxed);
    ring.output_tid.store(0, Ordering::Relaxed);
    #[cfg(windows)]
    if mode.api == Api::Wasapi {
        match crate::wasapi::WasapiOutput::open(ring.clone(), rate, channels, name, mode.exclusive) {
            Ok((w, device, period_ms)) => {
                return Ok((Output { backend: Backend::Wasapi(w), running: true, mode, period_ms }, device));
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
        cpal::SampleFormat::F32 => build::<f32>(&dev, config, ring.clone()),
        cpal::SampleFormat::I16 => build::<i16>(&dev, config, ring.clone()),
        cpal::SampleFormat::I32 => build::<i32>(&dev, config, ring.clone()),
        cpal::SampleFormat::U16 => build::<u16>(&dev, config, ring.clone()),
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
    Ok((Output { backend: Backend::Cpal(stream), running: true, mode: cpal_mode, period_ms }, device_name(&dev)))
}

fn build<T: SizedSample + FromSample<f32> + 'static>(
    dev: &cpal::Device,
    config: &cpal::StreamConfig,
    ring: Arc<Ring>,
) -> Result<cpal::Stream, String> {
    let mut state = CallbackState::new(config.channels as usize);
    let err_ring = ring.clone();
    dev.build_output_stream(
        *config,
        move |data: &mut [T], info: &cpal::OutputCallbackInfo| {
            let ts = info.timestamp();
            let latency = ts.playback.duration_since(ts.callback).as_secs_f64();
            ring.fill(data, &mut state, latency);
        },
        move |err| {
            eprintln!("[AudioEngine] device error: {err}");
            err_ring.failed.store(true, Ordering::Relaxed);
        },
        None,
    )
    .map_err(|e| e.to_string())
}
