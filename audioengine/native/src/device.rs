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
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

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
    pub underruns: AtomicU64,
    pub failed: AtomicBool,
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
            underruns: AtomicU64::new(0),
            failed: AtomicBool::new(false),
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
    fn fill<T: SizedSample + FromSample<f32>>(&self, out: &mut [T], state: &mut CallbackState, latency: f64) {
        let ch_out = state.channels;
        let frames = out.len() / ch_out;
        let generation = self.flush_gen.load(Ordering::Acquire);
        if generation != state.flush_seen {
            state.flush_seen = generation;
            state.cut = self.flush_cut.load(Ordering::Relaxed);
            state.fade_left = self.fade;
        }
        let mut read = self.read.load(Ordering::Relaxed);
        let write = self.write.load(Ordering::Acquire);
        // the report first: frame `read` is heard after the device's latency
        let now = clock::now();
        let s = self.cb_seq.load(Ordering::Relaxed);
        self.cb_seq.store(s + 1, Ordering::Relaxed);
        std::sync::atomic::fence(Ordering::Release);
        self.cb_read.store(read, Ordering::Relaxed);
        self.cb_at.store((now + latency).to_bits(), Ordering::Relaxed);
        self.cb_frames.store(frames as u64, Ordering::Relaxed);
        self.latency.store(latency.to_bits(), Ordering::Relaxed);
        self.cb_seq.store(s + 2, Ordering::Release);

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
        if missing > 0 {
            self.underruns.fetch_add(missing, Ordering::Relaxed);
        }
    }
}

struct CallbackState {
    channels: usize,
    flush_seen: u64,
    cut: u64,
    fade_left: u64,
}

pub struct Output {
    pub stream: cpal::Stream,
    pub running: bool,
}

/// A device's name as listed (its description's).
pub fn device_name(dev: &cpal::Device) -> String {
    dev.description().map(|d| d.name().to_string()).unwrap_or_default()
}

/// The output devices' names.
pub fn list() -> Vec<String> {
    let host = cpal::default_host();
    match host.output_devices() {
        Ok(devices) => devices.map(|d| device_name(&d)).filter(|n| !n.is_empty()).collect(),
        Err(_) => Vec::new(),
    }
}

/// The output device named ``name`` (None or "": the system's default; one not found: the default too), and
/// whether it is the one asked for.
fn find(name: Option<&str>) -> Result<(cpal::Device, bool), String> {
    let host = cpal::default_host();
    if let Some(wanted) = name.filter(|n| !n.is_empty()) {
        if let Ok(mut devices) = host.output_devices() {
            if let Some(dev) = devices.find(|d| device_name(d) == wanted) {
                return Ok((dev, true));
            }
        }
        let dev = host.default_output_device().ok_or("no output device")?;
        return Ok((dev, false));
    }
    let dev = host.default_output_device().ok_or("no output device")?;
    Ok((dev, true))
}

/// The preferred format of the device named ``name`` (see find): (sample rate, channels).
pub fn default_format(name: Option<&str>) -> Result<(u32, usize), String> {
    let (dev, _) = find(name)?;
    let cfg = dev.default_output_config().map_err(|e| e.to_string())?;
    Ok((cfg.sample_rate(), cfg.channels() as usize))
}

/// Opens the device named ``name`` (see find) at ``rate`` / ``channels`` (the session's: the track was
/// decoded for them; a device of another format converts, as WASAPI's shared mode does) and starts it reading
/// ``ring``. Returns the stream and the name of the device opened.
pub fn open(ring: Arc<Ring>, rate: u32, channels: usize, name: Option<&str>) -> Result<(Output, String), String> {
    let (dev, _) = find(name)?;
    let def = dev.default_output_config().map_err(|e| e.to_string())?;
    let config = cpal::StreamConfig {
        channels: channels as u16,
        sample_rate: rate,
        buffer_size: cpal::BufferSize::Default,
    };
    let stream = match def.sample_format() {
        cpal::SampleFormat::F32 => build::<f32>(&dev, &config, ring),
        cpal::SampleFormat::I16 => build::<i16>(&dev, &config, ring),
        cpal::SampleFormat::I32 => build::<i32>(&dev, &config, ring),
        cpal::SampleFormat::U16 => build::<u16>(&dev, &config, ring),
        other => return Err(format!("unsupported sample format {other:?}")),
    }?;
    stream.play().map_err(|e| e.to_string())?;
    Ok((Output { stream, running: true }, device_name(&dev)))
}

fn build<T: SizedSample + FromSample<f32> + 'static>(
    dev: &cpal::Device,
    config: &cpal::StreamConfig,
    ring: Arc<Ring>,
) -> Result<cpal::Stream, String> {
    let mut state = CallbackState { channels: config.channels as usize, flush_seen: 0, cut: 0, fade_left: 0 };
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
