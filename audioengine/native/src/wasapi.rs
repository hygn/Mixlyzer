//! The WASAPI output of the engine's own (Windows): the two low-latency paths cpal does not open.
//!
//! Shared: IAudioClient3's shared stream at the smallest period the driver allows (often 2.7-5 ms instead of
//! 10 ms), at the endpoint's mix format (which IAudioClient3 needs: no conversion on the way); other apps keep
//! playing. Exclusive: the device to this app alone, at the session's rate and channels in a format the device
//! takes (float32, int32, int16), its smallest period, the engine's audio straight to the driver; other apps
//! cannot play on it meanwhile. Both event driven, on a thread of their own registered with MMCSS as Pro Audio,
//! reading what the engine gives the device (device.rs Source: the main ring, or a bridge) as cpal's callback does.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::thread::JoinHandle;

use windows::Win32::Devices::Properties;
use windows::Win32::Foundation::{HANDLE, PROPERTYKEY, WAIT_OBJECT_0};
use windows::Win32::Media::Audio::{
    AUDCLNT_E_BUFFER_SIZE_NOT_ALIGNED, AUDCLNT_SHAREMODE_EXCLUSIVE, AUDCLNT_STREAMFLAGS_EVENTCALLBACK,
    DEVICE_STATE_ACTIVE, IAudioClient, IAudioClient3, IAudioRenderClient, IMMDevice, IMMDeviceEnumerator,
    MMDeviceEnumerator, WAVE_FORMAT_PCM, WAVEFORMATEX, WAVEFORMATEXTENSIBLE, WAVEFORMATEXTENSIBLE_0, eConsole,
    eRender,
};
use windows::Win32::Media::KernelStreaming::{
    KSAUDIO_SPEAKER_DIRECTOUT, KSDATAFORMAT_SUBTYPE_PCM, WAVE_FORMAT_EXTENSIBLE,
};
use windows::Win32::Media::Multimedia::KSDATAFORMAT_SUBTYPE_IEEE_FLOAT;
use windows::Win32::System::Com::StructuredStorage::PropVariantClear;
use windows::Win32::System::Com::{
    CLSCTX_ALL, COINIT_MULTITHREADED, CoCreateInstance, CoInitializeEx, CoTaskMemFree, STGM_READ,
};
use windows::Win32::System::Threading::{
    AvSetMmThreadCharacteristicsW, CreateEventW, WaitForSingleObject,
};
use windows::Win32::System::Variant::VT_LPWSTR;
use windows::core::w;

use crate::device::{CallbackState, Source};

const SPEAKER_STEREO: u32 = 0x3; // front left | front right

#[derive(Clone, Copy)]
enum Sample {
    F32,
    I32,
    I16,
}

/// A running WASAPI output; dropped, its thread stops and the device is let go.
pub struct WasapiOutput {
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl WasapiOutput {
    /// Opens the device named ``name`` (None or "": the default; one not found: the default too), ``exclusive``
    /// or shared, at ``rate`` / ``channels``; returns it, the device's name and its period (ms).
    pub fn open(source: Source, rate: u32, channels: usize, name: Option<&str>, exclusive: bool)
        -> Result<(WasapiOutput, String, f64), String> {
        let stop = Arc::new(AtomicBool::new(false));
        let (tx, rx) = mpsc::channel();
        let wanted = name.map(str::to_string);
        let s = stop.clone();
        let thread = std::thread::Builder::new()
            .name("WasapiOutput".into())
            .spawn(move || render_thread(source, rate, channels, wanted, exclusive, s, tx))
            .map_err(|e| e.to_string())?;
        match rx.recv() {
            Ok(Ok((device, period_ms))) => Ok((WasapiOutput { stop, thread: Some(thread) }, device, period_ms)),
            Ok(Err(e)) => {
                let _ = thread.join();
                Err(e)
            }
            Err(_) => {
                let _ = thread.join();
                Err("the output thread ended".into())
            }
        }
    }
}

impl Drop for WasapiOutput {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

unsafe fn friendly_name(device: &IMMDevice) -> Option<String> {
    unsafe {
        let store = device.OpenPropertyStore(STGM_READ).ok()?;
        let key = &Properties::DEVPKEY_Device_FriendlyName as *const _ as *const PROPERTYKEY;
        let mut value = store.GetValue(key).ok()?;
        let inner = &value.Anonymous.Anonymous;
        let name = if inner.vt == VT_LPWSTR {
            let p = *(&inner.Anonymous as *const _ as *const *const u16);
            let mut len = 0;
            while len < 32768 && *p.add(len) != 0 {
                len += 1;
            }
            Some(String::from_utf16_lossy(std::slice::from_raw_parts(p, len)))
        } else {
            None
        };
        let _ = PropVariantClear(&mut value);
        name
    }
}

unsafe fn find_device(enumerator: &IMMDeviceEnumerator, name: Option<&str>) -> Result<IMMDevice, String> {
    unsafe {
        if let Some(wanted) = name.filter(|n| !n.is_empty()) {
            if let Ok(collection) = enumerator.EnumAudioEndpoints(eRender, DEVICE_STATE_ACTIVE) {
                for i in 0..collection.GetCount().unwrap_or(0) {
                    if let Ok(d) = collection.Item(i) {
                        if friendly_name(&d).as_deref() == Some(wanted) {
                            return Ok(d);
                        }
                    }
                }
            }
        }
        enumerator.GetDefaultAudioEndpoint(eRender, eConsole).map_err(|e| format!("no output device: {e}"))
    }
}

/// A WAVEFORMATEXTENSIBLE of ``sample`` at ``rate`` / ``channels``.
fn extensible(sample: Sample, valid_bits: u16, rate: u32, channels: usize) -> WAVEFORMATEXTENSIBLE {
    let bytes: u16 = match sample {
        Sample::F32 | Sample::I32 => 4,
        Sample::I16 => 2,
    };
    let ch = channels as u16;
    WAVEFORMATEXTENSIBLE {
        Format: WAVEFORMATEX {
            wFormatTag: WAVE_FORMAT_EXTENSIBLE as u16,
            nChannels: ch,
            nSamplesPerSec: rate,
            nAvgBytesPerSec: rate * (ch * bytes) as u32,
            nBlockAlign: ch * bytes,
            wBitsPerSample: bytes * 8,
            cbSize: (std::mem::size_of::<WAVEFORMATEXTENSIBLE>() - std::mem::size_of::<WAVEFORMATEX>()) as u16,
        },
        Samples: WAVEFORMATEXTENSIBLE_0 { wValidBitsPerSample: valid_bits },
        dwChannelMask: if channels == 2 { SPEAKER_STEREO } else { KSAUDIO_SPEAKER_DIRECTOUT },
        SubFormat: if matches!(sample, Sample::F32) { KSDATAFORMAT_SUBTYPE_IEEE_FLOAT } else { KSDATAFORMAT_SUBTYPE_PCM },
    }
}

/// The sample type a format holds (the mix format: float32 or PCM).
unsafe fn sample_of(format: *const WAVEFORMATEX) -> Option<Sample> {
    unsafe {
        let f = &*format;
        let float = if f.wFormatTag == WAVE_FORMAT_EXTENSIBLE as u16 {
            let ex = &*(format as *const WAVEFORMATEXTENSIBLE);
            let sub = ex.SubFormat;
            sub == KSDATAFORMAT_SUBTYPE_IEEE_FLOAT
        } else {
            f.wFormatTag == 3 // WAVE_FORMAT_IEEE_FLOAT
        };
        match (float, f.wBitsPerSample) {
            (true, 32) => Some(Sample::F32),
            (false, 32) => Some(Sample::I32),
            (false, 16) if f.wFormatTag == WAVE_FORMAT_PCM as u16 || f.wFormatTag == WAVE_FORMAT_EXTENSIBLE as u16 => {
                Some(Sample::I16)
            }
            _ => None,
        }
    }
}

struct Opened {
    client: IAudioClient,
    render: IAudioRenderClient,
    event: HANDLE,
    buffer: u32,
    sample: Sample,
    period_ms: f64,
}

/// The exclusive stream: the session's format in a sample type the device takes, at its smallest period.
unsafe fn open_exclusive(device: &IMMDevice, rate: u32, channels: usize) -> Result<Opened, String> {
    unsafe {
        let client: IAudioClient = device.Activate(CLSCTX_ALL, None).map_err(|e| e.to_string())?;
        let candidates = [(Sample::F32, 32), (Sample::I32, 32), (Sample::I32, 24), (Sample::I16, 16)];
        let mut chosen = None;
        for (sample, bits) in candidates {
            let f = extensible(sample, bits, rate, channels);
            if client.IsFormatSupported(AUDCLNT_SHAREMODE_EXCLUSIVE, &f.Format, None).is_ok() {
                chosen = Some((sample, f));
                break;
            }
        }
        let Some((sample, format)) = chosen else {
            return Err(format!("the device takes no {rate} Hz / {channels} ch format in exclusive mode"));
        };
        let (mut default_period, mut min_period) = (0i64, 0i64);
        client.GetDevicePeriod(Some(&mut default_period), Some(&mut min_period)).map_err(|e| e.to_string())?;
        let mut period = min_period.max(1);
        let mut client = client;
        let flags = AUDCLNT_STREAMFLAGS_EVENTCALLBACK;
        if let Err(e) = client.Initialize(AUDCLNT_SHAREMODE_EXCLUSIVE, flags, period, period, &format.Format, None) {
            if e.code() != AUDCLNT_E_BUFFER_SIZE_NOT_ALIGNED {
                return Err(format!("exclusive mode refused: {e}"));
            }
            // the period the device's buffer alignment allows, with a new client
            let frames = client.GetBufferSize().map_err(|e| e.to_string())?;
            period = (10_000_000.0 * frames as f64 / rate as f64 + 0.5) as i64;
            client = device.Activate(CLSCTX_ALL, None).map_err(|e| e.to_string())?;
            client
                .Initialize(AUDCLNT_SHAREMODE_EXCLUSIVE, flags, period, period, &format.Format, None)
                .map_err(|e| format!("exclusive mode refused: {e}"))?;
        }
        finish(client, sample, period as f64 / 10_000.0)
    }
}

/// The low-latency shared stream (IAudioClient3), at the mix format: only when that is the session's format.
unsafe fn open_shared(device: &IMMDevice, rate: u32, channels: usize) -> Result<Opened, String> {
    unsafe {
        let client: IAudioClient3 = device.Activate(CLSCTX_ALL, None).map_err(|e| e.to_string())?;
        let mix = client.GetMixFormat().map_err(|e| e.to_string())?;
        let result = (|| {
            let (mix_rate, mix_channels) = ({ (*mix).nSamplesPerSec }, { (*mix).nChannels });
            if mix_rate != rate || mix_channels as usize != channels {
                return Err(format!(
                    "the mix format ({mix_rate} Hz / {mix_channels} ch) is not the one asked for ({rate} Hz / {channels} ch)"
                ));
            }
            let sample = sample_of(mix).ok_or("unsupported mix format")?;
            let (mut default, mut fundamental, mut min, mut max) = (0u32, 0u32, 0u32, 0u32);
            client
                .GetSharedModeEnginePeriod(mix, &mut default, &mut fundamental, &mut min, &mut max)
                .map_err(|e| e.to_string())?;
            client
                .InitializeSharedAudioStream(AUDCLNT_STREAMFLAGS_EVENTCALLBACK, min, mix, None)
                .map_err(|e| format!("low-latency shared stream refused: {e}"))?;
            let base: IAudioClient = windows::core::Interface::cast(&client).map_err(|e| e.to_string())?;
            finish(base, sample, min as f64 * 1000.0 / rate as f64)
        })();
        CoTaskMemFree(Some(mix as *const _));
        result
    }
}

unsafe fn finish(client: IAudioClient, sample: Sample, period_ms: f64) -> Result<Opened, String> {
    unsafe {
        let event = CreateEventW(None, false, false, None).map_err(|e| e.to_string())?;
        client.SetEventHandle(event).map_err(|e| e.to_string())?;
        let buffer = client.GetBufferSize().map_err(|e| e.to_string())?;
        let render: IAudioRenderClient = client.GetService().map_err(|e| e.to_string())?;
        Ok(Opened { client, render, event, buffer, sample, period_ms })
    }
}

#[allow(clippy::too_many_arguments)]
fn render_thread(
    source: Source,
    rate: u32,
    channels: usize,
    name: Option<String>,
    exclusive: bool,
    stop: Arc<AtomicBool>,
    ready: mpsc::Sender<Result<(String, f64), String>>,
) {
    unsafe {
        let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
        let mut task = 0u32;
        let mmcss = AvSetMmThreadCharacteristicsW(w!("Pro Audio"), &mut task).is_ok();
        let port = source.port();
        port.output_state.store(if mmcss { crate::sched::REALTIME } else { crate::sched::NORMAL },
            std::sync::atomic::Ordering::Relaxed);
        let opened = (|| {
            let enumerator: IMMDeviceEnumerator =
                CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL).map_err(|e| e.to_string())?;
            let device = find_device(&enumerator, name.as_deref())?;
            let device_name = friendly_name(&device).unwrap_or_default();
            let o = if exclusive { open_exclusive(&device, rate, channels)? } else { open_shared(&device, rate, channels)? };
            Ok::<_, String>((o, device_name))
        })();
        let (o, device_name) = match opened {
            Ok(v) => v,
            Err(e) => {
                let _ = ready.send(Err(e));
                return;
            }
        };
        let _ = ready.send(Ok((device_name, o.period_ms)));
        let mut state = CallbackState::new(channels, rate as f64);
        let latency_extra = o.client.GetStreamLatency().unwrap_or(0) as f64 / 1e7;
        let period_s = o.period_ms / 1000.0;
        let mut last_wake = 0.0f64;
        // the buffer filled (with silence) before it starts: an exclusive stream's events may not come otherwise
        // (a DJ controller's USB driver: one in 200 ms)
        let free = if exclusive { Ok(o.buffer) } else { o.client.GetCurrentPadding().map(|p| o.buffer - p) };
        if let Ok(frames) = free {
            if frames > 0 && o.render.GetBuffer(frames).is_ok() {
                let _ = o.render.ReleaseBuffer(frames, 2); // AUDCLNT_BUFFERFLAGS_SILENT
            }
        }
        if o.client.Start().is_err() {
            port.failed.store(true, Ordering::Relaxed);
        }
        while !stop.load(Ordering::Relaxed) && !port.failed.load(Ordering::Relaxed) {
            if WaitForSingleObject(o.event, 200) != WAIT_OBJECT_0 {
                continue;
            }
            let woke = crate::clock::now();
            if last_wake > 0.0 && port.sound_due.load(Ordering::Relaxed) {
                let gap = woke - last_wake;
                port.max_gap_us.fetch_max((gap * 1e6) as u64, Ordering::Relaxed);
                if gap > 1.8 * period_s {
                    port.late.fetch_add(1, Ordering::Relaxed);
                }
            }
            last_wake = woke;
            // exclusive: the whole buffer each period (the other half plays); shared: what is free
            let (frames, queued) = if exclusive {
                (o.buffer, o.buffer)
            } else {
                match o.client.GetCurrentPadding() {
                    Ok(p) => (o.buffer - p, p),
                    Err(_) => {
                        port.failed.store(true, Ordering::Relaxed);
                        break;
                    }
                }
            };
            if frames == 0 {
                continue;
            }
            let Ok(ptr) = o.render.GetBuffer(frames) else {
                port.failed.store(true, Ordering::Relaxed);
                break;
            };
            let n = frames as usize * channels;
            let latency = queued as f64 / rate as f64 + latency_extra;
            match o.sample {
                Sample::F32 => source.fill(std::slice::from_raw_parts_mut(ptr as *mut f32, n), &mut state, latency),
                Sample::I32 => source.fill(std::slice::from_raw_parts_mut(ptr as *mut i32, n), &mut state, latency),
                Sample::I16 => source.fill(std::slice::from_raw_parts_mut(ptr as *mut i16, n), &mut state, latency),
            }
            if o.render.ReleaseBuffer(frames, 0).is_err() {
                port.failed.store(true, Ordering::Relaxed);
                break;
            }
        }
        let _ = o.client.Stop();
        let _ = windows::Win32::Foundation::CloseHandle(o.event);
    }
}
