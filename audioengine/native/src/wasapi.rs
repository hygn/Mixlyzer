//! The WASAPI output of the engine's own (Windows): the two low-latency paths cpal does not open.
//!
//! Shared: IAudioClient3's shared stream at the smallest period the driver allows (often 2.7-5 ms instead of
//! 10 ms), at the endpoint's mix format (which IAudioClient3 needs: no conversion on the way); other apps keep
//! playing. Exclusive: the device to this app alone, at the session's rate and channels in a format the device
//! takes (float32, int32, int16), its smallest period, the engine's audio straight to the driver; other apps
//! cannot play on it meanwhile. Both event driven, on a thread of their own registered with MMCSS as Pro Audio,
//! reading what the engine gives the device (device.rs Source: the main ring, or a bridge) as cpal's callback does.

use std::sync::{Arc, Mutex};
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

/// The device period asked for is at least this many frames (5.8 ms at 44.1 kHz, as rekordbox asks the DDJ-400
/// for): below it a driver call held up now and then (the USB Audio class driver's GetBuffer: ~0.8 ms each period, at
/// times 10 ms) outlasts the period and runs the device dry. Measured on the DDJ-400, a real track, 4 x 20 s each: 132
/// frames (its smallest) 5 restarts, 192 one, 256 none.
const PERIOD_FRAMES_MIN: f64 = 256.0;

/// Every stream is watched while it runs, and started again in place (Stop, Reset, its buffer filled with silence,
/// Start: one short gap) when it is not keeping pace:
/// - its events thinned out: the last THIN_EVENTS (an exclusive stream's) came THIN_GAP periods apart on average (a
///   driver call held up longer than the period, the DDJ-400's USB driver: GetBuffer waits ~0.8 ms each period and
///   now and then 10 ms, let the device run dry, and its driver then signals every ~8 ms only, asking for 3 ms of
///   audio each time, for good);
/// - the frames it took over HEALTH_SEC (from HEALTH_SETTLE_SEC after a start) fell below HEALTH_MIN of its rate (its
///   events slower than the period without a long gap: seen on the DDJ-400 at a 10 ms period, events every 15.6 ms,
///   the device taking 64 % of its rate, its bridge's backlog growing by seconds; a window in which the output
///   thread itself woke late is left out, the next starting at that wake-up: the stream ran dry by its lateness,
///   not the device's, and a polled stream kept two periods full let a 60 ms hold count as 94 %), or no event came
///   for EVENT_TIMEOUT_MS (the stream stopped).
const THIN_GAP: f64 = 1.8;
const THIN_EVENTS: usize = 8;
const HEALTH_SEC: f64 = 1.0;
const HEALTH_SETTLE_SEC: f64 = 0.5;
const HEALTH_MIN: f64 = 0.95;
const EVENT_TIMEOUT_MS: u32 = 200;
/// More restarts than RESTARTS_MAX within RESTART_SPAN_SEC: the stream fails (its owner opens it again: the main
/// device by the engine thread, another by the outputs' thread), an exclusive one at the next of its PERIOD_LEVELS
/// (the smallest the engine allows, twice that, the device's default; after the last the first again): a device is
/// never kept at a period that does not keep pace.
const RESTARTS_MAX: usize = 3;
const RESTART_SPAN_SEC: f64 = 60.0;
const PERIOD_LEVELS: usize = 3;
/// Each device's (friendly name's) period level, kept for the engine's run.
static LEVELS: Mutex<Vec<(String, usize)>> = Mutex::new(Vec::new());

fn period_level(name: &str) -> usize {
    LEVELS.lock().ok().and_then(|l| l.iter().find(|(n, _)| n == name).map(|(_, k)| *k)).unwrap_or(0)
}

/// The device's next period level (kept); returns it.
fn step_period_level(name: &str) -> usize {
    let Ok(mut levels) = LEVELS.lock() else { return 0 };
    let next = (levels.iter().find(|(n, _)| n == name).map_or(0, |(_, k)| *k) + 1) % PERIOD_LEVELS;
    match levels.iter_mut().find(|(n, _)| n == name) {
        Some(entry) => entry.1 = next,
        None => levels.push((name.to_string(), next)),
    }
    next
}

/// Tests only (the ``fault`` feature): MIXLYZER_AUDIO_FAULT="<a device name's part>:<from s>:<to s>" holds that
/// device's output thread 1.56 periods after each event between those times from the first output opened (an
/// event coming meanwhile is taken at once, the others missed): the device taking 64 % of its rate, as the DDJ-400
/// was seen, for the watch and its recovery to be checked on hardware.
#[cfg(feature = "fault")]
fn fault_hold(device: &str, now: f64, period_s: f64) {
    static START: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    let start = *START.get_or_init(|| now);
    let Ok(spec) = std::env::var("MIXLYZER_AUDIO_FAULT") else { return };
    let mut parts = spec.split(':');
    let (Some(part), Some(from), Some(to)) = (parts.next(), parts.next(), parts.next()) else { return };
    let (from, to) = (from.parse::<f64>().unwrap_or(0.0), to.parse::<f64>().unwrap_or(0.0));
    let since = now - start;
    if device.contains(part) && since >= from && since < to {
        std::thread::sleep(std::time::Duration::from_secs_f64(1.56 * period_s));
    }
}

/// Tests only (the ``fault`` feature): a device's events recorded for a study of its driver.
/// MIXLYZER_AUDIO_PROBE="<a device name's part>": each event's wake-up, gap, GetBuffer / fill / ReleaseBuffer times,
/// frames and the device's own play position (IAudioClock) into %TEMP%/mixlyzer-probe-<pid>-<n>.csv;
/// MIXLYZER_AUDIO_STALL="<part>:<at s>:<ms>[,<at s>:<ms> ...]" holds its output thread once at each time (from the
/// first output opened: a deadline missed); MIXLYZER_AUDIO_NO_RECOVER=1 leaves a stream not keeping pace as it is
/// (the driver's own behaviour seen).
#[cfg(feature = "fault")]
mod probe {
    use std::io::Write;
    use windows::Win32::Media::Audio::{IAudioClient, IAudioClock};

    pub struct Probe {
        file: Option<std::io::BufWriter<std::fs::File>>,
        clock: Option<IAudioClock>,
        freq: f64,
        start: f64,
        stalls: Vec<(usize, f64, f64)>,
    }

    static START: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    static OPENED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    static STALLED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0); // each stall once a run

    /// MIXLYZER_AUDIO_QUERY=<device>: an event-driven stream of it asks for its padding every 1 ms between events.
    pub fn query_between(device: &str) -> bool {
        std::env::var("MIXLYZER_AUDIO_QUERY").is_ok_and(|v| !v.is_empty() && device.contains(&v))
    }

    pub fn no_recover() -> bool {
        std::env::var("MIXLYZER_AUDIO_NO_RECOVER").is_ok_and(|v| v == "1")
    }

    impl Probe {
        pub unsafe fn new(client: &IAudioClient, device: &str, now: f64) -> Probe {
            let start = *START.get_or_init(|| now);
            let wanted = |var: &str| std::env::var(var).ok().filter(|p| device.contains(p.split(':').next().unwrap_or("~")));
            let mut stalls = Vec::new();
            if let Some(spec) = std::env::var("MIXLYZER_AUDIO_STALL").ok().filter(|s| {
                device.contains(s.split(':').next().unwrap_or("~"))
            }) {
                let rest = spec.split_once(':').map(|(_, r)| r).unwrap_or("");
                for (k, item) in rest.split(',').enumerate() {
                    if let Some((at, ms)) = item.split_once(':') {
                        if let (Ok(at), Ok(ms)) = (at.parse::<f64>(), ms.parse::<f64>()) {
                            if STALLED.load(std::sync::atomic::Ordering::Relaxed) & (1 << k) == 0 {
                                stalls.push((k, at, ms));
                            }
                        }
                    }
                }
            }
            let mut probe = Probe { file: None, clock: None, freq: 1.0, start, stalls };
            if wanted("MIXLYZER_AUDIO_PROBE").is_some() {
                unsafe {
                    if let Ok(clock) = client.GetService::<IAudioClock>() {
                        probe.freq = clock.GetFrequency().unwrap_or(1) as f64;
                        probe.clock = Some(clock);
                    }
                }
                let n = OPENED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let path = std::env::temp_dir().join(format!("mixlyzer-probe-{}-{n}.csv", std::process::id()));
                if let Ok(f) = std::fs::File::create(&path) {
                    let mut w = std::io::BufWriter::new(f);
                    let _ = writeln!(w, "# {device}");
                    let _ = writeln!(w, "t,gap_ms,get_ms,fill_ms,release_ms,frames,position_s,position_qpc_s");
                    probe.file = Some(w);
                    log_line!("probe: {device} -> {}", path.display());
                }
            }
            probe
        }

        /// The stalls due by ``now`` (each once).
        pub fn stall(&mut self, now: f64) {
            let since = now - self.start;
            while let Some(i) = self.stalls.iter().position(|(_, at, _)| since >= *at) {
                let (k, _, ms) = self.stalls.remove(i);
                STALLED.fetch_or(1 << k, std::sync::atomic::Ordering::Relaxed);
                log_line!("probe: stalled {ms} ms at {since:.2} s");
                std::thread::sleep(std::time::Duration::from_secs_f64(ms / 1000.0));
            }
        }

        #[allow(clippy::too_many_arguments)]
        pub fn event(&mut self, woke: f64, gap: f64, get: f64, fill: f64, release: f64, frames: u32) {
            let Some(w) = self.file.as_mut() else { return };
            let (mut pos, mut qpc) = (0u64, 0u64);
            if let Some(c) = &self.clock {
                unsafe {
                    let _ = c.GetPosition(&mut pos, Some(&mut qpc));
                }
            }
            let _ = writeln!(w, "{:.6},{:.3},{:.3},{:.3},{:.3},{frames},{:.6},{:.6}", woke - self.start, gap * 1e3,
                             get * 1e3, fill * 1e3, release * 1e3, pos as f64 / self.freq, qpc as f64 / 1e7);
        }
    }

    impl Drop for Probe {
        fn drop(&mut self) {
            if let Some(w) = self.file.as_mut() {
                let _ = w.flush();
            }
        }
    }
}

/// The frames a stream took over a window, against its rate.
struct Health {
    since: f64, // the window's start (frames before it not counted)
    frames: f64,
}

impl Health {
    fn new(now: f64) -> Health {
        Health { since: now + HEALTH_SETTLE_SEC, frames: 0.0 }
    }

    /// A new window from ``now`` (the frames before it not counted).
    fn skip(&mut self, now: f64) {
        self.since = self.since.max(now);
        self.frames = 0.0;
    }

    fn took(&mut self, frames: u32, now: f64) {
        if now >= self.since {
            self.frames += frames as f64;
        }
    }

    /// At a window's end: the share of its rate the stream took (the next window starts), None before.
    fn check(&mut self, now: f64, rate: f64) -> Option<f64> {
        let span = now - self.since;
        if span < HEALTH_SEC {
            return None;
        }
        let share = self.frames / (span * rate);
        self.since = now;
        self.frames = 0.0;
        Some(share)
    }
}

const SPEAKER_STEREO: u32 = 0x3; // front left | front right

#[derive(Clone, Copy)]
enum Sample {
    F32,
    I32,
    I24, // packed in 3 bytes
    I16,
}

impl Sample {
    fn name(self) -> &'static str {
        match self {
            Sample::F32 => "float32",
            Sample::I32 => "int32",
            Sample::I24 => "int24",
            Sample::I16 => "int16",
        }
    }
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
        Sample::I24 => 3,
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
    polled: bool, // an exclusive stream without events: its buffer several periods, topped up every POLL_SEC
}

/// An exclusive stream is polled: asked for a buffer of POLL_PERIODS periods, kept POLL_FILL_PERIODS full (the
/// event-driven one's two periods: the same latency), its output thread topping it up every POLL_SEC; event driven
/// only where the device refuses that. Why: an event-driven exclusive stream whose output thread wakes once a
/// period late runs the device dry, and the DDJ-400's USB Audio class driver then falls back to a 10 ms tick taking
/// one buffer per tick, for good (a 5.8 ms period -> events every 10.00 ms, GetBuffer 0.04 -> 0.75 ms, the device
/// playing 58 % of its rate, its headphones a second behind; Stop / Reset / Start the only way out). Measured on the
/// DDJ-400, its thread held 15, 30 and 60 ms (the fault build, no recovery): event driven at 256 frames (rekordbox's
/// way: JUCE's exclusive stream, buffer = period) fell into that tick at the first 15 ms hold, every time; polled,
/// kept 2 periods (11.6 ms), 3 (17.4 ms) or the whole 2048-frame buffer (46.4 ms), it stayed at its rate through
/// all three (rekordbox has a polled mode too, off by default: WasapiPolling).
const POLL_PERIODS: u32 = 2;
const POLL_SEC: f64 = 0.001;
/// A polled stream is kept this many periods full, not its whole buffer: the driver may give a polled stream a
/// buffer far over the periods asked for (the DDJ-400: 2048 frames, 46 ms at 44.1 kHz, for 2 x 256), and filled
/// whole it would all be latency.
const POLL_FILL_PERIODS: u32 = 2;

/// The level a polled stream of ``device`` is kept at, in periods (0: its whole buffer).
fn poll_fill_periods(device: &str) -> u32 {
    #[cfg(feature = "fault")]
    if let Ok(spec) = std::env::var("MIXLYZER_AUDIO_POLL_FILL") {
        if let Some((part, n)) = spec.split_once(':') {
            if device.contains(part) {
                return n.parse().unwrap_or(POLL_FILL_PERIODS);
            }
        }
    }
    let _ = device;
    POLL_FILL_PERIODS
}

/// The fault build's MIXLYZER_AUDIO_FORMAT=<name>/<bits>,...: only those formats tried (tests).
fn format_allowed(sample: Sample, bits: u16) -> bool {
    #[cfg(feature = "fault")]
    if let Ok(list) = std::env::var("MIXLYZER_AUDIO_FORMAT") {
        return list.split(',').any(|f| f == format!("{}/{bits}", sample.name()));
    }
    let _ = (sample, bits);
    true
}

/// The buffer an exclusive stream of ``device`` is polled with, in periods (0: event driven).
fn poll_periods(device: &str) -> u32 {
    #[cfg(feature = "fault")]
    if let Ok(spec) = std::env::var("MIXLYZER_AUDIO_POLL") {
        if let Some((part, n)) = spec.split_once(':') {
            if device.contains(part) {
                return n.parse().unwrap_or(0);
            }
        }
    }
    let _ = device;
    POLL_PERIODS
}

/// The exclusive stream: the session's format in a sample type the device takes, at the period of its ``level``
/// (PERIOD_LEVELS: its smallest but PERIOD_FRAMES_MIN at least, twice that, its default).
unsafe fn open_exclusive(device: &IMMDevice, rate: u32, channels: usize, level: usize, poll: u32)
    -> Result<Opened, String> {
    unsafe {
        let client: IAudioClient = device.Activate(CLSCTX_ALL, None).map_err(|e| e.to_string())?;
        // JUCE's order (rekordbox's): 24 bits packed before 16, a USB device's native width (the DDJ-400 takes those
        // two; event driven, it fell into its 10 ms tick after one late wake-up at either)
        let candidates = [(Sample::F32, 32), (Sample::I32, 32), (Sample::I32, 24), (Sample::I24, 24), (Sample::I16, 16)];
        let mut chosen = None;
        let mut taken = Vec::new();
        for (sample, bits) in candidates {
            if !format_allowed(sample, bits) {
                continue;
            }
            let f = extensible(sample, bits, rate, channels);
            if client.IsFormatSupported(AUDCLNT_SHAREMODE_EXCLUSIVE, &f.Format, None).is_ok() {
                taken.push(format!("{}/{bits}", sample.name()));
                if chosen.is_none() {
                    chosen = Some((sample, f));
                }
            }
        }
        log_line!("exclusive formats taken at {rate} Hz {channels} ch: {}", taken.join(", "));
        let Some((sample, format)) = chosen else {
            return Err(format!("the device takes no {rate} Hz / {channels} ch format in exclusive mode"));
        };
        let (mut default_period, mut min_period) = (0i64, 0i64);
        client.GetDevicePeriod(Some(&mut default_period), Some(&mut min_period)).map_err(|e| e.to_string())?;
        let floor = (PERIOD_FRAMES_MIN / rate as f64 * 1e7).ceil() as i64;
        let mut period = match level {
            0 => min_period.max(floor),
            1 => min_period.max(2 * floor),
            _ => default_period.max(min_period),
        }
        .max(1);
        let mut client = client;
        let flags = if poll > 0 { Default::default() } else { AUDCLNT_STREAMFLAGS_EVENTCALLBACK };
        let periods = poll.max(1) as i64;
        if let Err(e) = client.Initialize(AUDCLNT_SHAREMODE_EXCLUSIVE, flags, period * periods, period, &format.Format,
                                          None) {
            if e.code() != AUDCLNT_E_BUFFER_SIZE_NOT_ALIGNED {
                return Err(format!("exclusive mode refused: {e}"));
            }
            // the period the device's buffer alignment allows, with a new client
            let frames = client.GetBufferSize().map_err(|e| e.to_string())?;
            period = (10_000_000.0 * frames as f64 / rate as f64 + 0.5) as i64;
            client = device.Activate(CLSCTX_ALL, None).map_err(|e| e.to_string())?;
            client
                .Initialize(AUDCLNT_SHAREMODE_EXCLUSIVE, flags, period * periods, period, &format.Format, None)
                .map_err(|e| format!("exclusive mode refused: {e}"))?;
        }
        finish(client, sample, period as f64 / 10_000.0, poll == 0)
    }
}

/// The low-latency shared stream (IAudioClient3), at the mix format: only when that is the session's format; its
/// period the smallest the engine allows of PERIOD_FRAMES_MIN at least (a multiple of its fundamental period).
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
            let step = fundamental.max(1);
            let wanted = (PERIOD_FRAMES_MIN as u32).max(min);
            let period = (wanted.div_ceil(step) * step).clamp(min, max.max(min));
            client
                .InitializeSharedAudioStream(AUDCLNT_STREAMFLAGS_EVENTCALLBACK, period, mix, None)
                .map_err(|e| format!("low-latency shared stream refused: {e}"))?;
            let base: IAudioClient = windows::core::Interface::cast(&client).map_err(|e| e.to_string())?;
            finish(base, sample, period as f64 * 1000.0 / rate as f64, true)
        })();
        CoTaskMemFree(Some(mix as *const _));
        result
    }
}

unsafe fn finish(client: IAudioClient, sample: Sample, period_ms: f64, evented: bool) -> Result<Opened, String> {
    unsafe {
        let event = CreateEventW(None, false, false, None).map_err(|e| e.to_string())?;
        if evented {
            client.SetEventHandle(event).map_err(|e| e.to_string())?;
        }
        let buffer = client.GetBufferSize().map_err(|e| e.to_string())?;
        let render: IAudioRenderClient = client.GetService().map_err(|e| e.to_string())?;
        Ok(Opened { client, render, event, buffer, sample, period_ms, polled: !evented })
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
            let o = if exclusive {
                let (level, poll) = (period_level(&device_name), poll_periods(&device_name));
                match open_exclusive(&device, rate, channels, level, poll) {
                    Err(e) if poll > 0 => {
                        log_line!("{device_name}: polled exclusive stream refused ({e}): event driven");
                        open_exclusive(&device, rate, channels, level, 0)?
                    }
                    other => other?,
                }
            } else {
                open_shared(&device, rate, channels)?
            };
            Ok::<_, String>((o, device_name))
        })();
        let (o, device_name) = match opened {
            Ok(v) => v,
            Err(e) => {
                let _ = ready.send(Err(e));
                return;
            }
        };
        let _ = ready.send(Ok((device_name.clone(), o.period_ms)));
        let mut state = CallbackState::new(channels, rate as f64);
        // a packed 24-bit stream's samples, made as i32 and packed (no allocation in the loop)
        let mut wide: Vec<i32> = if matches!(o.sample, Sample::I24) { vec![0; o.buffer as usize * channels] } else { vec![] };
        let latency_extra = o.client.GetStreamLatency().unwrap_or(0) as f64 / 1e7;
        // the level kept: a polled stream's POLL_FILL_PERIODS (within its buffer), a shared one's whole buffer
        let period_frames = (o.period_ms * rate as f64 / 1000.0).round() as u32;
        let fill = poll_fill_periods(&device_name);
        let keep = if o.polled && fill > 0 { (fill * period_frames).clamp(period_frames, o.buffer) } else { o.buffer };
        let kind = match (exclusive, o.polled) {
            (false, _) => "low-latency shared (IAudioClient3), event driven",
            (true, true) => "exclusive, polled",
            (true, false) => "exclusive, event driven",
        };
        log_line!("{device_name}: WASAPI {kind}, {} {rate} Hz {channels} ch, period {:.2} ms, buffer {} frames \
                   ({:.2} ms) kept {} frames ({:.2} ms), stream latency {:.2} ms, output thread {}", o.sample.name(),
                  o.period_ms, o.buffer, o.buffer as f64 * 1000.0 / rate as f64, keep,
                  keep as f64 * 1000.0 / rate as f64, latency_extra * 1000.0,
                  if mmcss { "MMCSS Pro Audio" } else { "normal (MMCSS refused)" });
        let period_s = o.period_ms / 1000.0;
        let mut last_wake = 0.0f64;
        // the buffer filled (with silence) before it starts: an exclusive stream's events may not come otherwise
        // (a DJ controller's USB driver: one in 200 ms)
        let whole = exclusive && !o.polled; // each event its whole buffer (the other plays)
        let free = if whole { Ok(o.buffer) } else { o.client.GetCurrentPadding().map(|p| keep.saturating_sub(p)) };
        if let Ok(frames) = free {
            if frames > 0 && o.render.GetBuffer(frames).is_ok() {
                let _ = o.render.ReleaseBuffer(frames, 2); // AUDCLNT_BUFFERFLAGS_SILENT
            }
        }
        if o.client.Start().is_err() {
            port.failed.store(true, Ordering::Relaxed);
        }
        // watched while it runs (THIN_GAP, HEALTH_MIN): started again in place, failed when that keeps happening
        let mut gaps = [0.0f64; THIN_EVENTS];
        let mut gap_count = 0usize;
        let mut restarts: Vec<f64> = Vec::with_capacity(RESTARTS_MAX + 1);
        #[cfg(feature = "fault")]
        let mut last_seen = crate::clock::now();
        let mut health = Health::new(crate::clock::now());
        let rate_f = rate as f64;
        #[cfg(feature = "fault")]
        let mut probe = probe::Probe::new(&o.client, &device_name, crate::clock::now());
        while !stop.load(Ordering::Relaxed) && !port.failed.load(Ordering::Relaxed) {
            let waited = if o.polled {
                std::thread::sleep(std::time::Duration::from_secs_f64(POLL_SEC));
                WAIT_OBJECT_0
            } else {
                #[cfg(feature = "fault")]
                if probe::query_between(&device_name) {
                    // tests: the event waited for 1 ms at a time, the device's padding asked for between (its answer
                    // unused): whether the queries alone keep the driver at its pace
                    let mut waited = windows::Win32::Foundation::WAIT_TIMEOUT;
                    for _ in 0..EVENT_TIMEOUT_MS {
                        waited = WaitForSingleObject(o.event, 1);
                        if waited == WAIT_OBJECT_0 {
                            break;
                        }
                        let _ = o.client.GetCurrentPadding();
                    }
                    waited
                } else {
                    WaitForSingleObject(o.event, EVENT_TIMEOUT_MS)
                }
                #[cfg(not(feature = "fault"))]
                WaitForSingleObject(o.event, EVENT_TIMEOUT_MS)
            };
            let woke = crate::clock::now();
            let mut trouble: Option<String> = None;
            if waited != WAIT_OBJECT_0 {
                trouble = Some(format!("no event for {EVENT_TIMEOUT_MS} ms"));
            } else {
                if last_wake > 0.0 && port.sound_due.load(Ordering::Relaxed) {
                    let gap = woke - last_wake;
                    port.max_gap_us.fetch_max((gap * 1e6) as u64, Ordering::Relaxed);
                    // late: an event 1.8 periods after the last; polled, half its level gone meanwhile
                    let late = if o.polled { 0.5 * keep as f64 / rate as f64 } else { 1.8 * period_s };
                    if gap > late {
                        port.late.fetch_add(1, Ordering::Relaxed);
                        health.skip(woke); // its own lateness: no sign of the device's pace
                    }
                }
                if whole && last_wake > 0.0 {
                    gaps[gap_count % THIN_EVENTS] = woke - last_wake;
                    gap_count += 1;
                    let mean = gaps.iter().sum::<f64>() / THIN_EVENTS as f64;
                    if gap_count >= THIN_EVENTS && mean > THIN_GAP * period_s {
                        trouble = Some(format!("its events {:.1} ms apart at a {:.1} ms period", mean * 1e3,
                                               period_s * 1e3));
                    }
                }
                if trouble.is_none() {
                    if let Some(share) = health.check(woke, rate_f) {
                        if share < HEALTH_MIN {
                            trouble = Some(format!("it took {:.0} % of its rate", share * 100.0));
                        }
                    }
                }
            }
            #[cfg(feature = "fault")]
            if probe::no_recover() {
                trouble = None;
            }
            if let Some(why) = trouble {
                restarts.retain(|&t| woke - t < RESTART_SPAN_SEC);
                restarts.push(woke);
                if restarts.len() > RESTARTS_MAX {
                    // opened again by its owner (an exclusive one at its next period level)
                    let level = if exclusive { step_period_level(&device_name) } else { 0 };
                    let at = if exclusive { format!(" at period level {level}") } else { String::new() };
                    log_line!("{device_name}: not keeping pace {} times in a minute ({why}): opened \
                               again{at}", restarts.len());
                    port.failed.store(true, Ordering::Relaxed);
                    break;
                }
                log_line!("{device_name}: not keeping pace ({why}): the stream started again");
                let _ = o.client.Stop();
                let _ = o.client.Reset();
                let free = if whole { Ok(o.buffer) } else { o.client.GetCurrentPadding().map(|p| keep.saturating_sub(p)) };
                if let Ok(frames) = free {
                    if frames > 0 && o.render.GetBuffer(frames).is_ok() {
                        let _ = o.render.ReleaseBuffer(frames, 2); // AUDCLNT_BUFFERFLAGS_SILENT
                    }
                }
                if o.client.Start().is_err() {
                    port.failed.store(true, Ordering::Relaxed);
                    break;
                }
                source.restarted();
                gap_count = 0;
                last_wake = 0.0;
                health = Health::new(crate::clock::now());
                continue;
            }
            last_wake = woke;
            // exclusive: the whole buffer each period (the other half plays); shared: what is free
            let (frames, queued) = if whole {
                (o.buffer, o.buffer)
            } else {
                match o.client.GetCurrentPadding() {
                    Ok(p) => (keep.saturating_sub(p), p),
                    Err(_) => {
                        port.failed.store(true, Ordering::Relaxed);
                        break;
                    }
                }
            };
            health.took(frames, woke);
            if frames == 0 {
                continue;
            }
            #[cfg(feature = "fault")]
            let gap_seen = woke - last_seen;
            #[cfg(feature = "fault")]
            {
                last_seen = woke;
            }
            let Ok(ptr) = o.render.GetBuffer(frames) else {
                port.failed.store(true, Ordering::Relaxed);
                break;
            };
            #[cfg(feature = "fault")]
            let got = crate::clock::now();
            let n = frames as usize * channels;
            let latency = queued as f64 / rate as f64 + latency_extra;
            match o.sample {
                Sample::F32 => source.fill(std::slice::from_raw_parts_mut(ptr as *mut f32, n), &mut state, latency),
                Sample::I32 => source.fill(std::slice::from_raw_parts_mut(ptr as *mut i32, n), &mut state, latency),
                Sample::I16 => source.fill(std::slice::from_raw_parts_mut(ptr as *mut i16, n), &mut state, latency),
                Sample::I24 => {
                    let wide = &mut wide[..n];
                    source.fill(wide, &mut state, latency);
                    let bytes = std::slice::from_raw_parts_mut(ptr, n * 3);
                    for (k, v) in wide.iter().enumerate() {
                        bytes[3 * k..3 * k + 3].copy_from_slice(&(v >> 8).to_le_bytes()[..3]);
                    }
                }
            }
            #[cfg(feature = "fault")]
            let filled = crate::clock::now();
            if o.render.ReleaseBuffer(frames, 0).is_err() {
                port.failed.store(true, Ordering::Relaxed);
                break;
            }
            #[cfg(feature = "fault")]
            {
                let released = crate::clock::now();
                probe.event(woke, gap_seen, got - woke, filled - got, released - filled, frames);
                probe.stall(released);
            }
            #[cfg(feature = "fault")]
            fault_hold(&device_name, crate::clock::now(), period_s);
        }
        let _ = o.client.Stop();
        let _ = windows::Win32::Foundation::CloseHandle(o.event);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Events every ``gap`` s taking ``frames`` each, until a window's share comes.
    fn share(gap: f64, frames: u32, rate: f64) -> f64 {
        let mut h = Health::new(0.0);
        let mut t = 0.0;
        loop {
            t += gap;
            h.took(frames, t);
            if let Some(s) = h.check(t, rate) {
                return s;
            }
        }
    }

    #[test]
    fn health_sees_a_device_not_keeping_pace() {
        // the DDJ-400 seen: a 10 ms period (441 frames at 44.1 kHz), its events every 15.6 ms: 64 %
        let slow = share(0.0156, 441, 44100.0);
        assert!((slow - 0.641).abs() < 0.02 && slow < HEALTH_MIN, "{slow}");
        // at its period: kept
        let ok = share(0.010, 441, 44100.0);
        assert!((ok - 1.0).abs() < 0.02 && ok >= HEALTH_MIN, "{ok}");
    }

    #[test]
    fn its_own_late_wake_up_is_left_out() {
        // a polled stream kept 512 frames at 44.1 kHz, topped up every 1 ms; its thread held 60 ms once
        let rate = 44100.0;
        let mut h = Health::new(0.0);
        let (mut t, mut worst) = (0.0f64, f64::INFINITY);
        let mut held = false;
        while t < 5.0 {
            let gap = if !held && t > 2.2 { held = true; 0.060 } else { 0.001 };
            t += gap;
            if gap > 0.5 * 512.0 / rate {
                h.skip(t);
            }
            h.took((gap * rate).round() as u32, t);
            if let Some(share) = h.check(t, rate) {
                worst = worst.min(share);
            }
        }
        assert!(worst >= HEALTH_MIN, "{worst}");
    }

    #[test]
    fn period_levels_go_round() {
        let name = "test device (levels)";
        assert_eq!(period_level(name), 0);
        let steps: Vec<usize> = (0..4).map(|_| step_period_level(name)).collect();
        assert_eq!(steps, vec![1, 2, 0, 1]);
        assert_eq!(period_level(name), 1);
    }
}
