//! The buses on devices of their own (the headphones', the booth's): the bridges the main callback writes them into
//! (bridge.rs) and the devices reading them. One thread of their own opens, keeps and lets go of the devices (an
//! open takes tens of ms: the engine thread, feeding the main device's small buffer, never waits for one): the buses
//! naming the same device (and audio API and mode) share one stream of it (a Sink: the headphones on a controller's
//! channels 3-4, the booth on its 1-2; exclusive mode lets a device have only one), at its own format (its mix
//! format, or the first bus's rate asked for: the bridges convert the bus's rate to it), opened again each second
//! while it fails; they run throughout (silence while nothing plays). The engine thread sends it commands and reads
//! what it publishes.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::bridge::{Bridge, Sink};
use crate::clock;
use crate::device::{self, BRIDGED_BUSES, Port, Source};
use crate::ipc::{Control, out};
use crate::monitor::Monitor;

const BUS_NAMES: [&str; BRIDGED_BUSES] = ["headphones", "booth", "headphones B"];

/// What a bus's own output is asked to be (op::SET_AUX_OUTPUT).
#[derive(Clone, PartialEq)]
pub struct AuxWanted {
    pub name: Option<String>,
    pub mode: device::Mode,
    pub buffer_ms: f64,
    pub left: u8,
    pub right: u8,
    pub rate: u32, // its device's rate (0: its mix format's)
}

impl AuxWanted {
    /// The stream it plays in: the device and how it is opened.
    fn device(&self) -> (Option<String>, device::Mode) {
        (self.name.clone(), self.mode)
    }
}

enum Command {
    Set(usize, Option<AuxWanted>),
}

/// What the outputs' thread publishes, per bus.
#[derive(Default)]
struct BusState {
    state: AtomicU32, // out::STATE_*
    mode: AtomicU32,
    period_ms: AtomicU64, // f64 bits
    channels: AtomicU32,
    rate: AtomicU32,
    errors: AtomicU32,
    device: Mutex<String>,
    port: Mutex<Option<Arc<Port>>>, // the stream it plays in (its late wake-ups, its output thread)
}

pub struct AuxOutputs {
    pub bridges: [Arc<Bridge>; BRIDGED_BUSES],
    wanted: [Option<AuxWanted>; BRIDGED_BUSES],
    tx: mpsc::Sender<Command>,
    shared: Arc<[BusState; BRIDGED_BUSES]>,
    monitors: [Monitor; BRIDGED_BUSES],
}

impl AuxOutputs {
    /// The buses at the session's ``rate``, their thread started.
    pub fn new(rate: u32) -> AuxOutputs {
        // a second of each bus: the largest buffer and the chunks around it
        let bridges: [Arc<Bridge>; BRIDGED_BUSES] = std::array::from_fn(|_| Arc::new(Bridge::new(rate as u64, rate as f64)));
        let shared: Arc<[BusState; BRIDGED_BUSES]> = Arc::new(std::array::from_fn(|_| BusState::default()));
        let (tx, rx) = mpsc::channel();
        let (b, s) = (bridges.clone(), shared.clone());
        let _ = std::thread::Builder::new().name("AuxOutputs".into()).spawn(move || worker(rx, b, s));
        AuxOutputs {
            bridges,
            wanted: std::array::from_fn(|_| None),
            tx,
            shared,
            monitors: std::array::from_fn(|_| Monitor::new(rate as f64)),
        }
    }

    /// Bus ``k`` asked for: another device, mode or rate opens its stream again; another buffer or channels apply
    /// as it plays.
    pub fn set(&mut self, k: usize, wanted: Option<AuxWanted>) {
        let Some(old) = self.wanted.get(k) else { return };
        match (old, &wanted) {
            (Some(a), Some(b)) if a.device() == b.device() && a.rate == b.rate => {
                self.bridges[k].set(buffer_frames(&self.bridges[k], b), b.left, b.right);
            }
            (None, None) => {}
            _ => {
                let _ = self.tx.send(Command::Set(k, wanted.clone()));
            }
        }
        self.wanted[k] = wanted;
    }

    /// Each tick: each bus's buffer observed while sound is due.
    pub fn tick(&mut self, due: bool) {
        for k in 0..BRIDGED_BUSES {
            let bridge = &self.bridges[k];
            bridge.sound_due.store(due, Ordering::Relaxed);
            // its device's port too: its late wake-ups and longest gap count while sound is due (never waited for:
            // the outputs' thread holds the lock only to swap the port)
            if let Ok(port) = self.shared[k].port.try_lock() {
                if let Some(p) = port.as_ref() {
                    p.sound_due.store(due, Ordering::Relaxed);
                }
            }
            let open = self.wanted[k].is_some() && self.shared[k].state.load(Ordering::Relaxed) == out::STATE_OPEN;
            let level = if due && open { Some(bridge.buffered(clock::now())) } else { None };
            self.monitors[k].observe(level, bridge.capacity_frames());
        }
    }

    pub fn publish(&self, ctl: &Control) {
        for k in 0..BRIDGED_BUSES {
            let f = |field| out::at(k, field);
            let s = &self.shared[k];
            let (bridge, monitor) = (&self.bridges[k], &self.monitors[k]);
            let state = if self.wanted[k].is_some() { s.state.load(Ordering::Relaxed) } else { out::STATE_OFF };
            ctl.put_u32(f(out::STATE), state);
            ctl.put_u32(f(out::MODE), s.mode.load(Ordering::Relaxed));
            ctl.put_f64(f(out::PERIOD_MS), f64::from_bits(s.period_ms.load(Ordering::Relaxed)));
            ctl.put_f64(f(out::LATENCY_MS), bridge.latency_ms());
            ctl.put_f64(f(out::QUEUED), bridge.buffered(clock::now()));
            ctl.put_f64(f(out::CAPACITY), bridge.capacity_frames());
            ctl.put_u64(f(out::UNDERRUNS), bridge.underruns.load(Ordering::Relaxed));
            if let Ok(port) = s.port.try_lock() {
                let (late, gap, thread) = port.as_ref().map_or((0, 0, 0), |p| {
                    (p.late.load(Ordering::Relaxed), p.max_gap_us.load(Ordering::Relaxed),
                     p.output_state.load(Ordering::Relaxed))
                });
                ctl.put_u64(f(out::LATE), late);
                ctl.put_f64(f(out::MAX_GAP_MS), gap as f64 / 1000.0);
                ctl.put_u32(f(out::THREAD), thread);
            }
            ctl.put_u32(f(out::CHANNELS), s.channels.load(Ordering::Relaxed));
            ctl.put_f64(f(out::DRIFT_PPM), bridge.drift_ppm());
            ctl.put_u64(f(out::BUFFER_GEN), monitor.buffer_gen);
            ctl.put_f64(f(out::OBS_SECONDS), monitor.obs_seconds);
            ctl.put_f64(f(out::OBS_LEVEL), monitor.obs_level);
            ctl.put_f64(f(out::HIST_BUCKET_MS), monitor.hist_bucket_ms);
            ctl.put_f64(f(out::LOWEST_MS), monitor.lowest_ms());
            ctl.put_u32(f(out::ERRORS), s.errors.load(Ordering::Relaxed));
            ctl.put_u32(f(out::RATE), s.rate.load(Ordering::Relaxed));
            if let Ok(name) = s.device.try_lock() {
                ctl.put_str(f(out::NAME), &name, out::NAME_MAX);
            }
            ctl.put_hist_at(f(out::HIST_LEN), f(out::HIST), &monitor.hist);
        }
    }
}

fn buffer_frames(bridge: &Bridge, w: &AuxWanted) -> u64 {
    (w.buffer_ms / 1000.0 * bridge.rate()).round() as u64
}

/// A device's stream: the buses on it (in bus order), the rate asked for (the first bus's), and its output; when
/// its bridges last skipped a surplus (their resyncs then).
struct Stream {
    device: (Option<String>, device::Mode),
    buses: Vec<usize>,
    rate: u32,
    output: Option<device::Output>,
    port: Arc<Port>,
    reopen_at: f64,
    resyncs: u64,
    resynced_at: Vec<f64>,
    last_error: String, // said once until it changes (an open is tried each second)
}

/// A stream whose bridges skipped a surplus more than RESYNCS_MAX times within RESYNC_SPAN_SEC is not keeping pace
/// with the main device (its device takes less than its rate: whatever its backend; the engine's own WASAPI output
/// also watches its events itself): it is opened again.
const RESYNCS_MAX: usize = 2;
const RESYNC_SPAN_SEC: f64 = 30.0;

/// The stream's bridges' resyncs since the last look: true when it has needed too many (RESYNCS_MAX).
fn behind(s: &mut Stream, bridges: &[Arc<Bridge>; BRIDGED_BUSES], now: f64) -> bool {
    let total: u64 = s.buses.iter().map(|&k| bridges[k].resyncs.load(Ordering::Relaxed)).sum();
    if total > s.resyncs {
        s.resynced_at.push(now);
    }
    s.resyncs = total;
    s.resynced_at.retain(|&t| now - t < RESYNC_SPAN_SEC);
    s.resynced_at.len() > RESYNCS_MAX
}

/// The outputs' thread: the buses grouped into a stream per device, each opened, kept and let go as told.
fn worker(rx: mpsc::Receiver<Command>, bridges: [Arc<Bridge>; BRIDGED_BUSES], shared: Arc<[BusState; BRIDGED_BUSES]>) {
    let mut wanted: [Option<AuxWanted>; BRIDGED_BUSES] = std::array::from_fn(|_| None);
    let mut streams: Vec<Stream> = Vec::new();
    loop {
        match rx.recv_timeout(Duration::from_millis(250)) {
            Ok(Command::Set(k, w)) => {
                if k < BRIDGED_BUSES {
                    wanted[k] = w;
                    regroup(&wanted, &mut streams, &bridges, &shared);
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => return,
        }
        let now = clock::now();
        for s in streams.iter_mut() {
            if s.output.is_some() && behind(s, &bridges, now) {
                let names: Vec<&str> = s.buses.iter().map(|&k| BUS_NAMES[k]).collect();
                log_line!("output of {}: its device kept falling behind: opened again", names.join(" + "));
                s.resynced_at.clear();
                s.port.failed.store(true, Ordering::Relaxed);
            }
            if s.output.is_some() && s.port.failed.load(Ordering::Relaxed) {
                s.output = None;
                s.buses.iter().for_each(|&k| bridges[k].stop());
            }
            if s.output.is_none() && now >= s.reopen_at {
                s.reopen_at = now + 1.0;
                open(s, &wanted, &bridges, &shared);
            }
        }
    }
}

/// The streams the buses asked for: a stream per device (its API and mode), its buses in order; one no longer
/// asked for as it is let go, a new one opened (by the caller's loop).
fn regroup(wanted: &[Option<AuxWanted>; BRIDGED_BUSES], streams: &mut Vec<Stream>, bridges: &[Arc<Bridge>; BRIDGED_BUSES],
           shared: &[BusState; BRIDGED_BUSES]) {
    let mut groups: Vec<((Option<String>, device::Mode), Vec<usize>)> = Vec::new();
    for (k, w) in wanted.iter().enumerate() {
        if let Some(w) = w {
            match groups.iter_mut().find(|(d, _)| *d == w.device()) {
                Some((_, buses)) => buses.push(k),
                None => groups.push((w.device(), vec![k])),
            }
        }
    }
    let rate_of = |buses: &[usize]| wanted[buses[0]].as_ref().map_or(0, |w| w.rate);
    // a stream kept only as it is: the same device, the same buses, the same rate
    streams.retain(|s| {
        let keep = groups.iter().any(|(d, b)| *d == s.device && *b == s.buses && rate_of(b) == s.rate);
        if !keep {
            s.buses.iter().for_each(|&k| bridges[k].stop());
        }
        keep
    });
    for (k, w) in wanted.iter().enumerate() {
        if w.is_none() {
            let s = &shared[k];
            s.state.store(out::STATE_OFF, Ordering::Relaxed);
            *s.device.lock().unwrap_or_else(|e| e.into_inner()) = String::new();
            *s.port.lock().unwrap_or_else(|e| e.into_inner()) = None;
        }
    }
    for (device, buses) in groups {
        if streams.iter().any(|s| s.device == device && s.buses == buses) {
            continue;
        }
        buses.iter().for_each(|&k| shared[k].state.store(out::STATE_OPENING, Ordering::Relaxed));
        let rate = rate_of(&buses);
        streams.push(Stream { device, buses, rate, output: None, port: Arc::new(Port::default()), reopen_at: 0.0,
                              resyncs: 0, resynced_at: Vec::new(), last_error: String::new() });
    }
}

fn start_bridges(s: &Stream, wanted: &[Option<AuxWanted>; BRIDGED_BUSES], bridges: &[Arc<Bridge>; BRIDGED_BUSES]) {
    for &k in &s.buses {
        if let Some(w) = &wanted[k] {
            bridges[k].start(buffer_frames(&bridges[k], w), w.left, w.right);
        }
    }
}

/// Opens a stream's device at its own format (its mix format, or the rate asked for), its buses' bridges prepared
/// for that rate; its device alone (not there: device::NOT_PRESENT, tried again each second).
fn open(s: &mut Stream, wanted: &[Option<AuxWanted>; BRIDGED_BUSES], bridges: &[Arc<Bridge>; BRIDGED_BUSES],
        shared: &[BusState; BRIDGED_BUSES]) {
    let (name, mode) = s.device.clone();
    let names: Vec<&str> = s.buses.iter().map(|&k| BUS_NAMES[k]).collect();
    let fail = |s: &mut Stream, e: String| {
        for &k in &s.buses {
            shared[k].errors.fetch_add(1, Ordering::Relaxed);
            shared[k].state.store(out::STATE_FAILED, Ordering::Relaxed);
        }
        if e != s.last_error {
            log_line!("output of {} not opened: {e} (tried again each second)", names.join(" + "));
            s.last_error = e;
        }
    };
    let (rate, channels) = match device::default_format(name.as_deref(), mode, false) {
        Ok((mix, ch)) => (if s.rate > 0 { s.rate } else { mix }, ch),
        Err(e) => return fail(s, e),
    };
    for &k in &s.buses {
        bridges[k].prepare(rate as f64);
    }
    s.port = Arc::new(Port::default());
    let sink = Sink::new(s.port.clone(), rate as f64, s.buses.iter().map(|&k| (k, bridges[k].clone())).collect());
    match device::open(Source::Sink(Arc::new(sink)), rate, channels, name.as_deref(), mode, false) {
        Ok((o, opened)) => {
            s.last_error.clear();
            log_line!("{} on {opened}: {}, {rate} Hz, {channels} ch", names.join(" + "), o.mode.label());
            for &k in &s.buses {
                let b = &shared[k];
                b.mode.store(o.mode.code(), Ordering::Relaxed);
                b.period_ms.store(o.period_ms.to_bits(), Ordering::Relaxed);
                b.channels.store(channels as u32, Ordering::Relaxed);
                b.rate.store(rate, Ordering::Relaxed);
                *b.device.lock().unwrap_or_else(|e| e.into_inner()) = opened.clone();
                *b.port.lock().unwrap_or_else(|e| e.into_inner()) = Some(s.port.clone());
            }
            start_bridges(s, wanted, bridges);
            s.output = Some(o);
            for &k in &s.buses {
                shared[k].state.store(out::STATE_OPEN, Ordering::Relaxed);
            }
        }
        Err(e) => fail(s, e),
    }
}
