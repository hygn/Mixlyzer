//! Mixlyzer's audio engine: a process of its own, so playback never waits for the GUI's Python (its GIL,
//! its garbage collector, its busy frames). The GUI process (audioengine/native/client.py) starts it with the
//! name of a shared memory block it made; commands come in and the engine's state goes out through it
//! (ipc.rs), each track's PCM through a block of its own. The engine exits when told to or when the GUI
//! process is gone.
//!
//! The signal flow: up to MAX_DECKS decks (deck.rs; the app uses deck 1), each a player (engine.rs) rendering its
//! lane ahead; the mixer (mixer.rs) in the main output's callback (device.rs): each deck's channel strip (trim, EQ,
//! meter, CUE, fader), the crossfader, the master, the headphones, the booth, soft clipping, the routing to the
//! device's channels; the headphones and the booth may play on devices of their own (bridge.rs, outputs.rs). A MIDI
//! controller drives the decks and the mixer (controller.rs). The decks play stereo whatever the device's
//! channels. The outputs run from the start to the end, silence while nothing plays.
//!
//! mixlyzer-audio --shm NAME --size BYTES --parent PID [--device NAME] [--api 0 WASAPI | 1 cpal | 2 JACK]
//!     [--exclusive 0|1] [--rate HZ: exclusive only, else the device's]

mod bridge;
mod clock;
mod controller;
mod cursor;
mod deck;
mod device;
mod dsp;
mod engine;
mod ipc;
mod midi;
mod mixer;
mod mixers;
mod monitor;
mod outputs;
mod sched;
mod shm;
mod stretch;
mod timeline;
mod vk;
#[cfg(windows)]
mod wasapi;

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use controller::Controller;
use deck::{Decks, Slot};
use device::{BRIDGED_BUSES, Port, Source};
use ipc::{Control, MAX_DECKS, Reader, op, st};
use mixer::{Assign, Curve, MixParams};
use outputs::{AuxOutputs, AuxWanted};

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if let Some(frames) = arg(&args, "--probe-device") {
        probe_device(frames.parse().unwrap_or(0));
        return;
    }
    if let Some(input) = arg(&args, "--scratch-file") {
        scratch_file(&args, &input);
        return;
    }
    if let Some(input) = arg(&args, "--stretch-file") {
        stretch_file(&args, &input);
        return;
    }
    let (Some(name), Some(size)) = (arg(&args, "--shm"), arg(&args, "--size").and_then(|s| s.parse::<usize>().ok()))
    else {
        eprintln!("usage: mixlyzer-audio --shm NAME --size BYTES [--parent PID]");
        std::process::exit(2);
    };
    let parent = arg(&args, "--parent").and_then(|s| s.parse::<u32>().ok()).map(Parent::watch);
    let block = match shm::Shm::open(&name, size, true) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("[AudioEngine] {e}");
            std::process::exit(1);
        }
    };
    let ctl = match Control::new(block.ptr(), block.len()) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[AudioEngine] {e}");
            std::process::exit(1);
        }
    };
    let machine = vk::warm();
    bridge::warm();
    let mut sched = sched::Sched::start();
    eprintln!("[AudioEngine] VOLK {} ({machine}), engine thread: {}", volk::volk_version(), sched.engine_state());
    // the output device asked for (None: the system's default); the session's format is its own
    let mut wanted_device: Option<String> = arg(&args, "--device").filter(|n| !n.is_empty());
    let code = |name: &str| arg(&args, name).and_then(|m| m.parse::<u32>().ok()).unwrap_or(0);
    let mut wanted_mode = device::Mode::from_codes(code("--api"), code("--exclusive"));
    // the session's format: the device's, at the rate asked for in exclusive mode (the GUI decodes for it)
    let asked_rate = code("--rate");
    let format = device::default_format(wanted_device.as_deref(), wanted_mode)
        .map(|(r, c)| (if wanted_mode.exclusive && asked_rate > 0 { asked_rate } else { r }, c));
    let (rate, ch) = match format {
        Ok(f) => f,
        Err(e) => {
            eprintln!("[AudioEngine] no output device: {e}");
            ctl.state_begin();
            ctl.put_u32(st::READY, 2);
            ctl.state_end();
            std::process::exit(1);
        }
    };
    // the decks play stereo (a mono device: mono); the device's channels take the buses as routed
    let out_ch = ch;
    let ch = out_ch.clamp(1, device::DECK_CH_MAX);
    // each deck's lane ([music, aids] a frame) holds the largest output buffer and what a flush leaves behind it;
    // the main device's ring is deck 1's lane, the others hang on it (sharing its port); the headphones' and the
    // booth's bridges hold a second
    let port = Arc::new(Port::default());
    let lane = || Arc::new(device::Ring::new(rate as u64 * 5 / 2, 2 * ch, rate as f64, port.clone()));
    let ring = lane();
    let lanes: Vec<Arc<device::Ring>> = (0..MAX_DECKS).map(|k| if k == 0 { ring.clone() } else { lane() }).collect();
    let params = Arc::new(MixParams::new(out_ch));
    let mut aux = AuxOutputs::new(rate);
    ring.attach_bus(device::Bus { decks: lanes[1..].to_vec(), params: params.clone(), bridges: aux.bridges.clone() });
    let _ = port.engine_thread.set(std::thread::current());
    let mut current_device = String::new();
    let mut output = match device::open(Source::Main(ring.clone()), rate, out_ch, wanted_device.as_deref(), wanted_mode) {
        Ok((o, name)) => {
            current_device = name;
            Some(o)
        }
        Err(e) => {
            eprintln!("[AudioEngine] device open failed: {e}");
            None
        }
    };
    let mut decks = Decks::new(rate, ch, lanes, params.clone());
    let mut controller = Controller::new();
    let mut cmd_done = 0u64;
    let mut device_errors = 0u32;
    let mut reopen_at = 0.0;
    let mut parent_checked = 0.0;
    ctl.state_begin();
    ctl.put_u32(st::RATE, rate);
    ctl.put_u32(st::CHANNELS, ch as u32);
    ctl.put_u32(st::OUT_CHANNELS, out_ch as u32);
    ctl.put_u32(st::READY, 1);
    ctl.state_end();

    let mut trace = Trace::new();
    let mut last_op = (0u32, 0.0f64);
    loop {
        let tick_start = clock::now();
        let level_before = ring.level();
        let mut batch = Vec::new();
        while let Some(cmd) = ctl.next_command() {
            batch.push(cmd);
        }
        for (i, cmd) in batch.iter().enumerate() {
            cmd_done = cmd.seq;
            last_op = (cmd.op, clock::now());
            if cmd.op == op::SHUTDOWN {
                decks.iter_mut().for_each(|s| s.engine.pause());
                controller.close_midi();
                publish(&ctl, &mut decks, &controller, cmd_done, device_errors, output.as_ref(), &sched, &aux);
                return;
            }
            // a seek another one in this batch overrides (a drag's positions come faster than the ticks): only
            // the last is made, each one costs a crossfade, with the time stretch a new stream
            let superseded = cmd.op == op::SEEK
                && batch[i + 1..].iter().any(|c| matches!(c.op, op::SEEK | op::STOP | op::LOAD | op::UNLOAD));
            match cmd.op {
                op::SET_DEVICE => {
                    // opened again on the next lines, at the session's format; what the ring holds plays on there
                    wanted_device = Reader::new(&cmd.payload).str().filter(|n| !n.is_empty());
                    output = None;
                    reopen_at = 0.0;
                }
                op::SET_OUTPUT_MODE => {
                    let mut r = Reader::new(&cmd.payload);
                    wanted_mode = device::Mode::from_codes(r.u32().unwrap_or(0), r.u32().unwrap_or(0));
                    output = None;
                    reopen_at = 0.0;
                }
                op::LIST_DEVICES => {
                    // the devices of the API asked for (none: the main output's)
                    let mode = match Reader::new(&cmd.payload).u32() {
                        Some(api) => device::Mode::from_codes(api, 0),
                        None => wanted_mode,
                    };
                    let (default, listed) = device::list(mode);
                    ctl.put_devices(&current_device, default.as_ref(), &listed);
                }
                op::SET_PROCESS => {
                    let mut r = Reader::new(&cmd.payload);
                    sched.set(r.u32().unwrap_or(sched::PRIORITY_HIGH), r.u32().unwrap_or(0) != 0, &port);
                }
                op::SET_AUX_OUTPUT => {
                    let mut r = Reader::new(&cmd.payload);
                    let (Some(k), Some(on), Some(api), Some(exclusive), Some(buffer_ms), Some(left), Some(right)) =
                        (r.u32(), r.u32(), r.u32(), r.u32(), r.f64(), r.u8(), r.u8()) else { continue };
                    let _pad = (r.u8(), r.u8());
                    let device_rate = r.u32().unwrap_or(0);
                    let name = r.str().filter(|n| !n.is_empty());
                    if (k as usize) < BRIDGED_BUSES {
                        let wanted = (on != 0).then(|| AuxWanted {
                            name,
                            mode: device::Mode::from_codes(api, exclusive),
                            buffer_ms: if buffer_ms.is_finite() { buffer_ms.clamp(2.0, 500.0) } else { 20.0 },
                            left,
                            right,
                            rate: device_rate.min(768_000),
                        });
                        aux.set(k as usize, wanted);
                    }
                }
                op::MIDI_LIST => ctl.put_midi_ports(&midi::ports()),
                op::MIDI_OPEN => {
                    let name = Reader::new(&cmd.payload).str().unwrap_or_default();
                    controller.open_midi(&name, port.engine_thread.get().cloned());
                }
                op::MIDI_MAP => controller.set_bindings(&cmd.payload),
                op::MIDI_OPTIONS => decks.set_jog_ticks(Reader::new(&cmd.payload).u32().unwrap_or(720)),
                op::SET_METER_GAIN => controller.set_meter_gain(Reader::new(&cmd.payload).f64().unwrap_or(1.0)),
                // what every deck takes alike
                op::SET_TIMESTRETCH => decks.set_timestretch(Reader::new(&cmd.payload).u32().unwrap_or(0) != 0),
                op::SET_SOFT_CLIP => params.set_soft_clip(Reader::new(&cmd.payload).u32().unwrap_or(0) != 0),
                op::SET_HEADPHONES => {
                    let mut r = Reader::new(&cmd.payload);
                    params.set_master_cue(r.u32().unwrap_or(1) != 0);
                    params.set_cue_mix(r.f64().unwrap_or(0.5));
                    params.set_cue_level(r.f64().unwrap_or(1.0));
                }
                op::SET_BUFFER_MS => decks.set_buffer_ms(Reader::new(&cmd.payload).f64().unwrap_or(40.0)),
                op::SET_VOLUME_TRIM => decks.set_volume_trim(Reader::new(&cmd.payload).f64().unwrap_or(0.0)),
                // the mix (the output callback's)
                op::SET_CROSSFADER => {
                    let mut r = Reader::new(&cmd.payload);
                    params.set_crossfader(r.f64().unwrap_or(0.5));
                    params.set_curve(if r.u32().unwrap_or(0) == 1 { Curve::Sharp } else { Curve::Smooth });
                }
                op::SET_ROUTING => {
                    let mut r = Reader::new(&cmd.payload);
                    let routes: [u8; 6] = std::array::from_fn(|_| r.u8().unwrap_or(mixer::NO_CHANNEL));
                    params.set_routes(routes);
                }
                op::SET_ASSIGN => {
                    let mut r = Reader::new(&cmd.payload);
                    let k = r.u32().unwrap_or(0) as usize;
                    let assign = match r.u32().unwrap_or(1) {
                        0 => Assign::A,
                        2 => Assign::B,
                        _ => Assign::Thru,
                    };
                    params.set_assign(k, assign);
                }
                op::SET_CHANNEL_CUE => {
                    let mut r = Reader::new(&cmd.payload);
                    let k = r.u32().unwrap_or(0) as usize;
                    params.set_cue(k, r.u32().unwrap_or(0) != 0);
                }
                // a deck's command (the others: deck 1's)
                op::FOR_DECK => {
                    let mut r = Reader::new(&cmd.payload);
                    let (Some(k), Some(inner)) = (r.u32(), r.u32()) else { continue };
                    if (k as usize) < MAX_DECKS && !matches!(inner, op::FOR_DECK | op::SHUTDOWN) {
                        let slot = decks.get(k as usize);
                        slot.engine.command_at = cmd.sent_at;
                        handle(slot, inner, r.rest());
                    }
                }
                _ if !superseded => {
                    let slot = decks.first();
                    slot.engine.command_at = cmd.sent_at;
                    handle(slot, cmd.op, &cmd.payload);
                }
                _ => {}
            }
        }
        let now = clock::now();
        // the device: started again when it failed (device loss; at most once a second); it runs throughout
        if port.failed.load(Ordering::Relaxed) || output.is_none() {
            if now >= reopen_at {
                reopen_at = now + 1.0;
                output = None;
                port.failed.store(false, Ordering::Relaxed);
                match device::open(Source::Main(ring.clone()), rate, out_ch, wanted_device.as_deref(), wanted_mode) {
                    Ok((o, name)) => {
                        output = Some(o);
                        if name != current_device {
                            eprintln!("[AudioEngine] output device: {name}");
                            current_device = name;
                        }
                    }
                    Err(e) => {
                        device_errors += 1;
                        eprintln!("[AudioEngine] device reopen failed: {e}");
                    }
                }
            }
        }
        // the buses on devices of their own: their buffers observed while sound is due
        let due = decks.iter_mut().any(|s| s.engine.sound_due());
        port.sound_due.store(due, Ordering::Relaxed);
        aux.tick(due);
        sched.follow_output(&port);
        // the controller's messages as soon as they come (its input thread wakes this one)
        controller.read_midi(&mut decks, &params, &ctl);
        for slot in decks.iter_mut() {
            slot.deck.tick(&mut slot.engine);
            slot.engine.tick();
            slot.engine.take_peak();
        }
        controller.tick(&decks, &params);
        publish(&ctl, &mut decks, &controller, cmd_done, device_errors, output.as_ref(), &sched, &aux);
        let playing = decks.first().engine.playing;
        trace.tick(&ring, tick_start, level_before, last_op, playing, output.as_ref().map(|o| o.mode.code()));
        if now - parent_checked > 0.5 {
            parent_checked = now;
            if parent.as_ref().is_some_and(|p| !p.alive()) {
                return;
            }
        }
        // woken by the device's read (the ring topped up at once) or a MIDI message, else every 2 ms (1 ms while
        // scratching, a jog wheel bending, or with a buffer under 10 ms)
        let fast = decks.iter_mut().any(|s| s.engine.scratching || s.engine.small_buffer() || s.deck.wants_fast_ticks());
        std::thread::park_timeout(Duration::from_micros(if fast { 1000 } else { 2000 }));
    }
}

/// Offline: the time stretch of a raw float32 file (interleaved), for checks against the Python engine's.
/// --stretch-file IN --out OUT --channels C --rate R --factor F [--block N]
fn stretch_file(args: &[String], input: &str) {
    let get = |n: &str, d: &str| arg(args, n).unwrap_or_else(|| d.to_string());
    let ch: usize = get("--channels", "2").parse().unwrap();
    let rate: u32 = get("--rate", "48000").parse().unwrap();
    let factor: f64 = get("--factor", "1.0").parse().unwrap();
    let block: usize = get("--block", "480").parse().unwrap();
    let bytes = std::fs::read(input).expect("read input");
    let pcm: Vec<f32> = bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
    vk::warm();
    let mut s = stretch::Stretcher::new(ch, rate);
    let mut pos = 0.0;
    let mut out: Vec<f32> = Vec::new();
    let t0 = clock::now();
    loop {
        let r = s.render(&pcm, pos, factor, block);
        if r.is_empty() {
            break;
        }
        pos = s.expected_input_position().unwrap_or(pos + factor * (r.len() / ch) as f64);
        out.extend_from_slice(&r);
    }
    let took = clock::now() - t0;
    let secs = out.len() as f64 / ch as f64 / rate as f64;
    eprintln!("rendered {secs:.2} s in {:.3} s ({:.2} % of real time)", took, 100.0 * took / secs.max(1e-9));
    let bytes: Vec<u8> = out.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(get("--out", "out.f32"), bytes).expect("write output");
}

/// Offline: the scratch path (platter + read_at) over a raw float32 file, for checks against the Python
/// engine's. Events: float64 triples (time s, input frame, rest); blocks of --block frames, the first heard at
/// time 0, each heard --block / rate later. --scratch-file IN --events EV --out OUT --channels C --rate R
fn scratch_file(args: &[String], input: &str) {
    let get = |n: &str, d: &str| arg(args, n).unwrap_or_else(|| d.to_string());
    let ch: usize = get("--channels", "2").parse().unwrap();
    let rate: f64 = get("--rate", "48000").parse().unwrap();
    let block: usize = get("--block", "48").parse().unwrap();
    let f32s = |path: &str| -> Vec<f32> {
        std::fs::read(path).unwrap().chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect()
    };
    let pcm = f32s(input);
    let ev: Vec<f64> = std::fs::read(get("--events", "events.f64"))
        .unwrap()
        .chunks_exact(8)
        .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let tables = dsp::ReadTables::new();
    let mut platter = dsp::Platter::new(rate);
    platter.reset(ev[1], 0.0, 0.0);
    let end = ev[ev.len() - 3] + 0.2;
    let mut next = 3;
    let mut out = Vec::new();
    let mut t = 0.0;
    while t < end {
        while next < ev.len() && ev[next] <= t {
            platter.set_target(ev[next + 1], ev[next], ev[next + 2] != 0.0);
            next += 3;
        }
        let start = platter.position;
        let pos = platter.positions(block, t);
        out.extend(dsp::read_at(&tables, &pcm, ch, &pos, Some(start)));
        t += block as f64 / rate;
    }
    let bytes: Vec<u8> = out.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(get("--out", "out.f32"), bytes).unwrap();
}

/// The device's latency with a requested buffer (0: the default): silence for a second, the latencies seen.
fn probe_device(frames: u32) {
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
    let host = cpal::default_host();
    let dev = host.default_output_device().unwrap();
    let def = dev.default_output_config().unwrap();
    let config = cpal::StreamConfig {
        channels: def.channels(),
        sample_rate: def.sample_rate(),
        buffer_size: if frames == 0 { cpal::BufferSize::Default } else { cpal::BufferSize::Fixed(frames) },
    };
    let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::<(f64, usize)>::new()));
    let s2 = seen.clone();
    let stream = dev
        .build_output_stream(
            config,
            move |data: &mut [f32], info: &cpal::OutputCallbackInfo| {
                data.iter_mut().for_each(|v| *v = 0.0);
                let ts = info.timestamp();
                s2.lock().unwrap().push((ts.playback.duration_since(ts.callback).as_secs_f64() * 1000.0, data.len()));
            },
            |e| eprintln!("{e}"),
            None,
        )
        .unwrap();
    stream.play().unwrap();
    std::thread::sleep(std::time::Duration::from_secs(1));
    drop(stream);
    let v = seen.lock().unwrap();
    let lat: Vec<f64> = v.iter().skip(10).map(|x| x.0).collect();
    let n = v.iter().skip(10).map(|x| x.1).max().unwrap_or(0);
    println!(
        "requested {frames}: callbacks {}, frames per callback up to {}, latency min {:.1} mean {:.1} max {:.1} ms",
        v.len(),
        n / def.channels() as usize,
        lat.iter().cloned().fold(f64::MAX, f64::min),
        lat.iter().sum::<f64>() / lat.len().max(1) as f64,
        lat.iter().cloned().fold(0.0, f64::max)
    );
}

/// The last ticks of the engine loop, written to the temp folder's mixlyzer-audio-trace.log when the device
/// played a gap (a dropout: the ring empty; or the output thread woken late): what the loop was doing before it.
struct Trace {
    ticks: std::collections::VecDeque<String>,
    underruns: u64,
    late: u64,
    last_end: f64,
    path: std::path::PathBuf,
}

impl Trace {
    fn new() -> Trace {
        Trace {
            ticks: std::collections::VecDeque::with_capacity(64),
            underruns: 0,
            late: 0,
            last_end: 0.0,
            path: std::env::temp_dir().join("mixlyzer-audio-trace.log"),
        }
    }

    fn tick(&mut self, ring: &device::Ring, start: f64, level_before: u64, last_op: (u32, f64), playing: bool, mode: Option<u32>) {
        let end = clock::now();
        let gap = if self.last_end > 0.0 { (start - self.last_end) * 1000.0 } else { 0.0 };
        self.last_end = end;
        if self.ticks.len() == 64 {
            self.ticks.pop_front();
        }
        self.ticks.push_back(format!(
            "{start:.4} work {:.2} ms, slept {gap:.2} ms, ring {level_before} -> {} frames, last op {} {:.0} ms ago",
            (end - start) * 1000.0,
            ring.level(),
            last_op.0,
            (end - last_op.1) * 1000.0
        ));
        let (u, l) = (ring.underruns.load(Ordering::Relaxed), ring.port.late.load(Ordering::Relaxed));
        if (u > self.underruns || l > self.late) && playing {
            use std::io::Write;
            if std::fs::metadata(&self.path).map(|m| m.len() > 2 << 20).unwrap_or(false) {
                let _ = std::fs::remove_file(&self.path);
            }
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&self.path) {
                let _ = writeln!(
                    f,
                    "== {end:.4}: {} frames played as silence (ring empty), {} late wake-ups, max gap {:.2} ms, output {:?}",
                    u - self.underruns,
                    l - self.late,
                    ring.port.max_gap_us.load(Ordering::Relaxed) as f64 / 1000.0,
                    mode
                );
                for t in &self.ticks {
                    let _ = writeln!(f, "   {t}");
                }
            }
        }
        self.underruns = u;
        self.late = l;
    }
}

/// A deck's command.
fn handle(slot: &mut Slot, op_code: u32, payload: &[u8]) {
    let Slot { engine: e, deck } = slot;
    let mut r = Reader::new(payload);
    if matches!(op_code, op::LOAD | op::UNLOAD) {
        deck.new_track();
    }
    match op_code {
        op::DECK_ACTION => {
            let (code, pressed) = (r.u32().unwrap_or(0), r.u32().unwrap_or(0) != 0);
            deck.action(e, code, pressed);
        }
        op::SET_TEMPO => deck.set_tempo(e, r.f64().unwrap_or(1.0)),
        op::SET_TEMPO_RANGE => deck.set_tempo_range(e, r.u32().unwrap_or(100)),
        op::SET_VOLUME => deck.set_volume(e, r.f64().unwrap_or(1.0)),
        op::DECK_MARKS => deck.set_marks(payload),
        op::JUMP_TO => deck.jump_to(e, r.f64().unwrap_or(f64::NAN)),
        op::CALL_AT => deck.call_at(e, r.f64().unwrap_or(f64::NAN)),
        op::LOAD => {
            let (Some(generation), Some(channels), Some(frames), Some(name)) = (r.u32(), r.u32(), r.u64(), r.str()) else {
                return;
            };
            if channels as usize != e.ch {
                eprintln!("[AudioEngine] track has {channels} channels, the device {}", e.ch);
                e.set_track(None, generation);
                return;
            }
            let Ok(frames) = usize::try_from(frames) else {
                e.set_track(None, generation);
                return;
            };
            match shm::Pcm::open(&name, frames, channels as usize) {
                Ok(pcm) => e.set_track(Some(pcm), generation),
                Err(err) => {
                    eprintln!("[AudioEngine] {err}");
                    e.set_track(None, generation);
                }
            }
        }
        op::UNLOAD => {
            let generation = r.u32().unwrap_or(0);
            e.set_track(None, generation);
        }
        op::PLAY => {
            e.play();
        }
        op::PAUSE => e.pause(),
        op::STOP => e.stop(),
        op::SEEK => {
            if let Some(f) = r.f64() {
                e.seek(f);
            }
        }
        op::SCRUB => e.set_scrubbing(r.u32().unwrap_or(0) != 0),
        op::SCRATCH => {
            let on = r.u32().unwrap_or(0) != 0;
            let fpp = r.f64().unwrap_or(0.0);
            e.set_scratching(on, fpp);
        }
        op::SCRATCH_TO => {
            if let (Some(f), Some(at)) = (r.f64(), r.f64()) {
                e.scratch_to(f, at);
            }
        }
        op::ARM_JUMP => {
            let (Some(start), Some(dest), Some(once), Some(n)) = (r.f64(), r.f64(), r.u32(), r.u32()) else { return };
            let n = r.count(n as u64, 8);
            let starts: Vec<f64> = (0..n).filter_map(|_| r.f64()).filter(|v| v.is_finite()).collect();
            e.arm_jump(if start.is_finite() { Some(start) } else { None }, dest, once != 0, starts);
        }
        op::DISARM_JUMP => e.disarm_jump(),
        op::SET_MODE => e.set_mode_speed(r.u32().unwrap_or(1) != 0),
        op::SET_FACTOR => e.set_factor(r.f64().unwrap_or(1.0)),
        op::TONE_HOLD => {
            let n = r.u32().unwrap_or(0) as u64;
            let n = r.count(n, 4);
            let notes: Vec<i32> = (0..n).filter_map(|_| r.i32()).collect();
            e.tones.hold(notes);
        }
        op::CLICK_SAMPLE => {
            let (frames, channels) = (r.u32().unwrap_or(0) as u64, r.u32().unwrap_or(0) as u64);
            let values = frames.saturating_mul(channels);
            // a sample is all there or not taken (the payload holds every value)
            if values == 0 || r.count(values, 4) as u64 != values {
                e.clicks.set_sample(None);
            } else {
                let data: Vec<f32> = (0..values).filter_map(|_| r.f32()).collect();
                e.clicks.set_sample(Some((data, channels as usize)));
            }
        }
        op::CLICK_PITCHES => {
            let (b, a) = (r.f64().unwrap_or(0.0), r.f64().unwrap_or(0.0));
            e.clicks.set_pitches(b, a);
        }
        op::CLICK_BEATS => {
            let n = r.u32().unwrap_or(0) as u64;
            let n = r.count(n, 8 + 4 + 1);
            let beats: Vec<f64> = (0..n).filter_map(|_| r.f64()).collect();
            let gains: Vec<f32> = (0..n).filter_map(|_| r.f32()).collect();
            let accents: Vec<bool> = (0..n).filter_map(|_| r.u8()).map(|v| v != 0).collect();
            e.clicks.set_beats(&beats, gains, accents);
        }
        op::CLICK_ENABLED => e.clicks.set_enabled(r.u32().unwrap_or(0) != 0),
        op::CLICK_OFFSET => e.clicks.set_offset_sec(r.f64().unwrap_or(0.0)),
        op::CLICK_DUCKING => e.clicks.set_ducking(r.u32().unwrap_or(0) != 0),
        op::CLICK_CLEAR => e.clicks.clear_voices(),
        op::WAKE => {}
        _ => {}
    }
}

#[allow(clippy::too_many_arguments)]
fn publish(ctl: &Control, decks: &mut Decks, controller: &Controller, cmd_done: u64, device_errors: u32,
           out: Option<&device::Output>, sched: &sched::Sched, aux: &AuxOutputs) {
    let now = clock::now();
    // every deck's, then deck 1's in full (the app's)
    let mut rows: [Option<(u32, u32, u32, [f64; 5])>; MAX_DECKS] = [None; MAX_DECKS];
    for (k, row) in rows.iter_mut().enumerate() {
        if let Some(s) = decks.slot(k) {
            let e = &s.engine;
            *row = Some((e.playing as u32, e.track_gen, s.deck.lamps(e, now), [
                e.playhead_frame(),
                e.factor(),
                s.deck.volume(),
                s.deck.cue_frame(e),
                e.jump_armed().unwrap_or(f64::NAN),
            ]));
        }
    }
    let Slot { engine: e, deck } = decks.first();
    let (display, display_rate) = e.display_frame(now);
    let playhead = e.playhead_frame();
    let peak = e.heard_peak_dbfs();
    let ring = e.ring();
    let queued = e.buffered(now);
    ctl.state_begin();
    ctl.put_u32(st::PLAYING, e.playing as u32);
    ctl.put_u32(st::TRACK_GEN, e.track_gen);
    ctl.put_u32(st::AT_END, e.at_end() as u32);
    ctl.put_u64(st::CMD_DONE, cmd_done);
    ctl.put_f64(st::PLAYHEAD, playhead);
    ctl.put_f64(st::DISPLAY, display);
    ctl.put_f64(st::DISPLAY_AT, now);
    ctl.put_f64(st::DISPLAY_RATE, display_rate);
    ctl.put_f64(st::PEAK, peak);
    ctl.put_f64(st::QUEUED, if e.idle() { 0.0 } else { queued });
    ctl.put_f64(st::CAPACITY, e.capacity_frames());
    ctl.put_u64(st::JUMPED, e.jumped_count);
    ctl.put_u64(st::FINISHED, e.finished_count);
    ctl.put_u64(st::BUFFER_GEN, e.monitor.buffer_gen);
    ctl.put_f64(st::OBS_SECONDS, e.monitor.obs_seconds);
    ctl.put_f64(st::OBS_LEVEL, e.monitor.obs_level);
    ctl.put_u64(st::UNDERRUNS, ring.underruns.load(Ordering::Relaxed));
    ctl.put_u32(st::DEVICE_ERRORS, device_errors);
    ctl.put_f64(st::STRETCH_MS, e.render_ms);
    ctl.put_f64(st::LATENCY_MS, ring.latency_frames() / e.rate * 1000.0);
    ctl.put_u32(st::OUTPUT_MODE, out.map(|o| o.mode.code()).unwrap_or(0));
    ctl.put_f64(st::PERIOD_MS, out.map(|o| o.period_ms).unwrap_or(0.0));
    ctl.put_f64(st::RESPONSE_PLAY, e.responses[0]);
    ctl.put_f64(st::RESPONSE_PAUSE, e.responses[1]);
    ctl.put_f64(st::RESPONSE_SEEK, e.responses[2]);
    ctl.put_f64(st::HIST_BUCKET_MS, e.monitor.hist_bucket_ms);
    ctl.put_f64(st::LOWEST_MS, e.monitor.lowest_ms());
    ctl.put_u32(st::OUTPUT_OPEN, out.is_some() as u32);
    ctl.put_u64(st::DEVICE_LATE, ring.port.late.load(Ordering::Relaxed));
    ctl.put_f64(st::MAX_GAP_MS, ring.port.max_gap_us.load(Ordering::Relaxed) as f64 / 1000.0);
    ctl.put_u32(st::THREADS, sched.engine_state() | sched.output_state(&ring.port) << 8);
    ctl.put_u32(st::POWER, sched.power_state());
    ctl.put_hist(&e.monitor.hist);
    ctl.put_f64(st::FACTOR, e.factor());
    ctl.put_f64(st::VOLUME, deck.volume());
    ctl.put_f64(st::CUE, deck.cue_frame(e));
    ctl.put_f64(st::JUMP_DEST, e.jump_armed().unwrap_or(f64::NAN));
    ctl.put_u32(st::LEDS, deck.lamps(e, now));
    ctl.put_u32(st::TEMPO_RANGE, deck.tempo_range);
    ctl.put_u32(st::MIDI_STATE, controller.midi_state);
    aux.publish(ctl);
    for (k, row) in rows.iter().enumerate() {
        let at = st::DECKS_STATE + k * st::DECK_STATE_SIZE;
        let (there, (playing, generation, lamps, values)) = match row {
            Some(r) => (1, *r),
            None => (0, (0, 0, 0, [0.0, 1.0, 1.0, f64::NAN, f64::NAN])),
        };
        ctl.put_u32(at, there);
        ctl.put_u32(at + 4, playing);
        ctl.put_u32(at + 8, generation);
        ctl.put_u32(at + 12, lamps);
        for (j, v) in values.iter().enumerate() {
            ctl.put_f64(at + 16 + 8 * j, *v);
        }
    }
    ctl.state_end();
}

/// The GUI process, watched: on Windows through a handle opened at the start (its PID opened again later could
/// be another process's by then), on Unix as this process's parent.
struct Parent {
    #[cfg(windows)]
    handle: windows_sys::Win32::Foundation::HANDLE,
    #[cfg(unix)]
    pid: u32,
}

impl Parent {
    #[cfg(windows)]
    fn watch(pid: u32) -> Parent {
        use windows_sys::Win32::System::Threading::{OpenProcess, PROCESS_SYNCHRONIZE};
        Parent { handle: unsafe { OpenProcess(PROCESS_SYNCHRONIZE, 0, pid) } }
    }

    #[cfg(windows)]
    fn alive(&self) -> bool {
        use windows_sys::Win32::Foundation::WAIT_TIMEOUT;
        use windows_sys::Win32::System::Threading::WaitForSingleObject;
        !self.handle.is_null() && unsafe { WaitForSingleObject(self.handle, 0) } == WAIT_TIMEOUT
    }

    #[cfg(unix)]
    fn watch(pid: u32) -> Parent {
        Parent { pid }
    }

    #[cfg(unix)]
    fn alive(&self) -> bool {
        unsafe { libc::getppid() as u32 == self.pid }
    }
}
