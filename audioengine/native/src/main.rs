//! Mixlyzer's audio engine: a process of its own, so playback never waits for the GUI's Python (its GIL,
//! its garbage collector, its busy frames). The GUI process (audioengine/native/client.py) starts it with the
//! name of a shared memory block it made; commands come in and the engine's state goes out through it
//! (ipc.rs), each track's PCM through a block of its own. The engine exits when told to or when the GUI
//! process is gone.
//!
//! mixlyzer-audio --shm NAME --size BYTES --parent PID [--device NAME]

mod clock;
mod cursor;
mod device;
mod dsp;
mod engine;
mod ipc;
mod mixers;
mod shm;
mod stretch;
mod timeline;
mod vk;

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use engine::Engine;
use ipc::{Control, Reader, op, st};

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
    let parent = arg(&args, "--parent").and_then(|s| s.parse::<u32>().ok());
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
    let realtime = set_realtime_priority();
    eprintln!("[AudioEngine] VOLK {} ({machine}), real-time thread: {realtime}", volk::volk_version());
    // the output device asked for (None: the system's default); the session's format is its own
    let mut wanted_device: Option<String> = arg(&args, "--device").filter(|n| !n.is_empty());
    let format = device::default_format(wanted_device.as_deref());
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
    // the ring holds the largest output buffer and what a flush leaves behind it
    let ring = Arc::new(device::Ring::new(rate as u64 * 5 / 2, ch, rate as f64));
    let mut current_device = String::new();
    let mut output = match device::open(ring.clone(), rate, ch, wanted_device.as_deref()) {
        Ok((o, name)) => {
            current_device = name;
            Some(o)
        }
        Err(e) => {
            eprintln!("[AudioEngine] device open failed: {e}");
            None
        }
    };
    let mut engine = Engine::new(rate, ch, ring.clone());
    let mut cmd_done = 0u64;
    let mut device_errors = 0u32;
    let mut reopen_at = 0.0;
    let mut idle_since: Option<f64> = None;
    let mut parent_checked = 0.0;
    ctl.state_begin();
    ctl.put_u32(st::RATE, rate);
    ctl.put_u32(st::CHANNELS, ch as u32);
    ctl.put_u32(st::READY, 1);
    ctl.state_end();

    loop {
        let mut batch = Vec::new();
        while let Some(cmd) = ctl.next_command() {
            batch.push(cmd);
        }
        for (i, cmd) in batch.iter().enumerate() {
            cmd_done = cmd.seq;
            if cmd.op == op::SHUTDOWN {
                engine.pause();
                publish(&ctl, &mut engine, cmd_done, device_errors);
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
                op::LIST_DEVICES => ctl.put_devices(&current_device, &device::list()),
                _ if !superseded => handle(&mut engine, cmd.op, &cmd.payload),
                _ => {}
            }
        }
        let now = clock::now();
        // the device: started again when it failed (device loss; at most once a second), stopped while idle
        if ring.failed.load(Ordering::Relaxed) || output.is_none() {
            if now >= reopen_at {
                reopen_at = now + 1.0;
                output = None;
                ring.failed.store(false, Ordering::Relaxed);
                match device::open(ring.clone(), rate, ch, wanted_device.as_deref()) {
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
        if let Some(out) = output.as_mut() {
            if engine.idle() {
                let since = *idle_since.get_or_insert(now);
                if out.running && now - since > 1.0 {
                    use cpal::traits::StreamTrait;
                    let _ = out.stream.pause();
                    out.running = false;
                }
            } else {
                idle_since = None;
                if !out.running {
                    use cpal::traits::StreamTrait;
                    let _ = out.stream.play();
                    out.running = true;
                }
            }
        }
        engine.tick();
        publish(&ctl, &mut engine, cmd_done, device_errors);
        if now - parent_checked > 0.5 {
            parent_checked = now;
            if let Some(pid) = parent {
                if !parent_alive(pid) {
                    return;
                }
            }
        }
        std::thread::sleep(Duration::from_micros(if engine.scratching { 1000 } else { 2000 }));
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

fn handle(e: &mut Engine, op_code: u32, payload: &[u8]) {
    let mut r = Reader::new(payload);
    match op_code {
        op::LOAD => {
            let (Some(generation), Some(channels), Some(frames), Some(name)) = (r.u32(), r.u32(), r.u64(), r.str()) else {
                return;
            };
            if channels as usize != e.ch {
                eprintln!("[AudioEngine] track has {channels} channels, the device {}", e.ch);
                e.set_track(None, generation);
                return;
            }
            match shm::Pcm::open(&name, frames as usize, channels as usize) {
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
            let starts: Vec<f64> = (0..n).filter_map(|_| r.f64()).collect();
            e.arm_jump(if start.is_finite() { Some(start) } else { None }, dest, once != 0, starts);
        }
        op::DISARM_JUMP => e.disarm_jump(),
        op::SET_MODE => e.set_mode_speed(r.u32().unwrap_or(1) != 0),
        op::SET_FACTOR => e.set_factor(r.f64().unwrap_or(1.0)),
        op::SET_TIMESTRETCH => e.set_timestretch(r.u32().unwrap_or(0) != 0),
        op::SET_SOFT_CLIP => e.set_soft_clip(r.u32().unwrap_or(0) != 0),
        op::SET_BUFFER_MS => e.set_buffer_ms(r.f64().unwrap_or(40.0)),
        op::TONE_HOLD => {
            let n = r.u32().unwrap_or(0);
            let notes: Vec<i32> = (0..n).filter_map(|_| r.i32()).collect();
            e.tones.hold(notes);
        }
        op::SET_MUSIC_GAIN => e.set_music_gain(r.f64().unwrap_or(1.0)),
        op::CLICK_SAMPLE => {
            let (frames, channels) = (r.u32().unwrap_or(0) as usize, r.u32().unwrap_or(0) as usize);
            if frames == 0 || channels == 0 {
                e.clicks.set_sample(None);
            } else {
                let data: Vec<f32> = (0..frames * channels).filter_map(|_| r.f32()).collect();
                e.clicks.set_sample(Some((data, channels)));
            }
        }
        op::CLICK_PITCHES => {
            let (b, a) = (r.f64().unwrap_or(0.0), r.f64().unwrap_or(0.0));
            e.clicks.set_pitches(b, a);
        }
        op::CLICK_BEATS => {
            let n = r.u32().unwrap_or(0) as usize;
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

fn publish(ctl: &Control, e: &mut Engine, cmd_done: u64, device_errors: u32) {
    let now = clock::now();
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
    ctl.put_u64(st::BUFFER_GEN, e.buffer_gen);
    ctl.put_f64(st::OBS_SECONDS, e.obs_seconds);
    ctl.put_f64(st::OBS_LEVEL, e.obs_level);
    ctl.put_u64(st::UNDERRUNS, ring.underruns.load(Ordering::Relaxed));
    ctl.put_u32(st::DEVICE_ERRORS, device_errors);
    ctl.put_f64(st::STRETCH_MS, e.render_ms);
    ctl.put_f64(st::LATENCY_MS, ring.latency_frames() / e.rate * 1000.0);
    ctl.put_hist(&e.hist);
    ctl.state_end();
}

#[cfg(windows)]
fn set_realtime_priority() -> bool {
    use windows_sys::Win32::System::Threading::{
        AvSetMmThreadCharacteristicsW, GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_TIME_CRITICAL,
    };
    let task: Vec<u16> = "Pro Audio".encode_utf16().chain(std::iter::once(0)).collect();
    let mut index = 0u32;
    let mmcss = unsafe { !AvSetMmThreadCharacteristicsW(task.as_ptr(), &mut index).is_null() };
    let prio = unsafe { SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL) != 0 };
    mmcss || prio
}

#[cfg(target_os = "macos")]
fn set_realtime_priority() -> bool {
    unsafe { libc::pthread_set_qos_class_self_np(libc::qos_class_t::QOS_CLASS_USER_INTERACTIVE, 0) == 0 }
}

#[cfg(all(unix, not(target_os = "macos")))]
fn set_realtime_priority() -> bool {
    // SCHED_FIFO needs rtprio (audio group / rtkit); without it the thread stays as it is
    let param = libc::sched_param { sched_priority: 20 };
    unsafe { libc::sched_setscheduler(0, libc::SCHED_FIFO, &param) == 0 }
}

#[cfg(windows)]
fn parent_alive(pid: u32) -> bool {
    use windows_sys::Win32::Foundation::{CloseHandle, WAIT_TIMEOUT};
    use windows_sys::Win32::System::Threading::{OpenProcess, PROCESS_SYNCHRONIZE, WaitForSingleObject};
    unsafe {
        let h = OpenProcess(PROCESS_SYNCHRONIZE, 0, pid);
        if h.is_null() {
            return false;
        }
        let alive = WaitForSingleObject(h, 0) == WAIT_TIMEOUT;
        CloseHandle(h);
        alive
    }
}

#[cfg(unix)]
fn parent_alive(pid: u32) -> bool {
    unsafe { libc::getppid() as u32 == pid }
}
