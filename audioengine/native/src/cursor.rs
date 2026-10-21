//! Scratch input from the mouse cursor, polled at ~1 kHz on a thread of its own (audioengine/python/cursor_poller.py
//! and _cursor_numba.py, moved here: the cursor is global, so the engine reads it itself). Keeps what is
//! exact about a whole-pixel cursor: the moment it crosses a pixel boundary, and the pixel it stops on.
//! Windows only (Win32 cursor, high-resolution timer); elsewhere scratching follows the GUI's mouse events.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

/// (time s on the shared clock, cursor x in physical px, rest)
pub type CursorEvent = (f64, f64, bool);

pub struct CursorPoller {
    run: Arc<AtomicBool>,
    events: Arc<Mutex<Vec<CursorEvent>>>,
    thread: Option<JoinHandle<()>>,
}

impl CursorPoller {
    pub fn available() -> bool {
        cfg!(windows)
    }

    pub fn start() -> Option<CursorPoller> {
        if !Self::available() {
            return None;
        }
        let run = Arc::new(AtomicBool::new(true));
        let events = Arc::new(Mutex::new(Vec::new()));
        let (r, e) = (run.clone(), events.clone());
        let thread = std::thread::Builder::new().name("ScratchCursorPoller".into()).spawn(move || poll(r, e)).ok()?;
        Some(CursorPoller { run, events, thread: Some(thread) })
    }

    pub fn drain(&self) -> Vec<CursorEvent> {
        std::mem::take(&mut *self.events.lock().unwrap())
    }
}

impl Drop for CursorPoller {
    fn drop(&mut self) {
        self.run.store(false, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

#[cfg(not(windows))]
fn poll(_run: Arc<AtomicBool>, _events: Arc<Mutex<Vec<CursorEvent>>>) {}

#[cfg(windows)]
fn poll(run: Arc<AtomicBool>, events: Arc<Mutex<Vec<CursorEvent>>>) {
    use windows_sys::Win32::Foundation::{CloseHandle, POINT};
    use windows_sys::Win32::System::Threading::{
        CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, CreateWaitableTimerExW, GetCurrentThread, INFINITE,
        SetThreadPriority, SetWaitableTimer, THREAD_PRIORITY_HIGHEST, TIMER_ALL_ACCESS, WaitForSingleObject,
    };
    use windows_sys::Win32::UI::WindowsAndMessaging::GetPhysicalCursorPos;

    const POLL_100NS: i64 = 5000; // relative due time; with the wake-up overhead ~1 ms per poll
    const RATE_SEC: f64 = 0.012;
    const SLOW_RATE: f64 = 50.0;
    // a stop is reported after 1.5 crossings' time at the rate it moved, within these
    const STOP_MIN: f64 = 0.015;
    const STOP_MAX: f64 = 0.05;
    let timer = unsafe {
        CreateWaitableTimerExW(std::ptr::null(), std::ptr::null(), CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS)
    };
    if timer.is_null() {
        return;
    }
    unsafe { SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST) };
    let emit = |at: f64, x: f64, rest: bool| events.lock().unwrap().push((at, x, rest));
    let mut hist: Vec<(f64, f64)> = Vec::with_capacity(64);
    let (mut first, mut last, mut last_dir, mut resting) = (true, 0i32, 0i32, true);
    let (mut prev_t, mut cross_t, mut rate) = (0.0, 0.0, SLOW_RATE);
    while run.load(Ordering::Relaxed) {
        let mut pt = POINT { x: 0, y: 0 };
        unsafe { GetPhysicalCursorPos(&mut pt) };
        let t = crate::clock::now();
        let x = pt.x;
        if first {
            first = false;
            last = x;
            emit(t, x as f64, true);
        } else if x != last {
            let d = if x > last { 1 } else { -1 };
            let b = x as f64 - 0.5 * d as f64;
            let at = 0.5 * (prev_t + t);
            if resting || d != last_dir {
                hist.clear();
            }
            if hist.len() == 64 {
                hist.remove(0);
            }
            hist.push((at, b));
            let mut k = 0;
            while k + 2 < hist.len() && hist[k + 1].0 <= at - RATE_SEC {
                k += 1;
            }
            rate = if hist.len() >= 2 && at > hist[k].0 { (b - hist[k].1).abs() / (at - hist[k].0) } else { SLOW_RATE };
            emit(at, b, false);
            last = x;
            last_dir = d;
            resting = false;
            cross_t = at;
        } else if !resting {
            let stop = (1.5 / rate).clamp(STOP_MIN, STOP_MAX);
            if t - cross_t > stop {
                emit(cross_t + (1.0 / rate).max(0.002).min(stop), x as f64, true);
                resting = true;
            }
        }
        prev_t = t;
        let due = -POLL_100NS;
        unsafe {
            SetWaitableTimer(timer, &due, 0, None, std::ptr::null(), 0);
            WaitForSingleObject(timer, INFINITE);
        }
    }
    unsafe { CloseHandle(timer) };
}
