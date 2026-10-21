//! The clock both processes read: seconds of Python's time.perf_counter() (QueryPerformanceCounter on
//! Windows, CLOCK_MONOTONIC on Linux, mach_absolute_time on macOS), so times cross the process boundary as
//! they are (the scratch hand's samples, the displayed playhead's extrapolation).

#[cfg(windows)]
pub fn now() -> f64 {
    use std::sync::OnceLock;
    use windows_sys::Win32::System::Performance::{QueryPerformanceCounter, QueryPerformanceFrequency};
    static FREQ: OnceLock<f64> = OnceLock::new();
    let freq = *FREQ.get_or_init(|| {
        let mut f = 0i64;
        unsafe { QueryPerformanceFrequency(&mut f) };
        f.max(1) as f64
    });
    let mut c = 0i64;
    unsafe { QueryPerformanceCounter(&mut c) };
    c as f64 / freq
}

#[cfg(unix)]
pub fn now() -> f64 {
    // macOS: the uptime clock, mach_absolute_time's (what perf_counter reads there)
    #[cfg(target_os = "macos")]
    const CLOCK: libc::clockid_t = libc::CLOCK_UPTIME_RAW;
    #[cfg(not(target_os = "macos"))]
    const CLOCK: libc::clockid_t = libc::CLOCK_MONOTONIC;
    let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    unsafe { libc::clock_gettime(CLOCK, &mut ts) };
    ts.tv_sec as f64 + ts.tv_nsec as f64 * 1e-9
}
