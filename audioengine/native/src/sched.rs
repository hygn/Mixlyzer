//! The scheduling of the engine's threads (the engine thread and the device's output thread): Settings >
//! Playback's process priority and power throttling (op::SET_PROCESS), on each system as it has them, and what
//! was got (published as st::THREADS / st::POWER, shown under the buffer meter).
//!
//! Windows: the engine thread and the engine's own WASAPI thread are MMCSS Pro Audio (always); the setting is the
//! process's priority class and its power throttling (EcoQoS, timer resolution).
//! Linux: per thread (the engine's and the ALSA output's; JACK's is the server's, real-time): High real-time
//! (SCHED_FIFO; else through rtkit), else a lower nice (-10, Above normal -5; else through rtkit's
//! MakeThreadHighPriority), Normal SCHED_OTHER at nice 0. Power throttling off: a 1 ns timer slack and
//! uclamp's minimum at the top (schedutil keeps the clock up while the thread runs).
//! macOS: the engine thread High a time-constraint (real-time) thread, Above normal QoS user-interactive, Normal
//! QoS default (CoreAudio's output thread is its own, real-time); power throttling off: an NSProcessInfo activity
//! (latency critical: no App Nap, no timer coalescing).

#![cfg_attr(not(target_os = "linux"), allow(dead_code))] // each system uses some of the states

use std::sync::atomic::Ordering;

use crate::device::Port;

// a thread's state (st::THREADS: the engine's in bits 0-7, the output's in bits 8-15)
pub const API_OWN: u32 = 0; // the audio API's own thread (not the engine's to change) / not known yet
pub const NORMAL: u32 = 1;
pub const RAISED: u32 = 2; // a lower nice / a higher QoS
pub const RAISED_RTKIT: u32 = 3;
pub const REALTIME: u32 = 4; // SCHED_FIFO / MMCSS / time-constraint
pub const REALTIME_RTKIT: u32 = 5;
pub const REFUSED: u32 = 6; // raising it was not permitted

// power throttling (st::POWER)
pub const POWER_LEFT: u32 = 0; // left to the system
pub const POWER_OFF: u32 = 1;
pub const POWER_PARTLY: u32 = 2; // some of it could not be turned off (Linux: uclamp)
pub const POWER_REFUSED: u32 = 3;

#[cfg_attr(target_os = "linux", allow(dead_code))] // Linux: the match's other arm
/// A thread's state (THREADS) as the console says it.
pub fn state_name(state: u32) -> &'static str {
    match state {
        NORMAL => "normal",
        RAISED if cfg!(windows) => "above normal",
        RAISED if cfg!(target_os = "macos") => "QoS user-interactive",
        RAISED => "nice -10",
        RAISED_RTKIT => "raised (rtkit)",
        REALTIME if cfg!(windows) => "MMCSS Pro Audio",
        REALTIME if cfg!(target_os = "macos") => "time-constraint",
        REALTIME => "SCHED_FIFO",
        REALTIME_RTKIT => "real-time (rtkit)",
        REFUSED => "not raised (refused)",
        _ => "the audio API's own",
    }
}

/// Power throttling's state (POWER) as the console says it.
pub fn power_name(state: u32) -> &'static str {
    match state {
        POWER_OFF => "off",
        POWER_PARTLY => "partly off",
        POWER_REFUSED => "not turned off (refused)",
        _ => "left to the system",
    }
}

pub const PRIORITY_NORMAL: u32 = 0;
pub const PRIORITY_ABOVE_NORMAL: u32 = 1;
pub const PRIORITY_HIGH: u32 = 2;

pub struct Sched {
    priority: u32,
    throttling: bool,
    asked: bool, // a setting arrived (until then the thread stays as it started)
    engine: u32,
    output: u32,
    power: u32,
    #[cfg(target_os = "linux")]
    linux: linux::State,
    #[cfg(target_os = "macos")]
    activity: Option<objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_foundation::NSObjectProtocol>>>,
}

impl Sched {
    /// On the engine thread, at the start: what every setting keeps (Windows: MMCSS and the 1 ms timer; macOS:
    /// QoS user-interactive until the setting arrives).
    pub fn start() -> Sched {
        let engine = start_engine_thread();
        Sched {
            priority: PRIORITY_HIGH,
            throttling: false,
            asked: false,
            engine,
            output: API_OWN,
            power: POWER_LEFT,
            #[cfg(target_os = "linux")]
            linux: linux::State::default(),
            #[cfg(target_os = "macos")]
            activity: None,
        }
    }

    pub fn engine_state(&self) -> u32 {
        self.engine
    }

    pub fn power_state(&self) -> u32 {
        self.power
    }

    /// The output thread's state: its backend's (Windows' own WASAPI thread) or the engine's doing (Linux).
    pub fn output_state(&self, port: &Port) -> u32 {
        let own = port.output_state.load(Ordering::Relaxed);
        if own != API_OWN { own } else { self.output }
    }

    /// The setting (op::SET_PROCESS), on the engine thread: applied to the process, this thread and the output
    /// thread known.
    pub fn set(&mut self, priority: u32, throttling: bool, port: &Port) {
        self.priority = priority.min(PRIORITY_HIGH);
        self.throttling = throttling;
        self.asked = true;
        self.apply(port);
        log_line!("priority {}, power throttling {}: engine thread {}, output thread {}, power throttling {}",
            ["normal", "above normal", "high"][self.priority as usize],
            if self.throttling { "the system's choice" } else { "off" }, state_name(self.engine),
            state_name(self.output_state(port)), power_name(self.power));
    }

    /// Each tick: an output thread not seen before (the device opened again) gets the setting.
    pub fn follow_output(&mut self, port: &Port) {
        #[cfg(target_os = "linux")]
        if self.asked {
            let tid = port.output_tid.load(Ordering::Relaxed);
            if tid != 0 && tid != self.linux.output_tid {
                self.output = self.linux.output(tid, self.priority, self.throttling, port);
                self.power = self.linux.power();
            }
        }
        #[cfg(not(target_os = "linux"))]
        let _ = port;
    }

    #[cfg(windows)]
    fn apply(&mut self, _port: &Port) {
        use windows_sys::Win32::System::Threading::{
            ABOVE_NORMAL_PRIORITY_CLASS, GetCurrentProcess, HIGH_PRIORITY_CLASS, NORMAL_PRIORITY_CLASS,
            PROCESS_POWER_THROTTLING_CURRENT_VERSION, PROCESS_POWER_THROTTLING_EXECUTION_SPEED,
            PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION, PROCESS_POWER_THROTTLING_STATE,
            ProcessPowerThrottling, SetPriorityClass, SetProcessInformation,
        };
        let class = match self.priority {
            PRIORITY_NORMAL => NORMAL_PRIORITY_CLASS,
            PRIORITY_ABOVE_NORMAL => ABOVE_NORMAL_PRIORITY_CLASS,
            _ => HIGH_PRIORITY_CLASS,
        };
        let controlled = PROCESS_POWER_THROTTLING_EXECUTION_SPEED | PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION;
        let state = PROCESS_POWER_THROTTLING_STATE {
            Version: PROCESS_POWER_THROTTLING_CURRENT_VERSION,
            ControlMask: if self.throttling { 0 } else { controlled }, // controlled and off, or left to the system
            StateMask: 0,
        };
        let power_ok = unsafe {
            SetPriorityClass(GetCurrentProcess(), class);
            SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling,
                &state as *const _ as *const core::ffi::c_void, std::mem::size_of_val(&state) as u32) != 0
        };
        self.power = match (self.throttling, power_ok) {
            (true, _) => POWER_LEFT,
            (false, true) => POWER_OFF,
            (false, false) => POWER_REFUSED,
        };
    }

    #[cfg(target_os = "linux")]
    fn apply(&mut self, port: &Port) {
        self.engine = self.linux.engine(self.priority, self.throttling);
        // the output thread takes its timer slack itself (only a thread's own can be set)
        port.output_slack.store(if self.throttling { 1 } else { 2 }, Ordering::Relaxed);
        let tid = port.output_tid.load(Ordering::Relaxed);
        if tid != 0 {
            self.output = self.linux.output(tid, self.priority, self.throttling, port);
        }
        self.power = self.linux.power();
    }

    #[cfg(target_os = "macos")]
    fn apply(&mut self, _port: &Port) {
        self.engine = mac::engine_thread(self.priority);
        if self.throttling {
            if let Some(activity) = self.activity.take() {
                mac::end_activity(&activity);
            }
            self.power = POWER_LEFT;
        } else {
            if self.activity.is_none() {
                self.activity = mac::begin_activity();
            }
            self.power = if self.activity.is_some() { POWER_OFF } else { POWER_REFUSED };
        }
    }

    #[cfg(not(any(windows, target_os = "linux", target_os = "macos")))]
    fn apply(&mut self, _port: &Port) {}
}

#[cfg(windows)]
fn start_engine_thread() -> u32 {
    use windows_sys::Win32::System::Threading::{
        AvSetMmThreadCharacteristicsW, GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_TIME_CRITICAL,
    };
    let task: Vec<u16> = "Pro Audio".encode_utf16().chain(std::iter::once(0)).collect();
    let mut index = 0u32;
    let mmcss = unsafe { !AvSetMmThreadCharacteristicsW(task.as_ptr(), &mut index).is_null() };
    let prio = unsafe { SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL) != 0 };
    // the 1 ms timer for this process (per process since Windows 10 2004): the engine thread's 1 ms park timeout
    unsafe { windows_sys::Win32::Media::timeBeginPeriod(1) };
    if mmcss || prio { REALTIME } else { NORMAL }
}

#[cfg(target_os = "macos")]
fn start_engine_thread() -> u32 {
    if mac::qos(libc::qos_class_t::QOS_CLASS_USER_INTERACTIVE) { RAISED } else { NORMAL }
}

#[cfg(not(any(windows, target_os = "macos")))]
fn start_engine_thread() -> u32 {
    NORMAL
}

/// In the output callback (Linux): its thread id for the engine thread, and the timer slack it was asked for.
#[cfg(target_os = "linux")]
pub fn output_thread(port: &Port) {
    use std::cell::Cell;
    thread_local! {
        static TID: Cell<u64> = const { Cell::new(0) };
        static SLACK: Cell<u64> = const { Cell::new(0) };
    }
    let tid = TID.with(|t| {
        if t.get() == 0 {
            t.set(linux::gettid());
        }
        t.get()
    });
    if port.output_tid.load(Ordering::Relaxed) != tid {
        port.output_tid.store(tid, Ordering::Relaxed);
    }
    let wanted = port.output_slack.load(Ordering::Relaxed);
    SLACK.with(|s| {
        if wanted != 0 && s.get() != wanted {
            s.set(wanted);
            linux::timer_slack(wanted == 1);
        }
    });
}

#[cfg(target_os = "linux")]
mod linux {
    use super::*;

    // real-time priorities: the output thread above the engine thread that feeds it (rtkit allows 20 by default)
    const RT_OUTPUT: i32 = 20;
    const NICE_HIGH: i32 = -10;
    const NICE_ABOVE: i32 = -5;
    // SIGXCPU's soft limit under rtkit's RLIMIT_RTTIME (its hard limit): the thread falls back to SCHED_OTHER
    const RTTIME_SOFT_SHARE: f64 = 0.9;

    pub fn gettid() -> u64 {
        unsafe { libc::syscall(libc::SYS_gettid) as u64 }
    }

    /// The calling thread's timer slack: 1 ns (on), or the default.
    pub fn timer_slack(on: bool) -> bool {
        unsafe { libc::prctl(libc::PR_SET_TIMERSLACK, if on { 1 } else { 0 } as libc::c_ulong, 0, 0, 0) == 0 }
    }

    #[derive(Default)]
    pub struct State {
        pub output_tid: u64,
        output_ours: bool, // the output thread's scheduling is the engine's (not the audio API's real-time)
        rtkit: Option<Option<RtKit>>, // None: not tried yet
        sigxcpu: bool,
        uclamp_ok: bool,
        slack_ok: bool,
        throttling: bool,
    }

    impl State {
        pub fn engine(&mut self, priority: u32, throttling: bool) -> u32 {
            let tid = gettid();
            let state = self.thread(tid, priority, true, RT_OUTPUT - 1);
            self.throttling = throttling;
            self.slack_ok = timer_slack(!throttling);
            self.uclamp_ok = uclamp(tid, !throttling);
            state
        }

        pub fn output(&mut self, tid: u64, priority: u32, throttling: bool, _port: &Port) -> u32 {
            if tid != self.output_tid {
                // a new output thread: the audio API's own when it is already real-time (JACK's)
                self.output_tid = tid;
                self.output_ours = !is_realtime(tid);
            }
            if !self.output_ours {
                return API_OWN;
            }
            let state = self.thread(tid, priority, false, RT_OUTPUT);
            self.uclamp_ok &= uclamp(tid, !throttling);
            state
        }

        pub fn power(&self) -> u32 {
            match (self.throttling, self.slack_ok, self.uclamp_ok) {
                (true, ..) => POWER_LEFT,
                (false, true, true) => POWER_OFF,
                (false, true, false) => POWER_PARTLY,
                (false, false, _) => POWER_REFUSED,
            }
        }

        fn rtkit(&mut self) -> Option<&RtKit> {
            self.rtkit.get_or_insert_with(RtKit::connect).as_ref()
        }

        fn thread(&mut self, tid: u64, priority: u32, _engine: bool, rt_priority: i32) -> u32 {
            match priority {
                PRIORITY_HIGH => {
                    if set_fifo(tid, rt_priority) {
                        self.catch_sigxcpu();
                        return REALTIME;
                    }
                    if let Some(rt) = self.rtkit() {
                        let prio = rt_priority.min(rt.max_priority).max(1);
                        if rt.limit_rttime() && rt.realtime(tid, prio) {
                            self.catch_sigxcpu();
                            return REALTIME_RTKIT;
                        }
                    }
                    self.nice(tid, NICE_HIGH)
                }
                PRIORITY_ABOVE_NORMAL => {
                    set_other(tid);
                    self.nice(tid, NICE_ABOVE)
                }
                _ => {
                    set_other(tid);
                    set_nice(tid, 0);
                    NORMAL
                }
            }
        }

        fn nice(&mut self, tid: u64, nice: i32) -> u32 {
            if set_nice(tid, nice) {
                return RAISED;
            }
            if let Some(rt) = self.rtkit() {
                if rt.high_priority(tid, nice.max(rt.min_nice)) {
                    return RAISED_RTKIT;
                }
            }
            REFUSED
        }

        /// SIGXCPU (a real-time thread over its RLIMIT_RTTIME soft limit, rtkit's condition): that thread back
        /// to SCHED_OTHER instead of the process killed.
        fn catch_sigxcpu(&mut self) {
            if self.sigxcpu {
                return;
            }
            self.sigxcpu = true;
            extern "C" fn demote(_sig: libc::c_int) {
                let param = libc::sched_param { sched_priority: 0 };
                unsafe { libc::sched_setscheduler(0, libc::SCHED_OTHER, &param) };
            }
            unsafe {
                let mut action: libc::sigaction = std::mem::zeroed();
                action.sa_sigaction = demote as extern "C" fn(libc::c_int) as usize;
                libc::sigemptyset(&mut action.sa_mask);
                libc::sigaction(libc::SIGXCPU, &action, std::ptr::null_mut());
            }
        }
    }

    fn is_realtime(tid: u64) -> bool {
        let policy = unsafe { libc::sched_getscheduler(tid as libc::pid_t) } & !libc::SCHED_RESET_ON_FORK;
        policy == libc::SCHED_FIFO || policy == libc::SCHED_RR
    }

    fn set_fifo(tid: u64, priority: i32) -> bool {
        let param = libc::sched_param { sched_priority: priority };
        unsafe { libc::sched_setscheduler(tid as libc::pid_t, libc::SCHED_FIFO | libc::SCHED_RESET_ON_FORK, &param) == 0 }
    }

    fn set_other(tid: u64) {
        if is_realtime(tid) {
            let param = libc::sched_param { sched_priority: 0 };
            unsafe { libc::sched_setscheduler(tid as libc::pid_t, libc::SCHED_OTHER, &param) };
        }
    }

    /// A thread's nice (Linux's nice is per thread): lowering it needs RLIMIT_NICE or CAP_SYS_NICE.
    fn set_nice(tid: u64, nice: i32) -> bool {
        unsafe { libc::setpriority(libc::PRIO_PROCESS, tid as libc::id_t, nice) == 0 }
    }

    #[repr(C)]
    struct SchedAttr {
        size: u32,
        policy: u32,
        flags: u64,
        nice: i32,
        priority: u32,
        runtime: u64,
        deadline: u64,
        period: u64,
        util_min: u32,
        util_max: u32,
    }

    /// uclamp's minimum (Linux 5.3+, CONFIG_UCLAMP_TASK): at the top, schedutil runs the CPU at full clock while
    /// the thread runs (and a hybrid CPU puts it on a big core); 0, the system's choice. Policy and parameters
    /// kept (passed as they are, for kernels before KEEP_PARAMS read them back).
    fn uclamp(tid: u64, on: bool) -> bool {
        const KEEP_POLICY: u64 = 0x08;
        const KEEP_PARAMS: u64 = 0x10;
        const UTIL_CLAMP_MIN: u64 = 0x20;
        let pid = tid as libc::pid_t;
        let mut param = libc::sched_param { sched_priority: 0 };
        unsafe {
            *libc::__errno_location() = 0;
            let nice = libc::getpriority(libc::PRIO_PROCESS, tid as libc::id_t);
            if *libc::__errno_location() != 0 || libc::sched_getparam(pid, &mut param) != 0 {
                return false;
            }
            let attr = SchedAttr {
                size: std::mem::size_of::<SchedAttr>() as u32,
                policy: (libc::sched_getscheduler(pid) & !libc::SCHED_RESET_ON_FORK) as u32,
                flags: KEEP_POLICY | KEEP_PARAMS | UTIL_CLAMP_MIN,
                nice,
                priority: param.sched_priority as u32,
                runtime: 0,
                deadline: 0,
                period: 0,
                util_min: if on { 1024 } else { 0 },
                util_max: 1024,
            };
            libc::syscall(libc::SYS_sched_setattr, pid, &attr as *const SchedAttr, 0u32) == 0
        }
    }

    /// rtkit (org.freedesktop.RealtimeKit1 on the system bus): real-time or a lower nice for a thread of this
    /// process, within its limits, for a user who may not set them.
    struct RtKit {
        proxy: zbus::blocking::Proxy<'static>,
        max_priority: i32,
        min_nice: i32,
        rttime_max_us: i64,
    }

    impl RtKit {
        fn connect() -> Option<RtKit> {
            let conn = zbus::blocking::Connection::system().ok()?;
            let proxy: zbus::blocking::Proxy<'static> = zbus::blocking::proxy::Builder::new(&conn)
                .destination("org.freedesktop.RealtimeKit1").ok()?
                .path("/org/freedesktop/RealtimeKit1").ok()?
                .interface("org.freedesktop.RealtimeKit1").ok()?
                .cache_properties(zbus::proxy::CacheProperties::No)
                .build().ok()?;
            let max_priority = proxy.get_property::<i32>("MaxRealtimePriority").unwrap_or(0);
            let min_nice = proxy.get_property::<i32>("MinNiceLevel").unwrap_or(0);
            let rttime_max_us = proxy.get_property::<i64>("RTTimeUSecMax").unwrap_or(0);
            Some(RtKit { proxy, max_priority, min_nice, rttime_max_us })
        }

        /// rtkit grants real-time only to a process whose RLIMIT_RTTIME is within its RTTimeUSecMax: set so
        /// (a lower limit cannot be raised again; SIGXCPU at the soft limit demotes the thread).
        fn limit_rttime(&self) -> bool {
            if self.max_priority < 1 || self.rttime_max_us <= 0 {
                return false;
            }
            let max = self.rttime_max_us as libc::rlim_t;
            unsafe {
                let mut now: libc::rlimit = std::mem::zeroed();
                if libc::getrlimit(libc::RLIMIT_RTTIME, &mut now) == 0 && now.rlim_max != libc::RLIM_INFINITY && now.rlim_max <= max {
                    return true;
                }
                let limit = libc::rlimit { rlim_cur: (max as f64 * RTTIME_SOFT_SHARE) as libc::rlim_t, rlim_max: max };
                libc::setrlimit(libc::RLIMIT_RTTIME, &limit) == 0
            }
        }

        fn realtime(&self, tid: u64, priority: i32) -> bool {
            self.proxy.call_method("MakeThreadRealtime", &(tid, priority as u32)).is_ok()
        }

        fn high_priority(&self, tid: u64, nice: i32) -> bool {
            self.proxy.call_method("MakeThreadHighPriority", &(tid, nice)).is_ok()
        }
    }
}

#[cfg(target_os = "macos")]
mod mac {
    use super::*;
    use objc2::rc::Retained;
    use objc2::runtime::ProtocolObject;
    use objc2_foundation::{NSActivityOptions, NSObjectProtocol, NSProcessInfo, NSString};

    // the time constraint: woken every 2 ms (the engine's tick), up to 1 ms of work each
    const PERIOD_NS: f64 = 2.0e6;
    const COMPUTATION_NS: f64 = 1.0e6;

    pub fn qos(class: libc::qos_class_t) -> bool {
        unsafe { libc::pthread_set_qos_class_self_np(class, 0) == 0 }
    }

    fn port() -> libc::mach_port_t {
        unsafe { libc::pthread_mach_thread_np(libc::pthread_self()) }
    }

    #[allow(deprecated)] // libc points to mach2 for the timebase; the call is the same
    fn time_constraint() -> bool {
        let mut base = libc::mach_timebase_info { numer: 0, denom: 0 };
        unsafe { libc::mach_timebase_info(&mut base) };
        if base.numer == 0 {
            return false;
        }
        let abs = |ns: f64| (ns * base.denom as f64 / base.numer as f64) as u32;
        let mut policy = libc::thread_time_constraint_policy {
            period: abs(PERIOD_NS),
            computation: abs(COMPUTATION_NS),
            constraint: abs(PERIOD_NS),
            preemptible: 1,
        };
        unsafe {
            libc::thread_policy_set(port(), libc::THREAD_TIME_CONSTRAINT_POLICY as libc::thread_policy_flavor_t,
                &mut policy as *mut _ as libc::thread_policy_t, libc::THREAD_TIME_CONSTRAINT_POLICY_COUNT)
                == libc::KERN_SUCCESS
        }
    }

    /// Back to a timesharing thread (from a time constraint), so its QoS class can be set.
    fn timeshare() {
        let mut policy = libc::thread_extended_policy { timeshare: 1 };
        unsafe {
            libc::thread_policy_set(port(), libc::THREAD_EXTENDED_POLICY as libc::thread_policy_flavor_t,
                &mut policy as *mut _ as libc::thread_policy_t, libc::THREAD_EXTENDED_POLICY_COUNT);
        }
    }

    pub fn engine_thread(priority: u32) -> u32 {
        if priority == PRIORITY_HIGH && time_constraint() {
            return REALTIME;
        }
        timeshare();
        if priority == PRIORITY_NORMAL {
            qos(libc::qos_class_t::QOS_CLASS_DEFAULT);
            return NORMAL;
        }
        if qos(libc::qos_class_t::QOS_CLASS_USER_INTERACTIVE) { RAISED } else { REFUSED }
    }

    pub type Activity = Retained<ProtocolObject<dyn NSObjectProtocol>>;

    /// An activity of this process (until ended): latency critical (no timer coalescing), user initiated (no App
    /// Nap), the system still free to sleep when idle.
    pub fn begin_activity() -> Option<Activity> {
        let options = NSActivityOptions::UserInitiatedAllowingIdleSystemSleep | NSActivityOptions::LatencyCritical;
        let reason = NSString::from_str("Audio playback");
        Some(NSProcessInfo::processInfo().beginActivityWithOptions_reason(options, &reason))
    }

    pub fn end_activity(activity: &Activity) {
        unsafe { NSProcessInfo::processInfo().endActivity(activity) };
    }
}
