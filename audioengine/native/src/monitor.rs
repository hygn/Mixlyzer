//! The buffer monitor (feeder.py's _observe_buffer): how long a buffer spent at each level while sound was due,
//! for the GUI's meter and histogram. The engine's (its own buffer ahead of the main device) and each output's of
//! a device of its own (the bridge ahead of that device: device.rs Bridge).

use crate::clock;

pub struct Monitor {
    rate: f64,
    obs_time: Option<f64>,
    obs_frames: f64,
    pub hist: Vec<f64>,
    pub hist_bucket_ms: f64, // each histogram bucket's width: 0.1 ms, wider when the buffer needs more buckets
    lows: std::collections::VecDeque<(i64, f64)>, // (second, lowest level in it, ms): the last LOW_WINDOW_SEC
    pub obs_seconds: f64,
    pub obs_level: f64,
    pub buffer_gen: u64,
}

impl Monitor {
    const LOW_WINDOW_SEC: i64 = 10;

    pub fn new(rate: f64) -> Monitor {
        Monitor {
            rate,
            obs_time: None,
            obs_frames: 0.0,
            hist: vec![],
            hist_bucket_ms: 0.1,
            lows: std::collections::VecDeque::new(),
            obs_seconds: 0.0,
            obs_level: 0.0,
            buffer_gen: 0,
        }
    }

    /// A reading of the level (``frames``, of a buffer of ``capacity`` frames) while sound is due; ``None``: not
    /// due (the next reading starts over).
    pub fn observe(&mut self, frames: Option<f64>, capacity: f64) {
        let Some(frames) = frames else {
            self.obs_time = None;
            return;
        };
        let now = clock::now();
        if let Some(t) = self.obs_time {
            self.accumulate(self.obs_frames, frames, now - t, capacity);
        }
        self.obs_time = Some(now);
        self.obs_frames = frames;
    }

    /// The lowest level (ms) over the last LOW_WINDOW_SEC of playing (-1: none).
    pub fn lowest_ms(&self) -> f64 {
        let low = self.lows.iter().map(|(_, low)| *low).fold(f64::INFINITY, f64::min);
        if low.is_finite() { low } else { -1.0 }
    }

    /// The level between two readings: drains at real time from the first until it reaches the second.
    fn accumulate(&mut self, start_frames: f64, end_frames: f64, seconds: f64, capacity: f64) {
        if seconds <= 0.0 {
            return;
        }
        // buckets of 0.1 ms up to twice the capacity (a shared device's own buffer shows above it), wider when that
        // would take more than HIST_MAX
        let span_ms = 2.0 * capacity * 1000.0 / self.rate + 2.0;
        let bucket = (0.1f64).max(span_ms / (crate::ipc::HIST_MAX - 2) as f64);
        let size = ((span_ms / bucket) as usize + 2).min(crate::ipc::HIST_MAX);
        // started again only when the span outgrows the histogram or falls under half of it (a capacity that moves a
        // little, as the devices' mean periods do, keeps it), a quarter of room made each time
        let fits = size <= self.hist.len() && 2 * size >= self.hist.len();
        if !fits || self.hist_bucket_ms != bucket {
            self.hist = vec![0.0; (size + size / 4).min(crate::ipc::HIST_MAX)];
            self.hist_bucket_ms = bucket;
            self.obs_seconds = 0.0;
            self.obs_level = 0.0;
            self.buffer_gen += 1;
        }
        let top = self.hist.len() - 1;
        let a = start_frames * 1000.0 / self.rate;
        let b = (end_frames * 1000.0 / self.rate).max(a - seconds * 1000.0);
        let drop = (a - b).max(0.0);
        let flat = seconds - drop / 1000.0;
        if drop > 0.0 {
            // a ms of level drains in a ms of time: each bucket the drop crosses holds its share
            let mut k = (b / bucket) as usize;
            while k <= (a / bucket) as usize && k <= top {
                let overlap = a.min((k + 1) as f64 * bucket) - b.max(k as f64 * bucket);
                if overlap > 0.0 {
                    self.hist[k] += overlap / 1000.0;
                }
                k += 1;
            }
        }
        if flat > 0.0 {
            self.hist[top.min((b / bucket) as usize)] += flat;
        }
        // the lowest level, per second, kept for the last LOW_WINDOW_SEC
        let second = clock::now().floor() as i64;
        match self.lows.back_mut() {
            Some((s, low)) if *s == second => *low = low.min(b),
            _ => self.lows.push_back((second, b)),
        }
        while self.lows.front().is_some_and(|(s, _)| *s <= second - Self::LOW_WINDOW_SEC) {
            self.lows.pop_front();
        }
        self.obs_level += (a + b) * 0.5 * drop / 1000.0 + b * flat.max(0.0);
        self.obs_seconds += seconds;
    }
}
