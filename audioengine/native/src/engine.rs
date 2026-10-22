//! The player's renderer (audioengine/python/feeder.py's PCMFeeder, its docs hold the design): a decoded track
//! rendered into the device's ring ahead of what is heard, every block recorded on the timeline, so the
//! playhead is the input position actually being heard.
//!
//! Modes: speed (varispeed, or the pitch-preserving time stretch; while scrubbing it plays the original at unity,
//! released it takes the tempo at once) / none (pass-through). With the time stretch off unity a seek or a jump while playing
//! goes through a second stretcher prepared ahead a little each tick while the current one plays on, then
//! the two are crossfaded. Scratching plays the record under the hand instead (the platter, read_at). Clicks
//! and reference tones are mixed in; the music volume is applied here, so the click level stays its own.

use std::sync::Arc;

use crate::clock;
use crate::cursor::CursorPoller;
use crate::device::Ring;
use crate::dsp::{self, Platter, ReadTables};
use crate::mixers::{Clicks, Tones};
use crate::shm::Pcm;
use crate::stretch::Stretcher;
use crate::timeline::Timeline;

#[derive(Clone, Copy, PartialEq)]
enum Standby {
    Seek,
    Jump,
    Resume, // a scrub let go: the tempo's stream from a point ahead (resume_at)
}

pub struct Engine {
    pub rate: f64,
    pub ch: usize,
    ring: Arc<Ring>,
    pub pcm: Option<Pcm>,
    pub track_gen: u32,
    len: usize,

    timeline: Timeline,
    pub clicks: Clicks,
    pub tones: Tones,
    tables: ReadTables,
    platter: Platter,

    stretch: Option<Box<Stretcher>>, // the speed renderer when the time stretch is on (else varispeed)
    standby: Option<Box<Stretcher>>,
    standby_for: Option<(Standby, f64)>,
    pending_seek: Option<f64>,
    // a scrub let go with the time stretch off unity: the original plays on at unity up to this input frame, where
    // the stream prepared for it (Standby::Resume) takes over
    resume_at: Option<f64>,
    prep_cost: f64, // one prepare_step's cost (s), smoothed: the standby's work stops before it would overrun

    pos: f64,
    pub playing: bool,
    scrubbing: bool,
    pub scratching: bool,
    scratch_target: f64,
    poller: Option<CursorPoller>,
    poll_frames_per_px: f64,
    poll_anchor: Option<f64>,
    grab_frame: f64,
    display: Option<(f64, f64)>,
    mode_speed: bool,
    factor: f64,
    bend: f64, // a jog wheel's pitch bend (deck.rs): the speed times 1 + bend
    halted: bool,
    last_step: f64,
    music_gain: f32,
    soft_clip: bool,
    jump_start: Option<f64>,
    jump_dest: Option<f64>,
    jump_once: bool,
    jump_starts: Option<Vec<f64>>,
    jump_dest_base: f64,
    end_out_frame: Option<u64>,
    sent_finished: bool,

    buffer_ms: f64,
    fade_frames: usize,
    chunk_frames: usize,
    idle_queue_frames: usize,
    scratch_queue_frames: usize,

    // the response measurement: when the command being handled was sent (the GUI's clock), and the frame whose
    // hearing answers a play / pause / seek: (ring index, sent at, 0 play | 1 pause | 2 seek)
    pub command_at: f64,
    probe: Option<(u64, f64, usize)>,
    pending_seek_at: f64,
    pub responses: [f64; 3],
    pub jumped_count: u64,
    pub finished_count: u64,
    pub render_ms: f64,

    // the buffer monitor (feeder.py's _observe_buffer): time spent at each 1 ms level
    obs_time: Option<f64>,
    obs_frames: f64,
    pub hist: Vec<f64>,
    pub hist_bucket_ms: f64, // each histogram bucket's width: 0.1 ms, wider when the buffer needs more buckets
    lows: std::collections::VecDeque<(i64, f64)>, // (second, lowest level in it, ms): the last LOW_WINDOW_SEC
    pub obs_seconds: f64,
    pub obs_level: f64,
    pub buffer_gen: u64,
}

impl Engine {
    pub const LATE_JUMP_SEC: f64 = 0.15;
    pub const PEAK_FLOOR_DBFS: f64 = -24.0;
    const SOFT_CLIP_KNEE: f32 = 0.8;
    pub const BUFFER_MS_RANGE: (f64, f64) = (2.0, 500.0);
    const DISPLAY_GAIN: f64 = 0.1;
    const DISPLAY_RESYNC_SEC: f64 = 0.04;
    const STANDBY_BUDGET_SEC: f64 = 0.0015;
    // the standby's work per tick: at most this share of the buffer (a small buffer keeps its ticks short)
    const STANDBY_BUFFER_SHARE: f64 = 0.3;
    // a scrub let go resumes the tempo this far ahead (input at unity): time for its stream to be prepared
    const RESUME_LEAD_SEC: f64 = 0.08;
    const LOW_WINDOW_SEC: i64 = 10;

    pub fn new(rate: u32, ch: usize, ring: Arc<Ring>) -> Engine {
        let r = rate as f64;
        Engine {
            rate: r,
            ch,
            ring,
            pcm: None,
            track_gen: 0,
            len: 0,
            timeline: Timeline::default(),
            clicks: Clicks::new(r, ch),
            tones: Tones::new(r),
            tables: ReadTables::new(),
            platter: Platter::new(r),
            stretch: None,
            standby: None,
            standby_for: None,
            pending_seek: None,
            resume_at: None,
            prep_cost: 0.0005,
            pos: 0.0,
            playing: false,
            scrubbing: false,
            scratching: false,
            scratch_target: 0.0,
            poller: None,
            poll_frames_per_px: 0.0,
            poll_anchor: None,
            grab_frame: 0.0,
            display: None,
            mode_speed: true,
            factor: 1.0,
            bend: 0.0,
            halted: false,
            last_step: 1.0,
            music_gain: 1.0,
            soft_clip: false,
            jump_start: None,
            jump_dest: None,
            jump_once: false,
            jump_starts: None,
            jump_dest_base: 0.0,
            end_out_frame: None,
            sent_finished: false,
            buffer_ms: 40.0,
            fade_frames: (rate as usize / 250).max(1),
            chunk_frames: 2048,
            idle_queue_frames: (rate as usize / 50).max(1),
            scratch_queue_frames: ((r * 0.02) as usize).max(1), // in the ring; the device's buffer comes after
            command_at: 0.0,
            probe: None,
            pending_seek_at: 0.0,
            responses: [0.0; 3],
            jumped_count: 0,
            finished_count: 0,
            render_ms: 0.0,
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

    pub fn ring(&self) -> &Arc<Ring> {
        &self.ring
    }

    // Configuration
    pub fn set_track(&mut self, pcm: Option<Pcm>, generation: u32) {
        self.pause();
        self.len = pcm.as_ref().map(|p| p.frames).unwrap_or(0);
        self.pcm = pcm;
        self.track_gen = generation;
        self.pos = 0.0;
        self.end_out_frame = None;
        self.sent_finished = false;
        self.scratching = false;
        self.poller = None;
        self.jump_start = None;
        self.jump_dest = None;
        self.jump_starts = None;
        self.cancel_standby();
        self.clicks.clear_voices();
        if let Some(s) = self.stretch.as_mut() {
            s.reset();
        }
    }

    pub fn set_mode_speed(&mut self, speed: bool) {
        self.mode_speed = speed;
    }

    pub fn set_timestretch(&mut self, on: bool) {
        if on == self.stretch.is_some() {
            return;
        }
        let rate = self.rate as u32;
        self.stretch = if on { Some(Box::new(Stretcher::new(self.ch, rate))) } else { None };
        self.standby = if on { Some(Box::new(Stretcher::new(self.ch, rate))) } else { None };
        self.cancel_standby();
    }

    pub fn set_buffer_ms(&mut self, ms: f64) {
        let (lo, hi) = Self::BUFFER_MS_RANGE;
        let ms = ms.clamp(lo, hi);
        let shrunk = ms < self.buffer_ms;
        self.buffer_ms = ms;
        // smaller: what is queued beyond it dropped at once (as a pause and play at the frame heard)
        if shrunk && self.playing && !self.scratching {
            self.pause();
            self.play();
        }
    }

    /// The buffer is small enough to need a tick every 1 ms (main.rs).
    pub fn small_buffer(&self) -> bool {
        self.buffer_ms < 10.0
    }

    /// The buffer kept: what the device takes at once (its period) and the setting on top of it, so after each
    /// of its reads the setting's worth is still queued: the engine may be that late to top it up (a buffer of
    /// the setting alone left only setting - period, 2 ms of 5 ms beside a 3 ms period: a thread woken late under
    /// load let the device read an empty ring).
    pub fn buffer_frames(&self) -> usize {
        let asked = (self.rate * self.buffer_ms / 1000.0).round() as usize;
        asked + self.ring.max_frames.load(std::sync::atomic::Ordering::Relaxed) as usize
    }

    pub fn set_factor(&mut self, f: f64) {
        self.factor = if f.is_finite() { f.clamp(0.0, 4.0) } else { 1.0 };
    }

    /// The deck's tempo (deck.rs): 1 exactly plays the original as it is (the mode "none"), else at the speed.
    pub fn set_tempo(&mut self, f: f64) {
        self.set_factor(f);
        self.mode_speed = (self.factor - 1.0).abs() > 1e-9;
    }

    pub fn factor(&self) -> f64 {
        if self.mode_speed { self.factor } else { 1.0 }
    }

    pub fn set_bend(&mut self, bend: f64) {
        self.bend = bend;
    }

    pub fn len_frames(&self) -> usize {
        self.len
    }

    pub fn has_track(&self) -> bool {
        self.pcm.is_some()
    }

    /// The JumpCUE jump armed: its destination (input frame).
    pub fn jump_armed(&self) -> Option<f64> {
        self.jump_dest.map(|_| self.jump_dest_base)
    }

    pub fn set_music_gain(&mut self, g: f64) {
        self.music_gain = g.max(0.0) as f32;
    }

    pub fn set_soft_clip(&mut self, on: bool) {
        self.soft_clip = on;
    }

    /// Scrubbing (the overview dragged): the playhead follows the drag, each position a seek. The time stretch
    /// plays the original at unity meanwhile (a new stream per position, ~10 ms of work each, would not keep up
    /// with the drag). Released, the original plays on at unity for RESUME_LEAD_SEC while the tempo's stream is
    /// prepared from there (a little each tick), and takes over at that very frame, crossfaded: the tempo comes
    /// at once, without ~10 ms of work in one tick (a small buffer runs dry) or a warm-up at unity catching up
    /// with the tempo over a few hundred ms (a tempo change's gentle path).
    pub fn set_scrubbing(&mut self, on: bool) {
        let released = self.scrubbing && !on;
        self.scrubbing = on;
        self.resume_at = None;
        if on {
            self.clicks.clear_voices();
        }
        let tempo = self.tempo();
        if released && self.stretch.is_some() && self.playing && !self.scratching && self.pcm.is_some()
            && tempo > 0.0 && (tempo - 1.0).abs() > 1e-8 && self.pos < self.len as f64
        {
            self.cancel_standby();
            let target = self.pos + Self::RESUME_LEAD_SEC * self.rate;
            if target < self.len as f64 {
                self.resume_at = Some(target);
                self.prepare_standby_at(Standby::Resume, target, tempo);
            } else {
                self.crossfade_to(self.pos);
            }
        }
    }

    /// Grab (true) or release the record. ``frames_per_px``: input frames the record moves per pixel of the
    /// drag (0: the GUI's mouse events are followed, scratch_to).
    pub fn set_scratching(&mut self, on: bool, frames_per_px: f64) {
        if on == self.scratching {
            return;
        }
        if on {
            if self.playing {
                let heard = self.playhead_frame();
                self.ring.flush();
                self.pos = heard.clamp(0.0, self.len as f64);
            }
            self.scratching = true;
            self.scratch_target = self.pos;
            self.grab_frame = self.pos;
            self.poller = if frames_per_px > 0.0 { CursorPoller::start() } else { None };
            self.poll_frames_per_px = if self.poller.is_some() { frames_per_px } else { 0.0 };
            self.poll_anchor = None;
            self.platter.reset(self.pos, clock::now(), self.poll_frames_per_px);
            self.end_out_frame = None;
            self.sent_finished = false;
            self.clicks.clear_voices();
            return;
        }
        self.drain_poller();
        self.poller = None;
        self.scratching = false;
        self.pos = self.scratch_target;
    }

    /// A jog wheel grabbed (true) or let go: scratching that follows scratch_to, the platter ``delay`` late.
    pub fn grab(&mut self, on: bool, delay: f64) {
        self.set_scratching(on, 0.0);
        self.platter.delay = if on { delay } else { Platter::DELAY };
    }

    /// The record held still where it is (a jog wheel stopped under the hand): no carrying on.
    pub fn scratch_rest(&mut self, at: f64) {
        if self.scratching && self.poller.is_none() {
            self.platter.set_target(self.scratch_target, at, true);
        }
    }

    pub fn scratch_target(&self) -> f64 {
        self.scratch_target
    }

    pub fn scratch_to(&mut self, frame: f64, at: f64) {
        if !self.scratching || self.poller.is_some() {
            return;
        }
        self.scratch_target = if self.len > 0 { frame.clamp(0.0, (self.len - 1) as f64) } else { 0.0 };
        self.platter.set_target(self.scratch_target, at, false);
    }

    fn drain_poller(&mut self) {
        let Some(poller) = &self.poller else { return };
        for (at, x, rest) in poller.drain() {
            let anchor = *self.poll_anchor.get_or_insert(x);
            let frame = self.grab_frame - (x - anchor) * self.poll_frames_per_px;
            self.scratch_target = if self.len > 0 { frame.clamp(0.0, (self.len - 1) as f64) } else { 0.0 };
            self.platter.set_target(self.scratch_target, at, rest);
        }
    }

    fn render_scratch(&mut self, n: usize) {
        self.drain_poller();
        let start = self.platter.position;
        let queued = self.ring.written() as f64 - self.ring.heard(clock::now());
        let heard_at = clock::now() + queued.max(0.0) / self.rate;
        let positions = self.platter.positions(n, heard_at);
        let out = match &self.pcm {
            Some(p) => dsp::read_at(&self.tables, p.data(), self.ch, &positions, Some(start)),
            None => vec![0.0; n * self.ch],
        };
        let step = (self.platter.position - start) / n.max(1) as f64;
        self.append(out, start, step, Some(true));
    }

    fn step(&self) -> f64 {
        if self.stretch.is_some() && (self.scrubbing || self.resume_at.is_some()) {
            return 1.0;
        }
        let tempo = if self.mode_speed { self.factor } else { 1.0 };
        if self.bend != 0.0 { tempo * (1.0 + self.bend) } else { tempo }
    }

    /// The deck's tempo (the speed it plays at, unity overrides aside).
    fn tempo(&self) -> f64 {
        if self.mode_speed { self.factor } else { 1.0 }
    }

    // Transport (input frames)
    pub fn at_end(&self) -> bool {
        self.pcm.is_none() || self.pos >= self.len as f64
    }

    pub fn play(&mut self) -> bool {
        if self.playing {
            return true;
        }
        if self.scratching {
            self.playing = true;
            return true;
        }
        if self.pcm.is_none() || self.pos >= self.len as f64 {
            return false;
        }
        self.playing = true;
        self.obs_time = None;
        self.probe = Some((self.ring.written(), self.command_at, 0));
        self.halted = self.step() <= 0.0;
        self.sent_finished = false;
        self.end_out_frame = None;
        let (mut head, step, after) = self.render(self.pos, self.fade_frames, None);
        dsp::fade(&mut head, self.ch, self.fade_frames, true);
        self.append(head, self.pos, step, None);
        self.pos = after;
        if self.pos < self.len as f64 {
            self.render_block(self.chunk_frames.min(self.ring.free() as usize));
        }
        true
    }

    /// Stops at the frame being heard now; what is queued after it is dropped (faded out in a few ms).
    pub fn pause(&mut self) {
        if !self.playing {
            return;
        }
        if self.scratching {
            self.playing = false;
            return;
        }
        let heard = self.playhead_frame();
        self.playing = false;
        self.obs_time = None;
        self.cancel_standby();
        // silent once the frame being read now and its fade have been heard (or what was queued, when less)
        let silent_at = (self.ring.read_index() + self.fade_frames as u64).min(self.ring.written());
        self.probe = Some((silent_at, self.command_at, 1));
        self.ring.flush();
        self.clicks.clear_voices();
        self.pos = heard.clamp(0.0, self.len as f64);
    }

    pub fn stop(&mut self) {
        self.pause();
        self.seek(0.0);
    }

    pub fn seek(&mut self, frame: f64) {
        if self.resume_at.is_some() {
            self.cancel_standby();
        }
        let target = frame.clamp(0.0, self.len as f64);
        self.end_out_frame = None;
        self.sent_finished = false;
        if self.playing && self.pcm.is_some() && self.can_defer() {
            self.prepare_standby(Standby::Seek, target);
            self.pending_seek = Some(target);
            self.pending_seek_at = self.command_at;
        } else if self.playing && self.pcm.is_some() {
            self.cancel_standby();
            self.probe = Some((self.ring.written(), self.command_at, 2));
            self.crossfade_to(target);
        } else {
            self.pos = target;
            self.timeline.hold_silence_at(target);
        }
    }

    /// Jumps from ``start`` to ``dest`` (input frames) whenever playback reaches it, or only the next time
    /// (``once``); with ``starts`` (candidate points) at the first the render head has not passed yet.
    pub fn arm_jump(&mut self, start: Option<f64>, dest: f64, once: bool, starts: Vec<f64>) {
        if !dest.is_finite() || (start.is_none() && starts.is_empty()) {
            self.disarm_jump();
            return;
        }
        if !starts.is_empty() {
            let mut s = starts;
            s.sort_by(f64::total_cmp);
            self.jump_start = Some(s[0]);
            self.jump_starts = Some(s);
        } else {
            self.jump_starts = None;
            self.jump_start = start;
        }
        self.jump_dest = Some(dest);
        self.jump_dest_base = dest;
        self.jump_once = once;
        if self.can_defer() && self.pending_seek.is_none() {
            self.prepare_standby(Standby::Jump, dest);
        }
    }

    pub fn disarm_jump(&mut self) {
        self.jump_start = None;
        self.jump_dest = None;
        self.jump_starts = None;
        if matches!(self.standby_for, Some((Standby::Jump, _))) {
            self.standby_for = None;
        }
    }

    // The standby stretcher
    fn can_defer(&self) -> bool {
        let step = self.step();
        self.standby.is_some()
            && self.playing
            && !self.scratching
            && !self.scrubbing
            && self.pcm.is_some()
            && step > 0.0
            && (step - 1.0).abs() > 1e-8
    }

    fn prepare_standby(&mut self, kind: Standby, target: f64) {
        let step = self.step();
        self.prepare_standby_at(kind, target, step);
    }

    fn prepare_standby_at(&mut self, kind: Standby, target: f64, step: f64) {
        let target = target.clamp(0.0, self.len as f64);
        if self.standby_for != Some((kind, target)) {
            if let (Some(sb), Some(pcm)) = (self.standby.as_mut(), self.pcm.as_ref()) {
                sb.prepare(pcm.data(), target, step);
            }
            self.standby_for = Some((kind, target));
        }
    }

    fn standby_ready(&mut self) -> bool {
        if self.standby_for.is_none() {
            return false;
        }
        // as much of its work as STANDBY_BUDGET_SEC allows each call (a tick): a seek is heard sooner than with a
        // step a tick, and the tick stays short next to the buffer
        let budget = Self::STANDBY_BUDGET_SEC.min(self.buffer_ms / 1000.0 * Self::STANDBY_BUFFER_SHARE);
        let (Some(sb), Some(pcm)) = (self.standby.as_mut(), self.pcm.as_ref()) else { return false };
        let start = clock::now();
        let mut done = 0;
        loop {
            // a step a tick at least; more while the next one (as long as they take) fits the budget
            let elapsed = clock::now() - start;
            if done > 0 && elapsed + self.prep_cost > budget {
                return false;
            }
            let t0 = clock::now();
            let ready = sb.prepare_step(pcm.data(), self.fade_frames);
            self.prep_cost += 0.2 * ((clock::now() - t0) - self.prep_cost);
            done += 1;
            if ready {
                return true;
            }
        }
    }

    fn cancel_standby(&mut self) {
        self.standby_for = None;
        self.pending_seek = None;
        self.resume_at = None;
    }

    /// The scrub's resume point reached (render head at ``target``): the original's tail at unity crossfaded
    /// into the stream prepared there at the tempo (the same frames: no jump).
    fn resume_switch(&mut self, target: f64) {
        let n = self.fade_frames;
        let tail = self.render(self.pos, n, Some(1.0)).0;
        std::mem::swap(&mut self.stretch, &mut self.standby);
        self.standby_for = None;
        self.resume_at = None;
        let (head, step, after) = self.render(target, n, None);
        let mix = self.crossfade(head, tail);
        self.append(mix, target, step, None);
        self.pos = after;
    }

    fn switch_to_standby(&mut self, target: f64) {
        if self.pending_seek.is_some() {
            self.probe = Some((self.ring.written(), self.pending_seek_at, 2));
        }
        let n = self.fade_frames;
        let tail = if self.pos < self.len as f64 { self.render(self.pos, n, None).0 } else { vec![] };
        std::mem::swap(&mut self.stretch, &mut self.standby);
        self.standby_for = None;
        self.pending_seek = None;
        let (head, step, after) = self.render(target, n, None);
        let mix = self.crossfade(head, tail);
        self.append(mix, target, step, None);
        self.pos = after;
    }

    /// ``head`` faded in over ``tail`` faded out.
    fn crossfade(&self, mut head: Vec<f32>, mut tail: Vec<f32>) -> Vec<f32> {
        let ch = self.ch;
        dsp::fade(&mut head, ch, self.fade_frames, true);
        let t = (head.len() / ch).min(tail.len() / ch);
        tail.truncate(t * ch);
        dsp::fade(&mut tail, ch, self.fade_frames, false);
        for (h, v) in head.iter_mut().zip(tail.iter()) {
            *h += v;
        }
        head
    }

    // What is being heard
    /// The input frame being heard (paused: where the deck stands; scratching: where the hand is).
    pub fn playhead_frame(&self) -> f64 {
        if self.scratching {
            return self.scratch_target;
        }
        if !self.playing {
            return self.pos;
        }
        let out_frame = self.ring.heard(clock::now());
        match self.timeline.segment_at(out_frame) {
            None => self.timeline.first().map(|s| s.in_start).unwrap_or(self.pos),
            Some(s) if s.step == 0.0 => s.in_start,
            Some(s) => s.in_start + (out_frame - s.out_start as f64).min(s.frames as f64) * s.step,
        }
    }

    /// playhead_frame for drawing: while the deck plays steadily it runs on its own clock at the speed being
    /// heard, nudged toward the reading (the device's queue drains in whole periods). (frame, frames per s.)
    pub fn display_frame(&mut self, now: f64) -> (f64, f64) {
        let raw = self.playhead_frame();
        let seg = if self.playing && !(self.scratching || self.scrubbing) {
            self.timeline.segment_at(self.ring.heard(now)).copied()
        } else {
            None
        };
        let Some(seg) = seg.filter(|s| s.step > 0.0) else {
            self.display = None;
            return (raw, 0.0);
        };
        let Some((mut frame, at)) = self.display else {
            self.display = Some((raw, now));
            return (raw, seg.step * self.rate);
        };
        frame += (now - at) * seg.step * self.rate;
        let error = raw - frame;
        if error.abs() > Self::DISPLAY_RESYNC_SEC * self.rate * seg.step.max(1.0) {
            frame = raw;
        } else {
            frame += Self::DISPLAY_GAIN * error;
        }
        self.display = Some((frame, now));
        (frame, seg.step * self.rate)
    }

    /// Pre-volume peak of the block being heard.
    pub fn heard_peak_dbfs(&self) -> f64 {
        if !self.playing && !self.scratching {
            return Self::PEAK_FLOOR_DBFS;
        }
        self.timeline.segment_at(self.ring.heard(clock::now())).map(|s| s.peak_dbfs).unwrap_or(Self::PEAK_FLOOR_DBFS)
    }

    // Rendering
    /// (output, input frames per output frame, input position after it) of up to ``n_out`` frames from
    /// ``pos`` at ``step`` (the deck's speed by default); the advance is the renderer's own.
    fn render(&mut self, pos: f64, n_out: usize, step: Option<f64>) -> (Vec<f32>, f64, f64) {
        let step = step.unwrap_or_else(|| self.step());
        let Some(pcm) = self.pcm.as_ref() else { return (vec![], step, pos) };
        if step <= 0.0 {
            return (vec![0.0; n_out * self.ch], 0.0, pos);
        }
        let t0 = clock::now();
        let (out, after) = match self.stretch.as_mut() {
            Some(s) => {
                let out = s.render(pcm.data(), pos, step, n_out);
                (out, s.expected_input_position())
            }
            None => (dsp::varispeed(pcm.data(), self.ch, pos, step, n_out), None),
        };
        self.render_ms += (clock::now() - t0) * 1000.0;
        let n = out.len() / self.ch;
        match after {
            Some(a) if n > 0 => (out, (a - pos) / n as f64, a),
            _ => (out, step, pos + step * n as f64),
        }
    }

    fn input_step(&self, step: f64) -> f64 {
        match &self.stretch {
            Some(s) => s.input_step(step),
            None => step,
        }
    }

    /// Queues ``out``: the music (by default when the record moves forward) from input frame ``in_start``
    /// on, ``step`` frames per output frame.
    fn append(&mut self, mut out: Vec<f32>, in_start: f64, step: f64, music: Option<bool>) {
        let n = out.len() / self.ch;
        if n == 0 {
            return;
        }
        let music = music.unwrap_or(step > 0.0);
        let peak = if music { dsp::peak_dbfs(&out, Self::PEAK_FLOOR_DBFS) } else { Self::PEAK_FLOOR_DBFS };
        if music {
            if self.music_gain != 1.0 {
                out.iter_mut().for_each(|v| *v *= self.music_gain);
            }
            if self.playing && !self.scrubbing && !self.scratching && step > 0.0 {
                self.clicks.queue(in_start, step, n);
            }
        }
        self.clicks.mix(&mut out);
        let ratio = self.tone_pitch_ratio();
        self.tones.mix(&mut out, self.ch, ratio);
        if self.soft_clip {
            dsp::soft_clip(&mut out, Self::SOFT_CLIP_KNEE);
        }
        self.timeline.append(self.ring.written(), n as u64, in_start, step, peak);
        self.ring.push(&out);
    }

    fn tone_pitch_ratio(&self) -> f64 {
        if self.mode_speed && self.stretch.is_none() && self.factor > 0.0 { self.factor } else { 1.0 }
    }

    fn append_silence(&mut self, n: usize) {
        let hold = self.pos.min(self.len as f64);
        self.append(vec![0.0; n * self.ch], hold, 0.0, None);
    }

    fn crossfade_to(&mut self, target: f64) {
        let target = target.clamp(0.0, self.len as f64);
        let n = self.fade_frames;
        let tail = if self.pos < self.len as f64 { self.render(self.pos, n, None).0 } else { vec![] };
        let (head, step, after) = self.render(target, n, None);
        let mix = self.crossfade(head, tail);
        self.append(mix, target, step, None);
        self.pos = after;
    }

    fn render_block(&mut self, mut n_out: usize) {
        let step = self.step();
        if step <= 0.0 {
            if !self.halted {
                // tempo pulled to 0: fade out at the last speed instead of cutting off
                self.halted = true;
                if self.last_step > 0.0 {
                    let (mut tail, tail_step, after) = self.render(self.pos, self.fade_frames, Some(self.last_step));
                    dsp::fade(&mut tail, self.ch, self.fade_frames, false);
                    let k = tail.len() / self.ch;
                    self.append(tail, self.pos, tail_step, None);
                    self.pos = after;
                    n_out = n_out.saturating_sub(k);
                    self.mark_end_if_done();
                }
            }
            if n_out > 0 {
                self.append_silence(n_out);
            }
            return;
        }
        if self.halted {
            self.halted = false;
            let (mut head, head_step, after) = self.render(self.pos, self.fade_frames, None);
            dsp::fade(&mut head, self.ch, self.fade_frames, true);
            let k = head.len() / self.ch;
            self.append(head, self.pos, head_step, None);
            self.pos = after;
            n_out = n_out.saturating_sub(k);
            if self.mark_end_if_done() || n_out == 0 {
                return;
            }
        }
        self.last_step = step;
        if let Some(target) = self.resume_at {
            // the scrub's resume point within this block: up to it at unity, then the stream at the tempo
            if self.pos + n_out as f64 >= target {
                let n1 = (target - self.pos).ceil().max(0.0) as usize;
                if n1 > 0 {
                    let (out, out_step, after) = self.render(self.pos, n1, None);
                    self.append(out, self.pos, out_step, None);
                    self.pos = after;
                }
                if matches!(self.standby_for, Some((Standby::Resume, _))) && self.standby_ready() {
                    self.resume_switch(target); // the frame it was prepared for (the head at it, under a frame)
                } else {
                    // not prepared in time (a busy computer): a lead further on
                    let tempo = self.tempo();
                    let next = self.pos + Self::RESUME_LEAD_SEC * self.rate;
                    self.standby_for = None;
                    self.resume_at = Some(next);
                    self.prepare_standby_at(Standby::Resume, next, tempo);
                }
                return;
            }
        }
        let deferring = self.can_defer();
        if let Some(starts) = &self.jump_starts {
            // the candidate just passed (by less than LATE_JUMP_SEC), taken late with the destination as late;
            // else the first one still ahead (always, through the standby stretcher)
            let late = Self::LATE_JUMP_SEC * self.rate;
            let k = starts.partition_point(|&s| s <= self.pos);
            let next = if k > 0 && self.pos < starts[k - 1] + late && !deferring {
                Some((starts[k - 1] + late, self.jump_dest_base + late))
            } else if k < starts.len() {
                Some((starts[k], self.jump_dest_base))
            } else {
                None
            };
            match next {
                Some((s, d)) => {
                    self.jump_start = Some(s);
                    self.jump_dest = Some(d);
                }
                None => self.disarm_jump(),
            }
        }
        let rate = self.input_step(step);
        if let (Some(js), Some(jd)) = (self.jump_start, self.jump_dest) {
            if self.pos < js && js <= self.pos + rate * n_out as f64 {
                let mut take = true;
                let mut ready = false;
                if deferring {
                    self.prepare_standby(Standby::Jump, jd);
                    ready = self.standby_ready();
                    let later = self.jump_starts.as_ref().is_some_and(|s| s.iter().any(|&v| v > js));
                    if !ready && later {
                        take = false; // not prepared yet: the next candidate point instead
                    }
                }
                if take {
                    let n1 = ((js - self.pos) / rate).ceil() as usize;
                    if n1 > 0 {
                        let (out, out_step, after) = self.render(self.pos, n1, None);
                        self.append(out, self.pos, out_step, None);
                        self.pos = after;
                    }
                    if ready {
                        self.switch_to_standby(jd);
                    } else {
                        self.cancel_standby();
                        self.crossfade_to(jd + (self.pos - js));
                    }
                    if self.jump_once {
                        self.disarm_jump();
                    }
                    self.jumped_count += 1;
                    return;
                }
            }
        }
        let (out, out_step, after) = self.render(self.pos, n_out, None);
        self.append(out, self.pos, out_step, None);
        self.pos = after;
        self.mark_end_if_done();
    }

    fn mark_end_if_done(&mut self) -> bool {
        if self.pos < self.len as f64 {
            return false;
        }
        if self.end_out_frame.is_none() {
            self.pos = self.len as f64;
            self.end_out_frame = Some(self.ring.written());
        }
        true
    }

    // The buffer monitor
    /// Frames rendered and not heard yet (this buffer and the device's).
    fn queued(&self, now: f64) -> f64 {
        (self.ring.written() as f64 - self.ring.heard(now)).max(0.0)
    }

    /// This engine's own buffer: what is queued ahead of the device's fixed latency (WASAPI's shared mode holds
    /// ~32 ms whatever is asked). The device takes it in whole periods, so the ring's own level saws; this one
    /// drains in real time, as the buffer setting means.
    pub fn buffered(&self, now: f64) -> f64 {
        (self.queued(now) - self.ring.latency_frames()).max(0.0)
    }

    /// The lowest level (ms) over the last LOW_WINDOW_SEC of playing (-1: none).
    pub fn lowest_ms(&self) -> f64 {
        let low = self.lows.iter().map(|(_, low)| *low).fold(f64::INFINITY, f64::min);
        if low.is_finite() { low } else { -1.0 }
    }

    pub fn capacity_frames(&self) -> f64 {
        self.buffer_frames() as f64
    }

    fn observe_buffer(&mut self) {
        if !self.playing {
            self.obs_time = None;
            return;
        }
        let now = clock::now();
        let frames = self.buffered(now);
        if let Some(t) = self.obs_time {
            self.accumulate(self.obs_frames, frames, now - t);
        }
        self.obs_time = Some(now);
        self.obs_frames = frames;
    }

    /// The level between two readings: drains at real time from the first until it reaches the second.
    fn accumulate(&mut self, start_frames: f64, end_frames: f64, seconds: f64) {
        if seconds <= 0.0 {
            return;
        }
        // buckets of 0.1 ms up to twice the capacity (a shared device's own buffer shows above it), wider when that
        // would take more than HIST_MAX
        let span_ms = 2.0 * self.capacity_frames() * 1000.0 / self.rate + 2.0;
        let bucket = (0.1f64).max(span_ms / (crate::ipc::HIST_MAX - 2) as f64);
        let size = ((span_ms / bucket) as usize + 2).min(crate::ipc::HIST_MAX);
        if self.hist.len() != size || self.hist_bucket_ms != bucket {
            self.hist = vec![0.0; size];
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

    // The feed: once a tick
    pub fn tick(&mut self) {
        let resuming = matches!(self.standby_for, Some((Standby::Resume, _)));
        if self.standby_for.is_some() && !self.can_defer() && !(resuming && self.playing && !self.scrubbing) {
            self.cancel_standby(); // scrubbing, scratching, back at unity: no longer wanted
        }
        if self.resume_at.is_some() && (!self.playing || (self.tempo() - 1.0).abs() <= 1e-8) {
            self.cancel_standby(); // paused, or the tempo back at unity: nothing to resume to
        }
        // the ring first (a small buffer cannot wait), then the standby stretcher's work (a jump's, a seek's, a
        // scrub's resume: a little each tick, so it is ready when wanted), then the ring again
        self.fill();
        if self.standby_for.is_some() && self.standby_ready() {
            if let Some(target) = self.pending_seek {
                self.switch_to_standby(target);
            }
        }
        self.fill();
        let now = clock::now();
        let heard = self.ring.heard(now);
        self.timeline.prune(heard);
        if let Some((frame, sent_at, kind)) = self.probe {
            // play / seek: the new frame being played out (past it: a report made before it ends exactly there);
            // pause: the last frame heard
            let reached = if kind == 1 { heard >= frame as f64 } else { heard >= frame as f64 + 1.0 };
            if reached {
                if sent_at > 0.0 {
                    self.responses[kind] = (now - sent_at) * 1000.0;
                }
                self.probe = None;
            }
        }
        self.ring.sound_due.store(
            self.playing || self.scratching || self.tones.active(),
            std::sync::atomic::Ordering::Relaxed,
        );
        if self.playing && !self.sent_finished {
            if let Some(end) = self.end_out_frame {
                if heard >= end as f64 {
                    self.sent_finished = true;
                    // the track's end heard: the deck plays on (silence), so the record dragged or the playhead
                    // moved back plays again from there
                    self.finished_count += 1;
                }
            }
        }
    }

    /// Tops the ring up to the buffer (while scratching: a short queue; paused with tones: a short one).
    fn fill(&mut self) {
        for _ in 0..4 {
            self.observe_buffer();
            let free = self.ring.free() as i64;
            let want = if self.scratching {
                // a short queue in the ring (the device's own buffer comes after it), so the record answers the
                // hand promptly
                self.scratch_queue_frames as i64 - self.ring.level() as i64
            } else if self.playing {
                self.buffer_frames() as i64 - self.ring.level() as i64
            } else if self.tones.active() {
                self.idle_queue_frames as i64 - self.ring.level() as i64
            } else {
                0
            };
            let n = want.min(free).min(self.chunk_frames as i64);
            if n <= 0 {
                break;
            }
            let n = n as usize;
            if self.scratching {
                self.render_scratch(n);
            } else if self.playing && self.pos < self.len as f64 {
                self.render_block(n);
            } else {
                self.append_silence(n);
            }
            self.observe_buffer();
        }
    }

    /// Nothing to play and nothing queued: the device may be stopped.
    pub fn idle(&self) -> bool {
        !self.playing && !self.scratching && !self.tones.active() && self.ring.level() == 0
    }
}
