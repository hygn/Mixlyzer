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

    pub jumped_count: u64,
    pub finished_count: u64,
    pub render_ms: f64,

    // the buffer monitor (feeder.py's _observe_buffer): time spent at each 1 ms level
    obs_time: Option<f64>,
    obs_frames: f64,
    pub hist: Vec<f64>,
    pub obs_seconds: f64,
    pub obs_level: f64,
    pub buffer_gen: u64,
}

impl Engine {
    pub const LATE_JUMP_SEC: f64 = 0.15;
    pub const PEAK_FLOOR_DBFS: f64 = -24.0;
    const SOFT_CLIP_KNEE: f32 = 0.8;
    pub const BUFFER_MS_RANGE: (f64, f64) = (10.0, 500.0);
    const DISPLAY_GAIN: f64 = 0.1;
    const DISPLAY_RESYNC_SEC: f64 = 0.04;
    const STANDBY_BUDGET_SEC: f64 = 0.0015;

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
            jumped_count: 0,
            finished_count: 0,
            render_ms: 0.0,
            obs_time: None,
            obs_frames: 0.0,
            hist: vec![],
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

    pub fn buffer_frames(&self) -> usize {
        (self.rate * self.buffer_ms / 1000.0).round() as usize
    }

    pub fn set_factor(&mut self, f: f64) {
        self.factor = if f.is_finite() { f.clamp(0.0, 4.0) } else { 1.0 };
    }

    pub fn set_music_gain(&mut self, g: f64) {
        self.music_gain = g.max(0.0) as f32;
    }

    pub fn set_soft_clip(&mut self, on: bool) {
        self.soft_clip = on;
    }

    /// Scrubbing (the overview dragged): the playhead follows the drag, each position a seek. The time stretch
    /// plays the original at unity meanwhile (a new stream per position, ~10 ms of work each, would not keep up
    /// with the drag); released, its stream starts at the tempo at once, crossfaded from the original, instead of
    /// warming up at unity and catching up with the tempo over a few hundred ms (a tempo change's gentle path).
    pub fn set_scrubbing(&mut self, on: bool) {
        let released = self.scrubbing && !on;
        self.scrubbing = on;
        if on {
            self.clicks.clear_voices();
        }
        let step = self.step();
        if released && self.stretch.is_some() && self.playing && !self.scratching && self.pcm.is_some()
            && step > 0.0 && (step - 1.0).abs() > 1e-8 && self.pos < self.len as f64
        {
            self.cancel_standby();
            self.crossfade_to(self.pos);
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
        if !self.mode_speed || (self.scrubbing && self.stretch.is_some()) {
            return 1.0;
        }
        self.factor
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
        self.ring.flush();
        self.clicks.clear_voices();
        self.pos = heard.clamp(0.0, self.len as f64);
    }

    pub fn stop(&mut self) {
        self.pause();
        self.seek(0.0);
    }

    pub fn seek(&mut self, frame: f64) {
        let target = frame.clamp(0.0, self.len as f64);
        self.end_out_frame = None;
        self.sent_finished = false;
        if self.playing && self.pcm.is_some() && self.can_defer() {
            self.prepare_standby(Standby::Seek, target);
            self.pending_seek = Some(target);
        } else if self.playing && self.pcm.is_some() {
            self.cancel_standby();
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
            s.sort_by(|a, b| a.partial_cmp(b).unwrap());
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
        let target = target.clamp(0.0, self.len as f64);
        if self.standby_for != Some((kind, target)) {
            let step = self.step();
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
        let (Some(sb), Some(pcm)) = (self.standby.as_mut(), self.pcm.as_ref()) else { return false };
        let until = clock::now() + Self::STANDBY_BUDGET_SEC;
        loop {
            if sb.prepare_step(pcm.data(), self.fade_frames) {
                return true;
            }
            if clock::now() >= until {
                return false;
            }
        }
    }

    fn cancel_standby(&mut self) {
        self.standby_for = None;
        self.pending_seek = None;
    }

    fn switch_to_standby(&mut self, target: f64) {
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
        let size = (self.capacity_frames() * 1000.0 / self.rate) as usize + 2;
        let size = size.min(crate::ipc::HIST_MAX);
        if self.hist.len() != size {
            self.hist = vec![0.0; size];
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
            let mut k = b as usize;
            while k <= a as usize && k <= top {
                let overlap = a.min(k as f64 + 1.0) - b.max(k as f64);
                if overlap > 0.0 {
                    self.hist[k] += overlap / 1000.0;
                }
                k += 1;
            }
        }
        if flat > 0.0 {
            self.hist[top.min(b as usize)] += flat;
        }
        self.obs_level += (a + b) * 0.5 * drop / 1000.0 + b * flat.max(0.0);
        self.obs_seconds += seconds;
    }

    // The feed: once a tick
    pub fn tick(&mut self) {
        if self.standby_for.is_some() && !self.can_defer() {
            self.cancel_standby(); // scrubbing, scratching, back at unity: no longer wanted
        }
        // the standby stretcher's work goes on every tick (a jump's as a seek's), so it is ready when wanted
        if self.standby_for.is_some() && self.standby_ready() {
            if let Some(target) = self.pending_seek {
                self.switch_to_standby(target);
            }
        }
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
        let heard = self.ring.heard(clock::now());
        self.timeline.prune(heard);
        self.ring.sound_due.store(
            self.playing || self.scratching || self.tones.active(),
            std::sync::atomic::Ordering::Relaxed,
        );
        if self.playing && !self.sent_finished {
            if let Some(end) = self.end_out_frame {
                if heard >= end as f64 {
                    self.sent_finished = true;
                    self.finished_count += 1;
                }
            }
        }
    }

    /// Nothing to play and nothing queued: the device may be stopped.
    pub fn idle(&self) -> bool {
        !self.playing && !self.scratching && !self.tones.active() && self.ring.level() == 0
    }
}
