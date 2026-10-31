//! The deck's player (audioengine/python/feeder.py's PCMFeeder, its docs hold the design): a decoded track rendered
//! into the deck's lane ahead of what is heard, every block recorded on the timeline, so the playhead is the input
//! position actually being heard. The player renders the music alone; its mixer channel (mixer.rs: trim, EQ, Sound
//! Color FX, the fader, CUE) is the main device's callback's.
//!
//! Modes: speed (varispeed, or the pitch-preserving time stretch; while scrubbing it plays the original at unity,
//! released it takes the tempo at once) / none (pass-through). With the time stretch off unity a seek or a jump while playing
//! goes through a second stretcher prepared ahead a little each tick while the current one plays on, then
//! the two are crossfaded. Scratching plays the record under the hand instead (the platter, read_at). The listening
//! aids (the metronome's clicks, ducking the music under them, and reference tones) go beside the music in the lane:
//! a lane frame is [music (ch), aids (ch)], the aids added after the channel fader (their level their own).

use std::sync::Arc;

use crate::clock;
use crate::cursor::CursorPoller;
use crate::device::Ring;
use crate::dsp::{self, Platter, ReadTables, SpeedGain};
use crate::mixer::StripParams;
use crate::mixers::{Clicks, Tones};
use crate::monitor::Monitor;
use crate::shm::Pcm;
use crate::keyshift::{self, Pitched};
use crate::timeline::Timeline;

#[derive(Clone, Copy, PartialEq)]
enum Standby {
    Seek,
    Jump,
    Resume, // a scrub let go: the tempo's stream from a point ahead (resume_at)
    // stopped off unity: the stream from where play would start (CUE's preview, PLAY), prepared while stopped, so
    // a start is no stream restart in one tick (~10 ms of work: CUE tapped fast ran a small buffer dry)
    Start,
}

pub struct Engine {
    pub rate: f64,
    pub ch: usize,
    ring: Arc<Ring>,
    pub strip: Arc<StripParams>, // its mixer channel's settings (deck.rs sets them) and meter
    peak: (f32, f64), // the meter's reading held (linear, since when): the strip's peaks taken each tick
    pub pcm: Option<Pcm>,
    pub track_gen: u32,
    len: usize,

    timeline: Timeline,
    pub clicks: Clicks,
    pub tones: Tones,
    tables: ReadTables,
    platter: Platter,
    scratch_gain: SpeedGain, // the record's level by its speed (a still record is silent)

    // the speed renderer when the time stretch is on or the key shifted (else varispeed), its pitch moved by the key
    stretch: Option<Box<Pitched>>,
    standby: Option<Box<Pitched>>,
    standby_for: Option<(Standby, f64)>,
    pending_seek: Option<f64>,
    // a seek while playing (not deferred): crossfaded in by the next block rendered, so the seeks between two
    // blocks are one crossfade, the last (each appended at once, a seek a tick ran the ring over)
    seek_to: Option<f64>,
    // a scrub let go with the time stretch off unity: the original plays on at unity up to this input frame, where
    // the stream prepared for it (Standby::Resume) takes over
    resume_at: Option<f64>,
    prep_cost: f64, // one prepare_step's cost (s), smoothed: the standby's work stops before it would overrun
    start_step: f64, // the step Standby::Start was prepared at (the tempo changed while stopped: prepared again)

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
    master_tempo: bool, // the time stretch on (MT): the pitch kept whatever the tempo
    key_shift: i32,     // semitones (keyshift::KEY_SHIFT_MAX either way): the pitch moved, the tempo kept
    halted: bool,
    last_step: f64,
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

    // the buffer monitor: time spent at each level of this engine's buffer while playing
    pub monitor: Monitor,
}

impl Engine {
    pub const LATE_JUMP_SEC: f64 = 0.15;
    // below the GUI's meter's lowest segment (-30 dB on a DJM-A9's scale: -42 dBFS)
    pub const PEAK_FLOOR_DBFS: f64 = -45.0;
    // the meter holds a peak this long (the GUI's meter reads at its frame rate, the engine takes peaks each tick)
    const PEAK_HOLD_SEC: f64 = 0.05;
    pub const BUFFER_MS_RANGE: (f64, f64) = (2.0, 500.0);
    const DISPLAY_GAIN: f64 = 0.1;
    const DISPLAY_RESYNC_SEC: f64 = 0.04;
    const STANDBY_BUDGET_SEC: f64 = 0.0015;
    // the standby's work per tick: at most this share of the buffer (a small buffer keeps its ticks short)
    const STANDBY_BUFFER_SHARE: f64 = 0.3;
    // a scrub let go resumes the tempo this far ahead (input at unity): time for its stream to be prepared
    const RESUME_LEAD_SEC: f64 = 0.08;

    pub fn new(rate: u32, ch: usize, ring: Arc<Ring>, strip: Arc<StripParams>) -> Engine {
        let r = rate as f64;
        Engine {
            rate: r,
            ch,
            ring,
            strip,
            peak: (0.0, 0.0),
            pcm: None,
            track_gen: 0,
            len: 0,
            timeline: Timeline::default(),
            clicks: Clicks::new(r, ch),
            tones: Tones::new(r),
            tables: ReadTables::new(),
            platter: Platter::new(r),
            scratch_gain: SpeedGain::new(r),
            stretch: None,
            standby: None,
            standby_for: None,
            pending_seek: None,
            seek_to: None,
            resume_at: None,
            prep_cost: 0.0005,
            start_step: 0.0,
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
            master_tempo: false,
            key_shift: 0,
            halted: false,
            last_step: 1.0,
            jump_start: None,
            jump_dest: None,
            jump_once: false,
            jump_starts: None,
            jump_dest_base: 0.0,
            end_out_frame: None,
            sent_finished: false,
            buffer_ms: 40.0,
            fade_frames: (rate as usize / 100).max(1), // 10 ms: start, stop, a seek's crossfade
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
            monitor: Monitor::new(r),
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
        self.seek_to = None;
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
        // the key shift kept: the next track at the same shift (the GUI keeps it in no file: 0 at each start)
    }

    pub fn set_mode_speed(&mut self, speed: bool) {
        self.mode_speed = speed;
    }

    /// The time stretch (MT) on or off.
    pub fn set_timestretch(&mut self, on: bool) {
        self.master_tempo = on;
        self.keep_stretcher();
    }

    /// The key shifted by ``semitones`` (within KEY_SHIFT_MAX either way): the pitch moved by the time stretch,
    /// the tempo as it is; MT off, the pitch the tempo's moved by it.
    pub fn set_key_shift(&mut self, semitones: i32) {
        self.key_shift = semitones.clamp(-keyshift::KEY_SHIFT_MAX, keyshift::KEY_SHIFT_MAX);
        self.keep_stretcher();
    }

    pub fn key_shift(&self) -> i32 {
        self.key_shift
    }

    /// The stretchers there while the time stretch is on or the key shifted (varispeed else).
    fn keep_stretcher(&mut self) {
        let wanted = self.master_tempo || self.key_shift != 0;
        if wanted == self.stretch.is_some() {
            return;
        }
        let rate = self.rate as u32;
        self.stretch = if wanted { Some(Box::new(Pitched::new(self.ch, rate))) } else { None };
        self.standby = if wanted { Some(Box::new(Pitched::new(self.ch, rate))) } else { None };
        self.cancel_standby();
    }

    /// The pitch ratio played at ``step`` (input frames per output frame): the key's, times the speed's with MT off.
    fn pitch_at(&self, step: f64) -> f64 {
        let key = keyshift::key_ratio(self.key_shift);
        if self.master_tempo || !(step > 0.0) { key } else { key * step }
    }

    /// Whether ``step`` is played through the stretcher's stream (off unity in tempo or in pitch).
    fn off_unity(&self, step: f64) -> bool {
        (step - 1.0).abs() > 1e-8 || (self.pitch_at(step) - 1.0).abs() > 1e-8
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
        asked + self.ring.port.max_frames.load(std::sync::atomic::Ordering::Relaxed) as usize
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


    /// Scrubbing (the overview dragged): the playhead follows the drag, each position a seek. The time stretch
    /// plays the original at unity meanwhile (a new stream per position, ~10 ms of work each, would not keep up
    /// with the drag). Released, the original plays on at unity for RESUME_LEAD_SEC while the tempo's stream is
    /// prepared from there (a little each tick), and takes over at that very frame, crossfaded: the tempo comes
    /// at once, without ~10 ms of work in one tick (a small buffer runs dry) or a warm-up at unity catching up
    /// with the tempo over a few hundred ms (a tempo change's gentle path).
    pub fn set_scrubbing(&mut self, on: bool) {
        let released = self.scrubbing && !on;
        if released && self.playing {
            if let Some(target) = self.seek_to.take() {
                self.crossfade_to(target); // the scrub's last seek in now: the resume below starts from it
            }
        }
        self.scrubbing = on;
        self.resume_at = None;
        if on {
            self.clicks.clear_voices();
        }
        let tempo = self.tempo();
        if released && self.stretch.is_some() && self.playing && !self.scratching && self.pcm.is_some()
            && tempo > 0.0 && self.off_unity(tempo) && self.pos < self.len as f64
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
                let sought = self.seek_to.take();
                self.queue_for_fade();
                let heard = self.playhead_frame();
                self.ring.flush();
                self.pos = sought.unwrap_or(heard).clamp(0.0, self.len as f64);
            }
            self.scratching = true;
            self.scratch_target = self.pos;
            self.grab_frame = self.pos;
            self.poller = if frames_per_px > 0.0 { CursorPoller::start() } else { None };
            self.poll_frames_per_px = if self.poller.is_some() { frames_per_px } else { 0.0 };
            self.poll_anchor = None;
            self.platter.reset(self.pos, clock::now(), self.poll_frames_per_px);
            self.scratch_gain.reset();
            self.end_out_frame = None;
            self.sent_finished = false;
            self.clicks.clear_voices();
            return;
        }
        self.drain_poller();
        self.poller = None;
        self.scratching = false;
        self.pos = self.scratch_target;
        if self.playing && self.scratch_gain.level() < 0.5 {
            self.halted = true; // let go of a record held (near) still: playing fades in (render_block)
        }
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
        let mut out = match &self.pcm {
            Some(p) => dsp::read_at(&self.tables, p.data(), self.ch, &positions, Some(start)),
            None => vec![0.0; n * self.ch],
        };
        self.scratch_gain.apply(&mut out, self.ch, &positions, start);
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
        self.take_start_standby();
        self.timeline.hold_until(self.ring.written(), self.pos);
        self.playing = true;
        self.monitor.observe(None, 0.0);
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
        let sought = self.seek_to.take(); // a seek not rendered yet: the deck stops where it was sent
        self.queue_for_fade();
        let heard = self.playhead_frame();
        self.playing = false;
        self.monitor.observe(None, 0.0);
        self.cancel_standby();
        // silent once the frame being read now and its fade have been heard (or what was queued, when less)
        let silent_at = (self.ring.read_index() + self.fade_frames as u64).min(self.ring.written());
        self.probe = Some((silent_at, self.command_at, 1));
        self.ring.flush();
        self.clicks.clear_voices();
        self.pos = sought.unwrap_or(heard).clamp(0.0, self.len as f64);
    }

    /// Before a flush (pause, the record grabbed): a fade's worth and more queued past the frame being read, so
    /// the callback's fade-out runs whole (a small buffer the device has just read holds a few frames only: the
    /// fade cut short clicks).
    fn queue_for_fade(&mut self) {
        let queued = self.ring.written().saturating_sub(self.ring.read_index());
        let want = 2 * self.fade_frames as u64;
        if self.playing && self.pcm.is_some() && self.pos < self.len as f64 && queued < want {
            self.render_block(((want - queued) as usize).min(self.ring.free() as usize));
        }
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
        if self.scratching {
            // the record under the hand put there (the platter starts from it; nothing rendered past the scratch)
            self.scratch_target = if self.len > 0 { target.min((self.len - 1) as f64) } else { 0.0 };
            self.grab_frame = self.scratch_target;
            self.poll_anchor = None;
            self.platter.reset(self.scratch_target, clock::now(), self.poll_frames_per_px);
            return;
        }
        self.end_out_frame = None;
        self.sent_finished = false;
        self.seek_to = None;
        if self.playing && self.pcm.is_some() && self.can_defer() {
            self.prepare_standby(Standby::Seek, target);
            self.pending_seek = Some(target);
            self.pending_seek_at = self.command_at;
        } else if self.playing && self.pcm.is_some() {
            self.cancel_standby();
            self.probe = Some((self.ring.written(), self.command_at, 2));
            self.seek_to = Some(target); // render_block
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
            && self.off_unity(step)
    }

    fn prepare_standby(&mut self, kind: Standby, target: f64) {
        let step = self.step();
        self.prepare_standby_at(kind, target, step);
    }

    fn prepare_standby_at(&mut self, kind: Standby, target: f64, step: f64) {
        let target = target.clamp(0.0, self.len as f64);
        if self.standby_for != Some((kind, target)) {
            let pitch = self.pitch_at(step);
            if let (Some(sb), Some(pcm)) = (self.standby.as_mut(), self.pcm.as_ref()) {
                sb.set_pitch(pitch);
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

    /// Stopped off unity with the time stretch: the stream from where play would start, prepared a little each
    /// tick (Standby::Start); else none wanted.
    fn keep_start_standby(&mut self) {
        let step = self.step();
        let wanted = self.standby.is_some() && !self.playing && !self.scratching && !self.scrubbing
            && self.pcm.is_some() && self.pos < self.len as f64 && step > 0.0 && self.off_unity(step);
        let starting = matches!(self.standby_for, Some((Standby::Start, _)));
        if !wanted {
            if starting {
                self.standby_for = None;
            }
            return;
        }
        if starting && step != self.start_step {
            self.standby_for = None; // the tempo moved: prepared again at the new one
        }
        if self.standby_for.is_none() || starting {
            self.start_step = step;
            self.prepare_standby_at(Standby::Start, self.pos, step);
        }
    }

    /// play(): the stream prepared here while stopped taken (the rest of its work done now: none once it had a
    /// few ticks), so the start renders on in sequence instead of restarting the stream.
    fn take_start_standby(&mut self) {
        let here = self.pos.clamp(0.0, self.len as f64);
        if self.standby_for != Some((Standby::Start, here)) || self.step() != self.start_step {
            if matches!(self.standby_for, Some((Standby::Start, _))) {
                self.standby_for = None;
            }
            return;
        }
        let (Some(sb), Some(pcm)) = (self.standby.as_mut(), self.pcm.as_ref()) else { return };
        while !sb.prepare_step(pcm.data(), self.fade_frames) {}
        std::mem::swap(&mut self.stretch, &mut self.standby);
        self.standby_for = None;
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

    /// The channel's meter (before the fader, after trim and EQ: the mixer's) as dBFS, held PEAK_HOLD_SEC.
    pub fn heard_peak_dbfs(&self) -> f64 {
        let p = self.peak.0 as f64;
        if p > 0.0 { (20.0 * p.log10()).clamp(Self::PEAK_FLOOR_DBFS, 0.0) } else { Self::PEAK_FLOOR_DBFS }
    }

    /// Each tick: the strip's peaks since the last into the meter's reading.
    pub fn take_peak(&mut self) {
        let p = self.strip.take_peak();
        let now = clock::now();
        if p >= self.peak.0 || now - self.peak.1 > Self::PEAK_HOLD_SEC {
            self.peak = (p, now);
        }
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
        // the original's pitch while scrubbing (its unity), else the key's (and the speed's with MT off)
        let pitch = if self.scrubbing || self.resume_at.is_some() { 1.0 } else { self.pitch_at(step) };
        let (out, after) = match self.stretch.as_mut() {
            Some(s) => {
                s.set_pitch(pitch);
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
    /// on, ``step`` frames per output frame; the listening aids beside it (the lane: [music, aids] a frame).
    fn append(&mut self, mut out: Vec<f32>, in_start: f64, step: f64, music: Option<bool>) {
        let ch = self.ch;
        let n = out.len() / ch;
        if n == 0 {
            return;
        }
        let music = music.unwrap_or(step > 0.0);
        if music && self.playing && !self.scrubbing && !self.scratching && step > 0.0 {
            self.clicks.queue(in_start, step, n);
        }
        let mut aids = vec![0f32; out.len()];
        self.clicks.mix(&mut out, &mut aids);
        let ratio = self.tone_pitch_ratio();
        self.tones.mix(&mut aids, ch, ratio);
        let mut lane = vec![0f32; 2 * out.len()];
        for f in 0..n {
            lane[2 * ch * f..2 * ch * f + ch].copy_from_slice(&out[ch * f..ch * (f + 1)]);
            lane[2 * ch * f + ch..2 * ch * (f + 1)].copy_from_slice(&aids[ch * f..ch * (f + 1)]);
        }
        self.timeline.append(self.ring.written(), n as u64, in_start, step);
        self.ring.push(&lane);
    }

    /// The reference tones' pitch: the music's (the key's, and the speed's with MT off).
    fn tone_pitch_ratio(&self) -> f64 {
        let speed = if self.mode_speed && !self.master_tempo && self.factor > 0.0 { self.factor } else { 1.0 };
        keyshift::key_ratio(self.key_shift) * speed
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
        if let Some(target) = self.seek_to.take() {
            if step > 0.0 && !self.halted {
                self.crossfade_to(target); // the block: the crossfade (a fade's length)
                self.mark_end_if_done();
                return;
            }
            self.pos = target.clamp(0.0, self.len as f64); // halted (silent): played from there when it starts again
        }
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

    pub fn capacity_frames(&self) -> f64 {
        self.buffer_frames() as f64
    }

    fn observe_buffer(&mut self) {
        let level = if self.playing { Some(self.buffered(clock::now())) } else { None };
        let capacity = self.capacity_frames();
        self.monitor.observe(level, capacity);
    }

    // The feed: once a tick
    pub fn tick(&mut self) {
        let resuming = matches!(self.standby_for, Some((Standby::Resume, _)));
        let starting = matches!(self.standby_for, Some((Standby::Start, _)));
        if self.standby_for.is_some() && !starting && !self.can_defer() && !(resuming && self.playing && !self.scrubbing) {
            self.cancel_standby(); // scrubbing, scratching, back at unity: no longer wanted
        }
        if self.resume_at.is_some() && (!self.playing || (self.tempo() - 1.0).abs() <= 1e-8) {
            self.cancel_standby(); // paused, or the tempo back at unity: nothing to resume to
        }
        self.keep_start_standby();
        // the ring first (a small buffer cannot wait), then the standby stretcher's work (a jump's, a seek's, a
        // scrub's resume, a start's: a little each tick, so it is ready when wanted), then the ring again
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
        self.ring.sound_due.store(self.sound_due(), std::sync::atomic::Ordering::Relaxed);
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
            } else if self.playing && (self.pos < self.len as f64 || self.seek_to.is_some()) {
                self.render_block(n);
            } else {
                self.append_silence(n);
            }
            self.observe_buffer();
        }
    }

    /// Sound is expected (a dropout counts): playing, scratching, or a tone held.
    pub fn sound_due(&self) -> bool {
        self.playing || self.scratching || self.tones.active()
    }

    /// Nothing to play and nothing queued: the device may be stopped.
    pub fn idle(&self) -> bool {
        !self.playing && !self.scratching && !self.tones.active() && self.ring.level() == 0
    }
}
