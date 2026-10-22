//! Everything rendered into the ring, by output frame (ring index): the track position each run of frames
//! carries, so the frame being heard maps back to the track at the speed it was rendered at.

use std::collections::VecDeque;

#[derive(Clone, Copy, Debug)]
pub struct Segment {
    pub out_start: u64,
    pub frames: u64,
    pub in_start: f64, // input frame at out_start
    pub step: f64,     // input frames per output frame; 0 = silence holding in_start
}

#[derive(Default)]
pub struct Timeline {
    segments: VecDeque<Segment>,
}

impl Timeline {
    pub fn append(&mut self, out_start: u64, frames: u64, in_start: f64, step: f64) {
        if frames > 0 {
            self.segments.push_back(Segment { out_start, frames, in_start, step });
        }
    }

    pub fn segment_at(&self, out_frame: f64) -> Option<&Segment> {
        self.segments.iter().rev().find(|s| s.out_start as f64 <= out_frame)
    }

    pub fn first(&self) -> Option<&Segment> {
        self.segments.front()
    }

    /// The silence queued while paused holds ``in_start``: the paused position moved.
    pub fn hold_silence_at(&mut self, in_start: f64) {
        for s in self.segments.iter_mut() {
            if s.step == 0.0 {
                s.in_start = in_start;
            }
        }
    }

    /// Drops the segments that finished playing, keeping the one being heard.
    pub fn prune(&mut self, heard: f64) {
        while self.segments.len() > 1 && self.segments[1].out_start as f64 <= heard {
            self.segments.pop_front();
        }
    }
}
