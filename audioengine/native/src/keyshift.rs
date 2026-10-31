//! The key shift: the time stretch's stream (stretch.rs) at the tempo over the pitch ratio, read at the pitch
//! ratio (dsp::read_at: a Kaiser-windowed sinc, anti-aliased when it reads faster): the tempo as asked, the pitch
//! moved by the ratio. ``Pitched`` is the stretcher with the same calls (render, prepare, prepare_step, the input
//! position); at a pitch of 1 it is the stretcher itself, sample for sample, until the pitch first moves; from then
//! on its stream goes through the reading (at 1 a plain copy where it falls on whole frames) until it starts again
//! (a seek, a jump, a stop: a new stream), so a key shift taken back to 0 while playing does not jump.

use crate::dsp::{self, ReadTables};
use crate::stretch::Stretcher;

/// The key shift's range in semitones (the deck's KEY - / + stop there).
pub const KEY_SHIFT_MAX: i32 = 6;

/// A key shift's pitch ratio (equal temperament).
pub fn key_ratio(semitones: i32) -> f64 {
    2f64.powf(semitones as f64 / 12.0)
}

pub struct Pitched {
    stretcher: Stretcher,
    ch: usize,
    pitch: f64,
    shifting: bool,
    // the stretched frames read from (interleaved), the read position in them, and where each chunk of them came
    // from: (its first frame here, its input position, input frames per frame)
    fifo: Vec<f32>,
    read: f64,
    chunks: Vec<(f64, f64, f64)>,
    next_input: f64, // the stretcher's input position after the fifo's last frame
    ended: bool,     // the stretcher gave nothing more: the source's end
    expected: Option<f64>,
    tables: ReadTables,
}

impl Pitched {
    // frames kept around the read position: the reader's reach at the fastest reading (read_at's REACH)
    const MARGIN: usize = 36;
    const POSITION_EPS: f64 = 1e-4;

    pub fn new(channels: usize, sample_rate: u32) -> Pitched {
        Pitched {
            stretcher: Stretcher::new(channels, sample_rate),
            ch: channels,
            pitch: 1.0,
            shifting: false,
            fifo: Vec::new(),
            read: 0.0,
            chunks: Vec::new(),
            next_input: 0.0,
            ended: false,
            expected: None,
            tables: ReadTables::new(),
        }
    }

    /// The pitch ratio of what is rendered from now on (1: as the stretcher plays it).
    pub fn set_pitch(&mut self, ratio: f64) {
        self.pitch = if ratio.is_finite() && ratio > 0.0 { ratio } else { 1.0 };
    }

    pub fn expected_input_position(&self) -> Option<f64> {
        if self.shifting { self.expected } else { self.stretcher.expected_input_position() }
    }

    /// Input frames per output frame the next call advances.
    pub fn input_step(&self, factor: f64) -> f64 {
        if self.shifting || self.pitch != 1.0 {
            self.stretcher.input_step(factor / self.pitch) * self.pitch
        } else {
            self.stretcher.input_step(factor)
        }
    }

    pub fn reset(&mut self) {
        self.stretcher.reset();
        self.restart(0.0);
        self.shifting = false;
        self.expected = None;
    }

    /// The reading begun anew from input ``pos`` (nothing stretched yet).
    fn restart(&mut self, pos: f64) {
        self.fifo.clear();
        self.chunks.clear();
        self.read = 0.0;
        self.next_input = pos;
        self.ended = false;
    }

    /// A stream from input ``pos`` to be read later (Stretcher::prepare).
    pub fn prepare(&mut self, source: &[f32], pos: f64, factor: f64) {
        self.shifting = self.pitch != 1.0;
        self.restart(pos);
        self.expected = Some(pos);
        self.stretcher.prepare(source, pos, if self.shifting { factor / self.pitch } else { factor });
    }

    /// Some of a prepared stream's work (Stretcher::prepare_step): true once its first ``count`` output frames can
    /// be read without more.
    pub fn prepare_step(&mut self, source: &[f32], count: usize) -> bool {
        let count = if self.shifting { (count as f64 * self.pitch).ceil() as usize + Self::MARGIN } else { count };
        self.stretcher.prepare_step(source, count)
    }

    /// Up to ``max_out`` frames from input ``pos`` at ``factor`` input frames per output frame, the pitch moved by
    /// the ratio set (Stretcher::render's contract).
    pub fn render(&mut self, source: &[f32], pos: f64, factor: f64, max_out: usize) -> Vec<f32> {
        if !self.shifting {
            if self.pitch == 1.0 {
                return self.stretcher.render(source, pos, factor, max_out);
            }
            self.shifting = true; // the stretcher's stream goes on (in sequence), read from here on
            self.restart(pos);
            self.expected = Some(pos);
        } else if self.expected.is_none_or(|e| (e - pos).abs() > Self::POSITION_EPS) {
            self.restart(pos); // not where the last call ended: a new stream
        }
        let frames = source.len() / self.ch;
        if max_out == 0 || !(factor > 0.0) || pos >= frames as f64 {
            return Vec::new();
        }
        let ch = self.ch;
        let p = self.pitch;
        let step = factor / p; // the stretched stream's input frames per frame
        let need = (self.read + max_out as f64 * p).ceil() as usize + Self::MARGIN;
        while !self.ended && self.fifo.len() / ch < need {
            let want = need - self.fifo.len() / ch;
            let at = self.next_input;
            let out = self.stretcher.render(source, at, step, want);
            let n = out.len() / ch;
            if n == 0 {
                self.ended = true;
                break;
            }
            let after = self.stretcher.expected_input_position().unwrap_or(at + step * n as f64);
            self.chunks.push(((self.fifo.len() / ch) as f64, at, (after - at) / n as f64));
            self.fifo.extend_from_slice(&out);
            self.next_input = after;
        }
        let have = (self.fifo.len() / ch) as f64;
        // what can be read: all asked while more is stretched; at the source's end up to its last frame
        let last = if self.ended { have - 1.0 } else { have - Self::MARGIN as f64 };
        if last < self.read {
            return Vec::new();
        }
        let n = max_out.min(((last - self.read) / p).floor() as usize + 1);
        let positions: Vec<f64> = (0..n).map(|k| self.read + p * k as f64).collect();
        let out = dsp::read_at(&self.tables, &self.fifo, ch, &positions, Some(self.read - p));
        self.read += p * n as f64;
        self.expected = Some(self.input_at(self.read));
        self.compact();
        out
    }

    /// The input position of a (fractional) frame of the stretched stream.
    fn input_at(&self, frame: f64) -> f64 {
        match self.chunks.iter().rev().find(|c| c.0 <= frame) {
            Some(&(first, input, step)) => input + (frame - first) * step,
            None => self.next_input,
        }
    }

    /// The frames read past (but the reader's reach) dropped.
    fn compact(&mut self) {
        let drop = (self.read.floor() as usize).saturating_sub(Self::MARGIN);
        if drop < 4096 {
            return;
        }
        self.fifo.drain(..drop * self.ch);
        self.read -= drop as f64;
        let d = drop as f64;
        for c in self.chunks.iter_mut() {
            c.0 -= d;
        }
        // the chunks wholly before the read position but the last of them
        let keep = self.chunks.iter().rposition(|c| c.0 <= self.read - Self::MARGIN as f64).unwrap_or(0);
        self.chunks.drain(..keep);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(rate: f64, hz: f64, secs: f64) -> Vec<f32> {
        let n = (rate * secs) as usize;
        (0..n).flat_map(|i| {
            let v = (0.3 * (2.0 * std::f64::consts::PI * hz * i as f64 / rate).sin()) as f32;
            [v, v]
        }).collect()
    }

    /// The strongest frequency of the left channel of ``x`` (frames from..to), by a fine DFT about ``around``.
    fn pitch_of(x: &[f32], rate: f64, from: usize, to: usize, around: f64) -> f64 {
        let mut best = (0.0, 0.0);
        let mut f = around * 0.8;
        while f < around * 1.25 {
            let (mut re, mut im) = (0.0, 0.0);
            for i in from..to {
                let w = 2.0 * std::f64::consts::PI * f * i as f64 / rate;
                re += x[2 * i] as f64 * w.cos();
                im += x[2 * i] as f64 * w.sin();
            }
            let m = re * re + im * im;
            if m > best.0 {
                best = (m, f);
            }
            f += 0.25;
        }
        best.1
    }

    fn play(p: &mut Pitched, src: &[f32], factor: f64, blocks: usize) -> (Vec<f32>, f64) {
        let mut pos = 0.0;
        let mut out = Vec::new();
        for _ in 0..blocks {
            let r = p.render(src, pos, factor, 480);
            if r.is_empty() {
                break;
            }
            pos = p.expected_input_position().unwrap();
            out.extend_from_slice(&r);
        }
        (out, pos)
    }

    #[test]
    fn a_semitone_up_at_the_same_tempo() {
        let rate = 48000.0;
        let src = tone(rate, 440.0, 6.0);
        for (semitones, factor) in [(1, 1.0), (-3, 1.0), (2, 0.8), (-6, 1.25)] {
            let mut p = Pitched::new(2, 48000);
            p.set_pitch(key_ratio(semitones));
            let (out, pos) = play(&mut p, &src, factor, 400);
            let frames = out.len() / 2;
            // the tempo: input consumed at the factor
            assert!((pos / frames as f64 - factor).abs() < 0.01, "{semitones} {factor}: {}", pos / frames as f64);
            let want = 440.0 * key_ratio(semitones);
            let got = pitch_of(&out, rate, frames / 3, frames / 3 + 9600, want);
            assert!((got / want - 1.0).abs() < 0.003, "{semitones} {factor}: {got} Hz, not {want}");
        }
    }

    #[test]
    fn at_a_pitch_of_one_it_is_the_stretcher() {
        let src = tone(48000.0, 440.0, 3.0);
        let mut p = Pitched::new(2, 48000);
        let mut s = Stretcher::new(2, 48000);
        let (a, _) = play(&mut p, &src, 0.8, 100);
        let mut pos = 0.0;
        let mut b = Vec::new();
        for _ in 0..100 {
            let r = s.render(&src, pos, 0.8, 480);
            if r.is_empty() {
                break;
            }
            pos = s.expected_input_position().unwrap();
            b.extend_from_slice(&r);
        }
        assert_eq!(a, b);
    }

    #[test]
    fn shifted_and_back_without_a_jump() {
        // the key shifted while playing and taken back: the reading goes on, no step in the input position (the
        // stretcher's gentle path off unity, a warm-up at unity caught up after, moves a block's advance a little)
        let src = tone(48000.0, 440.0, 4.0);
        let mut p = Pitched::new(2, 48000);
        let mut pos = 0.0;
        for k in 0..200 {
            p.set_pitch(if (50..120).contains(&k) { key_ratio(2) } else { 1.0 });
            let r = p.render(&src, pos, 1.0, 480);
            assert_eq!(r.len(), 960, "block {k}");
            let next = p.expected_input_position().unwrap();
            assert!((next - pos - 480.0).abs() < 0.2 * 480.0, "block {k}: {}", next - pos);
            pos = next;
        }
        assert!((pos / (200.0 * 480.0) - 1.0).abs() < 0.01, "{}", pos / (200.0 * 480.0));
    }
}
