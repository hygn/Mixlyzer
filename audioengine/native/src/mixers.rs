//! What is mixed into the music: the metronome's clicks and the reference tones (audioengine/python/clicks.py,
//! audioengine/python/tones.py).

/// Metronome clicks: a click starts at the output frame where the render head reaches its beat and then
/// plays at the device rate whatever the tempo. With ducking, the music under a click is lowered by up to
/// DUCK_DB, following the click's envelope scaled by its gain.
pub struct Clicks {
    rate: f64,
    ch: usize,
    raw: Option<(Vec<f32>, usize)>, // [L, raw channels]
    speeds: (f64, f64),
    sample: Option<Click>,
    accent: Option<Click>,
    ducking: bool,
    beats: Vec<f64>, // input frames, sorted
    gains: Vec<f32>,
    accents: Vec<bool>,
    enabled: bool,
    offset: f64,
    voices: Vec<Voice>,
}

#[derive(Clone)]
struct Click {
    data: Vec<f32>, // [L, ch]
    env: Vec<f32>,
}

struct Voice {
    p: i64,
    gain: f32,
    accent: bool,
}

impl Clicks {
    const MAX_PER_BLOCK: usize = 4;
    const DUCK_DB: f64 = -6.0;
    const DUCK_RELEASE_MS: f64 = 60.0;

    pub fn new(rate: f64, ch: usize) -> Clicks {
        Clicks {
            rate,
            ch,
            raw: None,
            speeds: (1.0, 1.0),
            sample: None,
            accent: None,
            ducking: false,
            beats: vec![],
            gains: vec![],
            accents: vec![],
            enabled: false,
            offset: 0.0,
            voices: vec![],
        }
    }

    /// The click as [frames, channels] at the device rate (None disables).
    pub fn set_sample(&mut self, sample: Option<(Vec<f32>, usize)>) {
        self.raw = sample;
        self.rebuild();
    }

    pub fn set_pitches(&mut self, beat_semitones: f64, accent_semitones: f64) {
        let speeds = (2f64.powf(beat_semitones / 12.0), 2f64.powf(accent_semitones / 12.0));
        if speeds != self.speeds {
            self.speeds = speeds;
            self.rebuild();
        }
    }

    fn rebuild(&mut self) {
        self.sample = None;
        self.accent = None;
        self.voices.clear();
        let Some((raw, rch)) = &self.raw else { return };
        let rch = (*rch).max(1);
        let len = raw.len() / rch;
        if len == 0 {
            return;
        }
        let mut variants = vec![];
        for speed in [self.speeds.0, self.speeds.1] {
            let n = ((len as f64 / speed) as usize).max(1);
            let mut s = vec![0f32; n * self.ch];
            for i in 0..n {
                let j = ((i as f64 * speed) as usize).min(len - 1);
                for c in 0..self.ch {
                    // the device's channels: the click's own, or its mean on every one
                    s[i * self.ch + c] = if rch == self.ch {
                        raw[j * rch + c]
                    } else {
                        raw[j * rch..(j + 1) * rch].iter().sum::<f32>() / rch as f32
                    };
                }
            }
            variants.push(self.with_envelope(s));
        }
        self.accent = variants.pop();
        self.sample = variants.pop();
    }

    /// The ducking envelope (peak 1, instant attack, exponential release to -60 dB), the sample zero-padded
    /// to its length.
    fn with_envelope(&self, sample: Vec<f32>) -> Click {
        let n = sample.len() / self.ch;
        let level: Vec<f64> =
            (0..n).map(|i| sample[i * self.ch..(i + 1) * self.ch].iter().fold(0f32, |m, v| m.max(v.abs())) as f64).collect();
        let peak = level.iter().cloned().fold(0.0, f64::max);
        if peak <= 0.0 {
            return Click { env: vec![0.0; n], data: sample };
        }
        let decay = (-1.0 / (Self::DUCK_RELEASE_MS * 1e-3 * self.rate)).exp();
        let tail = (1e-3f64.ln() / decay.ln()).ceil() as usize;
        let mut env = vec![0f32; n + tail];
        let mut e = 0f64;
        for i in 0..env.len() {
            e *= decay;
            if i < n && level[i] / peak > e {
                e = level[i] / peak;
            }
            env[i] = e as f32;
        }
        let mut data = vec![0f32; env.len() * self.ch];
        data[..sample.len()].copy_from_slice(&sample);
        Click { data, env }
    }

    pub fn set_ducking(&mut self, on: bool) {
        self.ducking = on;
    }

    pub fn set_beats(&mut self, beats_sec: &[f64], gains: Vec<f32>, accents: Vec<bool>) {
        self.beats = beats_sec.iter().map(|b| b * self.rate).collect();
        self.gains = gains;
        self.accents = accents;
    }

    pub fn set_enabled(&mut self, on: bool) {
        self.enabled = on;
        if !on {
            self.voices.clear();
        }
    }

    pub fn set_offset_sec(&mut self, sec: f64) {
        self.offset = sec * self.rate;
    }

    pub fn clear_voices(&mut self) {
        self.voices.clear();
    }

    /// Starts a click at the output frame where each beat inside this block is reached.
    pub fn queue(&mut self, in_start: f64, step: f64, n: usize) {
        if !self.enabled || self.sample.is_none() || self.beats.is_empty() {
            return;
        }
        let lo = in_start + self.offset;
        let hi = in_start + step * n as f64 + self.offset;
        let i0 = self.beats.partition_point(|&b| b < lo);
        let i1 = self.beats.partition_point(|&b| b < hi);
        for i in i0.max(i1.saturating_sub(Self::MAX_PER_BLOCK))..i1 {
            let k = ((self.beats[i] - lo) / step).ceil() as i64;
            let accent = self.accents.get(i).copied().unwrap_or(false);
            self.voices.push(Voice { p: -k, gain: self.gains.get(i).copied().unwrap_or(0.0), accent });
        }
    }

    /// The sounding clicks of a block about to be written into ``aids``, ducking ``music`` under them.
    pub fn mix(&mut self, music: &mut [f32], aids: &mut [f32]) {
        if self.voices.is_empty() {
            return;
        }
        let (Some(beat), Some(accent)) = (&self.sample, &self.accent) else {
            self.voices.clear();
            return;
        };
        let ch = self.ch;
        let n = (music.len() / ch) as i64;
        if self.ducking {
            let mut duck = vec![0f32; n as usize];
            let mut any = false;
            for v in &self.voices {
                let click = if v.accent { accent } else { beat };
                let len = click.env.len() as i64;
                let (a, b) = ((-v.p).max(0), n.min(len - v.p));
                for i in a..b {
                    let e = click.env[(v.p + i) as usize] * v.gain;
                    let d = &mut duck[i as usize];
                    if e > *d {
                        *d = e;
                        any = true;
                    }
                }
            }
            if any {
                let depth = (1.0 - 10f64.powf(Self::DUCK_DB / 20.0)) as f32;
                for (i, frame) in music.chunks_exact_mut(ch).enumerate() {
                    let g = 1.0 - depth * duck[i].min(1.0);
                    frame.iter_mut().for_each(|x| *x *= g);
                }
            }
        }
        for v in &self.voices {
            let click = if v.accent { accent } else { beat };
            let len = (click.data.len() / ch) as i64;
            let (a, b) = ((-v.p).max(0), n.min(len - v.p));
            for i in a..b {
                let s = ((v.p + i) as usize) * ch;
                for c in 0..ch {
                    aids[i as usize * ch + c] += click.data[s + c] * v.gain;
                }
            }
        }
        let (beat_len, accent_len) = ((beat.data.len() / ch) as i64, (accent.data.len() / ch) as i64);
        self.voices.retain_mut(|v| {
            v.p += n;
            v.p < if v.accent { accent_len } else { beat_len }
        });
    }
}

/// Reference tones for checking a key by ear: a held set of notes (MIDI numbers), a soft organ tone, each
/// note faded in / out; ``pitch_ratio`` follows varispeed.
pub struct Tones {
    rate: f64,
    hold: Vec<i32>,
    voices: Vec<(i32, [f64; 4])>, // note, [phase (cycles), gain, target, fade per frame]
}

impl Tones {
    const HARMONICS: [f64; 4] = [1.0, 0.45, 0.2, 0.08];
    const NOTE_GAIN: f64 = 0.07;
    const HOLD_FADE_S: f64 = 0.04;

    pub fn new(rate: f64) -> Tones {
        Tones { rate, hold: vec![], voices: vec![] }
    }

    pub fn active(&self) -> bool {
        !self.hold.is_empty() || !self.voices.is_empty()
    }

    pub fn hold(&mut self, notes: Vec<i32>) {
        self.hold = notes;
    }

    pub fn mix(&mut self, out: &mut [f32], ch: usize, pitch_ratio: f64) {
        let n = out.len() / ch;
        if n == 0 || !self.active() {
            return;
        }
        let step = Self::NOTE_GAIN / (Self::HOLD_FADE_S * self.rate).max(1.0);
        for &note in &self.hold {
            match self.voices.iter_mut().find(|v| v.0 == note) {
                Some(v) => {
                    v.1[2] = Self::NOTE_GAIN;
                    v.1[3] = step;
                }
                None => self.voices.push((note, [0.0, 0.0, Self::NOTE_GAIN, step])),
            }
        }
        for v in self.voices.iter_mut() {
            if !self.hold.contains(&v.0) {
                v.1[2] = 0.0;
            }
        }
        let ratio = if pitch_ratio.is_finite() && pitch_ratio > 0.0 { pitch_ratio } else { 1.0 };
        let norm: f64 = Self::HARMONICS.iter().sum();
        let rate = self.rate;
        for (note, v) in self.voices.iter_mut() {
            let [phase, gain, target, step] = *v;
            let freq = 440.0 * 2f64.powf((*note as f64 - 69.0) / 12.0) * ratio;
            let mut env_last = gain;
            for i in 0..n {
                let cyc = phase + i as f64 * (freq / rate);
                let env = if gain == target {
                    gain
                } else if target > gain {
                    (gain + step * (i + 1) as f64).min(target)
                } else {
                    (gain - step * (i + 1) as f64).max(target)
                };
                let mut wave = 0.0;
                for (k, amp) in Self::HARMONICS.iter().enumerate() {
                    let h = (k + 1) as f64;
                    if freq * h < rate * 0.45 {
                        wave += amp / norm * (2.0 * std::f64::consts::PI * h * cyc).sin();
                    }
                }
                let s = (wave * env) as f32;
                for c in 0..ch {
                    out[i * ch + c] += s;
                }
                env_last = env;
            }
            v[0] = (phase + n as f64 * freq / rate).rem_euclid(1.0);
            v[1] = env_last;
        }
        self.voices.retain(|(_, v)| !(v[1] <= 0.0 && v[2] <= 0.0));
    }
}
