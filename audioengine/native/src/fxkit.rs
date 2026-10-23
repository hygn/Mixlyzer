//! The effects' building blocks (colorfx.rs, beatfx.rs): a stereo delay line read anywhere between its samples, a
//! four-line feedback delay network for reverbs (the four lines a SIMD lane each, simd::F4, mixed by a Hadamard
//! matrix), an LFO, a one-pole smoother for parameters, white noise. Allocated when made (outside the callback);
//! the callback only reads and writes them.

use crate::simd::F4;

/// A parameter moved toward its target over ~``tau`` seconds, a block at a time.
#[derive(Clone, Copy)]
pub struct Smooth {
    pub value: f64,
    pole: f64,
}

impl Smooth {
    pub fn new(value: f64, tau: f64, rate: f64) -> Smooth {
        Smooth { value, pole: (-1.0 / (tau * rate).max(1.0)).exp() }
    }

    /// ``n`` frames toward ``target``; the value after them.
    #[inline]
    pub fn step(&mut self, target: f64, n: usize) -> f64 {
        self.value = target + self.pole.powi(n as i32) * (self.value - target);
        if (self.value - target).abs() < 1e-9 {
            self.value = target;
        }
        self.value
    }
}

/// A stereo delay line (interleaved frames), its length a power of two.
pub struct Delay {
    buf: Vec<f32>,
    mask: usize, // frames - 1
    pos: usize,  // the next frame written
}

impl Delay {
    /// At least ``frames`` long.
    pub fn new(frames: usize) -> Delay {
        let n = frames.max(4).next_power_of_two();
        Delay { buf: vec![0.0; n * 2], mask: n - 1, pos: 0 }
    }

    pub fn clear(&mut self) {
        self.buf.iter_mut().for_each(|v| *v = 0.0);
    }

    #[inline(always)]
    pub fn write(&mut self, l: f32, r: f32) {
        let i = self.pos * 2;
        self.buf[i] = l;
        self.buf[i + 1] = r;
        self.pos = (self.pos + 1) & self.mask;
    }

    /// The frame written ``delay`` frames before the next (1: the last written), linearly between frames.
    #[inline(always)]
    pub fn read(&self, delay: f64) -> (f32, f32) {
        let d = delay.clamp(1.0, (self.mask - 1) as f64);
        let whole = d.floor();
        let frac = (d - whole) as f32;
        let i0 = (self.pos + self.mask + 1 - whole as usize) & self.mask;
        let i1 = (i0 + self.mask) & self.mask; // one older
        let (a, b) = (i0 * 2, i1 * 2);
        (self.buf[a] + frac * (self.buf[b] - self.buf[a]), self.buf[a + 1] + frac * (self.buf[b + 1] - self.buf[a + 1]))
    }

    /// The frame at ``delay`` frames, whole.
    #[cfg(test)]
    pub fn read_whole(&self, delay: usize) -> (f32, f32) {
        let i = ((self.pos + self.mask + 1 - delay.clamp(1, self.mask)) & self.mask) * 2;
        (self.buf[i], self.buf[i + 1])
    }
}

/// A sine / triangle / saw LFO, its phase 0 .. 1.
#[derive(Clone, Copy, Default)]
pub struct Lfo {
    pub phase: f64,
}

impl Lfo {
    #[inline(always)]
    pub fn advance(&mut self, hz: f64, rate: f64) -> f64 {
        self.phase = (self.phase + hz / rate).fract();
        self.phase
    }

    pub fn triangle(&self) -> f64 {
        1.0 - 4.0 * (self.phase - 0.5).abs()
    }
}

/// White noise (xorshift32), -1 .. 1.
#[derive(Clone, Copy)]
pub struct Noise(u32);

impl Noise {
    pub fn new(seed: u32) -> Noise {
        Noise(seed.max(1))
    }

    #[inline(always)]
    pub fn next(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        (x as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

/// A reverb: four delay lines (prime lengths), their outputs mixed by a Hadamard matrix and fed back through a
/// damping low-pass, the four a SIMD lane each. Stereo in (summed), stereo out (lanes 0 + 2 left, 1 + 3 right).
pub struct Fdn4 {
    lines: [Vec<f32>; 4],
    pos: [usize; 4],
    lens: [usize; 4],
    damp_state: F4,
    pub feedback: f32, // the decay (0 .. <1)
    pub damp: f32,     // the feedback's low-pass (0 none .. 1 all)
    pre: Delay,        // the pre-delay
}

impl Fdn4 {
    /// Line lengths scaled by ``size`` (1: a room of ~ 35 - 60 ms lines).
    pub fn new(rate: f64, size: f64) -> Fdn4 {
        let base = [0.0297, 0.0371, 0.0411, 0.0437];
        let lens = base.map(|s| next_prime((s * size * rate) as usize).max(8));
        Fdn4 {
            lines: lens.map(|n| vec![0.0; n]),
            pos: [0; 4],
            lens,
            damp_state: F4::default(),
            feedback: 0.8,
            damp: 0.3,
            pre: Delay::new((0.05 * rate) as usize),
        }
    }

    pub fn clear(&mut self) {
        for l in &mut self.lines {
            l.iter_mut().for_each(|v| *v = 0.0);
        }
        self.damp_state = F4::default();
        self.pre.clear();
    }

    /// One frame in, the wet frame out.
    #[inline(always)]
    pub fn run(&mut self, l: f32, r: f32, pre_delay: f64) -> (f32, f32) {
        self.pre.write(l, r);
        let (pl, pr) = self.pre.read(pre_delay.max(1.0));
        let outs = F4::new(
            self.lines[0][self.pos[0]],
            self.lines[1][self.pos[1]],
            self.lines[2][self.pos[2]],
            self.lines[3][self.pos[3]],
        );
        // Hadamard 4 (normalised): mixes every line into every line, energy kept
        let [a, b, c, d] = outs.to_array();
        let mixed = F4::new(a + b + c + d, a - b + c - d, a + b - c - d, a - b - c + d) * F4::splat(0.5);
        // damping: a one-pole low-pass in the loop
        let damp = F4::splat(self.damp);
        self.damp_state = mixed + damp * (self.damp_state - mixed);
        let input = F4::new(pl, pr, pl, pr);
        let fb = self.damp_state * F4::splat(self.feedback) + input * F4::splat(0.5);
        let fb = fb.to_array();
        for k in 0..4 {
            self.lines[k][self.pos[k]] = fb[k];
            self.pos[k] += 1;
            if self.pos[k] == self.lens[k] {
                self.pos[k] = 0;
            }
        }
        ((a + c) * 0.5, (b + d) * 0.5)
    }
}

fn next_prime(n: usize) -> usize {
    let mut k = n.max(2);
    loop {
        if (2..).take_while(|d| d * d <= k).all(|d| k % d != 0) {
            return k;
        }
        k += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delay_reads_back() {
        let mut d = Delay::new(100);
        for i in 0..50 {
            d.write(i as f32, -(i as f32));
        }
        assert_eq!(d.read_whole(1), (49.0, -49.0));
        assert_eq!(d.read(10.0), (40.0, -40.0));
        let (l, _) = d.read(10.5);
        assert!((l - 39.5).abs() < 1e-6);
    }

    #[test]
    fn reverb_decays() {
        let mut f = Fdn4::new(48000.0, 1.0);
        f.feedback = 0.7;
        let mut energy = [0f64; 4];
        for i in 0..48000 {
            let x = if i == 0 { 1.0 } else { 0.0 };
            let (l, r) = f.run(x, x, 1.0);
            energy[i * 4 / 48000] += (l * l + r * r) as f64;
        }
        assert!(energy[0] > 0.0 && energy[3] < energy[0] * 0.01, "{energy:?}");
    }
}
