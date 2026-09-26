from __future__ import annotations
import math
from typing import Optional

import numpy as np


class ClickTrack:
    """
    Metronome clicks mixed into the player's output stream.

    Beats are positions on the track timeline. A click starts at the output frame where the
    render head reaches its beat and then plays at the device rate regardless of tempo, so it
    stays sample-aligned with the music without a second audio stream.

    With ducking on, the music under a click is lowered by up to DUCK_DB, following the
    click's envelope (instant attack, DUCK_RELEASE_MS release) scaled by its gain.
    """

    MAX_CLICKS_PER_BLOCK = 4
    DUCK_DB = -6.0                                      # music level under a full-gain click
    DUCK_RELEASE_MS = 60.0

    def __init__(self, rate: int, channels: int):
        self.rate = int(rate)
        self.ch = int(channels)
        self._raw: Optional[np.ndarray] = None          # click as loaded, [L, ch] float32 at device rate
        self._speeds = (1.0, 1.0)                       # playback speed of (beat, accent) clicks
        self._sample: Optional[np.ndarray] = None       # _raw at the beat speed (zero-padded)
        self._accent_sample: Optional[np.ndarray] = None  # _raw at the accent speed (zero-padded)
        self._envelopes: dict[int, np.ndarray] = {}     # id(sample) -> [L] envelope, peak 1
        self._ducking = False
        self._beats = np.zeros(0, dtype=np.float64)     # beat positions (input frames, sorted)
        self._gains = np.zeros(0, dtype=np.float32)
        self._accents = np.zeros(0, dtype=bool)
        self._enabled = False
        self._offset = 0.0                              # input frames; positive = clicks lead the beat
        self._voices: list[list] = []                   # [read index into sample, gain, sample, envelope]

    def set_sample(self, sample: Optional[np.ndarray]) -> None:
        """Click sound as [L, ch] float32 at the device rate (None disables)."""
        self._raw = None if sample is None else np.asarray(sample, dtype=np.float32)
        self._rebuild()

    def set_pitches(self, beat_semitones: float, accent_semitones: float) -> None:
        """Pitch of the beat / accent clicks, raised by playing them faster (and shorter)."""
        speeds = (2.0 ** (float(beat_semitones) / 12.0), 2.0 ** (float(accent_semitones) / 12.0))
        if speeds != self._speeds:
            self._speeds = speeds
            self._rebuild()

    def _rebuild(self) -> None:
        self._sample = self._accent_sample = None
        self._envelopes = {}
        self._voices.clear()
        if self._raw is None:
            return
        variants = []
        for speed in self._speeds:
            # Nearest-sample read at `speed`: pitch and speed change together.
            n = max(1, int(self._raw.shape[0] / speed))
            idx = np.minimum((np.arange(n) * speed).astype(np.int64), self._raw.shape[0] - 1)
            sample, env = self._with_envelope(self._raw[idx])
            self._envelopes[id(sample)] = env
            variants.append(sample)
        self._sample, self._accent_sample = variants

    def _with_envelope(self, sample: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Ducking envelope of a click (peak 1, instant attack, exponential release), with
        the sample zero-padded to the envelope's length so the release is not cut off."""
        level = np.max(np.abs(sample), axis=1).astype(np.float64)
        peak = float(level.max()) if level.size else 0.0
        if peak <= 0.0:
            return np.ascontiguousarray(sample), np.zeros(sample.shape[0], np.float32)
        level /= peak
        decay = math.exp(-1.0 / (self.DUCK_RELEASE_MS * 1e-3 * self.rate))
        tail = int(math.ceil(math.log(1e-3) / math.log(decay)))  # release down to -60 dB
        env = np.zeros(level.size + tail, np.float64)
        e = 0.0
        for i in range(env.size):
            e = e * decay
            if i < level.size and level[i] > e:
                e = level[i]
            env[i] = e
        padded = np.zeros((env.size, sample.shape[1]), np.float32)
        padded[:sample.shape[0]] = sample
        return padded, env.astype(np.float32)

    def set_ducking(self, enabled: bool) -> None:
        self._ducking = bool(enabled)

    def set_beats(self, beats_sec, gains, accents=None) -> None:
        """accents: per-beat bool, True plays the accent click (None: none)."""
        if beats_sec is None or len(beats_sec) == 0:
            self._beats = np.zeros(0, dtype=np.float64)
            self._gains = np.zeros(0, dtype=np.float32)
            self._accents = np.zeros(0, dtype=bool)
            return
        self._beats = np.asarray(beats_sec, dtype=np.float64) * self.rate
        self._gains = np.asarray(gains, dtype=np.float32)
        self._accents = (np.zeros(len(self._beats), dtype=bool) if accents is None
                         else np.asarray(accents, dtype=bool))

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)
        if not self._enabled:
            self._voices.clear()

    def set_offset_sec(self, offset_sec: float) -> None:
        self._offset = float(offset_sec) * self.rate

    def clear_voices(self) -> None:
        self._voices.clear()

    def queue(self, in_start: float, step: float, n: int) -> None:
        """Start a click at the output frame where each beat inside this block is reached."""
        if not self._enabled or self._sample is None or self._beats.size == 0:
            return
        lo = in_start + self._offset
        hi = in_start + step * n + self._offset
        i0 = int(np.searchsorted(self._beats, lo, side="left"))
        i1 = int(np.searchsorted(self._beats, hi, side="left"))
        for i in range(max(i0, i1 - self.MAX_CLICKS_PER_BLOCK), i1):
            k = int(math.ceil((self._beats[i] - lo) / step))
            sample = self._accent_sample if self._accents[i] else self._sample
            self._voices.append([-k, float(self._gains[i]), sample, self._envelopes[id(sample)]])

    def mix(self, out: np.ndarray) -> np.ndarray:
        """Add sounding clicks to a block about to be written (returns a new array),
        ducking the music under them when ducking is on."""
        if not self._voices:
            return out
        if self._sample is None:
            self._voices.clear()
            return out
        n = out.shape[0]
        out = np.array(out, dtype=np.float32)
        spans = []
        duck = None
        for p, g, sample, env in self._voices:
            a = max(0, -p)
            b = min(n, sample.shape[0] - p)
            if b > a:
                spans.append((a, b, p, g, sample))
                if self._ducking:
                    if duck is None:
                        duck = np.zeros(n, np.float32)
                    np.maximum(duck[a:b], env[p + a:p + b] * np.float32(g), out=duck[a:b])
        if duck is not None:
            depth = np.float32(1.0 - 10.0 ** (self.DUCK_DB / 20.0))
            out *= (1.0 - depth * np.minimum(duck, 1.0))[:, None]
        for a, b, p, g, sample in spans:
            out[a:b] += sample[p + a:p + b] * np.float32(g)
        keep = []
        for p, g, sample, env in self._voices:
            p += n
            if p < sample.shape[0]:
                keep.append([p, g, sample, env])
        self._voices = keep
        return out
