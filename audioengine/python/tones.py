from __future__ import annotations

import math

import numpy as np


class ToneTrack:
    """
    Reference tones mixed into the player's output stream, for checking a key by ear:
    a held set of notes (the root, root and fifth, or a chord, kept until changed).
    Notes are MIDI numbers; a soft organ tone (a few harmonics), every note faded in /
    out so changes do not click.

    `pitch_ratio` follows varispeed playback, so the tones stay in tune with the music.
    Only touch it from the audio thread.
    """

    HARMONICS = (1.0, 0.45, 0.2, 0.08)
    NOTE_GAIN = 0.07           # per note; a four-note chord stays well under full scale
    HOLD_FADE_S = 0.04

    def __init__(self, rate: int, channels: int):
        self.rate = int(rate)
        self.ch = int(channels)
        self._hold: tuple[int, ...] = ()
        # voice: note -> [phase (cycles), gain, target, fade per frame]
        self._voices: dict[int, list[float]] = {}
        norm = sum(self.HARMONICS)
        self._harm = [(k + 1, a / norm) for k, a in enumerate(self.HARMONICS)]

    @property
    def active(self) -> bool:
        return bool(self._hold or self._voices)

    def hold(self, notes) -> None:
        self._hold = tuple(int(n) for n in (notes or ()))

    def stop(self) -> None:
        self._hold = ()

    def mix(self, out: np.ndarray, pitch_ratio: float = 1.0) -> np.ndarray:
        """Add the sounding tones to a block about to be written (returns a new array)."""
        n = int(out.shape[0])
        if n <= 0 or not self.active:
            return out
        acc = np.zeros(n, np.float32)
        self._render(acc, n, pitch_ratio)
        out = np.array(out, dtype=np.float32)
        out += acc[:, None]
        return out

    def _render(self, acc: np.ndarray, run: int, pitch_ratio: float) -> None:
        step = self.NOTE_GAIN / max(1.0, self.HOLD_FADE_S * self.rate)
        for note in self._hold:
            v = self._voices.get(note)
            if v is None:
                self._voices[note] = [0.0, 0.0, self.NOTE_GAIN, step]
            else:
                v[2], v[3] = self.NOTE_GAIN, step
        for note, v in self._voices.items():
            if note not in self._hold:
                v[2] = 0.0
        ratio = pitch_ratio if math.isfinite(pitch_ratio) and pitch_ratio > 0 else 1.0
        idx = np.arange(run, dtype=np.float64)
        for note, v in list(self._voices.items()):
            phase, gain, target, step = v
            freq = 440.0 * 2.0 ** ((note - 69) / 12.0) * ratio
            cyc = phase + idx * (freq / self.rate)
            if gain == target:
                env = np.full(run, gain, np.float64)
            else:
                d = step if target > gain else -step
                env = gain + d * (idx + 1)
                env = np.minimum(env, target) if d > 0 else np.maximum(env, target)
            wave = np.zeros(run, np.float64)
            for h, amp in self._harm:
                if freq * h < self.rate * 0.45:
                    wave += amp * np.sin(2.0 * np.pi * h * cyc)
            acc += (wave * env).astype(np.float32)
            v[0] = float((phase + run * freq / self.rate) % 1.0)
            v[1] = float(env[-1])
            if v[1] <= 0.0 and target <= 0.0:
                del self._voices[note]
