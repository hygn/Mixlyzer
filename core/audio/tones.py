from __future__ import annotations

import math

import numpy as np


class ToneTrack:
    """
    Reference tones mixed into the player's output stream, for checking a key by ear:
    a held set of notes (a drone or a chord, kept until changed) and a sequence played
    once over it (a cadence). Notes are MIDI numbers; a soft organ tone (a few
    harmonics), every note faded in / out so chord changes do not click.

    `pitch_ratio` follows varispeed playback, so the tones stay in tune with the music.
    Only touch it from the audio thread.
    """

    HARMONICS = (1.0, 0.45, 0.2, 0.08)
    NOTE_GAIN = 0.07           # per note; a four-note chord stays well under full scale
    HOLD_FADE_S = 0.04
    SEQ_FADE_S = 0.012

    def __init__(self, rate: int, channels: int):
        self.rate = int(rate)
        self.ch = int(channels)
        self._hold: tuple[int, ...] = ()
        self._seq: list[tuple[tuple[int, ...], int]] = []   # (notes, frames left), first is sounding
        # voice key (layer, note) -> [phase (cycles), gain, target, fade per frame]
        self._voices: dict[tuple[str, int], list[float]] = {}
        norm = sum(self.HARMONICS)
        self._harm = [(k + 1, a / norm) for k, a in enumerate(self.HARMONICS)]

    @property
    def active(self) -> bool:
        return bool(self._hold or self._seq or self._voices)

    def hold(self, notes) -> None:
        self._hold = tuple(int(n) for n in (notes or ()))

    def play_sequence(self, steps) -> None:
        """``steps``: [(notes, seconds), ...]; replaces a sequence still sounding."""
        self._seq = [
            (tuple(int(n) for n in notes), max(1, int(float(sec) * self.rate)))
            for notes, sec in (steps or ())
        ]

    def stop(self) -> None:
        self._hold = ()
        self._seq = []

    def _targets(self) -> dict[tuple[str, int], float]:
        want = {("hold", n): self.NOTE_GAIN for n in self._hold}
        if self._seq:
            for n in self._seq[0][0]:
                want[("seq", n)] = self.NOTE_GAIN
        return want

    def mix(self, out: np.ndarray, pitch_ratio: float = 1.0) -> np.ndarray:
        """Add the sounding tones to a block about to be written (returns a new array)."""
        n = int(out.shape[0])
        if n <= 0 or not self.active:
            return out
        acc = np.zeros(n, np.float32)
        pos = 0
        while pos < n:
            # render up to the next sequence step change
            run = n - pos
            if self._seq:
                run = min(run, self._seq[0][1])
            self._render(acc, pos, run, pitch_ratio)
            pos += run
            if self._seq:
                notes, left = self._seq[0]
                left -= run
                if left <= 0:
                    self._seq.pop(0)
                else:
                    self._seq[0] = (notes, left)
        out = np.array(out, dtype=np.float32)
        out += acc[:, None]
        return out

    def _render(self, acc: np.ndarray, pos: int, run: int, pitch_ratio: float) -> None:
        targets = self._targets()
        for key, gain in targets.items():
            v = self._voices.get(key)
            fade = self.HOLD_FADE_S if key[0] == "hold" else self.SEQ_FADE_S
            step = self.NOTE_GAIN / max(1.0, fade * self.rate)
            if v is None:
                self._voices[key] = [0.0, 0.0, gain, step]
            else:
                v[2], v[3] = gain, step
        for key, v in list(self._voices.items()):
            if key not in targets:
                v[2] = 0.0
        ratio = pitch_ratio if math.isfinite(pitch_ratio) and pitch_ratio > 0 else 1.0
        idx = np.arange(run, dtype=np.float64)
        for key, v in list(self._voices.items()):
            phase, gain, target, step = v
            freq = 440.0 * 2.0 ** ((key[1] - 69) / 12.0) * ratio
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
            acc[pos:pos + run] += (wave * env).astype(np.float32)
            v[0] = float((phase + run * freq / self.rate) % 1.0)
            v[1] = float(env[-1])
            if v[1] <= 0.0 and target <= 0.0:
                del self._voices[key]
