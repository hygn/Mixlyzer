"""The deck of the Python engine (the compatibility mode): audioengine/native/src/deck.rs's requests and report
over PCMFeeder, so the GUI asks and shows the same on either engine. No MIDI controller here (the native engine
reads it); see deck.rs for the behaviour (rekordbox's CUE, the beat / bar steps, CALL, a JumpCUE's JUMP, the
tempo fader's range, the volume fader, the lamps)."""
from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np

from audioengine.native.client import (
    ACT_BAR_BACK as BAR_BACK, ACT_BAR_FORWARD as BAR_FORWARD, ACT_BEAT_BACK as BEAT_BACK,
    ACT_BEAT_FORWARD as BEAT_FORWARD, ACT_CALL_NEXT as CALL_NEXT, ACT_CALL_PREVIOUS as CALL_PREVIOUS, ACT_CUE as CUE,
    ACT_JUMP_A as JUMP_A, ACT_PLAY_PAUSE as PLAY_PAUSE, ACT_STOP as STOP, ACT_TEMPO_DOWN as TEMPO_DOWN,
    ACT_TEMPO_RANGE as TEMPO_RANGE, ACT_TEMPO_RESET as TEMPO_RESET, ACT_TEMPO_UP as TEMPO_UP, TEMPO_RANGES, DeckReport,
)
from utils.volume import slider_percent_to_linear
CUE_AT_SEC = 0.001
JUMP_CANDIDATE_BEATS = 8
JUMP_LANDING_SEC = 0.5
BLINK_SEC = 0.5


class PythonDeck:
    """deck.rs over ``feeder`` (a PCMFeeder; positions in its input frames). Lives in the player's audio thread;
    tick() once a frame."""

    def __init__(self, feeder) -> None:
        self._f = feeder
        self._volume = 1.0
        self._trim_db = 0.0
        self._tempo_range = 100
        self._factor = 1.0
        self._cue = 0.0
        self._previewing = False
        self._latched = False
        self._beats = np.empty(0)
        self._downbeats = np.empty(0)
        self._cue_points = np.empty(0)
        self._jumpcues = np.full(26, np.nan)
        self._armed: Optional[tuple[float, list[tuple[float, float]]]] = None
        self._landing: Optional[tuple[float, float]] = None
        self._jumped = False
        feeder.jumped.connect(self._on_jumped)

    @property
    def _rate(self) -> float:
        return float(self._f.rate)

    def _len(self) -> float:
        return float(getattr(self._f, "_len", 0))

    def _has_track(self) -> bool:
        return getattr(self._f, "_pcm", None) is not None

    # Requests
    def new_track(self) -> None:
        self._cue = 0.0
        self._previewing = self._latched = False
        self._beats = self._downbeats = self._cue_points = np.empty(0)
        self._jumpcues = np.full(26, np.nan)
        self._armed = self._landing = None

    def set_marks(self, beats_sec, downbeats, cue_points_sec, jumpcues_sec) -> None:
        rate = self._rate
        beats = np.asarray(beats_sec, dtype=np.float64).reshape(-1) * rate
        down = np.asarray(downbeats, dtype=bool).reshape(-1)[:beats.size]
        self._beats = beats
        self._downbeats = beats[:down.size][down]
        self._cue_points = np.sort(np.asarray(cue_points_sec, dtype=np.float64).reshape(-1) * rate)
        jc = np.full(26, np.nan)
        j = np.asarray(jumpcues_sec, dtype=np.float64).reshape(-1)[:26]
        jc[:j.size] = j * rate
        self._jumpcues = jc

    def set_volume(self, position: float) -> None:
        self._volume = min(1.0, max(0.0, float(position)))
        self._apply_volume()

    def set_volume_trim(self, db: float) -> None:
        self._trim_db = float(db)
        self._apply_volume()

    def _apply_volume(self) -> None:
        self._f.set_music_gain(slider_percent_to_linear(self._volume * 100.0, self._trim_db))

    def set_tempo_range(self, percent: int) -> None:
        self._tempo_range = int(percent) if int(percent) in TEMPO_RANGES else 100
        self.set_tempo(self._factor)

    def set_tempo(self, factor: float) -> None:
        r = self._tempo_range / 100.0
        f = float(factor) if math.isfinite(float(factor)) else 1.0
        self._factor = min(1.0 + r, max(max(0.0, 1.0 - r), f))
        self._f.set_factor(self._factor)
        self._f.set_mode("none" if abs(self._factor - 1.0) <= 1e-9 else "speed")

    def deck_action(self, code: int, pressed: bool = True) -> None:
        f = self._f
        if code == CUE:
            self._cue_down() if pressed else self._cue_up()
            return
        if not pressed:
            return
        if code == PLAY_PAUSE:
            if self._previewing:
                self._latched = True
            elif f.is_playing():
                f.pause()
            else:
                f.play()
        elif code == STOP:
            self._previewing = False
            f.pause()
            f.seek(0.0)
        elif code in (BEAT_BACK, BEAT_FORWARD, BAR_BACK, BAR_FORWARD):
            self._step_beats(-1 if code in (BEAT_BACK, BAR_BACK) else 1, code in (BAR_BACK, BAR_FORWARD))
        elif code in (TEMPO_DOWN, TEMPO_UP):
            self.set_tempo((round(self._factor * 100.0) + (1 if code == TEMPO_UP else -1)) / 100.0)
        elif code == TEMPO_RESET:
            self.set_tempo(1.0)
        elif code in (CALL_PREVIOUS, CALL_NEXT):
            if f.is_playing():
                return
            t, near = f.playhead_frame(), 0.001 * self._rate
            if code == CALL_NEXT:
                ahead = self._cue_points[self._cue_points > t + near]
                target = ahead[0] if ahead.size else None
            else:
                behind = self._cue_points[self._cue_points < t - near]
                target = behind[-1] if behind.size else None
            if target is not None:
                self.call_at(float(target) / self._rate)
        elif code == TEMPO_RANGE:
            i = TEMPO_RANGES.index(self._tempo_range) + 1 if self._tempo_range in TEMPO_RANGES else 0
            self.set_tempo_range(TEMPO_RANGES[i % len(TEMPO_RANGES)])
        elif JUMP_A <= code < JUMP_A + 8:
            dest = self._jumpcues[code - JUMP_A]
            if math.isfinite(dest):
                self.jump_to(dest / self._rate)

    def _cue_down(self) -> None:
        f = self._f
        if not self._has_track() or self._previewing:
            return
        if f.is_playing():
            f.pause()
            f.seek(self._cue)
            return
        pos = f.playhead_frame()
        if abs(pos - self._cue) <= CUE_AT_SEC * self._rate:
            if f.play():
                self._previewing, self._latched = True, False
        else:
            self._cue = self._quantize(pos)
            f.seek(self._cue)

    def _cue_up(self) -> None:
        if not self._previewing:
            return
        self._previewing = False
        if not self._latched:
            self._f.pause()
            self._f.seek(self._cue)

    def _quantize(self, pos: float) -> float:
        if self._beats.size == 0:
            return pos
        n = self._len()
        grid = np.concatenate((self._beats[(self._beats >= 0) & (self._beats <= n)], [0.0, n]))
        return float(grid[int(np.argmin(np.abs(grid - pos)))])

    def call_at(self, sec: float) -> None:
        f = self._f
        if f.is_playing() or not math.isfinite(float(sec)):
            return
        frame = min(self._len(), max(0.0, float(sec) * self._rate))
        f.seek(frame)
        self._cue = frame

    def _step_beats(self, step: int, bars: bool) -> None:
        f = self._f
        marks = self._downbeats if bars else self._beats
        if not self._has_track() or marks.size == 0:
            return
        t, near = f.playhead_frame(), 0.001 * self._rate
        i = int(np.searchsorted(marks, t, side="right")) - 1
        if f.is_playing() and i >= 0:
            j = int(np.clip(i + step, 0, marks.size - 1))
            target = t + float(marks[j] - marks[i])
        elif step > 0:
            ahead = marks[marks > t + near]
            if ahead.size == 0:
                return
            target = float(ahead[min(step, ahead.size) - 1])
        else:
            behind = marks[marks < t - near]
            if behind.size == 0:
                return
            target = float(behind[max(behind.size + step, 0)])
        f.seek(min(max(target, 0.0), self._len()))

    def jump_to(self, sec: float) -> None:
        f = self._f
        dest = float(sec) * self._rate
        if not self._has_track() or not math.isfinite(dest):
            return
        if not f.is_playing():
            f.seek(dest)
            return
        if self._armed is not None and abs(self._armed[0] - dest) < 0.001 * self._rate:
            return
        bases = [f.playhead_frame()]
        if self._landing is not None and time.monotonic() - self._landing[1] < JUMP_LANDING_SEC:
            bases.append(self._landing[0])
        ranges, starts = [], []
        for base in bases:
            ahead = self._beats[self._beats > base][:JUMP_CANDIDATE_BEATS]
            if ahead.size:
                ranges.append((base, float(ahead[-1])))
                starts.extend(ahead.tolist())
        if not starts:
            return
        rate = self._rate
        f.arm_jump(None, dest / rate, once=True, starts_sec=[s / rate for s in sorted(set(starts))])
        self._armed = (dest, ranges)

    def _on_jumped(self) -> None:
        self._jumped = True

    # Once a frame
    def tick(self) -> None:
        f = self._f
        if self._jumped:
            self._jumped = False
            if self._armed is not None:
                self._landing = (self._armed[0], time.monotonic())
                self._armed = None
        if self._armed is not None:
            t, rate = f.playhead_frame(), self._rate
            inside = any(lo - 0.1 * rate <= t <= hi + 0.02 * rate for lo, hi in self._armed[1])
            if not f.is_playing() or not inside:
                self._armed = None
                f.disarm_jump()
        if self._previewing and not f.is_playing():
            self._previewing = False

    def deck_report(self) -> DeckReport:
        f = self._f
        playing = f.is_playing()
        lamps = 0
        if self._has_track():
            blink = math.floor(time.monotonic() / BLINK_SEC) % 2 == 0
            on_cue = abs(f.playhead_frame() - self._cue) <= CUE_AT_SEC * self._rate
            lamps = int(playing or blink) | int(playing or self._previewing or on_cue or blink) << 1
        rate = self._rate
        cue = self._cue / rate if self._has_track() else math.nan
        dest = self._armed[0] / rate if self._armed is not None else math.nan
        return DeckReport(playing, self._factor, self._volume, cue, dest, lamps, self._tempo_range, 0)
