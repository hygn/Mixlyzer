from PySide6 import QtCore, QtMultimedia
import numpy as np


class MetronomeController(QtCore.QObject):
    sig_tick = QtCore.Signal(float, int)

    def __init__(self, click_wav_path: str, parent=None):
        super().__init__(parent)

        self.beats_time = None  # np.ndarray[float] (sec)
        self.next_idx = 0
        self.enabled = False

        self.latency_guard = 0.020
        self.seek_threshold = 0.200
        self.max_ticks_per_call = 4
        self._offset_sec = 0.0
        self._prev_time = None
        self._in_cb = False
        self.last_tick_time = -1.0
        self._current_time = 0.0

        self._click: QtMultimedia.QSoundEffect | None = None
        self._soundfile = click_wav_path
        self._base_volume = 0.9

        self.downbeat_cycle = None
        self.downbeat_indices: frozenset[int] | None = None

    @QtCore.Slot()
    def initialize_audio(self):
        if self._click is not None:
            return
        self._click = QtMultimedia.QSoundEffect(self)
        self._click.setSource(QtCore.QUrl.fromLocalFile(self._soundfile))
        self._click.setVolume(self._base_volume)

    # Public API
    @QtCore.Slot()
    def start(self):
        self.enabled = True
        self._reset_pointer()

    @QtCore.Slot()
    def stop(self):
        self.enabled = False

    @QtCore.Slot(float)
    def set_volume(self, v: float):
        self._base_volume = float(np.clip(v, 0.0, 1.0))
        if self._click is not None:
            self._click.setVolume(self._base_volume)

    @QtCore.Slot(float)
    def set_offset(self, offset_msec: float):
        """Shift click timing relative to the beat. Positive = clicks lead the beat."""
        try:
            self._offset_sec = float(offset_msec) / 1000.0
        except (TypeError, ValueError):
            self._offset_sec = 0.0
        self._reset_pointer()

    @QtCore.Slot(object)
    def set_downbeat_cycle(self, n_beats: int | None):
        self.downbeat_cycle = int(n_beats) if n_beats and n_beats > 0 else None

    @QtCore.Slot(str)
    def set_soundfile(self, click_wav_path):
        self._soundfile = click_wav_path
        if self._click is not None:
            self._click.setSource(QtCore.QUrl.fromLocalFile(click_wav_path))

    @QtCore.Slot(object, object, float)
    def set_beats(self, beats_time, downbeat_indices=None, current_time: float = 0.0):
        bt = beats_time
        if bt is None or len(bt) == 0:
            self.beats_time = None
            self.downbeat_indices = None
            self.next_idx = 0
            self._current_time = float(current_time)
            self._prev_time = self._current_time
            self.last_tick_time = -1.0
            return
        self.beats_time = np.asarray(bt, dtype=float)
        if downbeat_indices is None:
            self.downbeat_indices = None
        else:
            self.downbeat_indices = frozenset(
                int(idx) for idx in np.asarray(downbeat_indices, dtype=np.int64).ravel()
            )
        self._current_time = float(current_time)
        self._reset_pointer()

    @QtCore.Slot()
    def clear_beats(self):
        self.set_beats(None, None, self._current_time)

    def _reset_pointer(self):
        if self.beats_time is None or len(self.beats_time) == 0:
            self.next_idx = 0
            self._prev_time = None
            self.last_tick_time = -1.0
            return
        t = float(self._current_time) + self._offset_sec
        self.next_idx = int(np.searchsorted(self.beats_time, t, side="right"))
        self._prev_time = t
        self.last_tick_time = -1.0

    @QtCore.Slot(float)
    def _on_time_changed(self, t: float):
        self._current_time = float(t)
        if self._in_cb:
            return
        self._in_cb = True
        try:
            if not self.enabled or self.beats_time is None:
                self._prev_time = self._current_time
                return

            cur = self._current_time + self._offset_sec

            # Seek detection: big jumps skip pending ticks and realign pointer
            if self._prev_time is None or abs(cur - self._prev_time) >= self.seek_threshold:
                # Reset next tick after current position (skip all missed ticks)
                self.next_idx = int(np.searchsorted(self.beats_time, cur, side="right"))
                self.last_tick_time = -1.0
                self._prev_time = cur
                return

            # Normal progression: process only the necessary ticks
            ticks_done = 0
            # Upper beat index allowed for this frame (guard included)
            limit_idx = int(np.searchsorted(self.beats_time, cur + self.latency_guard, side="right"))

            # If lagging too far behind, jump forward instead of looping forever
            if limit_idx - self.next_idx > self.max_ticks_per_call:
                # Ring only the last few; skip the rest to avoid delay buildup
                self.next_idx = max(self.next_idx, limit_idx - self.max_ticks_per_call)

            while self.next_idx < len(self.beats_time) and self.next_idx < limit_idx:
                bt = float(self.beats_time[self.next_idx])
                if self.last_tick_time < 0 or (bt - self.last_tick_time) > self.latency_guard:
                    self._tick(bt, self.next_idx)
                    self.last_tick_time = bt
                self.next_idx += 1
                ticks_done += 1
                if ticks_done >= self.max_ticks_per_call:
                    break

            self._prev_time = cur
        finally:
            self._in_cb = False

    def _tick(self, when_sec: float, idx: int, is_sub: bool = False):
        if self._click is None:
            return
        if self.downbeat_indices is not None:
            accent = idx in self.downbeat_indices and not is_sub
        elif self.downbeat_cycle:
            accent = (idx % self.downbeat_cycle == 0) and not is_sub
        else:
            accent = (not is_sub)

        vol = (1.0 if accent else 0.4) * self._base_volume
        self._click.setVolume(max(0.0, min(1.0, vol)))
        self._click.play()

        self.sig_tick.emit(when_sec, idx)
