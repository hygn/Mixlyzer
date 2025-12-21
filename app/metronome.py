from PySide6 import QtCore, QtMultimedia
import numpy as np

class MetronomeController(QtCore.QObject):
    sig_tick = QtCore.Signal(float, int)

    def __init__(self, bus, tl, model, click_wav_path: str, parent=None):
        super().__init__(parent)
        self.bus = bus
        self.tl = tl
        self.model = model

        self.beats_time = None  # np.ndarray[float] (sec)
        self.next_idx = 0
        self.enabled = False

        # Timing/performance guardrails 
        self.latency_guard = 0.020
        self.seek_threshold = 0.200
        self.max_ticks_per_call = 4
        self._prev_time = None
        self._in_cb = False
        self.last_tick_time = -1.0

        # Sound 
        self.click = QtMultimedia.QSoundEffect(self)
        self.click.setSource(QtCore.QUrl.fromLocalFile(click_wav_path))
        self._base_volume = 0.9
        self.click.setVolume(self._base_volume)

        # Signal hookups
        self.bus.sig_features_loaded.connect(self._on_features_loaded)
        self.bus.sig_time_changed.connect(self._on_time_changed)
        self.bus.sig_beatgrid_edited.connect(self._on_beatgrid_edit)

        # Options
        self.downbeat_cycle = None

    # Public API
    def start(self):
        self.enabled = True
        self._reset_pointer()

    def stop(self):
        self.enabled = False

    def set_volume(self, v: float):
        self._base_volume = float(np.clip(v, 0.0, 1.0))
        self.click.setVolume(self._base_volume)

    def set_downbeat_cycle(self, n_beats: int | None):
        self.downbeat_cycle = int(n_beats) if n_beats and n_beats > 0 else None
        
    def set_soundfile(self, click_wav_path):
        self.click.setSource(QtCore.QUrl.fromLocalFile(click_wav_path))

    # Internals 
    def _on_features_loaded(self):
        f = self.model.features or {}
        bt = f.get("beats_time_sec")
        if bt is None or len(bt) == 0:
            self.beats_time = None
            self.next_idx = 0
            return
        self.beats_time = np.asarray(bt, dtype=float)
        self._reset_pointer()
    
    def _on_beatgrid_edit(self):
        f = self.model.features or {}
        bt = f.get("beats_time_sec")
        if bt is None or len(bt) == 0:
            self.beats_time = None
            self.next_idx = 0
            return
        self.beats_time = np.asarray(bt, dtype=float)
        self._reset_pointer()

    def _reset_pointer(self):
        if self.beats_time is None or len(self.beats_time) == 0:
            self.next_idx = 0
            self._prev_time = None
            self.last_tick_time = -1.0
            return
        t = float(self.tl.current_time)
        self.next_idx = int(np.searchsorted(self.beats_time, t, side="right"))
        self._prev_time = t
        self.last_tick_time = -1.0

    @QtCore.Slot(float)
    def _on_time_changed(self, t: float):
        if self._in_cb:
            return
        self._in_cb = True
        try:
            if not self.enabled or self.beats_time is None:
                self._prev_time = float(t)
                return

            cur = float(t)

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
        # Downbeat accent
        if self.downbeat_cycle:
            accent = (idx % self.downbeat_cycle == 0) and not is_sub
        else:
            accent = (not is_sub)

        vol = (1.0 if accent else 0.6) * self._base_volume
        self.click.setVolume(max(0.0, min(1.0, vol)))
        self.click.play()

        self.sig_tick.emit(when_sec, idx)
