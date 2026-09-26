from typing import Callable, Optional

from PySide6 import QtCore
import numpy as np

try:
    import soundfile as sf
except Exception:  # pragma: no cover
    sf = None


def _load_click(path: str, rate: int, channels: int) -> Optional[np.ndarray]:
    """Load the click sound as [L, channels] float32 at the output rate."""
    if sf is None or not path:
        return None
    try:
        data, sr = sf.read(path, dtype="float32", always_2d=True)
    except Exception as exc:
        print(f"[Metronome] Failed to load click '{path}': {exc}")
        return None
    if data.shape[0] == 0:
        return None
    if sr != rate:
        n_out = max(1, int(round(data.shape[0] * rate / sr)))
        t = np.arange(n_out, dtype=np.float64) * (sr / rate)
        src = np.arange(data.shape[0], dtype=np.float64)
        data = np.stack([np.interp(t, src, data[:, c]) for c in range(data.shape[1])], axis=1)
    if data.shape[1] != channels:
        data = np.repeat(data.mean(axis=1, keepdims=True), channels, axis=1)
    return np.ascontiguousarray(data, dtype=np.float32)


class MetronomeController(QtCore.QObject):
    """
    Metronome settings and beats, handed to the player's click mixer (ClickTrack).

    Clicks are mixed into the player's own output stream at each beat's input position,
    so they stay sample-aligned with the music at any tempo and no second audio stream is
    opened. Clicks only sound during normal playback.
    Lives in the player's audio thread, like the mixer it drives.
    """

    def __init__(self, click_wav_path: str, parent=None):
        super().__init__(parent)

        self.beats_time = None  # np.ndarray[float] (sec)
        self.enabled = False
        self._offset_sec = 0.0
        self._soundfile = click_wav_path
        # Downbeat (accent) and other beats: volume 0..1, pitch in semitones.
        self._downbeat_volume = 0.9
        self._beat_volume = 0.36
        self._downbeat_pitch = 7.0
        self._beat_pitch = 0.0

        self.downbeat_cycle = None
        self.downbeat_indices: frozenset[int] | None = None

        self._ducking = False

        self._mixer_provider: Optional[Callable[[], object]] = None
        self._initialized = False

    def bind_mixer(self, provider: Callable[[], object]) -> None:
        """provider() returns the ClickTrack (or None before the audio thread created it)."""
        self._mixer_provider = provider

    def _mixer(self):
        if not self._initialized or self._mixer_provider is None:
            return None
        return self._mixer_provider()

    @QtCore.Slot()
    def initialize_audio(self):
        self._initialized = True
        self._push_sample()
        self._push_beats()
        self._push_settings()
        mixer = self._mixer()
        if mixer is not None:
            mixer.set_ducking(self._ducking)

    # Public API
    @QtCore.Slot()
    def start(self):
        self.enabled = True
        self._push_settings()

    @QtCore.Slot()
    def stop(self):
        self.enabled = False
        self._push_settings()

    @QtCore.Slot(float, float, float, float)
    def set_click_levels(self, downbeat_volume: float, downbeat_pitch: float,
                         beat_volume: float, beat_pitch: float):
        """Volumes 0..1; pitches in semitones (the click is played faster / slower)."""
        self._downbeat_volume = float(np.clip(downbeat_volume, 0.0, 1.0))
        self._beat_volume = float(np.clip(beat_volume, 0.0, 1.0))
        self._downbeat_pitch = float(downbeat_pitch)
        self._beat_pitch = float(beat_pitch)
        self._push_sample_pitch()
        self._push_beats()

    @QtCore.Slot(bool)
    def set_ducking(self, enabled: bool):
        self._ducking = bool(enabled)
        mixer = self._mixer()
        if mixer is not None:
            mixer.set_ducking(self._ducking)

    @QtCore.Slot(float)
    def set_offset(self, offset_msec: float):
        """Shift click timing relative to the beat. Positive = clicks lead the beat."""
        try:
            self._offset_sec = float(offset_msec) / 1000.0
        except (TypeError, ValueError):
            self._offset_sec = 0.0
        self._push_settings()

    @QtCore.Slot(object)
    def set_downbeat_cycle(self, n_beats: int | None):
        self.downbeat_cycle = int(n_beats) if n_beats and n_beats > 0 else None
        self._push_beats()

    @QtCore.Slot(str)
    def set_soundfile(self, click_wav_path):
        self._soundfile = click_wav_path
        self._push_sample()

    @QtCore.Slot(object, object, float)
    def set_beats(self, beats_time, downbeat_indices=None, current_time: float = 0.0):
        bt = beats_time
        if bt is None or len(bt) == 0:
            self.beats_time = None
            self.downbeat_indices = None
        else:
            self.beats_time = np.asarray(bt, dtype=float)
            if downbeat_indices is None:
                self.downbeat_indices = None
            else:
                self.downbeat_indices = frozenset(
                    int(idx) for idx in np.asarray(downbeat_indices, dtype=np.int64).ravel()
                )
        self._push_beats()

    @QtCore.Slot()
    def clear_beats(self):
        self.set_beats(None, None)

    # Mixer sync
    def _beat_accents(self) -> np.ndarray:
        """Downbeats (all False when every beat would be one, i.e. no downbeat is known)."""
        n = len(self.beats_time)
        idx = np.arange(n)
        if self.downbeat_indices is not None:
            accent = np.isin(idx, np.fromiter(self.downbeat_indices, dtype=np.int64))
        elif self.downbeat_cycle:
            accent = (idx % self.downbeat_cycle) == 0
        else:
            accent = np.zeros(n, dtype=bool)
        if accent.all():
            accent[:] = False
        return accent

    def _beat_gains(self, accent: np.ndarray) -> np.ndarray:
        return np.where(accent, self._downbeat_volume, self._beat_volume).astype(np.float32)

    def _push_sample(self):
        mixer = self._mixer()
        if mixer is not None:
            mixer.set_pitches(self._beat_pitch, self._downbeat_pitch)
            mixer.set_sample(_load_click(self._soundfile, mixer.rate, mixer.ch))

    def _push_sample_pitch(self):
        mixer = self._mixer()
        if mixer is not None:
            mixer.set_pitches(self._beat_pitch, self._downbeat_pitch)

    def _push_beats(self):
        mixer = self._mixer()
        if mixer is None:
            return
        if self.beats_time is None:
            mixer.set_beats(None, None)
        else:
            accent = self._beat_accents()
            mixer.set_beats(self.beats_time, self._beat_gains(accent), accent)

    def _push_settings(self):
        mixer = self._mixer()
        if mixer is not None:
            mixer.set_offset_sec(self._offset_sec)
            mixer.set_enabled(self.enabled)
