"""Streaming, percussion-aware, pitch-preserving time scaling.

It exposes the same pull contract as core.audio.dsp.SpeedResampler

    render(pcm, input_position, speed_factor, maximum_output_frames)

``speed_factor`` is measured in input frames per output frame.  Consequently,
values above one make playback faster and values below one make it slower while
this processor keeps pitch approximately constant.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass
from typing import Final

import numpy as np
from scipy.fft import irfft, rfft
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfiltfilt


_EPS: Final = 1e-12
_TWO_PI: Final = 2.0 * np.pi

_NUMBA_LOCK = threading.Lock()
_NUMBA_KERNELS = None
_NUMBA_READY = False
_NUMBA_FAILED = False
_NUMBA_ERROR: BaseException | None = None


def _wrap_phase(value: np.ndarray) -> np.ndarray:
    return (value + np.pi) % _TWO_PI - np.pi


def _next_power_of_two(value: int) -> int:
    return 1 << (max(1, int(value)) - 1).bit_length()


def _periodic_hann(length: int) -> np.ndarray:
    if length <= 1:
        return np.ones(length, dtype=np.float32)
    phase = _TWO_PI * np.arange(length, dtype=np.float64) / float(length)
    return np.asarray(0.5 - 0.5 * np.cos(phase), dtype=np.float32)


def _ensure_numba_kernels():
    """Compile/load the numba kernels in the calling thread (see warm_timestretch_numba)."""
    global _NUMBA_KERNELS, _NUMBA_READY, _NUMBA_FAILED, _NUMBA_ERROR
    if _NUMBA_READY:
        return _NUMBA_KERNELS
    if _NUMBA_FAILED:
        return None

    with _NUMBA_LOCK:
        if _NUMBA_READY:
            return _NUMBA_KERNELS
        if _NUMBA_FAILED:
            return None
        try:
            from core.audio import _timestretch_numba as kernels

            kernels.warm()
            _NUMBA_KERNELS = kernels
            _NUMBA_READY = True
            return _NUMBA_KERNELS
        except Exception as exc:  # pragma: no cover - environment dependent
            _NUMBA_ERROR = exc
            _NUMBA_FAILED = True
            return None


def warm_timestretch_numba() -> bool:
    """Synchronously load/compile the Numba kernels.

    Call it outside the audio thread (the app does it on the main thread before
    enabling the stretch), so the audio thread never blocks on loading or compiling.
    A disk cache is used, so later launches normally load machine code (~0.5 s)
    instead of compiling. Failure is non-fatal and leaves the NumPy path (classic
    phase vocoder) active.
    """
    return _ensure_numba_kernels() is not None


def _ready_numba_kernels():
    """Lock-free audio-thread lookup of the compiled kernel module; never imports,
    compiles, or waits."""
    return _NUMBA_KERNELS if _NUMBA_READY else None


def timestretch_numba_error() -> BaseException | None:
    """Return the optional acceleration error after a failed warm-up."""
    return _NUMBA_ERROR


@dataclass(frozen=True, slots=True)
class TimeStretchConfig:
    """Quality and CPU controls for :class:`StreamingTimeStretcher`.

    Defaults follow the prototype's defaults.  Durations are converted to
    samples of ``sample_rate`` once during construction.
    """

    # Analysis/synthesis lattice.  The FFT covers the window times
    # frequency_oversampling (93 ms at 48 kHz -> 8192 points).  The prototype's
    # 5.33 ms hop costs ~40 % of a core at 48 kHz stereo (seek restart ~70 ms);
    # 10.67 ms (8.7x window overlap, 16x FFT redundancy) halves that.
    window_ms: float = 93.0
    synthesis_hop_ms: float = 10.67
    frequency_oversampling: float = 1.5
    # SELEBI window squeezing: S/V = 1 - lambda * r^gamma * (1 - 1/alpha), and for
    # confirmed kicks at least kick^2 * (1 - 1/kick_profile_stretch).
    minimum_window_ratio: float = 0.70
    squeeze_strength: float = 1.50
    squeeze_gamma: float = 1.50
    kick_profile_stretch: float = 2.0
    window_quantum: int = 16
    # Percussive-event detector.
    detector_magnitude_floor: float = 0.01
    mpd_lower: float = 0.50
    mpd_upper: float = 0.75
    detector_median_frames: int = 5
    peak_prominence: float = 0.10
    prominence_window_ms: float = 90.0  # bounded prominence search, each side
    kick_low_hz: float = 180.0
    # Phase generation.
    phase_lock_strength: float = 1.0
    stereo_coherence: bool = True
    rtpghi_tolerance: float = 1e-8
    numba_acceleration: bool = True
    # Attack preservation.
    dry_transient_amount: float = 0.70
    dry_transient_threshold: float = 0.40
    dry_transient_full_strength: float = 0.68
    dry_transient_pre_ms: float = 4.0
    dry_transient_post_ms: float = 24.0
    dry_transient_highpass_hz: float = 500.0
    kick_transient_bypass: float = 1.0
    kick_transient_pre_ms: float = 10.0
    kick_transient_post_ms: float = 24.0
    kick_transient_highpass_hz: float = 220.0
    attack_filter_pad_ms: float = 10.0

    def validate(self) -> None:
        if self.window_ms <= 0.0 or self.synthesis_hop_ms <= 0.0:
            raise ValueError("window_ms and synthesis_hop_ms must be > 0")
        if self.frequency_oversampling < 1.0:
            raise ValueError("frequency_oversampling must be >= 1")
        if not 0.0 < self.minimum_window_ratio <= 1.0:
            raise ValueError("minimum_window_ratio must be in (0, 1]")
        if self.squeeze_strength < 0.0 or self.squeeze_gamma <= 0.0:
            raise ValueError("squeeze strength/gamma must be nonnegative/positive")
        if self.kick_profile_stretch < 1.0:
            raise ValueError("kick_profile_stretch must be >= 1")
        if not 0.0 <= self.detector_magnitude_floor <= 1.0:
            raise ValueError("detector_magnitude_floor must be in [0, 1]")
        if self.mpd_lower <= 0.0 or self.mpd_upper <= 0.0:
            raise ValueError("MPD bounds must be > 0")
        if self.detector_median_frames < 1:
            raise ValueError("detector_median_frames must be >= 1")
        if not 0.0 <= self.phase_lock_strength <= 1.0:
            raise ValueError("phase_lock_strength must be in [0, 1]")
        if self.window_quantum < 2:
            raise ValueError("window_quantum must be >= 2")
        if not 0.0 <= self.dry_transient_amount <= 1.0:
            raise ValueError("dry_transient_amount must be in [0, 1]")
        if self.dry_transient_full_strength <= self.dry_transient_threshold:
            raise ValueError("dry transient full strength must exceed its threshold")
        if not 0.0 <= self.kick_transient_bypass <= 1.0:
            raise ValueError("kick_transient_bypass must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class _Event:
    frame: int        # detector frame; input centre = frame * detector hop
    strength: float   # median-filtered percussive ratio at the peak
    kick: float       # low-frequency kick score in [0, 1]


class _PercussionTrack:
    """SELEBI percussive-event detector on a fixed input-domain grid.

    Frame ``j`` is centred on input sample ``j * hop``.  Frames are computed lazily
    and ahead of the caller: a peak at ``j`` is confirmed once the median-filtered
    ratio is known up to ``j + prominence_frames``.
    """

    def __init__(
        self, sample_rate: int, window: int, hop: int, n_fft: int, config: TimeStretchConfig
    ) -> None:
        self.hop = hop
        self._window = window
        self._n_fft = n_fft
        self._config = config
        bins = n_fft // 2 + 1
        self._hann = _periodic_hann(window)
        omega = _TWO_PI * np.arange(bins, dtype=np.float64) / n_fft
        self._ramp = np.asarray(np.exp(1j * omega * (window // 2)), dtype=np.complex64)
        self._kick_bin = int(np.clip(round(config.kick_low_hz * n_fft / sample_rate), 2, bins - 1))
        wide_hz = min(12000.0, 0.5 * sample_rate)
        self._wide_bin = min(bins, max(self._kick_bin + 1, int(round(wide_hz * n_fft / sample_rate))))
        self._half_median = int(config.detector_median_frames) // 2
        self.prominence_frames = max(
            1, int(round(config.prominence_window_ms * sample_rate / 1000.0 / hop))
        )
        # The prototype normalises magnitudes by the whole track's maximum; here a
        # peak level that halves every 4 s stands in for it.
        self._level_decay = 0.5 ** (hop / (4.0 * sample_rate))
        self._no_phase = np.zeros(bins, dtype=np.float64)
        self.start(0)

    def start(self, first: int) -> None:
        """Start a new run of frames at ``first`` (after a seek or a new source)."""
        self._first = int(first)
        self._next_raw = self._first
        self._checked = self._first + 1
        self._raw: dict[int, float] = {}
        self._low: dict[int, float] = {}
        self._median: dict[int, float] = {}
        self._previous_phase: np.ndarray | None = None
        self._level = 0.0
        self.events: dict[int, _Event] = {}

    def ensure(self, source: np.ndarray, frame: int) -> None:
        """Confirm or reject peaks at every frame up to ``frame``."""
        lookahead = self.prominence_frames + self._half_median + 1
        while self._checked <= frame:
            while self._next_raw <= self._checked + lookahead:
                self._compute_raw(source, self._next_raw)
                self._next_raw += 1
            self._check_peak(self._checked)
            self._checked += 1
        self._prune(frame)

    def _compute_raw(self, source: np.ndarray, j: int) -> None:
        window = self._window
        left = j * self.hop - window // 2
        frame = np.zeros(window, dtype=np.float32)
        lo = max(0, left)
        hi = min(int(source.shape[0]), left + window)
        if hi > lo:
            frame[lo - left:hi - left] = np.mean(source[lo:hi], axis=1)
        spectrum = np.asarray(rfft(frame * self._hann, n=self._n_fft) * self._ramp, dtype=np.complex64)
        kernels = _ready_numba_kernels() if self._config.numba_acceleration else None
        if kernels is not None:
            config = self._config
            has_previous = self._previous_phase is not None
            ratio, low_ratio, self._level, phase = kernels.detector_frame(
                spectrum,
                self._previous_phase if has_previous else self._no_phase,
                has_previous,
                int(self.hop),
                int(self._n_fft),
                float(config.detector_magnitude_floor),
                float(self._level),
                float(self._level_decay),
                float(config.mpd_lower),
                float(config.mpd_upper),
                int(self._kick_bin),
                int(self._wide_bin),
            )
            self._previous_phase = phase
            self._raw[j] = float(ratio)
            self._low[j] = float(low_ratio)
            return

        magnitude = np.abs(spectrum).astype(np.float64)
        phase = np.angle(spectrum).astype(np.float64)
        self._level = max(float(magnitude.max()), self._level * self._level_decay)

        ratio = 0.0
        if self._previous_phase is not None:
            config = self._config
            dt = _wrap_phase(phase - self._previous_phase) / float(self.hop)
            mixed = _wrap_phase(dt[1:] - dt[:-1]) / (_TWO_PI / self._n_fft)
            mpd = np.empty_like(phase)
            mpd[:-1] = mixed
            mpd[-1] = mixed[-1]
            mask = (
                (magnitude > config.detector_magnitude_floor * self._level)
                & (mpd > 1.0 - config.mpd_lower)
                & (mpd < 1.0 + config.mpd_upper)
            )
            ratio = float(np.sum(magnitude[mask]) / (np.sum(magnitude) + _EPS))
        self._previous_phase = phase

        power = magnitude * magnitude
        low = float(np.sum(power[1:self._kick_bin]))
        wide = float(np.sum(power[1:self._wide_bin]))
        self._raw[j] = ratio
        self._low[j] = low / (wide + _EPS)

    def _median_at(self, j: int) -> float:
        value = self._median.get(j)
        if value is None:
            h = self._half_median
            # Edges repeat the nearest frame (mode="nearest").
            window = [self._raw[min(max(k, self._first), self._next_raw - 1)] for k in range(j - h, j + h + 1)]
            value = float(np.median(window))
            self._median[j] = value
        return value

    def _check_peak(self, j: int) -> None:
        value = self._median_at(j)
        if not (value > self._median_at(j - 1) and value >= self._median_at(j + 1)):
            return
        # scipy.signal.peak_prominences with a bounded window: walk each side until a
        # higher sample or the bound.
        span = self.prominence_frames
        left_min = value
        for k in range(j - 1, max(self._first, j - span) - 1, -1):
            other = self._median_at(k)
            if other > value:
                break
            left_min = min(left_min, other)
        right_min = value
        for k in range(j + 1, j + span + 1):
            other = self._median_at(k)
            if other > value:
                break
            right_min = min(right_min, other)
        if value - max(left_min, right_min) < self._config.peak_prominence:
            return
        low_score = float(np.clip((self._low[j] - 0.25) / 0.35, 0.0, 1.0))
        percussion_score = float(np.clip((value - 0.30) / 0.30, 0.0, 1.0))
        self.events[j] = _Event(j, value, math.sqrt(low_score * percussion_score))

    def _prune(self, frame: int) -> None:
        keep_from = frame - 8 * (self.prominence_frames + self._half_median + 8)
        if len(self._raw) < 4096 or keep_from <= self._first:
            return
        for table in (self._raw, self._low, self._median, self.events):
            for key in [key for key in table if key < keep_from]:
                del table[key]
        self._first = keep_from


class StreamingTimeStretcher:
    """Stateful pull renderer compatible with ``SpeedResampler.render``.

    The source PCM remains owned by the caller and must be a ``[frames,
    channels]`` array.  Consecutive calls should pass the position returned by
    ``expected_input_position``.  A seek or a source change is detected
    automatically and starts a new phase history with pre-roll.  A factor change
    keeps the phase history and the queued output (only later analysis frames
    follow the new factor), so a moving tempo control stays click-free.  At unity
    the input is read directly; the stream crossfades in when the factor leaves
    unity, and back out (one short crossfade) at the next onset once back at unity.

    This class is not thread-safe.  It is intended to be owned by one audio
    render thread, matching ``PCMFeeder``'s current execution model.
    """

    _UNITY_EPS: Final = 1e-8
    _POSITION_EPS: Final = 1e-4
    _COMPACT_AFTER: Final = 32_768
    # Stream <-> direct-read crossfade length (output frames).
    _XFADE_FRAMES: Final = 256
    # After a tempo change the analysis position moves at most this fraction of a hop
    # per frame faster/slower to reach the new mapping (a brief ~2 % tempo nudge).
    _CATCHUP_RATIO: Final = 0.02
    # Back at unity the stream keeps running unchanged (its phases differ from the
    # input's) and hands over to the direct read with one short crossfade at the next
    # onset, which masks the crossfade; without an onset it hands over after
    # _HANDOVER_TIMEOUT_SEC. Pulling the phases toward the input's instead is heard as
    # flanging while they converge.
    # Onset: normalised positive spectral flux sum(max(0, |X_t| - |X_t-1|)) / sum(|X_t|)
    # above _ONSET_FLUX (median ~0.11 on music; 0.2 is exceeded ~3-4 times a second).
    _ONSET_FLUX: Final = 0.2
    _HANDOVER_TIMEOUT_SEC: Final = 1.0
    # Kick-attack bypass: confidence ramps from 0 at kick score 0.70 to 1 at 0.95.
    _KICK_CONFIDENCE_FROM: Final = 0.70
    _KICK_CONFIDENCE_SPAN: Final = 0.25

    def __init__(
        self,
        channels: int,
        sample_rate: int = 48_000,
        config: TimeStretchConfig | None = None,
    ) -> None:
        self.channels = int(channels)
        self.sample_rate = int(sample_rate)
        if self.channels <= 0:
            raise ValueError("channels must be > 0")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be > 0")

        self.config = config or TimeStretchConfig()
        self.config.validate()
        config = self.config

        requested_window = max(128, int(round(self.sample_rate * config.window_ms / 1000.0)))
        self._max_window = requested_window + (requested_window & 1)
        self._hop = max(1, int(round(self.sample_rate * config.synthesis_hop_ms / 1000.0)))
        self._n_fft = _next_power_of_two(
            int(math.ceil(self._max_window * config.frequency_oversampling))
        )
        self._bins = self._n_fft // 2 + 1
        self._omega = np.asarray(
            _TWO_PI * np.arange(self._bins, dtype=np.float64) / self._n_fft,
            dtype=np.float64,
        )
        self._min_window = max(16, int(math.ceil(config.minimum_window_ratio * self._max_window)))
        # Events shorten windows within +/- half a window (SELEBI n_half on the
        # detector grid).
        self._detector = _PercussionTrack(
            self.sample_rate, self._max_window, self._hop, self._n_fft, config
        )
        self._event_reach = int(math.ceil(self._max_window / (2.0 * self._hop)))

        self._handover_timeout = max(
            1, int(round(self.sample_rate * self._HANDOVER_TIMEOUT_SEC / self._hop))
        )

        def ms(value: float) -> int:
            return max(1, int(round(value * self.sample_rate / 1000.0)))

        def highpass(hz: float):
            if hz <= 0.0:
                return None
            return butter(4, hz, btype="highpass", fs=self.sample_rate, output="sos")

        self._dry = (ms(config.dry_transient_pre_ms), ms(config.dry_transient_post_ms),
                     ms(3.0), highpass(config.dry_transient_highpass_hz))
        self._kick = (ms(config.kick_transient_pre_ms), ms(config.kick_transient_post_ms),
                      ms(4.0), highpass(config.kick_transient_highpass_hz))
        self._filter_pad = ms(config.attack_filter_pad_ms)
        self._envelope_window = max(3, ms(1.0))
        # Output is synthesised this far past the read point, so an event's
        # correction (its pre/post span plus filter padding) is complete before any
        # of it is read.
        self._attack_ahead = (
            max(self._dry[0], self._kick[0]) + max(self._dry[1], self._kick[1])
            + self._filter_pad + 1
        )

        self._window_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

        self._source_token: tuple[int, int, int] | None = None
        self._factor: float | None = None
        self._expected_pos: float | None = None
        self._next_analysis_center = 0.0
        self._next_synthesis_center = 0
        self._previous_analysis_center: int | None = None
        self._previous_analysis_phase: np.ndarray | None = None
        self._previous_synthesis_phase: np.ndarray | None = None
        self._previous_joint: np.ndarray | None = None

        self._output_read = 0
        self._finalized_until = 0
        self._ola_base = 0
        self._ola = np.zeros((0, self.channels), dtype=np.float32)
        self._weight = np.zeros(0, dtype=np.float32)
        self._correction = np.zeros((0, self.channels), dtype=np.float32)

        # Frame centres (input, output) for mapping events to output time, and events
        # waiting for their output span to be synthesised.
        self._frame_map: deque[tuple[int, int]] = deque(maxlen=512)
        self._next_event_frame = 0
        self._pending_events: deque[tuple[int, _Event]] = deque()

        # Stream vs. direct read (unity tempo), crossfaded by _stream_gain.
        self._streaming = False
        self._stream_gain = 0.0
        self._gain_target = 0.0
        self._unity_frames = 0  # stream frames analysed at unity
        self._handover_at: int | None = None  # output frame where the fade-out starts
        # Output->input mapping since the last tempo change (local output frames).
        self._anchor_in = 0.0
        self._anchor_out = 0

    @property
    def expected_input_position(self) -> float | None:
        """Input position expected by the next sequential ``render`` call."""
        return self._expected_pos

    @property
    def synthesis_hop(self) -> int:
        return self._hop

    @property
    def window_length(self) -> int:
        return self._max_window

    @property
    def fft_length(self) -> int:
        return self._n_fft

    def reset(self) -> None:
        """Forget phase/FIFO state; the next render call will pre-roll again."""
        self._source_token = None
        self._factor = None
        self._expected_pos = None
        self._previous_analysis_center = None
        self._previous_analysis_phase = None
        self._previous_synthesis_phase = None
        self._previous_joint = None
        self._output_read = 0
        self._finalized_until = 0
        self._ola_base = 0
        self._ola = np.zeros((0, self.channels), dtype=np.float32)
        self._weight = np.zeros(0, dtype=np.float32)
        self._correction = np.zeros((0, self.channels), dtype=np.float32)
        self._frame_map.clear()
        self._pending_events.clear()
        self._streaming = False
        self._stream_gain = 0.0
        self._gain_target = 0.0
        self._unity_frames = 0
        self._handover_at = None

    def render(
        self,
        pcm: np.ndarray,
        pos: float,
        factor: float,
        max_out_frames: int,
    ) -> np.ndarray:
        """Render up to ``max_out_frames`` beginning at absolute input ``pos``.

        As with ``SpeedResampler``, the next sequential input position is
        ``pos + factor * len(result)``.  End-of-track sizing follows that same
        contract, including a partial final block.
        """
        source = np.asarray(pcm)
        if source.ndim != 2 or source.shape[1] != self.channels:
            raise ValueError(
                f"pcm must have shape [frames, {self.channels}], got {source.shape}"
            )
        pos = float(pos)
        factor = float(factor)
        maximum = int(max_out_frames)
        if not math.isfinite(pos) or pos < 0.0:
            raise ValueError("pos must be a finite nonnegative input-frame position")
        if not math.isfinite(factor) or factor <= 0.0:
            raise ValueError("factor must be finite and > 0")
        if maximum <= 0 or pos >= source.shape[0]:
            return np.zeros((0, self.channels), dtype=np.float32)

        available = (source.shape[0] - pos) / factor
        # Sequential floating-point position updates can leave the final
        # quotient a few ulps above an integer.  Snap only that numerical dust
        # so block size cannot add a duplicated end sample.
        snap = 32.0 * np.finfo(np.float64).eps * max(
            1.0, float(source.shape[0]), abs(available)
        )
        count = min(maximum, int(math.ceil(available - snap)))
        if count <= 0:
            return np.zeros((0, self.channels), dtype=np.float32)

        token = self._pcm_token(source)
        unity = abs(factor - 1.0) <= self._UNITY_EPS
        sequential = self._continues(token, pos, factor)
        self._source_token = token
        if not sequential:
            # Seek, jump or new source: the caller crossfades those itself, so the
            # renderer switches at once (direct read at unity, fresh stream otherwise).
            self._streaming = False
            self._stream_gain = 0.0 if unity else 1.0

        if not self._streaming:
            if unity:
                result = self._read_direct(source, pos, count)
                self._factor = factor
                self._expected_pos = pos + factor * len(result)
                return result
            self._streaming = True
            if not sequential:
                self._start_stream(token, pos, factor)
            else:
                # Leaving unity (tempo slider moved off 100 %): pre-roll at unity, so
                # the stream reproduces the input up to output frame 0 and continues it
                # like a tempo change mid-stream (the short crossfade is between two
                # copies of the same input).
                self._start_stream(token, pos, 1.0)
                while self._next_synthesis_center < 0:
                    self._process_frame(source)
                self._retime(pos, factor)
                self._stream_gain = 0.0
        elif not math.isclose(factor, self._factor, rel_tol=1e-7, abs_tol=1e-9):
            self._retime(pos, factor)

        # Off unity the stream is (faded back) in; at unity _process_frame schedules
        # the handover to the direct read.
        if not unity:
            self._unity_frames = 0
            self._handover_at = None
            self._gain_target = 1.0

        start = self._output_read
        while self._finalized_until < start + count + self._attack_ahead:
            self._process_frame(source)
        self._apply_due_events(source)
        if unity and self._handover_at is not None:
            self._gain_target = 0.0
        result = self._read_output(count)
        self._compact_output()

        if self._stream_gain < 1.0 or self._gain_target < 1.0:
            result = self._mix_direct(source, pos, start, result)
            if self._stream_gain <= 0.0 and self._gain_target <= 0.0:
                self._streaming = False
        self._expected_pos = pos + factor * len(result)
        return result

    @staticmethod
    def _pcm_token(source: np.ndarray) -> tuple[int, int, int]:
        pointer = int(source.__array_interface__["data"][0])
        return pointer, int(source.shape[0]), int(source.shape[1])

    def _continues(
        self, token: tuple[int, int, int], pos: float, factor: float
    ) -> bool:
        """True when ``pos`` continues the previous render of the same source."""
        if self._source_token != token or self._expected_pos is None:
            return False
        return math.isclose(
            pos,
            self._expected_pos,
            rel_tol=0.0,
            abs_tol=max(self._POSITION_EPS, factor * self._POSITION_EPS),
        )

    def _retime(self, pos: float, factor: float) -> None:
        """Tempo change mid-stream: keep the phase history and the queued output.

        From here output frame ``_output_read`` maps to input ``pos`` at the new
        factor. Frames already overlap-added ahead of the read point (about half a
        window plus the attack look-ahead) were analysed at the old factor; the
        analysis centres catch up with the new mapping gradually (see
        ``_process_frame``) instead of jumping.
        """
        self._factor = factor
        self._anchor_in = pos
        self._anchor_out = self._output_read

    def _mix_direct(
        self, source: np.ndarray, pos: float, start: int, stream: np.ndarray
    ) -> np.ndarray:
        """Crossfade between the stream and the direct read, moving the stream gain
        linearly toward its target over ``_XFADE_FRAMES`` output frames. A fade-out
        starts at output frame ``_handover_at`` (``start`` is this block's first)."""
        count = len(stream)
        step = 1.0 / self._XFADE_FRAMES
        direction = 1.0 if self._gain_target > self._stream_gain else -1.0
        first = start
        if direction < 0.0 and self._handover_at is not None:
            first = max(start, self._handover_at)
        moved = np.maximum(0, np.arange(start, start + count) - first + 1)
        # The target is 0 or 1, so clipping to [0, 1] stops the ramp at the target.
        gain = np.clip(self._stream_gain + direction * step * moved, 0.0, 1.0)
        self._stream_gain = float(gain[-1])
        # Crossfades happen at (or just off) unity, where the stream follows the input
        # at unity rate, so the direct side is read at unity too.
        direct = self._read_direct(source, pos, count)
        g = gain.astype(np.float32)[:, None]
        return np.ascontiguousarray(direct + g * (stream - direct), dtype=np.float32)

    def _start_stream(
        self, token: tuple[int, int, int], pos: float, factor: float
    ) -> None:
        self._source_token = token
        self._factor = factor
        self._expected_pos = pos
        self._previous_analysis_center = None
        self._previous_analysis_phase = None
        self._previous_synthesis_phase = None
        self._previous_joint = None
        self._unity_frames = 0
        self._handover_at = None
        self._gain_target = 1.0

        # Pre-roll fills the left half of the first audible WOLA window and
        # establishes phase history before local output frame zero.
        pre_roll = math.ceil((self._max_window // 2) / self._hop)
        self._next_synthesis_center = -pre_roll * self._hop
        self._next_analysis_center = pos + factor * self._next_synthesis_center
        self._anchor_in = pos
        self._anchor_out = 0

        # The detector starts early enough for its median and prominence context.
        first_center = self._next_analysis_center
        detector = self._detector
        detector.start(
            math.floor(first_center / detector.hop)
            - self._event_reach - detector.prominence_frames - 2
        )
        self._next_event_frame = math.floor(first_center / detector.hop)
        self._frame_map.clear()
        self._pending_events.clear()

        self._output_read = 0
        self._finalized_until = 0
        self._ola_base = 0
        initial = self._max_window + 2 * self._hop + self._attack_ahead
        self._ola = np.zeros((initial, self.channels), dtype=np.float32)
        self._weight = np.zeros(initial, dtype=np.float32)
        self._correction = np.zeros((initial, self.channels), dtype=np.float32)

    def _process_frame(self, source: np.ndarray) -> None:
        assert self._factor is not None
        # Same rounding as _read_direct, so both read the same input samples at unity.
        input_center = math.floor(self._next_analysis_center + 0.5)
        output_center = self._next_synthesis_center
        if self._previous_analysis_center is None:
            analysis_hop = max(1, int(round(self._factor * self._hop)))
        else:
            analysis_hop = max(1, input_center - self._previous_analysis_center)

        window_length, lock_strength = self._frame_window(source, input_center)
        coeff = self._analyze(source, input_center, window_length)
        kernels = _ready_numba_kernels() if self.config.numba_acceleration else None
        if kernels is not None:
            analysis_phase, magnitude, joint_magnitude = kernels.polar(coeff)
        else:
            analysis_phase = np.angle(coeff).astype(np.float64)
            magnitude = np.abs(coeff).astype(np.float32)
            joint_magnitude = np.sqrt(
                np.sum(magnitude * magnitude, axis=0, dtype=np.float32)
            )
        synthesis_phase = self._propagate_phase(
            analysis_phase, magnitude, joint_magnitude, analysis_hop, lock_strength
        )
        if abs(self._factor - 1.0) <= self._UNITY_EPS and self._handover_at is None:
            self._unity_frames += 1
            previous = self._previous_joint
            flux = 0.0
            if previous is not None:
                flux = float(
                    np.sum(np.maximum(joint_magnitude - previous, 0.0))
                    / (np.sum(joint_magnitude) + _EPS)
                )
            if flux > self._ONSET_FLUX or self._unity_frames >= self._handover_timeout:
                # The onset is under this frame's centre; nothing before it has been
                # read yet (the read point trails the frames by half a window).
                self._handover_at = max(output_center, self._output_read)
        if kernels is not None:
            synthesis_coeff = kernels.rect(magnitude, synthesis_phase)
        else:
            synthesis_coeff = np.asarray(
                magnitude * np.exp(1j * synthesis_phase), dtype=np.complex64
            )
        self._overlap_add(synthesis_coeff, output_center, window_length)

        self._frame_map.append((input_center, output_center))
        self._queue_events(input_center)
        self._previous_analysis_center = input_center
        self._previous_analysis_phase = analysis_phase
        self._previous_synthesis_phase = synthesis_phase
        self._previous_joint = joint_magnitude
        self._next_synthesis_center += self._hop
        # Follow the output->input mapping of the latest tempo change; after a change
        # the offset is closed by at most _CATCHUP_RATIO of a hop per frame.
        nominal = self._next_analysis_center + self._factor * self._hop
        mapped = self._anchor_in + self._factor * (
            self._next_synthesis_center - self._anchor_out
        )
        limit = self._CATCHUP_RATIO * self._factor * self._hop
        self._next_analysis_center = nominal + min(limit, max(-limit, mapped - nominal))

        # A future frame can reach at most half the maximum window backwards.
        # Everything before its left edge is now immutable and safe to return.
        self._finalized_until = max(
            self._finalized_until,
            self._next_synthesis_center - self._max_window // 2,
        )

    def _frame_window(self, source: np.ndarray, input_center: int) -> tuple[int, float]:
        """SELEBI window length at ``input_center`` and the identity-lock strength.

        Around each event the window follows long -> short -> long: it is
        ``max(S_k, 2 * distance + V - 2 * hop * reach)``, so it starts shrinking
        ``reach`` hops before the event (look-ahead).  Windows are only shortened
        when stretching (alpha > 1), as in the prototype.
        """
        detector = self._detector
        center_frame = int(round(input_center / detector.hop))
        detector.ensure(source, center_frame + self._event_reach + 1)
        window = self._max_window
        alpha = 1.0 / self._factor
        strength = self.config.phase_lock_strength
        if alpha <= 1.0 + self._UNITY_EPS:
            return window, strength

        config = self.config
        max_reduction = 1.0 - config.minimum_window_ratio
        kick_profile = 1.0 - 1.0 / config.kick_profile_stretch
        ramp_offset = self._max_window - 2.0 * detector.hop * self._event_reach
        shortest = float(window)
        for frame in range(center_frame - self._event_reach - 1, center_frame + self._event_reach + 2):
            event = detector.events.get(frame)
            if event is None:
                continue
            compression = min(max_reduction, max(
                config.squeeze_strength * event.strength ** config.squeeze_gamma * (1.0 - 1.0 / alpha),
                event.kick * event.kick * kick_profile,
            ))
            target = math.floor(self._max_window * (1.0 - compression))
            target = max(self._min_window, min(self._max_window, target))
            distance = abs(input_center - frame * detector.hop)
            shortest = min(shortest, max(target, 2.0 * distance + ramp_offset))
        if shortest < window:
            quantum = int(config.window_quantum)
            window = int(round(shortest / quantum)) * quantum
            window = max(self._min_window, min(self._max_window, window))
            window += window & 1

        # Adaptive locking: none in the most percussive frames of this stretch.
        reduction_span = self._max_window * (1.0 - 1.0 / alpha)
        if reduction_span > 0.5:
            percussiveness = min(1.0, max(0.0, (self._max_window - window) / reduction_span))
            strength *= 1.0 - percussiveness
        return window, strength

    def _analyze(
        self, source: np.ndarray, center: int, window_length: int
    ) -> np.ndarray:
        window, center_ramp, _ = self._window_data(window_length)
        frame = np.zeros((window_length, self.channels), dtype=np.float32)
        left = center - window_length // 2
        src_start = max(0, left)
        src_stop = min(int(source.shape[0]), left + window_length)
        if src_stop > src_start:
            dst_start = src_start - left
            frame[dst_start:dst_start + src_stop - src_start] = source[
                src_start:src_stop
            ]
        frame *= window[:, None]
        spectrum = rfft(frame, n=self._n_fft, axis=0).T
        # Channel-major C layout: one numba specialization, contiguous bin loops.
        return np.ascontiguousarray(spectrum * center_ramp[None, :], dtype=np.complex64)

    def _propagate_phase(
        self,
        analysis_phase: np.ndarray,
        magnitude: np.ndarray,
        joint_magnitude: np.ndarray,
        analysis_hop: int,
        lock_strength: float,
    ) -> np.ndarray:
        previous_analysis = self._previous_analysis_phase
        previous_synthesis = self._previous_synthesis_phase
        if previous_analysis is None or previous_synthesis is None:
            return analysis_phase.copy()

        if self.config.numba_acceleration:
            # Never compile or wait for the warm-up lock in the render path; until
            # the warm-up completes the classic phase vocoder below is used.
            kernels = _ready_numba_kernels()
            if kernels is not None:
                return kernels.rtpghi_frame(
                    analysis_phase,
                    magnitude,
                    joint_magnitude,
                    previous_analysis,
                    previous_synthesis,
                    self._previous_joint,
                    int(analysis_hop),
                    int(self._hop),
                    int(self._n_fft),
                    float(self.config.rtpghi_tolerance),
                    float(lock_strength),
                    bool(self.config.stereo_coherence),
                )

        local_stretch = self._hop / float(analysis_hop)
        residual = _wrap_phase(
            analysis_phase
            - previous_analysis
            - self._omega[None, :] * analysis_hop
        )
        synthesis = (
            previous_synthesis
            + self._omega[None, :] * self._hop
            + local_stretch * residual
        )
        if lock_strength > 0.0:
            synthesis = self._identity_phase_lock(
                synthesis, analysis_phase, joint_magnitude, lock_strength
            )
        if self.config.stereo_coherence and self.channels > 1:
            rotation = _wrap_phase(synthesis - analysis_phase)
            common = np.angle(np.sum(magnitude * np.exp(1j * rotation), axis=0))
            synthesis = analysis_phase + common[None, :]
        return np.asarray(_wrap_phase(synthesis), dtype=np.float64)

    @staticmethod
    def _identity_phase_lock(
        synthesis: np.ndarray,
        analysis: np.ndarray,
        joint_magnitude: np.ndarray,
        strength: float,
    ) -> np.ndarray:
        if joint_magnitude.size < 3:
            return synthesis
        threshold = 1e-3 * max(float(np.max(joint_magnitude)), _EPS)
        peaks = np.flatnonzero(
            (joint_magnitude[1:-1] >= joint_magnitude[:-2])
            & (joint_magnitude[1:-1] > joint_magnitude[2:])
            & (joint_magnitude[1:-1] > threshold)
        ) + 1
        if peaks.size == 0:
            peaks = np.asarray([int(np.argmax(joint_magnitude))], dtype=np.int64)

        bins = np.arange(joint_magnitude.size)
        boundaries = ((peaks[:-1] + peaks[1:]) // 2) + 1
        owners = peaks[np.searchsorted(boundaries, bins, side="right")]
        locked = synthesis[:, owners] + _wrap_phase(
            analysis - analysis[:, owners]
        )
        return synthesis + strength * _wrap_phase(locked - synthesis)

    # Attack preservation
    def _queue_events(self, input_center: int) -> None:
        """Queue events whose input centre the analysis has now passed, with their
        output position interpolated from the frame centres."""
        detector = self._detector
        last = math.floor(input_center / detector.hop)
        if last < self._next_event_frame:
            return
        centers = None
        for frame in range(self._next_event_frame, last + 1):
            event = detector.events.get(frame)
            if event is None:
                continue
            if centers is None:
                centers = np.asarray(self._frame_map, dtype=np.float64)
            destination = int(round(np.interp(
                float(frame * detector.hop), centers[:, 0], centers[:, 1]
            )))
            self._pending_events.append((destination, event))
        self._next_event_frame = last + 1

    def _apply_due_events(self, source: np.ndarray) -> None:
        """Apply queued events whose correction span is fully synthesised."""
        pending = self._pending_events
        while pending:
            destination, event = pending[0]
            # Due once its whole filtered span [.., destination + post + pad] is final.
            if destination + self._attack_ahead - max(self._dry[0], self._kick[0]) > self._finalized_until:
                return
            pending.popleft()
            config = self.config
            center = event.frame * self._detector.hop
            span = config.dry_transient_full_strength - config.dry_transient_threshold
            confidence = min(1.0, max(0.0, (event.strength - config.dry_transient_threshold) / span))
            if config.dry_transient_amount > 0.0 and confidence > 0.0:
                self._restore_attack(
                    source, center, destination, self._dry,
                    config.dry_transient_amount * confidence, match_envelope=True,
                )
            confidence = min(1.0, max(0.0, (event.kick - self._KICK_CONFIDENCE_FROM)
                                      / self._KICK_CONFIDENCE_SPAN))
            if config.kick_transient_bypass > 0.0 and confidence > 0.0:
                self._restore_attack(
                    source, center, destination, self._kick,
                    config.kick_transient_bypass * confidence, match_envelope=False,
                )

    def _restore_attack(
        self,
        source: np.ndarray,
        input_center: int,
        output_center: int,
        spec: tuple,
        amount: float,
        match_envelope: bool,
    ) -> None:
        """Pull the output's high band around one attack toward the unstretched source.

        ``match_envelope``: scale the output's own high band toward the source's
        1 ms RMS envelope (dry transients; the output carrier is kept).  Otherwise
        crossfade the source's high band in for the output's (kick attack bypass).
        Offsets are 1:1 in time around the event, as in the prototype.
        """
        pre, post, fade, sos = spec
        pad = self._filter_pad
        # Output span that may still change: not yet read, already final.
        lo = max(output_center - pre, self._output_read, self._ola_base)
        hi = min(output_center + post + 1, self._finalized_until)
        if hi <= lo:
            return
        y_lo = max(self._ola_base, lo - pad)
        y_hi = min(self._finalized_until, hi + pad)
        y = self._output_span(y_lo, y_hi)
        x_lo = input_center - (output_center - y_lo)
        x = np.zeros((y_hi - y_lo, self.channels), dtype=np.float64)
        s_lo = max(0, x_lo)
        s_hi = min(int(source.shape[0]), x_lo + len(x))
        if s_hi > s_lo:
            x[s_lo - x_lo:s_hi - x_lo] = source[s_lo:s_hi]
        if sos is not None:
            if len(y) <= 3 * (2 * len(sos) + 1):
                return
            y_band = sosfiltfilt(sos, y, axis=0)
            x_band = sosfiltfilt(sos, x, axis=0)
        else:
            y_band, x_band = y, x

        offsets = np.arange(lo, hi) - output_center  # -pre .. post
        shape = np.ones(len(offsets), dtype=np.float64)
        ramp = np.sin(np.linspace(0.0, 0.5 * np.pi, fade, endpoint=False)) ** 2
        rise = offsets + pre < fade
        fall = post - offsets < fade
        shape[rise] = ramp[(offsets + pre)[rise]]
        shape[fall] = np.minimum(shape[fall], ramp[(post - offsets)[fall]])
        index = np.arange(lo, hi) - y_lo

        if match_envelope:
            def envelope(band: np.ndarray) -> np.ndarray:
                power = np.mean(band * band, axis=1)
                return np.sqrt(uniform_filter1d(power, size=self._envelope_window, mode="nearest") + _EPS)

            ratio = np.clip(envelope(x_band)[index] / (envelope(y_band)[index] + _EPS), 0.60, 1.80)
            delta = (amount * shape * (ratio - 1.0))[:, None] * y_band[index]
        else:
            delta = (amount * shape)[:, None] * (x_band[index] - y_band[index])
        start = lo - self._ola_base
        self._correction[start:start + len(delta)] += delta.astype(np.float32)

    def _output_span(self, lo: int, hi: int) -> np.ndarray:
        """Normalised output (with corrections) for absolute output frames [lo, hi)."""
        a = lo - self._ola_base
        b = hi - self._ola_base
        weight = self._weight[a:b].astype(np.float64)
        out = np.zeros((b - a, self.channels), dtype=np.float64)
        valid = weight > 1e-10
        out[valid] = self._ola[a:b][valid] / weight[valid, None]
        return out + self._correction[a:b]

    # Overlap-add output FIFO
    def _overlap_add(
        self, coeff: np.ndarray, center: int, window_length: int
    ) -> None:
        window, _, inverse_ramp = self._window_data(window_length)
        raw = np.asarray(coeff * inverse_ramp[None, :], dtype=np.complex64)
        frame = irfft(raw, n=self._n_fft, axis=1)[:, :window_length].T
        frame = np.asarray(frame, dtype=np.float32)

        left = center - window_length // 2
        right = left + window_length
        src_start = 0
        if left < self._ola_base:
            src_start = self._ola_base - left
            left = self._ola_base
        if right <= left or src_start >= window_length:
            return
        self._ensure_output_capacity(right)
        start = left - self._ola_base
        stop = right - self._ola_base
        active_window = window[src_start:src_start + stop - start]
        self._ola[start:stop] += (
            frame[src_start:src_start + stop - start] * active_window[:, None]
        )
        self._weight[start:stop] += active_window * active_window

    def _ensure_output_capacity(self, absolute_stop: int) -> None:
        required = absolute_stop - self._ola_base
        if required <= len(self._weight):
            return
        growth = max(required - len(self._weight), self._max_window, 4096)
        self._ola = np.pad(self._ola, ((0, growth), (0, 0)))
        self._weight = np.pad(self._weight, (0, growth))
        self._correction = np.pad(self._correction, ((0, growth), (0, 0)))

    def _read_output(self, count: int) -> np.ndarray:
        start = self._output_read - self._ola_base
        stop = start + count
        result = np.zeros((count, self.channels), dtype=np.float32)
        weight = self._weight[start:stop]
        valid = weight > 1e-10
        result[valid] = self._ola[start:stop][valid] / weight[valid, None]
        result += self._correction[start:stop]
        self._output_read += count
        return np.ascontiguousarray(result)

    def _compact_output(self) -> None:
        # Keep the attack pre-span and filter padding behind the read point, so the
        # span an attack correction filters never depends on when compaction ran.
        keep = max(self._dry[0], self._kick[0]) + self._filter_pad
        consumed = self._output_read - keep - self._ola_base
        if consumed < self._COMPACT_AFTER:
            return
        self._ola = np.ascontiguousarray(self._ola[consumed:])
        self._weight = np.ascontiguousarray(self._weight[consumed:])
        self._correction = np.ascontiguousarray(self._correction[consumed:])
        self._ola_base += consumed

    def _window_data(
        self, window_length: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cached = self._window_cache.get(window_length)
        if cached is not None:
            return cached
        window = _periodic_hann(window_length)
        phase = self._omega * (window_length // 2)
        center_ramp = np.asarray(np.exp(1j * phase), dtype=np.complex64)
        inverse_ramp = np.asarray(np.exp(-1j * phase), dtype=np.complex64)
        cached = window, center_ramp, inverse_ramp
        self._window_cache[window_length] = cached
        return cached

    def _read_direct(
        self, source: np.ndarray, pos: float, count: int
    ) -> np.ndarray:
        """The input itself at unity: ``count`` samples from ``pos`` rounded.

        Rounded exactly like the stream's analysis centres, so a stream at unity
        phases reproduces these same samples; the offset from the fractional
        position is under half a sample. No interpolation: nothing is filtered.
        """
        start = math.floor(pos + 0.5)
        out = np.zeros((count, self.channels), dtype=np.float32)
        chunk = source[start:start + count]
        out[: len(chunk)] = chunk
        return out


__all__ = [
    "TimeStretchConfig",
    "StreamingTimeStretcher",
    "timestretch_numba_error",
    "warm_timestretch_numba",
]
