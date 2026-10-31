from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import butter, sosfiltfilt

from core.settings.store import current as current_settings
from views.timeline import CLOSE_WINDOW_MAX_SEC, SHORT_WINDOW_MAX_SEC, WINDOW_STEPS_SEC
from views.render.wave import (
    RGB_DETAIL_COLOR_SMOOTH_SEC,
    RGB_DETAIL_SAT_FLOOR,
    THREE_BAND_DETAIL_RMS_SEC,
    WAVE_CENTER_LINE_RMS_SEC,
    add_center_line,
    band_ranges,
    frame_minmax,
    paint_layers,
    render_wave_detail_preview,
    three_band_detail_layers,
    wave_colors,
    wave_detail_band_ref,
    wave_coverage_columns,
)
from .base import WAVEFORM_LANE, ViewPlugin, register_view

_WAVE_Y = float(WAVEFORM_LANE[0])
_WAVE_H = float(WAVEFORM_LANE[1] - WAVEFORM_LANE[0])


RENDER_CHUNK_SEC = 1
RENDER_HEIGHT = 96
RENDER_BATCH_SIZE = 4           # chunks rendered at once
RENDER_SPAWN_INTERVAL_MS = 30   # one more started per interval (playback keeps its CPU)
SCENE_CULL_BUFFER = 2  # extra chunks kept in scene on each side of viewport
# A layer's rendered chunks farther from the playhead than this many of the widest view it draws are dropped,
# to bound memory (the long views' 24 s: 60 s; the short ones' 9 s: 22.5 s; the close ones' 3 s: 7.5 s, a chunk
# of their fine frame 288 KB, the long one's 72 KB); they are rendered again when the view comes back to them.
# Except as many again of it shown most recently: going back to a place left a moment ago (a JumpCUE, a CUE point
# called) shows it at once. A layer no view width draws any more (its frame changed) keeps only what is near.
RENDER_KEEP_VIEWS = 2.5
# Audio rendered on each side of a chunk and cropped away, so the band filters,
# frame RMS and blur see the neighbouring audio: chunks join without seams.
RENDER_CONTEXT_SEC = 0.2
# Drawing frames per image column (bars averaged into one column).
FRAMES_PER_COLUMN = 2


@dataclass(frozen=True)
class _WaveStyle:
    """Waveform settings read once from the config (not per chunk)."""

    bands: tuple[tuple[float, float], ...]  # low / mid / high band edges (Hz)
    filter_order: int
    frame_ms: float                         # drawing frame length
    waveform_style: str                     # "rgb" / "3band" (viewconfig.waveform_style)
    # The loudest band's track maximum RMS (from the stored overview): the
    # common scale of the three-band layers.
    band_ref: float | None

    @staticmethod
    def frame_ms_for(view, window_sec: float) -> float:
        """Drawing frame (ms) of a main view ``window_sec`` wide."""
        if float(window_sec) <= CLOSE_WINDOW_MAX_SEC:
            return float(view.mainview_wave_frame_ms_close)
        short = float(window_sec) <= SHORT_WINDOW_MAX_SEC
        return float(view.mainview_wave_frame_ms_short if short else view.mainview_wave_frame_ms_long)

    @classmethod
    def from_config(cls, wave, window_sec: float) -> "_WaveStyle":
        cfg = current_settings()
        v = cfg.viewconfig
        return cls(
            bands=tuple((float(low), float(high)) for low, high in (v.env_lo, v.env_mid, v.env_hi)),
            filter_order=int(v.mainview_wave_filter_order),
            frame_ms=cls.frame_ms_for(v, window_sec),
            waveform_style=v.waveform_style,
            band_ref=wave_detail_band_ref(wave),
        )

    def frame_hop(self, sample_rate: int) -> int:
        return max(1, int(round(self.frame_ms * 1e-3 * float(sample_rate))))


class _WaveRenderSignals(QtCore.QObject):
    finished = QtCore.Signal(int, object)


@dataclass
class _WaveRenderResult:
    image: np.ndarray
    chunk_index: int
    start_sec: float  # time span the image covers (column-aligned)
    end_sec: float
    style: _WaveStyle  # the layer it belongs to


class _WaveRenderJob(QtCore.QRunnable):
    def __init__(
        self,
        request_id: int,
        pcm: np.ndarray,
        sample_rate: int,
        chunk_index: int,
        style: _WaveStyle,
        *,
        height: int,
    ) -> None:
        super().__init__()
        self.request_id = int(request_id)
        self.pcm = pcm
        self.sample_rate = max(1, int(sample_rate))
        self.chunk_index = max(0, int(chunk_index))
        self.style = style
        self.height = max(16, int(height))
        self.signals = _WaveRenderSignals()

    def run(self) -> None:
        image, start_sec, end_sec = _render_waveform_chunk(
            self.pcm, self.sample_rate, self.chunk_index, self.style, height=self.height
        )
        self.signals.finished.emit(
            self.request_id,
            _WaveRenderResult(image=image, chunk_index=self.chunk_index, start_sec=start_sec, end_sec=end_sec,
                              style=self.style),
        )


def _chunk_columns(chunk_index: int, sample_rate: int, n_samples: int, column_hop: int) -> tuple[int, int]:
    """Image columns ``[c0, c1)`` of a chunk on the track-wide column grid
    (column ``c`` covers samples ``[c * column_hop, (c + 1) * column_hop)``).
    Neighbouring chunks share their boundary column edge."""
    total = -(-int(n_samples) // column_hop)
    chunk_samples = int(round(RENDER_CHUNK_SEC * sample_rate))
    c0 = min(total, chunk_index * chunk_samples // column_hop)
    end = (chunk_index + 1) * chunk_samples
    c1 = total if end >= n_samples else min(total, end // column_hop)
    return c0, c1


def _render_waveform_chunk(
    pcm: np.ndarray, sample_rate: int, chunk_index: int, style: _WaveStyle, *, height: int
) -> tuple[np.ndarray, float, float]:
    """Waveform image ``[columns, height, 3]`` of one chunk and the time span it covers.

    RGB: thin bars (min..max of each frame of ``style.frame_ms``) colored by the
    low / mid / high band levels (smoothed over RGB_DETAIL_COLOR_SMOOTH_SEC),
    averaged FRAMES_PER_COLUMN to a column.
    3band: the band RMS (over THREE_BAND_DETAIL_RMS_SEC) of each column on the
    bands' common scale, drawn as mirrored layers (views.render.wave.three_band_detail_layers). Blurred slightly along time.
    The chunk is computed with RENDER_CONTEXT_SEC of audio on each side, on the
    track-wide frame grid, and cropped.
    """
    n_samples = int(pcm.shape[0])
    column_hop = FRAMES_PER_COLUMN * style.frame_hop(sample_rate)
    c0, c1 = _chunk_columns(chunk_index, sample_rate, n_samples, column_hop)
    start_sec, end_sec = c0 * column_hop / sample_rate, c1 * column_hop / sample_rate
    if c1 <= c0:
        return np.zeros((1, height, 3), dtype=np.uint8), start_sec, max(end_sec, start_sec + 1e-3)
    return render_columns(pcm, sample_rate, c0, c1, style, height=height), start_sec, end_sec


def column_hop(style: _WaveStyle, sample_rate: int) -> int:
    """Samples per image column (column ``c`` covers ``[c * hop, (c + 1) * hop)`` of the track)."""
    return FRAMES_PER_COLUMN * style.frame_hop(sample_rate)


def render_columns(pcm: np.ndarray, sample_rate: int, c0: int, c1: int, style: _WaveStyle, *,
                   height: int) -> np.ndarray:
    """Waveform image ``[c1 - c0, height, 3]`` of the track's columns ``[c0, c1)`` (column_hop's grid), drawn as
    the main view draws its chunks (_render_waveform_chunk): the settings' preview draws any span with it."""
    n_samples = int(pcm.shape[0])
    frame_hop = style.frame_hop(sample_rate)
    column_hop = FRAMES_PER_COLUMN * frame_hop
    context = int(np.ceil(RENDER_CONTEXT_SEC * sample_rate / column_hop))
    p0 = max(0, c0 - context)
    p1 = c1 + context
    segment = np.asarray(pcm[p0 * column_hop: min(n_samples, p1 * column_hop)], dtype=np.float32)
    mono = segment if segment.ndim == 1 else segment.mean(axis=1, dtype=np.float32)

    n_frames = -(-mono.size // frame_hop)
    starts = np.arange(n_frames) * frame_hop
    counts = np.diff(np.append(starts, mono.size)).astype(np.float64)
    nyq = 0.5 * float(sample_rate)
    three_band = style.waveform_style == "3band" and bool(style.band_ref)
    window = max(1, int(round(THREE_BAND_DETAIL_RMS_SEC * sample_rate)))
    centers = np.minimum(starts + frame_hop // 2, mono.size - 1)
    band_rms = []
    for low, high in style.bands:
        filtered = _band_filter(mono, sample_rate, low, min(high, nyq * 0.98), style.filter_order)
        square = np.square(filtered, dtype=np.float64)
        if three_band:
            band_rms.append(np.sqrt(np.maximum(uniform_filter1d(square, size=window)[centers], 0.0)))
        else:
            band_rms.append(np.sqrt(np.add.reduceat(square, starts) / counts))
    column = np.arange(n_frames) // FRAMES_PER_COLUMN
    if three_band:
        per_column = np.maximum(np.bincount(column), 1)
        levels = [np.clip(np.bincount(column, rms) / per_column / style.band_ref, 0.0, 1.0) for rms in band_rms]
        image = paint_layers(three_band_detail_layers(*levels), height, mirrored=True).astype(np.float64)
    else:
        sigma = RGB_DETAIL_COLOR_SMOOTH_SEC * sample_rate / frame_hop
        levels = (np.sqrt(np.sqrt(gaussian_filter1d(np.square(rms), sigma))) for rms in band_rms)
        colors = wave_colors(*levels, white_threshold=0.0, sat_floor=RGB_DETAIL_SAT_FLOOR).astype(np.float64)
        min_env, max_env = frame_minmax(mono, frame_hop)
        image = wave_coverage_columns(min_env, max_env, colors, column, height)
    # the centre line where it is quiet (each column's loudest RMS over WAVE_CENTER_LINE_RMS_SEC), not where the
    # fine frames' waveform leaves the centre row dark
    rms_window = max(1, int(round(WAVE_CENTER_LINE_RMS_SEC * sample_rate)))
    level = np.sqrt(np.maximum(uniform_filter1d(np.square(mono, dtype=np.float64), size=rms_window), 0.0))[centers]
    loudness = np.zeros(int(column[-1]) + 1 if column.size else 0)
    np.maximum.at(loudness, column, level)
    image = add_center_line(gaussian_filter1d(image, sigma=0.8, axis=0)[c0 - p0: c1 - p0],
                            loudness[c0 - p0: c1 - p0])
    return np.ascontiguousarray(np.clip(image, 0.0, 255.0).astype(np.uint8))


@lru_cache(maxsize=32)
def _band_sos(sr: int, low: float, high: float, order: int) -> np.ndarray:
    """The band's Butterworth sections, designed once per band and rate (designed per chunk it took about 30 %
    of a chunk's render, all of it Python holding the GIL the playback needs)."""
    nyq = 0.5 * float(sr)
    low_n = max(float(low) / nyq, 1e-6)
    high_n = min(float(high) / nyq, 0.999999)
    return butter(int(order), [low_n, high_n], btype="band", output="sos")   # writable: scipy's filter needs it


def _band_filter(y: np.ndarray, sr: int, low: float, high: float, order: int) -> np.ndarray:
    try:
        return sosfiltfilt(_band_sos(int(sr), float(low), float(high), int(order)), y)
    except Exception:
        return y

# Render worker

@dataclass
class _ChunkSpec:
    chunk_index: int
    pcm: np.ndarray
    sample_rate: int
    style: _WaveStyle
    height: int


class _WaveformRenderWorker(QtCore.QObject):
    """Renders the requested chunks on a pool, from a dedicated thread. Chunks are tracked per (style, chunk
    index): each style is its own layer. The pending chunk nearest the playhead starts first (the one it is in,
    then at one distance the one ahead, which playback reaches, before the one behind), whenever fewer than
    RENDER_BATCH_SIZE run, one every RENDER_SPAWN_INTERVAL_MS (the first that long after a request, so a burst
    of requests is ordered as a whole): a request at the playhead never waits behind older ones farther away."""

    chunk_ready = QtCore.Signal(int, object)  # chunk_index, _WaveRenderResult

    def __init__(self) -> None:
        super().__init__()
        self.pool = QtCore.QThreadPool()
        self.pool.setMaxThreadCount(RENDER_BATCH_SIZE)
        self.extra_pool = QtCore.QThreadPool()   # the chunk playback goes on from, when the slots are busy
        self.extra_pool.setMaxThreadCount(1)
        self._pending: dict[tuple[_WaveStyle, int], _ChunkSpec] = {}
        self._inflight: set[tuple[_WaveStyle, int]] = set()
        self._rendered: set[tuple[_WaveStyle, int]] = set()
        self._playhead = 0  # the chunk the playhead is in
        self._request_seq = 0
        self._spawn_timer = QtCore.QTimer(self)
        self._spawn_timer.setInterval(RENDER_SPAWN_INTERVAL_MS)
        self._spawn_timer.timeout.connect(self._spawn_next)

    @QtCore.Slot(list, int)
    def enqueue(self, specs: list, playhead: int) -> None:
        self._playhead = int(playhead)
        for spec in specs:
            if not isinstance(spec, _ChunkSpec):
                continue
            key = (spec.style, spec.chunk_index)
            if key not in self._rendered and key not in self._inflight:
                self._pending[key] = spec
        if self._pending and not self._spawn_timer.isActive():
            self._spawn_timer.start()

    @QtCore.Slot(int)
    def set_playhead(self, playhead: int) -> None:
        self._playhead = int(playhead)

    @QtCore.Slot(list, int)
    def resume(self, specs: list, playhead: int) -> None:
        """Playback goes on from ``playhead`` (a scrub let go, a seek): the requests pending are dropped (made
        on the way there), and the playhead's chunk starts at once, on a thread of its own when every slot is
        busy; the rest of ``specs`` (the view there) after it, nearest first."""
        self._pending.clear()
        self.enqueue(specs, playhead)
        key = next((k for k in self._pending if k[1] == self._playhead), None)
        if key is not None:
            self._start(self._pending.pop(key), urgent=True)
        if not self._pending:
            self._spawn_timer.stop()

    def _urgency(self, key: tuple[_WaveStyle, int]) -> tuple[int, bool]:
        offset = key[1] - self._playhead
        return abs(offset), offset < 0

    @QtCore.Slot(list)
    def forget(self, keys: list) -> None:
        """The view dropped these (style, chunk) images or no longer needs them: render them again when asked."""
        for key in keys:
            self._rendered.discard(key)
            self._pending.pop(key, None)

    @QtCore.Slot()
    def drop_pending(self) -> None:
        """Forget the requests not started yet (the view switched layers);
        jobs running finish and are kept."""
        self._spawn_timer.stop()
        self._pending.clear()

    @QtCore.Slot()
    def cancel_all(self) -> None:
        self._spawn_timer.stop()
        self._pending.clear()
        self._inflight.clear()
        self._rendered.clear()
        self.pool.clear()
        self.extra_pool.clear()

    @QtCore.Slot()
    def _spawn_next(self) -> None:
        if len(self._inflight) >= RENDER_BATCH_SIZE:
            return  # a slot frees when a job finishes
        if not self._pending:
            self._spawn_timer.stop()
            return
        self._start(self._pending.pop(min(self._pending, key=self._urgency)))
        if not self._pending:
            self._spawn_timer.stop()

    def _start(self, spec: _ChunkSpec, *, urgent: bool = False) -> None:
        self._request_seq += 1
        job = _WaveRenderJob(
            self._request_seq,
            spec.pcm,
            spec.sample_rate,
            spec.chunk_index,
            spec.style,
            height=spec.height,
        )
        job.signals.finished.connect(self._on_job_finished, QtCore.Qt.ConnectionType.QueuedConnection)
        (self.extra_pool if urgent and len(self._inflight) >= RENDER_BATCH_SIZE else self.pool).start(job)
        self._inflight.add((spec.style, spec.chunk_index))

    def wait_for_jobs(self) -> None:
        """Block until the jobs running end (callable from any thread)."""
        self.pool.waitForDone()
        self.extra_pool.waitForDone()

    @QtCore.Slot(int, object)
    def _on_job_finished(self, _request_id: int, payload: object) -> None:
        if not isinstance(payload, _WaveRenderResult):
            return
        chunk_index = int(payload.chunk_index)
        self._inflight.discard((payload.style, chunk_index))
        self._rendered.add((payload.style, chunk_index))
        self.chunk_ready.emit(chunk_index, payload)


# View

@dataclass
class _ChunkLayer:
    """The chunk images of one style (e.g. the short or the long views' frame),
    kept while another layer is shown."""

    style: _WaveStyle
    items: list  # pg.ImageItem | None per chunk
    spans: list  # time span of each rendered image, or None
    shown: list  # when each chunk was last in the view (WaveformView._show_clock; 0 never)
    submitted: set = field(default_factory=set)  # chunks requested from the worker
    in_scene: set = field(default_factory=set)   # chunks shown


@register_view("WaveformView")
class WaveformView(ViewPlugin):
    _sig_enqueue = QtCore.Signal(list, int)   # specs, the playhead's chunk
    _sig_resume = QtCore.Signal(list, int)    # specs, the chunk playback goes on from
    _sig_playhead = QtCore.Signal(int)
    _sig_cancel = QtCore.Signal()
    _sig_drop_pending = QtCore.Signal()
    _sig_forget = QtCore.Signal(list)

    def __init__(self, bus, model, tl):
        super().__init__(bus, model, tl)
        self.plot: pg.PlotItem | None = None
        self.duration = 0.0
        self._left_offset = 0.0
        self._last_pcm = None
        self._wave_levels = (0, 255)
        self._chunk_count = 0
        # One layer of chunk images per style (the short and the long views draw
        # at different frames); switching shows the other layer, keeping both.
        self._layers: dict[_WaveStyle, _ChunkLayer] = {}
        self._layer: _ChunkLayer | None = None
        self._group: pg.ItemGroup | None = None
        self._preview_item: pg.ImageItem | None = None
        self._preview_source = None
        self._playhead_chunk: int | None = None  # the chunk the playhead was last seen in
        self._scrubbing = False                   # the playhead dragged (its moves are no jumps)
        self._jumps = 0                           # the frames in a row it jumped in (2 on: a search, rendered where it stops)
        self._show_clock = 0  # counts the scene passes: the order chunks were last shown in

        # Dedicated render worker thread
        self._worker = _WaveformRenderWorker()
        self._worker_thread = QtCore.QThread(self)
        self._worker_thread.setObjectName("WaveformRenderThread")
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.start()
        self._worker.chunk_ready.connect(self._on_chunk_ready, QtCore.Qt.ConnectionType.QueuedConnection)
        self._sig_enqueue.connect(self._worker.enqueue, QtCore.Qt.ConnectionType.QueuedConnection)
        self._sig_resume.connect(self._worker.resume, QtCore.Qt.ConnectionType.QueuedConnection)
        self._sig_playhead.connect(self._worker.set_playhead, QtCore.Qt.ConnectionType.QueuedConnection)
        self._sig_cancel.connect(self._worker.cancel_all, QtCore.Qt.ConnectionType.QueuedConnection)
        self._sig_drop_pending.connect(self._worker.drop_pending, QtCore.Qt.ConnectionType.QueuedConnection)
        self._sig_forget.connect(self._worker.forget, QtCore.Qt.ConnectionType.QueuedConnection)
        # The thread is the view's child: stopped before the view goes, also when the app quits without the
        # views detached (else Qt destroys it running).
        app = QtCore.QCoreApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._stop_worker)

        self.bus.sig_time_changed.connect(self.update_time)
        self.bus.sig_scrub_begin.connect(self._on_scrub_begin)
        self.bus.sig_scrub_update.connect(self._on_scrub_update)
        self.bus.sig_scrub_end.connect(self._on_scrub_end)
        self.bus.sig_scratch_begin.connect(self._on_scratch_begin)     # the main plot dragged: as the overview's
        self.bus.sig_scratch_end.connect(self._on_scrub_end)
        self.bus.sig_window_changed.connect(self.update_window)
        self.model.sig_loaded.connect(self.update_features)
        self.model.sig_audio_changed.connect(self._on_model_updated)

    # The shown layer
    @property
    def _style(self) -> _WaveStyle | None:
        return self._layer.style if self._layer is not None else None

    @property
    def _chunk_items(self) -> list:
        return self._layer.items if self._layer is not None else []

    @property
    def _chunk_spans(self) -> list:
        return self._layer.spans if self._layer is not None else []

    @property
    def _submitted_chunks(self) -> set:
        return self._layer.submitted if self._layer is not None else set()

    @property
    def _in_scene(self) -> set:
        return self._layer.in_scene if self._layer is not None else set()

    def _show_layer(self, style: _WaveStyle) -> None:
        """Show the layer of ``style`` (made empty the first time); the shown one
        is hidden and kept."""
        layer = self._layers.get(style)
        if layer is None:
            count = self._chunk_count
            layer = _ChunkLayer(style=style, items=[None] * count, spans=[None] * count, shown=[0] * count)
            self._layers[style] = layer
        old = self._layer
        if old is layer:
            return
        if old is not None:
            for i in old.in_scene:
                if old.items[i] is not None:
                    old.items[i].setVisible(False)
            old.in_scene.clear()
            # its requests not started yet are dropped below: requested again
            # when it is shown again (running ones still land in it)
            old.submitted = {i for i, item in enumerate(old.items) if item is not None}
            self._sig_drop_pending.emit()
        self._layer = layer

    def attach(self, plot: pg.PlotItem):
        self.plot = plot
        # Chunk images and the preview are laid out in track time under one group;
        # playback only moves the group by the view offset.
        self._group = pg.ItemGroup()
        self._group.setZValue(-1)
        plot.addItem(self._group)
        self._ensure_preview_item()
        scene = plot.scene()
        if scene:
            for view in scene.views():
                view.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform, True)

    def detach(self):
        self._clear_preview_item()
        self._remove_all_chunk_items()
        if self._group is not None and self.plot is not None:
            self.plot.removeItem(self._group)
        self._group = None
        self._stop_worker()
        self.plot = None

    @QtCore.Slot()
    def _stop_worker(self) -> None:
        """Stop the render thread: the requests dropped, the jobs running waited for (they hold the PCM), its
        event loop ended."""
        if not self._worker_thread.isRunning():
            return
        QtCore.QMetaObject.invokeMethod(
            self._worker, "cancel_all",
            QtCore.Qt.ConnectionType.BlockingQueuedConnection,
        )
        self._worker.wait_for_jobs()
        self._worker_thread.quit()
        self._worker_thread.wait()

    def render_initial(self):
        self.duration = float(self.model.duration_sec or 0.0)
        self._update_preview_image()
        self._allocate_canvas(force=True)
        self._evaluate_render_targets()
        self._set_rect(force=True)

    def update_time(self, _t: float):
        self._set_rect()
        self._follow_playhead()
        self._evaluate_render_targets()

    def _current_chunk(self) -> int:
        duration = max(0.0, float(self.model.duration_sec or 0.0))
        return int(np.floor(np.clip(float(self.tl.current_time), 0.0, duration) / RENDER_CHUNK_SEC))

    def _follow_playhead(self) -> None:
        """When the playhead enters another chunk: the worker renders nearest it first, and the far chunks
        are dropped. Jumped there (past the next chunk, not scrubbing: a seek, a CALL, a JumpCUE, a beat jump,
        External Sync's), playback goes on from there: rendered as a scrub let go (_resume_at). Jumping again the
        next frame, the playhead travels (a controller's jog search: no drag the view is told of), followed as a
        scrub and rendered from where it stops."""
        center = self._current_chunk()
        jumped = (self._playhead_chunk is not None and abs(center - self._playhead_chunk) > 1
                  and not self._scrubbing)
        travelled = self._jumps >= 2
        self._jumps = self._jumps + 1 if jumped else 0
        if self._jumps == 1 or (travelled and not jumped):     # a jump, or the end of a travel
            self._resume_at(float(self.tl.current_time))
            if self._playhead_chunk == center:
                return      # else no audio to render yet: followed as below
        if center == self._playhead_chunk:
            return
        self._playhead_chunk = center
        self._sig_playhead.emit(center)
        self._evict_far_chunks(center)

    def _evict_far_chunks(self, center: int) -> None:
        """Drop each layer's rendered chunks farther from the playhead than RENDER_KEEP_VIEWS of the widest view it
        draws, but as many of them shown most recently, and the requests that far."""
        view = current_settings().viewconfig
        forgotten = []
        dropped = []   # (style, chunk)
        for style, layer in self._layers.items():
            widths = [w for w in WINDOW_STEPS_SEC if _WaveStyle.frame_ms_for(view, w) == style.frame_ms]
            keep = int(np.ceil(RENDER_KEEP_VIEWS * max(widths) / RENDER_CHUNK_SEC)) if widths else 0
            far = []   # (when last shown, chunk)
            for i, item in enumerate(layer.items):
                if abs(i - center) <= keep:
                    continue
                if item is not None:
                    far.append((layer.shown[i], i))
                elif i in layer.submitted:
                    layer.submitted.discard(i)   # not rendered yet: not needed now
                    forgotten.append((style, i))
            far.sort(reverse=True)
            kept = sum(1 for shown, _ in far[:keep] if shown > 0)   # never shown: not recently used
            dropped += [(style, i) for _, i in far[kept:]]
        for style, i in dropped:
            layer = self._layers[style]
            item = layer.items[i]
            if item.scene() is not None:
                item.setParentItem(None)
                item.scene().removeItem(item)
            layer.items[i] = None
            layer.spans[i] = None
            layer.shown[i] = 0
            layer.submitted.discard(i)
            layer.in_scene.discard(i)
            forgotten.append((style, i))
        if forgotten:
            self._sig_forget.emit(forgotten)

    def update_window(self, _w: float):
        if self._layer is not None:
            # crossing between the close, short and long views shows the other frame's layer
            self._show_layer(_WaveStyle.from_config(self.model.analysis.wave, self.tl.window_sec))
        self._set_rect(force=True)
        self._evaluate_render_targets()  # a wider view shows chunks not rendered yet

    def update_features(self):
        self.render_initial()

    def _on_model_updated(self):
        new_duration = float(self.model.duration_sec or 0.0)
        new_pcm = getattr(self.model, "predecoded_pcm", None)
        if new_duration == self.duration and new_pcm is self._last_pcm:
            return
        self._last_pcm = new_pcm
        self.duration = new_duration
        self._update_preview_image()
        self._allocate_canvas(force=True)
        self._evaluate_render_targets()
        self._set_rect(force=True)

    def _remove_all_chunk_items(self) -> None:
        for layer in self._layers.values():
            for item in layer.items:
                if item is not None and item.scene() is not None:
                    item.setParentItem(None)
                    item.scene().removeItem(item)
        self._layers = {}
        self._layer = None
        self._playhead_chunk = None

    def _make_chunk_item(self) -> pg.ImageItem:
        item = pg.ImageItem()
        if hasattr(item, "setAutoDownsample"):
            item.setAutoDownsample(True)
        item.setOpts(interpolation="bilinear")
        item.setZValue(0)
        item.setParentItem(self._group)
        return item

    def _ensure_preview_item(self) -> None:
        if self.plot is None or self._group is None or self._preview_item is not None:
            return
        item = pg.ImageItem()
        if hasattr(item, "setAutoDownsample"):
            item.setAutoDownsample(True)
        if hasattr(item, "setCacheMode"):
            item.setCacheMode(QtWidgets.QGraphicsItem.DeviceCoordinateCache)
        item.setOpts(interpolation="bilinear")
        item.setZValue(-1)
        item.setParentItem(self._group)
        self._preview_item = item

    def _clear_preview_item(self) -> None:
        if self._preview_item is not None and self._preview_item.scene() is not None:
            self._preview_item.setParentItem(None)
            self._preview_item.scene().removeItem(self._preview_item)
        self._preview_item = None
        self._preview_source = None

    def _update_preview_image(self) -> None:
        if self.plot is None:
            return
        self._ensure_preview_item()
        if self._preview_item is None:
            return
        wave = self.model.analysis.wave
        if wave is None:
            self._preview_item.setVisible(False)
            self._preview_source = None
            return
        cfg = current_settings()
        ranges, style = band_ranges(cfg.viewconfig), cfg.viewconfig.waveform_style
        source = (id(wave), ranges, style)
        if self._preview_source != source:
            # Until chunks are rendered, the stored overview stands in, drawn like them.
            self._preview_item.setImage(
                # with the chunks' centre line, so the two join without a seam
                np.clip(add_center_line(render_wave_detail_preview(wave, ranges, style, RENDER_HEIGHT))
                        + 0.5, 0.0, 255.0).astype(np.uint8),
                autoLevels=False,
                levels=self._wave_levels,
            )
            self._preview_source = source
        self._preview_item.setRect(QtCore.QRectF(0.0, _WAVE_Y, max(self.duration, 1e-3), _WAVE_H))
        self._update_preview_visibility()

    def _update_preview_visibility(self) -> None:
        """Show the preview only where the view is not yet covered by rendered chunks
        (the opaque chunk images hide it; drawing it underneath costs a full scaled image)."""
        if self._preview_item is None:
            return
        if self._preview_source is None:
            self._preview_item.setVisible(False)
            return
        visible = self._visible_chunk_range()
        covered = visible is not None and all(
            0 <= i < len(self._chunk_items) and self._chunk_items[i] is not None
            for i in range(visible[0], visible[1] + 1)
        )
        if self._preview_item.isVisible() == covered:
            self._preview_item.setVisible(not covered)

    def _allocate_canvas(self, force: bool = False) -> int:
        """(Re)allocate the chunk slots for the track; returns the chunk count."""
        duration = max(0.0, float(self.model.duration_sec or 0.0))
        if duration <= 0.0:
            self._remove_all_chunk_items()
            self._chunk_count = 0
            self._sig_cancel.emit()
            return 0
        chunk_count = max(1, int(np.ceil(duration / RENDER_CHUNK_SEC)))
        if force or chunk_count != self._chunk_count or self._layer is None:
            self._remove_all_chunk_items()
            self._chunk_count = chunk_count
            self._sig_cancel.emit()
            self._show_layer(_WaveStyle.from_config(self.model.analysis.wave, self.tl.window_sec))
        return self._chunk_count

    def _on_scrub_begin(self) -> None:
        self._scrubbing = True
        self._jumps = 0

    def _on_scratch_begin(self, _sec_per_px: float) -> None:
        self._on_scrub_begin()

    def _on_scrub_update(self, _t: float) -> None:
        self._evaluate_render_targets()

    def _on_scrub_end(self, t: float) -> None:
        self._scrubbing = False
        self._resume_at(t)

    def _evaluate_render_targets(self) -> None:
        pcm = getattr(self.model, "predecoded_pcm", None)
        sample_rate = int(getattr(self.model, "predecoded_rate", 0) or 0)
        duration = float(self.model.duration_sec or 0.0)
        if pcm is None or sample_rate <= 0:
            return
        if self._allocate_canvas() <= 0:
            return
        specs = self._build_chunk_specs(pcm, sample_rate, duration)
        if specs:
            self._sig_enqueue.emit(specs, self._current_chunk())   # the worker orders them by the playhead

    def _resume_at(self, t: float) -> None:
        """Playback goes on from ``t`` (a scrub let go, a seek): its chunk is rendered first, whatever else was
        asked for (the requests made on the way there are dropped), and the view there after it."""
        pcm = getattr(self.model, "predecoded_pcm", None)
        sample_rate = int(getattr(self.model, "predecoded_rate", 0) or 0)
        duration = float(self.model.duration_sec or 0.0)
        if pcm is None or sample_rate <= 0 or self._allocate_canvas() <= 0:
            return
        t = float(np.clip(float(t), 0.0, duration))
        chunk = min(self._chunk_count - 1, int(np.floor(t / RENDER_CHUNK_SEC)))
        if chunk != self._playhead_chunk:
            self._playhead_chunk = chunk
            self._evict_far_chunks(chunk)
        # the worker forgets every request not started: asked again here (the view at t) and when shown
        self._layer.submitted = {i for i, item in enumerate(self._layer.items) if item is not None}
        self._sig_resume.emit(self._build_chunk_specs(pcm, sample_rate, duration, at=t), chunk)

    def _build_chunk_specs(self, pcm: np.ndarray, sample_rate: int, duration: float, *,
                           at: float | None = None) -> list:
        """Specs of the chunks in the view (the playhead at ``at`` if given, else where it is) not requested."""
        visible_start = max(0.0, -self.tl.left_edge() if at is None else at - self.tl.center_t)
        visible_end = min(duration, visible_start + max(0.1, float(self.tl.window_sec)))
        first_chunk = int(np.floor(visible_start / RENDER_CHUNK_SEC))
        last_chunk = int(np.floor(max(0.0, visible_end - 1e-6) / RENDER_CHUNK_SEC))
        # include one chunk past visible end as look-ahead
        target_chunks = range(first_chunk, last_chunk + 2)

        specs: list[_ChunkSpec] = []
        for chunk_index in target_chunks:
            if chunk_index < 0 or chunk_index >= self._chunk_count:
                continue
            if self._chunk_items[chunk_index] is not None:
                continue  # already displayed
            if chunk_index in self._submitted_chunks:
                continue  # already in worker queue
            specs.append(_ChunkSpec(
                chunk_index=chunk_index,
                pcm=pcm,
                sample_rate=sample_rate,
                style=self._style,
                height=RENDER_HEIGHT,
            ))
            self._submitted_chunks.add(chunk_index)
        return specs

    def _sync_scene_visibility(self) -> None:
        if self.plot is None or self._chunk_count <= 0:
            return
        visible = self._visible_chunk_range()
        if visible is None:
            return
        first_v, last_v = visible
        keep_start = max(0, first_v - SCENE_CULL_BUFFER)
        keep_end = min(self._chunk_count - 1, last_v + SCENE_CULL_BUFFER)
        self._show_clock += 1
        for i, item in enumerate(self._chunk_items):
            if item is None:
                continue
            if keep_start <= i <= keep_end:
                self._layer.shown[i] = self._show_clock
                if i not in self._in_scene:
                    item.setVisible(True)
                    self._in_scene.add(i)
            else:
                if i in self._in_scene:
                    item.setVisible(False)
                    self._in_scene.discard(i)
        self._update_preview_visibility()

    @QtCore.Slot(int, object)
    def _on_chunk_ready(self, chunk_index: int, payload: object) -> None:
        if not isinstance(payload, _WaveRenderResult) or self.plot is None:
            return
        layer = self._layers.get(payload.style)
        if layer is None or chunk_index < 0 or chunk_index >= len(layer.items):
            return  # a layer dropped meanwhile (new track or settings)
        layer.spans[chunk_index] = (payload.start_sec, payload.end_sec)
        patch = np.ascontiguousarray(payload.image, dtype=np.uint8)
        if patch.size == 0:
            return
        item = layer.items[chunk_index]
        is_new = item is None
        if is_new:
            item = self._make_chunk_item()
            layer.items[chunk_index] = item
        item.setImage(patch, autoLevels=False, levels=self._wave_levels)
        self._apply_chunk_rect(item, layer.spans[chunk_index], chunk_index)
        if is_new:
            item.setVisible(False)
            if layer is self._layer:
                self._sync_scene_visibility()

    def _apply_chunk_rect(self, item: pg.ImageItem, span, chunk_index: int) -> None:
        # The image covers its column-aligned span; neighbouring spans share their edge.
        if span is None:
            start_sec = chunk_index * RENDER_CHUNK_SEC
            end_sec = min(self.duration, start_sec + RENDER_CHUNK_SEC)
        else:
            start_sec, end_sec = span
        item.setRect(QtCore.QRectF(start_sec, _WAVE_Y, end_sec - start_sec, _WAVE_H))

    def _set_rect(self, force: bool = False):
        if self.duration <= 0.0:
            return
        left = self.tl.left_edge()
        if not force and abs(left - self._left_offset) < 1e-9:
            return
        self._left_offset = left
        if self._group is not None:
            self._group.setPos(left, 0.0)
        self._sync_scene_visibility()

    def _visible_chunk_range(self) -> tuple[int, int] | None:
        duration = float(self.model.duration_sec or 0.0)
        if duration <= 0.0 or self._chunk_count <= 0:
            return None
        visible_start = max(0.0, -self.tl.left_edge())
        visible_end = min(duration, visible_start + max(0.1, float(self.tl.window_sec)))
        first_visible_chunk = int(np.floor(visible_start / RENDER_CHUNK_SEC))
        last_visible_chunk = int(np.floor(max(0.0, visible_end - 1e-6) / RENDER_CHUNK_SEC))
        return first_visible_chunk, last_visible_chunk
