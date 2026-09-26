from __future__ import annotations

from dataclasses import dataclass
from collections import deque

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, sosfiltfilt

from core.config import load_cfg
from utils.wave import frame_minmax, wave_colors, wave_coverage_columns
from .base import ViewPlugin, register_view


RENDER_CHUNK_SEC = 1
RENDER_LAUNCH_DELAY_MS = 30
RENDER_HEIGHT = 96
RENDER_BATCH_SIZE = 4
RENDER_SPAWN_INTERVAL_MS = 30
SCENE_CULL_BUFFER = 2  # extra chunks kept in scene on each side of viewport
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

    @classmethod
    def from_config(cls) -> "_WaveStyle":
        cfg = load_cfg().analysisconfig
        return cls(
            bands=tuple((float(low), float(high)) for low, high in (cfg.env_lo, cfg.env_mid, cfg.env_hi)),
            filter_order=int(cfg.env_order),
            frame_ms=float(getattr(cfg, "env_frame_ms", 20)) / 2,
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
            _WaveRenderResult(image=image, chunk_index=self.chunk_index, start_sec=start_sec, end_sec=end_sec),
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

    Thin bars (min..max of each frame of ``style.frame_ms``) colored by the
    low / mid / high band levels, averaged FRAMES_PER_COLUMN to a column and
    blurred slightly along time. The chunk is computed with RENDER_CONTEXT_SEC
    of audio on each side, on the track-wide frame grid, and cropped.
    """
    n_samples = int(pcm.shape[0])
    frame_hop = style.frame_hop(sample_rate)
    column_hop = FRAMES_PER_COLUMN * frame_hop
    c0, c1 = _chunk_columns(chunk_index, sample_rate, n_samples, column_hop)
    start_sec, end_sec = c0 * column_hop / sample_rate, c1 * column_hop / sample_rate
    if c1 <= c0:
        return np.zeros((1, height, 3), dtype=np.uint8), start_sec, max(end_sec, start_sec + 1e-3)
    context = int(np.ceil(RENDER_CONTEXT_SEC * sample_rate / column_hop))
    p0 = max(0, c0 - context)
    p1 = c1 + context
    segment = np.asarray(pcm[p0 * column_hop: min(n_samples, p1 * column_hop)], dtype=np.float32)
    mono = segment if segment.ndim == 1 else segment.mean(axis=1, dtype=np.float32)

    n_frames = -(-mono.size // frame_hop)
    starts = np.arange(n_frames) * frame_hop
    counts = np.diff(np.append(starts, mono.size)).astype(np.float64)
    nyq = 0.5 * float(sample_rate)
    levels = []
    for low, high in style.bands:
        filtered = _band_filter(mono, sample_rate, low, min(high, nyq * 0.98), style.filter_order)
        rms = np.sqrt(np.add.reduceat(np.square(filtered, dtype=np.float64), starts) / counts)
        levels.append(np.sqrt(rms))
    colors = wave_colors(*levels, white_threshold=0.0).astype(np.float64)
    min_env, max_env = frame_minmax(mono, frame_hop)

    image = wave_coverage_columns(min_env, max_env, colors, np.arange(n_frames) // FRAMES_PER_COLUMN, height)
    image = gaussian_filter1d(image, sigma=0.8, axis=0)[c0 - p0: c1 - p0]
    return np.ascontiguousarray(np.clip(image, 0.0, 255.0).astype(np.uint8)), start_sec, end_sec


def _band_filter(y: np.ndarray, sr: int, low: float, high: float, order: int) -> np.ndarray:
    nyq = 0.5 * float(sr)
    low_n = max(float(low) / nyq, 1e-6)
    high_n = min(float(high) / nyq, 0.999999)
    try:
        sos = butter(int(order), [low_n, high_n], btype="band", output="sos")
        return sosfiltfilt(sos, y)
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
    """Manages render queue and QRunnable dispatch on a dedicated thread."""

    chunk_ready = QtCore.Signal(int, object)  # chunk_index, _WaveRenderResult

    def __init__(self) -> None:
        super().__init__()
        self._pool = QtCore.QThreadPool()
        self._pool.setMaxThreadCount(RENDER_BATCH_SIZE)
        self._play_pending: deque[_ChunkSpec] = deque()
        self._scrub_pending: deque[_ChunkSpec] = deque()
        self._spawn_queue: deque[_ChunkSpec] = deque()
        self._inflight: set[int] = set()
        self._rendered: set[int] = set()
        self._request_seq = 0
        self._launch_timer = QtCore.QTimer(self)
        self._launch_timer.setSingleShot(True)
        self._launch_timer.setInterval(RENDER_LAUNCH_DELAY_MS)
        self._launch_timer.timeout.connect(self._process_queue)
        self._spawn_timer = QtCore.QTimer(self)
        self._spawn_timer.setSingleShot(False)
        self._spawn_timer.setInterval(RENDER_SPAWN_INTERVAL_MS)
        self._spawn_timer.timeout.connect(self._spawn_next)

    @QtCore.Slot(list)
    def enqueue_play(self, specs: list) -> None:
        for spec in specs:
            if not isinstance(spec, _ChunkSpec):
                continue
            if spec.chunk_index in self._rendered or spec.chunk_index in self._inflight:
                continue
            self._play_pending.append(spec)
        if not self._launch_timer.isActive():
            self._launch_timer.start()

    @QtCore.Slot(list)
    def enqueue_scrub(self, specs: list) -> None:
        for spec in specs:
            if not isinstance(spec, _ChunkSpec):
                continue
            if spec.chunk_index in self._rendered or spec.chunk_index in self._inflight:
                continue
            self._scrub_pending.append(spec)
        if not self._launch_timer.isActive():
            self._launch_timer.start()

    @QtCore.Slot()
    def cancel_all(self) -> None:
        self._launch_timer.stop()
        self._spawn_timer.stop()
        self._play_pending.clear()
        self._scrub_pending.clear()
        self._spawn_queue.clear()
        self._inflight.clear()
        self._rendered.clear()
        self._pool.clear()

    @QtCore.Slot()
    def _process_queue(self) -> None:
        if self._inflight or self._spawn_timer.isActive():
            return
        # play queue has priority; fall back to scrub only when play is empty
        source = self._play_pending if self._play_pending else self._scrub_pending
        slots = RENDER_BATCH_SIZE
        while slots > 0 and source:
            spec = source.popleft()
            if spec.chunk_index in self._rendered or spec.chunk_index in self._inflight:
                continue
            self._spawn_queue.append(spec)
            slots -= 1
        if self._spawn_queue:
            self._spawn_next()
            if self._spawn_queue:
                self._spawn_timer.start()

    @QtCore.Slot()
    def _spawn_next(self) -> None:
        while self._spawn_queue:
            spec = self._spawn_queue.popleft()
            if spec.chunk_index in self._rendered or spec.chunk_index in self._inflight:
                continue
            self._request_seq += 1
            self._inflight.add(spec.chunk_index)
            job = _WaveRenderJob(
                self._request_seq,
                spec.pcm,
                spec.sample_rate,
                spec.chunk_index,
                spec.style,
                height=spec.height,
            )
            job.signals.finished.connect(self._on_job_finished, QtCore.Qt.ConnectionType.QueuedConnection)
            self._pool.start(job)
            return  # one job per tick
        # spawn_queue exhausted — all jobs of this batch were already submitted
        self._spawn_timer.stop()
        # if all inflight jobs also finished while we were still spawning, kick next batch
        has_pending = bool(self._play_pending or self._scrub_pending)
        if has_pending and not self._inflight and not self._launch_timer.isActive():
            self._launch_timer.start()

    @QtCore.Slot(int, object)
    def _on_job_finished(self, _request_id: int, payload: object) -> None:
        if not isinstance(payload, _WaveRenderResult):
            return
        chunk_index = int(payload.chunk_index)
        self._inflight.discard(chunk_index)
        self._rendered.add(chunk_index)
        self.chunk_ready.emit(chunk_index, payload)
        # only schedule next batch when spawn timer is also done
        # (if spawn timer is still running, _spawn_next will handle continuation)
        has_pending = bool(self._play_pending or self._scrub_pending)
        if has_pending and not self._inflight and not self._spawn_timer.isActive() and not self._launch_timer.isActive():
            self._launch_timer.start()


# View

@register_view("WaveformView")
class WaveformView(ViewPlugin):
    _sig_enqueue_play = QtCore.Signal(list)
    _sig_enqueue_scrub = QtCore.Signal(list)
    _sig_cancel = QtCore.Signal()

    def __init__(self, bus, model, tl):
        super().__init__(bus, model, tl)
        self.plot: pg.PlotItem | None = None
        self.duration = 0.0
        self._left_offset = 0.0
        self._last_pcm = None
        self._wave_levels = (0, 255)
        self._chunk_count = 0
        self._chunk_items: list[pg.ImageItem | None] = []
        self._chunk_spans: list[tuple[float, float] | None] = []  # time span of each rendered image
        self._style: _WaveStyle | None = None
        self._preview_item: pg.ImageItem | None = None
        self._preview_source = None
        self._in_scene: set[int] = set()
        self._submitted_chunks: set[int] = set()
        self._scrubbing = False

        # Dedicated render worker thread
        self._worker = _WaveformRenderWorker()
        self._worker_thread = QtCore.QThread(self)
        self._worker_thread.setObjectName("WaveformRenderThread")
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.start()
        self._worker.chunk_ready.connect(self._on_chunk_ready, QtCore.Qt.ConnectionType.QueuedConnection)
        self._sig_enqueue_play.connect(self._worker.enqueue_play, QtCore.Qt.ConnectionType.QueuedConnection)
        self._sig_enqueue_scrub.connect(self._worker.enqueue_scrub, QtCore.Qt.ConnectionType.QueuedConnection)
        self._sig_cancel.connect(self._worker.cancel_all, QtCore.Qt.ConnectionType.QueuedConnection)

        self.bus.sig_time_changed.connect(self.update_time)
        self.bus.sig_seek_requested.connect(self._on_seek_requested)
        self.bus.sig_scrub_begin.connect(self._on_scrub_begin)
        self.bus.sig_scrub_update.connect(self._on_scrub_update)
        self.bus.sig_scrub_end.connect(self._on_scrub_end)
        self.bus.sig_window_changed.connect(self.update_window)
        self.bus.sig_features_loaded.connect(self.update_features)
        self.model.sig_updated.connect(self._on_model_updated)

    def attach(self, plot: pg.PlotItem):
        self.plot = plot
        self._ensure_preview_item()
        scene = plot.scene()
        if scene:
            for view in scene.views():
                view.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform, True)

    def detach(self):
        self._clear_preview_item()
        self._remove_all_chunk_items()
        if self._worker_thread.isRunning():
            QtCore.QMetaObject.invokeMethod(
                self._worker, "cancel_all",
                QtCore.Qt.ConnectionType.BlockingQueuedConnection,
            )
            self._worker_thread.quit()
            self._worker_thread.wait(3000)
        self.plot = None

    def render_initial(self):
        self.duration = float(self.model.duration_sec or 0.0)
        self._update_preview_image()
        self._allocate_canvas(force=True)
        self._evaluate_render_targets()
        self._set_rect(force=True)

    def update_time(self, _t: float):
        self._set_rect()
        self._evaluate_render_targets()

    def update_window(self, _w: float):
        self._set_rect(force=True)

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
        for item in self._chunk_items:
            if item is not None and item.scene() is not None:
                item.scene().removeItem(item)
        self._chunk_items = []
        self._chunk_spans = []
        self._in_scene.clear()

    def _make_chunk_item(self) -> pg.ImageItem:
        item = pg.ImageItem()
        if hasattr(item, "setAutoDownsample"):
            item.setAutoDownsample(True)
        item.setOpts(interpolation="bilinear")
        item.setZValue(-1)
        return item

    def _ensure_preview_item(self) -> None:
        if self.plot is None or self._preview_item is not None:
            return
        item = pg.ImageItem()
        if hasattr(item, "setAutoDownsample"):
            item.setAutoDownsample(True)
        if hasattr(item, "setCacheMode"):
            item.setCacheMode(QtWidgets.QGraphicsItem.DeviceCoordinateCache)
        item.setOpts(interpolation="bilinear")
        item.setZValue(-2)
        self.plot.addItem(item)
        self._preview_item = item

    def _clear_preview_item(self) -> None:
        if self._preview_item is not None and self._preview_item.scene() is not None:
            self._preview_item.scene().removeItem(self._preview_item)
        self._preview_item = None
        self._preview_source = None

    def _update_preview_image(self) -> None:
        if self.plot is None:
            return
        self._ensure_preview_item()
        if self._preview_item is None:
            return
        wave_data = self.model.features.get("wave_img_np_preview") if isinstance(self.model.features, dict) else None
        if wave_data is None:
            self._preview_item.setVisible(False)
            self._preview_source = None
            return
        wave_arr = wave_data if isinstance(wave_data, np.ndarray) else np.asarray(wave_data)
        if wave_arr.ndim != 3 or wave_arr.size == 0:
            self._preview_item.setVisible(False)
            self._preview_source = None
            return
        if self._preview_source is not wave_data:
            self._preview_item.setImage(
                np.ascontiguousarray(wave_arr, dtype=np.uint8),
                autoLevels=False,
                levels=self._wave_levels,
            )
            self._preview_source = wave_data
        self._preview_item.setVisible(True)
        self._preview_item.setRect(
            QtCore.QRectF(self._left_offset, 0.14, max(self.duration, 1e-3), 1.0 - 0.24)
        )

    def _allocate_canvas(self, force: bool = False) -> int:
        """(Re)allocate the chunk slots for the track; returns the chunk count."""
        duration = max(0.0, float(self.model.duration_sec or 0.0))
        if duration <= 0.0:
            self._remove_all_chunk_items()
            self._chunk_count = 0
            self._sig_cancel.emit()
            return 0
        chunk_count = max(1, int(np.ceil(duration / RENDER_CHUNK_SEC)))
        if force or chunk_count != self._chunk_count:
            self._remove_all_chunk_items()
            self._chunk_count = chunk_count
            self._chunk_items = [None] * chunk_count
            self._chunk_spans = [None] * chunk_count
            self._submitted_chunks.clear()
            self._style = _WaveStyle.from_config()
            self._sig_cancel.emit()
        return self._chunk_count

    def _on_seek_requested(self, _t: float) -> None:
        self._evaluate_render_targets()

    def _on_scrub_begin(self) -> None:
        self._scrubbing = True

    def _on_scrub_update(self, _t: float) -> None:
        if not self._scrubbing:
            return
        self._evaluate_render_targets()

    def _on_scrub_end(self, _t: float) -> None:
        self._scrubbing = False
        self._evaluate_render_targets()

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
            if self._scrubbing:
                self._sig_enqueue_scrub.emit(specs)
            else:
                self._sig_enqueue_play.emit(specs)

    def _build_chunk_specs(self, pcm: np.ndarray, sample_rate: int, duration: float) -> list:
        visible_start = max(0.0, float(self.tl.current_time) - float(self.tl.center_t))
        visible_end = min(duration, visible_start + max(0.1, float(self.tl.window_sec)))
        first_chunk = int(np.floor(visible_start / RENDER_CHUNK_SEC))
        last_chunk = int(np.floor(max(0.0, visible_end - 1e-6) / RENDER_CHUNK_SEC))
        # include one chunk past visible end as look-ahead
        target_chunks = list(range(first_chunk, last_chunk + 2))
        current_chunk = int(np.floor(np.clip(float(self.tl.current_time), 0.0, duration) / RENDER_CHUNK_SEC))
        target_chunks.sort(key=lambda idx: (abs(int(idx) - current_chunk), int(idx)))

        specs: list[_ChunkSpec] = []
        for chunk_index in target_chunks:
            if chunk_index < 0 or chunk_index >= self._chunk_count:
                continue
            if self._chunk_items[chunk_index] is not None:
                continue  # already displayed
            if chunk_index in self._submitted_chunks:
                continue  # already in worker queue
            if self._style is None:
                self._style = _WaveStyle.from_config()
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
        for i, item in enumerate(self._chunk_items):
            if item is None:
                continue
            if keep_start <= i <= keep_end:
                if i not in self._in_scene:
                    self.plot.addItem(item)
                    self._apply_chunk_rect(item, i)
                    self._in_scene.add(i)
            else:
                if i in self._in_scene:
                    self.plot.removeItem(item)
                    self._in_scene.discard(i)

    @QtCore.Slot(int, object)
    def _on_chunk_ready(self, chunk_index: int, payload: object) -> None:
        if not isinstance(payload, _WaveRenderResult) or self.plot is None:
            return
        if chunk_index < 0 or chunk_index >= self._chunk_count:
            return
        self._chunk_spans[chunk_index] = (payload.start_sec, payload.end_sec)
        patch = np.ascontiguousarray(payload.image, dtype=np.uint8)
        if patch.size == 0:
            return
        item = self._chunk_items[chunk_index]
        is_new = item is None
        if is_new:
            item = self._make_chunk_item()
            self._chunk_items[chunk_index] = item
        item.setImage(patch, autoLevels=False, levels=self._wave_levels)
        if is_new:
            visible = self._visible_chunk_range()
            if visible is not None:
                first_v, last_v = visible
                keep_start = max(0, first_v - SCENE_CULL_BUFFER)
                keep_end = min(self._chunk_count - 1, last_v + SCENE_CULL_BUFFER)
                if keep_start <= chunk_index <= keep_end:
                    self.plot.addItem(item)
                    self._apply_chunk_rect(item, chunk_index)
                    self._in_scene.add(chunk_index)

    def _apply_chunk_rect(self, item: pg.ImageItem, chunk_index: int) -> None:
        # The image covers its column-aligned span; neighbouring spans share their edge.
        span = self._chunk_spans[chunk_index] if chunk_index < len(self._chunk_spans) else None
        if span is None:
            start_sec = chunk_index * RENDER_CHUNK_SEC
            end_sec = min(self.duration, start_sec + RENDER_CHUNK_SEC)
        else:
            start_sec, end_sec = span
        item.setRect(QtCore.QRectF(
            self._left_offset + start_sec, 0.14,
            end_sec - start_sec, 1.0 - 0.24,
        ))

    def _set_rect(self, force: bool = False):
        if self.duration <= 0.0:
            return
        left = float(self.tl.center_t) - float(self.tl.current_time)
        if not force and abs(left - self._left_offset) < 1e-9:
            return
        self._left_offset = left
        if self._preview_item is not None:
            self._preview_item.setRect(
                QtCore.QRectF(self._left_offset, 0.14, max(self.duration, 1e-3), 1.0 - 0.24)
            )
        for i in self._in_scene:
            item = self._chunk_items[i]
            if item is not None:
                self._apply_chunk_rect(item, i)
        self._sync_scene_visibility()

    def _visible_chunk_range(self) -> tuple[int, int] | None:
        duration = float(self.model.duration_sec or 0.0)
        if duration <= 0.0 or self._chunk_count <= 0:
            return None
        visible_start = max(0.0, float(self.tl.current_time) - float(self.tl.center_t))
        visible_end = min(duration, visible_start + max(0.1, float(self.tl.window_sec)))
        first_visible_chunk = int(np.floor(visible_start / RENDER_CHUNK_SEC))
        last_visible_chunk = int(np.floor(max(0.0, visible_end - 1e-6) / RENDER_CHUNK_SEC))
        return first_visible_chunk, last_visible_chunk
