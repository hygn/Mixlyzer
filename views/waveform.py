from __future__ import annotations

from dataclasses import dataclass
from collections import deque

import librosa
import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui
from scipy.signal import butter, sosfiltfilt

from analyzer_core.global_analyzer import _band_envelope_rms, frame_minmax
from core.config import load_cfg
from utils.wave import build_wave_image
from .base import ViewPlugin, register_view


RENDER_CHUNK_SEC = 1
RENDER_LAUNCH_DELAY_MS = 30
RENDER_HEIGHT = 96
RENDER_WIDTH = 2048
MIN_CANVAS_WIDTH = 2048
MAX_CANVAS_WIDTH = 1048576
CANVAS_PIXELS_PER_SEC = 256.0
RENDER_BATCH_SIZE = 4
RENDER_SPAWN_INTERVAL_MS = 30
SCENE_CULL_BUFFER = 2  # extra chunks kept in scene on each side of viewport


class _WaveRenderSignals(QtCore.QObject):
    finished = QtCore.Signal(int, object)


@dataclass
class _WaveRenderResult:
    image: np.ndarray
    chunk_index: int
    start_sec: float
    span_sec: float


class _WaveRenderJob(QtCore.QRunnable):
    def __init__(
        self,
        request_id: int,
        pcm: np.ndarray,
        sample_rate: int,
        start_sec: float,
        span_sec: float,
        chunk_index: int,
        duration_sec: float,
        *,
        canvas_width: int,
        height: int,
    ) -> None:
        super().__init__()
        self.request_id = int(request_id)
        self.pcm = pcm
        self.sample_rate = max(1, int(sample_rate))
        self.start_sec = max(0.0, float(start_sec))
        self.span_sec = max(0.001, float(span_sec))
        self.chunk_index = max(0, int(chunk_index))
        self.duration_sec = max(self.span_sec, float(duration_sec))
        self.canvas_width = max(64, int(canvas_width))
        self.height = max(16, int(height))
        self.signals = _WaveRenderSignals()

    def run(self) -> None:
        x0, x1 = _time_span_to_canvas_columns(
            self.start_sec,
            self.span_sec,
            duration_sec=self.duration_sec,
            canvas_width=self.canvas_width,
        )
        width = max(1, x1 - x0)
        frame_start = int(round(self.start_sec * self.sample_rate))
        frame_span = max(1, int(round(self.span_sec * self.sample_rate)))
        frame_end = min(int(self.pcm.shape[0]), frame_start + frame_span)
        segment = np.asarray(self.pcm[frame_start:frame_end], dtype=np.float32)
        image = _render_waveform_segment(segment, self.sample_rate, width=width, height=self.height)
        self.signals.finished.emit(
            self.request_id,
            _WaveRenderResult(
                image=image,
                chunk_index=self.chunk_index,
                start_sec=self.start_sec,
                span_sec=self.span_sec,
            ),
        )


def _time_span_to_canvas_columns(
    start_sec: float,
    span_sec: float,
    *,
    duration_sec: float,
    canvas_width: int,
) -> tuple[int, int]:
    duration_sec = max(0.001, float(duration_sec))
    canvas_width = max(1, int(canvas_width))
    start_ratio = np.clip(float(start_sec) / duration_sec, 0.0, 1.0)
    end_ratio = np.clip((float(start_sec) + float(span_sec)) / duration_sec, 0.0, 1.0)
    x0 = int(np.floor(start_ratio * canvas_width))
    x1 = int(np.ceil(end_ratio * canvas_width))
    x0 = int(np.clip(x0, 0, max(0, canvas_width - 1)))
    x1 = int(np.clip(x1, x0 + 1, canvas_width))
    return x0, x1


def _render_waveform_segment(segment: np.ndarray, sample_rate: int, *, width: int, height: int) -> np.ndarray:
    if segment.size == 0:
        return np.zeros((width, height, 3), dtype=np.uint8)
    if segment.ndim == 1:
        mono = segment.astype(np.float32, copy=False)
    else:
        mono = np.mean(segment.astype(np.float32, copy=False), axis=1)
    if mono.size == 0:
        return np.zeros((width, height, 3), dtype=np.uint8)
    cfg = load_cfg().analysisconfig
    lo_low, lo_high = cfg.env_lo
    mid_low, mid_high = cfg.env_mid
    hi_low, hi_high = cfg.env_hi
    nyq = 0.5 * float(sample_rate)
    lo_high = min(float(lo_high), nyq * 0.98)
    mid_high = min(float(mid_high), nyq * 0.98)
    hi_high = min(float(hi_high), nyq * 0.98)

    frame_ms = float(getattr(cfg, "env_frame_ms", 20))/2
    env_frame_len = max(1, int(round(frame_ms * 1e-3 * float(sample_rate))))
    env_hop = env_frame_len

    low_input = mono
    if mono.size > 1:
        fade_len = min(int(round(0.01 * sample_rate)), mono.size // 2)
        fade = np.ones(mono.size, dtype=np.float32)
        if fade_len > 0:
            ramp = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
            fade[:fade_len] = ramp
            fade[-fade_len:] = ramp[::-1]
        low_input = mono * fade

    lo_env = _band_envelope_rms_safe(low_input, sample_rate, lo_low, lo_high, env_frame_len, env_hop, cfg.env_order)
    mid_env = _band_envelope_rms_safe(mono, sample_rate, mid_low, mid_high, env_frame_len, env_hop, cfg.env_order)
    hi_env = _band_envelope_rms_safe(mono, sample_rate, hi_low, hi_high, env_frame_len, env_hop, cfg.env_order)
    min_env, max_env = frame_minmax(mono, env_hop)
    min_env = np.nan_to_num(min_env, nan=0.0, posinf=0.0, neginf=0.0)
    max_env = np.nan_to_num(max_env, nan=0.0, posinf=0.0, neginf=0.0)

    def _norm(x: np.ndarray) -> np.ndarray:
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        m = float(np.max(x)) if x.size else 0.0
        return (x / m) if m > 1e-12 else x

    lo_env = _norm(lo_env)
    mid_env = _norm(mid_env)
    hi_env = _norm(hi_env)

    t_len = min(len(lo_env), len(mid_env), len(hi_env), len(min_env), len(max_env))
    if t_len <= 0:
        return np.zeros((width, height, 3), dtype=np.uint8)

    lo_env = lo_env[:t_len]
    mid_env = mid_env[:t_len]
    hi_env = hi_env[:t_len]
    min_env = min_env[:t_len]
    max_env = max_env[:t_len]

    ds_scale = 2
    img = build_wave_image(
        lo_env,
        mid_env,
        hi_env,
        min_env,
        max_env,
        height_px=height,
        downsample=ds_scale,
        white_threshold=0.0,
    )
    return np.ascontiguousarray(img.swapaxes(0, 1), dtype=np.uint8)


def _band_envelope_rms_safe(
    y: np.ndarray,
    sr: int,
    low: float,
    high: float,
    frame_length: int,
    hop_length: int,
    order: int,
) -> np.ndarray:
    nyq = 0.5 * float(sr)
    low_n = max(float(low) / nyq, 1e-6)
    high_n = min(float(high) / nyq, 0.999999)
    try:
        sos = butter(int(order), [low_n, high_n], btype="band", output="sos")
        yb = sosfiltfilt(sos, y).astype(np.float32, copy=False)
    except Exception:
        yb = y.astype(np.float32, copy=False)
    fl = int(max(16, frame_length))
    if fl % 2 == 0:
        fl += 1
    env = np.sqrt(
        np.maximum(
            0.0,
            np.asarray(
                librosa.feature.rms(
                    y=yb,
                    frame_length=fl,
                    hop_length=int(max(1, hop_length)),
                    center=True,
                )[0],
                dtype=np.float32,
            ),
        )
    )
    return np.nan_to_num(env, nan=0.0, posinf=0.0, neginf=0.0)

# Render worker

@dataclass
class _ChunkSpec:
    chunk_index: int
    pcm: np.ndarray
    sample_rate: int
    start_sec: float
    span_sec: float
    duration_sec: float
    canvas_width: int
    height: int


class _WaveformRenderWorker(QtCore.QObject):
    """Manages render queue and QRunnable dispatch on a dedicated thread."""

    chunk_ready = QtCore.Signal(int, object)  # chunk_index, np.ndarray image

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
                spec.start_sec,
                spec.span_sec,
                spec.chunk_index,
                spec.duration_sec,
                canvas_width=spec.canvas_width,
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
        self.chunk_ready.emit(chunk_index, payload.image)
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
        self._canvas_width = 0
        self._chunk_count = 0
        self._chunk_items: list[pg.ImageItem | None] = []
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
        scene = plot.scene()
        if scene:
            for view in scene.views():
                view.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform, True)

    def detach(self):
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
        self._allocate_canvas(force=True)
        self._evaluate_render_targets()
        self._set_rect(force=True)

    def _remove_all_chunk_items(self) -> None:
        for item in self._chunk_items:
            if item is not None and item.scene() is not None:
                item.scene().removeItem(item)
        self._chunk_items = []
        self._in_scene.clear()

    def _make_chunk_item(self) -> pg.ImageItem:
        item = pg.ImageItem()
        if hasattr(item, "setAutoDownsample"):
            item.setAutoDownsample(True)
        item.setOpts(interpolation="bilinear")
        item.setZValue(-1)
        return item

    def _allocate_canvas(self, force: bool = False) -> int:
        duration = max(0.0, float(self.model.duration_sec or 0.0))
        if duration <= 0.0:
            self._remove_all_chunk_items()
            self._canvas_width = 0
            self._chunk_count = 0
            self._sig_cancel.emit()
            return 0
        canvas_width = int(np.clip(np.ceil(duration * CANVAS_PIXELS_PER_SEC), MIN_CANVAS_WIDTH, MAX_CANVAS_WIDTH))
        chunk_count = max(1, int(np.ceil(duration / RENDER_CHUNK_SEC)))
        if force or canvas_width != self._canvas_width or chunk_count != self._chunk_count:
            self._remove_all_chunk_items()
            self._canvas_width = canvas_width
            self._chunk_count = chunk_count
            self._chunk_items = [None] * chunk_count
            self._submitted_chunks.clear()
            self._sig_cancel.emit()
        return self._canvas_width

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
            start_sec = chunk_index * RENDER_CHUNK_SEC
            end_sec = min(duration, start_sec + RENDER_CHUNK_SEC)
            span_sec = max(0.001, end_sec - start_sec)
            specs.append(_ChunkSpec(
                chunk_index=chunk_index,
                pcm=pcm,
                sample_rate=sample_rate,
                start_sec=start_sec,
                span_sec=span_sec,
                duration_sec=duration,
                canvas_width=self._canvas_width,
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
    def _on_chunk_ready(self, chunk_index: int, image: object) -> None:
        if not isinstance(image, np.ndarray) or self.plot is None:
            return
        if chunk_index < 0 or chunk_index >= self._chunk_count:
            return
        patch = np.ascontiguousarray(image, dtype=np.uint8)
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
        start_sec = chunk_index * RENDER_CHUNK_SEC
        end_sec = min(self.duration, start_sec + RENDER_CHUNK_SEC)
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
