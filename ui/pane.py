from __future__ import annotations

from typing import List, Optional, Type
import time

from PySide6 import QtWidgets, QtGui, QtCore, QtMultimedia
import pyqtgraph as pg
import numpy as np

from views.base import REGISTRY, ViewPlugin
from views.overview import OverviewWidget
from ui.library import LibraryWidget
from utils.cue_points import extract_cue_points
from utils.labels import idx_to_labels
from utils.semitone import speed_to_semitone
from utils.jump_cues import extract_jump_cue_pairs
from utils.phrases import extract_phrase_segments, number_phrase_labels, phrase_color
from core.event_bus import EventBus
from core.config import config
from ui.mainplotvbx import ScratchViewBox
from ui.track_info_panel import TrackInfoPanel
from ui.edit_panel import EditPanel
from utils.volume import slider_percent_to_linear, trim_dbfs_to_linear
from core.resource_paths import resource_path
from core.beat_geometry import downbeat_beat_indices, segment_time_signature
from core.tempo_rows import segment_end_bpm

def _tempo_text(percent: int) -> str:
    """Tempo fader reading as DJ gear shows it: change from 100 %, +000% / −050% / +100%
    (U+2212 minus: as wide as "+")."""
    return f"{int(percent) - 100:+04d}%".replace("-", "−")


_transport_icons: dict[str, QtGui.QIcon] = {}


def _transport_icon(name: str) -> QtGui.QIcon:
    """Deck button icon from assets/icons/transport_<name>.png (stop / play / pause)."""
    icon = _transport_icons.get(name)
    if icon is None:
        icon = QtGui.QIcon(str(resource_path(f"assets/icons/transport_{name}.png")))
        _transport_icons[name] = icon
    return icon


pg.setConfigOptions(antialias=False, useOpenGL=False)


class _TrackedGraphicsLayoutWidget(pg.GraphicsLayoutWidget):
    def __init__(self, bus: EventBus, *args, **kwargs):
        self._bus = bus
        self._last_paint_ts = None
        super().__init__(*args, **kwargs)

    def paintEvent(self, event):
        now = time.perf_counter()
        if self._last_paint_ts is not None:
            dt_ms = (now - self._last_paint_ts) * 1000.0
            if 1.0 <= dt_ms <= 1000.0:
                try:
                    self._bus.sig_ui_draw_interval.emit(dt_ms)
                except Exception:
                    pass
        self._last_paint_ts = now
        super().paintEvent(event)


class _OneStepWheelSlider(QtWidgets.QSlider):
    """QSlider forcing wheel events to adjust by exactly one step regardless of delta."""

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        delta = 0
        if event is not None:
            if event.angleDelta().y():
                delta = event.angleDelta().y()
            elif event.pixelDelta().y():
                delta = event.pixelDelta().y()
        if delta == 0:
            event.ignore()
            return
        step = 1 if delta > 0 else -1
        new_value = int(self.value() + step)
        new_value = max(self.minimum(), min(self.maximum(), new_value))
        if new_value != self.value():
            self.setValue(new_value)
        event.accept()


class MainPane(QtWidgets.QWidget):
    """Main pane containing header controls, overview, and library."""

    def __init__(self, bus: EventBus, model, tl, cfg: config):
        super().__init__()
        self.bus = bus
        self.model = model
        self.tl = tl
        self.cfg = cfg
        self.current_time = 0.0
        # the edit pages emphasized (EditPanel, while unlocked), for the views created later
        self._edit_focus: frozenset = frozenset()
        bus.sig_edit_focus_changed.connect(self._on_edit_focus)
        self._tempo_segments = np.empty((0, 5), dtype=float)
        self._key_segments = None

        # Plot container
        self.gl = _TrackedGraphicsLayoutWidget(self.bus, show=True)
        self.p = self.gl.addPlot(row=0, col=0, viewBox=ScratchViewBox(
            self.bus,
            get_time=lambda: self.current_time,
            get_duration=lambda: float(self.model.duration_sec or 0.0),
            can_scratch=self._can_use_transport,
            get_window=lambda: self.tl.window_sec,
        ))
        self.p.hideAxis("left")
        self.p.hideAxis("bottom")
        self.p.hideButtons()
        self.gl.setMaximumHeight(150)

        # Overview (timeline summary)
        self.ov = OverviewWidget(bus, tl=self.tl, model=self.model)
        self.ov.setMinimumHeight(103)
        self.ov.setMaximumHeight(103)

        # Header (track info + transport + editors)
        self._init_header_ui()

        # Library
        self.lib = LibraryWidget(bus, self.cfg, model=self.model)

        # Layout
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(6, 6, 6, 6)
        v.setSpacing(6)
        v.addWidget(self.gl)
        v.addWidget(self.header)
        v.addWidget(self.lib, stretch=5)

        self._is_playing = False
        self._transport_enabled = True
        self._external_sync_enabled = bool(
            getattr(getattr(self.cfg, "externalsyncconfig", None), "enabled", False)
        )
        self.plugins: List[ViewPlugin] = []

        # Signals
        self.bus.sig_time_changed.connect(self._on_time)
        self.bus.sig_window_changed.connect(self._on_window)
        self.p.getViewBox().sigResized.connect(self._update_pixel_sec)
        self._update_pixel_sec()
        self.bus.sig_output_peak_dbfs.connect(self._on_output_peak_dbfs)
        self.bus.sig_features_loaded.connect(self._on_features)
        self.bus.sig_properties_loaded.connect(self._on_properties)
        self.bus.sig_albumart_loaded.connect(self._on_albumart)
        self.bus.sig_playback_status.connect(self._on_playback_state_changed)
        self.bus.sig_reload_UI.connect(self._on_reload_requested)
        self.bus.sig_beatgrid_edited.connect(self._on_beatgrid_edited)
        self.bus.sig_key_segments_updated.connect(self._on_key_segments_updated)
        self.bus.sig_phrase_segments_updated.connect(self._on_phrases_updated)
        try:
            self.bus.sig_transport_enabled.connect(self._on_transport_enabled)
            self.bus.sig_external_sync_enabled.connect(self._on_external_sync_enabled)
        except Exception:
            pass

        # Default plot range
        self.p.setYRange(-0.14, 1.0, padding=0.0)

    # UI layout
    def _init_header_ui(self) -> None:
        ov_height = self.ov.maximumHeight() if hasattr(self, "ov") else 0
        # Provisional height; the real height (accounting for actual child sizes
        # and inter-widget spacing) is applied by _fit_header_height from the
        # header layout's own sizeHint once all children exist.
        HEADER_H = TrackInfoPanel.HEADER_H + ov_height
        self.header = QtWidgets.QWidget()
        self.header.setMinimumHeight(HEADER_H)
        self.header.setMaximumHeight(HEADER_H)

        header_v = QtWidgets.QVBoxLayout(self.header)
        header_v.setContentsMargins(0, 0, 0, 0)
        header_v.setSpacing(6)

        # breathing room around the deck row: off the waveforms above / below
        # and off the window edges
        top_row = QtWidgets.QHBoxLayout()
        top_row.setContentsMargins(8, 8, 14, 8)
        top_row.setSpacing(12)
        top_row.setAlignment(QtCore.Qt.AlignVCenter)
        header_v.addLayout(top_row, stretch=0)

        self.track_info = TrackInfoPanel(self.cfg)
        self.track_info.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.track_info.unload_button.clicked.connect(self.bus.sig_unload_track_requested.emit)
        top_row.addWidget(self.track_info, 1, QtCore.Qt.AlignVCenter)

        # the edit panel sits in the track info header, between the info and the deck
        self.track_edit = EditPanel(bus=self.bus, model=self.model)
        self.track_info.editor_slot.addWidget(self.track_edit, 0, QtCore.Qt.AlignVCenter)
        self.track_edit.set_metronome_checked(self._metronome_enabled())
        # more edit pages side by side when the header has room for them
        self.track_info.installEventFilter(self)

        def _match_track_info_height(widget: QtWidgets.QWidget) -> None:
            widget.setMinimumHeight(TrackInfoPanel.HEADER_H)
            widget.setMaximumHeight(TrackInfoPanel.HEADER_H)

        # Volume column
        volume_box = QtWidgets.QWidget()
        volume_v = QtWidgets.QVBoxLayout(volume_box)
        volume_v.setContentsMargins(0, 0, 0, 0)
        volume_v.setSpacing(2)
        playback_cfg = getattr(self.cfg, "playbackconfig", None)
        default_volume_slider = int(
            max(
                0,
                min(
                    100,
                    int(getattr(playback_cfg, "default_volume_percent", 100) or 100),
                ),
            )
        )

        self.vol_slider = _OneStepWheelSlider(QtCore.Qt.Vertical)
        self.vol_slider.setRange(0, 100)
        self.vol_slider.setValue(default_volume_slider)
        self.vol_slider.setToolTip("Volume")

        self.l_volume = QtWidgets.QLabel("Volume")
        f = self.l_volume.font()
        f.setPointSize(9)
        f.setBold(True)
        self.l_volume.setFont(f)
        self.l_volume.setAlignment(QtCore.Qt.AlignHCenter)

        self.v_volume = QtWidgets.QLabel("50%")
        f = self.v_volume.font()
        f.setPointSize(9)
        f.setBold(True)
        self.v_volume.setFont(f)
        self.v_volume.setAlignment(QtCore.Qt.AlignHCenter)

        volume_v.addWidget(self.l_volume, 0, QtCore.Qt.AlignHCenter)
        volume_v.addWidget(self.vol_slider, 0, QtCore.Qt.AlignHCenter)
        volume_v.addWidget(self.v_volume, 0, QtCore.Qt.AlignHCenter)

        _match_track_info_height(volume_box)

        self.vol_slider.valueChanged.connect(self._on_volume_slider_changed)
        QtCore.QTimer.singleShot(0, lambda: self._on_volume_slider_changed(self.vol_slider.value()))

        # Tempo column
        tempo_box = QtWidgets.QWidget()
        tempo_v = QtWidgets.QVBoxLayout(tempo_box)
        tempo_v.setContentsMargins(0, 0, 0, 0)
        tempo_v.setSpacing(2)

        self.l_tempo = QtWidgets.QLabel("Tempo")
        f = self.l_tempo.font()
        f.setPointSize(9)
        f.setBold(True)
        self.l_tempo.setFont(f)
        self.l_tempo.setAlignment(QtCore.Qt.AlignHCenter)

        self.v_tempo = QtWidgets.QLabel(_tempo_text(100))
        f = self.v_tempo.font()
        f.setPointSize(9)
        f.setBold(True)
        self.v_tempo.setFont(f)
        self.v_tempo.setAlignment(QtCore.Qt.AlignHCenter)
        # fixed width: the widest reading, so the fader column never shifts
        self.v_tempo.setFixedWidth(QtGui.QFontMetrics(f).horizontalAdvance(_tempo_text(0)) + 4)

        self.tempo_slider = _OneStepWheelSlider(QtCore.Qt.Vertical)
        self.tempo_slider.setRange(0, 200)  # 0 halts the deck
        self.tempo_slider.setValue(100)
        self.tempo_slider.setToolTip("Tempo (0%-200%, 0 stops)")

        tempo_v.addWidget(self.l_tempo, 0, QtCore.Qt.AlignHCenter)
        tempo_v.addWidget(self.tempo_slider, 0, QtCore.Qt.AlignHCenter)
        tempo_v.addWidget(self.v_tempo, 0, QtCore.Qt.AlignHCenter)

        _match_track_info_height(tempo_box)
        top_row.addWidget(volume_box, 0, QtCore.Qt.AlignVCenter)
        top_row.addWidget(tempo_box, 0, QtCore.Qt.AlignVCenter)

        self.tempo_slider.valueChanged.connect(
            lambda value: (
                self.v_tempo.setText(_tempo_text(value)),
                self.bus.sig_tempo_factor_changed.emit(value / 100.0)
            )
        )
        self.tempo_slider.valueChanged.connect(
            lambda value: self.bus.sig_tempo_mode_changed.emit("none" if abs(value-100) < 1e-9 else "speed")
        )

        # Deck buttons, after a DJ deck: square MT (time stretch) / RST (tempo
        # reset) in a row above the record, round stop / play left of it.
        ROUND_D = int(TrackInfoPanel.HEADER_H / 3.5)
        SQUARE_H = int(TrackInfoPanel.HEADER_H / 6)
        COLUMN_W = int(ROUND_D * 1.15)

        def _mk_round_btn(icon: str, ring: str) -> QtWidgets.QToolButton:
            btn = QtWidgets.QToolButton()
            btn.setIcon(_transport_icon(icon))
            btn.setIconSize(QtCore.QSize(int(ROUND_D * 0.6), int(ROUND_D * 0.6)))
            btn.setFixedSize(ROUND_D, ROUND_D)
            radius = int(ROUND_D // 2)
            btn.setStyleSheet(
                f"QToolButton {{ border: 2px solid {ring}; background: #141414; border-radius: {radius}px; }}\n"
                "QToolButton:hover  { background: #1c1c1c; }\n"
                "QToolButton:pressed{ background: #0c0c0c; }\n"
                "QToolButton:disabled { border-color: #3a3a3a; }\n"
                f"QToolButton[playing=true] {{ background: {ring}; }}\n"
            )
            btn.setProperty("playing", False)
            return btn

        def _mk_square_btn(text: str, checkable: bool = False) -> QtWidgets.QToolButton:
            btn = QtWidgets.QToolButton()
            btn.setText(text)
            font_btn = btn.font()
            font_btn.setPointSize(9)
            font_btn.setBold(True)
            btn.setFont(font_btn)
            btn.setFixedSize(COLUMN_W, SQUARE_H)
            btn.setCheckable(checkable)
            btn.setStyleSheet(
                "QToolButton { border: 1px solid #4a4a4a; background: #181818; border-radius: 3px; color: #e6e6e6; }\n"
                "QToolButton:hover  { background: #222; }\n"
                "QToolButton:pressed{ background: #0c0c0c; }\n"
                "QToolButton:checked { background: #1f6fd1; border-color: #3d8bf0; color: #fff; }\n"
            )
            return btn

        self.btn_stop = _mk_round_btn("stop", "#8a8a8a")
        self.btn_play = _mk_round_btn("play", "#3fbf5f")
        self.btn_play.setCheckable(True)

        self.btn_tempo_reset = _mk_square_btn("RST")
        self.btn_tempo_reset.setToolTip("Tempo reset (100%)")
        self.btn_mt = _mk_square_btn("MT", checkable=True)
        self.btn_mt.setToolTip("Master tempo: time stretch keeps the pitch")
        self.btn_mt.setChecked(self._timestretch_enabled())

        for btn in (self.btn_mt, self.btn_tempo_reset):
            self.track_info.record_buttons_slot.addWidget(btn, 0, QtCore.Qt.AlignVCenter)
        for btn in (self.btn_stop, self.btn_play):
            self.track_info.transport_slot.addWidget(btn, 0, QtCore.Qt.AlignHCenter)

        header_v.addWidget(self.ov, 0, QtCore.Qt.AlignTop)

        # Wire transport buttons
        self.btn_tempo_reset.clicked.connect(self._on_click_tempo_reset)
        self.btn_mt.toggled.connect(self.bus.sig_timestretch_toggle_requested.emit)
        self.btn_stop.clicked.connect(self._on_click_stop)
        self.btn_play.clicked.connect(self._on_click_play)
        self._fit_header_height()

    # View management
    def install_default(self) -> None:
        """Install the default set of analysis views based on the current config."""
        self._install_views_from_config(self.cfg)

    def _install_views_from_config(self, cfg: config) -> None:
        self.clear_views()
        view_cfg = cfg.viewconfig
        if view_cfg.display_waveform:
            self.add_view("WaveformView")
        if view_cfg.display_beatgrid:
            self.add_view("BeatgridView")
        if view_cfg.display_keystrip:
            self.add_view("KeyStripView")
        if view_cfg.display_JumpCUE:
            self.add_view("JumpCUEView")
        if getattr(view_cfg, "display_phrase", True):
            self.add_view("PhraseView")
        self.add_view("CUEPointView")
        if hasattr(self, "ov") and hasattr(self.ov, "set_phrase_visible"):
            self.ov.set_phrase_visible(getattr(view_cfg, "display_phrase", True))
        # last, so the playhead is drawn over everything (it also has the top z value)
        self.add_view("PlayHead")
        self.p.setYRange(0, 1.0, padding=0.05)

    def clear_views(self) -> None:
        """Detach and forget all currently attached view plugins."""
        for plugin in self.plugins:
            try:
                plugin.detach()
            except Exception:
                pass
        self.plugins.clear()

    def add_view(self, name: str) -> Optional[ViewPlugin]:
        """Instantiate and attach a view plugin by name."""
        cls: Optional[Type[ViewPlugin]] = REGISTRY.get(name)
        if cls is None:
            print(f"[MainPane] view '{name}' is not registered")
            return None
        plugin = cls(self.bus, self.model, self.tl)
        plugin.attach(self.p)
        try:
            plugin.render_initial()
        except Exception:
            # Not all plugins implement render_initial
            pass
        plugin.apply_edit_focus(self._edit_focus)
        self.plugins.append(plugin)
        return plugin

    def _on_edit_focus(self, focus) -> None:
        self._edit_focus = frozenset(focus or ())

    # Transport handlers
    def _on_click_tempo_reset(self):
        self.tempo_slider.blockSignals(True)
        self.tempo_slider.setValue(100)
        self.tempo_slider.blockSignals(False)
        self.v_tempo.setText(_tempo_text(100))
        self.bus.sig_tempo_factor_changed.emit(1.0)
        self.bus.sig_tempo_mode_changed.emit("none")

    def _on_click_stop(self):
        self.bus.sig_stop_requested.emit(True)
        self._on_playback_state_changed(False)

    def _on_click_play(self):
        if self._external_sync_enabled:
            QtWidgets.QMessageBox.information(
                self,
                "External Sync",
                "Playback is disabled while External Sync is enabled.",
            )
            return
        want_play = not bool(getattr(self, "_is_playing", False))
        if want_play:
            self.bus.sig_play_requested.emit(True)
        else:
            self.bus.sig_pause_requested.emit(True)

    def _on_volume_slider_changed(self, value: int) -> None:
        self.v_volume.setText(f"{value}%")
        trim_dbfs = float(
            getattr(
                getattr(self.cfg, "playbackconfig", None),
                "volume_trim_dbfs",
                -6.0,
            )
        )
        linear_volume = slider_percent_to_linear(value, trim_dbfs)
        self.bus.sig_volume_changed.emit(float(linear_volume))
        use_output_volume = bool(
            getattr(
                getattr(self.cfg, "viewconfig", None),
                "use_output_volume_as_peak_meter_input",
                True,
            )
        )
        peak_meter_gain = (
            linear_volume if use_output_volume else trim_dbfs_to_linear(trim_dbfs)
        )
        self.bus.sig_peak_meter_gain_changed.emit(float(peak_meter_gain))

    # Signals
    def _on_window(self, _window_sec: float) -> None:
        self.p.setXRange(0.0, self.tl.window_sec, padding=0)
        self._update_pixel_sec()

    def _update_pixel_sec(self, *_args) -> None:
        """The main view's track seconds per device pixel, for the timeline's
        whole-pixel offset (TimelineCoordinator.left_edge)."""
        width = float(self.p.getViewBox().width()) * float(self.gl.devicePixelRatioF())
        self.tl.set_pixel_sec(self.tl.window_sec / width if width > 0.0 else 0.0)

    def _on_time(self, t: float):
        self.current_time = t
        self.track_info.update_record_pix(t)
        f = self.model.features
        if hasattr(self, "track_edit"):
            self.track_edit.update_time(t)
        self.p.setXRange(0.0, self.tl.window_sec, padding=0)

        self._update_bpm_display(t)
        
        active_key = self._key_from_segments(self._key_segments, t)
        speed = self.tempo_slider.value() / 100
        if active_key is not None and speed <= 0.0 and not self._timestretch_enabled():
            # varispeed at tempo 0: no pitch to show (12 * log2(0) is -inf)
            self.track_info.set_key(None)
        elif active_key is not None:
            if self._timestretch_enabled():
                # The time stretch keeps the pitch: the key does not follow the tempo.
                semitone, semitone_int = 0.0, 0
            else:
                semitone, semitone_int, _ = speed_to_semitone(speed)
            if active_key < 12:
                ks_v = (active_key + semitone_int) % 12
            else:
                ks_v = ((active_key - 12) + semitone_int) % 12 + 12
            keyl_value = idx_to_labels(int(ks_v))
            keyorig_value = idx_to_labels(int(active_key))
            self.track_info.set_key(keyl_value, keyorig_value, semitone)
        else:
            self.track_info.set_key(None)

        dur = max(self.model.duration_sec, 0.0)
        self.track_info.set_time(float(np.clip(t, 0.0, dur)), dur)
        self._update_phrase_display(t)
        self._update_beat_display(t)

    def _rebuild_beat_positions(self) -> None:
        """Per beat: its 0-based position in the bar and the bar's meter, from the same
        downbeats the beatgrid view draws (downbeat_beat_indices)."""
        f = self.model.features if isinstance(self.model.features, dict) else {}
        beats = np.asarray(f.get("beats_time_sec") if f.get("beats_time_sec") is not None else [], dtype=float)
        segments = f.get("tempo_segments")
        self._beat_times = beats
        self._beat_in_bar = np.zeros(beats.size, dtype=np.int64)
        self._beat_meter = np.full(beats.size, 4, dtype=np.int64)
        if beats.size == 0:
            return
        downs = downbeat_beat_indices(beats, segments)
        if downs.size:
            last = np.searchsorted(downs, np.arange(beats.size), side="right") - 1
            has = last >= 0
            self._beat_in_bar[has] = np.arange(beats.size)[has] - downs[last[has]]
            # beats before the first downbeat count back from it
            first = int(downs[0])
            self._beat_in_bar[~has] = (np.arange(beats.size)[~has] - first)
        try:
            seg = np.asarray(segments, dtype=float)
            if seg.ndim == 2 and seg.shape[0]:
                idx = np.clip(np.searchsorted(seg[:, 0], beats, side="right") - 1, 0, seg.shape[0] - 1)
                meters = np.asarray([segment_time_signature(r) for r in seg], dtype=np.int64)
                self._beat_meter = meters[idx]
        except Exception:
            pass
        self._beat_in_bar = np.mod(self._beat_in_bar, np.maximum(self._beat_meter, 1))

    def _update_beat_display(self, t: float) -> None:
        beats = getattr(self, "_beat_times", None)
        if beats is None or beats.size == 0:
            self.track_info.set_beat(None)
            return
        i = int(np.searchsorted(beats, float(t) + 1e-4, side="right")) - 1
        if i < 0:
            self.track_info.set_beat(None, int(self._beat_meter[0]))
            return
        self.track_info.set_beat(int(self._beat_in_bar[i]), int(self._beat_meter[i]))

    def _on_phrases_updated(self) -> None:
        """Cache the displayed phrases (numbered like the phrase view)."""
        f = self.model.features if isinstance(self.model.features, dict) else {}
        rows = extract_phrase_segments(f, self.model.duration_sec)
        self._phrase_starts = np.asarray([float(r["start"]) for r in rows], dtype=float)
        self._phrase_ends = np.asarray([float(r["end"]) for r in rows], dtype=float)
        self._phrase_names = number_phrase_labels(rows)
        self._phrase_colors = [phrase_color(r["label"]) for r in rows]
        self._update_phrase_display(self.current_time)

    def _update_phrase_display(self, t: float) -> None:
        starts = getattr(self, "_phrase_starts", None)
        if starts is None or starts.size == 0:
            self.track_info.set_phrase(None)
            return
        i = int(np.searchsorted(starts, float(t), side="right")) - 1
        if i < 0 or float(t) >= self._phrase_ends[i]:
            self.track_info.set_phrase(None)
            return
        self.track_info.set_phrase(self._phrase_names[i], self._phrase_colors[i])

    def _on_features(self):
        f = self.model.features
        self.ov.set_data()
        self._on_phrases_updated()
        self._rebuild_beat_positions()
        segments = f.get("tempo_segments")
        if hasattr(self, "track_edit"):
            self.track_edit.set_key_segments(f.get("key_segments"), initialize=True)
            self.track_edit.set_JumpCUE(extract_jump_cue_pairs(f))
            self.track_edit.set_cue_points(extract_cue_points(f))
            self.track_edit.set_phrase_segments(extract_phrase_segments(f, self.model.duration_sec), initialize=True)
            # Build the initial undo snapshot only after all editor-owned state
            # has been loaded, so undo cannot restore stale Phrase/JumpCUE data.
            self.track_edit.set_segments(f.get("beats_time_sec"), segments)
        self._set_tempo_segments(segments)
        key_segments = f.get("key_segments")
        self._key_segments = None
        if key_segments is not None:
            try:
                self._key_segments = np.asarray(key_segments, dtype=float)
            except Exception:
                self._key_segments = None
        if self._key_segments is None:
            self.track_info.set_key(None)  # e.g. unloaded

    def _on_beatgrid_edited(self):
        if not isinstance(self.model.features, dict):
            return
        beatgrid = self.model.features.get("beats_time_sec")
        segments = self.model.features.get("tempo_segments")
        self._set_tempo_segments(segments)
        self._rebuild_beat_positions()
        self._update_beat_display(self.current_time)

    def _set_tempo_segments(self, segments) -> None:
        """Keep [start, end, bpm, ts_num, bpm_end] rows (ts_num 4 / steady when absent) for
        the BPM line; a ramp's BPM is read at the playhead."""
        self._tempo_segments = np.empty((0, 5), dtype=float)
        if segments is not None:
            arr = np.asarray(segments, dtype=float)
            if arr.ndim == 2 and arr.shape[1] >= 3:
                rows = np.full((arr.shape[0], 5), 4.0)
                rows[:, :3] = arr[:, :3]
                if arr.shape[1] >= 5:
                    rows[:, 3] = arr[:, 4]
                rows[:, 4] = [segment_end_bpm(row) for row in arr] if arr.shape[1] > 5 else arr[:, 2]
                self._tempo_segments = rows
        self._update_bpm_display()

    def _key_from_segments(self, key_segments, t: float):
        try:
            arr = np.asarray(key_segments, dtype=float)
        except Exception:
            return None
        if arr.ndim != 2 or arr.shape[1] < 4:
            return None
        t_val = float(t)
        for pitch, mode_flag, start, end in arr:
            if not (np.isfinite(start) and np.isfinite(end)):
                continue
            if start <= t_val < end or (t_val >= end and np.isclose(t_val, end)):
                if not np.isfinite(pitch):
                    continue
                key_idx = int(pitch)
                if np.isfinite(mode_flag) and int(mode_flag) == 1:
                    key_idx += 12
                return key_idx
        return None

    def _current_segment(self, t: float) -> tuple[float, int]:
        """(bpm, beats per bar) of the tempo segment at ``t``; bpm is nan when unknown."""
        segs = getattr(self, "_tempo_segments", None)
        if segs is None or segs.size == 0:
            return float("nan"), 4
        starts = segs[:, 0]
        ends = segs[:, 1]
        within = (starts <= t) & (t < ends)
        if np.any(within):
            idx = int(np.argmax(within))
        elif t < starts[0]:
            idx = 0
        elif t >= ends[-1]:
            idx = len(starts) - 1
        else:
            idx = int(np.searchsorted(starts, t, side="left"))
            idx = min(idx, len(starts) - 1)
        bpm = float(segs[idx, 2])
        start, end, end_bpm = float(segs[idx, 0]), float(segs[idx, 1]), float(segs[idx, 4])
        if abs(end_bpm - bpm) > 1e-6 and end > start:
            # Tempo ramp: the tempo at the playhead, held at the ramp's ends outside it.
            frac = min(max((float(t) - start) / (end - start), 0.0), 1.0)
            bpm = bpm + (end_bpm - bpm) * frac
        meter = int(round(segs[idx, 3])) if np.isfinite(segs[idx, 3]) and segs[idx, 3] >= 1 else 4
        return (bpm if np.isfinite(bpm) and bpm > 0 else float("nan")), meter

    def _update_bpm_display(self, t: float | None = None) -> None:
        if not hasattr(self, "track_info"):
            return
        cur_t = self.current_time if t is None else float(t)
        bpm, meter = self._current_segment(cur_t)
        factor = self.tempo_slider.value() / 100.0
        if not np.isfinite(bpm):
            self.track_info.set_bpm(None)
        else:
            self.track_info.set_bpm(bpm * factor, bpm, factor * 100.0 - 100.0, meter)

    def _on_properties(self):
        p = self.model.properties
        title_text = p.get("title") or "Not Loaded" if isinstance(p, dict) else "Not Loaded"
        artist_text = p.get("artist") or "" if isinstance(p, dict) else ""
        self.track_info.set_title(title_text.strip())
        self.track_info.set_artist(artist_text.strip())
        self.track_edit.set_track_uid(p.get("uid"))
        self.track_edit.set_track_duration(p.get("duration_sec"))
        self.track_info.set_track_loaded(float(self.model.duration_sec or 0.0) > 0.0)

    def _on_albumart(self):
        self.track_info.update_album_art(getattr(self.model, "album_art", None))

    def _on_output_peak_dbfs(self, dbfs: float) -> None:
        self.track_info.set_output_peak_dbfs(dbfs)

    def _on_playback_state_changed(self, is_playing: bool):
        self._is_playing = bool(is_playing)
        self.track_info.update_record_pix(self.current_time)
        if hasattr(self, "btn_play"):
            self.btn_play.blockSignals(True)
            self.btn_play.setProperty("playing", self._is_playing)
            self.btn_play.setChecked(self._is_playing)
            self.btn_play.setIcon(_transport_icon("pause" if self._is_playing else "play"))
            self.btn_play.style().unpolish(self.btn_play)
            self.btn_play.style().polish(self.btn_play)
            self.btn_play.update()
            self.btn_play.blockSignals(False)

    def _on_transport_enabled(self, enabled: bool) -> None:
        self._transport_enabled = bool(enabled)
        self._refresh_transport_buttons()

    def _on_external_sync_enabled(self, enabled: bool) -> None:
        self._external_sync_enabled = bool(enabled)
        self._refresh_transport_buttons()

    # Spare width (px) the track info keeps beyond its widest row before the edit
    # panel shows more pages side by side.
    EDIT_COLUMN_SLACK = 80

    def eventFilter(self, obj, event):
        if obj is self.track_info and event.type() == QtCore.QEvent.Resize:
            QtCore.QTimer.singleShot(0, self._fit_edit_columns)  # after the layout has run
        return super().eventFilter(obj, event)

    def _fit_edit_columns(self) -> None:
        """Show the edit pages 1, 2 or 4 at a time: more while the track info has
        room to spare for them, fewer once the header is too narrow."""
        info, edit = self.track_info, self.track_edit
        count = edit.column_count()
        squeezed = edit.width() < edit.sizeHint().width()
        spare = info.info_area.width() - info.info_area.minimumWidth() - self.EDIT_COLUMN_SLACK
        if squeezed and count > 1:
            new = 2 if count == 4 else 1
        elif count < 4 and spare >= ((4 if count == 2 else 2) - count) * edit.column_step():
            new = 4 if count == 2 else 2
        else:
            return
        edit.set_column_count(new)
        QtCore.QTimer.singleShot(0, self._fit_edit_columns)  # one step at a time

    def _can_use_transport(self) -> bool:
        return self._transport_enabled and not self._external_sync_enabled

    def _refresh_transport_buttons(self) -> None:
        can_use_transport = self._can_use_transport()
        for btn_name in ("btn_play", "btn_stop"):
            btn = getattr(self, btn_name, None)
            if btn is not None:
                btn.setEnabled(can_use_transport)

    def _on_reload_requested(self, cfg: config):
        self.cfg = cfg
        self._external_sync_enabled = bool(
            getattr(getattr(cfg, "externalsyncconfig", None), "enabled", False)
        )
        self.track_info.set_config(cfg)
        self._install_views_from_config(cfg)
        self._refresh_transport_buttons()
        print("UI Reloaded")

    def apply_playback_config(self, cfg: config) -> None:
        self.cfg = cfg
        current_volume = int(self.vol_slider.value())
        current_volume = max(0, min(100, current_volume))
        self._on_volume_slider_changed(current_volume)
        self._on_time(self.current_time)  # key display depends on use_timestretch
        self.btn_mt.blockSignals(True)
        self.btn_mt.setChecked(self._timestretch_enabled())
        self.btn_mt.blockSignals(False)
        self.track_edit.set_metronome_checked(self._metronome_enabled())

    def _metronome_enabled(self) -> bool:
        return bool(getattr(getattr(self.cfg, "playbackconfig", None), "enable_metronome", False))

    def _timestretch_enabled(self) -> bool:
        return bool(getattr(getattr(self.cfg, "playbackconfig", None), "use_timestretch", False))

    def _fit_header_height(self) -> None:
        # Let the header layout report the exact height it needs. It accounts
        # for the real top-row height, the inter-widget spacing, and whether the
        # editor panel is currently shown. Manual sums under-counted these and
        # clipped the editor panel's bottom status line.
        lay = self.header.layout()
        if lay is not None:
            lay.activate()
            header_h = lay.sizeHint().height()
            self.header.setMinimumHeight(header_h)
            self.header.setMaximumHeight(header_h)

    def _on_key_segments_updated(self) -> None:
        key_segments = self.model.features.get("key_segments")
        try:
            self._key_segments = np.asarray(key_segments, dtype=float)
        except Exception:
            self._key_segments = None
        if hasattr(self.ov, "update_key_strip"):
            self.ov.update_key_strip()

