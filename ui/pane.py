from __future__ import annotations

from typing import List, Optional, Type
import time

from PySide6 import QtWidgets, QtGui, QtCore, QtMultimedia
import pyqtgraph as pg
import numpy as np

from views.base import REGISTRY, ViewPlugin
from views.overview import OverviewWidget
from ui.library import LibraryWidget
from utils.labels import idx_to_labels
from utils.semitone import speed_to_semitone
from utils.jump_cues import extract_jump_cue_pairs
from utils.phrases import extract_phrase_segments
from core.event_bus import EventBus
from core.config import config
from ui.mainplotvbx import NoDragNoWheelViewBox
from ui.track_info_panel import TrackInfoPanel
from ui.beatgrid_edit_panel import BeatgridEditPanel
from utils.volume import slider_percent_to_linear, trim_dbfs_to_linear


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
        self._tempo_segments = np.empty((0, 4), dtype=float)
        self._key_segments = None

        # Plot container
        self.gl = _TrackedGraphicsLayoutWidget(self.bus, show=True)
        self.p = self.gl.addPlot(row=0, col=0, viewBox=NoDragNoWheelViewBox())
        self.p.hideAxis("left")
        self.p.hideAxis("bottom")
        self.p.hideButtons()
        self.gl.setMaximumHeight(150)

        # Overview (timeline summary)
        self.ov = OverviewWidget(bus, tl=self.tl, model=self.model)
        self.ov.setMaximumHeight(80)

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
        self.bus.sig_output_peak_dbfs.connect(self._on_output_peak_dbfs)
        self.bus.sig_features_loaded.connect(self._on_features)
        self.bus.sig_properties_loaded.connect(self._on_properties)
        self.bus.sig_albumart_loaded.connect(self._on_albumart)
        self.bus.sig_playback_status.connect(self._on_playback_state_changed)
        self.bus.sig_reload_UI.connect(self._on_reload_requested)
        self.bus.sig_beatgrid_edited.connect(self._on_beatgrid_edited)
        self.bus.sig_key_segments_updated.connect(self._on_key_segments_updated)
        self.bus.sig_jumpcue_updated.connect(self._on_jumpcue_updated)
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
        editor_toggle_h = 24
        HEADER_H = TrackInfoPanel.HEADER_H + ov_height + editor_toggle_h + BeatgridEditPanel.ROW_H
        self.header = QtWidgets.QWidget()
        self.header.setMinimumHeight(HEADER_H)
        self.header.setMaximumHeight(HEADER_H)

        header_v = QtWidgets.QVBoxLayout(self.header)
        header_v.setContentsMargins(0, 0, 0, 0)
        header_v.setSpacing(4)

        top_row = QtWidgets.QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(8)
        top_row.setAlignment(QtCore.Qt.AlignVCenter)
        header_v.addLayout(top_row, stretch=0)

        self.track_info = TrackInfoPanel(self.cfg)
        self.track_info.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        top_row.addWidget(self.track_info, 1, QtCore.Qt.AlignVCenter)

        self.track_edit = BeatgridEditPanel(bus=self.bus, model=self.model)

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
        top_row.addWidget(volume_box, 0, QtCore.Qt.AlignVCenter)

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

        self.v_tempo = QtWidgets.QLabel("100%")
        f = self.v_tempo.font()
        f.setPointSize(9)
        f.setBold(True)
        self.v_tempo.setFont(f)
        self.v_tempo.setAlignment(QtCore.Qt.AlignHCenter)

        self.tempo_slider = _OneStepWheelSlider(QtCore.Qt.Vertical)
        self.tempo_slider.setRange(50, 200)
        self.tempo_slider.setValue(100)
        self.tempo_slider.setToolTip("Tempo (50%-200%)")

        tempo_v.addWidget(self.l_tempo, 0, QtCore.Qt.AlignHCenter)
        tempo_v.addWidget(self.tempo_slider, 0, QtCore.Qt.AlignHCenter)
        tempo_v.addWidget(self.v_tempo, 0, QtCore.Qt.AlignHCenter)

        _match_track_info_height(tempo_box)
        top_row.addWidget(tempo_box, 0, QtCore.Qt.AlignVCenter)

        self.tempo_slider.valueChanged.connect(
            lambda value: (
                self.v_tempo.setText(f"{value}%"),
                self.bus.sig_tempo_factor_changed.emit(value / 100.0)
            )
        )
        self.tempo_slider.valueChanged.connect(
            lambda value: self.bus.sig_tempo_mode_changed.emit("none" if abs(value-100) < 1e-9 else "speed")
        )

        # Transport controls
        ctrl_box = QtWidgets.QWidget()
        ctrl_v = QtWidgets.QVBoxLayout(ctrl_box)
        ctrl_v.setContentsMargins(0, 0, 0, 0)
        ctrl_v.setSpacing(6)
        ctrl_v.setAlignment(QtCore.Qt.AlignVCenter)

        BTN_D = TrackInfoPanel.HEADER_H / 3.5

        def _mk_round_btn(symbol: str, fontsize: int = 14) -> QtWidgets.QToolButton:
            btn = QtWidgets.QToolButton()
            btn.setText(symbol)
            font_btn = btn.font()
            font_btn.setPointSize(fontsize)
            font_btn.setBold(True)
            btn.setFont(font_btn)
            btn.setFixedSize(BTN_D, BTN_D)
            btn.setCheckable(False)
            radius = int(BTN_D // 2)
            style = (
                "QToolButton {{\n"
                "    border: 1px solid #555;\n"
                "    background: #111;\n"
                "    border-radius: {radius}px;\n"
                "    color: #ddd;\n"
                "}}\n"
                "QToolButton:hover  {{ background: #1a1a1a; }}\n"
                "QToolButton:pressed{{ background: #0c0c0c; }}\n"
                "QToolButton[playing=true] {{\n"
                "    background: #1f6f1f; border-color:#3faa3f; color: #fff;\n"
                "}}\n"
            ).format(radius=radius)
            btn.setStyleSheet(style)
            btn.setProperty("playing", False)
            return btn

        self.btn_tempo_reset = _mk_round_btn("↻", 12)
        self.btn_stop = _mk_round_btn("■", 10)
        self.btn_play = _mk_round_btn("▶", 10)
        self.btn_play.setCheckable(True)

        ctrl_v.addStretch(1)
        ctrl_v.addWidget(self.btn_tempo_reset, 0, QtCore.Qt.AlignHCenter)
        ctrl_v.addWidget(self.btn_stop, 0, QtCore.Qt.AlignHCenter)
        ctrl_v.addWidget(self.btn_play, 0, QtCore.Qt.AlignHCenter)
        ctrl_v.addStretch(1)

        _match_track_info_height(ctrl_box)
        top_row.addWidget(ctrl_box, 0, QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        header_v.addWidget(self.ov, 0, QtCore.Qt.AlignTop)
        toggle_row = QtWidgets.QHBoxLayout()
        toggle_row.setContentsMargins(0, 0, 0, 0)
        toggle_row.setSpacing(12)
        self.cb_show_editors = QtWidgets.QCheckBox("Show Editors")
        self.cb_show_editors.setChecked(False)
        self.cb_show_editors.setMinimumHeight(editor_toggle_h)
        self.cb_show_editors.setMaximumHeight(editor_toggle_h)

        # Editor page toggle (moved here, next to "Show Editors"):
        # Beat/Key | Phrase/JumpCUE -> switches the editor panel's stacked page.
        self.btn_page_beatkey = QtWidgets.QToolButton()
        self.btn_page_beatkey.setText("Beat/Key")
        self.btn_page_beatkey.setCheckable(True)
        self.btn_page_beatkey.setChecked(True)
        self.btn_page_phrasejc = QtWidgets.QToolButton()
        self.btn_page_phrasejc.setText("Phrase/JumpCUE")
        self.btn_page_phrasejc.setCheckable(True)
        for _b in (self.btn_page_beatkey, self.btn_page_phrasejc):
            _b.setMinimumHeight(editor_toggle_h)
            _b.setMaximumHeight(editor_toggle_h)
        self._editor_page_group = QtWidgets.QButtonGroup(self)
        self._editor_page_group.setExclusive(True)
        self._editor_page_group.addButton(self.btn_page_beatkey, 0)
        self._editor_page_group.addButton(self.btn_page_phrasejc, 1)
        self._editor_page_group.idClicked.connect(self.track_edit.set_active_page)

        self.cb_show_transition_search = QtWidgets.QCheckBox("Show Transition Search")
        self.cb_show_transition_search.setChecked(False)
        self.cb_show_transition_search.setMinimumHeight(editor_toggle_h)
        self.cb_show_transition_search.setMaximumHeight(editor_toggle_h)
        toggle_row.addWidget(self.cb_show_editors, 0, QtCore.Qt.AlignLeft)
        toggle_row.addWidget(self.btn_page_beatkey, 0, QtCore.Qt.AlignLeft)
        toggle_row.addWidget(self.btn_page_phrasejc, 0, QtCore.Qt.AlignLeft)
        toggle_row.addWidget(self.cb_show_transition_search, 0, QtCore.Qt.AlignLeft)
        toggle_row.addStretch(1)
        header_v.addLayout(toggle_row)
        header_v.addWidget(self.track_edit, 0, QtCore.Qt.AlignTop)

        # Wire transport buttons
        self.btn_tempo_reset.clicked.connect(self._on_click_tempo_reset)
        self.btn_stop.clicked.connect(self._on_click_stop)
        self.btn_play.clicked.connect(self._on_click_play)
        self.cb_show_editors.toggled.connect(self._set_editor_panel_visible)
        self.cb_show_transition_search.toggled.connect(self._set_transition_search_visible)
        self._set_editor_panel_visible(False)
        self._set_transition_search_visible(True)

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
        if hasattr(self, "ov") and hasattr(self.ov, "set_phrase_visible"):
            self.ov.set_phrase_visible(getattr(view_cfg, "display_phrase", True))
        self.add_view("PlayHead")
        self.add_view("SelectionOverlayView")
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
        self.plugins.append(plugin)
        return plugin

    # Transport handlers
    def _on_click_tempo_reset(self):
        self.tempo_slider.blockSignals(True)
        self.tempo_slider.setValue(100)
        self.tempo_slider.blockSignals(False)
        self.v_tempo.setText("100%")
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
    def _on_time(self, t: float):
        self.current_time = t
        self.track_info.update_record_pix(t)
        f = self.model.features
        if hasattr(self, "track_edit"):
            self.track_edit.update_time(t)
        self.p.setXRange(0.0, self.tl.window_sec, padding=0)

        self._update_bpm_display(t)
        
        active_key = self._key_from_segments(self._key_segments, t)
        if active_key is not None:
            _, semitone_int, semitone_frac = speed_to_semitone(self.tempo_slider.value() / 100)
            if active_key < 12:
                ks_v = (active_key + semitone_int) % 12
            else:
                ks_v = ((active_key - 12) + semitone_int) % 12 + 12
            keyl_value = idx_to_labels(int(ks_v))
            keyorig_value = idx_to_labels(int(active_key))
            self.track_info.set_key_text(f"KEY: {keyl_value[0]} / {keyl_value[1]} ({semitone_int:0=+3d})")
            self.track_info.set_cent_text(f"{keyorig_value[0]} / {keyorig_value[1]} {semitone_int:0=+3d} Cent = {int(np.round(semitone_frac * 100)):0=+3d}")
        else:
            self.track_info.set_key_text("KEY: --")

        dur = max(self.model.duration_sec, 0.0)
        cur = np.clip(t, 0.0, dur)

        def fmt(sec: float) -> str:
            m = int(sec // 60)
            s = int(sec - 60 * m)
            ms = int((sec - 60 * m - s) * 100)
            return f"{m:02d}:{s:02d}.{ms:02d}"

        self.track_info.set_time_text(f"{fmt(cur)} / {fmt(dur)}")

    def _on_features(self):
        f = self.model.features
        self.ov.set_data()
        segments = f.get("tempo_segments")
        if hasattr(self, "track_edit"):
            self.track_edit.set_key_segments(f.get("key_segments"), initialize=True)
            self.track_edit.set_segments(f.get("beats_time_sec"), segments)
            self.track_edit.set_JumpCUE(extract_jump_cue_pairs(f))
            self.track_edit.set_phrase_segments(extract_phrase_segments(f), initialize=True)
        self._set_tempo_segments(segments)
        key_segments = f.get("key_segments")
        if key_segments is not None:
            try:
                self._key_segments = np.asarray(key_segments, dtype=float)
            except Exception:
                self._key_segments = None

    def _on_beatgrid_edited(self):
        if not isinstance(self.model.features, dict):
            return
        beatgrid = self.model.features.get("beats_time_sec")
        segments = self.model.features.get("tempo_segments")
        self._set_tempo_segments(segments)

    def _set_tempo_segments(self, segments) -> None:
        if segments is None:
            self._tempo_segments = np.empty((0, 4), dtype=float)
            self._update_bpm_display()
            return
        arr = np.asarray(segments, dtype=float)
        if arr.ndim == 1:
            cols = 4 if arr.size % 4 == 0 else 3 if arr.size % 3 == 0 else 0
            if cols:
                arr = arr.reshape((-1, cols))
        if arr.ndim != 2 or arr.shape[1] < 3:
            self._tempo_segments = np.empty((0, 4), dtype=float)
        else:
            if arr.shape[1] >= 4:
                self._tempo_segments = arr[:, :4].copy()
        padded = np.zeros((arr.shape[0], 4), dtype=float)
        padded[:, :3] = arr[:, :3]
        self._tempo_segments = padded
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

    def _current_segment_bpm(self, t: float) -> float:
        segs = getattr(self, "_tempo_segments", None)
        if segs is None or segs.size == 0:
            return float("nan")
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
        return bpm if np.isfinite(bpm) and bpm > 0 else float("nan")

    def _update_bpm_display(self, t: float | None = None) -> None:
        if not hasattr(self, "track_info"):
            return
        cur_t = self.current_time if t is None else float(t)
        bpm = self._current_segment_bpm(cur_t)
        factor = self.tempo_slider.value() / 100.0
        text = "BPM: --" if not np.isfinite(bpm) else f"BPM: {bpm * factor:.2f} ({bpm:.2f} BPM {int(np.round(factor*100-100)):0=+3d}%p)"
        self.track_info.set_bpm_text(text)

    def _on_properties(self):
        p = self.model.properties
        title_text = p.get("title") or "Not Loaded" if isinstance(p, dict) else "Not Loaded"
        artist_text = p.get("artist") or "" if isinstance(p, dict) else ""
        self.track_info.set_title(title_text.strip())
        self.track_info.set_artist(artist_text.strip())
        self.track_edit.set_track_uid(p.get("uid"))
        self.track_edit.set_track_duration(p.get("duration_sec"))

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
            self.btn_play.setText("||" if self._is_playing else "▶")
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

    def _refresh_transport_buttons(self) -> None:
        can_use_transport = self._transport_enabled and not self._external_sync_enabled
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

    def _set_editor_panel_visible(self, visible: bool) -> None:
        show = bool(visible)
        if hasattr(self, "track_edit"):
            self.track_edit.setVisible(show)
        for _attr in ("btn_page_beatkey", "btn_page_phrasejc"):
            _btn = getattr(self, _attr, None)
            if _btn is not None:
                _btn.setEnabled(show)
        ov_height = self.ov.maximumHeight() if hasattr(self, "ov") else 0
        toggle_h = self.cb_show_editors.maximumHeight() if hasattr(self, "cb_show_editors") else 24
        header_h = TrackInfoPanel.HEADER_H + ov_height + toggle_h + (BeatgridEditPanel.ROW_H if show else 0)
        self.header.setMinimumHeight(header_h)
        self.header.setMaximumHeight(header_h)

    def _set_transition_search_visible(self, visible: bool) -> None:
        if hasattr(self, "lib") and hasattr(self.lib, "set_transition_search_visible"):
            self.lib.set_transition_search_visible(bool(visible))

    def _on_key_segments_updated(self) -> None:
        key_segments = self.model.features.get("key_segments")
        key_image = self.model.features.get("key_np")
        try:
            self._key_segments = np.asarray(key_segments, dtype=float)
        except Exception:
            self._key_segments = None
        if key_image is not None and hasattr(self.ov, "update_key_image"):
            self.ov.update_key_image(key_image)

    def _on_jumpcue_updated(self) -> None:
        jc_block = self.model.features.get("jump_cues_np")
        extracted = self.model.features.get("jump_cues_extracted")
        if extracted is None and jc_block is not None:
            extracted = extract_jump_cue_pairs({"jump_cues_np": jc_block})
