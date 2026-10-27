from __future__ import annotations

from typing import List, Optional, Type
import time

from PySide6 import QtWidgets, QtGui, QtCore
import pyqtgraph as pg
import numpy as np

from audioengine import controls as deck

from views.base import REGISTRY, ViewPlugin
from views.overview import OverviewWidget
from ui.library import LibraryWidget
from analyzer.music.key_names import idx_to_labels
from ui.render.semitone import speed_to_semitone
from core.marks.phrases import number_phrase_labels, phrase_color
from core.workflows.event_bus import EventBus
from core.settings.config import config
from ui.deck.mainplotvbx import ScratchViewBox
from ui.deck.track_info_panel import TrackInfoPanel
from ui.edit_panel import EditPanel
from audioengine.volume import slider_percent_to_linear, trim_dbfs_to_linear
from system.resource_paths import resource_path
from analyzer.music.beat_geometry import downbeat_beat_indices, segment_time_signature
from analyzer.music.tempo_rows import segment_end_bpm
from views.timeline import step_window
from system.i18n import tr
from ui.theme import current_theme

# the tempo fader's step in each range (+-%), as rekordbox's; WIDE (100: 0 - 200 %) by whole percents
TEMPO_STEP = {6: 0.02, 10: 0.05, 16: 0.05, 100: 1.0}
CUE_COLOR = "#f2c014"   # the CUE button: yellow, as a CDJ's


def _tempo_text(factor: float, tempo_range: int = 100) -> str:
    """Tempo fader reading as DJ gear shows it: change from 100 %, WIDE in whole percents (+000% / −050% /
    +100%), a narrow range to its step (+3.40%); U+2212 minus, as wide as "+"."""
    change = (float(factor) - 1.0) * 100.0
    if tempo_range >= 100:
        return f"{int(round(change)):+04d}%".replace("-", "−")
    return f"{change:+.2f}%".replace("-", "−")


def _range_text(tempo_range: int) -> str:
    return "WIDE" if tempo_range >= 100 else f"±{tempo_range}%"


_transport_icons: dict[str, QtGui.QIcon] = {}


def _transport_icon(name: str) -> QtGui.QIcon:
    """Deck button icon from assets/icons/transport_<name>.png (stop / play / pause / cue, cue_lit)."""
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


class _DropZone(QtWidgets.QWidget):
    """Over the deck (everything above the library) while files are dragged: where they would load, lit while the
    pointer is over it."""

    def __init__(self, parent: QtWidgets.QWidget) -> None:
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.lit = False
        self.hide()

    def show_at(self, rect: QtCore.QRect, lit: bool) -> None:
        self.lit = lit
        self.setGeometry(rect)
        self.raise_()
        self.show()
        self.update()

    def paintEvent(self, event) -> None:
        theme = current_theme()
        accent = QtGui.QColor(theme.accent)
        p = QtGui.QPainter(self)
        font = p.font()
        font.setPointSize(16)
        font.setBold(True)
        p.setFont(font)
        rect = self.rect().adjusted(4, 4, -5, -5)
        fill = QtGui.QColor(accent)
        fill.setAlpha(70 if self.lit else 25)
        p.fillRect(rect, fill)
        edge = QtGui.QColor(accent)
        edge.setAlpha(230 if self.lit else 120)
        p.setPen(QtGui.QPen(edge, 2, QtCore.Qt.DashLine))
        p.drawRect(rect)
        text = QtGui.QColor(theme.text)
        text.setAlpha(230 if self.lit else 150)
        p.setPen(text)
        p.drawText(self.rect(), QtCore.Qt.AlignCenter, tr("Load to Deck"))


def _match_track_info_height(widget: QtWidgets.QWidget) -> None:
    """A deck column as tall as the track info beside it."""
    widget.setMinimumHeight(TrackInfoPanel.HEADER_H)
    widget.setMaximumHeight(TrackInfoPanel.HEADER_H)


class MainPane(QtWidgets.QWidget):
    """Main pane containing header controls, overview, and library. Files dragged over the deck (the area above
    the library) show where they would load (drop_zone_at / show_drop_zone); the window loads them dropped there,
    and the library's own drop only analyzes."""

    def __init__(self, bus: EventBus, model, tl, cfg: config, library):
        super().__init__()
        self.bus = bus
        self.model = model
        self.tl = tl
        self.cfg = cfg
        self.library = library      # the library's owner (core.library.service)
        self.current_time = 0.0
        # the edit pages emphasized (EditPanel, while unlocked), for the views created later
        self._edit_focus: frozenset = frozenset()
        bus.sig_edit_focus_changed.connect(self._on_edit_focus)
        self._tempo_segments = np.empty((0, 5), dtype=float)
        self._key_segments = None
        # the deck as the player reports it (sig_deck_state): what the faders and buttons show
        self._tempo_factor = 1.0
        self._tempo_range = int(cfg.deckconfig.tempo_range or 100)
        self._volume_position: Optional[float] = None   # none reported yet: the first report is shown, whatever it is
        self._cue_lit = False

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
        self.lib = LibraryWidget(bus, self.cfg, library, model=self.model)

        # Layout
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(6, 6, 6, 6)
        v.setSpacing(6)
        v.addWidget(self.gl)
        v.addWidget(self.header)
        v.addWidget(self.lib, stretch=5)
        self._drop_zone = _DropZone(self)

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
        self.model.sig_loaded.connect(self._on_features)
        self.model.sig_properties_changed.connect(self._on_properties)
        self.model.sig_album_art_changed.connect(self._on_albumart)
        self.bus.sig_playback_status.connect(self._on_playback_state_changed)
        self.bus.sig_deck_state.connect(self._on_deck_state)
        self.bus.sig_reload_UI.connect(self._on_reload_requested)
        self.model.sig_beatgrid_changed.connect(self._on_beatgrid_edited)
        self.model.sig_keys_changed.connect(self._on_key_segments_updated)
        self.model.sig_phrases_changed.connect(self._on_phrases_updated)
        self.bus.sig_phrase_labels_changed.connect(self._on_phrases_updated)
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
        # its channel's TRIM / EQ knobs: requests (the player's report moves them: _on_deck_state)
        self.track_info.knob_moved.connect(self.bus.sig_knob_requested.emit)
        top_row.addWidget(self.track_info, 1, QtCore.Qt.AlignVCenter)

        # the edit panel sits in the track info header, between the info and the deck
        self.track_edit = EditPanel(bus=self.bus, model=self.model, library=self.library)
        self.track_info.editor_slot.addWidget(self.track_edit, 0, QtCore.Qt.AlignVCenter)
        self.track_edit.set_metronome_checked(self._metronome_enabled())
        # more edit pages side by side when the header has room for them
        self.track_info.installEventFilter(self)

        volume_box = self._build_volume_column()
        tempo_box = self._build_tempo_column()
        top_row.addWidget(volume_box, 0, QtCore.Qt.AlignVCenter)
        top_row.addWidget(tempo_box, 0, QtCore.Qt.AlignVCenter)

        self._build_transport_buttons()

        header_v.addWidget(self.ov, 0, QtCore.Qt.AlignTop)
        self._fit_header_height()

    def _build_volume_column(self) -> QtWidgets.QWidget:
        """The Volume column: its head, the fader (a request: it shows the volume the player reports), the reading."""
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

        self.v_volume = QtWidgets.QLabel(f"{default_volume_slider}%")   # until the player reports its volume
        f = self.v_volume.font()
        f.setPointSize(9)
        f.setBold(True)
        self.v_volume.setFont(f)
        self.v_volume.setAlignment(QtCore.Qt.AlignHCenter)

        volume_v.addWidget(self.l_volume, 0, QtCore.Qt.AlignHCenter)
        volume_v.addWidget(self.vol_slider, 0, QtCore.Qt.AlignHCenter)
        volume_v.addWidget(self.v_volume, 0, QtCore.Qt.AlignHCenter)

        _match_track_info_height(volume_box)

        # a request: the fader shows the volume the player then reports
        self.vol_slider.valueChanged.connect(lambda value: self.bus.sig_volume_requested.emit(value / 100.0))
        return volume_box

    def _build_tempo_column(self) -> QtWidgets.QWidget:
        """The Tempo column: the range button as its head, the fader (a request: it shows the tempo the player reports), the reading."""
        tempo_box = QtWidgets.QWidget()
        tempo_v = QtWidgets.QVBoxLayout(tempo_box)
        tempo_v.setContentsMargins(0, 0, 0, 0)
        tempo_v.setSpacing(2)

        # the column's head: the fader's range (as rekordbox's TEMPO RANGE: +-6 / 10 / 16 % or WIDE), a button
        # that takes the next; as wide as its widest text, so WIDE fits
        self.btn_tempo_range = QtWidgets.QToolButton()
        f = self.btn_tempo_range.font()
        f.setPointSize(9)
        f.setBold(True)
        self.btn_tempo_range.setFont(f)
        fm = QtGui.QFontMetrics(f)
        widest = max(fm.horizontalAdvance(_range_text(r)) for r in TEMPO_STEP)
        self.btn_tempo_range.setFixedSize(widest + 2 * 4 + 2 + 4, fm.height() + 4)
        self.btn_tempo_range.setText(_range_text(self._tempo_range))
        self.btn_tempo_range.setToolTip(tr("Tempo range: ±6% / ±10% / ±16% / WIDE; click for the next"))
        self.btn_tempo_range.setStyleSheet(
            "QToolButton { border: 1px solid #4a4a4a; background: #181818; border-radius: 3px; color: #e6e6e6;"
            " padding: 0px 4px; }\n"
            "QToolButton:hover { background: #222; }\n"
            "QToolButton:pressed { background: #0c0c0c; }\n"
        )
        self.btn_tempo_range.clicked.connect(lambda: self._deck_action(deck.ACT_TEMPO_RANGE))

        self.v_tempo = QtWidgets.QLabel(_tempo_text(1.0))
        f = self.v_tempo.font()
        f.setPointSize(9)
        f.setBold(True)
        self.v_tempo.setFont(f)
        self.v_tempo.setAlignment(QtCore.Qt.AlignHCenter)
        # fixed width: the widest reading, so the fader column never shifts
        widest = max((_tempo_text(0.0), _tempo_text(0.84, 16)), key=QtGui.QFontMetrics(f).horizontalAdvance)
        self.v_tempo.setFixedWidth(QtGui.QFontMetrics(f).horizontalAdvance(widest) + 4)

        self.tempo_slider = _OneStepWheelSlider(QtCore.Qt.Vertical)
        self.tempo_slider.setToolTip(tr("Tempo (WIDE: 0%-200%, 0 stops)"))
        self._set_tempo_fader(self._tempo_factor, self._tempo_range)

        tempo_v.addWidget(self.btn_tempo_range, 0, QtCore.Qt.AlignHCenter)
        tempo_v.addWidget(self.tempo_slider, 0, QtCore.Qt.AlignHCenter)
        tempo_v.addWidget(self.v_tempo, 0, QtCore.Qt.AlignHCenter)

        _match_track_info_height(tempo_box)
        # a request: the fader shows the tempo the player then reports
        self.tempo_slider.valueChanged.connect(
            lambda value: self.bus.sig_tempo_requested.emit(1.0 + value * TEMPO_STEP[self._tempo_range] / 100.0))
        return tempo_box

    def _build_transport_buttons(self) -> None:
        """The deck's buttons, after a DJ deck: square MT (time stretch) / RST (tempo reset) in a row above the record, round stop / CUE / play left of it; wired."""
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
        # CUE as rekordbox's (the player's deck: audioengine/native/src/deck.rs): pressed and let go are both
        # sent; lit as the player reports its lamp (blinking: a new cue point can be set)
        # the word as an icon (transport_cue / _lit: dark on the lit yellow), larger than the others' glyphs
        self.btn_cue = _mk_round_btn("cue", CUE_COLOR)
        self.btn_cue.setIconSize(QtCore.QSize(int(ROUND_D * 0.76), int(ROUND_D * 0.76)))
        self.btn_cue.setStyleSheet(
            self.btn_cue.styleSheet() + f"QToolButton[lit=true] {{ background: {CUE_COLOR}; }}\n")
        self.btn_cue.setProperty("lit", False)
        self.btn_cue.setToolTip(tr(
            "CUE: stopped, sets the cue point here / held on the cue point, plays until let go; playing, back to "
            "the cue point and stop"))

        self.btn_tempo_reset = _mk_square_btn("RST")
        self.btn_tempo_reset.setToolTip(tr("Tempo reset (100%)"))
        self.btn_mt = _mk_square_btn("MT", checkable=True)
        self.btn_mt.setToolTip(tr("Master tempo: time stretch keeps the pitch"))
        self.btn_mt.setChecked(self._timestretch_enabled())

        for btn in (self.btn_mt, self.btn_tempo_reset):
            self.track_info.record_buttons_slot.addWidget(btn, 0, QtCore.Qt.AlignVCenter)
        for btn in (self.btn_stop, self.btn_cue, self.btn_play):
            self.track_info.transport_slot.addWidget(btn, 0, QtCore.Qt.AlignHCenter)

        # Wire transport buttons
        self.btn_tempo_reset.clicked.connect(self._on_click_tempo_reset)
        self.btn_mt.toggled.connect(self.bus.sig_timestretch_toggle_requested.emit)
        self.btn_stop.clicked.connect(self._on_click_stop)
        self.btn_play.clicked.connect(self._on_click_play)
        self.btn_cue.pressed.connect(lambda: self._cue(True))
        self.btn_cue.released.connect(lambda: self._cue(False))


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
            self.add_view("TempoLabelView")   # the tempo at the playhead, right of it (accel. / rit.)
        if view_cfg.display_keystrip:
            self.add_view("KeyStripView")
        if view_cfg.display_JumpCUE:
            self.add_view("JumpCUEView")
        if getattr(view_cfg, "display_phrase", True):
            self.add_view("PhraseView")
        self.add_view("CUEPointView")
        self.add_view("DeckCueView")   # the deck's cue point (the CUE button's): above the phrase strip
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

    # Transport handlers: requests to the player's deck, which combines them with a MIDI controller's; the
    # buttons and faders show what it then reports (_on_deck_state, _on_playback_state_changed)
    def _deck_action(self, code: int, pressed: bool = True) -> None:
        self.bus.sig_deck_action.emit(int(code), bool(pressed))

    def _on_click_tempo_reset(self):
        self._deck_action(deck.ACT_TEMPO_RESET)

    def _on_click_stop(self):
        self._deck_action(deck.ACT_STOP)

    def _on_click_play(self):
        if self._external_sync_enabled:
            QtWidgets.QMessageBox.information(
                self,
                tr("External Sync"),
                tr("Playback is disabled while External Sync is enabled."),
            )
            return
        self._deck_action(deck.ACT_PLAY_PAUSE)

    def _cue(self, pressed: bool) -> None:
        """The CUE button (or its key) pressed or let go."""
        if pressed and not self._can_use_transport():
            return
        self._deck_action(deck.ACT_CUE, pressed)

    def _set_tempo_fader(self, factor: float, tempo_range: int) -> None:
        """The tempo fader at ``factor`` in ``tempo_range``'s steps (no request sent)."""
        step = TEMPO_STEP.get(tempo_range, 1.0)
        n = int(round(tempo_range / step))
        self.tempo_slider.blockSignals(True)
        self.tempo_slider.setRange(-n, n)
        self.tempo_slider.setValue(int(round((factor - 1.0) * 100.0 / step)))
        self.tempo_slider.blockSignals(False)
        self.v_tempo.setText(_tempo_text(factor, tempo_range))

    def _on_deck_state(self, report) -> None:
        """The player's deck as it is (DeckReport): the faders, the tempo range, its channel's knobs, the CUE lamp."""
        factor, tempo_range = float(report.factor), int(report.tempo_range)
        if (factor, tempo_range) != (self._tempo_factor, self._tempo_range):
            self._tempo_factor, self._tempo_range = factor, tempo_range
            if not self.tempo_slider.isSliderDown():   # the hand on it: it shows the hand's position
                self._set_tempo_fader(factor, tempo_range)
            else:
                self.v_tempo.setText(_tempo_text(factor, tempo_range))
            self.btn_tempo_range.setText(_range_text(tempo_range))
            self._update_bpm_display()
            self._on_time(self.current_time)   # the key follows the tempo (varispeed)
        if report.volume != self._volume_position:
            self._volume_position = float(report.volume)
            if not self.vol_slider.isSliderDown():
                self.vol_slider.blockSignals(True)
                self.vol_slider.setValue(int(round(self._volume_position * 100.0)))
                self.vol_slider.blockSignals(False)
            self._apply_volume_report()
        self.track_info.set_knobs(report.knobs)
        lit = bool(report.lamps & 2)
        if lit != self._cue_lit:
            self._cue_lit = lit
            self.btn_cue.setProperty("lit", lit)
            self.btn_cue.setIcon(_transport_icon("cue_lit" if lit else "cue"))
            self.btn_cue.style().unpolish(self.btn_cue)
            self.btn_cue.style().polish(self.btn_cue)

    # Keyboard shortcuts (ui.shortcuts)
    def shortcut_handlers(self) -> dict:
        """Action id -> handler of every shortcut: the deck's here, the editor's and the library's from them."""
        return {
            "playback.play_pause": lambda: self._on_click_play() if self.btn_play.isEnabled() else None,
            # pressed and let go (ui.shortcuts: a momentary action)
            "playback.cue": self._cue,
            "playback.stop": lambda: self._on_click_stop() if self.btn_stop.isEnabled() else None,
            **{action: (lambda code=code: self._deck_action(code) if self._can_use_transport() else None)
               for action, code in (("playback.beat_back", deck.ACT_BEAT_BACK),
                                    ("playback.beat_forward", deck.ACT_BEAT_FORWARD),
                                    ("playback.bar_back", deck.ACT_BAR_BACK),
                                    ("playback.bar_forward", deck.ACT_BAR_FORWARD))},
            "playback.tempo_down": lambda: self._deck_action(deck.ACT_TEMPO_DOWN),
            "playback.tempo_up": lambda: self._deck_action(deck.ACT_TEMPO_UP),
            "playback.tempo_reset": self._on_click_tempo_reset,
            "playback.tempo_range": lambda: self._deck_action(deck.ACT_TEMPO_RANGE),
            "playback.master_tempo": self.btn_mt.toggle,
            "playback.zoom_in": lambda: self._zoom(-1),
            "playback.zoom_out": lambda: self._zoom(1),
            "playback.unload": lambda: (self.bus.sig_unload_track_requested.emit()
                                        if float(self.model.duration_sec or 0.0) > 0.0 else None),
            "library.leave_search": self._leave_text_field,
            **self.track_edit.shortcut_handlers(),
            **self.lib.shortcut_handlers(),
        }

    def _leave_text_field(self) -> None:
        """Out of the search field, the sources panel or the track panel (to the library's tracks), or out of
        another text / value field here; a text is kept: the keys work the shortcuts again."""
        focus = QtWidgets.QApplication.focusWidget()
        if focus is None or not self.isAncestorOf(focus):
            return
        if focus is self.lib.e_search or focus is self.lib.source_tree or self.lib.track_panel_has_focus():
            self.lib.focus_tracks()
        elif isinstance(focus, (QtWidgets.QLineEdit, QtWidgets.QAbstractSpinBox)):
            self.setFocus(QtCore.Qt.ShortcutFocusReason)

    def _zoom(self, steps: int) -> None:
        """The main view ``steps`` widths narrower (< 0) or wider, as the wheel."""
        window = float(self.tl.window_sec)
        new_window = step_window(window, steps)
        if new_window != window:
            self.bus.sig_window_changed.emit(new_window)

    def _apply_volume_report(self) -> None:
        """The volume the player reports: its reading, and the peak meter's gain from it."""
        if self._volume_position is None:
            return
        value = int(round(self._volume_position * 100.0))
        self.v_volume.setText(f"{value}%")
        trim_dbfs = float(self.cfg.deckconfig.volume_trim_dbfs)
        linear_volume = slider_percent_to_linear(value, trim_dbfs)
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
        self._x_range_window = self.tl.window_sec
        self._update_pixel_sec()

    def _update_pixel_sec(self, *_args) -> None:
        """The main view's track seconds per device pixel, for the timeline's
        whole-pixel offset (TimelineCoordinator.left_edge)."""
        width = float(self.p.getViewBox().width()) * float(self.gl.devicePixelRatioF())
        self.tl.set_pixel_sec(self.tl.window_sec / width if width > 0.0 else 0.0)

    def _on_time(self, t: float):
        self.current_time = t
        self.track_info.update_record_pix(t)
        if hasattr(self, "track_edit"):
            self.track_edit.update_time(t)
        # the main view's range is set when the window changes (auto-range is off): here only
        # before the first window change
        if getattr(self, "_x_range_window", None) != self.tl.window_sec:
            self._on_window(self.tl.window_sec)

        self._update_bpm_display(t)
        
        active_key = self._key_from_segments(self._key_segments, t)
        speed = self._tempo_factor
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
        beats = self.model.beats
        segments = self.model.tempo_segments
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
        rows = self.model.phrases
        self._phrase_starts = np.asarray([float(r.start) for r in rows], dtype=float)
        self._phrase_ends = np.asarray([float(r.end) for r in rows], dtype=float)
        self._phrase_names = number_phrase_labels(rows)
        self._phrase_colors = [phrase_color(r.label) for r in rows]
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
        """A track (or its analysis) loaded: the overview, the phrase / BPM / key lines read it (the editor follows
        the model itself)."""
        self.ov.set_data()
        self._on_phrases_updated()
        self._rebuild_beat_positions()
        self._set_tempo_segments(self.model.tempo_segments)
        self._on_key_segments_updated()
        if self._key_segments is None or self._key_segments.size == 0:
            self.track_info.set_key(None)  # e.g. unloaded

    def _on_beatgrid_edited(self):
        self._set_tempo_segments(self.model.tempo_segments)
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
        factor = self._tempo_factor
        if not np.isfinite(bpm):
            self.track_info.set_bpm(None)
        else:
            self.track_info.set_bpm(bpm * factor, bpm, factor * 100.0 - 100.0, meter)

    def _on_properties(self):
        p = self.model.properties
        title_text = p.get("title") or tr("Not Loaded") if isinstance(p, dict) else tr("Not Loaded")
        artist_text = p.get("artist") or "" if isinstance(p, dict) else ""
        self.track_info.set_title(title_text.strip())
        self.track_info.set_artist(artist_text.strip())
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
        for btn_name in ("btn_play", "btn_stop", "btn_cue"):
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

    # Files dragged over the deck
    def _deck_rect(self) -> QtCore.QRect:
        """The deck: the pane above the library."""
        return QtCore.QRect(0, 0, self.width(), max(0, self.lib.geometry().top() - 3))

    def drop_zone_at(self, pos: QtCore.QPoint) -> bool:
        """``pos`` (the pane's) is over the deck."""
        return self._deck_rect().contains(pos)

    def show_drop_zone(self, lit: bool) -> None:
        self._drop_zone.show_at(self._deck_rect(), lit)

    def hide_drop_zone(self) -> None:
        self._drop_zone.hide()

    def apply_playback_config(self, cfg: config) -> None:
        self.cfg = cfg
        self._apply_volume_report()   # the trim changes the peak meter's gain
        self._on_time(self.current_time)  # key display depends on use_timestretch
        self.btn_mt.blockSignals(True)
        self.btn_mt.setChecked(self._timestretch_enabled())
        self.btn_mt.blockSignals(False)
        self.track_edit.set_metronome_checked(self._metronome_enabled())

    def _metronome_enabled(self) -> bool:
        return bool(self.cfg.deckconfig.enable_metronome)

    def _timestretch_enabled(self) -> bool:
        return bool(self.cfg.deckconfig.use_timestretch)

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
        self._key_segments = self.model.key_segments
        if hasattr(self.ov, "update_key_strip"):
            self.ov.update_key_strip()

