"""Deck Monitor's window, in a process of its own (core.deck_monitor.controller starts it with ``run``): another DJ
program's decks 1 - 4 laid out as rekordbox's four decks: their main views stacked at the top, the decks under them
two by two. Each deck's main view and overview are Mixlyzer's own (the main view's plugins, views.base.REGISTRY, and
views.overview), on a bus, a model and a timeline of the deck's own: its library track's stored analysis, its audio
decoded here, its position as the deck plays it. The main views keep one span of time on screen: a deck played faster
shows more of its track (its tempo from the sender), so decks at one BPM show their beats alike, as rekordbox does;
the wheel over them zooms all four. A deck shows its album art, title and artist, BPM and tempo, key, time, whether
it is the master deck, and the phrase it is in and the next one (core.deck_monitor.readout). A track not in the
library shows what the sender names. Where the decks are received is set here, at the top (the app keeps it in
deck.json). Everything here (the OSC receiver, the decoding, the drawing) runs in this process: the app's GIL and GUI
thread are not shared."""

from __future__ import annotations

import ipaddress
import sys
from collections import OrderedDict
from dataclasses import dataclass
from queue import Empty
from typing import Any, Callable, Optional

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from analyzer.music.key_names import idx_to_labels
from core.analysis_library.stored import StoredAnalysis
from core.deck_monitor.decks import TrackInfo
from core.deck_monitor.readout import LABEL, SECTION, UNITS, TrackPhrases, bars_beats, bpm_at, key_at
from core.deck_monitor.receivers import OSC, PROTOCOL_NAMES, PRODJLINK, Where, interfaces
from core.deck_monitor.source import DeckSource, MonitorDeck
from core.deck_monitor.source_marks import SourceMarks, source_analysis
from core.library.records import TrackRow
from core.marks.phrases import PHRASE_LABELS, phrase_display_name
from core.settings.store import SettingsStore, current as current_settings
from core.track.model import DataModel
from core.workflows.event_bus import EventBus
from system.i18n import tr, tr_ctx
from ui.deck.deck_controls import Lamp, label as deck_label
from ui.deck.track_info_panel import (_ARTIST as INFO_ARTIST, _DIM as INFO_DIM, _DOWN as INFO_DOWN, _UP as INFO_UP,
                                      _VALUE as INFO_VALUE, MarqueeLabel)
from ui.theme import current_theme
from ui.widgets.elided_label import ElidedLabel
from ui.widgets.live_label import LiveLabel
from ui.widgets.record_disc import RecordDisc
from views.base import REGISTRY
from views.overview import OverviewWidget
from views.render.keystrip import key_segment_rows
from views.timeline import DEFAULT_WINDOW_SEC, TimelineCoordinator, step_window

RATE = 48000             # a track decoded at the engine's rate, its channels averaged as the main view draws them
MAX_FPS = 60             # redrawn at most this often (the view's fps setting), on a precise timer
KEPT_TRACKS = 6          # tracks kept decoded, with their analysis (the decks' and the ones just before)
COMMAND_MS, STATUS_MS = 50, 500
MAIN_VIEW_MIN_H = 48     # a deck's main view at the least
OVERVIEW_MIN_H = 64      # a deck's overview at the least (the main views and the overviews share the height left)
RECORD_D = 83            # the turning record, as large as the main window's (TrackInfoPanel.RECORD_D)
COVER = 52               # a deck's album art beside its title (rekordbox's)
DECKS = 4
# the main views' plugins, as the app's pane installs them (views.base.REGISTRY), without the app's own deck's cue
# and the editor's change flashes
VIEWS_BY_SETTING = (("display_waveform", ("WaveformView",)), ("display_beatgrid", ("BeatgridView", "TempoLabelView")),
                    ("display_keystrip", ("KeyStripView",)), ("display_JumpCUE", ("JumpCUEView",)),
                    ("display_phrase", ("PhraseView",)))
ALWAYS_VIEWS = ("CUEPointView", "PlayHead")
MINUS = "−"          # as wide as "+"
SHOWN_PROTOCOLS = (OSC, PRODJLINK)        # the connection bar's choices, in this order
_STORE: Optional["ReadOnlyStore"] = None    # the settings as this process was given them (run installs it)


def local_address(text: str) -> bool:
    """Whether the address only takes messages from this computer (a loopback address, localhost)."""
    text = text.strip()
    if text in ("", "localhost"):
        return True
    try:
        return ipaddress.ip_address(text).is_loopback
    except ValueError:
        return False


class ReadOnlyStore(SettingsStore):
    """The app's settings as this process was given them (core.settings.store.current() reads them here); nothing
    is written: the files are the app's."""

    def save(self, new) -> None:
        pass

    def deck_changed(self) -> None:
        pass

    def flush(self) -> None:
        pass

    def replace(self, cfg) -> None:
        self._cfg = cfg


@dataclass
class TrackData:
    """A library track as the decks show it: its stored analysis (asked for once), its audio and cover (read once)."""
    uid: str
    path: str
    analysis: Optional[StoredAnalysis] = None
    asked: bool = False                      # its analysis asked for and not answered
    unread: bool = False                     # none could be read (asked again once the library changes)
    stale: bool = False                      # the library changed since it was read: asked again, shown meanwhile
    pcm: Optional[np.ndarray] = None
    art: Optional[QtGui.QImage] = None
    decoding: bool = False
    failed: bool = False                     # its audio could not be decoded


class _Signals(QtCore.QObject):
    decoded = QtCore.Signal(str, object, object)      # uid, its PCM (None: not decoded), its cover (None: none)


class _Decode(QtCore.QRunnable):
    def __init__(self, signals: _Signals, uid: str, path: str) -> None:
        super().__init__()
        self.signals, self.uid, self.path = signals, uid, path

    def run(self) -> None:
        pcm = art = None
        try:
            from analyzer.audio.decoder import decode_to_memmap
            stereo = decode_to_memmap(self.path, RATE, 2)
            pcm = np.ascontiguousarray(stereo.mean(axis=1, dtype=np.float32))   # as the main view averages them
            del stereo
        except Exception as exc:
            print(f"[DeckMonitor] {self.path} could not be decoded: {exc}")
        try:
            from core.tags.reader import getAlbumArt
            art = getAlbumArt(self.path)
        except Exception:
            art = None
        self.signals.decoded.emit(self.uid, pcm, art)


class _ZoomViewBox(pg.ViewBox):
    """A deck's main view box: it shows another program's deck, so nothing is dragged or clicked; the wheel zooms
    every deck's view (up: narrower)."""

    def __init__(self, on_wheel: Callable[[int], None]) -> None:
        super().__init__(enableMouse=False, enableMenu=False)
        self._on_wheel = on_wheel
        self._delta = 0

    def wheelEvent(self, ev, axis=None) -> None:
        self._delta += int(ev.delta())
        steps = int(self._delta / 120)
        if steps:
            self._delta -= steps * 120
            self._on_wheel(-steps)
        ev.accept()

    def mouseDragEvent(self, ev, axis=None) -> None:
        ev.ignore()

    def mouseClickEvent(self, ev) -> None:
        ev.ignore()


class DeckViews(QtCore.QObject):
    """A deck's main view and overview, the app's own, on a bus, a model and a timeline of the deck's: ``show`` its
    track (loaded into the model as the app loads one), ``set_time`` / ``set_window`` / ``set_played`` as it plays."""

    def __init__(self, on_wheel: Callable[[int], None]) -> None:
        super().__init__()
        self.bus = EventBus()
        self.tl = TimelineCoordinator(self.bus)       # first: it takes the time before the views draw it
        self.model = DataModel()
        self.main = pg.GraphicsLayoutWidget()
        self.main.ci.layout.setContentsMargins(0, 0, 0, 0)
        self.plot = self.main.addPlot(row=0, col=0, viewBox=_ZoomViewBox(on_wheel))
        self.plot.hideAxis("left")
        self.plot.hideAxis("bottom")
        self.plot.hideButtons()
        self.plot.setMenuEnabled(False)
        self.main.setMinimumHeight(MAIN_VIEW_MIN_H)
        self.main.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.overview = OverviewWidget(self.bus, tl=self.tl, model=self.model)
        self.overview.setMinimumHeight(OVERVIEW_MIN_H)
        self.overview.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.overview.canvas.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)    # it seeks nothing
        self.model.sig_loaded.connect(self.overview.set_data)
        self.model.sig_loaded.connect(self._read_marks)
        self.plot.getViewBox().sigResized.connect(self._pixel_sec)
        self.plugins: list = []
        self.uid: Optional[str] = None
        self.analysis: Optional[StoredAnalysis] = None
        self.pcm = None
        self.time: Optional[float] = None
        self.window: Optional[float] = None
        self.played: Optional[float] = None
        self.phrases = TrackPhrases((), None, None)
        self.key_rows: list = []

    def install(self, cfg) -> None:
        """The main view's plugins as Settings > View names them (the app's pane's)."""
        self.clear()
        view = cfg.viewconfig
        names = [n for setting, group in VIEWS_BY_SETTING if getattr(view, setting, True) for n in group]
        for name in (*names, *ALWAYS_VIEWS):
            cls = REGISTRY.get(name)
            if cls is None:
                continue
            plugin = cls(self.bus, self.model, self.tl)
            plugin.attach(self.plot)
            try:
                plugin.render_initial()
            except Exception:
                pass
            self.plugins.append(plugin)
        self.overview.set_phrase_visible(getattr(view, "display_phrase", True))
        self.plot.setYRange(0, 1.0, padding=0.05)
        self.played = None                            # the new views take the track as played

    def clear(self) -> None:
        for plugin in self.plugins:
            try:
                plugin.detach()
            except Exception:
                pass
        self.plugins = []

    def show(self, data: Optional[TrackData], row) -> None:
        """The deck's library track (None: none): loaded as the app loads one, its analysis and audio as they
        come."""
        if data is None:
            if self.uid is not None:
                self.uid, self.analysis, self.pcm = None, None, None
                self.model.load(StoredAnalysis(), {}, 0.0)
            return
        analysis = data.analysis
        if data.uid != self.uid:
            self.uid, self.analysis, self.pcm = data.uid, analysis, None
            self.model.load(analysis or StoredAnalysis(), row.to_meta(), _duration(analysis, row))
        elif analysis is not None and analysis is not self.analysis:
            self.analysis = analysis
            self.model.load(analysis, row.to_meta(), _duration(analysis, row), keep_audio=True)
        if data.pcm is not None and data.pcm is not self.pcm:
            self.pcm = data.pcm
            self.model.set_predecoded_audio(data.pcm, RATE)

    def set_time(self, t: Optional[float]) -> None:
        if t is not None and t != self.time:
            self.time = float(t)
            self.bus.sig_time_changed.emit(self.time)

    def set_window(self, window_sec: float) -> None:
        if window_sec != self.window:
            self.window = window_sec
            self.bus.sig_window_changed.emit(window_sec)
            self.plot.setXRange(0.0, self.tl.window_sec, padding=0)
            self._pixel_sec()

    def set_played(self, speed: float) -> None:
        """The track as played (its tempo; its key as the sender gives none): the views show it so where Settings >
        View says (bus.sig_played_changed)."""
        if speed != self.played:
            self.played = speed
            self.bus.sig_played_changed.emit(float(speed), 0)

    def _pixel_sec(self, *_args) -> None:
        width = float(self.plot.getViewBox().width()) * float(self.main.devicePixelRatioF())
        self.tl.set_pixel_sec(self.tl.window_sec / width if width > 0.0 else 0.0)

    def _read_marks(self) -> None:
        self.phrases = TrackPhrases(self.model.phrases, self.model.beats, self.model.tempo_segments)
        self.key_rows = key_segment_rows(self.model.key_segments)


def _duration(analysis: Optional[StoredAnalysis], row) -> float:
    if analysis is not None and float(analysis.duration_sec or 0.0) > 0.0:
        return float(analysis.duration_sec)
    return float(getattr(row, "duration", 0.0) or 0.0)


def _label(point_size: int = 0, bold: bool = False, cls=QtWidgets.QLabel, text: str = ""):
    label = cls(text) if cls is not ElidedLabel else cls()
    font = label.font()
    if point_size:
        font.setPointSize(point_size)
    font.setBold(bold)
    label.setFont(font)
    return label


def deck_clock(sec: Optional[float]) -> str:
    """A time as a CDJ / rekordbox deck shows it: 02:46.4; before the track's start −00:01.2."""
    sec = float(sec or 0.0)
    if sec <= -0.05:                                   # shown as 0.1 or more
        return MINUS + deck_clock(-sec)
    minutes, rest = divmod(max(0.0, sec), 60.0)
    return f"{int(minutes):02d}:{rest:04.1f}"


def tempo_text(change: float) -> str:
    """The tempo as a CDJ shows it: +1.60%, −1.63%, 0.00% (U+2212: as wide as "+")."""
    return "0.00%" if abs(change) < 0.005 else f"{change:+.2f}%".replace("-", MINUS)


class DeckPanel(QtWidgets.QFrame):
    """One deck laid out as a Pioneer (rekordbox / CDJ) deck, in the app's look (its palette and fonts, its marquee
    titles, lamps and turning record): the header with the number, album art, title and artist, the time left
    (large) and played at its right and MASTER; the phrase now and next; the overview taking the rest. On the side
    facing the window's middle (``inner_right``: right for decks 1 and 3, left for 2 and 4) the turning record and
    the BPM (large), the tempo and the key: the record above them on the top decks (``top``: 1, 2), under them on the
    bottom ones (3, 4), so the four decks' BPMs and keys meet at the window's middle."""

    def __init__(self, deck: int, overview: QtWidgets.QWidget, inner_right: bool, top: bool, parent=None) -> None:
        super().__init__(parent)
        theme = current_theme()
        self.deck = deck
        self.setObjectName("deckPanel")
        self.setStyleSheet(f"QFrame#deckPanel {{ background: {theme.base}; border: 1px solid {theme.border};"
                           f" border-radius: 3px; }}")
        self.body = body = QtWidgets.QHBoxLayout(self)
        body.setContentsMargins(10, 8, 10, 8)
        body.setSpacing(12)

        main = QtWidgets.QVBoxLayout()                    # the header, the phrases, the overview
        main.setSpacing(4)
        head = QtWidgets.QHBoxLayout()
        head.setSpacing(8)
        self.number = _label(11, True, text=str(deck))
        self.cover = QtWidgets.QLabel()                   # the album art, as the track info frames it
        self.cover.setFixedSize(COVER, COVER)
        self.cover.setAlignment(QtCore.Qt.AlignCenter)
        self.cover.setStyleSheet("background:#1c1c1c; border:1px solid #333; border-radius:6px; padding:2px;")
        names = QtWidgets.QVBoxLayout()
        names.setSpacing(0)
        self.title = MarqueeLabel()
        self.artist = MarqueeLabel()
        for lbl, size, bold, color in ((self.title, 13, True, INFO_VALUE), (self.artist, 11, False, INFO_ARTIST)):
            f = lbl.font()
            f.setPointSize(size)
            f.setBold(bold)
            lbl.setFont(f)
            lbl.setStyleSheet(f"color: {color};")
            lbl.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
            names.addWidget(lbl)
        self.found = _label(max(7, self.font().pointSize() - 2))
        self.found.setStyleSheet(f"color: {theme.text_dim};")
        names.addWidget(self.found)
        times = QtWidgets.QVBoxLayout()                   # the time left, large, and played (rekordbox's)
        times.setSpacing(0)
        self.remaining = _label(16, True, cls=LiveLabel)
        self.remaining.setFixedWidth(self.remaining.fontMetrics().horizontalAdvance(MINUS + "88:88.8") + 4)
        self.elapsed = _label(11, False, cls=LiveLabel)
        self.elapsed.setFixedWidth(self.remaining.width())
        self.elapsed.setStyleSheet(f"color: {INFO_DIM};")
        for w in (self.remaining, self.elapsed):
            w.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            times.addWidget(w)
        self.master = Lamp("MASTER")
        self.master.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)   # it shows the sender's master deck
        self.master.setFocusPolicy(QtCore.Qt.NoFocus)
        head.addWidget(self.number, 0, QtCore.Qt.AlignTop)
        head.addWidget(self.cover, 0, QtCore.Qt.AlignTop)
        head.addLayout(names, 1)
        head.addLayout(times)
        head.addWidget(self.master, 0, QtCore.Qt.AlignTop)
        main.addLayout(head)
        self.phrase_line = PhraseLine()
        main.addWidget(self.phrase_line)
        main.addWidget(overview, 1)

        inner = QtWidgets.QVBoxLayout()                   # facing the window's middle: the record, BPM, tempo, key
        inner.setSpacing(6)
        center = QtCore.Qt.AlignHCenter                   # one axis: the record over the BPM, the tempo and key
        self.record = RecordDisc(RECORD_D, lambda: getattr(current_settings().viewconfig, "record_img_path", None))
        reading = QtWidgets.QVBoxLayout()
        reading.setSpacing(0)
        self.bpm = _label(24, True, cls=LiveLabel)
        self.bpm.setFixedWidth(self.bpm.fontMetrics().horizontalAdvance("888.88") + 6)
        self.bpm.setStyleSheet(f"color: {INFO_VALUE};")
        under = QtWidgets.QHBoxLayout()
        under.setSpacing(12)
        self.pitch = _label(11, True, cls=LiveLabel)
        self.pitch.setFixedWidth(self.pitch.fontMetrics().horizontalAdvance(MINUS + "88.88%") + 4)
        self.key = _label(13, True, cls=LiveLabel)
        self.key.setFixedWidth(self.key.fontMetrics().horizontalAdvance("12A") + 4)
        self.key.setStyleSheet(f"color: {INFO_VALUE};")
        self.pitch.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.key.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        under.addStretch(1)
        under.addWidget(self.pitch)
        under.addWidget(self.key)
        under.addStretch(1)
        self.bpm.setAlignment(QtCore.Qt.AlignCenter)
        reading.addWidget(self.bpm, 0, center)
        reading.addLayout(under)
        align = center
        if top:
            inner.addWidget(self.record, 0, align)
            inner.addStretch(1)
            inner.addLayout(reading)
        else:
            inner.addLayout(reading)
            inner.addStretch(1)
            inner.addWidget(self.record, 0, align)
        self.inner = QtWidgets.QWidget()                  # one width on every deck: the four line up
        self.inner.setLayout(inner)
        inner.setContentsMargins(0, 0, 0, 0)
        self.inner.setFixedWidth(max(RECORD_D, self.bpm.width()) + 8)
        if inner_right:
            body.addLayout(main, 1)
            body.addWidget(self.inner)
        else:
            body.addWidget(self.inner)
            body.addLayout(main, 1)
        self._shown: Optional[tuple] = None
        self._art_of: Any = False
        self._pitch_sign: Optional[int] = None
        self.show_deck(MonitorDeck(deck, None, True, None, None, 0.0), None, None)

    def show_deck(self, d: MonitorDeck, data: Optional[TrackData], views: Optional[DeckViews]) -> None:
        theme = current_theme()
        state = d.state
        if not d.shown:
            title = tr("Not heard") if state is None or d.stale else tr("No track")
            names = (title, "", "")
        elif d.row is not None:
            names = (d.row.title or d.row.path, d.row.artist, "")
        else:
            track = state.track
            names = (track.title or track.path or tr("(no name)"), track.artist, tr("Not in the library"))
        master = bool(d.shown and state is not None and state.master)
        shown = (d.shown, names, master)
        if shown != self._shown:           # the names and the look change with the track or the state only
            self._shown = shown
            self.title.set_marquee_text(names[0])
            self.title.setStyleSheet(f"color: {INFO_VALUE if d.shown else theme.text_disabled};")
            self.artist.set_marquee_text(names[1])
            self.found.setText(names[2])
            self.number.setStyleSheet(f"color: {theme.text if d.shown else theme.text_disabled};")
            self.master.set_lit(master)
            self.master.setVisible(d.shown)
        self._show_art(data if d.shown else None)
        if not d.shown:
            for label in (self.bpm, self.pitch, self.key, self.remaining, self.elapsed):
                label.setText("")
            self._show_phrases(None, None)
            return
        position = d.position_sec
        if position is not None:
            self.record.set_time(position)
        analysed = data is not None and data.analysis is not None and views is not None
        pitch = state.pitch
        tempo = bpm_at(views.model.tempo_segments, position) if analysed and position is not None else None
        if tempo is not None:
            bpm = tempo * (pitch if pitch is not None else 1.0)     # the sender's tempo, never one guessed from
                                                                     # the motion (it shook)
        else:
            bpm = state.bpm if state.bpm > 0.0 else None
        self.bpm.setText(f"{bpm:.2f}" if bpm is not None else "--")
        change = (pitch - 1.0) * 100.0 if pitch is not None else None
        self.pitch.setText(tempo_text(change) if change is not None else "")
        sign = 0 if change is None or abs(change) < 0.005 else (1 if change > 0 else -1)
        if sign != self._pitch_sign:                  # coloured as the track info colours a change
            self._pitch_sign = sign
            self.pitch.setStyleSheet(f"color: {INFO_UP if sign > 0 else INFO_DOWN if sign < 0 else INFO_DIM};")
        # before the track's start: the key it starts in
        key = key_at(views.key_rows, max(0.0, position)) if analysed and position is not None else None
        self.key.setText(idx_to_labels(key)[0] if key is not None else "--")
        duration = (float(d.row.duration) if d.row is not None and d.row.duration
                    else float(state.track.duration_sec or 0.0))
        left = duration - position if duration > 0.0 and position is not None else None
        self.remaining.setText(MINUS + deck_clock(left) if left is not None else "")
        self.elapsed.setText(deck_clock(position) if position is not None else "")
        self._show_phrases(views if analysed else None, position)

    def _show_art(self, data: Optional[TrackData]) -> None:
        art = data.art if data is not None else None
        if art is self._art_of:
            return
        self._art_of = art
        usable = art is not None and not art.isNull()
        self.record.set_album_art(art if usable else None)
        if not usable:
            self.cover.clear()
            return
        size = self.cover.contentsRect().size()
        if not size.isValid() or size.isEmpty():
            size = QtCore.QSize(COVER - 6, COVER - 6)
        self.cover.setPixmap(QtGui.QPixmap.fromImage(art).scaled(size, QtCore.Qt.KeepAspectRatio,
                                                                 QtCore.Qt.SmoothTransformation))

    def _show_phrases(self, views: Optional[DeckViews], position: Optional[float]) -> None:
        now = views.phrases.at(position, self.phrase_line.unit) if views is not None and position is not None \
            else None
        self.phrase_line.show_phrases(now, position)


class _Swatch(QtWidgets.QLabel):
    """A phrase's colour, set only when it changes."""

    def __init__(self) -> None:
        super().__init__()
        self.setFixedSize(10, 10)
        self._color: Optional[tuple] = None

    def set_color(self, color: Optional[tuple]) -> None:
        if color == self._color:
            return
        self._color = color
        self.setStyleSheet("" if color is None else
                           f"background: rgb({color[0]},{color[1]},{color[2]}); border-radius: 2px;")


class PhraseLine(QtWidgets.QWidget):
    """The phrase now and the next, in cells set apart by rules, in the order a DJ reads them: NOW (the one playing,
    its length in bars), NEXT IN (bars.beats and the time until the next), NEXT (the next one, its length); each
    cell's name small over its value, as the toolbar names its blocks, the unit (section or label) in NOW's and
    NEXT's. Every value has a cell of fixed width, sized for the widest value it can take (the track info's rule):
    a new phrase, count or time moves nothing on the line. Clicked: ``clicked`` (the window switches every deck to
    the other unit)."""

    clicked = QtCore.Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        theme = current_theme()
        self.unit = SECTION
        self._unit_texts = {SECTION: tr_ctx("phrase unit", "Section"), LABEL: tr_ctx("phrase unit", "Label")}
        grid = QtWidgets.QGridLayout(self)
        grid.setContentsMargins(0, 2, 0, 2)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(0)
        self.now_swatch, self.next_swatch = _Swatch(), _Swatch()
        self.now_name = _label(11, True, cls=LiveLabel)
        self.next_name = _label(11, True, cls=LiveLabel)
        self.now_bars = _label(11, True, cls=LiveLabel)
        self.next_bars = _label(11, True, cls=LiveLabel)
        self.left = _label(13, True, cls=LiveLabel)
        self.left_time = _label(10, False, cls=LiveLabel)
        self.left_time.setStyleSheet(f"color: {INFO_DIM};")
        for w in (self.now_name, self.next_name, self.now_bars, self.next_bars, self.left):
            w.setStyleSheet(f"color: {INFO_VALUE};")
        widest_name = max((phrase_display_name(label) + "10" for label in PHRASE_LABELS),
                          key=self.now_name.fontMetrics().horizontalAdvance)
        for w, widest in ((self.now_name, widest_name), (self.next_name, widest_name), (self.now_bars, "888"),
                          (self.next_bars, "888"), (self.left, "888.8"), (self.left_time, "88:88.8")):
            w.setFixedWidth(w.fontMetrics().horizontalAdvance(widest) + 2)
        for w in (self.now_bars, self.next_bars, self.left):
            w.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        def bars_unit() -> QtWidgets.QLabel:              # the unit after a count: a DJ's word, untranslated
            unit = _label(8, False, text="Bars")
            unit.setFixedWidth(unit.fontMetrics().horizontalAdvance("Bars") + 2)
            unit.setStyleSheet(f"color: {INFO_DIM};")
            return unit

        def cell(*widgets) -> QtWidgets.QHBoxLayout:
            row = QtWidgets.QHBoxLayout()
            row.setSpacing(5)
            for w in widgets:
                row.addWidget(w, 0, QtCore.Qt.AlignBottom if isinstance(w, QtWidgets.QLabel) and
                              not isinstance(w, _Swatch) else QtCore.Qt.AlignVCenter)
            row.addStretch(1)
            return row

        def rule() -> QtWidgets.QFrame:
            line = QtWidgets.QFrame()
            line.setFixedWidth(1)
            line.setStyleSheet(f"background: {theme.border};")
            return line

        self.units = [bars_unit() for _ in range(3)]     # NOW's, NEXT IN's, NEXT's: shown with a count only
        self._units_shown: Optional[tuple] = None
        self.now_caption, self.left_caption, self.next_caption = (deck_label(""), deck_label(tr("Next in")),
                                                                  deck_label(""))
        for caption, text in ((self.now_caption, tr("Now · {unit}")), (self.next_caption, tr("Next · {unit}"))):
            caption.setFixedWidth(max(caption.fontMetrics().horizontalAdvance(text.format(unit=u))
                                      for u in self._unit_texts.values()) + 2)    # either unit: the same room
        cells = [(self.now_caption, cell(self.now_swatch, self.now_name, self.now_bars, self.units[0])),
                 (self.left_caption, cell(self.left, self.units[1], self.left_time)),
                 (self.next_caption, cell(self.next_swatch, self.next_name, self.next_bars, self.units[2]))]
        column = 0
        for i, (caption, values) in enumerate(cells):
            if i:
                grid.addWidget(rule(), 0, column, 2, 1)
                column += 1
            grid.addWidget(caption, 0, column)
            grid.addLayout(values, 1, column)
            column += 1
        grid.setColumnStretch(column, 1)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setToolTip(tr("Click: the phrases by section or by label, on every deck."))
        self.set_unit(SECTION)

    def set_unit(self, unit: str) -> None:
        self.unit = unit if unit in UNITS else SECTION
        name = self._unit_texts[self.unit]
        self.now_caption.setText(tr("Now · {unit}").format(unit=name))
        self.next_caption.setText(tr("Next · {unit}").format(unit=name))

    def mousePressEvent(self, event) -> None:
        if event.button() == QtCore.Qt.LeftButton:
            self.clicked.emit()
            event.accept()
            return
        super().mousePressEvent(event)

    def show_phrases(self, now, position: Optional[float] = None) -> None:
        """``now`` (core.deck_monitor.readout.PhraseNow, None: nothing to show) at ``position``."""
        current = now.now if now is not None else None
        nxt = now.next if now is not None else None
        left = now.left_beats if now is not None else None
        self.now_swatch.set_color(current.color if current is not None else None)
        self.now_name.setText(self._fitted(self.now_name, current.name) if current is not None else "--")
        self.now_bars.setText(str(current.bars) if current is not None else "")
        self.next_swatch.set_color(nxt.color if nxt is not None else None)
        self.next_name.setText(self._fitted(self.next_name, nxt.name) if nxt is not None else "--")
        self.next_bars.setText(str(nxt.bars) if nxt is not None else "")
        self.left.setText(bars_beats(left, now.meter) if nxt is not None and left is not None else "--")
        self.left_time.setText(deck_clock(max(0.0, nxt.start - position))
                               if nxt is not None and position is not None else "")
        units = (current is not None, nxt is not None and left is not None, nxt is not None)
        if units != self._units_shown:                # "Bars" after a count, none after "--" (its room kept)
            self._units_shown = units
            for unit, shown in zip(self.units, units):
                unit.setText("Bars" if shown else "")

    @staticmethod
    def _fitted(label: QtWidgets.QLabel, text: str) -> str:
        """``text`` cut to its cell (a long alias of the user's elided, the cell never wider)."""
        return label.fontMetrics().elidedText(text, QtCore.Qt.ElideRight, label.width())


class ConnectionBar(QtWidgets.QWidget):
    """Where the decks are received (set here: ``listen(where)`` asked of the window: OSC at an address and port, or a
    DJ network on the interface picked), the status."""

    def __init__(self, listen: Callable[[Where], None], parent=None) -> None:
        super().__init__(parent)
        self._listen = listen
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(6)
        self.protocol = QtWidgets.QComboBox()
        for protocol in SHOWN_PROTOCOLS:
            self.protocol.addItem("OSC Basic / Full" if protocol == OSC else PROTOCOL_NAMES[protocol], protocol)
        self.protocol.setToolTip(tr("How the other program's decks come: OSC from a sender, or a DJ network."))
        self.protocol.currentIndexChanged.connect(self._sync_protocol)
        self.interface = QtWidgets.QComboBox()
        self.interface.setToolTip(tr("The network the DJ equipment is on."))
        self.interface_label = QtWidgets.QLabel(tr("Network"))
        self.host = QtWidgets.QLineEdit()
        self.host.setPlaceholderText("127.0.0.1")
        self.host.setFixedWidth(self.host.fontMetrics().horizontalAdvance("255.255.255.255") + 24)
        self.host.setToolTip(
            tr("The address the messages are received on. 127.0.0.1 takes them from this computer only; the address "
               "of a network interface (or 0.0.0.0, all of them) takes them from that network too."))
        self.port = QtWidgets.QSpinBox()
        self.port.setRange(1, 65535)
        self.port.setToolTip(tr("The UDP port the messages are sent to."))
        self.apply = QtWidgets.QPushButton(tr("Listen"))
        self.apply.clicked.connect(self._apply)
        self.host.returnPressed.connect(self._apply)
        self.status = LiveLabel("")
        self.status.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        self.host_label, self.port_label = QtWidgets.QLabel(tr("Address")), QtWidgets.QLabel(tr("Port"))
        for w in (self.protocol, self.host_label, self.host, self.port_label, self.port, self.interface_label,
                  self.interface, self.apply):
            row.addWidget(w)
        row.addWidget(self.status, 1)
        root.addLayout(row)
        self.network = QtWidgets.QLabel(tr("Messages are taken from the network: any device on it can send them."))
        self.network.setStyleSheet("color: #d8a441;")
        root.addWidget(self.network)
        self.host.textChanged.connect(self._sync_network)

    def set_where(self, where: Where) -> None:
        """The fields as ``where`` says (the protocol, OSC's address and port, the network picked)."""
        self.host.setText(where.host)
        self.port.setValue(int(where.port))
        self._fill_interfaces(where.interface)
        index = self.protocol.findData(where.protocol)
        self.protocol.setCurrentIndex(index if index >= 0 else 0)
        self._sync_protocol()

    def where(self) -> Where:
        return Where(str(self.protocol.currentData() or OSC), self.host.text().strip() or "127.0.0.1",
                     int(self.port.value()), str(self.interface.currentData() or ""))

    def _fill_interfaces(self, chosen: str) -> None:
        """The networks with an IPv4 address (the first: whichever comes first, as kept when "" is chosen); the one
        chosen kept even when it is gone (said so)."""
        self.interface.blockSignals(True)
        self.interface.clear()
        self.interface.addItem(tr("First found"), "")
        for found in interfaces():
            self.interface.addItem(f"{found.name} — {found.address}", found.address)
        if chosen and self.interface.findData(chosen) < 0:
            self.interface.addItem(tr("{address} (not found)").format(address=chosen), chosen)
        self.interface.setCurrentIndex(max(0, self.interface.findData(chosen)))
        self.interface.blockSignals(False)

    def _sync_protocol(self, *_args) -> None:
        osc = self.protocol.currentData() == OSC
        for w in (self.host_label, self.host, self.port_label, self.port):
            w.setVisible(osc)
        for w in (self.interface_label, self.interface):
            w.setVisible(not osc)
        self._sync_network()

    def _sync_network(self) -> None:
        osc = self.protocol.currentData() == OSC
        self.network.setVisible(not osc or not local_address(self.host.text()))

    def _apply(self) -> None:
        self._listen(self.where())


class DeckMonitorWindow(QtWidgets.QWidget):
    """The window of decks 1 - 4. ``cfg`` the app's settings (where to listen first, how the views draw),
    ``messages`` the queue to the app (None: none)."""

    def __init__(self, cfg, messages=None, parent=None) -> None:
        super().__init__(parent, QtCore.Qt.Window)
        self.setWindowTitle(tr("Deck Monitor"))
        self.cfg, self.messages = cfg, messages
        self.source: Optional[DeckSource] = None
        self.error = ""                 # why it does not listen (shown as the status)
        self.track_rows: tuple = ()     # the library's tracks as last given
        self.tracks: OrderedDict[str, TrackData] = OrderedDict()
        # each deck's track the library does not hold, shown by the sender's own marks: (its track, its marks,
        # what the views show of it)
        self.sent: dict[int, tuple[TrackInfo, SourceMarks, TrackData, TrackRow]] = {}
        self.sent_count = 0             # those shown so far (each one's own uid: the views load it as a new track)
        self.gen = 0                    # the library's tracks as last given (an analysis for an older one dropped)
        self.window_sec = DEFAULT_WINDOW_SEC     # the span of time a main view shows (the wheel steps it)
        self.signals = _Signals()
        self.signals.decoded.connect(self._decoded)
        self.decoders = QtCore.QThreadPool(self)
        self.decoders.setMaxThreadCount(1)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)
        self.connection = ConnectionBar(self.listen)
        root.addWidget(self.connection)
        self.views = [DeckViews(self.zoom) for _ in range(DECKS)]
        lanes = QtWidgets.QFrame()
        lanes.setObjectName("deckLanes")
        theme = current_theme()
        lanes.setStyleSheet(f"QFrame#deckLanes {{ background: {theme.base}; border: 1px solid {theme.border};"
                            f" border-radius: 3px; }}")
        lanes_lay = QtWidgets.QVBoxLayout(lanes)
        lanes_lay.setContentsMargins(4, 4, 4, 4)
        lanes_lay.setSpacing(2)
        for n, views in enumerate(self.views, start=1):
            row = QtWidgets.QHBoxLayout()
            row.setSpacing(4)
            number = _label(12, True, text=str(n))
            number.setFixedWidth(number.fontMetrics().horizontalAdvance("88") + 4)
            number.setAlignment(QtCore.Qt.AlignCenter)
            row.addWidget(number)
            row.addWidget(views.main, 1)
            lanes_lay.addLayout(row, 1)
        root.addWidget(lanes, 1)
        grid = QtWidgets.QGridLayout()
        grid.setSpacing(6)
        # the decks two by two, each one's record, BPM and key on the side facing the middle, the BPMs and keys
        # nearest it (the top decks' under their records, the bottom ones' over theirs): they meet there
        self.panels = [DeckPanel(n, views.overview, inner_right=bool(n % 2), top=n <= DECKS // 2)
                       for n, views in enumerate(self.views, start=1)]
        for i, panel in enumerate(self.panels):
            grid.addWidget(panel, i // 2, i % 2)
            panel.phrase_line.clicked.connect(self.switch_phrase_unit)
        self.phrase_unit = SECTION
        self.set_phrase_unit(getattr(cfg.deckconfig, "monitor_phrase_unit", SECTION))
        root.addLayout(grid, 1)                         # the main views and the decks' overviews share the height
        for views in self.views:
            views.install(cfg)
            views.set_window(self.window_sec)
        self.resize(1360, 960)
        self._timer = QtCore.QTimer(self)
        self._timer.setTimerType(QtCore.Qt.PreciseTimer)     # a coarse one rounds to Windows' 15.6 ms ticks
        self._timer.timeout.connect(self.refresh)
        self._timer.setInterval(max(1, round(1000 / max(1, min(MAX_FPS, int(cfg.viewconfig.fps or 60))))))
        self._status_timer = QtCore.QTimer(self)
        self._status_timer.setInterval(STATUS_MS)
        self._status_timer.timeout.connect(self._show_status)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.refresh()
        self._show_status()
        self._timer.start()
        self._status_timer.start()

    def closeEvent(self, event) -> None:
        self._timer.stop()
        self._status_timer.stop()
        super().closeEvent(event)
        QtWidgets.QApplication.quit()

    def shut_down(self) -> None:
        """Its views' render threads stopped, its decoding waited for, its receiver closed."""
        for views in self.views:
            views.clear()
        self.decoders.waitForDone()
        if self.source is not None:
            self.source.close()
            self.source = None

    # ---- where the decks are received ----

    def start(self) -> None:
        """Listening where the settings say (the last place set here)."""
        deck = self.cfg.deckconfig
        where = Where(str(getattr(deck, "monitor_protocol", OSC) or OSC), str(deck.monitor_host or "127.0.0.1"),
                      int(deck.monitor_port), str(getattr(deck, "monitor_interface", "") or ""))
        self.connection.set_where(where)
        self.listen(where)

    def listen(self, where: Where) -> bool:
        """The receiver ``where`` names (the app told, to keep it); one that cannot listen there: said, the one before
        kept."""
        if self.source is not None and self.source.where == where:
            return True
        old, self.source = self.source, None
        if old is not None:
            old.close()                  # its ports free first: the same port on another address
        try:
            self.source = DeckSource(where, self.track_rows)
        except (OSError, ValueError, OverflowError) as exc:
            self.error = cannot_listen(where, exc)
            if old is not None:                # the one before, again
                try:
                    self.source = DeckSource(old.where, self.track_rows)
                except (OSError, ValueError, OverflowError):
                    self.source = None
            self._show_status()
            return False
        self.error = ""
        self._ask({"type": "listening", "protocol": where.protocol, "host": where.host, "port": int(where.port),
                   "interface": where.interface})
        self._show_status()
        return True

    def _show_status(self) -> None:
        self.connection.status.setText(status_text(self.source, self.error))
        color = "#d8a441" if self.error else current_theme().text_dim
        self.connection.status.setStyleSheet(f"color: {color};")

    # ---- the app's messages ----

    def command(self, message: dict) -> None:
        kind = message.get("type")
        if kind == "tracks":
            self.gen = int(message.get("gen") or 0)
            self.track_rows = tuple(message.get("tracks") or ())
            if self.source is not None:
                self.source.set_tracks(self.track_rows)
            for data in self.tracks.values():        # the library changed: the analyses asked again (shown meanwhile)
                data.asked, data.unread, data.stale = False, False, True
        elif kind == "analysis":
            data = self.tracks.get(str(message.get("uid") or ""))
            if data is None:
                return
            data.asked = False
            if int(message.get("gen") or 0) == self.gen:
                new = message.get("analysis")
                data.stale, data.unread = False, new is None
                if new is None or data.analysis is None or not same_marks(data.analysis, new):
                    data.analysis = new              # the same as shown: kept (the views not loaded again)
            # an answer for an older library: asked again by the next refresh
        elif kind == "settings":
            self.apply_settings(message.get("cfg"))
        elif kind == "close":
            self.close()

    def apply_settings(self, cfg) -> None:
        """The app's settings changed: read here (the installed store) and the views made again with them."""
        if cfg is None:
            return
        self.cfg = cfg
        if _STORE is not None:
            _STORE.replace(cfg)
        from core.marks.phrases import set_phrase_label_names
        set_phrase_label_names(cfg.viewconfig.phrase_labels.by_label())
        self.set_phrase_unit(getattr(cfg.deckconfig, "monitor_phrase_unit", self.phrase_unit))
        for views in self.views:
            views.install(cfg)                       # each view draws the loaded track again (render_initial)

    def _ask(self, message: dict) -> None:
        if self.messages is not None:
            try:
                self.messages.put(message)
            except (ValueError, OSError):
                pass

    # ---- the decks ----

    def zoom(self, steps: int) -> None:
        """Every main view ``steps`` widths narrower (< 0) or wider."""
        self.window_sec = step_window(self.window_sec, steps)

    def set_phrase_unit(self, unit: str) -> None:
        """Every deck's phrase now and next by ``unit`` (section or label)."""
        self.phrase_unit = unit if unit in UNITS else SECTION
        for panel in self.panels:
            panel.phrase_line.set_unit(self.phrase_unit)

    def switch_phrase_unit(self) -> None:
        """A deck's phrase line clicked: every deck by the other unit, the app told (kept in deck.json)."""
        self.set_phrase_unit(LABEL if self.phrase_unit == SECTION else SECTION)
        self._ask({"type": "phrase_unit", "unit": self.phrase_unit})
        self.refresh()

    def refresh(self) -> None:
        decks = self.source.decks() if self.source is not None else \
            [MonitorDeck(n, None, True, None, None, 0.0) for n in range(1, DECKS + 1)]
        used = set()
        for views, panel, d in zip(self.views, self.panels, decks):
            row = d.row
            if d.shown and row is not None:
                data = self._track(row)
                used.add(data.uid)
            else:
                data, row = self._sent(d) if d.shown else (None, None)
            views.show(data, row)
            pitch = d.state.pitch if d.state is not None else None
            views.set_played(pitch or 1.0)
            views.set_window(self.window_sec * (pitch or 1.0))     # one span of time: a faster deck shows more
            if data is not None:
                views.set_time(d.position_sec)
            panel.show_deck(d, data, views)
        self._forget(used)

    def _track(self, row) -> TrackData:
        data = self.tracks.get(row.uid)
        if data is None:
            data = self.tracks[row.uid] = TrackData(row.uid, row.path)
        self.tracks.move_to_end(row.uid)
        if (data.analysis is None or data.stale) and not data.asked and not data.unread:
            data.asked = True
            self._ask({"type": "analysis?", "uid": data.uid})
        if data.pcm is None and not data.decoding and not data.failed:
            data.decoding = True
            self.decoders.start(_Decode(self.signals, data.uid, data.path))
        return data

    def _sent(self, d: MonitorDeck) -> tuple[Optional[TrackData], Optional[TrackRow]]:
        """The deck's track the library does not hold, as the sender's own grid, CUE points and key show it
        (OSC Full, Pro DJ Link: no audio); (None, None) when it sends none or its length is not known."""
        state = d.state
        marks = state.marks if state is not None else None
        kept = self.sent.get(d.deck)
        if kept is not None and kept[1] is marks and kept[0] == state.track:
            return kept[2], kept[3]
        analysis = source_analysis(marks, state.track.duration_sec) if marks is not None and not marks.empty() \
            else None
        if analysis is None:
            self.sent.pop(d.deck, None)
            return None, None
        if kept is not None and kept[0] == state.track:
            data, row = kept[2], kept[3]             # the same track's marks again: shown on the same track
            data.analysis = analysis
        else:
            self.sent_count += 1
            uid = f"sent:{d.deck}:{self.sent_count}"
            data = TrackData(uid, "", analysis=analysis, failed=True)      # nothing to decode
            row = TrackRow(uid=uid, path="", title=state.track.title, artist=state.track.artist,
                           duration=state.track.duration_sec)
        self.sent[d.deck] = (state.track, marks, data, row)
        return data, row

    def _forget(self, used: set) -> None:
        """The tracks beyond KEPT_TRACKS no deck shows let go (the oldest first)."""
        for uid in list(self.tracks):
            if len(self.tracks) <= KEPT_TRACKS:
                break
            if uid not in used:
                del self.tracks[uid]

    def _decoded(self, uid: str, pcm, art) -> None:
        data = self.tracks.get(uid)
        if data is None:
            return
        data.decoding = False
        data.pcm, data.art, data.failed = pcm, art, pcm is None


def same_marks(a: StoredAnalysis, b: StoredAnalysis) -> bool:
    """Whether two reads of a track's analysis show the same (its length, grid, keys and marks; a reanalysis changes
    those too): the library read again after another track's change gives the same."""
    try:
        return (a.duration_sec == b.duration_sec and a.phrases == b.phrases and a.cue_points == b.cue_points
                and a.jump_cues == b.jump_cues
                and all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in
                        ((a.beats, b.beats), (a.tempo_segments, b.tempo_segments),
                         (a.key_segments, b.key_segments))))
    except Exception:
        return False


def cannot_listen(where: Where, error: Exception) -> str:
    if where.protocol == OSC:
        return tr("Deck Monitor could not listen for OSC on {host}:{port} ({error}).").format(
            host=where.host, port=where.port, error=error)
    return tr("Deck Monitor could not join {protocol} on {network} ({error}).").format(
        protocol=PROTOCOL_NAMES.get(where.protocol, where.protocol), network=where.interface or tr("the first network"),
        error=error)


def status_text(source: Optional[DeckSource], error: str = "") -> str:
    if error:
        return error
    if source is None:
        return tr("Not listening.")
    state = source.status()
    listening = str(state.get("listening") or "")
    decks = state.get("decks") or {}
    if not decks:
        return tr("Listening on {address}; nothing received yet.").format(address=listening)
    newest = min(float(age) for age in decks.values())
    return tr("Listening on {address}; decks {decks} heard (last message {age} s ago, from {sender}).").format(
        address=listening, decks=", ".join(str(n) for n in decks), age=f"{newest:.1f}",
        sender=state.get("sender") or "?")


def run(settings: dict, commands, messages) -> None:
    """The window's process (spawned by core.deck_monitor.controller): it listens, shows the decks until its window
    closes or the app says close, then says done."""
    from system import i18n, os_support
    from system.process_worker import install_parent_watchdog
    from system.resource_paths import APP_ICON, resource_path
    install_parent_watchdog()
    i18n.install(str(settings.get("language") or ""))
    os_support.set_taskbar_identity()               # the app's taskbar button, not Python's
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    app.setWindowIcon(QtGui.QIcon(str(resource_path(APP_ICON))))
    from ui.fonts import load_fonts_and_set_global
    from ui.theme import apply_theme
    apply_theme(app)
    load_fonts_and_set_global(app)
    i18n.install_qt(app)
    from core.marks.phrases import set_phrase_label_names
    from core.settings import store as settings_store
    global _STORE
    cfg = settings["cfg"]
    _STORE = ReadOnlyStore(cfg)
    settings_store.install(_STORE)
    set_phrase_label_names(cfg.viewconfig.phrase_labels.by_label())
    window = DeckMonitorWindow(cfg, messages)
    window.start()

    def take_commands() -> None:
        while True:
            try:
                message = commands.get_nowait()
            except Empty:
                return
            except (OSError, ValueError, EOFError):
                window.close()              # the app's side gone
                return
            window.command(message)

    timer = QtCore.QTimer()
    timer.setInterval(COMMAND_MS)
    timer.timeout.connect(take_commands)
    timer.start()
    window.show()
    app.exec()
    timer.stop()
    window.shut_down()
    try:
        messages.put({"type": "done"})
    except (OSError, ValueError):
        pass
