"""Search by Music in the library's search bar: the microphone button turns the search field into the microphone's
waveform and level meter and listens (a hum, singing, a whistle, or the song itself played from a phone or a
speaker); the last seconds heard (searchconfig.music_listen_window) are searched every few seconds
(music_listen_interval) and the library shows the songs found, best first: the searches joined over time
(core.music_search.listening), so a song sung on and on leads. The microphone off, the results stay and
are no longer followed; the close button beside it is the plain library again.

`MusicListener` listens and searches (core.music_search) over music_search.db: the index read on a thread when
first needed and again after Index Now or a library change, the reference lines held in memory; a search runs on a
thread of its own (its kernels leave the interpreter lock), one at a time: one due while another runs is skipped.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Callable

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtMultimedia import QAudioFormat, QAudioSource, QMediaDevices

from core.config import save_deck
from core.event_bus import EventBus
from core.i18n import tr
from core.library import jobs
from core.music_search.listening import Listening
from core.music_search.search import SearchResult, search_audio
from ui.elided_label import ElidedLabel
from ui.theme import current_theme

SEEK_WAIT_SEC = 20.0          # a result loaded onto the deck: seeked once its audio is ready, if within this
WAVE_SEC = 4.0                # the waveform shows this much of what was heard last
WAVE_BARS = 160
QUIET_RMS = 10 ** (-55 / 20)  # the window this quiet (-55 dBFS): nothing heard, nothing searched


def mmss(sec: float) -> str:
    sec = max(0, int(round(sec)))
    return f"{sec // 60}:{sec % 60:02d}"


def _mono_float(data: bytes, fmt: QAudioFormat) -> np.ndarray:
    sf = fmt.sampleFormat()
    if sf == QAudioFormat.SampleFormat.Int16:
        a = np.frombuffer(data, np.int16).astype(np.float32) / 32768.0
    elif sf == QAudioFormat.SampleFormat.Int32:
        a = np.frombuffer(data, np.int32).astype(np.float32) / 2147483648.0
    elif sf == QAudioFormat.SampleFormat.Float:
        a = np.frombuffer(data, np.float32).astype(np.float32)
    else:
        a = (np.frombuffer(data, np.uint8).astype(np.float32) - 128.0) / 128.0
    ch = max(fmt.channelCount(), 1)
    return a[: a.size // ch * ch].reshape(-1, ch).mean(axis=1)


# --- icons (drawn in the theme's colours, crisp at a high pixel ratio) -----------------------------------------------
def _icon(draw, size: int = 18) -> QtGui.QIcon:
    icon = QtGui.QIcon()
    for ratio in (1.0, 2.0):
        pm = QtGui.QPixmap(int(size * ratio), int(size * ratio))
        pm.setDevicePixelRatio(ratio)
        pm.fill(QtCore.Qt.transparent)
        p = QtGui.QPainter(pm)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        draw(p, size)
        p.end()
        icon.addPixmap(pm)
    return icon


def mic_icon(color: str | None = None) -> QtGui.QIcon:
    def draw(p: QtGui.QPainter, s: int) -> None:
        c = QtGui.QColor(color or current_theme().text)
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(c)
        w = s * 0.34
        p.drawRoundedRect(QtCore.QRectF((s - w) / 2, s * 0.08, w, s * 0.52), w / 2, w / 2)
        pen = QtGui.QPen(c, max(1.4, s * 0.09))
        pen.setCapStyle(QtCore.Qt.RoundCap)
        p.setPen(pen)
        p.setBrush(QtCore.Qt.NoBrush)
        p.drawArc(QtCore.QRectF(s * 0.22, s * 0.22, s * 0.56, s * 0.52), 200 * 16, 140 * 16)
        p.drawLine(QtCore.QPointF(s / 2, s * 0.74), QtCore.QPointF(s / 2, s * 0.9))
        p.drawLine(QtCore.QPointF(s * 0.34, s * 0.9), QtCore.QPointF(s * 0.66, s * 0.9))
    return _icon(draw)


def close_icon() -> QtGui.QIcon:
    def draw(p: QtGui.QPainter, s: int) -> None:
        pen = QtGui.QPen(QtGui.QColor(current_theme().text), max(1.4, s * 0.1))
        pen.setCapStyle(QtCore.Qt.RoundCap)
        p.setPen(pen)
        m = s * 0.28
        p.drawLine(QtCore.QPointF(m, m), QtCore.QPointF(s - m, s - m))
        p.drawLine(QtCore.QPointF(s - m, m), QtCore.QPointF(m, s - m))
    return _icon(draw)


# --- the waveform and the level --------------------------------------------------------------------------------------
class MicWaveform(QtWidgets.QWidget):
    """The last WAVE_SEC heard as bars (a peak each, on a dB scale), the newest at the right; its level meter at the
    right end."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._bars = np.zeros(WAVE_BARS, np.float32)
        self._level = 0.0
        self._carry = np.zeros(0, np.float32)
        self._rate = 48000
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setMinimumSize(160, 12)

    def reset(self, rate: int) -> None:
        self._bars[:] = 0
        self._level = 0.0
        self._carry = np.zeros(0, np.float32)
        self._rate = max(int(rate), 1)
        self.update()

    def add(self, y: np.ndarray) -> None:
        per = max(1, int(self._rate * WAVE_SEC / WAVE_BARS))
        rms = float(np.sqrt(np.mean(y ** 2))) if y.size else 0.0
        self._level = float(np.clip((20 * np.log10(max(rms, 1e-6)) + 60) / 60, 0, 1))
        y = np.concatenate([self._carry, np.abs(y)])
        n = y.size // per
        if n:
            peaks = y[: n * per].reshape(n, per).max(axis=1)[-WAVE_BARS:]
            self._bars = np.roll(self._bars, -peaks.size)
            self._bars[-peaks.size:] = peaks
        self._carry = y[n * per:]
        self.update()

    def silence(self) -> None:
        self._level = 0.0
        self.update()

    def paintEvent(self, _event) -> None:
        theme = current_theme()
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        r = QtCore.QRectF(self.rect()).adjusted(2, 4, -2, -4)
        meter_w = 6.0
        wave = QtCore.QRectF(r.left(), r.top(), r.width() - meter_w - 8, r.height())
        mid = wave.center().y()
        step = wave.width() / WAVE_BARS
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(QtGui.QColor(theme.accent))
        heights = np.clip((20 * np.log10(np.maximum(self._bars, 1e-5)) + 54) / 54, 0.04, 1.0)
        for i, h in enumerate(heights):
            hh = float(h) * wave.height()
            p.drawRoundedRect(QtCore.QRectF(wave.left() + i * step, mid - hh / 2, max(1.0, step * 0.6), hh), 1, 1)
        meter = QtCore.QRectF(r.right() - meter_w, r.top(), meter_w, r.height())
        p.setBrush(QtGui.QColor(theme.separator))
        p.drawRoundedRect(meter, 2, 2)
        lit = meter.height() * self._level
        p.setBrush(QtGui.QColor(theme.danger if self._level > 0.92 else theme.success))
        p.drawRoundedRect(QtCore.QRectF(meter.left(), meter.bottom() - lit, meter_w, lit), 2, 2)
        p.end()


class MusicSearchBar(QtWidgets.QFrame):
    """What the search field's place holds in music mode: the waveform with its level meter, the input listened to
    ("Listening on <input>": the input's name a link to the system's inputs, one picked from them) and the status."""

    device_picked = QtCore.Signal(str)        # an input's name, "" the system's default

    DEVICE_WIDTH = 280                        # px of the input's name at most (cut in its middle)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("musicSearchBar")
        theme = current_theme()
        self.setStyleSheet(f"#musicSearchBar {{ border: 1px solid {theme.separator}; border-radius: 4px; }}")
        self.wave = MicWaveform()
        self.device = QtWidgets.QLabel()
        self.device.setTextFormat(QtCore.Qt.RichText)
        self.device.setTextInteractionFlags(QtCore.Qt.LinksAccessibleByMouse | QtCore.Qt.LinksAccessibleByKeyboard)
        self.device.setStyleSheet(f"color: {theme.text_dim}; background: transparent;")
        self.device.setToolTip(tr("The microphone Search by Music listens to: click its name to pick another."))
        self.device.linkActivated.connect(lambda _link: self._menu())
        self.status = ElidedLabel()
        self.status.setStyleSheet(f"color: {theme.text_dim}; background: transparent;")
        self._inputs: Callable[[], list[str]] = list
        self._picked = ""
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(4, 0, 6, 0)
        lay.setSpacing(8)
        lay.addWidget(self.wave, 3)
        lay.addWidget(self.device, 0)
        lay.addWidget(self.status, 2)

    def show_device(self, name: str, picked: str, listening: bool, inputs: Callable[[], list[str]]) -> None:
        """The input heard (``name``), the one picked ("": the system's default), whether it is listened to now, and
        how to list the inputs for the menu."""
        import html
        theme = current_theme()
        self._inputs, self._picked = inputs, picked
        shown = self.fontMetrics().elidedText(name or tr("no microphone"), QtCore.Qt.ElideMiddle, self.DEVICE_WIDTH)
        link = (f'<a href="device" style="color: {theme.accent}; text-decoration: none;">'
                f'{html.escape(shown)}&nbsp;▾</a>')
        text = tr("Listening on {device}") if listening else tr("Microphone: {device}")
        self.device.setText(html.escape(text).replace("{device}", link))

    def _menu(self) -> None:
        menu = QtWidgets.QMenu(self)
        group = QtGui.QActionGroup(menu)
        names = self._inputs()
        entries = [("", tr("System default"))] + [(n, n) for n in names]
        if self._picked and self._picked not in names:
            entries.append((self._picked, tr("{name} (not connected)").format(name=self._picked)))
        for value, label in entries:
            act = menu.addAction(label)
            act.setCheckable(True)
            act.setChecked(value == self._picked)
            act.setData(value)
            group.addAction(act)
        chosen = menu.exec(self.device.mapToGlobal(QtCore.QPoint(0, self.device.height())))
        if chosen is not None and chosen.data() != self._picked:
            self.device_picked.emit(str(chosen.data()))


# --- listening and searching -----------------------------------------------------------------------------------------
class _Bridge(QtCore.QObject):
    loaded = QtCore.Signal(object)     # (index, connection, uids of the library) or an error text
    done = QtCore.Signal(object)       # (SearchResult, ms) or an error text


class MusicListener(QtCore.QObject):
    """The microphone listened to and its last seconds searched every few seconds."""

    audio = QtCore.Signal(object)             # a chunk heard: mono float at `rate`
    rate_changed = QtCore.Signal(int)         # an input opened at this rate
    listening_changed = QtCore.Signal(bool)
    device_changed = QtCore.Signal()          # another input picked, listening started or stopped: device() / picked()
    status = QtCore.Signal(str)
    results = QtCore.Signal(object)           # SearchResult, while listening (none when nothing was found)

    def __init__(self, bus: EventBus, cfg_getter: Callable[[], object], library, parent=None):
        """``library``: the library's owner (core.library.service), which opens the index and lends its read-only
        connection to the searches (they read the landmarks every few seconds: on the owner's thread they would
        hold its other jobs)."""
        super().__init__(parent)
        self._cfg_getter = cfg_getter
        self._owner = library
        self._index = None
        self._landmarks = None             # the index's landmarks (core.library.music_index.LandmarkIndex), lent
        self._library: set[str] = set()
        self._stale = True
        self._loading = False
        self._searching = False
        self._source: QAudioSource | None = None
        self._io = None
        self._live: list[np.ndarray] = []
        self._live_n = 0
        self.rate = 0
        self._t0 = 0.0
        self._listening = Listening()
        self._bridge = _Bridge()
        self._bridge.loaded.connect(self._on_loaded)
        self._bridge.done.connect(self._on_done)
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._on_tick)
        bus.sig_music_index_finished.connect(lambda *_: self._mark_stale())
        bus.sig_lib_updated.connect(lambda *_: self._mark_stale())

    def listening(self) -> bool:
        return self._source is not None

    # the index
    def _mark_stale(self) -> None:
        self._stale = True
        if self.listening():
            self._load_index()

    def _load_index(self) -> None:
        if self._loading or self._searching:
            return
        self._loading = True
        self._stale = False
        threading.Thread(target=self._read_index, name="MusicSearchIndex", daemon=True).start()

    def _read_index(self) -> None:
        try:
            # opened and read by the library's owner (its landmarks lent to the searches); the library's tracks
            # from its snapshot
            opened = self._owner.call(jobs.open_music_search)
            con, index = opened if opened is not None else (None, None)
            library = set(self._owner.snapshot.by_uid)
            if index is not None and len(index):
                import numba
                from core.music_search.search import SEARCH_THREADS
                numba.set_num_threads(SEARCH_THREADS)
                index.search(np.full(40, 60.0), top=1)          # the kernels: from numba's cache, else compiled
            self._bridge.loaded.emit((index, con, library))
        except Exception as exc:
            self._bridge.loaded.emit(f"{type(exc).__name__}: {exc}")

    def _on_loaded(self, res) -> None:
        self._loading = False
        if isinstance(res, str):
            self.status.emit(tr("The index could not be read: {error}").format(error=res))
            return
        old = self._landmarks
        self._index, self._landmarks, self._library = res
        if old is not None and old is not self._landmarks:
            old.close()
        indexed = len(set(self._index.uids) & self._library) if self._index is not None else 0
        if not indexed:
            self.status.emit(tr("No track is indexed for Search by Music yet: Settings > Search > Index Now "
                                "(new tracks are indexed when analyzed)."))
        if self._stale:
            self._load_index()

    # the microphone
    @staticmethod
    def inputs() -> list[str]:
        """The system's input devices by name."""
        return [d.description() for d in QMediaDevices.audioInputs()]

    def picked(self) -> str:
        """The input picked ("": the system's default)."""
        return str(getattr(self._cfg_getter().deckconfig, "music_input_device", "") or "")

    def device(self):
        """The input listened to: the one picked when the system has it, else the system's default."""
        name = self.picked()
        for dev in QMediaDevices.audioInputs():
            if name and dev.description() == name:
                return dev
        return QMediaDevices.defaultAudioInput()

    def pick(self, name: str) -> None:
        """Listen to another input ("": the system's default), kept in deck.json; while listening, from now on."""
        deck = self._cfg_getter().deckconfig
        deck.music_input_device = str(name or "")
        save_deck(self._cfg_getter())
        if self._source is not None:
            self._close_source()
            if not self._open_source():
                self._timer.stop()
                self.listening_changed.emit(False)
                return
        self.device_changed.emit()

    def _open_source(self) -> bool:
        dev = self.device()
        if dev.isNull():
            self.status.emit(tr("No microphone found."))
            return False
        fmt = dev.preferredFormat()
        self._source = QAudioSource(dev, fmt, self)
        self.rate = fmt.sampleRate()
        self._live, self._live_n = [], 0
        self._io = self._source.start()
        self._io.readyRead.connect(self._on_audio)
        self.rate_changed.emit(self.rate)
        return True

    def _close_source(self) -> None:
        self._source.stop()
        self._source.deleteLater()
        self._source = self._io = None
        self._live, self._live_n = [], 0

    def start(self) -> bool:
        if self._source is not None:
            return True
        if not self._open_source():
            return False
        self._t0 = time.monotonic()
        self._listening = Listening(float(self._cfg_getter().searchconfig.music_listen_window))
        self._timer.start(int(float(self._cfg_getter().searchconfig.music_listen_interval) * 1000))
        if self._index is None or self._stale:
            self._load_index()
        self.status.emit("")
        self.listening_changed.emit(True)
        self.device_changed.emit()
        return True

    def stop(self) -> None:
        self._timer.stop()
        if self._source is None:
            return
        self._close_source()
        self.listening_changed.emit(False)
        self.device_changed.emit()

    def _on_audio(self) -> None:
        if self._io is None:
            return
        data = bytes(self._io.readAll())
        if not data:
            return
        y = _mono_float(data, self._source.format())
        self._live.append(y)
        self._live_n += y.size
        keep = int((float(self._cfg_getter().searchconfig.music_listen_window) + 1.0) * self.rate)
        while self._live and self._live_n - self._live[0].size >= keep:
            self._live_n -= self._live.pop(0).size
        self.audio.emit(y)

    def _on_tick(self) -> None:
        if self._source is None:
            return
        self._on_audio()
        if self._searching or self._loading or self._index is None:
            return
        n = int(float(self._cfg_getter().searchconfig.music_listen_window) * self.rate)
        if self._live_n < min(n, 2 * self.rate):
            return
        y = np.concatenate(self._live)[-n:]
        if float(np.sqrt(np.mean(y ** 2))) < QUIET_RMS:
            return         # nothing heard: no search
        self._searching = True
        threading.Thread(target=self._run_search, args=(y, self.rate, self._index, self._landmarks, time.monotonic()),
                         name="MusicSearch", daemon=True).start()

    def _run_search(self, y, rate, index, con, heard_at) -> None:
        try:
            result = search_audio(y, rate, index, con, top=20)
            self._bridge.done.emit((result, heard_at))
        except Exception as exc:
            self._bridge.done.emit(f"{type(exc).__name__}: {exc}")

    def _on_done(self, res) -> None:
        self._searching = False
        if self._stale:
            self._load_index()
        if self._source is None:
            return         # turned off meanwhile: no more following
        if isinstance(res, str):
            self.status.emit(tr("The search failed: {error}").format(error=res))
            return
        result, heard_at = res
        result: SearchResult
        # joined with the searches before (core.music_search.listening): a song heard on and on leads
        joined = self._listening.update(result, heard_at)
        listening = mmss(time.monotonic() - self._t0)
        if not joined:
            listening += "  ·  " + tr("Nothing found yet: hum or whistle for 8 seconds or more, or play the song.")
        self.status.emit(listening)
        if joined:
            self.results.emit(SearchResult(joined[0].kind, tuple(joined), result.voiced_sec))


class DeckSeekAfterLoad(QtCore.QObject):
    """A track loaded onto the deck and seeked to a place once its audio is ready (the deck takes no seek before)."""

    def __init__(self, bus: EventBus, model=None, parent=None):
        super().__init__(parent)
        self.bus = bus
        self.model = model
        self._pending: tuple[str, float, float] | None = None
        bus.sig_duration_changed.connect(self._on_duration)

    def _held(self) -> str:
        props = getattr(self.model, "properties", None) or {}
        return str(props.get("path") or "") if isinstance(props, dict) else ""

    @staticmethod
    def _same(a: str, b: str) -> bool:
        return bool(a and b) and os.path.normcase(os.path.normpath(a)) == os.path.normcase(os.path.normpath(b))

    def load(self, path: str, at: float) -> None:
        if self._same(self._held(), path):
            self.bus.sig_seek_requested.emit(at)
            return
        self._pending = (path, at, time.monotonic() + SEEK_WAIT_SEC)
        self.bus.sig_request_load_track.emit(path)

    def _on_duration(self, duration: float) -> None:
        pending = self._pending
        if pending is None or duration <= 0:
            return
        path, at, until = pending
        self._pending = None
        if time.monotonic() <= until and self._same(self._held(), path):
            self.bus.sig_seek_requested.emit(at)
