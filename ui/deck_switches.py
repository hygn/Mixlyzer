"""The deck's and its mixer channel's switches, a row between the player and the overview, as a DJM's and a CDJ's
are: the jog wheel's mode (VINYL: its top holds the record, CDJ: a bend only), the EQ (ISO: an isolator, a band down
to a kill; EQ: -26 dB at the bottom), the channel fader's curve (each drawn: the app's default, steep: up fast,
slow: most of the rise at the top), the headphones (the channel's CUE, their level) and the master's level. A click
or a turn sends a request (sig_jog_mode_requested, sig_mixer_requested, sig_channel_cue_requested); what is lit and
where a knob points is what the deck reports (DeckReport: its lamps' VINYL bit, channel_cue, the mixer's values), so
a controller's switch or knob shows too. The native engine's (the Python engine has none of them): disabled
there."""
from __future__ import annotations

import math

from PySide6 import QtCore, QtGui, QtWidgets

from audioengine.native import client
from core.event_bus import EventBus
from core.i18n import tr
from ui.knob import Knob
from ui.theme import current_theme

GLYPH_W, GLYPH_H = 26, 14


def _curve_icon(curve: int, color: str) -> QtGui.QIcon:
    """The channel fader's curve ``curve`` (FADER_CURVES' place) drawn: its gain over its travel."""
    ratio = 2.0
    pix = QtGui.QPixmap(int(GLYPH_W * ratio), int(GLYPH_H * ratio))
    pix.setDevicePixelRatio(ratio)
    pix.fill(QtCore.Qt.transparent)
    p = QtGui.QPainter(pix)
    p.setRenderHint(QtGui.QPainter.Antialiasing)
    p.setPen(QtGui.QPen(QtGui.QColor(color), 1.6, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
    path = QtGui.QPainterPath()
    for k in range(41):
        x = k / 40
        if curve == 1:
            y = 1 - (1 - x) ** 3
        elif curve == 2:
            y = x ** 3
        else:   # the app's default (utils/volume.py; src/mixer.rs fader_gain)
            v = x ** 0.4
            y = 1.0 if v > 0.99 else -math.log(1 - v) / math.log(100)
        point = QtCore.QPointF(2 + x * (GLYPH_W - 4), GLYPH_H - 2 - y * (GLYPH_H - 4))
        path.moveTo(point) if k == 0 else path.lineTo(point)
    p.drawPath(path)
    p.end()
    return QtGui.QIcon(pix)


class _Segment(QtWidgets.QToolButton):
    """One choice of a switch, lit as reported."""

    def __init__(self, text: str = "", parent=None) -> None:
        super().__init__(parent)
        self.setText(text)
        f = self.font()
        f.setPointSize(8)
        f.setBold(True)
        self.setFont(f)
        self.setFixedHeight(22)
        self.setMinimumWidth(40)
        theme = current_theme()
        self.setProperty("lit", False)
        self.setStyleSheet(
            f"QToolButton {{ border: 1px solid {theme.border}; background: {theme.raised}; color: {theme.text_dim};"
            f" padding: 0 6px; }}\n"
            f"QToolButton:hover {{ background: {theme.raised_hover}; color: {theme.text}; }}\n"
            f"QToolButton[lit=true] {{ background: {theme.accent}; border-color: {theme.accent};"
            f" color: {theme.on_accent}; }}\n"
            f"QToolButton:disabled {{ color: {theme.text_disabled}; }}")

    def set_lit(self, lit: bool) -> None:
        if bool(self.property("lit")) != bool(lit):
            self.setProperty("lit", bool(lit))
            self.style().unpolish(self)
            self.style().polish(self)


def _label(text: str) -> QtWidgets.QLabel:
    label = QtWidgets.QLabel(text)
    f = label.font()
    f.setPointSize(7)
    f.setBold(True)
    label.setFont(f)
    label.setStyleSheet(f"color: {current_theme().text_dim};")
    return label


class DeckSwitches(QtWidgets.QWidget):
    """The row (see the module). ``set_report`` shows the deck's report; ``set_available`` on the native engine."""

    def __init__(self, bus: EventBus, parent=None) -> None:
        super().__init__(parent)
        self.bus = bus
        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(8, 0, 8, 0)
        row.setSpacing(0)

        def group(title: str, segments: list[_Segment]) -> None:
            row.addWidget(_label(title), 0, QtCore.Qt.AlignVCenter)
            row.addSpacing(6)
            for s in segments:
                row.addWidget(s, 0, QtCore.Qt.AlignVCenter)
            row.addSpacing(18)

        self.vinyl, self.cdj = _Segment("VINYL"), _Segment("CDJ")
        self.vinyl.setToolTip(tr("JOG MODE VINYL: touching the jog wheel's top holds the record (scratch)."))
        self.cdj.setToolTip(tr("JOG MODE CDJ: the jog wheel bends the tempo only, its top's touch ignored."))
        self.vinyl.clicked.connect(lambda: self.bus.sig_jog_mode_requested.emit(True))
        self.cdj.clicked.connect(lambda: self.bus.sig_jog_mode_requested.emit(False))
        group("JOG MODE", [self.vinyl, self.cdj])

        self.eq = [_Segment("ISO"), _Segment("EQ")]
        self.eq[0].setToolTip(tr("EQ CURVE ISO: an isolator, each band from a kill to +6 dB."))
        self.eq[1].setToolTip(tr("EQ CURVE EQ: an equalizer, each band from -26 dB to +6 dB."))
        for k, b in enumerate(self.eq):
            b.clicked.connect(lambda _c=False, k=k: self.bus.sig_mixer_requested.emit("eq_mode", float(k)))
        group("EQ CURVE", self.eq)

        self.fader = []
        tips = (tr("CH FADER curve: the app's (slow at first, full at the top)."),
                tr("CH FADER curve: steep (up fast)."),
                tr("CH FADER curve: slow (most of the rise at the top)."))
        for k, tip in enumerate(tips):
            b = _Segment()
            b.setIcon(_curve_icon(k, current_theme().text))
            b.setIconSize(QtCore.QSize(GLYPH_W, GLYPH_H))
            b.setToolTip(tip)
            b.clicked.connect(lambda _c=False, k=k: self.bus.sig_mixer_requested.emit("fader_curve", float(k)))
            self.fader.append(b)
        group("CH FADER", self.fader)

        # the headphones: the channel's CUE and their level; the master's level
        self.cue = _Segment("CUE")
        self.cue.setToolTip(tr("Headphones CUE: the deck's channel, before its fader, in the headphones."))
        self.cue.clicked.connect(self.toggle_cue)
        self.phones = self._level_knob(lambda v: self.bus.sig_mixer_requested.emit("hp_a_level", v))
        self.master = self._level_knob(lambda v: self.bus.sig_mixer_requested.emit("master_level", v))
        row.addWidget(_label("PHONES"), 0, QtCore.Qt.AlignVCenter)
        row.addSpacing(6)
        row.addWidget(self.cue, 0, QtCore.Qt.AlignVCenter)
        row.addSpacing(6)
        row.addWidget(self.phones, 0, QtCore.Qt.AlignVCenter)
        row.addSpacing(18)
        row.addWidget(_label("MASTER"), 0, QtCore.Qt.AlignVCenter)
        row.addSpacing(6)
        row.addWidget(self.master, 0, QtCore.Qt.AlignVCenter)
        self._cue_on = False
        self._show_level(self.phones, tr("Headphones level"), client.LEVEL_UNITY)
        self._show_level(self.master, tr("Master level"), client.LEVEL_UNITY)
        row.addStretch(1)
        self.setFixedHeight(26)

    def _level_knob(self, request) -> Knob:
        knob = Knob(20, neutral=client.LEVEL_UNITY)
        knob.moved.connect(request)
        return knob

    @staticmethod
    def _show_level(knob: Knob, title: str, value: float) -> None:
        knob.set_value(value)
        db = client.level_knob_to_db(knob.value())
        text = tr("-inf dB") if db <= client.LEVEL_DB_MIN else f"{db:+.1f} dB"
        knob.setToolTip(tr("{name}: {level} (double-click: 0 dB)").format(name=title, level=text))

    def toggle_cue(self) -> None:
        """The headphones CUE switched (a request; lit as the deck then reports)."""
        if self.isEnabled():
            self.bus.sig_channel_cue_requested.emit(not self._cue_on)

    def shortcut_handlers(self) -> dict:
        return {"mixer.headphones_cue": self.toggle_cue}

    def set_available(self, available: bool) -> None:
        self.setEnabled(available)
        if not available:
            self.setToolTip(tr("The jog wheel's mode, the EQ, the fader curve, the headphones and the master level: "
                               "the native audio engine's only (Settings > Audio > Engine)."))

    def set_report(self, report) -> None:
        """The deck's report (DeckReport): the jog's mode, the mixer's EQ and fader curve, the headphones CUE and the
        levels."""
        if not report.mix:
            return
        self._cue_on = bool(report.channel_cue)
        self.cue.set_lit(self._cue_on)
        for knob, key, title in ((self.phones, "hp_a_level", tr("Headphones level")),
                                 (self.master, "master_level", tr("Master level"))):
            value = report.mix_value(key, client.LEVEL_UNITY)
            if abs(value - knob.value()) > 1e-6:
                self._show_level(knob, title, value)
        self.vinyl.set_lit(report.vinyl)
        self.cdj.set_lit(not report.vinyl)
        eq = int(report.mix_value("eq_mode"))
        for k, b in enumerate(self.eq):
            b.set_lit(k == eq)
        curve = int(report.mix_value("fader_curve"))
        for k, b in enumerate(self.fader):
            b.set_lit(k == curve)
