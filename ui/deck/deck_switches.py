"""The deck's and its mixer channel's switches, the toolbar's CH 1 row, as a DJM's and a CDJ's are: the jog wheel's
mode (VINYL: its top holds the record, CDJ: a bend only), the EQ (ISO: an isolator, a band down to a kill; EQ:
-26 dB at the bottom), the channel fader's curve (each drawn: the app's default, steep: up fast, slow: most of the
rise at the top), Sound Color FX (the effect, the channel's COLOR knob, PARAMETER) and the channel's headphones CUE;
``MasterLevels`` the MASTER row's levels (the master, the headphones). A click or a turn sends a request
(sig_jog_mode_requested, sig_mixer_requested, sig_knob_requested, sig_channel_cue_requested); what is lit and where a
knob points is what the deck reports (DeckReport: its lamps' VINYL bit, channel_cue, color, the mixer's values), so
a controller's switch or knob shows too. The native engine's (the Python engine has none of them): disabled
there. The controls' looks: ui/deck/deck_controls.py."""
from __future__ import annotations

import math

from PySide6 import QtCore, QtGui, QtWidgets

from audioengine import controls
from core.workflows.event_bus import EventBus
from system.i18n import tr
from ui.deck.deck_controls import KNOB, Lamp, Segment, block, blocks_row, label, segments
from ui.widgets.knob import Knob
from ui.theme import current_theme

GLYPH_W, GLYPH_H = 26, 14
COLOR_KNOB = 4   # the deck's channel knob for its COLOR (player.set_knob)


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
        else:   # the app's default (audioengine/volume.py; src/mixer.rs fader_gain)
            v = x ** 0.4
            y = 1.0 if v > 0.99 else -math.log(1 - v) / math.log(100)
        point = QtCore.QPointF(2 + x * (GLYPH_W - 4), GLYPH_H - 2 - y * (GLYPH_H - 4))
        path.moveTo(point) if k == 0 else path.lineTo(point)
    p.drawPath(path)
    p.end()
    return QtGui.QIcon(pix)


class DeckSwitches(QtWidgets.QWidget):
    """The CH 1 row's switches (see the module). ``set_report`` shows the deck's report; ``set_available`` on the
    native engine."""

    def __init__(self, bus: EventBus, parent=None) -> None:
        super().__init__(parent)
        self.bus = bus
        self._report = None
        self._cue_on = False

        self.vinyl, self.cdj = Segment("VINYL"), Segment("CDJ")
        self.vinyl.setToolTip(tr("JOG MODE VINYL: touching the jog wheel's top holds the record (scratch)."))
        self.cdj.setToolTip(tr("JOG MODE CDJ: the jog wheel bends the tempo only, its top's touch ignored."))
        self.vinyl.clicked.connect(lambda: self.bus.sig_jog_mode_requested.emit(True))
        self.cdj.clicked.connect(lambda: self.bus.sig_jog_mode_requested.emit(False))

        self.eq = [Segment("ISO"), Segment("EQ")]
        self.eq[0].setToolTip(tr("EQ CURVE ISO: an isolator, each band from a kill to +6 dB."))
        self.eq[1].setToolTip(tr("EQ CURVE EQ: an equalizer, each band from -26 dB to +6 dB."))
        for k, b in enumerate(self.eq):
            b.clicked.connect(lambda _c=False, k=k: self.bus.sig_mixer_requested.emit("eq_mode", float(k)))

        self.fader = []
        tips = (tr("CH FADER curve: the app's (slow at first, full at the top)."),
                tr("CH FADER curve: steep (up fast)."),
                tr("CH FADER curve: slow (most of the rise at the top)."))
        for k, tip in enumerate(tips):
            b = Segment()
            b.setIcon(_curve_icon(k, current_theme().text))
            b.setIconSize(QtCore.QSize(GLYPH_W, GLYPH_H))
            b.setToolTip(tip)
            b.clicked.connect(lambda _c=False, k=k: self.bus.sig_mixer_requested.emit("fader_curve", float(k)))
            self.fader.append(b)

        # Sound Color FX: the effect, the channel's COLOR knob, PARAMETER
        self.color_type = QtWidgets.QComboBox()
        for code, name in enumerate(controls.COLOR_TYPES):
            self.color_type.addItem(name, code)
        self.color_type.addItem("OFF", controls.COLOR_OFF)
        self.color_type.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.color_type.setToolTip(tr("Sound Color FX: the effect the COLOR knob turns on (OFF: none)."))
        self.color_type.activated.connect(lambda _i: self._mix("color_type", self.color_type.currentData()))
        self.color = Knob(KNOB)
        self.color.setToolTip(tr("COLOR: the effect on the deck's channel, its two flavours left and right of the "
                                 "middle (off). Double-click: off."))
        self.color.moved.connect(lambda v: self.bus.sig_knob_requested.emit(COLOR_KNOB, v))
        self.param = Knob(KNOB)
        self.param.setToolTip(tr("PARAMETER: the effect's quantity."))
        self.param.moved.connect(lambda v: self._mix("color_param", v))

        # the channel's headphones CUE (the headphones' level: MasterLevels)
        self.cue = Lamp("CUE", current_theme().warning)
        self.cue.setToolTip(tr("Headphones CUE: the deck's channel, before its fader, in the headphones."))
        self.cue.clicked.connect(self.toggle_cue)
        blocks_row(self, [
            block("JOG MODE", [segments([self.vinyl, self.cdj])]),
            block("EQ CURVE", [segments(self.eq)]),
            block("CH FADER", [segments(self.fader)]),
            block("COLOR FX", [self.color_type, 8, label("COLOR"), self.color, 8, label("PARAMETER"), self.param]),
            block("PHONES", [self.cue]),
        ])

    def _mix(self, key: str, value) -> None:
        if value is not None:
            self.bus.sig_mixer_requested.emit(key, float(value))

    def step_color_type(self, by: int) -> None:
        """The next / previous Sound Color FX, OFF among them."""
        codes = list(range(len(controls.COLOR_TYPES))) + [controls.COLOR_OFF]
        report = self._report
        now = int(report.mix_value("color_type", 5.0)) if report is not None and report.mix else 5
        at = codes.index(now) if now in codes else 0
        self._mix("color_type", codes[(at + by) % len(codes)])

    def toggle_cue(self) -> None:
        """The headphones CUE switched (a request; lit as the deck then reports)."""
        if self.isEnabled():
            self.bus.sig_channel_cue_requested.emit(not self._cue_on)

    def shortcut_handlers(self) -> dict:
        return {"mixer.headphones_cue": self.toggle_cue,
                "fx.color_next": lambda: self.step_color_type(1),
                "fx.color_previous": lambda: self.step_color_type(-1)}

    def set_available(self, available: bool) -> None:
        self.setEnabled(available)
        if not available:
            self.setToolTip(tr("Available with the native audio engine only."))

    def set_report(self, report) -> None:
        """The deck's report (DeckReport): the jog's mode, the mixer's EQ, Sound Color FX and fader curve, the
        headphones CUE."""
        if not report.mix:
            return
        self._report = report
        if not self.color_type.view().isVisible():   # open: the hand is on it
            k = self.color_type.findData(int(report.mix_value("color_type")))
            if k >= 0 and k != self.color_type.currentIndex():
                self.color_type.setCurrentIndex(k)
        self.color.set_value(report.color)
        self.param.set_value(report.mix_value("color_param", 0.5))
        self._cue_on = bool(report.channel_cue)
        self.cue.set_lit(self._cue_on)
        self.vinyl.set_lit(report.vinyl)
        self.cdj.set_lit(not report.vinyl)
        eq = int(report.mix_value("eq_mode"))
        for k, b in enumerate(self.eq):
            b.set_lit(k == eq)
        curve = int(report.mix_value("fader_curve"))
        for k, b in enumerate(self.fader):
            b.set_lit(k == curve)


class MasterLevels(QtWidgets.QWidget):
    """The MASTER row's levels: the master's and the headphones' (a knob each, its middle 0 dB)."""

    def __init__(self, bus: EventBus, parent=None) -> None:
        super().__init__(parent)
        self.bus = bus
        self.master = self._level_knob(lambda v: self.bus.sig_mixer_requested.emit("master_level", v))
        self.phones = self._level_knob(lambda v: self.bus.sig_mixer_requested.emit("hp_level", v))
        blocks_row(self, [block("LEVEL", [label("MASTER"), self.master, 10, label("PHONES"), self.phones])])
        self._show_level(self.master, tr("Master level"), controls.LEVEL_UNITY)
        self._show_level(self.phones, tr("Headphones level"), controls.LEVEL_UNITY)

    def _level_knob(self, request) -> Knob:
        knob = Knob(KNOB, neutral=controls.LEVEL_UNITY)
        knob.moved.connect(request)
        return knob

    @staticmethod
    def _show_level(knob: Knob, title: str, value: float) -> None:
        knob.set_value(value)
        db = controls.level_knob_to_db(knob.value())
        text = tr("-inf dB") if db <= controls.LEVEL_DB_MIN else f"{db:+.1f} dB"
        knob.setToolTip(tr("{name}: {level} (double-click: 0 dB)").format(name=title, level=text))

    def set_available(self, available: bool) -> None:
        self.setEnabled(available)

    def set_report(self, report) -> None:
        if not report.mix:
            return
        for knob, key, title in ((self.master, "master_level", tr("Master level")),
                                 (self.phones, "hp_level", tr("Headphones level"))):
            value = report.mix_value(key, controls.LEVEL_UNITY)
            if abs(value - knob.value()) > 1e-6:
                self._show_level(knob, title, value)
