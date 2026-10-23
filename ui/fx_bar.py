"""The mixer's effects in the toolbar, as a DJM-A9's (the native engine's mixer, src/mixer.rs): Sound Color FX (the
effect, the deck's COLOR knob, PARAMETER) and Beat FX on the master (the effect, its BEAT, the X-PAD, FX FREQUENCY
LOW / MID / HI, LEVEL / DEPTH, the BPM with TAP / AUTO, ON). Every control sends a request (sig_mixer_requested,
sig_knob_requested) and shows what the deck then reports (DeckReport.mix / color / beat_bpm), so a controller's
changes show too. Beat FX's channel is the master's alone (MST): a report of another is asked back to it. On the
Python engine (no mixer) the bar is disabled."""
from __future__ import annotations

import time

from PySide6 import QtCore, QtGui, QtWidgets

from audioengine.native import client
from core.event_bus import EventBus
from core.i18n import tr
from ui.knob import Knob
from ui.theme import current_theme

COLOR_KNOB = 4   # the deck's channel knob for its COLOR (player.set_knob)
TAP_GAP_SEC = 2.0   # taps further apart start anew (src/controller.rs TAP_GAP_SEC)
TAP_KEEP = 8
XPAD_W, XPAD_H = 180, 24
MOBIUS_WAVES = ("SQR", "SAW", "TRI")   # MOBIUS's waveforms across the X-PAD (left .. right)


def _small_label(text: str) -> QtWidgets.QLabel:
    label = QtWidgets.QLabel(text)
    f = label.font()
    f.setPointSize(7)
    f.setBold(True)
    label.setFont(f)
    label.setStyleSheet(f"color: {current_theme().text_dim};")
    return label


class _LitButton(QtWidgets.QToolButton):
    """A button lit as the report says (not toggled by its click: the click is a request)."""

    def __init__(self, text: str, parent=None) -> None:
        super().__init__(parent)
        self.setText(text)
        f = self.font()
        f.setPointSize(8)
        f.setBold(True)
        self.setFont(f)
        self.setFixedHeight(22)
        self.setMinimumWidth(30)
        theme = current_theme()
        self.setProperty("lit", False)
        self.setStyleSheet(
            f"QToolButton {{ border: 1px solid {theme.border}; background: {theme.raised}; color: {theme.text};"
            f" border-radius: 3px; padding: 0 4px; }}\n"
            f"QToolButton:hover {{ background: {theme.raised_hover}; }}\n"
            f"QToolButton[lit=true] {{ background: {theme.accent}; border-color: {theme.accent};"
            f" color: {theme.on_accent}; }}\n"
            f"QToolButton:disabled {{ color: {theme.text_disabled}; }}")

    def set_lit(self, lit: bool) -> None:
        if bool(self.property("lit")) != bool(lit):
            self.setProperty("lit", bool(lit))
            self.style().unpolish(self)
            self.style().polish(self)


class XPad(QtWidgets.QWidget):
    """The X-PAD: a strip split into the effect's BEAT values (MOBIUS: its waveforms; REVERB: its filter, from
    none at the left to the highest cutoff at the right); touched, the effect on while held, the place under the
    finger its BEAT (or waveform, filter). ``touched`` (True / False), ``moved`` (0 .. 1) while held."""

    touched = QtCore.Signal(bool)
    moved = QtCore.Signal(float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedSize(XPAD_W, XPAD_H)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self._labels: tuple[str, ...] = ()
        self._lit: int | None = None     # the segment the report has
        self._marker: float | None = None   # REVERB's filter as reported (0 .. 1)
        self._held: float | None = None   # the finger's place

    def set_state(self, labels: tuple[str, ...], lit: int | None, marker: float | None) -> None:
        if (labels, lit, marker) != (self._labels, self._lit, self._marker):
            self._labels, self._lit, self._marker = labels, lit, marker
            self.update()

    def _at(self, event) -> float:
        return min(1.0, max(0.0, event.position().x() / max(1.0, float(self.width()))))

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.LeftButton and self.isEnabled():
            self._held = self._at(event)
            self.touched.emit(True)
            self.moved.emit(self._held)
            self.update()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._held is not None:
            self._held = self._at(event)
            self.moved.emit(self._held)
            self.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._held is not None:
            self._held = None
            self.touched.emit(False)
            self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        theme = current_theme()
        enabled = self.isEnabled()
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        r = QtCore.QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        p.setPen(QtGui.QPen(QtGui.QColor(theme.border), 1))
        p.setBrush(QtGui.QColor(theme.base))
        p.drawRoundedRect(r, 3, 3)
        n = max(1, len(self._labels))
        w = r.width() / n
        f = p.font()
        f.setPointSize(7)
        p.setFont(f)
        accent = QtGui.QColor(theme.accent)
        # where the labels do not fit their cells, every n-th labelled (the lit one always)
        widest = max((p.fontMetrics().horizontalAdvance(t) for t in self._labels), default=0)
        every = max(1, -(-(widest + 4) // max(1, int(w))))
        for k, text in enumerate(self._labels):
            cell = QtCore.QRectF(r.left() + k * w, r.top(), w, r.height())
            if k == self._lit and enabled:
                fill = QtGui.QColor(accent)
                fill.setAlpha(110)
                p.fillRect(cell.adjusted(1, 1, -1, -1), fill)
            if k:
                p.setPen(QtGui.QPen(QtGui.QColor(theme.separator), 1))
                p.drawLine(cell.topLeft() + QtCore.QPointF(0, 3), cell.bottomLeft() - QtCore.QPointF(0, 3))
            if k % every and k != self._lit:
                continue
            p.setPen(QtGui.QColor(theme.text if enabled else theme.text_disabled))
            p.drawText(cell.adjusted(-w, 0, w, 0), QtCore.Qt.AlignCenter, text)
        if self._marker is not None and enabled:
            x = r.left() + self._marker * r.width()
            p.setPen(QtGui.QPen(accent, 2))
            p.drawLine(QtCore.QPointF(x, r.top() + 2), QtCore.QPointF(x, r.bottom() - 2))
        if self._held is not None:
            x = r.left() + self._held * r.width()
            p.setPen(QtGui.QPen(QtGui.QColor(theme.text), 2))
            p.drawLine(QtCore.QPointF(x, r.top() + 1), QtCore.QPointF(x, r.bottom() - 1))


class FxBar(QtWidgets.QWidget):
    """The toolbar's effects (see the module). ``set_report`` shows the deck's report; ``set_available`` on the
    native engine's mixer only."""

    def __init__(self, bus: EventBus, parent=None) -> None:
        super().__init__(parent)
        self.bus = bus
        self._report: client.DeckReport | None = None
        self._taps: list[float] = []
        self._xpad_turned_on = False
        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(6, 0, 6, 0)
        row.setSpacing(4)

        def knob(text: str, tip: str) -> Knob:
            k = Knob(22)
            k.setToolTip(tip)
            row.addWidget(_small_label(text), 0, QtCore.Qt.AlignVCenter)
            row.addWidget(k, 0, QtCore.Qt.AlignVCenter)
            return k

        # Sound Color FX
        row.addWidget(_small_label("SOUND COLOR FX"), 0, QtCore.Qt.AlignVCenter)
        self.color_type = QtWidgets.QComboBox()
        for code, name in enumerate(client.COLOR_TYPES):
            self.color_type.addItem(name, code)
        self.color_type.addItem("OFF", client.COLOR_OFF)
        self.color_type.setToolTip(tr("Sound Color FX: the effect the COLOR knob turns on (OFF: none)."))
        self.color_type.activated.connect(
            lambda _i: self._mix("color_type", self.color_type.currentData()))
        row.addWidget(self.color_type)
        self.color = knob("COLOR", tr("COLOR: the effect on the deck's channel, its two flavours left and right of "
                                      "the middle (off). Double-click: off."))
        self.color.moved.connect(lambda v: self.bus.sig_knob_requested.emit(COLOR_KNOB, v))
        self.param = knob("PARAMETER", tr("PARAMETER: the effect's quantity (SPACE / DUB ECHO: feedback, SWEEP: the "
                                          "centre frequency, NOISE: its volume, CRUSH: how crushed, FILTER: "
                                          "resonance, JET: feedback)."))
        self.param.moved.connect(lambda v: self._mix("color_param", v))
        row.addSpacing(6)
        rule = QtWidgets.QFrame()
        rule.setFixedSize(1, 22)
        rule.setStyleSheet(f"background: {current_theme().separator};")
        row.addWidget(rule, 0, QtCore.Qt.AlignVCenter)
        row.addSpacing(6)

        # Beat FX (on the master)
        row.addWidget(_small_label("BEAT FX  MST"), 0, QtCore.Qt.AlignVCenter)
        self.beat_type = QtWidgets.QComboBox()
        for code, name in enumerate(client.BEAT_TYPES):
            self.beat_type.addItem(name, code)
        self.beat_type.setToolTip(tr("Beat FX: the effect, on the master."))
        self.beat_type.activated.connect(lambda _i: self._select_beat_type(self.beat_type.currentData()))
        row.addWidget(self.beat_type)
        self.beat_value = QtWidgets.QComboBox()
        self.beat_value.setToolTip(tr("BEAT: the effect's time in beats of the BPM (REVERB: its level)."))
        self.beat_value.activated.connect(lambda _i: self._set_beat(self.beat_value.currentData()))
        row.addWidget(self.beat_value)
        self.xpad = XPad()
        self.xpad.setToolTip(tr("X-PAD: touched, the effect is on while held; the place under the finger is its "
                                "BEAT (MOBIUS: its waveform, REVERB: its filter)."))
        self.xpad.touched.connect(self._xpad_touched)
        self.xpad.moved.connect(self._xpad_moved)
        row.addWidget(self.xpad, 0, QtCore.Qt.AlignVCenter)
        self.bands = []
        for k, name in enumerate(("LOW", "MID", "HI")):
            b = _LitButton(name)
            b.setToolTip(tr("FX FREQUENCY {band}: the effect works on this band (lit), the others pass by.").format(
                band=name))
            b.clicked.connect(lambda _c=False, k=k: self.toggle_band(k))
            self.bands.append(b)
            row.addWidget(b, 0, QtCore.Qt.AlignVCenter)
        self.level = knob("LEVEL/DEPTH", tr("LEVEL / DEPTH: the balance of the sound and the effect (the effect's "
                                            "depth, MOBIUS's volume, HELIX's overlay, as the effect has it)."))
        self.level.moved.connect(lambda v: self._mix("beat_level", v))
        self.bpm = QtWidgets.QLabel()
        f = self.bpm.font()
        f.setPointSize(9)
        f.setBold(True)
        self.bpm.setFont(f)
        self.bpm.setFixedWidth(QtGui.QFontMetrics(f).horizontalAdvance("888.8 BPM") + 4)
        self.bpm.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.bpm.setToolTip(tr("The BPM Beat FX plays at: the deck's (AUTO) or the one tapped (TAP)."))
        row.addWidget(self.bpm, 0, QtCore.Qt.AlignVCenter)
        self.tap_button = _LitButton("TAP")
        self.tap_button.setToolTip(tr("TAP: tap more than twice in time with the beat to set the BPM (lit while "
                                      "tapped); BEAT goes to 1."))
        self.tap_button.clicked.connect(self.tap)
        row.addWidget(self.tap_button, 0, QtCore.Qt.AlignVCenter)
        self.auto_button = _LitButton("AUTO")
        self.auto_button.setToolTip(tr("AUTO: the deck's BPM again (lit while it is used)."))
        self.auto_button.clicked.connect(self.auto)
        row.addWidget(self.auto_button, 0, QtCore.Qt.AlignVCenter)
        self.on_button = _LitButton("ON")
        self.on_button.setMinimumWidth(40)
        self.on_button.setToolTip(tr("Beat FX on / off (lit while on)."))
        self.on_button.clicked.connect(self.toggle_on)
        row.addWidget(self.on_button, 0, QtCore.Qt.AlignVCenter)
        self._show_beat_values(client.BEAT_TYPES.index("ECHO"))
        for combo in (self.color_type, self.beat_type):
            combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        longest = max((client.beat_label(k, v) for k, values in enumerate(client.BEAT_VALUES) for v in values),
                      key=self.beat_value.fontMetrics().horizontalAdvance)
        self.beat_value.setMinimumContentsLength(len(longest))
        self.beat_value.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.set_available(True)

    # Requests
    def _mix(self, key: str, value) -> None:
        if value is not None:
            self.bus.sig_mixer_requested.emit(key, float(value))

    def _value(self, key: str, default: float = 0.0) -> float:
        return self._report.mix_value(key, default) if self._report is not None and self._report.mix else default

    def _kind(self) -> int:
        return int(self._value("beat_type", 1.0))

    def _select_beat_type(self, kind) -> None:
        """An effect picked: its BEAT at its default, the TIME knob's time and the X-PAD let go (as the engine's
        controller picks one)."""
        if kind is None:
            return
        kind = int(kind)
        self._mix("beat_type", kind)
        self._mix("beat_beat", client.BEAT_DEFAULTS[kind])
        self._mix("beat_time", 0.0)
        self._mix("beat_xpad", -1.0)

    def _set_beat(self, value) -> None:
        if value is not None:
            self._mix("beat_beat", value)
            self._mix("beat_time", 0.0)

    def toggle_on(self) -> None:
        self._mix("beat_on", 0.0 if self._value("beat_on") else 1.0)

    def toggle_band(self, band: int) -> None:
        self._mix("beat_bands", int(self._value("beat_bands", 7.0)) ^ (1 << band))

    def step_beat_type(self, by: int) -> None:
        self._select_beat_type((self._kind() + by) % len(client.BEAT_TYPES))

    def step_beat(self, by: int) -> None:
        """The effect's next / previous BEAT value (held at the ends)."""
        kind = self._kind()
        values = client.BEAT_VALUES[kind]
        now = client.snap_beat(kind, self._value("beat_beat", 1.0))
        at = min(range(len(values)), key=lambda k: abs(values[k] - now))
        self._set_beat(values[min(len(values) - 1, max(0, at + by))])

    def step_color_type(self, by: int) -> None:
        """The next / previous Sound Color FX, OFF among them."""
        codes = list(range(len(client.COLOR_TYPES))) + [client.COLOR_OFF]
        now = int(self._value("color_type", 5.0))
        at = codes.index(now) if now in codes else 0
        self._mix("color_type", codes[(at + by) % len(codes)])

    def tap(self) -> None:
        """TAP: the BPM from the taps (more than two, none further apart than TAP_GAP_SEC), BEAT then 1 (the
        manual's; REVERB's level kept)."""
        now = time.monotonic()
        if self._taps and now - self._taps[-1] > TAP_GAP_SEC:
            self._taps.clear()
        self._taps = (self._taps + [now])[-TAP_KEEP:]
        if len(self._taps) >= 3:
            bpm = 60.0 * (len(self._taps) - 1) / max(1e-3, self._taps[-1] - self._taps[0])
            self._mix("beat_bpm", bpm)
            if self._kind() != client.BEAT_REVERB:
                self._mix("beat_beat", 1.0)
            self._mix("beat_time", 0.0)

    def auto(self) -> None:
        self._taps.clear()
        self._mix("beat_bpm", 0.0)

    def _xpad_touched(self, down: bool) -> None:
        if down and not self._value("beat_on"):
            self._mix("beat_on", 1.0)
            self._xpad_turned_on = True
        elif not down:
            if self._xpad_turned_on:
                self._mix("beat_on", 0.0)
                self._xpad_turned_on = False
            if self._kind() == client.BEAT_REVERB:
                self._mix("beat_xpad", -1.0)   # its filter off with the finger

    def _xpad_moved(self, place: float) -> None:
        kind = self._kind()
        if kind in (client.BEAT_REVERB, client.BEAT_MOBIUS):
            self._mix("beat_xpad", place)
        else:
            values = client.BEAT_VALUES[kind]
            self._set_beat(values[min(len(values) - 1, int(place * len(values)))])

    # The report
    def set_available(self, available: bool) -> None:
        """On the native engine's mixer only (the Python engine has none)."""
        self.setEnabled(available)
        if not available:
            self.setToolTip(tr("Sound Color FX and Beat FX: the native audio engine's mixer only (Settings > "
                               "Audio > Engine)."))

    def _show_beat_values(self, kind: int) -> None:
        if self.beat_value.property("kind") == kind:
            return
        self.beat_value.setProperty("kind", kind)
        self.beat_value.clear()
        for v in client.BEAT_VALUES[kind]:
            self.beat_value.addItem(client.beat_label(kind, v), v)

    @staticmethod
    def _show_combo(combo: QtWidgets.QComboBox, data) -> None:
        if combo.view().isVisible():
            return   # open: the hand is on it
        k = combo.findData(data)
        if k >= 0 and k != combo.currentIndex():
            combo.setCurrentIndex(k)

    def set_report(self, report) -> None:
        """The deck's report (DeckReport): the mixer's settings, the COLOR knob, the BPM."""
        if not report.mix:
            return
        self._report = report
        v = report.mix_value
        if int(v("beat_channel", client.BEAT_MASTER)) != client.BEAT_MASTER:
            self._mix("beat_channel", client.BEAT_MASTER)   # Beat FX on the master alone
        self._show_combo(self.color_type, int(v("color_type")))
        self.color.set_value(report.color)
        self.param.set_value(v("color_param", 0.5))
        kind = int(v("beat_type", 1.0))
        self._show_combo(self.beat_type, kind)
        self._show_beat_values(kind)
        beat = client.snap_beat(kind, v("beat_beat", 1.0))
        values = client.BEAT_VALUES[kind]
        timed = v("beat_time") != 0.0 and kind != client.BEAT_REVERB
        lit = None if timed else min(range(len(values)), key=lambda k: abs(values[k] - beat))
        if kind == client.BEAT_REVERB:
            self._show_combo(self.beat_value, values[lit])
            xpad = v("beat_xpad", -1.0)
            self.xpad.set_state(("HPF",), None, xpad if xpad >= 0.0 else None)
        elif kind == client.BEAT_MOBIUS:
            self._show_combo(self.beat_value, beat)
            xpad = v("beat_xpad", -1.0)
            wave = 1 if xpad < 0.0 else min(2, int(xpad * 3))
            self.xpad.set_state(MOBIUS_WAVES, wave, None)
        else:
            if lit is not None:
                self._show_combo(self.beat_value, values[lit])
            self.xpad.set_state(tuple(client.beat_label(kind, b) for b in values), lit, None)
        bands = int(v("beat_bands", 7.0))
        for k, b in enumerate(self.bands):
            b.set_lit(bool(bands & (1 << k)))
        self.level.set_value(v("beat_level", 0.5))
        tapped = v("beat_bpm") > 0.0
        self.tap_button.set_lit(tapped)
        self.auto_button.set_lit(not tapped)
        self.bpm.setText(f"{report.beat_bpm:.1f} BPM" if report.beat_bpm > 0.0 else "--- BPM")
        self.on_button.set_lit(bool(v("beat_on")))

    # Keyboard shortcuts (ui.shortcuts)
    def shortcut_handlers(self) -> dict:
        return {
            "fx.beat_on": self.toggle_on,
            "fx.beat_tap": self.tap,
            "fx.beat_auto": self.auto,
            "fx.beat_next": lambda: self.step_beat_type(1),
            "fx.beat_previous": lambda: self.step_beat_type(-1),
            "fx.beat_up": lambda: self.step_beat(1),
            "fx.beat_down": lambda: self.step_beat(-1),
            "fx.band_low": lambda: self.toggle_band(0),
            "fx.band_mid": lambda: self.toggle_band(1),
            "fx.band_hi": lambda: self.toggle_band(2),
            "fx.color_next": lambda: self.step_color_type(1),
            "fx.color_previous": lambda: self.step_color_type(-1),
        }
