from __future__ import annotations

import html
import math
from typing import Optional

from PySide6 import QtWidgets, QtGui, QtCore

from ui.theme import current_theme
from core.settings.config import config
from analyzer.music.key_names import ALPHANUMERIC_LABELS, CLASSICAL_LABELS
from system.i18n import tr
from system.resource_paths import setting_path
from ui.widgets.knob import Knob


_VALUE = "#f0f3f6"   # current values, title
_ARTIST = "#aab3bc"
_DIM = "#8b949e"     # original values, meter, total time
_UP = "#5fd38d"
_DOWN = "#ff7b72"
_BEAT_ON = "#3d8bf0"
_BEAT_IN_BAR = "#e6e6e6"
_BEAT_OFF = "#4a4a4a"
_GAP = " "  # en space between the change and the meter
_THIN = " "  # thin space around the "/" of a key: narrower key column
KEY_LABEL_SAMPLES = [f"{num}{_THIN}/{_THIN}{cls}" for num, cls in zip(ALPHANUMERIC_LABELS, CLASSICAL_LABELS)]


def _span(text: str, color: str) -> str:
    return f'<span style="color:{color}">{html.escape(text)}</span>'


def _bpm_text(bpm: float) -> str:
    """Six characters at most: 999.99 below 1000 BPM, then one decimal up to 9999.9."""
    bpm = min(max(float(bpm), 0.0), 9999.9)
    return f"{bpm:.2f}" if bpm < 999.995 else f"{bpm:.1f}"


def _signed(value: float, fmt: str) -> str:
    """Signed, zero-padded number whose width never changes: "-" becomes the
    minus sign U+2212, as wide as "+" and the digits (a hyphen is narrower)."""
    return format(float(value), fmt).replace("-", "−")


def _pp_text(pp: float) -> str:
    """+002.00pp, −050.00pp"""
    return _signed(max(-999.99, min(999.99, float(pp))), "+07.2f") + "pp"


def _signed_color(value: float) -> str:
    return _UP if value > 0 else _DOWN if value < 0 else _DIM


def _clock(sec: float) -> str:
    sec = max(0.0, float(sec))
    m = int(sec // 60)
    s = int(sec - 60 * m)
    cs = int((sec - 60 * m - s) * 100)
    return f"{m:02d}:{s:02d}.{cs:02d}"


class MarqueeLabel(QtWidgets.QLabel):
    """QLabel variant that scrolls long text after a short pause."""

    HOLD_MS = 5000
    STEP_MS = 200
    PAD = "   "

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(self.STEP_MS)
        self._timer.timeout.connect(self._on_tick)
        self._full_text = ""
        self._buffer = ""
        self._scroll_idx = 0
        self._needs_scroll = False
        self._hold_ticks = 0
        self._pending_geometry_check = False

    def set_marquee_text(self, text: str):
        self._full_text = text or ""
        self._buffer = self._full_text + self.PAD
        self._scroll_idx = 0
        self._reset_hold()
        self.setText(self._full_text)
        self._update_scroll_state(force=True)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_scroll_state()

    def _update_scroll_state(self, force: bool = False):
        if not self._full_text:
            self._timer.stop()
            return
        width = self.contentsRect().width()
        if width <= 0:
            if not self._pending_geometry_check:
                self._pending_geometry_check = True
                QtCore.QTimer.singleShot(
                    0, lambda: self._reset_geometry_flag_and_update()
                )
            return
        fm = self.fontMetrics()
        needs = fm.horizontalAdvance(self._full_text) > width
        changed = (needs != self._needs_scroll) or force
        self._needs_scroll = needs
        if not needs:
            self._timer.stop()
            self.setText(self._full_text)
            return
        if changed:
            self._reset_hold()
            self._scroll_idx = 0
            self.setText(self._full_text)
        if not self._timer.isActive():
            self._timer.start()

    def _on_tick(self):
        if not (self._needs_scroll and self._buffer):
            return
        if self._hold_ticks > 0:
            self._hold_ticks -= 1
            return
        self._scroll_idx = (self._scroll_idx + 1) % len(self._buffer)
        if self._scroll_idx == 0:
            self.setText(self._full_text)
            self._reset_hold()
            return
        rotated = self._buffer[self._scroll_idx :] + self._buffer[: self._scroll_idx]
        self.setText(rotated)

    def _reset_geometry_flag_and_update(self):
        self._pending_geometry_check = False
        self._update_scroll_state()

    def _reset_hold(self):
        self._hold_ticks = max(0, math.ceil(self.HOLD_MS / self.STEP_MS))


class VerticalPeakMeter(QtWidgets.QWidget):
    """The channel's level meter as a DJM-A9's channel level indicator: 15 segments, -30 to +12 dB in 3 dB steps
    (a segment lit at its level and above), its 0 dB at REFERENCE_DBFS: -12 dBFS (one of the A9's DIGITAL MASTER OUT
    reference levels), so +12, the top, is 0 dBFS: the output's full scale, where it clips. The peak held 0.5 s."""
    REFERENCE_DBFS = -12.0
    SEGMENTS_DB = tuple(range(-30, 13, 3))
    FLOOR_DBFS = REFERENCE_DBFS + SEGMENTS_DB[0] - 3.0   # nothing lit

    def __init__(self, parent=None):
        super().__init__(parent)
        self._display_dbfs = self.FLOOR_DBFS
        self._target_dbfs = self.FLOOR_DBFS
        self._peak_hold_dbfs = self.FLOOR_DBFS
        self._hold_ms = 500
        self._fall_db_per_sec = 24.0
        self._peak_fall_db_per_sec = 36.0
        self._peak_hold_deadline = QtCore.QDeadlineTimer(self._hold_ms)
        self._anim = QtCore.QTimer(self)
        self._anim.setInterval(16)
        self._anim.timeout.connect(self._tick)
        self.setFixedWidth(10)
        self.setMinimumHeight(TrackInfoPanel.HEADER_H - 6)
        self.setMaximumHeight(TrackInfoPanel.HEADER_H - 6)
        self.setToolTip(tr("Channel level (before the fader): -30 to +12 dB."))
        self._anim.start()

    def set_level_dbfs(self, dbfs: float) -> None:
        value = max(self.FLOOR_DBFS, min(0.0, float(dbfs)))   # the engine reports 0 dBFS at most
        self._target_dbfs = value
        if value >= self._display_dbfs:
            self._display_dbfs = value
        if value >= self._peak_hold_dbfs:
            self._peak_hold_dbfs = value
            self._peak_hold_deadline = QtCore.QDeadlineTimer(self._hold_ms)
        self.update()

    def _tick(self) -> None:
        dt = self._anim.interval() / 1000.0
        changed = False
        if self._display_dbfs > self._target_dbfs:
            next_dbfs = max(self._target_dbfs, self._display_dbfs - (self._fall_db_per_sec * dt))
            if abs(next_dbfs - self._display_dbfs) > 1e-6:
                self._display_dbfs = next_dbfs
                changed = True
        if self._peak_hold_deadline.hasExpired():
            next_peak = max(self._display_dbfs, self._peak_hold_dbfs - (self._peak_fall_db_per_sec * dt))
            if abs(next_peak - self._peak_hold_dbfs) > 1e-6:
                self._peak_hold_dbfs = next_peak
                changed = True
        if changed:
            self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        rect = QtCore.QRectF(self.rect()).adjusted(2, 2, -2, -2)
        painter.fillRect(rect, QtGui.QColor("#111"))
        level = self._display_dbfs - self.REFERENCE_DBFS
        held = self._peak_hold_dbfs - self.REFERENCE_DBFS
        n = len(self.SEGMENTS_DB)
        pitch = (rect.height() - 1) / n
        for k, db in enumerate(self.SEGMENTS_DB):
            # green below 0 dB, amber 0 to +9, red at +12 (clipping)
            color = QtGui.QColor("#ff4d4d" if db >= 12 else "#ffb020" if db >= 0 else "#34d399")
            lit = level >= db - 1e-6 or db - 1e-6 <= held < db + 3.0   # the level, and the peak held
            if not lit:
                color.setAlpha(38)
            seg = QtCore.QRectF(rect.left() + 1, rect.bottom() - (k + 1) * pitch + 1, rect.width() - 2, pitch - 1)
            painter.fillRect(seg, color)
        painter.end()


class HorizontalBufferMeter(QtWidgets.QWidget):
    """The peak meter's bar turned sideways, for how full the output buffer is.

    Colours run the other way (empty = red, full = green), and the optional hold marker
    keeps the lowest recent level instead of the highest.
    """

    def __init__(self, parent=None, hold_marker: bool = True):
        super().__init__(parent)
        self._hold_marker = bool(hold_marker)
        self._ratio = 0.0
        self._low_hold = 1.0
        self._hold_ms = 1000
        self._rise_per_sec = 0.5  # hold marker returns to the bar at half the scale per second
        self._low_hold_deadline = QtCore.QDeadlineTimer(self._hold_ms)
        self._anim = QtCore.QTimer(self)
        self._anim.setInterval(16)
        self._anim.timeout.connect(self._tick)
        self.setFixedHeight(14)
        self.setMinimumWidth(160)

    def set_level(self, ratio: float, lowest_ratio: float | None = None) -> None:
        """``ratio``: filled part now (0..1); ``lowest_ratio``: lowest since the last call."""
        self._ratio = max(0.0, min(1.0, float(ratio)))
        low = self._ratio if lowest_ratio is None else min(self._ratio, max(0.0, float(lowest_ratio)))
        if low <= self._low_hold:
            self._low_hold = low
            self._low_hold_deadline = QtCore.QDeadlineTimer(self._hold_ms)
        self.update()

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        if self._hold_marker:
            self._anim.start()

    def hideEvent(self, event: QtGui.QHideEvent) -> None:
        super().hideEvent(event)
        self._anim.stop()

    def _tick(self) -> None:
        if self._low_hold_deadline.hasExpired() and self._low_hold < self._ratio:
            dt = self._anim.interval() / 1000.0
            self._low_hold = min(self._ratio, self._low_hold + self._rise_per_sec * dt)
            self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        rect = self.rect().adjusted(2, 3, -2, -3)
        painter.fillRect(rect, QtGui.QColor("#111"))
        painter.setPen(QtGui.QPen(QtGui.QColor("#444"), 1))
        painter.drawRect(rect)

        fill_w = int(round(rect.width() * self._ratio))
        if fill_w > 0:
            fill_rect = QtCore.QRect(rect.left() + 1, rect.top() + 1, fill_w - 1, rect.height() - 1)
            grad_rect = rect.adjusted(1, 1, -1, -1)
            grad = QtGui.QLinearGradient(grad_rect.topLeft(), grad_rect.topRight())
            grad.setColorAt(0.0, QtGui.QColor("#ff4d4d"))
            grad.setColorAt(0.35, QtGui.QColor("#ffcf40"))
            grad.setColorAt(1.0, QtGui.QColor("#34d399"))
            painter.save()
            painter.setClipRect(fill_rect)
            painter.fillRect(grad_rect, grad)
            painter.restore()

        if not self._hold_marker:
            painter.end()
            return
        low_x = rect.left() + int(round(rect.width() * self._low_hold))
        painter.setPen(QtGui.QPen(QtGui.QColor("#f8fafc"), 2))
        painter.drawLine(low_x, rect.top() + 1, low_x, rect.bottom() - 1)
        painter.end()


class BeatDots(QtWidgets.QWidget):
    """2 x 4 short bars under the record: the bar's beats (white up to the meter,
    grey after it) with the current beat lit blue."""

    COLS, ROWS = 4, 2
    BAR_W, BAR_H, GAP_X, GAP_Y = 12, 2, 5, 3

    def __init__(self, parent=None):
        super().__init__(parent)
        self._meter = 4
        self._current: Optional[int] = None
        self.setFixedSize(self.COLS * self.BAR_W + (self.COLS - 1) * self.GAP_X,
                          self.ROWS * self.BAR_H + (self.ROWS - 1) * self.GAP_Y)

    def set_beat(self, beat_in_bar: Optional[int], meter: int = 4) -> None:
        """``beat_in_bar`` 0-based (None: no beat); repaints only when it changes."""
        meter = max(1, int(meter))
        if beat_in_bar is not None:
            beat_in_bar = int(beat_in_bar) % (self.COLS * self.ROWS)
        if (beat_in_bar, meter) == (self._current, self._meter):
            return
        self._current, self._meter = beat_in_bar, meter
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        p.setPen(QtCore.Qt.NoPen)
        for k in range(self.COLS * self.ROWS):
            if k == self._current:
                color = _BEAT_ON
            elif k < self._meter:
                color = _BEAT_IN_BAR
            else:
                color = _BEAT_OFF
            p.setBrush(QtGui.QColor(color))
            x = (k % self.COLS) * (self.BAR_W + self.GAP_X)
            y = (k // self.COLS) * (self.BAR_H + self.GAP_Y)
            p.drawRoundedRect(QtCore.QRectF(x, y, self.BAR_W, self.BAR_H), 1.0, 1.0)
        p.end()


def _vrule() -> QtWidgets.QFrame:
    """Thin vertical rule between header groups."""
    rule = QtWidgets.QFrame()
    rule.setFixedSize(1, TrackInfoPanel.HEADER_H - 24)
    rule.setStyleSheet(f"background: {current_theme().separator};")
    return rule


def _gap() -> QtWidgets.QWidget:
    """Empty space that grows with the header up to GAP_MAX."""
    gap = QtWidgets.QWidget()
    gap.setMaximumWidth(GAP_MAX)
    gap.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
    return gap


GAP_MAX = 32


class _UnloadButton(QtWidgets.QAbstractButton):
    """Small square over the middle of the album art: invisible until hovered,
    then an eject icon on a dark tile; clicked, the track is unloaded."""

    SIZE = 40

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(self.SIZE, self.SIZE)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setToolTip(tr("Unload track"))
        self.setAttribute(QtCore.Qt.WA_Hover, True)

    def enterEvent(self, event) -> None:
        super().enterEvent(event)
        self.update()

    def leaveEvent(self, event) -> None:
        super().leaveEvent(event)
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        if not self.underMouse():
            return
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        r = QtCore.QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 90), 1))
        p.setBrush(QtGui.QColor(0, 0, 0, 230 if self.isDown() else 170))
        p.drawRoundedRect(r, 6, 6)
        # eject: a triangle over a bar
        s = float(self.SIZE)
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(QtGui.QColor("#f0f3f6"))
        p.drawPolygon(QtGui.QPolygonF([
            QtCore.QPointF(s * 0.5, s * 0.24),
            QtCore.QPointF(s * 0.76, s * 0.54),
            QtCore.QPointF(s * 0.24, s * 0.54),
        ]))
        p.drawRect(QtCore.QRectF(s * 0.24, s * 0.62, s * 0.52, s * 0.12))
        p.end()


# the player's channel knobs right of its peak meter, top to bottom as a DJ mixer's: (label, the player's knob:
# 0 TRIM, 1 EQ LOW, 2 MID, 3 HIGH)
CHANNEL_KNOBS = (("TRIM", 0), ("HI", 3), ("MID", 2), ("LOW", 1))
KNOB_SIZE = 18


def _knob_label(text: str) -> QtWidgets.QLabel:
    label = QtWidgets.QLabel(text)
    f = label.font()
    f.setPointSize(7)
    f.setBold(True)
    label.setFont(f)
    label.setAlignment(QtCore.Qt.AlignHCenter)
    label.setStyleSheet(f"color: {current_theme().text_dim};")
    label.setFixedHeight(QtGui.QFontMetrics(f).height())
    return label


class TrackInfoPanel(QtWidgets.QWidget):
    """Header widget that displays central track information."""

    # the player's channel knob turned by the hand: (0 TRIM, 1 EQ LOW, 2 MID, 3 HIGH; its position 0 .. 1)
    knob_moved = QtCore.Signal(int, float)

    HEADER_H = 120
    # record diameter: the header minus the MT / RST row above and the beat bars below
    RECORD_D = 83

    def __init__(self, cfg: Optional[config] = None, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self._record_rotation_cache: dict[int, QtGui.QPixmap] = {}
        self._record_last_cache_key: Optional[int] = None
        self._record_cache_step_deg = 2  # rotate in coarse steps to limit redraw work
        self._record_base_pix = QtGui.QPixmap()
        self._record_base_dpr = 1.0
        self._current_time = 0.0
        self._last_album_art: Optional[QtGui.QImage] = None

        self._init_ui()
        self._set_placeholder_cover()
        self._load_record_pixmap(None)

    def _init_ui(self):
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        layout.setAlignment(QtCore.Qt.AlignVCenter)

        self._build_cover(layout)
        layout.addWidget(self._build_info_area(), 1, QtCore.Qt.AlignVCenter)

        # the edit panel (filled in by the main pane) between the track info and
        # the deck, set off by thin rules
        self.editor_slot = QtWidgets.QHBoxLayout()
        self.editor_slot.setContentsMargins(0, 0, 0, 0)
        # gaps either side of each rule grow with the window, evenly and only up
        # to GAP_MAX; the info area takes the rest
        for item in (_gap(), _vrule(), _gap(), self.editor_slot, _gap(), _vrule(), _gap()):
            if isinstance(item, QtWidgets.QLayout):
                layout.addLayout(item)
            elif isinstance(item, QtWidgets.QFrame):
                layout.addWidget(item, 0, QtCore.Qt.AlignVCenter)
            else:
                layout.addWidget(item, 1)

        # round stop / play in a column left of the record, filled in by the
        # main pane
        self.transport_slot = QtWidgets.QVBoxLayout()
        self.transport_slot.setContentsMargins(0, 0, 0, 0)
        self.transport_slot.setSpacing(4)
        self.transport_slot.setAlignment(QtCore.Qt.AlignBottom)
        layout.addLayout(self.transport_slot)

        layout.addLayout(self._build_record_column())

        self.peak_meter = VerticalPeakMeter()
        layout.addWidget(self.peak_meter, 0, QtCore.Qt.AlignVCenter)

        layout.addLayout(self._build_knob_column())

    def _build_cover(self, layout: QtWidgets.QHBoxLayout) -> None:
        """The album art, and over its middle the eject button shown while hovered (unload)."""
        self.cover_label = QtWidgets.QLabel()
        self.cover_label.setFixedSize(self.HEADER_H, self.HEADER_H)
        self.cover_label.setAlignment(QtCore.Qt.AlignCenter)
        self.cover_label.setScaledContents(False)
        self.cover_label.setStyleSheet("background:#1c1c1c; border:1px solid #333; border-radius:6px; padding:3px;")
        layout.addWidget(self.cover_label, 0, QtCore.Qt.AlignVCenter)
        # unload: hovering the middle of the album art shows an eject button
        self.unload_button = _UnloadButton(self.cover_label)
        self.unload_button.move((self.HEADER_H - _UnloadButton.SIZE) // 2, (self.HEADER_H - _UnloadButton.SIZE) // 2)
        self.unload_button.setVisible(False)

    def _build_info_area(self) -> QtWidgets.QWidget:
        """The track's info: title and artist (scrolling), the BPM / KEY rows and the time row with the current phrase, every value in a cell of fixed width."""
        center = QtWidgets.QWidget()
        center.setMinimumHeight(self.HEADER_H)
        center.setMaximumHeight(self.HEADER_H)
        rows = QtWidgets.QVBoxLayout(center)
        rows.setContentsMargins(0, 0, 0, 0)
        rows.setSpacing(0)
        rows.setAlignment(QtCore.Qt.AlignVCenter)

        def _style(lbl: QtWidgets.QLabel, size: int, bold: bool, color: str, *,
                   ignore_width: bool = False) -> None:
            f = lbl.font()
            f.setPointSize(size)
            f.setBold(bold)
            lbl.setFont(f)
            lbl.setWordWrap(False)
            lbl.setStyleSheet(f"color: {color};")
            h_policy = QtWidgets.QSizePolicy.Ignored if ignore_width else QtWidgets.QSizePolicy.Preferred
            lbl.setSizePolicy(h_policy, QtWidgets.QSizePolicy.Fixed)

        self.title_label = MarqueeLabel()
        self.artist_label = MarqueeLabel()
        _style(self.title_label, 15, True, _VALUE, ignore_width=True)
        _style(self.artist_label, 12, False, _ARTIST, ignore_width=True)
        # BPM / KEY / time: every element has its own fixed-width cell, sized for
        # the widest value it can take, so no value change moves anything. No
        # brackets: groups are set apart by a gap and by colour (label, current
        # value, original value, change).
        info_font = QtGui.QFont(self.title_label.font())
        info_font.setPointSize(12)
        info_font.setBold(False)
        fm = QtGui.QFontMetrics(info_font)
        GROUP_GAP = 4

        def _cell(samples: list[str], color: str, *, right: bool = False, lead: int = 0) -> QtWidgets.QLabel:
            lbl = QtWidgets.QLabel()
            lbl.setFont(info_font)
            lbl.setTextFormat(QtCore.Qt.RichText)
            lbl.setStyleSheet(f"color: {color};")
            lbl.setAlignment((QtCore.Qt.AlignRight if right else QtCore.Qt.AlignLeft) | QtCore.Qt.AlignVCenter)
            lbl.setContentsMargins(lead, 0, 0, 0)
            lbl.setFixedWidth(max(fm.horizontalAdvance(t) for t in samples) + 2 + lead)
            # fixed in both directions: a new text (the time, every frame) asks for no layout (Qt skips
            # the request only for a widget whose size cannot change)
            lbl.setFixedHeight(lbl.sizeHint().height())
            lbl.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
            return lbl

        # BPM and KEY share the columns label | current | original | change |
        # meter, each column as wide as its widest value in either row (BPM is
        # always six characters, see _bpm_text), all left-aligned so the rows
        # line up like a table.
        widest_key = max(KEY_LABEL_SAMPLES, key=fm.horizontalAdvance)
        # The original value, its change and the meter run on in one last cell
        # ("205.00 +00pp  4/4"): no gaps inside the group, and the cell's fixed
        # width still keeps everything else in place.
        columns = [
            (["BPM", "KEY"], _VALUE, 0),
            (["9999.9", "999.99", widest_key, "--"], _VALUE, 0),
            ([f"9999.9 {_pp_text(-100)}{_GAP}8/4", f"{widest_key} {_signed(-12, '+06.2f')}"], _DIM, GROUP_GAP),
        ]
        self._bpm_cells = [_cell(w, color, lead=lead) for w, color, lead in columns]
        self._key_cells = [_cell(w, color, lead=lead) for w, color, lead in columns]
        self._time_cells = [_cell(["88:88.88"], _VALUE), _cell(["/ 88:88.88"], _DIM)]
        for cell in self._time_cells:
            # plain text (drawn the same): a rich-text label builds a new text document per change, every frame here
            cell.setTextFormat(QtCore.Qt.PlainText)
        # current phrase, last on the time row (its width never moves anything):
        # a squircle in the phrase colour, then the label in white
        self._phrase_swatch = QtWidgets.QLabel()
        self._phrase_swatch.setFixedSize(11, 11)
        policy = self._phrase_swatch.sizePolicy()
        policy.setRetainSizeWhenHidden(True)  # keeps its slot: counted in the width cap, label never shifts
        self._phrase_swatch.setSizePolicy(policy)
        self._phrase_swatch.hide()
        self._phrase_swatch_color: Optional[tuple[int, int, int]] = None
        self._phrase_label = QtWidgets.QLabel()
        self._phrase_label.setFont(info_font)
        self._phrase_label.setStyleSheet(f"color: {_VALUE};")
        self._phrase_label.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        # as wide as the longest standard label; longer custom labels are elided
        self._phrase_label.setFixedWidth(fm.horizontalAdvance("BREAK_CHORUS10") + 2)
        self._bpm_cells[0].setText("BPM")
        self._key_cells[0].setText("KEY")

        def _row(cells: list[QtWidgets.QLabel]) -> QtWidgets.QHBoxLayout:
            row = QtWidgets.QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(6)
            for cell in cells:
                row.addWidget(cell, 0, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
            row.addStretch(1)
            return row

        self.set_bpm(None)
        self.set_key(None)
        self.set_time(0.0, 0.0)

        rows.addWidget(self.title_label, 0, QtCore.Qt.AlignVCenter)
        rows.addWidget(self.artist_label, 0, QtCore.Qt.AlignVCenter)
        rows.addLayout(_row(self._bpm_cells))
        rows.addLayout(_row(self._key_cells))
        time_row = _row(self._time_cells)
        time_row.insertSpacing(time_row.count() - 1, GROUP_GAP + 6)
        time_row.insertWidget(time_row.count() - 1, self._phrase_swatch, 0, QtCore.Qt.AlignVCenter)
        time_row.insertWidget(time_row.count() - 1, self._phrase_label, 0, QtCore.Qt.AlignVCenter)
        rows.addLayout(time_row)
        # The info area takes all the width the window gives the header, never
        # less than its widest fixed row (BPM / KEY / time + phrase); title and
        # artist scroll inside it.
        info_rows = (rows.itemAt(i).layout() for i in range(rows.count()))
        self._info_width = max(r.sizeHint().width() for r in info_rows if r is not None)
        center.setMinimumWidth(self._info_width)
        self.info_area = center  # the main pane fits the edit panel's columns to its spare width
        center.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        return center

    def _build_record_column(self) -> QtWidgets.QVBoxLayout:
        """The record with a row of deck buttons (MT / RST, filled in by the main pane) above it and the beat dots under it."""
        record_col = QtWidgets.QVBoxLayout()
        record_col.setContentsMargins(0, 0, 0, 0)
        record_col.setSpacing(4)
        self.record_buttons_slot = QtWidgets.QHBoxLayout()
        self.record_buttons_slot.setContentsMargins(0, 0, 0, 0)
        self.record_buttons_slot.setSpacing(6)
        self.record_buttons_slot.setAlignment(QtCore.Qt.AlignHCenter)
        record_col.addLayout(self.record_buttons_slot)
        self.record_label = QtWidgets.QLabel()
        self.record_label.setFixedSize(self.RECORD_D, self.RECORD_D)
        self.record_label.setAlignment(QtCore.Qt.AlignCenter)
        record_col.addWidget(self.record_label, 0, QtCore.Qt.AlignHCenter)
        self.beat_dots = BeatDots()
        record_col.addWidget(self.beat_dots, 0, QtCore.Qt.AlignHCenter)
        return record_col

    def _build_knob_column(self) -> QtWidgets.QVBoxLayout:
        """Its channel's knobs right of the meter (TRIM, HI, MID, LOW: requests; the player reports them, set_knobs)."""
        knob_col = QtWidgets.QVBoxLayout()
        knob_col.setContentsMargins(0, 0, 0, 0)
        knob_col.setSpacing(0)
        knob_col.setAlignment(QtCore.Qt.AlignVCenter)
        self.knobs: dict[int, Knob] = {}
        for text, which in CHANNEL_KNOBS:
            knob = Knob(KNOB_SIZE)
            knob.moved.connect(lambda v, which=which: self.knob_moved.emit(which, v))
            self.knobs[which] = knob
            knob_col.addWidget(knob, 0, QtCore.Qt.AlignHCenter)
            knob_col.addWidget(_knob_label(text))
        self.set_eq_available(True)
        return knob_col


    # External
    def set_title(self, text: str):
        self.title_label.set_marquee_text((text or "").strip())

    def set_artist(self, text: str):
        self.artist_label.set_marquee_text((text or "").strip())

    def set_bpm(self, current: Optional[float], original: float = 0.0, pp: float = 0.0, meter: int = 4):
        """BPM  current  original +NNpp  meter; ``current`` None shows ``--``."""
        _, value, rest = self._bpm_cells
        if current is None:
            value.setText("--")
            rest.setText("")
            return
        value.setText(_bpm_text(current))
        rest.setText(f"{_bpm_text(original)} {_span(_pp_text(pp), _signed_color(round(pp, 2)))}{_GAP}{int(meter)}/4")

    def set_key(self, current: Optional[tuple[str, str]], original: tuple[str, str] = ("", ""),
                semitones: float = 0.0):
        """KEY  current  original +N.NN; keys as (Alphanumeric, classical); None shows ``--``."""
        _, value, rest = self._key_cells
        if current is None:
            value.setText("--")
            rest.setText("")
            return
        shift = round(float(semitones), 2)
        value.setText(html.escape(f"{current[0]}{_THIN}/{_THIN}{current[1]}"))
        rest.setText(f"{html.escape(f'{original[0]}{_THIN}/{_THIN}{original[1]}')} "
                     f"{_span(_signed(shift, '+06.2f'), _signed_color(shift))}")

    def set_phrase(self, label: Optional[str], color: tuple[int, int, int] = (110, 110, 110)):
        """Current phrase next to the time: a squircle in its colour and the
        numbered label; None clears both."""
        text = label or ""
        width = self._phrase_label.width()
        self._phrase_label.setText(
            QtGui.QFontMetrics(self._phrase_label.font()).elidedText(text, QtCore.Qt.ElideRight, width) if text else ""
        )
        color = tuple(color) if label else None
        if color == self._phrase_swatch_color:
            return
        self._phrase_swatch_color = color
        if color is None:
            self._phrase_swatch.hide()
            return
        self._phrase_swatch.setStyleSheet(f"background: rgb({color[0]},{color[1]},{color[2]}); border-radius: 3px;")
        self._phrase_swatch.show()

    def set_beat(self, beat_in_bar: Optional[int], meter: int = 4):
        self.beat_dots.set_beat(beat_in_bar, meter)

    def set_time(self, current_sec: float, duration_sec: float):
        self._time_cells[0].setText(_clock(current_sec))
        self._time_cells[1].setText("/ " + _clock(duration_sec))

    def update_album_art(self, album_art: Optional[QtGui.QImage]):
        self._last_album_art = album_art
        if album_art is not None and not album_art.isNull():
            target = self.cover_label.contentsRect().size()
            if not target.isValid():
                target = self.cover_label.size()
            pix = QtGui.QPixmap.fromImage(album_art).scaled(
                target,
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
            self.cover_label.setPixmap(pix)
        else:
            self._set_placeholder_cover()
        self._load_record_pixmap(album_art)

    def set_track_loaded(self, loaded: bool) -> None:
        """The unload button is there only while a track is loaded."""
        self.unload_button.setVisible(bool(loaded))

    def update_record_pix(self, time_sec: float):
        self._current_time = time_sec
        base = self._record_base_pix
        if base.isNull():
            return
        dpr = self.record_label.devicePixelRatioF()
        if abs(dpr - self._record_base_dpr) > 1e-6:
            self._load_record_pixmap(self._last_album_art)
            base = self._record_base_pix
            if base.isNull():
                return
        angle = ((time_sec * (33.3333333 / 60) * 360) % 360)
        step = max(1, int(getattr(self, "_record_cache_step_deg", 2)))
        cache_key = int(round(angle / step) * step) % 360
        if cache_key == self._record_last_cache_key:
            return
        self._record_last_cache_key = cache_key
        rotated = self._record_rotation_cache.get(cache_key)
        if rotated is None:
            logical_size = self.record_label.size()
            pixel_w = max(1, int(round(logical_size.width() * dpr)))
            pixel_h = max(1, int(round(logical_size.height() * dpr)))
            rotated = QtGui.QPixmap(pixel_w, pixel_h)
            rotated.setDevicePixelRatio(dpr)
            rotated.fill(QtCore.Qt.transparent)

            painter = QtGui.QPainter(rotated)
            painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
            painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)
            painter.translate(logical_size.width() / 2, logical_size.height() / 2)
            painter.rotate(cache_key)
            painter.translate(-logical_size.width() / 2, -logical_size.height() / 2)
            painter.drawPixmap(0, 0, base)
            painter.end()
            self._record_rotation_cache[cache_key] = rotated
        self.record_label.setPixmap(rotated)

    def set_config(self, cfg: config):
        self.cfg = cfg
        self._load_record_pixmap(self._last_album_art)

    def set_output_peak_dbfs(self, dbfs: float):
        self.peak_meter.set_level_dbfs(dbfs)

    def set_knobs(self, positions) -> None:
        """The channel's knobs as the player reports them (TRIM, EQ LOW, MID, HIGH: 0 .. 1); a knob in the hand
        shows the hand's position."""
        for which, position in enumerate(tuple(positions)[:4]):
            self.knobs[which].set_value(position)

    def set_eq_available(self, available: bool) -> None:
        """The EQ knobs work on the native engine's mixer alone (the Python engine has no EQ)."""
        self.knobs[0].setToolTip(tr("TRIM: the level before the volume fader, -24 to +9 dB. Double-click: 0 dB."))
        for text, which in CHANNEL_KNOBS[1:]:
            knob = self.knobs[which]
            knob.setEnabled(available)
            knob.setToolTip(tr("EQ {band}: the isolator's band, from a kill to +6 dB (0 dB in the middle). "
                               "Double-click: 0 dB.").format(band=text) if available else
                            tr("EQ {band}: the native audio engine's mixer only (Settings > Audio > "
                               "Engine).").format(band=text))

    # Internal helper
    def _set_placeholder_cover(self):
        pix = QtGui.QPixmap(self.HEADER_H-4, self.HEADER_H-4)
        pix.fill(QtGui.QColor("#333"))
        painter = QtGui.QPainter(pix)
        painter.setPen(QtGui.QPen(QtGui.QColor("#666")))
        painter.drawRect(0, 0, self.HEADER_H - 5, self.HEADER_H - 5)
        painter.setPen(QtGui.QPen(QtGui.QColor("#aaa")))
        painter.drawText(pix.rect(), QtCore.Qt.AlignCenter, tr("No Art"))
        painter.end()
        self.cover_label.setPixmap(pix)

    def _load_record_pixmap(
        self,
        album_art: Optional[QtGui.QImage],
        size: int | None = None,
        vinyl_path: Optional[str] = None,
    ):
        size = size or self.RECORD_D
        dpr = self.record_label.devicePixelRatioF() if hasattr(self, "record_label") else 1.0
        pixel_size = max(1, int(round(size * dpr)))
        base = QtGui.QPixmap(pixel_size, pixel_size)
        base.setDevicePixelRatio(dpr)
        base.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(base)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

        outer_rect = QtCore.QRectF(2, 2, size - 4, size - 4)

        painter.setBrush(QtGui.QColor("#111"))
        painter.setPen(QtGui.QPen(QtGui.QColor("#444"), 2))
        painter.drawEllipse(outer_rect)

        used_any = False
        if album_art is not None and not album_art.isNull():
            src_pm = QtGui.QPixmap.fromImage(album_art)
            w, h = src_pm.width(), src_pm.height()
            side = min(w, h)
            x = (w - side) // 2
            y = (h - side) // 2
            label_margin = int(size * 0.19)
            label_size = size - 2 * label_margin
            square = src_pm.copy(x, y, side, side).scaled(
                label_size,
                label_size,
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
            circ = QtGui.QPixmap(label_size, label_size)
            circ.fill(QtCore.Qt.transparent)
            pp = QtGui.QPainter(circ)
            pp.setRenderHint(QtGui.QPainter.Antialiasing, True)
            path = QtGui.QPainterPath()
            path.addEllipse(0, 0, label_size, label_size)
            pp.setClipPath(path)
            pp.drawPixmap(0, 0, square)
            pp.end()
            painter.drawPixmap(label_margin, label_margin, circ)
            used_any = True

        if not used_any:
            cfg_path = (
                getattr(self.cfg.viewconfig, "record_img_path", None)
                if self.cfg is not None and getattr(self.cfg, "viewconfig", None) is not None
                else None
            )
            path = vinyl_path if vinyl_path is not None else cfg_path or "assets/vinyl.png"
            vinyl = QtGui.QPixmap(setting_path(path)) if path else QtGui.QPixmap()
            if not vinyl.isNull():
                vinyl = vinyl.scaled(
                    size - 6,
                    size - 6,
                    QtCore.Qt.KeepAspectRatio,
                    QtCore.Qt.SmoothTransformation,
                )
                vx = (size - vinyl.width()) // 2
                vy = (size - vinyl.height()) // 2
                painter.drawPixmap(vx, vy, vinyl)
            else:
                painter.setBrush(QtGui.QColor("#333"))
                painter.setPen(QtGui.QPen(QtGui.QColor("#666"), 1))
                painter.drawEllipse(
                    outer_rect.adjusted(
                        size * 0.17, size * 0.17, -size * 0.17, -size * 0.17
                    )
                )
                painter.setPen(QtGui.QPen(QtGui.QColor("#555"), 1))
                for r in range(8, (size // 2) - 8, 3):
                    painter.drawEllipse(QtCore.QRectF(size / 2 - r, size / 2 - r, 2 * r, 2 * r))

        # Rim: one even white ring (balanced, so the spinning record never looks
        # off-centre), with the upright marker as a red stretch of the same ring
        # at 12 o'clock, turning with the record.
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.setPen(QtGui.QPen(QtGui.QColor("#e6e6e6"), 2.5))
        painter.drawEllipse(outer_rect)
        marker_pen = QtGui.QPen(QtGui.QColor("#e5484d"), 2.5)
        marker_pen.setCapStyle(QtCore.Qt.FlatCap)
        painter.setPen(marker_pen)
        painter.drawArc(outer_rect, (90 - 10) * 16, 20 * 16)

        painter.end()

        self._record_base_pix = base
        self._record_base_dpr = dpr
        self._record_rotation_cache.clear()
        self._record_last_cache_key = None
        self.update_record_pix(self._current_time)
