"""The drawing grammar of the audio system diagram (Settings > Audio > Overview; the analysis pipeline map's SVG,
build/pipeline_map_svg.py, borrows its look): a model as a Simulink model is drawn, in the app's dark theme. On a
dotted canvas, rounded blocks with thin borders of their domain's colour, their names under them and their parameters
inside (a gain as a triangle, a sum as a circle, a subsystem as a box of its own blocks), port marks at their edges,
signal lines routed at right angles (their name above them, a line hop where two cross, a dot where one branches), a
port value label (a small yellow pill), a block ringed when in trouble, a path idle drawn as Simulink's commented-out
blocks, a status bar with a legend of the domains. Drawn in model coordinates (W x H) scaled to the widget's width."""
from __future__ import annotations

import math

from PySide6 import QtCore, QtGui, QtWidgets

from ui.theme import current_theme
from ui.widgets.background_paint import BackgroundPaint

FONT_MAIN, FONT_SUB, FONT_NAME, FONT_LABEL = 8.4, 7.6, 7.8, 7.0
# the domains' colours, as Simulink colours sample times (kept clear of the states' green / yellow / red)
DOMAINS = (QtGui.QColor("#5aa9ff"), QtGui.QColor("#b58cff"), QtGui.QColor("#f0a35a"), QtGui.QColor("#4fd1c5"),
           QtGui.QColor("#e879b9"))


class _Palette:
    """The theme's colours for the diagrams (``use_theme`` sets them from the current theme at each paint)."""

    def __init__(self) -> None:
        self.ink = QtGui.QColor("#e6e6e6")        # a block's title
        self.dim = QtGui.QColor("#8a8a8a")        # its parameters, the names under the blocks
        self.paper = QtGui.QColor("#181818")      # a block's face
        self.canvas = QtGui.QColor("#0b0b0b")
        self.grid = QtGui.QColor("#1c1c1c")
        self.commented = QtGui.QColor("#4c4c4c")  # a path idle (Simulink's commented-out blocks)
        self.shadow = QtGui.QColor(0, 0, 0, 0)
        self.value_fill = QtGui.QColor("#2b2410")
        self.value_edge = QtGui.QColor("#6b5a1a")
        self.value_ink = QtGui.QColor("#f2c94c")
        self.warn = QtGui.QColor("#e6c229")
        self.bad = QtGui.QColor("#e5484d")
        self.ok = QtGui.QColor("#3fbf5f")
        self.accent = QtGui.QColor("#1f7fe5")


PAL = _Palette()


def use_theme() -> None:
    t = current_theme()
    PAL.ink, PAL.dim, PAL.commented = QtGui.QColor(t.text), QtGui.QColor(t.text_dim), QtGui.QColor(t.text_disabled)
    PAL.canvas = QtGui.QColor(t.base)
    PAL.paper = QtGui.QColor(t.raised)
    PAL.grid = QtGui.QColor(t.border)
    PAL.grid.setAlphaF(0.45)
    PAL.warn, PAL.bad, PAL.ok = QtGui.QColor(t.warning), QtGui.QColor(t.danger), QtGui.QColor(t.success)
    PAL.accent = QtGui.QColor(t.accent)


def soft(c: QtGui.QColor, alpha: float) -> QtGui.QColor:
    c = QtGui.QColor(c)
    c.setAlphaF(alpha)
    return c


class ModelCanvas(QtWidgets.QWidget):
    """A model drawn in W x H coordinates, scaled to the widget's width (its height follows). A subclass sets W, H
    and draws in ``paint_model(p, m)``; ``set_model(m)`` gives it what to show (``m["status"]``, ``m["legend"]``: the
    status bar's text and its (colour, text) entries)."""

    W, H = 1000.0, 600.0

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._m: dict = {}
        self.setMinimumWidth(640)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.setFixedHeight(int(940 * self.H / self.W))
        # painted off the GUI thread (ui.widgets.background_paint): ~20 ms a model, four a second while shown
        self._paint = BackgroundPaint(self, self._draw)
        self._paint_font = QtGui.QFont(self.font())     # the widget's font as the paint thread's helpers use it

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(940, int(940 * self.H / self.W))

    def resizeEvent(self, event) -> None:
        height = int(event.size().width() * self.H / self.W)
        if height != self.height():
            self.setFixedHeight(height)
        super().resizeEvent(event)
        self._repaint()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._repaint()

    def set_model(self, m: dict) -> None:
        self._m = m
        self._repaint()

    def _repaint(self) -> None:
        """The model painted again on the paint thread, where it is shown (its theme's colours and the widget's
        font taken here, on the GUI thread)."""
        if not self._m or not self.isVisible():
            return
        use_theme()
        self._paint.request((self._m, QtGui.QFont(self.font())))

    def paintEvent(self, _event) -> None:
        p = QtGui.QPainter(self)
        if not self._paint.draw_on(p, QtCore.QRectF(self.rect())):
            p.fillRect(self.rect(), PAL.canvas)          # until the first picture is painted
        p.end()

    def _draw(self, p: QtGui.QPainter, size: QtCore.QSizeF, state) -> None:
        """The picture (on the paint thread): ``state`` the model and the widget's font; the model's coordinates
        scaled to ``size``'s width. Reads nothing of the widget."""
        m, self._paint_font = state
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setRenderHint(QtGui.QPainter.TextAntialiasing)
        scale = size.width() / self.W
        p.scale(scale, scale)
        frame = QtGui.QPainterPath()
        frame.addRoundedRect(QtCore.QRectF(0.5, 0.5, self.W - 1, self.H - 1), 10, 10)
        p.setClipPath(frame)
        p.fillRect(QtCore.QRectF(0, 0, self.W, self.H), PAL.canvas)
        p.setPen(QtGui.QPen(PAL.grid, 1.2, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
        for gx in range(20, int(self.W), 20):
            for gy in range(20, int(self.H) - 26, 20):
                p.drawPoint(QtCore.QPointF(gx, gy))
        p.setClipping(False)
        p.setPen(QtGui.QPen(soft(PAL.grid, 1.0), 1.0))
        p.setBrush(QtCore.Qt.NoBrush)
        p.drawPath(frame)
        self.paint_model(p, m)
        self._status(p, m)

    def paint_model(self, p: QtGui.QPainter, m: dict) -> None:
        """The model (on the paint thread: the helpers below and the module's constants only, never the widget)."""
        raise NotImplementedError

    # drawing helpers (the model's coordinates)
    def _font(self, size: float, bold: bool = False, italic: bool = False) -> QtGui.QFont:
        f = QtGui.QFont(self._paint_font)
        f.setPointSizeF(size)
        f.setBold(bold)
        f.setItalic(italic)
        return f

    def _text(self, p, rect: QtCore.QRectF, text: str, size: float = FONT_SUB, bold=False, italic=False,
              align=QtCore.Qt.AlignCenter, color: QtGui.QColor = PAL.ink, elide: bool = True) -> None:
        p.setFont(self._font(size, bold, italic))
        p.setPen(color)
        fm = QtGui.QFontMetricsF(p.font())
        p.drawText(rect, align, fm.elidedText(text, QtCore.Qt.ElideRight, rect.width()) if elide else text)

    def _shadowed(self, p, path: QtGui.QPainterPath, edge: QtGui.QColor, live: bool, width: float = 1.1) -> None:
        if live:
            # a faint glow of its domain's colour around a live block
            p.setPen(QtGui.QPen(soft(edge, 0.10), 5.0))
            p.setBrush(QtCore.Qt.NoBrush)
            p.drawPath(path)
        p.setBrush(PAL.paper if live else soft(PAL.paper, 0.4))
        p.setPen(QtGui.QPen(soft(edge, 0.85) if live else soft(PAL.commented, 0.8), width))
        p.drawPath(path)

    def _ports(self, p, rect: QtCore.QRectF, ins, outs, color: QtGui.QColor) -> None:
        """Simulink's port marks: a small chevron at each input on the left edge and output on the right."""
        p.setPen(QtGui.QPen(soft(color, 0.7), 0.9))
        for y in ins:
            p.drawPolyline([QtCore.QPointF(rect.left() + 1.8, y - 3.0), QtCore.QPointF(rect.left() + 5.2, y),
                            QtCore.QPointF(rect.left() + 1.8, y + 3.0)])
        for y in outs:
            p.drawPolyline([QtCore.QPointF(rect.right() - 5.2, y - 3.0), QtCore.QPointF(rect.right() - 1.8, y),
                            QtCore.QPointF(rect.right() - 5.2, y + 3.0)])

    def _name(self, p, rect: QtCore.QRectF, name: str, live: bool) -> None:
        self._text(p, QtCore.QRectF(rect.center().x() - 90, rect.bottom() + 3, 180, 12), name, FONT_NAME,
                   color=PAL.dim if live else PAL.commented, elide=False)

    def _highlight(self, p, rect: QtCore.QRectF, state: str) -> None:
        if state not in ("warn", "bad"):
            return
        c = QtGui.QColor(PAL.warn if state == "warn" else PAL.bad)
        p.setBrush(QtCore.Qt.NoBrush)
        for width, alpha in ((7.0, 0.18), (2.0, 0.9)):
            p.setPen(QtGui.QPen(soft(c, alpha), width))
            p.drawRoundedRect(rect.adjusted(-4, -4, 4, 4), 7, 7)

    def _block(self, p, rect, name, lines, color, live=True, ins=(0.5,), outs=(0.5,), state="ok", bold_first=True):
        path = QtGui.QPainterPath()
        path.addRoundedRect(rect, 5, 5)
        self._highlight(p, rect, state)
        self._shadowed(p, path, color, live)
        self._ports(p, rect, [rect.top() + f * rect.height() for f in ins],
                    [rect.top() + f * rect.height() for f in outs], color if live else PAL.commented)
        n = max(1, len(lines))
        lh = min(12.0, (rect.height() - 6) / n)
        top = rect.center().y() - lh * n / 2
        for k, line in enumerate(lines):
            self._text(p, QtCore.QRectF(rect.left() + 7, top + k * lh, rect.width() - 14, lh), line,
                       FONT_MAIN if k == 0 and bold_first else FONT_SUB, bold=bold_first and k == 0,
                       color=(PAL.ink if k == 0 or not bold_first else PAL.dim) if live else PAL.commented)
        if name:
            self._name(p, rect, name, live)

    def _gain(self, p, rect, name, text, color, live=True, state="ok"):
        path = QtGui.QPainterPath()
        path.moveTo(rect.left() + 3, rect.top())
        path.lineTo(rect.right(), rect.center().y())
        path.lineTo(rect.left() + 3, rect.bottom())
        path.quadTo(rect.left(), rect.bottom(), rect.left(), rect.bottom() - 3)
        path.lineTo(rect.left(), rect.top() + 3)
        path.quadTo(rect.left(), rect.top(), rect.left() + 3, rect.top())
        self._highlight(p, rect, state)
        self._shadowed(p, path, color, live)
        self._text(p, QtCore.QRectF(rect.left() + 1, rect.top(), rect.width() * 0.74, rect.height()), text,
                   FONT_SUB, color=PAL.ink if live else PAL.commented)
        self._name(p, rect, name, live)

    def _sum(self, p, c: QtCore.QPointF, r: float, color, live=True):
        path = QtGui.QPainterPath()
        path.addEllipse(c, r, r)
        self._shadowed(p, path, color, live)
        self._text(p, QtCore.QRectF(c.x() - r, c.y() - r - 0.5, 2 * r, 2 * r), "+", 10.0, bold=True,
                   color=PAL.ink if live else PAL.commented)

    def _arrowhead(self, p, a: QtCore.QPointF, b: QtCore.QPointF, c: QtGui.QColor) -> None:
        dx, dy = b.x() - a.x(), b.y() - a.y()
        length = max(1e-6, math.hypot(dx, dy))
        ux, uy = dx / length, dy / length
        head = QtGui.QPolygonF([b, QtCore.QPointF(b.x() - 6 * ux + 3 * uy, b.y() - 6 * uy - 3 * ux),
                                QtCore.QPointF(b.x() - 6 * ux - 3 * uy, b.y() - 6 * uy + 3 * ux)])
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(c)
        p.drawPolygon(head)

    def _line(self, p, pts, color, live=True, arrow=True, label: str = "", hops=(), width=1.3):
        """A signal line through ``pts`` (right angles), an arrowhead at its end; ``hops``: (x, y) where it crosses
        another line (a vertical segment's hop, Simulink's line-hop crossing style); ``label`` above its longest
        horizontal segment."""
        c = soft(color, 0.9) if live else PAL.commented
        p.setPen(QtGui.QPen(c, width, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
        p.setBrush(QtCore.Qt.NoBrush)
        path = QtGui.QPainterPath()
        path.moveTo(*pts[0])
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            for hx, hy in hops:
                if abs(x0 - x1) < 1e-6 and abs(hx - x0) < 1e-6 and min(y0, y1) < hy < max(y0, y1):
                    sign = 1 if y1 > y0 else -1
                    path.lineTo(x0, hy - 4 * sign)
                    path.arcTo(QtCore.QRectF(x0 - 4, hy - 4, 8, 8), 90 if sign > 0 else 270, -180 * sign)
            path.lineTo(x1, y1)
        p.drawPath(path)
        q = [QtCore.QPointF(x, y) for x, y in pts]
        if arrow:
            self._arrowhead(p, q[-2], q[-1], c)
        if label:
            best = max(range(len(q) - 1), key=lambda k: abs(q[k + 1].x() - q[k].x()))
            a, b = q[best], q[best + 1]
            cx = (a.x() + b.x()) / 2
            self._text(p, QtCore.QRectF(cx - 70, a.y() - 13, 140, 11), label, FONT_LABEL, italic=True, color=c,
                       elide=False)

    def _dot(self, p, x, y, color, live=True):
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(color if live else PAL.commented)
        p.drawEllipse(QtCore.QPointF(x, y), 2.4, 2.4)

    def _value(self, p, x, y, text):
        """A port value label (Simulink's live signal value): a small yellow box centred on (x, y)."""
        p.setFont(self._font(7.2, bold=True))
        fm = QtGui.QFontMetricsF(p.font())
        w = fm.horizontalAdvance(text) + 12
        r = QtCore.QRectF(x - w / 2, y - 7, w, 14)
        p.setPen(QtGui.QPen(PAL.value_edge, 0.8))
        p.setBrush(PAL.value_fill)
        p.drawRoundedRect(r, 7, 7)
        p.setPen(PAL.value_ink)
        p.drawText(r, QtCore.Qt.AlignCenter, text)

    def _status(self, p, m):
        """Simulink's status bar: the engine's state on the left, the sample-time legend on the right."""
        bar = QtCore.QRectF(1, self.H - 26, self.W - 2, 25)
        bg = QtGui.QPainterPath()
        bg.addRoundedRect(QtCore.QRectF(0.5, self.H - 26, self.W - 1, 25.5), 10, 10)
        bg.addRect(QtCore.QRectF(0.5, self.H - 26, self.W - 1, 12))
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(PAL.paper)
        p.drawPath(bg.simplified())
        p.setPen(QtGui.QPen(soft(PAL.grid, 1.0), 0.8))
        p.drawLine(bar.topLeft(), bar.topRight())
        legend = m.get("legend", [])
        legend_w = 0.0
        p.setFont(self._font(7.4))
        fm = QtGui.QFontMetricsF(p.font())
        widths = [fm.horizontalAdvance(t) + 24 for _c, t in legend]
        legend_w = sum(widths)
        self._text(p, QtCore.QRectF(12, bar.top(), self.W - legend_w - 30, bar.height()), m.get("status", ""), 7.4,
                   align=QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, color=PAL.dim)
        x = self.W - legend_w - 8
        for (color, text), w in zip(legend, widths):
            p.setPen(QtCore.Qt.NoPen)
            p.setBrush(color)
            p.drawRoundedRect(QtCore.QRectF(x, bar.center().y() - 4, 8, 8), 2, 2)
            self._text(p, QtCore.QRectF(x + 12, bar.top(), w - 12, bar.height()), text, 7.4,
                       align=QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, color=PAL.dim, elide=False)
            x += w
