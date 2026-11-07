"""Settings > Audio > Overview: the audio system drawn as a Simulink model is, in the app's dark theme. On a dotted
canvas, rounded blocks with thin borders, their names under them and their parameters inside (gains as triangles, a sum as a
circle, the channel strip as a subsystem of its own blocks), signal lines routed at right angles and the signal's
format above them, coloured by their clock domain as Simulink colours sample times (D1: the device's clock). The
output ends in an Audio Device Writer and a Scope whose screen is its buffer's histogram; live values show in port
value labels (the deck's tempo, its channel's peak); a block in trouble is highlighted (yellow: late wake-ups or a
fallback, red: dropouts or not opened). A status bar under it: the engine, its threads, power throttling, the
measured responses, the MIDI input, and the sample-time legend.

Fed by the buffer stats the player sends every 0.25 s (PlaybackSettingsMixin._on_output_buffer) and the deck's
report; drawn only while shown. Texts shown go through tr(); the DJ terms stay as they are."""
from __future__ import annotations

import math
from collections import deque

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from audioengine import controls
from system.i18n import tr
from ui.config.output_group import BufferMonitor, THREAD_STATE_NAMES
from ui.widgets.model_canvas import DOMAINS, FONT_LABEL, FONT_MAIN, FONT_NAME, FONT_SUB, PAL, ModelCanvas, soft
from audioengine.volume import slider_percent_to_linear, trim_knob_to_db

W, H = 1000.0, 310.0          # the model's coordinates (scaled to the widget's width)
# its lines: the deck and its strip, the master
ROW_1, ROW_M = 80.0, 222.0
# the clock domain (model_canvas.DOMAINS): the device's clock
D1 = DOMAINS[0]
# the right column: the output's Audio Device Writer and, beside it, its Scope
WRITER_X, WRITER_W = 612.0, 178.0


def _db(v: float) -> str:
    return "-inf" if v <= -99.0 else f"{v:+.1f}"


class SimulinkCanvas(ModelCanvas):
    """The audio system's model (see the module); ``set_model(m)``: what AudioOverview builds."""

    W, H = W, H

    def paint_model(self, p, m):
        if m.get("native"):
            self._paint_native(p, m)
        else:
            self._paint_python(p, m)

    def _speaker(self, p, x, y, color, live):
        c = color if live else PAL.commented
        p.setPen(QtGui.QPen(c, 0.9))
        p.setBrush(QtCore.Qt.NoBrush)
        p.drawPolygon(QtGui.QPolygonF([QtCore.QPointF(x, y - 3), QtCore.QPointF(x + 3.5, y - 3),
                                       QtCore.QPointF(x + 8, y - 7), QtCore.QPointF(x + 8, y + 7),
                                       QtCore.QPointF(x + 3.5, y + 3), QtCore.QPointF(x, y + 3)]))
        for r in (4.5, 8.0):
            p.drawArc(QtCore.QRectF(x + 8 - r, y - r, 2 * r, 2 * r), -45 * 16, 90 * 16)

    def _writer(self, p, row, out: dict, color) -> QtCore.QRectF:
        """An output's Audio Device Writer at ``row`` (its input on the row): the device, its API and mode, its
        format and period, its buffer and latency."""
        rect = QtCore.QRectF(WRITER_X, row - 28, WRITER_W, 56)
        live = out.get("live", True)
        path = QtGui.QPainterPath()
        path.addRoundedRect(rect, 5, 5)
        self._highlight(p, rect, out.get("state", "ok"))
        self._shadowed(p, path, color, live)
        self._ports(p, rect, [row], [], color if live else PAL.commented)
        self._speaker(p, rect.left() + 9, rect.top() + 12, color, live)
        ink = PAL.ink if live else PAL.commented
        self._text(p, QtCore.QRectF(rect.left() + 24, rect.top() + 4, rect.width() - 28, 13), out.get("device", ""),
                   FONT_MAIN, bold=True, align=QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, color=ink)
        for k, line in enumerate(out.get("lines", [])[:3]):
            x = rect.left() + (24 if k == 0 else 8)
            self._text(p, QtCore.QRectF(x, rect.top() + 17 + 11.5 * k, rect.right() - x - 4, 11.5), line, FONT_SUB,
                       align=QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, color=PAL.dim if live else PAL.commented)
        self._text(p, QtCore.QRectF(rect.left(), rect.bottom() + 3, rect.width(), 12), tr("Audio Device Writer"),
                   FONT_NAME, color=PAL.dim if live else PAL.commented, elide=False)
        return rect

    def _scope(self, p, writer: QtCore.QRectF, out: dict, color) -> None:
        """Beside the writer, a Scope of its buffer: its screen the histogram (the time at each level over the last
        10 s, log scale; dashed: the size; yellow: the 1 % low; red near empty), its readout under the screen."""
        live = out.get("live", True)
        rect = QtCore.QRectF(writer.right() + 18, writer.top(), W - 10 - writer.right() - 18, writer.height())
        path = QtGui.QPainterPath()
        path.addRoundedRect(rect, 5, 5)
        self._shadowed(p, path, color, live)
        row = writer.center().y()
        self._ports(p, rect, [row], [], color if live else PAL.commented)
        # its input: the writer's buffer (a measurement: thin and dotted)
        c = color if live and out.get("scoped") else PAL.commented
        p.setPen(QtGui.QPen(soft(c, 0.8), 1.0, QtCore.Qt.DotLine))
        p.drawLine(QtCore.QPointF(writer.right(), row), QtCore.QPointF(rect.left(), row))
        self._arrowhead(p, QtCore.QPointF(writer.right(), row), QtCore.QPointF(rect.left(), row), c)
        screen = QtCore.QRectF(rect.left() + 9, rect.top() + 4, rect.width() - 16, rect.height() - 19)
        p.setPen(QtGui.QPen(soft(PAL.grid, 1.0), 0.8))
        p.setBrush(QtGui.QColor("#000000") if live else PAL.canvas)
        p.drawRoundedRect(screen, 3, 3)
        hist = out.get("hist")
        if live and hist is not None and hist.size and hist.sum() > 0:
            bucket, cap, low = out["bucket"], out["capacity"], out["low"]
            span = max(cap * 1.25, (np.flatnonzero(hist)[-1] + 1) * bucket)
            cols = max(8, int(screen.width() * 1.4))
            share = np.zeros(cols)
            idx = np.minimum((np.arange(hist.size) * bucket / span * cols).astype(int), cols - 1)
            np.add.at(share, idx, hist / hist.sum())
            bar_w = screen.width() / cols
            for k in np.flatnonzero(share):
                h = (max(-4.0, math.log10(share[k])) + 4.0) / 4.0 * (screen.height() - 2)
                frac = k / cols * span / max(cap, 1e-6)
                col = PAL.bad if frac < 0.15 else PAL.warn if frac < 0.35 else PAL.ok
                p.fillRect(QtCore.QRectF(screen.left() + k * bar_w, screen.bottom() - 1 - h, max(bar_w, 0.7), h), col)
            for x_ms, pen in ((cap, QtGui.QPen(soft(PAL.ink, 0.6), 0.7, QtCore.Qt.DashLine)),
                              (low, QtGui.QPen(PAL.warn, 0.8))):
                x = screen.left() + min(1.0, x_ms / span) * screen.width()
                p.setPen(pen)
                p.drawLine(QtCore.QPointF(x, screen.top() + 1), QtCore.QPointF(x, screen.bottom() - 1))
        self._text(p, QtCore.QRectF(rect.left() + 3, rect.bottom() - 14, rect.width() - 6, 12),
                   out.get("readout", ""), 6.8, color=PAL.dim if live else PAL.commented)
        self._name(p, rect, tr("Scope (buffer)"), live)

    # the model
    def _paint_python(self, p, m):
        c = D1
        y = ROW_1
        deck = QtCore.QRectF(12, y - 27, 92, 54)
        self._block(p, deck, tr("Deck 1"), m["deck_lines"], c, ins=())
        ts = QtCore.QRectF(130, y - 27, 92, 54)
        self._block(p, ts, tr("Time Stretch"), m["ts_lines"], c, live=m["timestretch"])
        self._line(p, [(deck.right(), y), (ts.left(), y)], c)
        g = QtCore.QRectF(260, y - 21, 52, 42)
        self._gain(p, g, tr("Trim / Volume"), m.get("trim_text", ""), c)
        self._line(p, [(ts.right(), y), (g.left(), y)], c)
        wr = self._writer(p, y, m["main"], c)
        self._line(p, [(g.right(), y), (wr.left(), y)], c, label=m["main"].get("signal", ""))
        self._scope(p, wr, m["main"], c)
        self._text(p, QtCore.QRectF(12, ROW_M - 20, 700, 40),
                   tr("The Python engine plays deck 1 on one output: no mixer, no MIDI."),
                   9.0, align=QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, color=PAL.commented)

    def _paint_native(self, p, m):
        c1 = D1
        # line 1: the deck, its time stretch, its channel strip (the A9's: trim, a Color FX place, EQ, a Color FX
        # place, the fader, a Color FX place; the effect picked plays in its own place, the others idle)
        y = ROW_1
        deck = QtCore.QRectF(12, y - 27, 88, 54)
        self._block(p, deck, tr("Deck 1"), m["deck_lines"], c1, ins=())
        ts = QtCore.QRectF(116, y - 27, 84, 54)
        self._block(p, ts, tr("Time Stretch"), m["ts_lines"], c1)
        self._line(p, [(deck.right(), y), (ts.left(), y)], c1)
        strip = QtCore.QRectF(218, y - 46, 770, 100)
        path = QtGui.QPainterPath()
        path.addRoundedRect(strip, 8, 8)
        self._shadowed(p, path, c1, True)
        self._ports(p, strip, [y], [y], c1)
        self._name(p, strip, tr("Channel Strip 1 (subsystem)"), True)
        self._line(p, [(ts.right(), y), (strip.left(), y)], c1)
        h = 34.0
        trim = QtCore.QRectF(240, y - 17, 44, h)
        slots = {"pre_eq": QtCore.QRectF(304, y - h / 2, 84, h), "post_eq": QtCore.QRectF(508, y - h / 2, 150, h),
                 "post_fader": QtCore.QRectF(752, y - h / 2, 112, h)}
        eq = QtCore.QRectF(406, y - h / 2, 84, h)
        fader = QtCore.QRectF(690, y - 17, 44, h)
        self._gain(p, trim, "TRIM", m["trim_text"], c1)
        idle = {"pre_eq": "NOISE", "post_eq": "SWEEP · CRUSH · FILTER · JET", "post_fader": "SPACE · DUB ECHO"}
        names = {"pre_eq": tr("COLOR FX before EQ"), "post_eq": tr("COLOR FX after EQ"),
                 "post_fader": tr("COLOR FX after fader")}
        for slot, rect in slots.items():
            active = m.get("color_slot") == slot
            self._block(p, rect, names[slot], m["color_lines"] if active else [idle[slot]], c1, live=active,
                        bold_first=active)
        self._block(p, eq, m["eq_name"], m["eq_lines"], c1, bold_first=False)
        self._gain(p, fader, tr("Fader"), m["fader_text"], c1)
        chain = [(strip.left() + 6, trim.left()), (trim.right(), slots["pre_eq"].left()),
                 (slots["pre_eq"].right(), eq.left()), (eq.right(), slots["post_eq"].left()),
                 (slots["post_eq"].right(), fader.left()), (fader.right(), slots["post_fader"].left())]
        for a, b in chain:
            self._line(p, [(a, y), (b, y)], c1)
        self._line(p, [(slots["post_fader"].right(), y), (strip.right() - 6, y)], c1, arrow=False)
        self._value(p, strip.right() - 60, y - 24, m["peak_value"])
        # the strip's output: the ㄹ turn, from line 1's right end down the right edge, back left, down to line 2
        lane = strip.bottom() + 24
        self._line(p, [(strip.right(), y), (W - 6, y), (W - 6, lane), (26, lane), (26, ROW_M), (40, ROW_M)], c1)
        # line 2: the master: the channels summed, its level, its last stage, its device
        y = ROW_M
        sx = 50.0
        self._sum(p, QtCore.QPointF(sx, y), 10, c1)
        mg = QtCore.QRectF(92, y - 21, 50, 42)
        self._gain(p, mg, tr("Master"), m["master_text"], c1)
        self._line(p, [(sx + 10, y), (mg.left(), y)], c1)
        clip = QtCore.QRectF(172, y - 19, 46, 38)
        self._last_stage(p, clip, m["clip"], c1)
        self._line(p, [(mg.right(), y), (clip.left(), y)], c1)
        main = m["main"]
        wr = self._writer(p, y, main, c1)
        self._line(p, [(clip.right(), y), (wr.left(), y)], c1)
        self._text(p, QtCore.QRectF(wr.left(), y - 44, wr.width(), 12), main.get("signal", ""), FONT_LABEL,
                   italic=True, color=c1, elide=False)
        self._scope(p, wr, main, c1)

    def _last_stage(self, p, rect, kind, c1, live=True):
        """The master's last stage (audioengine.controls.CLIPS) as a block with its transfer curve: flat above the
        ceiling (a hard clip), tanh (the soft clip), or the line under it with its lookahead (the limiter)."""
        name = {"hard": tr("Hard Clip"), "limiter": tr("Limiter")}.get(kind, tr("Soft Clip"))
        self._block(p, rect, name, [], c1, live=live)
        p.setPen(QtGui.QPen(c1 if live else PAL.commented, 1.0))
        curve = QtGui.QPainterPath()
        half = rect.height() / 2 - 6
        for k in range(21):
            x = -1.0 + k / 10.0
            px = rect.left() + 7 + (x + 1) / 2 * (rect.width() - 14)
            if kind == "hard":
                v = max(-1.0, min(1.0, 1.6 * x))
            elif kind == "limiter":
                v = 0.85 * x
            else:
                v = math.tanh(2.2 * x) / math.tanh(2.2)
            py = rect.center().y() - v * half
            curve.moveTo(px, py) if k == 0 else curve.lineTo(px, py)
        p.setBrush(QtCore.Qt.NoBrush)
        p.drawPath(curve)
        if kind == "limiter":
            self._text(p, QtCore.QRectF(rect.left(), rect.bottom() - 13, rect.width() - 4, 11),
                       tr("{ms:g} ms").format(ms=controls.LIMITER_DELAY_SEC * 1000.0), 7.0,
                       color=PAL.dim if live else PAL.commented)


class AudioOverview(QtWidgets.QWidget):
    """The page: the model on its canvas. ``update_view(stats, capacity_ms, ctx)`` with the player's buffer stats
    (the engine client's take_buffer_stats' dict; None: none yet) and what the settings know (``ctx``: native, timestretch,
    main_device, api_names, midi, mix, deck, clip: the master's last stage)."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        self.canvas = SimulinkCanvas()
        lay.addWidget(self.canvas)
        lay.addStretch(1)
        self._monitor = BufferMonitor()

    def reset(self) -> None:
        self._monitor.reset()

    def _observe(self, observed: dict, capacity_ms: float, micro: bool) -> dict:
        """The window's histogram and readout values of the output (BufferMonitor's: the last 10 s)."""
        monitor = self._monitor
        values = monitor.observe(observed, capacity_ms, micro=micro)
        window: deque = monitor._window
        size = max(len(h) for _, h in window) if window else 0
        total = np.zeros(size)
        for _, h in window:
            total[:len(h)] += h
        return dict(values, hist=total, bucket=float(monitor._bucket_ms or 1.0), capacity=capacity_ms)

    def update_view(self, stats: dict | None, capacity_ms: float, ctx: dict) -> None:
        native = bool(ctx.get("native"))
        mix = ctx.get("mix") or {}
        deck = ctx.get("deck") or {}
        m = {"native": native, "timestretch": bool(ctx.get("timestretch")), "clip": ctx.get("clip") or "tanh"}
        factor = float(deck.get("factor", 1.0))
        playing = bool(deck.get("playing", False))
        m["deck_lines"] = [("▶ " if playing else "■ ") + f"x{factor:.3f}",
                           f"{(factor - 1) * 100:+.2f} %"]
        m["tempo_value"] = f"{factor * 100:.1f} %"
        m["ts_lines"] = [tr("pitch kept") if ctx.get("timestretch") else tr("varispeed"),
                         "PV + RTPGHI" if ctx.get("timestretch") else f"x{factor:.3f}"]
        knobs = deck.get("knobs") or (0.5, 0.5, 0.5, 0.5)
        trim_db = trim_knob_to_db(knobs[0])
        m["trim_text"] = _db(trim_db)
        iso = int(mix.get("eq_mode", 0)) == 0

        def eq_db(k: float) -> str:
            if iso and k <= 0.0:
                return "kill"
            v = -26.0 * (1.0 - 2.0 * k) if k < 0.5 else 6.0 * (2.0 * k - 1.0)
            return f"{v:+.0f}"

        m["eq_name"] = "ISO" if iso else "EQ"
        m["eq_lines"] = [f"H {eq_db(knobs[3])}", f"M {eq_db(knobs[2])}  L {eq_db(knobs[1])}"]
        ctype = int(mix.get("color_type", controls.COLOR_OFF))
        cknob = float(deck.get("color", 0.5))
        m["color_on"] = ctype < len(controls.COLOR_TYPES) and abs(cknob - 0.5) > 0.01
        m["color_slot"] = controls.COLOR_SLOTS[ctype] if m["color_on"] else None
        m["color_lines"] = [controls.COLOR_TYPES[ctype] if ctype < len(controls.COLOR_TYPES) else "OFF",
                            f"{(cknob - 0.5) * 200:+.0f} %"]
        volume = float(deck.get("volume", 1.0))
        g = slider_percent_to_linear(volume * 100.0, 0.0)
        m["fader_text"] = _db(20 * math.log10(g) if g > 0 else -120.0)
        peak = float(deck.get("peak", controls.PEAK_FLOOR_DBFS))
        m["peak_value"] = f"{peak:.1f} dBFS"
        m["master_text"] = _db(controls.level_knob_to_db(mix.get("master_level", controls.LEVEL_UNITY)))
        m["main"] = self._main(stats, capacity_ms, ctx)
        m["status"] = self._status_text(stats, ctx)
        m["legend"] = [(D1, tr("D1 {rate} kHz: {device}").format(rate=m["main"].get("rate_text", "-"),
                                                                    device=m["main"].get("device", "")))]
        self.canvas.set_model(m)

    def _main(self, stats, capacity_ms, ctx) -> dict:
        device = ctx.get("main_device") or tr("System default")
        out = {"device": device, "live": True, "state": "ok", "scoped": stats is not None, "lines": [],
               "readout": tr("waiting for playback")}
        if stats is None:
            return out
        if stats.get("output_open") is False:
            out.update(state="bad", lines=[tr("not opened: retried every second")])
            return out
        rate = int(stats.get("rate", 0))
        out["rate_text"] = f"{rate / 1000:g}" if rate else "-"
        if "device_ms" in stats:
            api = ctx.get("api_names", {}).get(stats.get("output_api"), stats.get("output_api", ""))
            mode = tr("Exclusive") if stats.get("exclusive") else tr("Shared")
            out["lines"] = [f"{api} · {mode}",
                            tr("{rate} kHz · {channels} ch · {period} ms").format(
                                rate=out["rate_text"], channels=int(stats.get("channels", 0)),
                                period=f"{float(stats.get('period_ms', 0.0)):.2f}"),
                            tr("buffer {buffer} + device {device} ms").format(
                                buffer=f"{capacity_ms:.0f}", device=f"{float(stats['device_ms']):.1f}")]
            out["signal"] = tr("{rate} kHz · {channels} ch").format(rate=out["rate_text"],
                                                                   channels=int(stats.get("channels", 0)))
        if "hist" in stats and capacity_ms > 0.0:
            v = self._observe(stats, capacity_ms, "device_ms" in stats)
            out.update(hist=v["hist"], bucket=v["bucket"], capacity=capacity_ms, low=v["low"])
            out["readout"] = tr("{ema} / {cap} ms · 1% {low} · drop {drop}").format(
                ema=f"{v['ema']:.1f}", cap=f"{capacity_ms:.0f}", low=f"{v['low']:.1f}", drop=f"{v['dropout']:.0f}")
        if int(stats.get("late", 0)) > 0:
            out["state"] = "warn"
        if float(stats.get("dropout_sec", 0.0)) > 0.0:
            out["state"] = "bad"
        return out

    @staticmethod
    def _status_text(stats, ctx) -> str:
        parts = [tr("Native engine") if ctx.get("native") else tr("Python engine")]
        if stats is not None and "engine_thread" in stats:
            parts.append(tr("engine thread {state}").format(
                state=tr(THREAD_STATE_NAMES.get(stats.get("engine_thread"), THREAD_STATE_NAMES["api"]))))
            parts.append(tr("output thread {state}").format(
                state=tr(THREAD_STATE_NAMES.get(stats.get("output_thread"), THREAD_STATE_NAMES["api"]))))
            parts.append({"off": tr("power throttling off"), "left": tr("power throttling left to the system"),
                          "partly": tr("power throttling off in part"),
                          "refused": tr("power throttling not turned off")}.get(stats.get("power"), ""))
            response = stats.get("response_ms") or {}
            if any(v > 0.0 for v in response.values()):
                parts.append(tr("response play {play} / pause {pause} / seek {seek} ms").format(
                    **{k: (f"{v:.0f}" if v > 0.0 else "-") for k, v in response.items()}))
        port, state = ctx.get("midi") or ("", 0)
        if ctx.get("native"):
            parts.append(tr("MIDI: no input") if not port else
                         {1: tr("MIDI: {port} (open)"), 2: tr("MIDI: {port} (could not be opened)")}.get(
                             int(state), tr("MIDI: {port} (closed)")).format(port=port))
        if stats is None:
            parts.append(tr("waiting for playback"))
        return "  ·  ".join(p for p in parts if p)
