"""A widget's picture painted off the GUI thread: into a QImage on the interface's one paint thread, the widget then
only draws the image (well under a millisecond). For a picture that takes long to paint (the audio overview's model,
the analysis pipeline's image): painted on the GUI thread, each paint held the main view's frames (~20 ms, a dropped
frame each time: a stutter).

    painter = BackgroundPaint(widget, draw)   # draw(p: QPainter, size: QSizeF, state) -> None, run on the thread
    painter.request(state)                    # what to paint (a snapshot: the GUI may change its own after)
    painter.image                             # the last picture (None before the first), drawn in paintEvent

The newest request only: one painted at a time, a request while it runs painted once it ends (the ones between
dropped). ``draw`` reads ``state`` and its own module's constants only, never the widget (its methods are not the
thread's to call); the widget's font, size and pixel ratio go in the request. Qt paints into a QImage, text included,
on any thread."""
from __future__ import annotations

import atexit
from typing import Any, Callable, Optional

from PySide6 import QtCore, QtGui, QtWidgets

Draw = Callable[[QtGui.QPainter, QtCore.QSizeF, Any], None]


class _Worker(QtCore.QObject):
    done = QtCore.Signal(object, int, QtGui.QImage)       # the BackgroundPaint, its request's number, the picture

    @QtCore.Slot(object)
    def paint(self, job) -> None:
        owner, serial, draw, size, ratio, state = job
        image = QtGui.QImage(max(1, round(size.width() * ratio)), max(1, round(size.height() * ratio)),
                             QtGui.QImage.Format_ARGB32_Premultiplied)
        image.setDevicePixelRatio(ratio)
        image.fill(QtCore.Qt.transparent)
        p = QtGui.QPainter(image)
        try:
            draw(p, size, state)
        except Exception as exc:          # a picture that does not paint leaves the last one shown
            print(f"[UI] a picture not painted: {exc!r}")
        finally:
            p.end()
        self.done.emit(owner, serial, image)


class _Thread(QtCore.QObject):
    """The interface's one paint thread (made with the first picture, stopped when the app quits)."""
    send = QtCore.Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.thread = QtCore.QThread()
        self.thread.setObjectName("UiPaintThread")
        self.worker = _Worker()
        self.worker.moveToThread(self.thread)
        self.send.connect(self.worker.paint, QtCore.Qt.QueuedConnection)
        self.worker.done.connect(self._done, QtCore.Qt.QueuedConnection)
        self.thread.start()
        # stopped before Qt destroys it (a QThread destroyed running aborts the process): when the app quits, and
        # when Python ends without an event loop's end (tests, scripts)
        app = QtCore.QCoreApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.stop)
        atexit.register(self.stop)

    @QtCore.Slot(object, int, QtGui.QImage)
    def _done(self, owner, serial, image) -> None:
        owner._painted(serial, image)

    @QtCore.Slot()
    def stop(self) -> None:
        if self.thread.isRunning():
            self.thread.quit()
            self.thread.wait()


_thread: Optional[_Thread] = None


def _paint_thread() -> _Thread:
    global _thread
    if _thread is None:
        _thread = _Thread()
    return _thread


class BackgroundPaint:
    """``widget``'s picture painted by ``draw`` on the paint thread (the module)."""

    def __init__(self, widget: QtWidgets.QWidget, draw: Draw) -> None:
        self._widget = widget
        self._draw = draw
        self.image: Optional[QtGui.QImage] = None
        self._serial = 0
        self._busy = False
        self._pending = None            # the newest request while one is painted: (size, ratio, state)

    def request(self, state: Any, size: Optional[QtCore.QSizeF] = None) -> None:
        """Paint ``state`` at the widget's size (or ``size``) and pixel ratio."""
        w = self._widget
        job = (QtCore.QSizeF(size if size is not None else QtCore.QSizeF(w.size())), float(w.devicePixelRatioF()),
               state)
        if self._busy:
            self._pending = job
            return
        self._send(job)

    def _send(self, job) -> None:
        size, ratio, state = job
        self._busy = True
        self._serial += 1
        _paint_thread().send.emit((self, self._serial, self._draw, size, ratio, state))

    def _painted(self, serial: int, image: QtGui.QImage) -> None:
        self._busy = False
        if serial == self._serial:
            self.image = image
            if self._widget is not None:
                try:
                    self._widget.update()
                except RuntimeError:        # the widget is gone
                    self._widget = None
                    return
        if self._pending is not None:
            job, self._pending = self._pending, None
            self._send(job)

    def draw_on(self, p: QtGui.QPainter, rect: QtCore.QRectF) -> bool:
        """The last picture drawn into ``rect`` (scaled while a new size is painted); False before the first."""
        if self.image is None:
            return False
        p.drawImage(rect, self.image)
        return True
