"""A view box's drag held by the left button ends however the button is lost.

Let go over another window (a box opened while the button is held takes the release; a modal one blocks this window
too) or with the app left, the release never reaches the view, and the item may keep the scene's mouse grab: the drag
held on (the deck scratching, the waveform following a scrub) until the next click. So the drag ends where it was
when another window takes the focus or the item loses the grab otherwise than by a release."""
from PySide6 import QtGui


class HeldDrag:
    """Mixed into a view box before its pyqtgraph base: the box says whether its drag is held (``_held``) and ends
    it where it was (``_let_go``); ``_watch_focus()`` from its ``__init__``."""

    def _held(self) -> bool:
        raise NotImplementedError

    def _let_go(self) -> None:
        raise NotImplementedError

    def _watch_focus(self) -> None:
        app = QtGui.QGuiApplication.instance()
        if app is not None:
            app.focusWindowChanged.connect(self._focus_moved)

    def _focus_moved(self, window) -> None:
        if not self._held():
            return
        scene = self.scene()
        views = scene.views() if scene is not None else []
        if not any(view.window().windowHandle() is window for view in views):
            self._end_held()

    def ungrabMouseEvent(self, ev) -> None:
        if self._held():
            self._end_held()        # the grab taken away: a release ends the drag before the scene ungrabs
        super().ungrabMouseEvent(ev)

    def _end_held(self) -> None:
        self._let_go()
        scene = self.scene()
        if scene is not None and scene.mouseGrabberItem() is self:
            self.ungrabMouse()      # else the next press still goes here, and nothing else under it is hovered
