from PySide6 import QtCore
import pyqtgraph as pg

REGISTRY = {}

# Main plot lanes (y in plot units, bottom to top): key strip, phrase band of the
# same height, waveform, and a top margin for the JumpCUE blocks / cue markers.
KEY_LANE = (0.0, 0.10)
PHRASE_LANE = (0.10, 0.20)
WAVEFORM_LANE = (0.20, 0.94)

# Edit focus (EventBus.sig_edit_focus_changed): while the edit panel is unlocked, the
# visualizations of the pages it shows are emphasized -- their lines drawn thicker, their
# markers larger -- and the other pages' dimmed; locked, all are drawn as usual. Shapes and
# colours never change. The waveform and the playhead belong to no page.
EDIT_PAGES = ("beatgrid", "key", "phrase", "cue")
DIM_OPACITY = 0.3
FOCUS_SCALE = 2.0    # line width of an emphasized visualization, x its usual
FOCUS_MARKER = 1.3   # marker size of an emphasized visualization, x its usual


def edit_emphasis(page, focus) -> str:
    """``page``'s emphasis under ``focus`` (the pages shown while unlocked): "normal" (no
    focus), "focus" (shown) or "dim"."""
    if not focus or page is None:
        return "normal"
    return "focus" if page in focus else "dim"


def emphasis_opacity(state: str, base: float = 1.0) -> float:
    return base * (DIM_OPACITY if state == "dim" else 1.0)


def emphasis_width(state: str, base: float) -> float:
    return base * (FOCUS_SCALE if state == "focus" else 1.0)


def emphasis_size(state: str, base: float) -> float:
    return base * (FOCUS_MARKER if state == "focus" else 1.0)


def register_view(name: str):
    def deco(cls):
        REGISTRY[name] = cls
        return cls
    return deco

class ViewPlugin(QtCore.QObject):
    # the edit page whose visualization this view draws (EDIT_PAGES), None for none
    EDIT_PAGE = None

    def __init__(self, bus, model, tl):
        super().__init__()
        self.bus = bus
        self.model = model
        self.tl = tl
        self.emphasis = "normal"
        if self.EDIT_PAGE is not None:
            bus.sig_edit_focus_changed.connect(self.apply_edit_focus)

    def apply_edit_focus(self, focus) -> None:
        state = edit_emphasis(self.EDIT_PAGE, focus)
        if state != self.emphasis:
            self.emphasis = state
            self._apply_emphasis()

    def _apply_emphasis(self) -> None:
        """Style the view's items for ``self.emphasis`` (also called after it creates items)."""

    def attach(self, plot: pg.PlotItem): ...
    def detach(self): ...
    def render_initial(self): ...
    def update_time(self, t: float): ...
    def update_window(self, w: float): ...
    def update_features(self, features: dict): ...
