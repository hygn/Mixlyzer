from PySide6 import QtCore
import pyqtgraph as pg

REGISTRY = {}

# Main plot lanes (y in plot units, bottom to top): key strip, phrase band of the
# same height, waveform, and a top margin for the JumpCUE blocks / cue markers.
KEY_LANE = (0.0, 0.10)
PHRASE_LANE = (0.10, 0.20)
WAVEFORM_LANE = (0.20, 0.94)

def register_view(name: str):
    def deco(cls):
        REGISTRY[name] = cls
        return cls
    return deco

class ViewPlugin(QtCore.QObject):
    def __init__(self, bus, model, tl):
        super().__init__()
        self.bus = bus
        self.model = model
        self.tl = tl

    def attach(self, plot: pg.PlotItem): ...
    def detach(self): ...
    def render_initial(self): ...
    def update_time(self, t: float): ...
    def update_window(self, w: float): ...
    def update_features(self, features: dict): ...