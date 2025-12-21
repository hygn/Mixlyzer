from PySide6 import QtCore
import pyqtgraph as pg

REGISTRY = {}

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