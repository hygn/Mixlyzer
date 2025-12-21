import pyqtgraph as pg

class NoDragNoWheelViewBox(pg.ViewBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMouseEnabled(x=False, y=False)
        if hasattr(self, "setWheelEnabled"):
            self.setWheelEnabled(False)

    def wheelEvent(self, ev):
        ev.ignore()

    def mouseDragEvent(self, ev, *args, **kwargs):
        ev.ignore()

    def mouseClickEvent(self, ev, *args, **kwargs):
        ev.ignore()