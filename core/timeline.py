from PySide6 import QtCore

class TimelineCoordinator(QtCore.QObject):
    """윈도우/센터/현재시간을 관리하고, left-edge(이미지 Rect 평행이동)를 계산."""
    def __init__(self, bus):
        super().__init__()
        self.bus = bus
        self.window_sec: float = 12.0
        self.center_t: float = self.window_sec / 2.0
        self.current_time: float = 0.0

        self.bus.sig_time_changed.connect(self._on_time)
        self.bus.sig_window_changed.connect(self._on_window)
        self.bus.sig_center_changed.connect(self._on_center)

    def left_edge(self) -> float:
        return self.center_t - self.current_time

    def _on_time(self, t: float):
        self.current_time = t

    def _on_window(self, w: float):
        self.window_sec = max(0.2, float(w))
        self.center_t = self.window_sec / 2.0

    def _on_center(self, c: float):
        self.center_t = float(c)