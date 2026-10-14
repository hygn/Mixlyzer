from PySide6 import QtCore

# Main view widths (s) the wheel steps through over the main view.
WINDOW_STEPS_SEC = (3.0, 6.0, 9.0, 12.0, 24.0)
DEFAULT_WINDOW_SEC = 12.0
# Views up to this wide are "short" (the main view waveform's short-view frame).
SHORT_WINDOW_MAX_SEC = 6.0


def step_window(window_sec: float, steps: int) -> float:
    """The main view width ``steps`` WINDOW_STEPS_SEC narrower (< 0) or wider (> 0)
    than ``window_sec``, held at the ends."""
    current = min(range(len(WINDOW_STEPS_SEC)), key=lambda i: abs(WINDOW_STEPS_SEC[i] - float(window_sec)))
    return WINDOW_STEPS_SEC[max(0, min(len(WINDOW_STEPS_SEC) - 1, current + int(steps)))]


class TimelineCoordinator(QtCore.QObject):
    """윈도우/센터/현재시간을 관리하고, left-edge(이미지 Rect 평행이동)를 계산."""
    def __init__(self, bus):
        super().__init__()
        self.bus = bus
        self.window_sec: float = DEFAULT_WINDOW_SEC
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