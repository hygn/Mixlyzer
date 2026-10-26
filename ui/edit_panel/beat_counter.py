from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtWidgets

from foundation.beat_geometry import bar_beat_position, downbeat_beat_indices
from ui.edit_panel.icons import icon
from foundation.i18n import tr

class BeatCounterMixin:
    """BeatCounter section: hold Count to count beats passing under the playhead."""

    def _build_beat_counter_ui(self, grid: QtWidgets.QGridLayout) -> list[QtWidgets.QWidget]:
        """BeatCounter widgets in the Phrase page's middle row; returns the edit buttons."""

        self.beat_counter_button = QtWidgets.QToolButton()
        self.beat_counter_button.setIcon(icon("phrase_count"))
        self.beat_counter_button.setIconSize(QtCore.QSize(36, 18))
        self.beat_counter_button.setToolTip(tr("Hold to count beats passing under the playhead"))
        self.beat_counter_button.setAutoRepeat(False)

        self.beat_counter_display = QtWidgets.QLineEdit("0")
        self.beat_counter_display.setReadOnly(True)
        self.beat_counter_display.setFocusPolicy(QtCore.Qt.NoFocus)
        self.beat_counter_display.setAlignment(QtCore.Qt.AlignCenter)
        self.beat_counter_display.setToolTip(tr("Beats counted while the Count button is held"))
        self.beat_counter_display.setFixedSize(self.BTN_WIDTH, self.BTN_HEIGHT)

        self.beat_counter_barbeat_display = QtWidgets.QLineEdit("--")
        self.beat_counter_barbeat_display.setReadOnly(True)
        self.beat_counter_barbeat_display.setFocusPolicy(QtCore.Qt.NoFocus)
        self.beat_counter_barbeat_display.setAlignment(QtCore.Qt.AlignCenter)
        self.beat_counter_barbeat_display.setToolTip(tr("Bar.Beat of the current playhead position"))
        self.beat_counter_barbeat_display.setFixedSize(self.BTN_WIDTH, self.BTN_HEIGHT)

        _AL = QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter
        # the first cells of the Phrase page's middle tool row (after its mode column and rule)
        grid.addWidget(self.beat_counter_button, 1, 2, _AL)
        grid.addWidget(self.beat_counter_display, 1, 3, _AL)
        grid.addWidget(self.beat_counter_barbeat_display, 1, 4, _AL)

        self.beat_counter_button.pressed.connect(self._on_beat_counter_pressed)
        self.beat_counter_button.released.connect(self._on_beat_counter_released)

        return [
            self.beat_counter_button,
        ]

    def _beats_before(self, t: float) -> int:
        """Number of beatgrid beats at or before ``t``."""
        beats = np.asarray(self._beatgrid, dtype=float)
        if beats.size == 0:
            return 0
        return int(np.searchsorted(beats, float(t), side="right"))

    def _on_beat_counter_pressed(self) -> None:
        self._beat_counting = True
        press_beat = self._beats_before(self._current_time)
        self._beat_count_anchor_idx = press_beat
        self._beat_count = 0
        # Bar.Beat counts from the first bar that starts at/after the press
        # point: keep only the downbeat indices from here on, so the first one
        # becomes bar 1 (counting begins once that bar's downbeat is reached).
        db_idx = downbeat_beat_indices(self._beatgrid, self._segments)
        self._bb_downbeats = db_idx[db_idx >= press_beat]
        self._refresh_beat_counter_display()
        self._update_barbeat_display(self._current_time)

    def _on_beat_counter_released(self) -> None:
        self._beat_counting = False

    def _update_beat_counter(self, t: float) -> None:
        if not self._beat_counting:
            return
        count = max(0, self._beats_before(t) - self._beat_count_anchor_idx)
        if count != self._beat_count:
            self._beat_count = count
            self._refresh_beat_counter_display()

    def _refresh_beat_counter_display(self) -> None:
        self.beat_counter_display.setText(str(self._beat_count))

    def _update_barbeat_display(self, t: float) -> None:
        # Edit-panel Bar.Beat only runs while the Count button is held, and
        # starts counting once the first bar after the press begins.
        if not self._beat_counting:
            return
        pos = bar_beat_position(
            self._beatgrid, t, downbeat_indices=self._bb_downbeats
        )
        if pos is None or pos[0] <= 0:
            self.beat_counter_barbeat_display.setText("--")
        else:
            self.beat_counter_barbeat_display.setText(f"{pos[0]}.{pos[1]}")
