from collections import deque

from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QSpinBox,
    QWidget,
)



def build_view_tab(dialog) -> None:
    dialog.tab_view = QWidget()
    form = QFormLayout(dialog.tab_view)

    dialog.cb_waveform = QCheckBox("Display waveform")
    dialog.cb_beatgrid = QCheckBox("Display beatgrid")
    dialog.cb_keystrip = QCheckBox("Display keystrip")
    dialog.cb_JumpCUE = QCheckBox("Display JumpCUE")
    dialog.cb_phrase = QCheckBox("Display phrase overlay")
    dialog.cb_use_output_volume_as_peak_meter_input = QCheckBox(
        "Use output volume as peak meter input"
    )
    dialog.cb_reduce_fps_when_occluded = QCheckBox("Reduce FPS when occluded")
    dialog.sp_fps = QSpinBox()
    dialog.sp_fps.setRange(1, 240)
    dialog.sp_fps.setSingleStep(5)

    dialog.ed_record_img_path = QLineEdit()
    dialog.lbl_fps = QLabel("Waiting for FPS...")
    dialog.lbl_fps.setToolTip(
        "Measured from actual pyqtgraph plot repaint intervals."
    )
    dialog.cb_waveform.setToolTip("Show the detailed waveform view.")
    dialog.cb_beatgrid.setToolTip("Show the beat grid view.")
    dialog.cb_keystrip.setToolTip("Show the key strip (detected key over time).")
    dialog.cb_JumpCUE.setToolTip("Show the JumpCUE view (detected repeat jump points).")
    dialog.cb_phrase.setToolTip("Overlay the detected phrases on the views.")
    dialog.cb_use_output_volume_as_peak_meter_input.setToolTip(
        "Checked: the peak meter follows the volume slider (what you hear).\n"
        "Unchecked: the peak meter shows the level after Trim only."
    )
    dialog.cb_reduce_fps_when_occluded.setToolTip(
        "Lower the redraw rate while the window is hidden behind other windows\n"
        "or minimized, to save CPU."
    )
    dialog.sp_fps.setToolTip("Target redraw rate of the views.")
    dialog.ed_record_img_path.setToolTip(
        "Image drawn as the spinning record when a track has no album art."
    )

    grp_display = QGroupBox("Display")
    f_display = QFormLayout(grp_display)
    f_display.addRow(dialog.cb_waveform)
    f_display.addRow(dialog.cb_beatgrid)
    f_display.addRow(dialog.cb_keystrip)
    f_display.addRow(dialog.cb_JumpCUE)
    f_display.addRow(dialog.cb_phrase)
    f_display.addRow(dialog.cb_use_output_volume_as_peak_meter_input)

    grp_performance = QGroupBox("Performance")
    f_performance = QFormLayout(grp_performance)
    f_performance.addRow(dialog.cb_reduce_fps_when_occluded)
    f_performance.addRow("FPS", dialog.sp_fps)
    f_performance.addRow("Actual FPS", dialog.lbl_fps)

    grp_advanced = QGroupBox("Advanced")
    f_advanced = QFormLayout(grp_advanced)
    f_advanced.addRow("Record image path", dialog.ed_record_img_path)

    form.addRow(grp_display)
    form.addRow(grp_performance)
    form.addRow(grp_advanced)
    dialog.tabs.addTab(dialog.tab_view, "View")


class ViewSettingsMixin:
    def _initialize_view_settings(self, bus) -> None:
        self._refresh_samples = deque(maxlen=32)
        if bus is not None:
            bus.sig_ui_draw_interval.connect(self._on_ui_draw_interval)

    def _on_ui_draw_interval(self, dt_ms: float):
        if 1.0 <= dt_ms <= 1000.0:
            self._refresh_samples.append(float(dt_ms))
            avg_ms = sum(self._refresh_samples) / len(self._refresh_samples)
            fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0
            self.lbl_fps.setText(f"{avg_ms:.1f} ms ({fps:.1f} FPS)")
