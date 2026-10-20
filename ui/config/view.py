from collections import deque

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QWidget,
)

from core.config import phraselabelname, phraselabelnames
from core.timeline import SHORT_WINDOW_MAX_SEC
from ui.config.scroll import scroll_page


def build_view_tab(dialog) -> None:
    dialog.tab_view = QWidget()
    form = QFormLayout(dialog.tab_view)

    dialog.cb_waveform = QCheckBox("Display waveform")
    dialog.cb_beatgrid = QCheckBox("Display beatgrid")
    dialog.cb_keystrip = QCheckBox("Display keystrip")
    dialog.cb_JumpCUE = QCheckBox("Display JumpCUE")
    dialog.cb_phrase = QCheckBox("Display phrase overlay")
    dialog.cmb_waveform_style = QComboBox()
    dialog.cmb_waveform_style.addItem("RGB", "rgb")
    dialog.cmb_waveform_style.addItem("3Band", "3band")
    dialog.cmb_waveform_style.setToolTip(
        "RGB: the waveform colored by its low / mid / high mix.\n"
        "3Band: the low (blue), mid (amber) and high (white) bands drawn as layers."
    )
    dialog.sp_mainview_wave_frame_ms_short = QDoubleSpinBox()
    dialog.sp_mainview_wave_frame_ms_long = QDoubleSpinBox()
    for spin, views, minimum in ((dialog.sp_mainview_wave_frame_ms_short, f"up to {SHORT_WINDOW_MAX_SEC:g} s", 0.3),
                                 (dialog.sp_mainview_wave_frame_ms_long, f"wider than {SHORT_WINDOW_MAX_SEC:g} s", 1.0)):
        spin.setRange(minimum, 10.0)
        spin.setDecimals(1)
        spin.setSingleStep(0.1)
        spin.setToolTip(
            f"Time resolution of the main view waveform bars when the view is {views} wide.\n"
            "Smaller is more detailed."
        )
    dialog.sp_mainview_wave_filter_order = QSpinBox(); dialog.sp_mainview_wave_filter_order.setRange(1, 12)
    dialog.sp_mainview_wave_filter_order.setToolTip(
        "Order of the main view waveform's band filters. Higher separates the bands more sharply."
    )
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
    f_display.addRow("Waveform style", dialog.cmb_waveform_style)
    f_display.addRow(dialog.cb_use_output_volume_as_peak_meter_input)

    # the waveforms' low / mid / high ranges: the overview and the preview redraw from the stored bands at once
    def make_band_row(parent_form, label: str):
        box = QWidget(); h = QHBoxLayout(box); h.setContentsMargins(0, 0, 0, 0); h.setSpacing(6)
        lo = QDoubleSpinBox(); hi = QDoubleSpinBox()
        box.setToolTip(f"Frequency range (Hz) drawn as the {label.lower()} band of the waveform.")
        lo.setToolTip(f"Lower edge of the {label.lower()} band (Hz).")
        hi.setToolTip(f"Upper edge of the {label.lower()} band (Hz).")
        for sp in (lo, hi):
            sp.setDecimals(3); sp.setRange(0.0, 1e6); sp.setSingleStep(1.0); sp.setMinimumWidth(100)
        h.addWidget(QLabel("lo")); h.addWidget(lo)
        h.addWidget(QLabel("hi")); h.addWidget(hi)
        parent_form.addRow(label, box)
        return lo, hi
    grp_wave_bands = QGroupBox("Waveform bands")
    grp_wave_bands.setToolTip(
        "Low / mid / high bands of the waveforms' colours. The overview and the preview sum them from the\n"
        "12 stored bands, so a change redraws them at once (no re-analysis)."
    )
    f_wave_bands = QFormLayout(grp_wave_bands)
    dialog.sp_env_lo_lo,  dialog.sp_env_lo_hi  = make_band_row(f_wave_bands, "Low")
    dialog.sp_env_mid_lo, dialog.sp_env_mid_hi = make_band_row(f_wave_bands, "Mid")
    dialog.sp_env_hi_lo,  dialog.sp_env_hi_hi  = make_band_row(f_wave_bands, "High")

    grp_mainview_wave = QGroupBox("Main view waveform")
    f_mainview_wave = QFormLayout(grp_mainview_wave)
    f_mainview_wave.addRow("Frame, short view (ms)", dialog.sp_mainview_wave_frame_ms_short)
    f_mainview_wave.addRow("Frame, long view (ms)", dialog.sp_mainview_wave_frame_ms_long)
    f_mainview_wave.addRow("Filter order", dialog.sp_mainview_wave_filter_order)

    grp_performance = QGroupBox("Performance")
    f_performance = QFormLayout(grp_performance)
    f_performance.addRow(dialog.cb_reduce_fps_when_occluded)
    f_performance.addRow("FPS", dialog.sp_fps)
    f_performance.addRow("Actual FPS", dialog.lbl_fps)

    grp_advanced = QGroupBox("Advanced")
    f_advanced = QFormLayout(grp_advanced)
    f_advanced.addRow("Record image path", dialog.ed_record_img_path)

    form.addRow(grp_display)
    form.addRow(grp_wave_bands)
    form.addRow(grp_mainview_wave)
    form.addRow(_build_phrase_label_names(dialog))
    form.addRow(grp_performance)
    form.addRow(grp_advanced)
    dialog.tabs.addTab(scroll_page(dialog.tab_view), "View")


def _build_phrase_label_names(dialog) -> QGroupBox:
    """Per phrase label: an alias shown in its place, and its overview abbreviation
    (empty: the label, and the first letter of its name)."""
    grp = QGroupBox("Phrase labels")
    grid = QGridLayout(grp)
    grid.setHorizontalSpacing(12)
    grid.addWidget(QLabel("Alias"), 0, 1)
    grid.addWidget(QLabel("Overview"), 0, 2)
    dialog.phrase_name_edits = {}
    for row, name in enumerate((f for f in phraselabelnames.__dataclass_fields__), start=1):
        label = name.upper()
        alias = QLineEdit()
        alias.setPlaceholderText(label)
        alias.setToolTip(f"Shown in place of {label} (empty: {label})")
        abbrev = QLineEdit()
        abbrev.setMaxLength(4)
        abbrev.setPlaceholderText(label[:1])
        abbrev.setFixedWidth(64)
        abbrev.setToolTip(f"{label} in the overview, numbered when it occurs more than once\n"
                          "(empty: the first letter of its name)")
        grid.addWidget(QLabel(label), row, 0)
        grid.addWidget(alias, row, 1)
        grid.addWidget(abbrev, row, 2)
        dialog.phrase_name_edits[name] = (alias, abbrev)
    grid.setColumnStretch(1, 1)
    return grp


class ViewSettingsMixin:
    def _set_phrase_label_names(self, names: phraselabelnames) -> None:
        for name, (alias, abbrev) in self.phrase_name_edits.items():
            value = getattr(names, name)
            alias.setText(value.alias)
            abbrev.setText(value.abbrev)

    def _get_phrase_label_names(self) -> phraselabelnames:
        return phraselabelnames(**{
            name: phraselabelname(alias=alias.text().strip(), abbrev=abbrev.text().strip())
            for name, (alias, abbrev) in self.phrase_name_edits.items()
        })

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
