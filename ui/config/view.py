from collections import deque
from typing import Callable, Optional

from PySide6.QtWidgets import (
    QCheckBox,
    QButtonGroup,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QRadioButton,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from PySide6 import QtCore

from core.settings.config import phraselabelname, phraselabelnames
from system.i18n import LANGUAGES, system_language, tr
from views.timeline import CLOSE_WINDOW_MAX_SEC, SHORT_WINDOW_MAX_SEC
from views.render.wave import wave_detail_band_ref
from views.waveform import _WaveStyle
from ui.config.scroll import groups_page
from ui.config.wave_bands import F_MAX, F_MIN, BandSpectrum, WavePreview

# the preview spans the main view's width its frame is for: up to SHORT_WINDOW_MAX_SEC, or a wider one
PREVIEW_LONG_SEC = 16.0
WAVEFORM_STYLES = (("rgb", "RGB"), ("3band", "3Band"))   # viewconfig.waveform_style, as named
SPECTRUM_POLL_MS = 33


def build_view_tab(dialog) -> None:
    """Settings > View: its pages (General, Display, Waveform, Phrase labels)."""
    dialog.tab_view = QWidget()

    # each language named in itself; "" follows the operating system's
    dialog.cmb_language = QComboBox()
    dialog.cmb_language.addItem(tr("System ({language})").format(language=LANGUAGES[system_language()]), "")
    for code, name in LANGUAGES.items():
        dialog.cmb_language.addItem(name, code)
    dialog.cmb_language.setToolTip(tr("The language of the app's texts. Applied at the next start."))

    dialog.cb_waveform = QCheckBox(tr("Display waveform"))
    dialog.cb_beatgrid = QCheckBox(tr("Display beatgrid"))
    dialog.cb_keystrip = QCheckBox(tr("Display keystrip"))
    dialog.cb_JumpCUE = QCheckBox(tr("Display JumpCUE"))
    dialog.cb_phrase = QCheckBox(tr("Display phrase overlay"))
    dialog.cb_played_tempo_key = QCheckBox(tr("Show the tempo and key as played"))
    # the waveform style: one radio button each, in a row (the value its button's: WAVEFORM_STYLES)
    dialog.waveform_style_box = QWidget()
    style_row = QHBoxLayout(dialog.waveform_style_box)
    style_row.setContentsMargins(0, 0, 0, 0)
    dialog.waveform_style_group = QButtonGroup(dialog.waveform_style_box)
    dialog.rb_waveform_style = {}
    for value, name in WAVEFORM_STYLES:
        button = QRadioButton(name)
        dialog.rb_waveform_style[value] = button
        dialog.waveform_style_group.addButton(button)
        style_row.addWidget(button)
    style_row.addStretch(1)
    dialog.rb_waveform_style["rgb"].setChecked(True)
    dialog.waveform_style_box.setToolTip(tr(
        "RGB: the waveform colored by its low / mid / high mix.\n"
        "3Band: the low (blue), mid (amber) and high (white) bands drawn as layers."
    ))
    dialog.sp_mainview_wave_frame_ms_close = QDoubleSpinBox()
    dialog.sp_mainview_wave_frame_ms_short = QDoubleSpinBox()
    dialog.sp_mainview_wave_frame_ms_long = QDoubleSpinBox()
    short_max, close_max = f"{SHORT_WINDOW_MAX_SEC:g}", f"{CLOSE_WINDOW_MAX_SEC:g}"
    for spin, tooltip, minimum in (
            (dialog.sp_mainview_wave_frame_ms_close,
             tr("Time resolution of the main view waveform bars when the view is up to {seconds} s wide.\n"
                "Smaller is more detailed.").format(seconds=close_max), 0.1),
            (dialog.sp_mainview_wave_frame_ms_short,
             tr("Time resolution of the main view waveform bars when the view is wider than {close} s and up to "
                "{seconds} s wide.\nSmaller is more detailed.").format(close=close_max, seconds=short_max), 0.3),
            (dialog.sp_mainview_wave_frame_ms_long,
             tr("Time resolution of the main view waveform bars when the view is wider than {seconds} s.\n"
                "Smaller is more detailed.").format(seconds=short_max), 1.0)):
        spin.setRange(minimum, 10.0)
        spin.setDecimals(1)
        spin.setSingleStep(0.1)
        spin.setToolTip(tooltip)
    dialog.sp_mainview_wave_filter_order = QSpinBox(); dialog.sp_mainview_wave_filter_order.setRange(1, 12)
    dialog.sp_mainview_wave_filter_order.setToolTip(tr(
        "Order of the main view waveform's band filters. Higher separates the bands more sharply."
    ))
    dialog.cb_use_output_volume_as_peak_meter_input = QCheckBox(
        tr("Use output volume as peak meter input")
    )
    dialog.cb_reduce_fps_when_occluded = QCheckBox(tr("Reduce FPS when occluded"))
    dialog.sp_fps = QSpinBox()
    dialog.sp_fps.setRange(1, 240)
    dialog.sp_fps.setSingleStep(5)

    dialog.ed_record_img_path = QLineEdit()
    dialog.lbl_fps = QLabel(tr("Waiting for FPS..."))
    dialog.lbl_fps.setToolTip(tr(
        "Measured from actual pyqtgraph plot repaint intervals."
    ))
    dialog.cb_waveform.setToolTip(tr("Show the detailed waveform view."))
    dialog.cb_beatgrid.setToolTip(tr("Show the beat grid view."))
    dialog.cb_keystrip.setToolTip(tr("Show the key strip (detected key over time)."))
    dialog.cb_JumpCUE.setToolTip(tr("Show the JumpCUE view (detected repeat jump points)."))
    dialog.cb_phrase.setToolTip(tr("Overlay the detected phrases on the views."))
    dialog.cb_played_tempo_key.setToolTip(tr(
        "<p>The overview's and the main view's BPM and keys as heard: the tempo changed by the fader, the key moved "
        "by the key shift (and by the tempo with MT off). Off: the track's own.</p>"))
    dialog.cb_use_output_volume_as_peak_meter_input.setToolTip(tr(
        "Checked: the peak meter follows the volume slider (what you hear).\n"
        "Unchecked: the peak meter shows the level after Trim only."
    ))
    dialog.cb_reduce_fps_when_occluded.setToolTip(tr(
        "Lower the redraw rate while the window is hidden behind other windows\n"
        "or minimized, to save CPU."
    ))
    dialog.sp_fps.setToolTip(tr("Target redraw rate of the views."))
    dialog.ed_record_img_path.setToolTip(tr(
        "Image drawn as the spinning record when a track has no album art."
    ))

    grp_language = QGroupBox(tr("Language"))
    f_language = QFormLayout(grp_language)
    f_language.addRow(tr("Language"), dialog.cmb_language)

    grp_display = QGroupBox(tr("Display"))
    f_display = QFormLayout(grp_display)
    f_display.addRow(dialog.cb_waveform)
    f_display.addRow(dialog.cb_beatgrid)
    f_display.addRow(dialog.cb_keystrip)
    f_display.addRow(dialog.cb_JumpCUE)
    f_display.addRow(dialog.cb_phrase)
    f_display.addRow(dialog.cb_played_tempo_key)
    f_display.addRow(dialog.cb_use_output_volume_as_peak_meter_input)

    # the waveforms' low / mid / high ranges: the overview and the preview redraw from the stored bands at once
    def make_band_row(parent_form, label: str, range_tip: str, lo_tip: str, hi_tip: str):
        box = QWidget(); h = QHBoxLayout(box); h.setContentsMargins(0, 0, 0, 0); h.setSpacing(6)
        lo = QDoubleSpinBox(); hi = QDoubleSpinBox()
        box.setToolTip(range_tip)
        lo.setToolTip(lo_tip)
        hi.setToolTip(hi_tip)
        for sp in (lo, hi):
            sp.setDecimals(3); sp.setRange(F_MIN, F_MAX); sp.setSingleStep(1.0); sp.setMinimumWidth(100)
        h.addWidget(QLabel(tr("lo"))); h.addWidget(lo)
        h.addWidget(QLabel(tr("hi"))); h.addWidget(hi)
        parent_form.addRow(label, box)
        return lo, hi
    # the waveform, top to bottom: its style, the spectrum under the bands' sliders, the main view's frames and filter
    # order (a slider beside each field), the preview drawn with them, the bands' edges typed
    grp_wave_bands = QGroupBox(tr("Waveform"))
    grp_wave_bands.setToolTip(tr(
        "Low / mid / high bands of the waveforms' colors."
    ))
    f_wave_bands = QFormLayout(grp_wave_bands)
    f_wave_bands.addRow(tr("Waveform style"), dialog.waveform_style_box)
    dialog.band_spectrum = BandSpectrum()
    dialog.band_spectrum.setToolTip(tr("<p>The master output's spectrum as it plays. Drag a band's edges to set "
                                       "it.</p>"))
    f_wave_bands.addRow(dialog.band_spectrum)
    f_wave_bands.addRow(tr("Frame, close view (ms)"), _with_slider(dialog.sp_mainview_wave_frame_ms_close, 10))
    f_wave_bands.addRow(tr("Frame, short view (ms)"), _with_slider(dialog.sp_mainview_wave_frame_ms_short, 10))
    f_wave_bands.addRow(tr("Frame, long view (ms)"), _with_slider(dialog.sp_mainview_wave_frame_ms_long, 10))
    f_wave_bands.addRow(tr("Filter order"), _with_slider(dialog.sp_mainview_wave_filter_order, 1))
    dialog.wave_preview = WavePreview(dialog._preview_style)
    dialog.wave_preview.setToolTip(tr("<p>The loaded track at the playhead, drawn with these settings. What is "
                                      "already drawn keeps the settings it was drawn with.</p>"))
    f_wave_bands.addRow(dialog.wave_preview)
    dialog.sp_env_lo_lo,  dialog.sp_env_lo_hi  = make_band_row(
        f_wave_bands, tr("Low"), tr("Frequency range (Hz) drawn as the low band of the waveform."),
        tr("Lower edge of the low band (Hz)."), tr("Upper edge of the low band (Hz)."))
    dialog.sp_env_mid_lo, dialog.sp_env_mid_hi = make_band_row(
        f_wave_bands, tr("Mid"), tr("Frequency range (Hz) drawn as the mid band of the waveform."),
        tr("Lower edge of the mid band (Hz)."), tr("Upper edge of the mid band (Hz)."))
    dialog.sp_env_hi_lo,  dialog.sp_env_hi_hi  = make_band_row(
        f_wave_bands, tr("High"), tr("Frequency range (Hz) drawn as the high band of the waveform."),
        tr("Lower edge of the high band (Hz)."), tr("Upper edge of the high band (Hz)."))

    grp_performance = QGroupBox(tr("Performance"))
    f_performance = QFormLayout(grp_performance)
    f_performance.addRow(dialog.cb_reduce_fps_when_occluded)
    f_performance.addRow("FPS", dialog.sp_fps)
    f_performance.addRow(tr("Actual FPS"), dialog.lbl_fps)

    grp_advanced = QGroupBox(tr("Advanced"))
    f_advanced = QFormLayout(grp_advanced)
    f_advanced.addRow(tr("Record image path"), dialog.ed_record_img_path)

    dialog.view_tabs = QTabWidget(dialog.tab_view)
    dialog.view_tabs.addTab(groups_page(grp_language, grp_performance, grp_advanced), tr("General"))
    dialog.view_tabs.addTab(groups_page(grp_display), tr("Display"))
    dialog.view_tabs.addTab(groups_page(grp_wave_bands), tr("Waveform"))
    dialog.view_tabs.addTab(groups_page(_build_phrase_label_names(dialog)), tr("Phrase labels"))
    root = QVBoxLayout(dialog.tab_view)
    root.addWidget(dialog.view_tabs)
    dialog.tabs.addTab(dialog.tab_view, tr("View"))


def _with_slider(spin, scale: int) -> QWidget:
    """``spin`` with a slider beside it over its range (``scale`` slider steps a unit), each following the other:
    the field stays what the settings read."""
    box = QWidget()
    h = QHBoxLayout(box)
    h.setContentsMargins(0, 0, 0, 0)
    h.setSpacing(6)
    slider = QSlider(QtCore.Qt.Horizontal)
    slider.setRange(round(spin.minimum() * scale), round(spin.maximum() * scale))
    slider.setValue(round(spin.value() * scale))
    slider.setToolTip(spin.toolTip())
    slider.valueChanged.connect(lambda v: spin.setValue(v / scale if scale != 1 else v))
    spin.valueChanged.connect(lambda v: slider.setValue(round(v * scale)))
    h.addWidget(slider, 1)
    spin.setFixedWidth(90)
    h.addWidget(spin)
    return box


def _build_phrase_label_names(dialog) -> QGroupBox:
    """Per phrase label: an alias shown in its place, and its overview abbreviation
    (empty: the label, and the first letter of its name)."""
    grp = QGroupBox(tr("Phrase labels"))
    grid = QGridLayout(grp)
    grid.setHorizontalSpacing(12)
    grid.addWidget(QLabel(tr("Alias")), 0, 1)
    grid.addWidget(QLabel(tr("Overview")), 0, 2)
    dialog.phrase_name_edits = {}
    for row, name in enumerate((f for f in phraselabelnames.__dataclass_fields__), start=1):
        label = name.upper()
        alias = QLineEdit()
        alias.setPlaceholderText(label)
        alias.setToolTip(tr("Shown in place of {label} (empty: {label})").format(label=label))
        abbrev = QLineEdit()
        abbrev.setMaxLength(4)
        abbrev.setPlaceholderText(label[:1])
        abbrev.setFixedWidth(64)
        abbrev.setToolTip(tr("{label} in the overview, numbered when it occurs more than once\n"
                             "(empty: the first letter of its name)").format(label=label))
        grid.addWidget(QLabel(label), row, 0)
        grid.addWidget(alias, row, 1)
        grid.addWidget(abbrev, row, 2)
        dialog.phrase_name_edits[name] = (alias, abbrev)
    grid.setColumnStretch(1, 1)
    return grp


class ViewSettingsMixin:
    def _set_language(self, language: str) -> None:
        self.cmb_language.setCurrentIndex(max(0, self.cmb_language.findData(language)))

    def _set_waveform_style(self, style: str) -> None:
        self.rb_waveform_style.get(style, self.rb_waveform_style["rgb"]).setChecked(True)

    def _get_waveform_style(self) -> str:
        return next((value for value, button in self.rb_waveform_style.items() if button.isChecked()), "rgb")

    def _get_language(self) -> str:
        return str(self.cmb_language.currentData() or "")

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
            bus.sig_time_changed.connect(self.wave_preview.set_time)
        # the waveform bands: the spectrum's and the preview's sources (set_waveform_sources), the frame the preview
        # draws with (the one edited last: the short view's at first)
        self._spectrum_native = False
        self._spectrum_switch: Optional[Callable[[bool], None]] = None
        self._spectrum_frames: Optional[Callable] = None
        self._track_model = None
        self._preview_view = "short"     # the view width whose frame was edited last: "close", "short", "long"
        self._spectrum_timer = QtCore.QTimer(self)
        self._spectrum_timer.setInterval(SPECTRUM_POLL_MS)
        self._spectrum_timer.timeout.connect(self._poll_spectrum)
        self.band_spectrum.shown.connect(self._waveform_bands_shown)
        self.band_spectrum.edited.connect(self._band_dragged)
        for band, (lo, hi) in enumerate(self._band_spins()):
            for spin in (lo, hi):
                spin.valueChanged.connect(self._bands_typed)
        self.waveform_style_group.buttonToggled.connect(
            lambda _button, on: on and self.band_spectrum.set_style(self._get_waveform_style()))
        self.sp_mainview_wave_frame_ms_close.valueChanged.connect(lambda _v: self._preview_frame("close"))
        self.sp_mainview_wave_frame_ms_short.valueChanged.connect(lambda _v: self._preview_frame("short"))
        self.sp_mainview_wave_frame_ms_long.valueChanged.connect(lambda _v: self._preview_frame("long"))

    def set_waveform_sources(self, native: bool, spectrum_switch: Callable[[bool], None], spectrum_frames: Callable,
                             model) -> None:
        """The waveform bands' sources: the native engine's master spectrum (switched on while shown; the Python
        engine has none) and the loaded track (DataModel: its audio and overview) for the preview."""
        self._spectrum_native = bool(native)
        self._spectrum_switch = spectrum_switch
        self._spectrum_frames = spectrum_frames
        self._track_model = model
        self.band_spectrum.set_note("" if native else tr("The spectrum needs the native audio engine."))
        model.sig_audio_changed.connect(self._preview_track)
        model.sig_loaded.connect(self._preview_track)
        self._preview_track()

    # ---- the waveform bands ----
    def _band_spins(self):
        return ((self.sp_env_lo_lo, self.sp_env_lo_hi), (self.sp_env_mid_lo, self.sp_env_mid_hi),
                (self.sp_env_hi_lo, self.sp_env_hi_hi))

    def _bands_typed(self, *_args) -> None:
        self.band_spectrum.set_bands([(lo.value(), hi.value()) for lo, hi in self._band_spins()])

    def _band_dragged(self, band: int, low: float, high: float) -> None:
        lo, hi = self._band_spins()[band]
        lo.setValue(low)
        hi.setValue(high)

    def _preview_frame(self, view: str) -> None:
        self._preview_view = view

    def _preview_style(self) -> tuple[_WaveStyle, float]:
        """The preview's waveform settings as edited, and the seconds it spans."""
        model = self._track_model
        wave = model.analysis.wave if model is not None else None
        spin, seconds = {
            "close": (self.sp_mainview_wave_frame_ms_close, float(CLOSE_WINDOW_MAX_SEC)),
            "short": (self.sp_mainview_wave_frame_ms_short, float(SHORT_WINDOW_MAX_SEC)),
            "long": (self.sp_mainview_wave_frame_ms_long, PREVIEW_LONG_SEC),
        }[self._preview_view]
        style = _WaveStyle(
            bands=tuple((float(lo.value()), float(hi.value())) for lo, hi in self._band_spins()),
            filter_order=int(self.sp_mainview_wave_filter_order.value()),
            frame_ms=float(spin.value()),
            waveform_style=self._get_waveform_style(),
            band_ref=wave_detail_band_ref(wave),
        )
        return style, seconds

    def _preview_track(self) -> None:
        model = self._track_model
        if model is not None:
            self.wave_preview.set_track(model.predecoded_pcm, model.predecoded_rate)

    def _waveform_bands_shown(self, shown: bool) -> None:
        """In sight: the engine makes the spectrum and the preview draws; out of it, neither."""
        if shown:
            self._bands_typed()
            self.band_spectrum.set_style(self._get_waveform_style())
            self.wave_preview.start()
        else:
            self.wave_preview.stop()
        if self._spectrum_native and self._spectrum_switch is not None:
            self._spectrum_switch(shown)
            if shown:
                self._spectrum_timer.start()
            else:
                self._spectrum_timer.stop()
                self.band_spectrum.clear()

    def _poll_spectrum(self) -> None:
        if self._spectrum_frames is not None:
            self.band_spectrum.add_frames(self._spectrum_frames())

    def _on_ui_draw_interval(self, dt_ms: float):
        if 1.0 <= dt_ms <= 1000.0:
            self._refresh_samples.append(float(dt_ms))
            avg_ms = sum(self._refresh_samples) / len(self._refresh_samples)
            fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0
            self.lbl_fps.setText(f"{avg_ms:.1f} ms ({fps:.1f} FPS)")
