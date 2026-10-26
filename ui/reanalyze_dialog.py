"""Reanalyze Track: the selected tracks' full reanalysis, with some analysis settings set for it only (the saved
settings stay as they are): the BPM range, dynamic beat / downbeat / key analysis, AutoCUE."""

from __future__ import annotations

from PySide6 import QtWidgets

from core.config import analysisconfig
from foundation.i18n import tr, tr_n
from ui.elided_label import ElidedLabel

class ReanalyzeDialog(QtWidgets.QDialog):
    def __init__(self, analysis: analysisconfig, titles: list[str], parent: QtWidgets.QWidget | None = None):
        """``analysis``: the saved settings the fields start from; ``titles``: the tracks reanalyzed."""
        super().__init__(parent)
        self.setWindowTitle(tr("Reanalyze Track"))
        self.setModal(True)

        what = (ElidedLabel(titles[0]) if len(titles) == 1 else
                QtWidgets.QLabel(tr_n("{count} track", "{count} tracks", len(titles)).format(count=len(titles))))
        font = what.font()
        font.setBold(True)
        what.setFont(font)
        warning = QtWidgets.QLabel(tr_n("Its stored analysis and edits are replaced; this cannot be undone.",
                                        "Their stored analysis and edits are replaced; this cannot be undone.",
                                        len(titles)))
        warning.setWordWrap(True)

        self.sp_bpm_min = QtWidgets.QSpinBox()
        self.sp_bpm_min.setRange(60, 400)
        self.sp_bpm_min.setToolTip(tr(
            "The beat tracker reads every tempo in the octave from here (BPM min .. 2 x BPM min);\n"
            "a song outside it is read at half or double tempo."
        ))
        self.lbl_bpm_range = QtWidgets.QLabel()
        self.sp_bpm_min.valueChanged.connect(
            lambda value: self.lbl_bpm_range.setText(
                tr("Reads {low} - {high} BPM").format(low=int(value), high=2 * int(value))))
        self.cb_bpm_dynamic = QtWidgets.QCheckBox(tr("Dynamic Analysis"))
        self.cb_bpm_dynamic.setToolTip(tr(
            "Checked: tempo changes within a track (sections, ramps).\n"
            "Unchecked: one steady BPM for the whole track."
        ))
        self.cb_dynamic_downbeat = QtWidgets.QCheckBox(tr("Dynamic Downbeat"))
        self.cb_dynamic_downbeat.setToolTip(tr(
            "Checked: odd bars and meter changes within a track.\n"
            "Unchecked: one bar length and downbeat phase for the whole track.\n"
            "Always on with Dynamic Analysis."
        ))
        self.cb_key_dynamic = QtWidgets.QCheckBox(tr("Dynamic Analysis"))
        self.cb_key_dynamic.setToolTip(tr(
            "Key changes within a track. Off: one key for the whole track -- the key the dynamic analysis finds in "
            "the intro, outro and interludes (where tracks are mixed), or the key it finds most when a track has none. "
            "Applies to the analysis and the editor's whole-track key reanalysis."
        ))
        self.cmb_autocue = QtWidgets.QComboBox()
        self.cmb_autocue.addItem(tr("CUE point"), "cue_point")
        self.cmb_autocue.addItem("JumpCUE", "jumpcue")
        self.cmb_autocue.addItem(tr("Off"), "off")
        self.cmb_autocue.setToolTip(tr(
            "What the analysis puts at the automatic start position:\n"
            "the lead-in end, an early first downbeat, or 0:00.\n"
            "CUE point: a CUE point (memory cue).\n"
            "JumpCUE: a JumpCUE (hot cue) of its own.\n"
            "Off: nothing."
        ))

        # a dynamic grid always has dynamic bars; static analysis may have either (as in the settings)
        self._static_downbeat = bool(analysis.dynamic_downbeat)

        def follow_dynamic(on: bool) -> None:
            if on:
                self._static_downbeat = self.cb_dynamic_downbeat.isChecked()
                self.cb_dynamic_downbeat.setChecked(True)
            else:
                self.cb_dynamic_downbeat.setChecked(self._static_downbeat)
            self.cb_dynamic_downbeat.setEnabled(not on)

        self.cb_bpm_dynamic.toggled.connect(follow_dynamic)
        self.sp_bpm_min.setValue(int(analysis.bpm_min))
        self.lbl_bpm_range.setText(tr("Reads {low} - {high} BPM").format(low=int(analysis.bpm_min),
                                                                          high=2 * int(analysis.bpm_min)))
        self.cb_dynamic_downbeat.setChecked(bool(analysis.dynamic_downbeat) or bool(analysis.bpm_dynamic))
        self.cb_bpm_dynamic.setChecked(bool(analysis.bpm_dynamic))
        self.cb_dynamic_downbeat.setEnabled(not analysis.bpm_dynamic)
        self.cb_key_dynamic.setChecked(bool(analysis.key_dynamic))
        self.cmb_autocue.setCurrentIndex(max(0, self.cmb_autocue.findData(analysis.autocue)))

        beat = QtWidgets.QGroupBox(tr("Beat"))
        f_beat = QtWidgets.QFormLayout(beat)
        f_beat.addRow(tr("BPM min"), self.sp_bpm_min)
        f_beat.addRow("", self.lbl_bpm_range)
        f_beat.addRow(self.cb_bpm_dynamic)
        f_beat.addRow(self.cb_dynamic_downbeat)
        key = QtWidgets.QGroupBox("Key")
        QtWidgets.QFormLayout(key).addRow(self.cb_key_dynamic)
        autocue = QtWidgets.QGroupBox("AutoCUE")
        QtWidgets.QFormLayout(autocue).addRow(tr("Automatic start"), self.cmb_autocue)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.button(QtWidgets.QDialogButtonBox.Ok).setText(tr("Reanalyze"))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        root = QtWidgets.QVBoxLayout(self)
        root.addWidget(what)
        root.addWidget(warning)
        root.addWidget(beat)
        root.addWidget(key)
        root.addWidget(autocue)
        root.addWidget(buttons)
        self.setMinimumWidth(360)
        self.setMaximumWidth(560)

    def analysis(self) -> dict:
        """The settings set here (analysisconfig fields) for this reanalysis."""
        return {
            "bpm_min": int(self.sp_bpm_min.value()),
            "bpm_dynamic": self.cb_bpm_dynamic.isChecked(),
            "dynamic_downbeat": self.cb_dynamic_downbeat.isChecked(),
            "key_dynamic": self.cb_key_dynamic.isChecked(),
            "autocue": str(self.cmb_autocue.currentData()),
        }

