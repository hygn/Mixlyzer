from PySide6 import QtCore
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QGridLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class ParameterOptimizeProgressDialog(QDialog):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(False)

        self.lbl_feature = QLabel("Feature build pending")
        self.lbl_feature.setWordWrap(True)
        self.bar_feature = QProgressBar()
        self.bar_feature.setRange(0, 100)
        self.lbl_optimize = QLabel("Optimize pending")
        self.lbl_optimize.setWordWrap(True)
        self.bar_optimize = QProgressBar()
        self.bar_optimize.setRange(0, 100)
        self.btn_close = QPushButton("Close")
        self.btn_close.setEnabled(False)
        self.btn_close.clicked.connect(self.accept)

        layout = QVBoxLayout(self)
        layout.addWidget(self.lbl_feature)
        layout.addWidget(self.bar_feature)
        layout.addWidget(self.lbl_optimize)
        layout.addWidget(self.bar_optimize)
        layout.addWidget(self.btn_close, alignment=Qt.AlignRight)
        self.resize(520, 210)

    @QtCore.Slot(int, str)
    def set_feature_progress(self, value: int, text: str) -> None:
        self.bar_feature.setValue(max(0, min(100, int(value))))
        self.lbl_feature.setText(str(text or "Building features"))

    @QtCore.Slot(int, str)
    def set_optimize_progress(self, value: int, text: str) -> None:
        self.bar_optimize.setValue(max(0, min(100, int(value))))
        self.lbl_optimize.setText(str(text or "Optimizing"))

    @QtCore.Slot(str)
    def set_failed(self, message: str) -> None:
        self.lbl_optimize.setText("Failed")
        self.btn_close.setEnabled(True)

    @QtCore.Slot(object)
    def set_finished(self, result) -> None:
        self.bar_feature.setValue(100)
        self.bar_optimize.setValue(100)
        skipped_count = len(getattr(result, "skipped_tracks", ()))
        skipped_suffix = f", {skipped_count} skipped" if skipped_count else ""
        metric_kind = "CV" if getattr(result, "cross_validated", True) else "Training"
        if hasattr(result, "top1_accuracy"):
            text = (
                f"Done — {metric_kind} top-1 {result.top1_accuracy:.1%}, "
                f"cross-entropy {result.cross_entropy:.4f}{skipped_suffix}"
            )
        elif hasattr(result, "track_top1_accuracy"):
            text = (
                f"Done — {metric_kind} track phase {result.track_top1_accuracy:.1%}, "
                f"beat top-1 {result.beat_top1_accuracy:.1%}{skipped_suffix}"
            )
        elif hasattr(result, "beat_average_precision"):
            text = (
                f"Done — {metric_kind} grid top-1 {result.grid_top1_accuracy:.1%}, "
                f"beat AP {result.beat_average_precision:.1%}{skipped_suffix}"
            )
        elif hasattr(result, "boundary_average_precision"):
            text = (
                f"Done — {metric_kind} boundary AP {result.boundary_average_precision:.1%}, "
                f"label accuracy {result.label_accuracy:.1%}{skipped_suffix}"
            )
        else:
            text = "Done"
        self.lbl_optimize.setText(text)
        self.btn_close.setEnabled(True)


class BeatParameterOptimizeDialog(QDialog):
    ITEMS = (("onset", "Onset"), ("beat_phase", "Beat Phase"), ("downbeat", "Downbeat"))

    def __init__(self, default_l2: dict[str, float], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Reoptimize Beat Parameters")
        self._checks: dict[str, QCheckBox] = {}
        self._folds: dict[str, QSpinBox] = {}
        self._sweeps: dict[str, QCheckBox] = {}
        self._l2: dict[str, QDoubleSpinBox] = {}

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Models to retrain on the library's beat grids:"))
        for key, name in self.ITEMS:
            check = QCheckBox(name)
            check.toggled.connect(self._update_enabled)
            self._checks[key] = check
            layout.addWidget(check)
        self.cb_backup = QCheckBox("Back up existing parameter files")
        self.cb_backup.setChecked(True)
        layout.addWidget(self.cb_backup)

        self.btn_advanced = QToolButton()
        self.btn_advanced.setText("Advanced")
        self.btn_advanced.setCheckable(True)
        self.btn_advanced.setArrowType(Qt.RightArrow)
        self.btn_advanced.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.btn_advanced.setAutoRaise(True)
        self.btn_advanced.toggled.connect(self._toggle_advanced)
        layout.addWidget(self.btn_advanced)

        self.advanced = QWidget()
        grid = QGridLayout(self.advanced)
        grid.setContentsMargins(16, 0, 0, 0)
        for column, text in enumerate(("", "CV folds", "L2 sweep", "Fixed L2")):
            grid.addWidget(QLabel(text), 0, column)
        for row, (key, name) in enumerate(self.ITEMS, start=1):
            folds = QSpinBox()
            folds.setRange(1, 20)
            folds.setSpecialValueText("Off")
            folds.setToolTip(
                "Cross-validation folds by track. Off: one fit on the whole library."
            )
            sweep = QCheckBox()
            sweep.setToolTip(
                "Pick the L2 among the candidates by cross-validation (needs CV)."
            )
            l2 = QDoubleSpinBox()
            l2.setDecimals(6)
            l2.setRange(1e-6, 10.0)
            l2.setSingleStep(0.0001)
            l2.setValue(float(default_l2.get(key, 0.001)))
            folds.valueChanged.connect(self._update_enabled)
            sweep.toggled.connect(self._update_enabled)
            self._folds[key], self._sweeps[key], self._l2[key] = folds, sweep, l2
            grid.addWidget(QLabel(name), row, 0)
            grid.addWidget(folds, row, 1)
            grid.addWidget(sweep, row, 2, alignment=Qt.AlignCenter)
            grid.addWidget(l2, row, 3)
        self.advanced.setVisible(False)
        layout.addWidget(self.advanced)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self._update_enabled()

    def _toggle_advanced(self, shown: bool) -> None:
        self.btn_advanced.setArrowType(Qt.DownArrow if shown else Qt.RightArrow)
        self.advanced.setVisible(shown)
        self.adjustSize()

    def _update_enabled(self, *_args) -> None:
        for key, _name in self.ITEMS:
            selected = self._checks[key].isChecked()
            cross_validated = self._folds[key].value() >= 2
            if not cross_validated and self._sweeps[key].isChecked():
                self._sweeps[key].setChecked(False)
            self._folds[key].setEnabled(selected)
            self._sweeps[key].setEnabled(selected and cross_validated)
            self._l2[key].setEnabled(
                selected and not self._sweeps[key].isChecked()
            )
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(
            any(check.isChecked() for check in self._checks.values())
        )

    def jobs(self) -> list[tuple[str, str, dict]]:
        jobs = []
        for key, name in self.ITEMS:
            if not self._checks[key].isChecked():
                continue
            folds = self._folds[key].value()
            jobs.append(
                (
                    key,
                    name,
                    {
                        "cv_folds": folds if folds >= 2 else 0,
                        "l2_sweep": bool(
                            folds >= 2 and self._sweeps[key].isChecked()
                        ),
                        "l2_strength": float(self._l2[key].value()),
                    },
                )
            )
        return jobs

    def backup_existing(self) -> bool:
        return self.cb_backup.isChecked()
