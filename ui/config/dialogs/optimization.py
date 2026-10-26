import html

from PySide6 import QtCore
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from foundation.i18n import tr, tr_mark


def wrapped_tooltip(text: str) -> str:
    """The tooltip as rich text, so Qt word-wraps it (a plain-text tooltip shows each paragraph on one line)."""
    return "<qt>" + html.escape(text, quote=False).replace("\n", "<br>") + "</qt>"


class ParameterOptimizeProgressDialog(QDialog):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(False)

        self.lbl_feature = QLabel(tr("Feature build pending"))
        self.lbl_feature.setWordWrap(True)
        self.bar_feature = QProgressBar()
        self.bar_feature.setRange(0, 100)
        self.lbl_optimize = QLabel(tr("Optimize pending"))
        self.lbl_optimize.setWordWrap(True)
        self.bar_optimize = QProgressBar()
        self.bar_optimize.setRange(0, 100)
        self.btn_close = QPushButton(tr("Close"))
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
        self.lbl_feature.setText(str(text or tr("Building features")))

    @QtCore.Slot(int, str)
    def set_optimize_progress(self, value: int, text: str) -> None:
        self.bar_optimize.setValue(max(0, min(100, int(value))))
        self.lbl_optimize.setText(str(text or tr("Optimizing")))

    @QtCore.Slot(str)
    def set_failed(self, message: str) -> None:
        self.lbl_optimize.setText(tr("Failed"))
        self.btn_close.setEnabled(True)

    @QtCore.Slot(object)
    def set_finished(self, result) -> None:
        self.bar_feature.setValue(100)
        self.bar_optimize.setValue(100)
        skipped_count = len(getattr(result, "skipped_tracks", ()))
        skipped_suffix = tr(", {count} skipped").format(count=skipped_count) if skipped_count else ""
        kind = tr("CV") if getattr(result, "cross_validated", True) else tr("Training")
        if hasattr(result, "top1_accuracy"):
            text = tr("Done — {kind} top-1 {top1}, cross-entropy {entropy}").format(
                kind=kind, top1=f"{result.top1_accuracy:.1%}", entropy=f"{result.cross_entropy:.4f}") + skipped_suffix
        elif hasattr(result, "track_top1_accuracy"):
            text = tr("Done — {kind} track phase {phase}, beat top-1 {top1}").format(
                kind=kind, phase=f"{result.track_top1_accuracy:.1%}", top1=f"{result.beat_top1_accuracy:.1%}"
            ) + skipped_suffix
        elif hasattr(result, "beat_average_precision"):
            text = tr("Done — {kind} grid top-1 {top1}, beat AP {ap}").format(
                kind=kind, top1=f"{result.grid_top1_accuracy:.1%}", ap=f"{result.beat_average_precision:.1%}"
            ) + skipped_suffix
        elif hasattr(result, "accuracy_after"):
            text = tr("Done — {kind} key accuracy {before} -> {after}").format(
                kind=kind, before=f"{result.accuracy_before:.1%}", after=f"{result.accuracy_after:.1%}")
            text += ("" if getattr(result, "saved", True) else tr(" (not saved)")) + skipped_suffix
        elif hasattr(result, "label_change_f1"):
            text = tr("Done — {kind} key-section change F1 {f1}, label accuracy {accuracy}").format(
                kind=kind, f1=f"{result.key_change_f1:.1%}", accuracy=f"{result.label_accuracy:.1%}") + skipped_suffix
        else:
            text = tr("Done")
        self.lbl_optimize.setText(text)
        self.btn_close.setEnabled(True)


class BeatParameterOptimizeDialog(QDialog):
    # The beat tracker's retrainable models (the texts translated where shown).
    ITEMS = (
        ("beat_decoder", tr_mark("Decoder costs")),
        ("downbeat_ctx", tr_mark("Contextual downbeat model")),
        ("refine_onset", tr_mark("Tempo-change onset model")),
        ("beat_tcn", tr_mark("Beat activation network (TCN)")),
        ("beat_critic", tr_mark("Metronome critic")),
    )
    TIPS = {
        "beat_decoder": tr_mark(
            "How much evidence the decoder needs before it adds a new tempo section, a half-beat reset or an odd "
            "bar. Refit them after retraining the TCN. About 40 minutes (Quick: about 10)."
        ),
        "downbeat_ctx": tr_mark(
            "Gradient-boosted classifier: the probability that a bar starts on each beat, from harmony, bass and "
            "drum changes. A few minutes; at least 50 songs."
        ),
        "refine_onset": tr_mark(
            "Gradient-boosted onset model that places each tempo change exactly. A few minutes; at least 50 songs."
        ),
        "beat_tcn": tr_mark(
            "The network that hears beats and downbeats; every later stage reads it. Fine-tuning about 20 minutes, "
            "from scratch 1-2 hours (at least 200 songs)."
        ),
        "beat_critic": tr_mark(
            "Judges whether a grid sounds right and corrects 3:2 / 4:3 tempo misreads. Fine-tuning about 45 "
            "minutes; from scratch at least 200 songs."
        ),
    }
    # where a training starts: (value, label) per model; the others are always learned anew in minutes
    STARTS = {
        "beat_tcn": (("current", tr_mark("Fine-tune the current model")),
                     ("shipped", tr_mark("Fine-tune the shipped model")), ("scratch", tr_mark("Train from scratch"))),
        "beat_critic": (("current", tr_mark("Fine-tune the current model")),
                        ("shipped", tr_mark("Fine-tune the shipped model")), ("scratch", tr_mark("Train from scratch"))),
        "beat_decoder": (("current", tr_mark("Refit around the current costs")),
                         ("shipped", tr_mark("Refit around the shipped costs"))),
    }
    START_TIP = tr_mark(
        "Current: continue from the model in use (earlier retraining builds up).\n"
        "Shipped: continue from the model that came with Mixlyzer.\n"
        "From scratch: learn from your grids alone; needs many songs."
    )
    INTRO = tr_mark(
        "Retrains the beat analysis's models on the beat grids in your library (the grids you checked and "
        "corrected are the training data). The shipped models are kept (default_* files). With cross-validation a "
        "retrained model is saved only when the held-out songs' grids are not worse. Hover over an item for details."
    )

    def __init__(self, parent=None, track_count: int | None = None,
                 model_paths: dict[str, str] | None = None):
        super().__init__(parent)
        self.setWindowTitle(tr("Reoptimize Beat Models"))
        self._checks: dict[str, QCheckBox] = {}
        self._starts: dict[str, QComboBox] = {}
        self._folds: dict[str, QSpinBox] = {}
        self._track_count = track_count
        self._model_paths = dict(model_paths or {})

        layout = QVBoxLayout(self)
        intro = QLabel(tr(self.INTRO) + ("" if track_count is None else
                                          " " + tr("Songs with beat grids: {count}.").format(count=track_count)))
        intro.setWordWrap(True)
        layout.addWidget(intro)
        rows = QGridLayout()
        for row, (key, name) in enumerate(self.ITEMS):
            check = QCheckBox(tr(name))
            check.setToolTip(wrapped_tooltip(tr(self.TIPS[key])))
            check.toggled.connect(self._update_enabled)
            self._checks[key] = check
            rows.addWidget(check, 2 * row, 0, 1, 2)
            if key in self.STARTS:
                combo = QComboBox()
                for value, label in self.STARTS[key]:
                    combo.addItem(tr(label), value)
                combo.setToolTip(wrapped_tooltip(tr(self.START_TIP) if key != "beat_decoder" else tr(
                    "The candidates are centered on the current or the shipped costs.")))
                combo.currentIndexChanged.connect(self._update_enabled)
                self._starts[key] = combo
                rows.addWidget(combo, 2 * row + 1, 1)
            if key == "beat_decoder":
                self.cb_decoder_quick = QCheckBox(tr("Quick: half-beat reset and odd-bar costs only"))
                self.cb_decoder_quick.setToolTip(wrapped_tooltip(tr(
                    "Keeps the section cost (the slow one to refit) and refits the other two."
                )))
                self.cb_decoder_quick.setChecked(True)
                self.cb_decoder_quick.toggled.connect(self._update_enabled)
                rows.addWidget(self.cb_decoder_quick, 2 * row + 1, 0)
        rows.setColumnMinimumWidth(0, 24)
        layout.addLayout(rows)

        self.lbl_warnings = QLabel()
        self.lbl_warnings.setWordWrap(True)
        self.lbl_warnings.setStyleSheet("QLabel { color: #e0a030; }")
        self.lbl_warnings.setVisible(False)
        layout.addWidget(self.lbl_warnings)

        self.cb_backup = QCheckBox(tr("Back up the files being replaced"))
        self.cb_backup.setToolTip(wrapped_tooltip(tr("Copies each model file before it is overwritten.")))
        self.cb_backup.setChecked(True)
        layout.addWidget(self.cb_backup)

        self.btn_advanced = QToolButton()
        self.btn_advanced.setText(tr("Advanced"))
        self.btn_advanced.setCheckable(True)
        self.btn_advanced.setArrowType(Qt.RightArrow)
        self.btn_advanced.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.btn_advanced.setAutoRaise(True)
        self.btn_advanced.toggled.connect(self._toggle_advanced)
        layout.addWidget(self.btn_advanced)

        self.advanced = QWidget()
        grid = QGridLayout(self.advanced)
        grid.setContentsMargins(16, 0, 0, 0)
        for column, text in enumerate(("", tr("CV folds"))):
            grid.addWidget(QLabel(text), 0, column)
        for row, (key, name) in enumerate(self.ITEMS, start=1):
            folds = QSpinBox()
            folds.setRange(1, 20)
            folds.setSpecialValueText(tr("Off"))
            folds.setToolTip(wrapped_tooltip(tr(
                "Cross-validation folds: the new model is saved only if held-out songs' grids are not worse. "
                "Off: always saved."
            )))
            if key in ("beat_tcn", "beat_critic", "beat_decoder"):
                folds.setValue(5)
            self._folds[key] = folds
            grid.addWidget(QLabel(tr(name)), row, 0)
            grid.addWidget(folds, row, 1)
        self.advanced.setVisible(False)
        layout.addWidget(self.advanced)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        self.buttons.accepted.connect(self._confirm)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self._update_enabled()

    def _start(self, key: str) -> str | None:
        combo = self._starts.get(key)
        return None if combo is None else str(combo.currentData())

    def _warnings(self) -> tuple[list[str], bool]:
        """What to warn about for the chosen trainings, and whether one of them cannot run."""
        from optimizer import model_start as MS
        if self._track_count is None:
            return [], False
        out, blocked = [], False
        for key, name in self.ITEMS:
            if not self._checks[key].isChecked():
                continue
            start = self._start(key)
            kind = {"beat_decoder": "search", "refine_onset": "gbm", "downbeat_ctx": "gbm"}.get(key, MS.kind_of(start or ""))
            messages, stop = MS.warnings(self._track_count, kind, start, self._model_paths.get(key))
            out += [f"{tr(name)}: {m}" for m in messages]
            blocked = blocked or stop
        return out, blocked

    def _confirm(self) -> None:
        messages, blocked = self._warnings()
        if blocked:
            return
        if messages:
            answer = QMessageBox.warning(
                self, tr("Reoptimize Beat Models"), "\n\n".join(messages) + "\n\n" + tr("Continue?"),
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
        self.accept()

    def _toggle_advanced(self, shown: bool) -> None:
        self.btn_advanced.setArrowType(Qt.DownArrow if shown else Qt.RightArrow)
        self.advanced.setVisible(shown)
        self.adjustSize()

    def _update_enabled(self, *_args) -> None:
        for key, _name in self.ITEMS:
            self._folds[key].setEnabled(self._checks[key].isChecked())
        self.cb_decoder_quick.setEnabled(self._checks["beat_decoder"].isChecked())
        for key, combo in self._starts.items():
            combo.setEnabled(self._checks[key].isChecked())
        messages, blocked = self._warnings()
        self.lbl_warnings.setText("\n".join(messages))
        self.lbl_warnings.setVisible(bool(messages))
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(
            any(check.isChecked() for check in self._checks.values()) and not blocked
        )

    def jobs(self) -> list[tuple[str, str, dict]]:
        jobs = []
        for key, name in self.ITEMS:
            if not self._checks[key].isChecked():
                continue
            folds = self._folds[key].value()
            options = {"cv_folds": folds if folds >= 2 else 0}
            if key in self._starts:
                options["start"] = self._start(key)
            if key == "beat_decoder":
                options["quick"] = bool(self.cb_decoder_quick.isChecked())
            jobs.append((key, tr(name), options))
        return jobs

    def backup_existing(self) -> bool:
        return self.cb_backup.isChecked()


class KeyModelOptimizeDialog(QDialog):
    """Options of a key model retraining: where it starts, the held-out check, a backup."""
    STARTS = (("current", tr_mark("Fine-tune the current model")), ("shipped", tr_mark("Fine-tune the shipped model")),
              ("scratch", tr_mark("Train from scratch")))
    INTRO = tr_mark(
        "Retrains the key model on the keys you checked in your library: tracks whose keys you changed from the "
        "automatic result, and tracks analyzed before the automatic result was kept whose keys are not the old "
        "analyzer's raw output. Only keys train it, no chord labels. The shipped model is kept (default_* file). "
        "With cross-validation the retrained model is saved only when the held-out songs' keys are not worse. "
        "Features take a few seconds per song the first time; the training a few minutes."
    )

    def __init__(self, parent=None, track_count: int | None = None, model_path: str | None = None):
        super().__init__(parent)
        self.setWindowTitle(tr("Reoptimize Key Model"))
        self._track_count = track_count
        self._model_path = model_path
        layout = QVBoxLayout(self)
        intro = QLabel(tr(self.INTRO) + ("" if track_count is None else
                                          " " + tr("Analyzed songs: {count}.").format(count=track_count)))
        intro.setWordWrap(True)
        layout.addWidget(intro)
        grid = QGridLayout()
        self.cmb_start = QComboBox()
        for value, label in self.STARTS:
            self.cmb_start.addItem(tr(label), value)
        self.cmb_start.setToolTip(wrapped_tooltip(tr(
            "Current: continue from the model in use (earlier retraining builds up).\n"
            "Shipped: continue from the model that came with Mixlyzer.\n"
            "From scratch: learn from your keys alone; needs many songs."
        )))
        self.cmb_start.currentIndexChanged.connect(self._update_enabled)
        self.sp_folds = QSpinBox()
        self.sp_folds.setRange(1, 20)
        self.sp_folds.setSpecialValueText(tr("Off"))
        self.sp_folds.setValue(5)
        self.sp_folds.setToolTip(wrapped_tooltip(tr(
            "Cross-validation folds: one group of songs is held out, and the new model is saved only if the whole "
            "key analysis gets their keys at least as right as the current one. Off: always saved."
        )))
        grid.addWidget(QLabel(tr("Start")), 0, 0)
        grid.addWidget(self.cmb_start, 0, 1)
        grid.addWidget(QLabel(tr("CV folds")), 1, 0)
        grid.addWidget(self.sp_folds, 1, 1)
        layout.addLayout(grid)
        self.lbl_warnings = QLabel()
        self.lbl_warnings.setWordWrap(True)
        self.lbl_warnings.setStyleSheet("QLabel { color: #e0a030; }")
        self.lbl_warnings.setVisible(False)
        layout.addWidget(self.lbl_warnings)
        self.cb_backup = QCheckBox(tr("Back up the file being replaced"))
        self.cb_backup.setToolTip(wrapped_tooltip(tr("Copies the key model file before it is overwritten.")))
        self.cb_backup.setChecked(True)
        layout.addWidget(self.cb_backup)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self._confirm)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self._update_enabled()

    def _warnings(self) -> tuple[list[str], bool]:
        from optimizer import model_start as MS
        if self._track_count is None:
            return [], False
        start = str(self.cmb_start.currentData())
        # a key model trains in minutes, not hours
        return MS.warnings(self._track_count, MS.kind_of(start), start, self._model_path, MS.KEY_SONGS,
                           scratch_takes_hours=False)

    def _confirm(self) -> None:
        messages, blocked = self._warnings()
        if blocked:
            return
        if messages:
            answer = QMessageBox.warning(self, tr("Reoptimize Key Model"),
                                         "\n\n".join(messages) + "\n\n" + tr("Continue?"),
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer != QMessageBox.Yes:
                return
        self.accept()

    def _update_enabled(self, *_args) -> None:
        messages, blocked = self._warnings()
        self.lbl_warnings.setText("\n".join(messages))
        self.lbl_warnings.setVisible(bool(messages))
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(not blocked)

    def options(self) -> dict:
        folds = self.sp_folds.value()
        return {"start": str(self.cmb_start.currentData()), "cv_folds": folds if folds >= 2 else 0}

    def backup_existing(self) -> bool:
        return self.cb_backup.isChecked()
