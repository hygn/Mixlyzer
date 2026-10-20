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
        elif hasattr(result, "label_change_f1"):
            text = (
                f"Done — {metric_kind} key-section change F1 {result.key_change_f1:.1%}, "
                f"label accuracy {result.label_accuracy:.1%}{skipped_suffix}"
            )
        else:
            text = "Done"
        self.lbl_optimize.setText(text)
        self.btn_close.setEnabled(True)


class BeatParameterOptimizeDialog(QDialog):
    # The beat tracker's retrainable models.
    ITEMS = (
        ("beat_decoder", "Decoder costs"),
        ("downbeat_ctx", "Contextual downbeat model"),
        ("refine_onset", "Tempo-change onset model"),
        ("beat_tcn", "Beat activation network (TCN)"),
        ("beat_critic", "Metronome critic"),
    )
    TIPS = {
        "beat_decoder": (
            "The three costs the decoder charges, in units of beat evidence (the network's activation summed over "
            "the beats), before it accepts a structural change in the grid:\n"
            "- section cost: a new tempo section (a tempo change);\n"
            "- half-beat reset cost: a jump of half a beat while the tempo stays the same;\n"
            "- odd-bar cost: a bar of a different length (e.g. a 2-beat bar in 4/4).\n"
            "Higher costs give steadier grids that miss real changes; lower costs follow real changes but also make "
            "false ones. Refitting tracks every song with combinations of candidate costs and keeps the combination "
            "with the least time where your grid and the analysis disagree (beats off by more than 25 ms, half or "
            "double tempo allowed; downbeats off by more than 70 ms).\n"
            "The costs belong to the activation network: refit them after retraining it. No training data is needed "
            "beyond your grids; about 40 minutes for 400 songs (Quick: about 10)."
        ),
        "downbeat_ctx": (
            "A gradient-boosted classifier that estimates, for each beat, the probability that a bar starts there, "
            "from the frame features around the beat (bass, harmony, drums) and their differences to the "
            "neighbouring beats. The bar decoding combines it with the network's downbeat activation.\n"
            "Retraining fits it anew on the beats and downbeats of your grids (a few minutes; needs at least 50 songs)."
        ),
        "refine_onset": (
            "A gradient-boosted onset model used to place each tempo change exactly: where a new tempo section starts "
            "is moved to the onset evidence of this model.\n"
            "Retraining fits it anew on the beats of your grids (a few minutes; needs at least 50 songs)."
        ),
        "beat_tcn": (
            "The temporal convolutional network (three independently trained copies) that turns the frame features "
            "into a beat and a downbeat activation per frame, hearing about 12 s of context on each side. Every later "
            "stage reads its activation.\n"
            "Fine-tuning continues its training on your grids (about 20 minutes); training from scratch starts from "
            "random weights (one to two hours, at least 200 songs). With cross-validation, a group of songs is held "
            "out and the whole analysis is run with the old and the new network; the new one is saved only if those "
            "songs' grids are not worse."
        ),
        "beat_critic": (
            "A small convolutional network (five copies) that lays the music out in the time of a candidate grid, bar "
            "by bar, and judges whether the grid sounds right. It decides between a decoded tempo and its 3:2, 2:3, "
            "4:3 and 3:4 readings.\n"
            "Training examples are windows of your grids (right) and of deliberately wrong grids made from them "
            "(half/quarter-beat shifts, ratio readings, drifts) plus the decoder's own mistakes. Fine-tuning takes "
            "about 45 minutes; from scratch longer and needs at least 200 songs. Saved only if the held-out songs' "
            "grids are not worse."
        ),
    }
    # where a training starts: (value, label) per model; the others are always learned anew in minutes
    STARTS = {
        "beat_tcn": (("current", "Fine-tune the current model"), ("shipped", "Fine-tune the shipped model"),
                     ("scratch", "Train from scratch")),
        "beat_critic": (("current", "Fine-tune the current model"), ("shipped", "Fine-tune the shipped model"),
                        ("scratch", "Train from scratch")),
        "beat_decoder": (("current", "Refit around the current costs"), ("shipped", "Refit around the shipped costs")),
    }
    START_TIP = (
        "Fine-tune the current model: continue from the model in use; if it was already retrained, the changes "
        "build on each other.\n"
        "Fine-tune the shipped model: continue from the model that came with Mixlyzer (kept as default_*, never "
        "overwritten); an earlier retrained model is replaced.\n"
        "Train from scratch: random initial weights, learned from your grids alone. The shipped models were "
        "trained on about 340 songs; with fewer songs a new model is usually worse."
    )
    INTRO = (
        "Retrains the beat analysis's models on the beat grids in your library (the grids you checked and "
        "corrected are the training data). The shipped models are kept (default_* files). With cross-validation a "
        "retrained model is saved only when the held-out songs' grids are not worse. Hover over an item for details."
    )

    def __init__(self, parent=None, track_count: int | None = None,
                 model_paths: dict[str, str] | None = None):
        super().__init__(parent)
        self.setWindowTitle("Reoptimize Beat Models")
        self._checks: dict[str, QCheckBox] = {}
        self._starts: dict[str, QComboBox] = {}
        self._folds: dict[str, QSpinBox] = {}
        self._track_count = track_count
        self._model_paths = dict(model_paths or {})

        layout = QVBoxLayout(self)
        intro = QLabel(self.INTRO + ("" if track_count is None else f" Songs with beat grids: {track_count}."))
        intro.setWordWrap(True)
        layout.addWidget(intro)
        rows = QGridLayout()
        for row, (key, name) in enumerate(self.ITEMS):
            check = QCheckBox(name)
            check.setToolTip(self.TIPS[key])
            check.toggled.connect(self._update_enabled)
            self._checks[key] = check
            rows.addWidget(check, 2 * row, 0, 1, 2)
            if key in self.STARTS:
                combo = QComboBox()
                for value, label in self.STARTS[key]:
                    combo.addItem(label, value)
                combo.setToolTip(self.START_TIP if key != "beat_decoder" else (
                    "The candidate costs are centred on, and ties keep, the current costs or the shipped ones "
                    "(default_beat_decoder.json)."))
                combo.currentIndexChanged.connect(self._update_enabled)
                self._starts[key] = combo
                rows.addWidget(combo, 2 * row + 1, 1)
            if key == "beat_decoder":
                self.cb_decoder_quick = QCheckBox("Quick: half-beat reset and odd-bar costs only")
                self.cb_decoder_quick.setToolTip(
                    "Keeps the section cost and refits the half-beat reset and odd-bar costs. The section cost is the "
                    "expensive one (every value needs the section decoding and everything after it again); the other "
                    "two reuse it, so this takes about a quarter of the time."
                )
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

        self.cb_backup = QCheckBox("Back up the files being replaced")
        self.cb_backup.setToolTip("Copies each model file that a retraining overwrites before it is written.")
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
        for column, text in enumerate(("", "CV folds")):
            grid.addWidget(QLabel(text), 0, column)
        for row, (key, name) in enumerate(self.ITEMS, start=1):
            folds = QSpinBox()
            folds.setRange(1, 20)
            folds.setSpecialValueText("Off")
            folds.setToolTip(
                "Cross-validation: the songs are split into this many groups by artist and title. For the networks "
                "and the decoder costs, held-out songs are tracked with the old and the new model and the new one is "
                "saved only if their grids are not worse. Off: no check, the result is always saved."
            )
            if key in ("beat_tcn", "beat_critic", "beat_decoder"):
                folds.setValue(5)
            self._folds[key] = folds
            grid.addWidget(QLabel(name), row, 0)
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
            out += [f"{name}: {m}" for m in messages]
            blocked = blocked or stop
        return out, blocked

    def _confirm(self) -> None:
        messages, blocked = self._warnings()
        if blocked:
            return
        if messages:
            answer = QMessageBox.warning(
                self, "Reoptimize Beat Models", "\n\n".join(messages) + "\n\nContinue?",
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
            jobs.append((key, name, options))
        return jobs

    def backup_existing(self) -> bool:
        return self.cb_backup.isChecked()
