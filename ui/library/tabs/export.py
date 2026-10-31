from __future__ import annotations
import os
import threading
from pathlib import Path
from typing import Optional
from urllib.parse import unquote, urlparse

from PySide6 import QtCore, QtWidgets
from system import os_support
from system.os_support import path_key
from system.atomic_io import atomic_output_path
from core.analysis_library import jobs as analysis_jobs
from core.analysis_library.stored import StoredAnalysis
from core.library.records import TrackRow
from core.library.service import LibraryService
from core.playback.player import native_engine_built
from core.track.model import DataModel
from third_party.midi_tempo_map import TempoMapError, midi_tempo_map
from third_party.rekordbox import sanitize_filename
from core.marks.colors import jump_cue_colors
from ui.config.scroll import scroll_page
from ui.widgets.elided_label import ElidedLabel
from ui.widgets.jumpcue_picker import JumpCuePickButton
from analyzer.music.marks import JumpCue, JumpCueGroup
from ui.render.track_audio import MAX_SECONDS, render_track, write_wav
from system.i18n import tr

# the tempo change in +-% of the track's, the deck's tempo fader's WIDE range (the slider in 0.01 % steps); -100 %
# (a standstill, the fader's bottom) cannot be rendered: its least is a step above it
CHANGE_MAX = 100.0
CHANGE_MIN = -99.99


class _ChangeSpinBox(QtWidgets.QDoubleSpinBox):
    """A tempo change shown signed (+5.00 %), as the deck's fader reads."""

    def textFromValue(self, value: float) -> str:
        return f"{value:+.2f}"


class ExportTrackPanel(QtWidgets.QWidget):
    """The library panel's Export tab: the track's audio rendered (WAV) as it is, with a JumpCUE's jump baked in
    (the cue jumped from and the cue jumped to picked apart, each in a picker of the cues' colours as the key's) and
    / or at the tempo changed; its beatgrid as a DAW's tempo map (a MIDI file: third_party.midi_tempo_map), at the
    track's tempo or the changed one. The tempo is changed by +-% (the deck's fader's WIDE range, -99.99 - +100 %) or
    as the BPM aimed at, from the track's BPM as the library stores it; a file made at the changed tempo is named
    with its target BPM (150BPM; its change in % where the track has no BPM). The marks are
    the deck's (as edited) when it holds the track, else the stored ones. The rendering and the MIDI file are made
    on a thread of their own, never the GUI's. Its file's tags are the Sync tab's (ui.library.tabs.sync)."""

    _sig_written = QtCore.Signal(str, str, str)    # what was made ("audio" / "midi"), the file written, the error

    def __init__(self, library: LibraryService, parent: Optional[QtWidgets.QWidget] = None, *,
                 model: Optional[DataModel] = None) -> None:
        """``library``: the library's owner (core.library.service: the analysis)."""
        super().__init__(parent)
        self._library = library
        self._model: Optional[DataModel] = model
        self._track: Optional[TrackRow] = None
        self._source: Optional[Path] = None
        self._jump_cues: Optional[list[JumpCueGroup]] = None   # the track's JumpCUE groups (None: not read yet)
        self._analysis: Optional[StoredAnalysis] = None        # the track's stored analysis (None: not read yet)
        self._busy: set[str] = set()                           # what is being made ("audio", "midi")
        self._syncing = False                                  # the tempo's fields set from one another
        self._sig_written.connect(self._on_written)

        self.lbl_title = ElidedLabel("-")
        self.lbl_artist = ElidedLabel("-")
        self.lbl_source = ElidedLabel("-", mode=QtCore.Qt.ElideMiddle)

        self.dest_edit = QtWidgets.QLineEdit()
        self.dest_browse = QtWidgets.QPushButton(tr("Browse..."))
        self.dest_browse.clicked.connect(self._on_browse)

        self.status_label = QtWidgets.QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #ff8080;")

        dest_layout = QtWidgets.QHBoxLayout()
        dest_layout.addWidget(self.dest_edit, 1)
        dest_layout.addWidget(self.dest_browse)

        track_form = QtWidgets.QFormLayout()
        track_form.addRow(tr("Title"), self.lbl_title)
        track_form.addRow(tr("Artist"), self.lbl_artist)
        track_form.addRow(tr("Source"), self.lbl_source)
        track_form.addRow(tr("Destination Folder"), dest_layout)

        # the tempo change: the rendering's and the tempo map's (each says whether it takes it)
        tempo = QtWidgets.QGroupBox(tr("Tempo Change"))
        tempo.setToolTip(tr("<p>The tempo the audio is rendered at and the tempo map written for: a change of "
                            "±100% at most from the track's, as the deck's tempo fader in WIDE, or the BPM aimed "
                            "at.</p>"))
        self.sld_tempo = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sld_tempo.setRange(-int(round(CHANGE_MAX * 100)), int(round(CHANGE_MAX * 100)))
        self.sld_tempo.setValue(0)
        self.sld_tempo.setPageStep(100)
        self.spn_tempo = _ChangeSpinBox()
        self.spn_tempo.setDecimals(2)
        self.spn_tempo.setRange(CHANGE_MIN, CHANGE_MAX)
        self.spn_tempo.setSingleStep(0.01)
        self.spn_tempo.setSuffix(" %")
        self.spn_tempo.setValue(0.0)
        self.spn_bpm = QtWidgets.QDoubleSpinBox()
        self.spn_bpm.setDecimals(2)
        self.spn_bpm.setRange(0.01, 9999.99)
        self.spn_bpm.setSingleStep(0.1)
        self.spn_bpm.setSuffix(" BPM")
        self.lbl_source_bpm = QtWidgets.QLabel("")
        self.sld_tempo.valueChanged.connect(lambda v: self._set_tempo(v / 100.0, self.sld_tempo))
        self.spn_tempo.valueChanged.connect(lambda v: self._set_tempo(v, self.spn_tempo))
        self.spn_bpm.valueChanged.connect(self._on_target_bpm)
        tempo_row = QtWidgets.QHBoxLayout()
        tempo_row.addWidget(self.sld_tempo, 1)
        tempo_row.addWidget(self.spn_tempo)
        bpm_row = QtWidgets.QHBoxLayout()
        bpm_row.addWidget(self.spn_bpm)
        bpm_row.addWidget(self.lbl_source_bpm, 1)
        tempo_form = QtWidgets.QFormLayout(tempo)
        tempo_form.addRow(tr("Tempo"), tempo_row)
        tempo_form.addRow(tr("Target BPM"), bpm_row)

        # the track's audio rendered
        render = QtWidgets.QGroupBox(tr("Render Audio"))
        render.setToolTip(tr("<p>The track written as a WAV file: as it is, with a JumpCUE's jump baked in (the "
                             "audio up to the cue jumped from, then on from the cue jumped to), at the tempo "
                             "changed, or both.</p>"))
        self.btn_jump_from = JumpCuePickButton(none_text=tr("No jump"))
        self.btn_jump_to = JumpCuePickButton()
        self.btn_jump_from.setToolTip(tr("The JumpCUE the audio jumps from (No jump: the whole track)."))
        self.btn_jump_to.setToolTip(tr("The JumpCUE the audio jumps to."))
        self.btn_jump_from.cueChanged.connect(self._update_jump_to)
        self.chk_render_tempo = QtWidgets.QCheckBox(tr("At the changed tempo"))
        self.chk_render_tempo.setChecked(True)
        if not native_engine_built():     # the tempo is changed by the native engine, as the deck plays it
            self.chk_render_tempo.setChecked(False)
            self.chk_render_tempo.setEnabled(False)
            self.chk_render_tempo.setToolTip(tr("The tempo is changed by the native audio engine, which is not "
                                                "built."))
        self.btn_render = QtWidgets.QPushButton(tr("Render"))
        self.btn_render.clicked.connect(self.render_audio)
        render_form = QtWidgets.QFormLayout()
        render_form.addRow(tr("Jump from"), self.btn_jump_from)
        render_form.addRow(tr("Jump to"), self.btn_jump_to)
        render_btns = QtWidgets.QHBoxLayout()
        render_btns.addWidget(self.chk_render_tempo)
        render_btns.addStretch(1)
        render_btns.addWidget(self.btn_render)
        render_layout = QtWidgets.QVBoxLayout(render)
        render_layout.addLayout(render_form)
        render_layout.addLayout(render_btns)

        tempo_map = QtWidgets.QGroupBox(tr("DAW Tempo Map"))
        tempo_map.setToolTip(tr(
            "<p>A MIDI file of the track's beatgrid for a DAW: its tempo (ramps too), bars and key changes. Put the "
            "audio at the project's start and import the file's tempo map there.</p>"))
        self.chk_map_tempo = QtWidgets.QCheckBox(tr("At the changed tempo"))
        self.chk_map_tempo.setChecked(True)
        self.chk_map_tempo.setToolTip(tr("<p>The tempo map of the audio rendered at the tempo changed; off, of the "
                                         "track as it is.</p>"))
        self.btn_tempo_map = QtWidgets.QPushButton(tr("Export MIDI"))
        self.btn_tempo_map.clicked.connect(self.export_tempo_map)
        tempo_btns = QtWidgets.QHBoxLayout()
        tempo_btns.addWidget(self.chk_map_tempo)
        tempo_btns.addStretch(1)
        tempo_btns.addWidget(self.btn_tempo_map)
        QtWidgets.QVBoxLayout(tempo_map).addLayout(tempo_btns)

        self.lbl_progress = QtWidgets.QLabel("")
        self.lbl_progress.setWordWrap(True)

        # the fields scroll (as the Edit tab's): the panel never holds the library taller than a small window
        body = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(body)
        layout.addLayout(track_form)
        layout.addWidget(tempo)
        layout.addWidget(render)
        layout.addWidget(tempo_map)
        layout.addWidget(self.lbl_progress)
        layout.addWidget(self.status_label)
        layout.addStretch(1)
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(scroll_page(body), 1)
        self._set_tempo(0.0, None)
        self._populate_jump_cues()

    def track_uid(self) -> Optional[str]:
        return self._track.uid if self._track is not None else None

    # ---- the tempo change ----

    def _source_bpm(self) -> float:
        """The track's BPM as the library stores it (its main tempo); 0: none."""
        bpm = self._track.bpm if self._track is not None else None
        return float(bpm) if bpm and bpm > 0 else 0.0

    def _set_tempo(self, change: float, origin: Optional[QtWidgets.QWidget]) -> None:
        """The tempo set to ``change`` (+-% of the track's) from ``origin`` (the slider, the % field, the target BPM;
        None: the program): the other fields follow (the target BPM from the track's BPM)."""
        if self._syncing:
            return
        self._syncing = True
        try:
            change = min(CHANGE_MAX, max(CHANGE_MIN, float(change)))
            if origin is not self.sld_tempo:
                self.sld_tempo.setValue(int(round(change * 100)))
            if origin is not self.spn_tempo:
                self.spn_tempo.setValue(change)
            source = self._source_bpm()
            if origin is not self.spn_bpm and source:
                self.spn_bpm.setValue(source * (1.0 + change / 100.0))
        finally:
            self._syncing = False

    def _on_target_bpm(self, bpm: float) -> None:
        source = self._source_bpm()
        if source and not self._syncing:
            self._set_tempo(100.0 * (bpm / source - 1.0), self.spn_bpm)

    def _show_source_bpm(self) -> None:
        source = self._source_bpm()
        self.spn_bpm.setEnabled(bool(source))
        self._syncing = True          # a value clamped to the new range is no target typed
        self.spn_bpm.setRange(0.01, max(0.02, 2.0 * source) if source else 9999.99)
        self._syncing = False
        self.lbl_source_bpm.setText(tr("from {bpm} BPM").format(bpm=f"{source:.2f}") if source
                                    else tr("The track has no BPM."))
        self._set_tempo(self.spn_tempo.value(), None)

    def speed(self) -> float:
        """The tempo changed, as times the track's speed (1: as it is): the target BPM over the track's (the BPM
        rendered to exactly as shown; the % field rounds to its 0.01 %), the % where the track has none."""
        source = self._source_bpm()
        if source and self.spn_tempo.value() != 0.0:
            return self.spn_bpm.value() / source
        return 1.0 + self.spn_tempo.value() / 100.0

    def _tempo_suffix(self, speed: float) -> str:
        """A file name's mark of the tempo changed ("" at the track's own): the target BPM (150BPM, 150.25BPM),
        or the change where the track has no BPM (+5.00%)."""
        if speed == 1.0:
            return ""
        if self._source_bpm():
            return "_" + f"{self.spn_bpm.value():.2f}".rstrip("0").rstrip(".") + "BPM"
        return f"_{100.0 * (speed - 1.0):+.2f}%"

    def set_track(self, track: TrackRow) -> None:
        self._track = track
        self._jump_cues = None
        self._analysis = None
        self._show_source_bpm()
        self.lbl_title.setText(track.title or "-")
        self.lbl_artist.setText(track.artist or "-")

        resolved = track_source(track, self._model)
        self._source = resolved if resolved else None
        self.lbl_source.setText(str(resolved) if resolved else (track.path or "-"))
        self.status_label.clear()

        if resolved and resolved.exists():
            self.dest_edit.setText(str(resolved.parent))
        elif resolved:
            self.dest_edit.setText(str(resolved.parent))
            self.status_label.setText(tr("Source file not found at the specified path."))
        else:
            self.dest_edit.clear()
            self.status_label.setText(tr("Source path is invalid."))
        self._populate_jump_cues()
        self._read_jump_cues()

    def _on_browse(self) -> None:
        current = self.dest_edit.text().strip()
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            tr("Select Destination Folder"),
            current or (str(self._source.parent) if self._source else os.getcwd()),
        )
        if directory:
            self.dest_edit.setText(directory)

    def _destination(self) -> Optional[Path]:
        """The destination folder (made if need be), or None (said)."""
        if self._track is None:
            QtWidgets.QMessageBox.warning(self, tr("Export Track"), tr("No track is selected."))
            return None
        dest_text = self.dest_edit.text().strip()
        if not dest_text:
            QtWidgets.QMessageBox.warning(self, tr("Export Track"), tr("Please select a destination folder."))
            return None
        dest_dir = Path(dest_text).expanduser()
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            QtWidgets.QMessageBox.critical(self, tr("Export Track"), tr("Cannot create destination folder:\n{error}").format(error=exc))
            return None
        return dest_dir

    def _overwrite(self, path: Path) -> bool:
        """``path`` may be written: it does not exist, or the user said to overwrite it."""
        if not path.exists():
            return True
        reply = QtWidgets.QMessageBox.question(
            self,
            tr("Export Track"),
            tr("'{name}' already exists.\nOverwrite?").format(name=path.name),
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
        )
        return reply == QtWidgets.QMessageBox.StandardButton.Yes

    def export_tempo_map(self) -> None:
        """The track's beatgrid as a MIDI file (third_party.midi_tempo_map): the deck's marks when it holds the
        track (as edited), else the stored ones."""
        dest_dir = self._destination()
        if dest_dir is None:
            return
        if self._model_holds_track():
            m = self._model
            marks = (m.beats, m.tempo_segments, m.key_segments, m.duration_sec)
        elif self._analysis is not None:
            a = self._analysis
            marks = (a.beats, a.tempo_segments, a.key_segments, a.duration_sec)
        else:        # its stored analysis still being read (set_track asked for it): written when it comes
            uid = self._track.uid

            def came(analysis) -> None:
                if self._track is None or self._track.uid != uid:
                    return
                if analysis is None:
                    QtWidgets.QMessageBox.warning(self, tr("Export Track"),
                                                  tr("The track's analysis could not be read."))
                    return
                self._analysis = analysis
                self.export_tempo_map()

            self._library.read(analysis_jobs.analysis, uid, done=came)
            return
        if "midi" in self._busy:
            return
        title = self._track.title or (self._source.stem if self._source else "")
        speed = self.speed() if self.chk_map_tempo.isChecked() else 1.0
        out_path = dest_dir / f"{sanitize_filename(title or 'track')}_tempo_map{self._tempo_suffix(speed)}.mid"
        if not self._overwrite(out_path):
            return
        self._start("midi", self._tempo_map_job, out_path, marks, title, speed)

    def _tempo_map_job(self, out_path: Path, marks: tuple, title: str, speed: float) -> None:
        """(a thread of its own) The tempo map made and written."""
        try:
            data = midi_tempo_map(*marks[:3], title=title, duration_sec=float(marks[3] or 0.0), speed=speed)
            with atomic_output_path(out_path) as temporary:
                temporary.write_bytes(data)
        except TempoMapError:
            self._sig_written.emit("midi", "", tr("The track has no beatgrid to write."))
        except OSError as exc:
            self._sig_written.emit("midi", "", tr("The file could not be written: {error}").format(error=exc))
        else:
            self._sig_written.emit("midi", str(out_path), "")

    def render_audio(self) -> None:
        """The track's audio rendered to a WAV file in the destination folder: the jump picked baked in (none: the
        whole track), at the tempo changed when asked."""
        if "audio" in self._busy:
            return
        dest_dir = self._destination()
        if dest_dir is None:
            return
        source = self._source
        if source is None or not source.is_file():
            QtWidgets.QMessageBox.warning(self, tr("Export Track"), tr("Source file not found at the specified path."))
            return
        jump_from, jump_to = self.btn_jump_from.cue(), self.btn_jump_to.cue()
        jump = None
        if isinstance(jump_from, JumpCue):
            if not isinstance(jump_to, JumpCue) or jump_to == jump_from:
                QtWidgets.QMessageBox.warning(self, tr("Export Track"),
                                              tr("Pick the JumpCUE jumped to, another than the one jumped from."))
                return
            jump = (float(jump_from.point), float(jump_to.point))
        speed = self.speed() if self.chk_render_tempo.isChecked() and self.chk_render_tempo.isEnabled() else 1.0
        duration = float(self._track.duration or 0.0)
        if duration / speed > MAX_SECONDS:
            QtWidgets.QMessageBox.warning(self, tr("Export Track"), tr(
                "At this tempo the audio would be longer than {minutes} minutes: not rendered.").format(
                minutes=MAX_SECONDS // 60))
            return
        name = sanitize_filename(self._track.title or source.stem)
        if jump is not None:
            name += "_jump_" + sanitize_filename(f"{jump_from.label.strip()}_to_{jump_to.label.strip()}")
        name += self._tempo_suffix(speed) or ("" if jump is not None else "_render")
        out_path = dest_dir / f"{name}.wav"
        if path_key(out_path) == path_key(source):     # never the track's own file
            QtWidgets.QMessageBox.warning(self, tr("Export Track"), tr("The track's own file is not written over."))
            return
        if not self._overwrite(out_path):
            return
        self._start("audio", self._render_job, out_path, str(source), jump, speed)

    def _render_job(self, out_path: Path, source: str, jump: Optional[tuple[float, float]], speed: float) -> None:
        """(a thread of its own) The audio rendered and written."""
        try:
            pcm, rate = render_track(source, jump, speed)
            write_wav(out_path, pcm, rate)
        except Exception as exc:          # a decode, the engine's render or the write: said, nothing left half
            self._sig_written.emit("audio", "", tr("The audio could not be rendered: {error}").format(error=exc))
        else:
            self._sig_written.emit("audio", str(out_path), "")

    def _start(self, kind: str, job, *args) -> None:
        """``job(*args)`` on a thread of its own, its button off and the panel saying so until it is done."""
        self._busy.add(kind)
        (self.btn_render if kind == "audio" else self.btn_tempo_map).setEnabled(False)
        self._show_progress()
        threading.Thread(target=job, args=args, daemon=True).start()

    def _show_progress(self) -> None:
        self.lbl_progress.setText(" ".join(text for kind, text in (
            ("audio", tr("Rendering the audio...")), ("midi", tr("Writing the tempo map..."))) if kind in self._busy))

    def _on_written(self, kind: str, path: str, error: str) -> None:
        self._busy.discard(kind)
        (self.btn_render if kind == "audio" else self.btn_tempo_map).setEnabled(True)
        self._show_progress()
        if error:
            QtWidgets.QMessageBox.warning(self, tr("Export Track"), error)
        else:
            QtWidgets.QMessageBox.information(self, tr("Export Track"),
                                              tr("Export completed:\n{files}").format(files=path))

    def _load_jump_cues(self) -> Optional[list[JumpCueGroup]]:
        """The track's JumpCUE groups: the deck's (as edited) when it holds the track, else the stored ones as read
        on the library thread (set_track asks for them; None until they come)."""
        if self._jump_cues is not None:
            return self._jump_cues
        if self._track is None or not self._track.uid:
            return None
        # Prefer in-memory model data when it matches the requested track.
        if self._model_holds_track():
            self._jump_cues = list(self._model.jump_cues)
            return self._jump_cues
        return None

    def _model_holds_track(self) -> bool:
        return model_holds(self._model, self._track)

    def _read_jump_cues(self) -> None:
        """The track's stored analysis asked of the library's owner (not the deck's track); the JumpCUEs listed
        when it comes."""
        if self._track is None or not self._track.uid or self._load_jump_cues() is not None:
            return
        uid = self._track.uid

        def came(analysis) -> None:
            if self._track is None or self._track.uid != uid:
                return
            if analysis is None:
                self.status_label.setText(tr("The track's analysis could not be read."))
                return
            self._analysis = analysis
            self._jump_cues = list(analysis.jump_cues)
            self._populate_jump_cues()

        self._library.read(analysis_jobs.analysis, uid, done=came)

    def _populate_jump_cues(self) -> None:
        """The JumpCUEs in both pickers, each in its group's colour, by letter (any two may be picked: the jump need
        not be one of a group); the cue jumped from may be none (no jump)."""
        entries = jump_cue_colors(list(self._load_jump_cues() or []))
        self.btn_jump_from.setCues(entries)
        self.btn_jump_to.setCues(entries)
        self.btn_jump_from.setEnabled(bool(entries))
        self._update_jump_to()

    def _update_jump_to(self) -> None:
        """The cue jumped to picked only with a cue jumped from (by default the next one after it)."""
        jump_from = self.btn_jump_from.cue()
        self.btn_jump_to.setEnabled(jump_from is not None)
        cues = self.btn_jump_to.cues()
        if jump_from is not None and self.btn_jump_to.cue() in (None, jump_from) and len(cues) > 1:
            self.btn_jump_to.setCue(cues[(cues.index(jump_from) + 1) % len(cues)])


def resolve_track_path(raw: str | None) -> Optional[Path]:
    """A stored path (a file:// URL too) as a local path; None: none or not a path."""
    if not raw:
        return None
    candidate: Optional[str]
    if raw.startswith("file://"):
        parsed = urlparse(raw)
        path_part = unquote(parsed.path or "")
        if parsed.netloc and parsed.netloc not in ("", "localhost"):
            path_part = f"/{parsed.netloc}{path_part}"
        if os_support.IS_WINDOWS and path_part.startswith("/"):
            path_part = path_part.lstrip("/")
        candidate = path_part
    else:
        candidate = unquote(raw)
    try:
        return Path(candidate).expanduser()
    except (TypeError, ValueError):
        return None


def track_source(track: TrackRow, model: Optional[DataModel]) -> Optional[Path]:
    """The track's audio file: its stored path, else the deck's track's."""
    return resolve_track_path(track.path) or resolve_track_path(
        getattr(model, "properties", {}).get("path") if model else None)


def model_holds(model: Optional[DataModel], track: Optional[TrackRow]) -> bool:
    """The deck holds ``track`` (its marks as edited are the model's)."""
    if model is None or track is None:
        return False
    model_path = (getattr(model, "properties", {}) or {}).get("path")
    return bool(model_path and track.path and path_key(model_path) == path_key(track.path))
