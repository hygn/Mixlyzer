from __future__ import annotations

import hashlib
import traceback
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PySide6 import QtCore

from core import tieup
from core.analysis_library import jobs as analysis_jobs
from core.library import jobs
from core.library.records import TrackRow
from third_party.rekordbox import (
    RekordboxXmlDocument,
    build_rekordbox_track_element,
    rekordbox_track_key_for_row,
)
from utils.atomic_io import atomic_write_text
from foundation.i18n import tr


@dataclass(frozen=True)
class RekordboxSyncRequest:
    library_dir: Path
    xml_path: Path
    mode: str
    ramp_tolerance_ms: float = 0.0
    comment_tieups: bool = False     # each track's comment with its song's tie-ups after it (core.tieup)
    full_rebuild: bool = False
    rows: tuple[TrackRow, ...] = ()
    uid: str = ""
    show_progress: bool = False


@dataclass(frozen=True)
class RekordboxSyncResult:
    output_path: Path
    processed_count: int
    reused_count: int
    entry_count: int


class RekordboxSyncWorker(QtCore.QObject):
    progress = QtCore.Signal(int, str)
    finished = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, request: RekordboxSyncRequest, library, parent=None) -> None:
        """``library``: the library's owner (core.library.service), asked for the rows and features."""
        super().__init__(parent)
        self._request = request
        self._library = library

    @QtCore.Slot()
    def run(self) -> None:
        try:
            engine = _RekordboxSyncEngine(self._request, self.progress.emit, self._library)
            self.finished.emit(engine.run())
        except Exception:
            self.failed.emit(traceback.format_exc())


class _RekordboxSyncEngine:
    def __init__(
        self,
        request: RekordboxSyncRequest,
        progress: Callable[[int, str], None],
        library,
    ) -> None:
        """``library``: the library's owner: every read of the library asked of it (LibraryService.call, from the
        sync's thread), the XML written here."""
        self.request = request
        self.progress = progress
        self.library = library
        self._track_percent: Optional[int] = None
        self._suffixes: Optional[dict[str, str]] = None   # uid -> its tie-ups' comment suffix, read once a run

    def _track_progress(self, index: int, total: int, text: str) -> None:
        """A track's progress, reported when its percent changes: a report is a queued event
        for the GUI, and a library has thousands of tracks."""
        percent = 5 + int(round(index * 85 / max(total, 1)))
        if percent != self._track_percent or index == total:
            self._track_percent = percent
            self.progress(percent, text)

    def _with_tieups(self, row: TrackRow) -> TrackRow:
        """The row as the XML writes it: with the comment option, its song's tie-ups after its comment."""
        if not self.request.comment_tieups:
            return row
        if self._suffixes is None:
            self._suffixes = self.library.call(jobs.comment_suffixes)
        suffix = self._suffixes.get(str(row.uid or ""), "")
        return replace(row, comment=tieup.with_suffix(row.comment or "", suffix)) if suffix else row

    def run(self) -> RekordboxSyncResult:
        self.request.xml_path.parent.mkdir(parents=True, exist_ok=True)
        if self.request.mode == "library":
            self.progress(1, tr("Loading Mixlyzer library"))
            rows = [self._with_tieups(row) for row in self._load_library_rows()]
            return self._sync_library(rows)
        if self.request.mode == "incremental":
            return self._sync_incremental([self._with_tieups(row) for row in self.request.rows])
        if self.request.mode == "track":
            return self._sync_track(self.request.uid)
        raise ValueError(f"Unsupported Rekordbox sync mode: {self.request.mode}")

    def _sync_library(self, rows: list[TrackRow]) -> RekordboxSyncResult:
        self.progress(4, tr("Loading Rekordbox XML"))
        document = RekordboxXmlDocument.load(
            self.request.xml_path, force_new=self.request.full_rebuild
        )
        nodes = []
        reused = 0
        total = len(rows)
        for index, row in enumerate(rows, start=1):
            track_key = rekordbox_track_key_for_row(row)
            track_hash = self._track_hash(row)
            previous = document.existing_track(track_key)
            if (
                previous is not None
                and not self.request.full_rebuild
                and str(previous.attrib.get("MixlyzerHash", "")) == track_hash
            ):
                node = previous
                reused += 1
            else:
                node = self._build_track(row, track_hash)
            nodes.append(node)
            self._track_progress(index, total, tr("Syncing tracks {done}/{total}").format(done=index, total=total))
        document.replace_tracks(nodes)
        self._write(document)
        return RekordboxSyncResult(
            self.request.xml_path, total, reused, len(nodes)
        )

    def _sync_incremental(self, rows: list[TrackRow]) -> RekordboxSyncResult:
        self.progress(4, tr("Loading Rekordbox XML"))
        document = RekordboxXmlDocument.load(
            self.request.xml_path, force_new=False
        )
        total = len(rows)
        for index, row in enumerate(rows, start=1):
            document.remove_track(rekordbox_track_key_for_row(row))
            document.append_track(self._build_track(row, self._track_hash(row)))
            self._track_progress(index, total, tr("Syncing changed tracks {done}/{total}").format(done=index, total=total))
        entry_count = document.entry_count
        self._write(document)
        return RekordboxSyncResult(self.request.xml_path, total, 0, entry_count)

    def _sync_track(self, uid: str) -> RekordboxSyncResult:
        self.progress(4, tr("Loading Rekordbox XML"))
        document = RekordboxXmlDocument.load(
            self.request.xml_path, force_new=False
        )
        row = self._load_track_row(uid)
        if row is not None:
            row = self._with_tieups(row)
        document.remove_track(uid)
        processed = 0
        if row is not None:
            self.progress(45, tr("Building Rekordbox track entry"))
            document.append_track(self._build_track(row, self._track_hash(row)))
            processed = 1
        entry_count = document.entry_count
        self._write(document)
        return RekordboxSyncResult(self.request.xml_path, processed, 0, entry_count)

    def _build_track(self, row: TrackRow, track_hash: str):
        return build_rekordbox_track_element(
            row,
            self._load_features(row.uid),
            resolve_path=self._resolve_audio_path,
            extra_attributes={"MixlyzerUID": str(row.uid or ""), "MixlyzerHash": track_hash},
            ramp_tolerance_sec=max(0.0, float(self.request.ramp_tolerance_ms)) / 1000.0,
        )

    def _write(self, document: RekordboxXmlDocument) -> None:
        self.progress(92, tr("Formatting Rekordbox XML"))
        xml_text = document.to_xml()
        self.progress(97, tr("Writing Rekordbox XML"))
        atomic_write_text(self.request.xml_path, xml_text)
        self.progress(100, tr("Rekordbox XML sync complete"))

    def _load_features(self, uid: Optional[str]) -> dict[str, np.ndarray]:
        if not uid:
            return {}
        return self.library.call(analysis_jobs.features, uid) or {}

    def _track_hash(self, row: TrackRow) -> str:
        parts = [str(v) for k, v in sorted(asdict(row).items())]
        parts.append(f"ramp_tolerance_ms={float(self.request.ramp_tolerance_ms):g}")   # the TEMPO steps
        uid = str(row.uid or "").strip()
        stamp = self.library.call(analysis_jobs.features_stamp, uid) if uid else None
        if stamp is not None:
            parts += [str(stamp[0]), str(stamp[1])]
        digest = hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()
        return digest

    def _load_library_rows(self) -> list[TrackRow]:
        return self.library.call(jobs.all_tracks)

    def _load_track_row(self, uid: str) -> TrackRow | None:
        return self.library.call(jobs.track, uid)

    @staticmethod
    def _resolve_audio_path(path: str | None) -> Optional[Path]:
        raw = str(path or "").strip()
        if not raw:
            return None
        # Intentionally returns the candidate even when it does not exist:
        # callers re-check .exists() and fall back to the stored path/size.
        return Path(raw).expanduser()
