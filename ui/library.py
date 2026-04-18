from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
from PySide6 import QtCore, QtGui, QtWidgets
import os, csv
from utils.labels import idx_to_labels
from core.library_handler import TrackRow, LibraryDB
from core.analysis_lib_handler import FeatureNPZStore
from core.event_bus import EventBus
from core.config import config
from core.model import DataModel
from ui.edit_song import EditSongDialog
from ui.export import ExportTrackDialog


# Data schema
@dataclass
class Track:
    track_id: str
    title: str
    artist: str
    album: str = ""
    bpm: Optional[float] = None
    key: Optional[int] = None
    duration_sec: Optional[float] = None
    rating: int = 0
    added: Optional[QtCore.QDateTime] = None
    path: str = ""
    comment: str = ""

    _disp_bpm: str = field(init=False, default="--")
    _disp_key_camelot: str = field(init=False, default="--")
    _disp_duration: str = field(init=False, default="00:00")
    _disp_rating: str = field(init=False, default="☆☆☆☆☆")
    _disp_added: str = field(init=False, default="")

    def __post_init__(self):
        # BPM
        self._disp_bpm = f"{self.bpm:.2f}" if isinstance(self.bpm, (int, float)) else "--"
        # Key
        try:
            if isinstance(self.key, int) and 0 <= self.key <= 23:
                self._disp_key_camelot = idx_to_labels(self.key)[0]
        except Exception:
            pass
        # Duration
        if self.duration_sec and self.duration_sec > 0:
            s = int(round(self.duration_sec))
            m, s = divmod(s, 60)
            h, m = divmod(m, 60)
            self._disp_duration = f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
        # Rating
        r = max(0, min(5, int(self.rating or 0)))
        self._disp_rating = "★"*r + "☆"*(5-r)
        # Added
        if self.added:
            self._disp_added = self.added.toString("yyyy-MM-dd HH:mm")

def row_to_track(row: TrackRow) -> Track:
    added_qt = QtCore.QDateTime.fromSecsSinceEpoch(int(row.added_ts)) if row.added_ts else None
    return Track(
        track_id=row.path, title=row.title or "", artist=row.artist or "",
        album=row.album or "", bpm=row.bpm, key=row.key, duration_sec=row.duration,
        rating=int(row.rating or 0), added=added_qt, path=row.path, comment=row.comment or "",
    )
class FastDelegate(QtWidgets.QStyledItemDelegate):
    def paint(self, painter, option, index):
        painter.save()
        # Highlight when selected
        if option.state & QtWidgets.QStyle.State_Selected:
            painter.fillRect(option.rect, option.palette.highlight())
            painter.setPen(option.palette.highlightedText().color())
        else:
            painter.setPen(QtGui.QColor("#ffffff"))
        text = index.data(QtCore.Qt.DisplayRole)
        if text:
            painter.drawText(option.rect.adjusted(4, 0, -4, 0), QtCore.Qt.AlignVCenter | QtCore.Qt.TextSingleLine, str(text))
        painter.restore()
# QTableWidget-based implementation
class LibraryWidget(QtWidgets.QWidget):
    HEADERS = ["Title", "Artist", "Album", "BPM", "Key", "Duration", "Rating", "Added", "Path", "Comment"]
    SEARCH_MODES = ["All"] + HEADERS

    def __init__(self, bus: EventBus, cfg: config, model: DataModel | None = None, parent=None):
        super().__init__(parent)
        self.bus = bus
        self.cfg = cfg
        self.model = model
        self._external_sync_enabled = bool(getattr(cfg, "externalsyncconfig", None) and cfg.externalsyncconfig.enabled)
        self._all_tracks: List[Track] = []
        self._edit_dlg: EditSongDialog | None = None
        self._export_dlg: ExportTrackDialog | None = None

        # Search UI
        self.e_search = QtWidgets.QLineEdit(placeholderText="Search (Title / Artist / Key / ...)")
        self.cb_column = QtWidgets.QComboBox()
        self.cb_column.addItems(self.SEARCH_MODES)

        self.transition_box = QtWidgets.QWidget()
        transition_lay = QtWidgets.QGridLayout(self.transition_box)
        transition_lay.setContentsMargins(0, 0, 0, 0)
        transition_lay.setHorizontalSpacing(6)
        transition_lay.setVerticalSpacing(6)

        self.lbl_transition_bpm = QtWidgets.QLabel("BPM Transition")
        self.lbl_transition_bpm.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.lbl_transition_bpm.setMinimumWidth(110)
        self.cb_transition_bpm = QtWidgets.QCheckBox("BPM")
        self.cb_transition_bpm.setChecked(False)
        self.lbl_from_bpm = QtWidgets.QLabel("From BPM")
        self.lbl_from_bpm.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.lbl_from_bpm.setMinimumWidth(72)
        self.sp_from_bpm = QtWidgets.QDoubleSpinBox()
        self.sp_from_bpm.setRange(0.0, 400.0)
        self.sp_from_bpm.setDecimals(2)
        self.sp_from_bpm.setValue(125.0)
        self.lbl_bpm_arrow = QtWidgets.QLabel("→")
        self.lbl_bpm_arrow.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_bpm_arrow.setMinimumWidth(24)
        self.sp_to_bpm = QtWidgets.QDoubleSpinBox()
        self.sp_to_bpm.setRange(0.0, 400.0)
        self.sp_to_bpm.setDecimals(2)
        self.sp_to_bpm.setValue(129.0)
        self.lbl_bpm_tol = QtWidgets.QLabel("Tolerance %p")
        self.lbl_bpm_tol.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.lbl_bpm_tol.setMinimumWidth(92)
        self.sp_bpm_tolerance = QtWidgets.QDoubleSpinBox()
        self.sp_bpm_tolerance.setRange(0.0, 100.0)
        self.sp_bpm_tolerance.setDecimals(2)
        self.sp_bpm_tolerance.setValue(2.0)
        self.cb_bpm_first_segment = QtWidgets.QCheckBox("Fix Start Segment to Initial Segment")
        self.cb_bpm_first_segment.setChecked(False)

        self.lbl_transition_key = QtWidgets.QLabel("Key Transition")
        self.lbl_transition_key.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.lbl_transition_key.setMinimumWidth(110)
        self.cb_transition_key = QtWidgets.QCheckBox("Harmonic Key")
        self.cb_transition_key.setChecked(False)
        self.lbl_from_key = QtWidgets.QLabel("From Key")
        self.lbl_from_key.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.lbl_from_key.setMinimumWidth(72)
        self.cmb_from_key = QtWidgets.QComboBox()
        self.lbl_arrow = QtWidgets.QLabel("→")
        self.lbl_arrow.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_arrow.setMinimumWidth(24)
        self.cmb_to_key = QtWidgets.QComboBox()
        for key_idx in range(24):
            camelot, classical = idx_to_labels(key_idx)
            label = f"{camelot} / {classical}"
            self.cmb_from_key.addItem(label, key_idx)
            self.cmb_to_key.addItem(label, key_idx)
        self.lbl_min_sec = QtWidgets.QLabel("Min sec")
        self.lbl_min_sec.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.lbl_min_sec.setMinimumWidth(92)
        self.sp_transition_min_duration = QtWidgets.QDoubleSpinBox()
        self.sp_transition_min_duration.setRange(0.0, 600.0)
        self.sp_transition_min_duration.setDecimals(2)
        self.sp_transition_min_duration.setValue(4.0)
        self.cb_key_first_segment = QtWidgets.QCheckBox("Fix Start Segment to Initial Segment")
        self.cb_key_first_segment.setChecked(False)

        transition_lay.addWidget(self.lbl_transition_bpm, 0, 0)
        transition_lay.addWidget(self.cb_transition_bpm, 0, 1)
        transition_lay.addWidget(self.lbl_from_bpm, 0, 2)
        transition_lay.addWidget(self.sp_from_bpm, 0, 3)
        transition_lay.addWidget(self.lbl_bpm_arrow, 0, 4)
        transition_lay.addWidget(self.sp_to_bpm, 0, 5)
        transition_lay.addWidget(self.lbl_bpm_tol, 0, 6)
        transition_lay.addWidget(self.sp_bpm_tolerance, 0, 7)
        transition_lay.addWidget(self.cb_bpm_first_segment, 0, 8)

        transition_lay.addWidget(self.lbl_transition_key, 1, 0)
        transition_lay.addWidget(self.cb_transition_key, 1, 1)
        transition_lay.addWidget(self.lbl_from_key, 1, 2)
        transition_lay.addWidget(self.cmb_from_key, 1, 3)
        transition_lay.addWidget(self.lbl_arrow, 1, 4)
        transition_lay.addWidget(self.cmb_to_key, 1, 5)
        transition_lay.addWidget(self.lbl_min_sec, 1, 6)
        transition_lay.addWidget(self.sp_transition_min_duration, 1, 7)
        transition_lay.addWidget(self.cb_key_first_segment, 1, 8)
        transition_lay.setColumnStretch(9, 1)
        self.transition_box.setVisible(True)

        # Table
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(len(self.HEADERS))
        self.table.setHorizontalHeaderLabels(self.HEADERS)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSortingEnabled(True)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setDefaultSectionSize(140)
        self.table.setItemDelegate(FastDelegate(self.table))
        self.table.doubleClicked.connect(self._on_activate)
        self.table.verticalScrollBar().valueChanged.connect(self._on_scroll)

        # Toolbar
        self.btn_remove = QtWidgets.QToolButton(text="Remove")
        self.btn_export = QtWidgets.QToolButton(text="Export CSV")

        # Layout
        top = QtWidgets.QHBoxLayout()
        top.addWidget(self.e_search, 1)
        top.addWidget(self.cb_column)
        tools = QtWidgets.QHBoxLayout()
        tools.addWidget(self.btn_remove)
        tools.addStretch(1)
        tools.addWidget(self.btn_export)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(self.transition_box)
        lay.addLayout(tools)
        lay.addWidget(self.table, 1)

        # Signals
        self.e_search.textChanged.connect(self._on_search_changed)
        self.cb_column.currentIndexChanged.connect(self._on_search_changed)
        self.cb_transition_bpm.toggled.connect(self._sync_transition_type_ui)
        self.cb_transition_bpm.toggled.connect(self._on_search_changed)
        self.cb_transition_key.toggled.connect(self._sync_transition_type_ui)
        self.cb_transition_key.toggled.connect(self._on_search_changed)
        self.sp_from_bpm.valueChanged.connect(self._on_search_changed)
        self.sp_to_bpm.valueChanged.connect(self._on_search_changed)
        self.sp_bpm_tolerance.valueChanged.connect(self._on_search_changed)
        self.cb_bpm_first_segment.toggled.connect(self._on_search_changed)
        self.cmb_from_key.currentIndexChanged.connect(self._on_search_changed)
        self.cmb_to_key.currentIndexChanged.connect(self._on_search_changed)
        self.sp_transition_min_duration.valueChanged.connect(self._on_search_changed)
        self.cb_key_first_segment.toggled.connect(self._on_search_changed)
        self.btn_remove.clicked.connect(self._on_remove_selected)
        self.btn_export.clicked.connect(self._on_export_csv)
        self.table.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._on_context_menu)

        # Initial load
        self.bus.sig_lib_updated.connect(self.reload)
        self.bus.sig_external_sync_enabled.connect(self._on_external_sync_enabled)
        self.reload_from_db()
        self._sync_transition_type_ui()

    # Data loading
    def reload_from_db(self):
        l = LibraryDB(os.path.join(self.cfg.libconfig.libpath, "library.db"))
        l.connect()
        lib_full = l.list_all()
        l.close()
        self.reload(lib_full)

    def reload(self, data):
        """Hard reload with full reset."""
        # Lock UI
        self.table.setSortingEnabled(False)
        self.table.setUpdatesEnabled(False)

        # Full reset
        self.table.clearContents()
        self.table.setRowCount(0)
        # Prepare new data
        tracks = [row_to_track(r) if isinstance(r, TrackRow) else r for r in data]
        self._all_tracks = tracks
        # Re-render
        self._populate_table(tracks)
        # Default sort (e.g., Added desc)
        col_idx = self.HEADERS.index("Added")
        self.table.sortItems(col_idx, QtCore.Qt.SortOrder.DescendingOrder)
        # Unlock UI
        self.table.setSortingEnabled(True)
        self.table.setUpdatesEnabled(True)

    def _populate_table(self, tracks: List[Track]):
        header = self.table.horizontalHeader()
        sorting_enabled = self.table.isSortingEnabled()
        sort_section = header.sortIndicatorSection() if header else -1
        sort_order = header.sortIndicatorOrder() if header else QtCore.Qt.SortOrder.AscendingOrder
        if sort_section < 0:
            sort_section = self.HEADERS.index("Added")
            sort_order = QtCore.Qt.SortOrder.DescendingOrder
        if sorting_enabled:
            self.table.setSortingEnabled(False)

        self.table.setRowCount(len(tracks))
        for r, t in enumerate(tracks):
            values = [
                t.title, t.artist, t.album, t._disp_bpm, t._disp_key_camelot,
                t._disp_duration, t._disp_rating, t._disp_added, t.path, t.comment
            ]
            for c, v in enumerate(values):
                item = QtWidgets.QTableWidgetItem(str(v))
                if c == 3:  # for BPM sorting
                    item.setData(QtCore.Qt.UserRole, float(t.bpm or -1))
                self.table.setItem(r, c, item)
        if sorting_enabled:
            self.table.setSortingEnabled(True)
            self.table.sortItems(sort_section, sort_order)

    # Filter
    def _on_search_changed(self):
        text = self.e_search.text().strip().lower()
        col = self.cb_column.currentText()
        # Field mapping for search
        field_map = {
            "Title": "title",
            "Artist": "artist",
            "Album": "album",
            "Comment": "comment",
            "Key": "_disp_key_camelot",
            "BPM": "_disp_bpm",
        }
        if text:
            results = []
            for t in self._all_tracks:
                cols = self.HEADERS if col == "All" else [col]
                for name in cols:
                    attr = field_map.get(name)
                    if not attr:
                        continue
                    val = getattr(t, attr, "")
                    if val and text in str(val).lower():
                        results.append(t)
                        break
        else:
            results = list(self._all_tracks)

        transition_tracks = self._search_transition_tracks()
        if transition_tracks is not None:
            allowed = {track.path for track in transition_tracks}
            results = [track for track in results if track.path in allowed]
        self._populate_table(results)

    def _sync_transition_type_ui(self):
        is_bpm = self.cb_transition_bpm.isChecked()
        for widget in (
            self.lbl_transition_bpm,
            self.lbl_from_bpm,
            self.sp_from_bpm,
            self.lbl_bpm_arrow,
            self.sp_to_bpm,
            self.lbl_bpm_tol,
            self.sp_bpm_tolerance,
            self.cb_bpm_first_segment,
        ):
            widget.setEnabled(is_bpm)
        is_key = self.cb_transition_key.isChecked()
        for widget in (
            self.lbl_transition_key,
            self.lbl_from_key,
            self.cmb_from_key,
            self.lbl_arrow,
            self.cmb_to_key,
            self.lbl_min_sec,
            self.sp_transition_min_duration,
            self.cb_key_first_segment,
        ):
            widget.setEnabled(is_key)

    def _search_transition_tracks(self) -> List[Track]:
        if not self.cb_transition_bpm.isChecked() and not self.cb_transition_key.isChecked():
            return None
        db = LibraryDB(os.path.join(self.cfg.libconfig.libpath, "library.db"))
        db.connect()
        try:
            min_duration = float(self.sp_transition_min_duration.value())
            matched_paths: set[str] | None = None
            if self.cb_transition_bpm.isChecked():
                center_from = float(self.sp_from_bpm.value())
                center_to = float(self.sp_to_bpm.value())
                tol_pct = max(0.0, float(self.sp_bpm_tolerance.value()))
                scale = tol_pct / 100.0
                bpm_rows = db.search_bpm_transitions(
                    center_from * (1.0 - scale),
                    center_from * (1.0 + scale),
                    center_to * (1.0 - scale),
                    center_to * (1.0 + scale),
                    min_duration_sec=min_duration,
                    require_first_segment=self.cb_bpm_first_segment.isChecked(),
                )
                matched_paths = {str(row.path) for row in bpm_rows}
            if self.cb_transition_key.isChecked():
                key_rows = db.search_harmonic_key_transitions(
                    int(self.cmb_from_key.currentData()),
                    int(self.cmb_to_key.currentData()),
                    min_duration_sec=min_duration,
                    require_first_segment=self.cb_key_first_segment.isChecked(),
                )
                key_paths = {str(row.path) for row in key_rows}
                matched_paths = key_paths if matched_paths is None else (matched_paths & key_paths)
        finally:
            db.close()
        if matched_paths is None:
            return None
        if not matched_paths:
            return []
        return [track for track in self._all_tracks if track.path in matched_paths]

    def _on_activate(self, idx: QtCore.QModelIndex):
        if self._external_sync_enabled:
            QtWidgets.QMessageBox.information(
                self,
                "External Sync",
                "Track loading is disabled while External Sync is enabled.",
            )
            return
        r = idx.row()
        if not (0 <= r < self.table.rowCount()):
            return
        path_col = self.HEADERS.index("Path")
        item = self.table.item(r, path_col)
        if not item:
            return
        path = item.text()
        if path:
            self.bus.sig_request_load_track.emit(path)

    def _on_external_sync_enabled(self, enabled: bool) -> None:
        self._external_sync_enabled = bool(enabled)

    def _on_context_menu(self, pos):
        menu = QtWidgets.QMenu(self)
        act_reveal = menu.addAction("Reveal in Explorer/Finder")
        act_remove = menu.addAction("Remove from Library")
        act_edit = menu.addAction("Edit Track")
        act_reanalyze = menu.addAction("Reanalyze Track")
        act_export = menu.addAction("Export Track")
        act = menu.exec_(self.table.viewport().mapToGlobal(pos))
        if act == act_remove:
            self._on_remove_selected()
        elif act == act_reveal:
            self._reveal_selected()
        elif act == act_reanalyze:
            self._reanalyze_selected()
        elif act == act_edit:
            self._edit_selected()
        elif act == act_export:
            self._export_selected()

    def _on_remove_selected(self):
        sel = self.table.selectionModel().selectedRows()
        if not sel:
            return
        rows = [ix.row() for ix in sel]
        paths = [self.table.item(r, self.HEADERS.index("Path")).text() for r in rows]
        l = LibraryDB(os.path.join(self.cfg.libconfig.libpath, "library.db"))
        l.connect()
        uids = []
        for p in paths:
            rec = l.get(p)
            if rec:
                uids.append(rec.uid)
        l.delete_paths(paths)
        l.close()
        n = FeatureNPZStore(base_dir=self.cfg.libconfig.libpath, compressed=True)
        for uid in uids:
            n.delete(uid)
        self.reload_from_db()

    # CSV
    def _on_export_csv(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export CSV", "", "CSV Files (*.csv)")
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(self.HEADERS)
            for r in range(self.table.rowCount()):
                row = [self.table.item(r, c).text() if self.table.item(r, c) else "" for c in range(len(self.HEADERS))]
                w.writerow(row)

    # Explorer
    def _reveal_selected(self):
        import subprocess, platform
        sel = self.table.selectionModel().selectedRows()
        for ix in sel:
            path = self.table.item(ix.row(), self.HEADERS.index("Path")).text()
            if not path:
                continue
            if platform.system() == "Windows":
                subprocess.Popen(["explorer", "/select,", os.path.normpath(path)])
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", "-R", path])
            else:
                folder = os.path.dirname(path)
                subprocess.Popen(["xdg-open", folder])

    def _export_selected(self):
        sel = self.table.selectionModel().selectedRows()
        if not sel:
            return
        row_idx = sel[0].row()
        path_item = self.table.item(row_idx, self.HEADERS.index("Path"))
        path = path_item.text() if path_item else ""
        if not path:
            return

        db = LibraryDB(os.path.join(self.cfg.libconfig.libpath, "library.db"))
        db.connect()
        track_row = db.get(path)
        db.close()
        if track_row is None:
            QtWidgets.QMessageBox.warning(self, "Export Track", "Track metadata could not be found in the library.")
            return

        if self._export_dlg is None:
            self._export_dlg = ExportTrackDialog(self.cfg.libconfig.libpath, self, model=self.model)
        self._export_dlg.set_track(track_row)
        self._export_dlg.show()
        self._export_dlg.raise_()
        self._export_dlg.activateWindow()

    # Reanalyze
    def _reanalyze_selected(self):
        sel = self.table.selectionModel().selectedRows()
        for ix in sel:
            path = self.table.item(ix.row(), self.HEADERS.index("Path")).text()
            if path:
                self.bus.sig_reanalyze_requested.emit(path)

    # Edit
    def _edit_selected(self):
        sel = self.table.selectionModel().selectedRows()
        if not sel:
            return
        # First selected row
        ix = sel[0]
        path = self.table.item(ix.row(), self.HEADERS.index("Path")).text()
        if not path:
            return
        l = LibraryDB(os.path.join(self.cfg.libconfig.libpath, "library.db"))
        l.connect()
        row = l.get(path)
        l.close()
        if not row:
            return
        if self._edit_dlg is None:
            self._edit_dlg = EditSongDialog(bus=self.bus, libpath=self.cfg.libconfig.libpath)
        self._edit_dlg.set_track(row)
        self._edit_dlg.show()
    def _on_scroll(self):
        start = self.table.verticalScrollBar().value()
        visible_rows = int(self.table.viewport().height() / self.table.rowHeight(0))
        end = start + visible_rows + 5  # small buffer
        for r in range(start, min(end, self.table.rowCount())):
            if not self.table.item(r, 0):  # cells not yet created
                t = self._all_tracks[r]
                vals = [t.title, t.artist, t.album, t._disp_bpm, t._disp_key_camelot,
                        t._disp_duration, t._disp_rating, t._disp_added, t.path, t.comment]
                for c, v in enumerate(vals):
                    self.table.setItem(r, c, QtWidgets.QTableWidgetItem(str(v)))
