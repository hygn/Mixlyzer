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

    def __init__(self, bus: EventBus, cfg: config, model: DataModel | None = None, parent=None):
        super().__init__(parent)
        self.bus = bus
        self.cfg = cfg
        self.model = model
        self._all_tracks: List[Track] = []
        self._edit_dlg: EditSongDialog | None = None
        self._export_dlg: ExportTrackDialog | None = None

        # Search UI
        self.e_search = QtWidgets.QLineEdit(placeholderText="Search (Title / Artist / Key / ...)")
        self.cb_column = QtWidgets.QComboBox()
        self.cb_column.addItems(["All"] + self.HEADERS)

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
        lay.addLayout(tools)
        lay.addWidget(self.table, 1)

        # Signals
        self.e_search.textChanged.connect(self._on_search_changed)
        self.cb_column.currentIndexChanged.connect(self._on_search_changed)
        self.btn_remove.clicked.connect(self._on_remove_selected)
        self.btn_export.clicked.connect(self._on_export_csv)
        self.table.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._on_context_menu)

        # Initial load
        self.bus.sig_lib_updated.connect(self.reload)
        self.reload_from_db()

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
        if not text:
            self._populate_table(self._all_tracks)
            return
    
        # Field mapping for search
        field_map = {
            "Title": "title",
            "Artist": "artist",
            "Album": "album",
            "Comment": "comment",
            "Key": "_disp_key_camelot",
            "BPM": "_disp_bpm",
        }
    
        results = []
        for t in self._all_tracks:
            # Decide which columns to search
            cols = self.HEADERS if col == "All" else [col]
            for name in cols:
                attr = field_map.get(name)
                if not attr:
                    continue  # skip unsupported fields
                val = getattr(t, attr, "")
                if val and text in str(val).lower():
                    results.append(t)
                    break
    
        self._populate_table(results)



    def _on_activate(self, idx: QtCore.QModelIndex):
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
