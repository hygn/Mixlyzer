"""The tie-ups' works window (the menu bar's Tie-up Works): every work the library's tie-ups name, found by any of its
names or their readings, edited as a whole (its category, titles, aliases, references: every track using it follows),
merged into another (two entries of one work), or deleted with its tie-ups. Written at once; the library and the
Rekordbox XML's comments follow."""
from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget,
    QMessageBox, QPushButton, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from core import reading, tieup
from core.event_bus import EventBus
from foundation.i18n import tr, tr_n
from core.library import jobs
from core.library.service import LibraryService
from ui.tieup_editor import FIND_PAUSE_MS, LANG, _TextTable

ALL = ""    # the category filter's "all"


class WorksWindow(QDialog):
    """The works (the module)."""

    def __init__(self, bus: Optional[EventBus], cfg: Callable[[], object], library: LibraryService, parent=None):
        """``library``: where the works are read and written (core.library.service)."""
        super().__init__(parent)
        self.setWindowTitle(tr("Tie-up Works"))
        self.setWindowFlag(Qt.Window, True)
        self._bus = bus
        self._cfg = cfg
        self._library = library
        self._works: dict[str, tieup.Work] = {}
        self._uses: dict[str, int] = {}
        self._keys: list = []
        self._current: Optional[tieup.Work] = None     # the work being edited, as stored

        # the works: found by any name, in any script
        self.ed_find = QLineEdit()
        self.ed_find.setPlaceholderText(tr("Find a work by any of its names"))
        self.ed_find.setClearButtonEnabled(True)
        self.cmb_filter = QComboBox()
        self.cmb_filter.addItem(tr("All categories"), ALL)
        for code, name in tieup.CATEGORIES:
            self.cmb_filter.addItem(tr(name), code)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels([tr("Work"), tr("Category"), tr("Tracks")])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for c in (1, 2):
            self.table.horizontalHeader().setSectionResizeMode(c, QHeaderView.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.horizontalHeader().setSortIndicator(0, Qt.AscendingOrder)
        self.table.setSortingEnabled(True)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        find_row = QHBoxLayout()
        find_row.addWidget(self.ed_find, 1)
        find_row.addWidget(self.cmb_filter)
        ll.addLayout(find_row)
        ll.addWidget(self.table, 1)
        self.lbl_count = QLabel()
        ll.addWidget(self.lbl_count)

        # the work picked: edited as a whole
        self.cmb_category = QComboBox()
        for code, name in tieup.CATEGORIES:
            self.cmb_category.addItem(tr(name), code)
        self.tbl_titles = _TextTable(tr("Language"), tr("Title"), rows_shown=4, picks={0: LANG})
        self.tbl_titles.setToolTip(tr("<p>The work's official titles, each with its language as language-REGION "
                                      "(ja-JP, ko-KR, en-US). The library and the comment show the first title in "
                                      "the interface's language, else the first of all.</p>"))
        self.tbl_aliases = _TextTable(tr("Type"), tr("Language"), tr("Alias"), rows_shown=4,
                                      picks={0: tieup.ALIAS_TYPES, 1: LANG})
        self.tbl_aliases.setToolTip(tr("<p>Other names the work is found by, never shown: a name it is commonly "
                                       "called by (a romanization), an abbreviation, or a search hint (a misspelling, "
                                       "a typing slip). The search finds the work by them, also by how they read.</p>"))
        self.tbl_urls = _TextTable(tr("Label"), tr("URL"), rows_shown=3)
        self.list_tracks = QListWidget()
        self.list_tracks.setFixedHeight(self.list_tracks.fontMetrics().lineSpacing() * 5 + 12)
        self.list_tracks.setToolTip(tr("The tracks with a tie-up to this work: a change to the work is theirs."))
        self.btn_save = QPushButton(tr("Save"))
        self.btn_revert = QPushButton(tr("Discard Changes"))
        self.btn_merge = QPushButton(tr("Merge Into…"))
        self.btn_merge.setToolTip(tr("<p>The same work entered twice: this one's names and references are added to "
                                     "the one picked, its tracks' tie-ups moved there, and this one deleted.</p>"))
        self.btn_delete = QPushButton(tr("Delete Work"))
        self.btn_delete.setToolTip(tr("The work deleted with every track's tie-up to it."))
        right = QWidget()
        form = QFormLayout(right)
        form.setContentsMargins(0, 0, 0, 0)
        form.addRow(tr("Category"), self.cmb_category)
        form.addRow(tr("Titles"), self.tbl_titles)
        form.addRow(tr("Aliases"), self.tbl_aliases)
        form.addRow(tr("References"), self.tbl_urls)
        form.addRow(tr("Tracks"), self.list_tracks)
        buttons = QHBoxLayout()
        buttons.addWidget(self.btn_save)
        buttons.addWidget(self.btn_revert)
        buttons.addStretch(1)
        buttons.addWidget(self.btn_merge)
        buttons.addWidget(self.btn_delete)
        form.addRow(buttons)

        split = QSplitter(Qt.Horizontal)
        split.addWidget(left)
        split.addWidget(right)
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 3)
        split.setChildrenCollapsible(False)
        root = QVBoxLayout(self)
        root.addWidget(split, 1)
        self.resize(1000, 640)

        # the list narrowed: the work being edited kept in the editor (picked again when still listed)
        self._find_timer = QTimer(self)         # the list narrowed once typing pauses
        self._find_timer.setSingleShot(True)
        self._find_timer.setInterval(FIND_PAUSE_MS)
        self._find_timer.timeout.connect(lambda: self._fill(self._current.id if self._current else None))
        self.ed_find.textChanged.connect(lambda _text: self._find_timer.start())
        self.cmb_filter.currentIndexChanged.connect(lambda _i: self._fill(self._current.id if self._current else None))
        self.table.itemSelectionChanged.connect(self._on_picked)
        self.btn_save.clicked.connect(self._save)
        self.btn_revert.clicked.connect(lambda: self._show(self._current))
        self.btn_merge.clicked.connect(self._merge)
        self.btn_delete.clicked.connect(self._delete)
        self._show(None)

    # the library
    def show_works(self) -> None:
        """Shown (raised when open) with the works as stored."""
        self._load()
        self.show()
        self.raise_()
        self.activateWindow()
        self.ed_find.setFocus(Qt.OtherFocusReason)

    def _load(self, select: Optional[str] = None) -> None:
        """The works read on the library thread, listed when they come."""
        self._library.read(jobs.tieup_catalog, done=lambda catalog: self._loaded(catalog, select))

    def _loaded(self, catalog: tieup.Catalog, select: Optional[str]) -> None:
        self._works, self._uses, self._keys = catalog.works, catalog.uses, catalog.keys
        self._fill(select if select is not None else (self._current.id if self._current else None))

    def _found(self) -> list[tieup.Work]:
        """The works the find field and the category filter leave."""
        text = self.ed_find.text().strip()
        if text:
            s = self._cfg().searchconfig
            query = reading.query_keys(text, s.query_languages, hangul_korean=s.query_hangul_korean,
                                       hangul_foreign=s.query_hangul_foreign, keyboard=s.query_keyboard_layout,
                                       han_run=s.han_run())
            ids = {self._keys[i][0] for i in reading.find(query, [k for _w, k in self._keys], similar=True)}
            needle = text.casefold()
            ids |= {w.id for w in self._works.values() if any(needle in n.casefold() for n in w.names())}
        else:
            ids = set(self._works)
        category = self.cmb_filter.currentData()
        return [self._works[i] for i in ids if i in self._works and (category == ALL or self._works[i].category == category)]

    def _fill(self, select: Optional[str] = None) -> None:
        names = dict(tieup.CATEGORIES)
        shown = sorted(self._found(), key=lambda w: w.title().casefold())
        self.table.blockSignals(True)
        self.table.clearSelection()     # rows refilled under a selection would keep it on another work, unsaid
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(shown))
        for r, w in enumerate(shown):
            title = QTableWidgetItem(w.title())
            title.setData(Qt.UserRole, w.id)
            title.setToolTip(" / ".join(w.names()))
            uses = QTableWidgetItem()
            uses.setData(Qt.DisplayRole, self._uses.get(w.id, 0))
            for c, item in enumerate((title, QTableWidgetItem(tr(names.get(w.category, "Other"))), uses)):
                self.table.setItem(r, c, item)
        self.table.setSortingEnabled(True)
        self.table.blockSignals(False)
        self.lbl_count.setText(tr_n("{n} work", "{n} works", len(shown)).format(n=len(shown)))
        for r in range(self.table.rowCount()):
            if self.table.item(r, 0).data(Qt.UserRole) == select:
                self.table.selectRow(r)
                return
        if select is None or select not in self._works:
            self._show(None)

    # the work picked
    def _picked_id(self) -> Optional[str]:
        rows = self.table.selectionModel().selectedRows()
        return str(self.table.item(rows[0].row(), 0).data(Qt.UserRole)) if rows else None

    def _on_picked(self) -> None:
        work_id = self._picked_id()
        if work_id is None or (self._current is not None and work_id == self._current.id):
            return      # the one in the editor (picked again after the list was refilled): its edits kept
        if self._current is not None and self._changed():
            answer = QMessageBox.question(
                self, tr("Tie-up Works"),
                tr('Save the changes to "{title}"? (No: they are dropped)').format(title=self._current.title()),
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if answer == QMessageBox.Yes and not self._save(reselect=False):
                return
        self._show(self._works.get(work_id) if work_id else None)

    def _show(self, w: Optional[tieup.Work]) -> None:
        """The work in the editor, as stored (the catalog's); its tracks read on the library thread."""
        self._current = w
        self.cmb_category.setCurrentIndex(max(0, self.cmb_category.findData(w.category if w else "other")))
        self.tbl_titles.set_values(w.titles if w else [])
        self.tbl_aliases.set_values(w.aliases if w else [])
        self.tbl_urls.set_values(w.urls if w else [])
        self.list_tracks.clear()
        for widget in (self.cmb_category, self.tbl_titles, self.tbl_aliases, self.tbl_urls, self.btn_save,
                       self.btn_revert, self.btn_merge, self.btn_delete):
            widget.setEnabled(w is not None)
        if w is not None:
            work_id = w.id
            self._library.read(jobs.work_tracks, work_id, done=lambda uids: self._show_tracks(work_id, uids))

    def _show_tracks(self, work_id: str, uids: list[str]) -> None:
        if self._current is None or self._current.id != work_id:
            return
        snap = self._library.snapshot
        self.list_tracks.clear()
        self.list_tracks.addItems(sorted(((snap.track(u).title if snap.track(u) else "") or u for u in uids),
                                         key=str.casefold))

    def _edited(self) -> tieup.Work:
        return tieup.Work(self._current.id, str(self.cmb_category.currentData()), self.tbl_titles.values(),
                          self.tbl_aliases.values(), self.tbl_urls.values())

    def _changed(self) -> bool:
        if self._current is None:
            return False
        e, w = self._edited(), self._current
        return (e.category, e.titles, e.aliases, e.urls) != (w.category, list(w.titles), list(w.aliases), list(w.urls))

    def _written(self, tracks: list[str], select: Optional[str]) -> None:
        """After a write (the library's owner showed it): the works read again, the Rekordbox XML's comments of the
        tracks."""
        self._load(select)
        if self._bus is None:
            return
        for uid in tracks:
            self._bus.sig_rekordbox_sync_track_requested.emit(str(uid))

    def _save(self, reselect: bool = True) -> bool:
        """The work as edited, written on the library thread (checked here first: a title, ll-RR languages)."""
        if self._current is None:
            return True
        w = self._edited()
        if not w.titles:
            QMessageBox.warning(self, tr("Tie-up Works"), tr("Give the work a title."))
            return False
        bad = [lang for lang, _t in w.titles if not tieup.is_lang(lang)] + \
              [lang for _k, lang, _a in w.aliases if not tieup.is_lang(lang)]
        if bad:
            QMessageBox.warning(self, tr("Tie-up Works"),
                                tr("A language is written as ko-KR (language-REGION): {code}").format(code=bad[0]))
            return False
        before, self._current = self._current, w      # meanwhile stored as edited (nothing counts as changed)
        select = w.id if reselect else self._picked_id()

        def refused(exc: Exception) -> None:
            if self._current is w:
                self._current = before
            QMessageBox.warning(self, tr("Tie-up Works"), str(exc))

        self._library.write(jobs.save_work, w, done=lambda tracks: self._written(tracks, select), failed=refused)
        return True

    def _merge(self) -> None:
        if self._current is None:
            return
        others = sorted((w for w in self._works.values() if w.id != self._current.id), key=lambda w: w.title().casefold())
        if not others:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle(tr("Merge Into…"))
        pick = QComboBox()
        pick.setEditable(True)                  # typed to narrow: Qt's completer on the titles
        pick.setInsertPolicy(QComboBox.NoInsert)
        for w in others:
            pick.addItem(w.title(), w.id)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel(tr('Merge "{title}" into:').format(title=self._current.title())))
        layout.addWidget(pick)
        layout.addWidget(buttons)
        if dialog.exec() != QDialog.Accepted or pick.currentIndex() < 0 or pick.currentText() != pick.itemText(pick.currentIndex()):
            return
        keep = str(pick.currentData())
        n = self._uses.get(self._current.id, 0)
        answer = QMessageBox.question(
            self, tr("Merge Into…"),
            tr_n('Merge "{title}" into "{into}"? Its {n} track\'s tie-up moves there and "{title}" is deleted.',
                 'Merge "{title}" into "{into}"? Its {n} tracks\' tie-ups move there and "{title}" is deleted.',
                 n).format(title=self._current.title(), into=self._works[keep].title(), n=n),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer != QMessageBox.Yes:
            return
        drop = self._current.id
        self._current = None
        self._library.write(jobs.merge_works, keep, drop, done=lambda moved: self._written(moved, keep))

    def _delete(self) -> None:
        if self._current is None:
            return
        n = self._uses.get(self._current.id, 0)
        answer = QMessageBox.question(
            self, tr("Delete Work"),
            tr_n('Delete "{title}"? Its tie-up on {n} track is deleted too.',
                 'Delete "{title}"? Its tie-ups on {n} tracks are deleted too.', n).format(
                title=self._current.title(), n=n),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer != QMessageBox.Yes:
            return
        work_id = self._current.id
        self._current = None
        self._library.write(jobs.delete_work, work_id, done=lambda tracks: self._written(tracks, None))

    def reject(self) -> None:
        self.close()        # Esc: asked about the changes as a close is

    def closeEvent(self, event) -> None:
        if self._changed():
            answer = QMessageBox.question(
                self, tr("Tie-up Works"),
                tr('Save the changes to "{title}"? (No: they are dropped)').format(title=self._current.title()),
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.Yes)
            if answer == QMessageBox.Cancel or (answer == QMessageBox.Yes and not self._save()):
                event.ignore()
                return
        self._show(None)
        event.accept()      # not QDialog's: it calls reject(), which closes (above)
