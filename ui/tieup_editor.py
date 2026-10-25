"""The Edit tab's tie-ups (core.tieup): the works the track is used in, edited in a dialog; written at once (not by
the tab's Apply). A tie-up added goes to the track's versions (its Variants) too when Settings > Library says so."""
from __future__ import annotations

import os
from typing import Callable, Optional
import urllib.parse

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QPushButton, QSpinBox, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from core import reading, tieup, tieup_web
from core.config import searchconfig
from core.i18n import tr, tr_n
from core.library_handler import LibraryDB
from core.variants import store as variant_store
from core.workflows.tieup_web import TieupWebLookup

MATCHES_SHOWN = 30
LANG = "lang"   # a column of ll-RR language tags (a pick of tieup.LANGUAGES, or one typed)


class _TextTable(QTableWidget):
    """Text columns, rows as many as needed: an empty row always last, a new one once its last column is typed in.
    A row counts when its last column is filled. ``picks`` make columns a combo box: column -> (code, name) choices,
    or LANG."""

    def __init__(self, *headers: str, rows_shown: int = 3, picks: Optional[dict] = None, parent=None):
        super().__init__(0, len(headers), parent)
        self._picks = picks or {}
        self.setHorizontalHeaderLabels(list(headers))
        for c in range(len(headers) - 1):
            self.horizontalHeader().setSectionResizeMode(c, QHeaderView.ResizeToContents)
        self.horizontalHeader().setSectionResizeMode(len(headers) - 1, QHeaderView.Stretch)
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QAbstractItemView.AllEditTriggers)
        self.setFixedHeight(self.horizontalHeader().sizeHint().height()
                            + self.verticalHeader().defaultSectionSize() * rows_shown + 4)
        self.itemChanged.connect(self._keep_empty_row)
        self.set_values([])

    def _combo(self, c: int, value: str) -> QComboBox:
        combo = QComboBox()
        if self._picks[c] == LANG:
            combo.setEditable(True)
            combo.addItems(list(tieup.LANGUAGES))
            combo.setEditText(value or tieup.LANGUAGES[0])
        else:
            for code, name in self._picks[c]:
                combo.addItem(tr(name), code)
            combo.setCurrentIndex(max(0, combo.findData(value)))
        return combo

    def _append(self, row: tuple) -> None:
        r = self.rowCount()
        self.insertRow(r)
        for c in range(self.columnCount()):
            text = str(row[c]) if c < len(row) else ""
            if c in self._picks:
                self.setCellWidget(r, c, self._combo(c, text))
            else:
                self.setItem(r, c, QTableWidgetItem(text))

    def set_values(self, values: list[tuple]) -> None:
        self.blockSignals(True)
        self.setRowCount(0)
        for row in list(values) + [()]:
            self._append(row)
        self.blockSignals(False)

    def _cell(self, r: int, c: int) -> str:
        if c in self._picks:
            combo = self.cellWidget(r, c)
            return combo.currentText().strip() if self._picks[c] == LANG else str(combo.currentData())
        return self.item(r, c).text().strip() if self.item(r, c) else ""

    def values(self) -> list[tuple]:
        """The rows whose last column is filled."""
        rows = (tuple(self._cell(r, c) for c in range(self.columnCount())) for r in range(self.rowCount()))
        return [row for row in rows if row[-1]]

    def _keep_empty_row(self, _item) -> None:
        if self.rowCount() == 0 or self._cell(self.rowCount() - 1, self.columnCount() - 1):
            self.blockSignals(True)
            self._append(())
            self.blockSignals(False)


class TieupDialog(QDialog):
    """One tie-up: its work (found by a name in any script, or a new one) and its role in it."""

    def __init__(self, con, t: tieup.TieUp, settings: searchconfig, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("Tie-up"))
        self._tieup = t
        self._settings = settings
        self._works = {w.id: w for w in tieup.works(con)}
        self._keys = tieup.work_keys(con)
        self._con = con

        # the work: typed to find one, or a new one
        self.ed_find = QLineEdit()
        self.ed_find.setPlaceholderText(tr("Type a title in any language or script"))
        self.ed_find.setToolTip(tr("<p>Finds the works already entered by any of their titles, also by how they "
                                   "read (あのね by anone or 아노네). Pick one to use it, or New work.</p>"))
        self.list_works = QListWidget()
        self.list_works.setFixedHeight(self.list_works.fontMetrics().lineSpacing() * 6 + 12)
        self.cmb_category = QComboBox()
        for code, name in tieup.CATEGORIES:
            self.cmb_category.addItem(tr(name), code)
        self.tbl_titles = _TextTable(tr("Language"), tr("Title"), picks={0: LANG})
        self.tbl_titles.setToolTip(tr("<p>The work's official titles, each with its language as language-REGION "
                                      "(ja-JP, ko-KR, en-US). The library and the comment show the first title in "
                                      "the interface's language, else the first of all.</p>"))
        self.tbl_aliases = _TextTable(tr("Type"), tr("Language"), tr("Alias"), rows_shown=2,
                                      picks={0: tieup.ALIAS_TYPES, 1: LANG})
        self.tbl_aliases.setToolTip(tr("<p>Other names the work is found by, never shown: a name it is commonly "
                                       "called by (ぼざろ, a romanization), an abbreviation, or a search hint (a "
                                       "misspelling, a typing slip). The search finds the work by them, also by how "
                                       "they read.</p>"))
        self.tbl_urls = _TextTable(tr("Label"), tr("URL"), rows_shown=2)

        self.cmb_role = QComboBox()
        for code, name, _short in tieup.ROLES:
            self.cmb_role.addItem(tr(name), code)
        self.sp_seq = QSpinBox()
        self.sp_seq.setRange(0, 99)
        self.sp_seq.setSpecialValueText("—")
        self.sp_seq.setToolTip(tr("Which one, when the work has several (OP2: 2)."))
        self.ed_detail = QLineEdit()
        self.ed_detail.setPlaceholderText(tr("Season 2, episodes 1-12; TV size"))
        self.ed_detail.setToolTip(tr("Where and how it is used: the episodes, the version of the song (a TV size, "
                                     "a character's version)."))

        work_box = QGroupBox(tr("Work"))
        wf = QFormLayout(work_box)
        wf.addRow(tr("Find"), self.ed_find)
        wf.addRow("", self.list_works)
        wf.addRow(tr("Category"), self.cmb_category)
        wf.addRow(tr("Titles"), self.tbl_titles)
        wf.addRow(tr("Aliases"), self.tbl_aliases)
        wf.addRow(tr("References"), self.tbl_urls)
        self.lbl_shared = QLabel()
        self.lbl_shared.setWordWrap(True)
        wf.addRow(self.lbl_shared)

        use_box = QGroupBox(tr("Use"))
        uf = QFormLayout(use_box)
        role_row = QWidget()
        rl = QHBoxLayout(role_row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(self.cmb_role, 1)
        rl.addWidget(QLabel(tr("No.")))
        rl.addWidget(self.sp_seq)
        uf.addRow(tr("Role"), role_row)
        uf.addRow(tr("Detail"), self.ed_detail)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        root = QVBoxLayout(self)
        root.addWidget(work_box)
        root.addWidget(use_box)
        root.addWidget(buttons)
        self.resize(600, 680)

        self.ed_find.textChanged.connect(self._show_matches)
        self.list_works.currentItemChanged.connect(self._work_picked)
        self._loading = False
        self._set_tieup(t)

    # the work
    def _set_tieup(self, t: tieup.TieUp) -> None:
        self.cmb_role.setCurrentIndex(max(0, self.cmb_role.findData(t.role)))
        self.sp_seq.setValue(int(t.seq))
        self.ed_detail.setText(t.detail)
        self._loading = True
        self.ed_find.setText(t.work.title())
        self._loading = False
        self._show_matches(select=t.work.id or None)

    def _show_matches(self, _text: str = "", select: Optional[str] = None) -> None:
        if self._loading:
            return
        text = self.ed_find.text().strip()
        if text:
            s = self._settings
            query = reading.query_keys(text, s.query_languages, hangul_korean=s.query_hangul_korean,
                                       hangul_foreign=s.query_hangul_foreign, keyboard=s.query_keyboard_layout,
                                       han_run=s.han_run())
            found = {self._keys[i][0] for i in reading.find(query, [k for _w, k in self._keys], similar=True)}
            needle = text.casefold()
            found |= {w.id for w in self._works.values() if any(needle in n.casefold() for n in w.names())}
        else:
            found = set(self._works)
        shown = sorted((self._works[w] for w in found if w in self._works), key=lambda w: w.title().casefold())
        self.list_works.blockSignals(True)
        self.list_works.clear()
        new = QListWidgetItem(tr("New work: “{title}”").format(title=text) if text else tr("New work"))
        new.setData(Qt.UserRole, "")
        self.list_works.addItem(new)
        for w in shown[:MATCHES_SHOWN]:
            others = " / ".join(n for n in w.names() if n != w.title())[:80]
            item = QListWidgetItem(f"{w.title()}  ({tr(dict(tieup.CATEGORIES).get(w.category, 'Other'))})"
                                   + (f" — {others}" if others else ""))
            item.setData(Qt.UserRole, w.id)
            self.list_works.addItem(item)
        self.list_works.blockSignals(False)
        target = select if select is not None else (shown[0].id if shown and self._exact(text, shown[0]) else "")
        for r in range(self.list_works.count()):
            if self.list_works.item(r).data(Qt.UserRole) == target:
                self.list_works.setCurrentRow(r)
                break

    @staticmethod
    def _exact(text: str, w: tieup.Work) -> bool:
        return any(n.casefold() == text.casefold() for n in w.names())

    def _work_picked(self, current, _previous=None) -> None:
        if current is None:
            return
        work_id = str(current.data(Qt.UserRole) or "")
        if work_id:
            w = tieup.work(self._con, work_id) or self._works[work_id]
            self.cmb_category.setCurrentIndex(max(0, self.cmb_category.findData(w.category)))
            self.tbl_titles.set_values(w.titles)
            self.tbl_aliases.set_values(w.aliases)
            self.tbl_urls.set_values(w.urls)
            uses = self._con.execute("SELECT COUNT(DISTINCT track_uid) FROM tieup WHERE work_id=?;",
                                     (work_id,)).fetchone()[0]
            self.lbl_shared.setText(tr_n("Changes to the work apply to the {n} track using it.",
                                         "Changes to the work apply to the {n} tracks using it.",
                                         int(uses)).format(n=int(uses)))
        else:
            text = self.ed_find.text().strip()
            self.tbl_titles.set_values([(tieup.guess_lang(text), text)] if text else [])
            self.tbl_aliases.set_values([])
            self.tbl_urls.set_values([])
            self.lbl_shared.setText("")

    def _accept(self) -> None:
        titles = self.tbl_titles.values()
        if not titles:
            QMessageBox.warning(self, tr("Tie-up"), tr("Give the work a title."))
            return
        aliases = self.tbl_aliases.values()
        bad = [lang for lang, _t in titles if not tieup.is_lang(lang)] + \
              [lang for _k, lang, _a in aliases if not tieup.is_lang(lang)]
        if bad:
            QMessageBox.warning(self, tr("Tie-up"),
                                tr("A language is written as ko-KR (language-REGION): {code}").format(code=bad[0]))
            return
        current = self.list_works.currentItem()
        work_id = str(current.data(Qt.UserRole) or "") if current is not None else ""
        t = self._tieup
        t.work = tieup.Work(work_id, str(self.cmb_category.currentData()), titles, aliases, self.tbl_urls.values())
        t.role = str(self.cmb_role.currentData())
        t.seq = int(self.sp_seq.value())
        t.detail = self.ed_detail.text().strip()
        self.accept()

    def tieup(self) -> tieup.TieUp:
        return self._tieup


class WebCandidatesDialog(QDialog):
    """The tie-ups Wikimedia gave for the track (core.tieup_web), each to be added or not (a sure one checked)."""

    def __init__(self, candidates: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("Tie-ups on the Web"))
        self._candidates = candidates
        roles = {code: name for code, name, _short in tieup.ROLES}
        self.list = QListWidget()
        for c in candidates:
            role = tr(roles.get(c.role, "Other")) + (f" {c.seq}" if c.seq else "")
            text = f"{c.work.title()} — {role}" + (f" ({c.detail})" if c.detail else "")
            item = QListWidgetItem(text)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if c.sure else Qt.Unchecked)
            item.setToolTip(tr("Read on {page}").format(page=urllib.parse.unquote(c.source)))
            self.list.addItem(item)
        note = QLabel(tr("From Japanese Wikipedia and Wikidata. A checked one is added with the work's names in "
                         "other languages and its page as a reference."))
        note.setWordWrap(True)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(note)
        layout.addWidget(self.list)
        layout.addWidget(buttons)
        self.resize(560, 360)

    def chosen(self) -> list:
        return [c for i, c in enumerate(self._candidates) if self.list.item(i).checkState() == Qt.Checked]


class TieupBox(QGroupBox):
    """The track's tie-ups: listed, added, edited, removed (written at once), found on the web."""

    changed = Signal(list)   # the tracks whose tie-ups changed

    def __init__(self, libpath: Callable[[], Optional[str]], settings: Callable[[], searchconfig],
                 to_versions: Callable[[], bool], parent=None):
        super().__init__(tr("Tie-ups"), parent)
        self.setToolTip(tr("<p>The works this track is used in: an anime's opening, a game's title screen, a "
                           "character song.</p>"))
        self._libpath = libpath
        self._settings = settings
        self._to_versions = to_versions
        self._uid: Optional[str] = None
        self._tieups: list[tieup.TieUp] = []
        self._versions: set[str] = set()
        self._looking_up: Optional[str] = None   # the track being looked up on the web

        self.lbl_versions = QLabel()
        self.lbl_versions.setWordWrap(True)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels([tr("Work"), tr("Role"), tr("Detail")])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setFixedHeight(self.table.horizontalHeader().sizeHint().height()
                                  + self.table.verticalHeader().defaultSectionSize() * 4 + 4)
        self.table.itemDoubleClicked.connect(lambda _item: self._edit())
        self.btn_add = QPushButton(tr("Add…"))
        self.btn_edit = QPushButton(tr("Edit…"))
        self.btn_remove = QPushButton(tr("Remove"))
        self.btn_web = QPushButton(tr("Find on the Web"))
        self.btn_web.setToolTip(tr("<p>Look the track up on Japanese Wikipedia and Wikidata (its title and artist are "
                                   "sent there) and pick the tie-ups to add.</p>"))
        self.btn_add.clicked.connect(self._add)
        self.btn_edit.clicked.connect(self._edit)
        self.btn_remove.clicked.connect(self._remove)
        self.btn_web.clicked.connect(self._find_on_web)
        self._web = TieupWebLookup(libpath=lambda: self._libpath() or "", settings=self._settings, parent=self)
        self._web.found.connect(self._on_found)
        self._web.failed.connect(self._on_web_failed)
        buttons = QHBoxLayout()
        for b in (self.btn_add, self.btn_edit, self.btn_remove):
            buttons.addWidget(b)
        buttons.addStretch(1)
        buttons.addWidget(self.btn_web)
        layout = QVBoxLayout(self)
        layout.addWidget(self.table)
        layout.addLayout(buttons)
        layout.addWidget(self.lbl_versions)
        self.set_track(None)

    def _db(self) -> Optional[LibraryDB]:
        libpath = self._libpath()
        if not libpath:
            return None
        s = self._settings()
        db = LibraryDB(os.path.join(libpath, "library.db"), reading_languages=s.db_languages, han_run=s.han_run())
        db.connect()
        return db

    def set_track(self, uid: Optional[str]) -> None:
        self._uid = uid or None
        self._tieups = []
        self._versions = set()
        db = self._db() if self._uid else None
        if db is not None:
            try:
                self._tieups = tieup.track_tieups(db.conn, self._uid)
            finally:
                db.close()
            self._versions = tieup.versions_of(self._uid, variant_store.pairs(variant_store.path_for(self._libpath())))
        self.table.setRowCount(len(self._tieups))
        roles = {code: name for code, name, _short in tieup.ROLES}
        for r, t in enumerate(self._tieups):
            role = tr(roles.get(t.role, "Other")) + (f" {t.seq}" if t.seq else "")
            for c, text in enumerate((t.work.title(), role, t.detail)):
                item = QTableWidgetItem(text)
                item.setToolTip(text)
                self.table.setItem(r, c, item)
        n = len(self._versions)
        self.lbl_versions.setText(
            "" if not n else
            tr_n("A tie-up added here is added to its {n} other version too (Settings > Library).",
                 "A tie-up added here is added to its {n} other versions too (Settings > Library).", n).format(n=n)
            if self._to_versions() else
            tr_n("{n} other version of this track (Variants).", "{n} other versions of this track (Variants).",
                 n).format(n=n))
        enabled = self._uid is not None and self._libpath() is not None
        self.btn_add.setEnabled(enabled)
        self.btn_edit.setEnabled(enabled and bool(self._tieups))
        self.btn_remove.setEnabled(enabled and bool(self._tieups))
        self.btn_web.setEnabled(enabled and self._looking_up is None)

    def _selected(self) -> Optional[tieup.TieUp]:
        r = self.table.currentRow()
        return self._tieups[r] if 0 <= r < len(self._tieups) else None

    def _open(self, t: tieup.TieUp, old: Optional[tuple] = None) -> None:
        db = self._db()
        if db is None or self._uid is None:
            return
        try:
            dialog = TieupDialog(db.conn, t, self._settings(), self)
            if dialog.exec() != QDialog.Accepted:
                return
            s = self._settings()
            versions = self._versions if old is None and self._to_versions() else ()
            with db.tx():
                written = tieup.save_tieup(db.conn, self._uid, dialog.tieup(), old, versions,
                                           s.db_languages, s.han_run())
        finally:
            db.close()
        self.set_track(self._uid)
        self.changed.emit(written)

    def _add(self) -> None:
        self._open(tieup.TieUp())

    def _edit(self) -> None:
        t = self._selected()
        if t is not None:
            self._open(t, t.key())

    def _find_on_web(self) -> None:
        if self._uid is None:
            return
        self._looking_up = self._uid
        self.btn_web.setEnabled(False)
        self.btn_web.setText(tr("Looking up…"))
        self._web.find(self._uid)

    def _web_done(self) -> None:
        self._looking_up = None
        self.btn_web.setText(tr("Find on the Web"))
        self.btn_web.setEnabled(self._uid is not None and self._libpath() is not None)

    def _on_web_failed(self, uid: str, error: str) -> None:
        self._web_done()
        if uid == self._uid:
            QMessageBox.warning(self, tr("Tie-ups on the Web"), tr("The lookup failed:\n{error}").format(error=error))

    def _on_found(self, uid: str, candidates: list) -> None:
        self._web_done()
        if uid != self._uid:
            return
        if not candidates:
            QMessageBox.information(self, tr("Tie-ups on the Web"), tr("Wikipedia has no tie-up for this track."))
            return
        dialog = WebCandidatesDialog(candidates, self)
        if dialog.exec() != QDialog.Accepted or not dialog.chosen():
            return
        db = self._db()
        if db is None:
            return
        s = self._settings()
        tracks = [uid] + (sorted(self._versions) if self._to_versions() else [])
        try:
            with db.tx():
                for c in dialog.chosen():
                    for track in tracks:
                        tieup_web.add(db.conn, track, c, s.db_languages, s.han_run())
        finally:
            db.close()
        self.set_track(uid)
        self.changed.emit(tracks)

    def _remove(self) -> None:
        t = self._selected()
        db = self._db()
        if t is None or db is None:
            return
        try:
            with db.tx():
                tieup.delete_tieup(db.conn, self._uid, t.key())
        finally:
            db.close()
        self.set_track(self._uid)
        self.changed.emit([self._uid])
