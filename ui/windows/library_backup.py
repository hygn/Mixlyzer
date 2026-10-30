"""Settings > Library's Backup: the backup's progress and the restore's window (core.workflows.library_backup): the
backup picked, the new library folder, the backup's music folders joined to this device's."""
from __future__ import annotations

import os
from datetime import datetime

from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt

from core.library.package import PackageError, PackageInfo, path_roots
from core.workflows.library_backup import LibraryBackup, default_folder
from system import os_support
from system.i18n import tr, tr_n
from ui.widgets.elided_label import ElidedLabel


def reason_text(exc: Exception) -> str:
    """A refusal or failure in words."""
    reason = getattr(exc, "reason", None)
    if reason == PackageError.NOT_A_PACKAGE:
        return tr("This file is not a Mixlyzer library backup.")
    if reason == PackageError.DAMAGED:
        return tr("The backup is damaged: its files do not match its list of contents.")
    if reason == PackageError.NEWER_FORMAT:
        return tr("The backup was made by a later version of Mixlyzer. Update Mixlyzer to restore it.")
    if reason == PackageError.LIBRARY_VERSION:
        return tr("The backup's library is of another version than this Mixlyzer's ({versions}), so it cannot be "
                  "restored.").format(versions=exc.detail)
    if reason == PackageError.UNEXPECTED:
        return tr("The backup holds files no Mixlyzer backup has, so it was not restored.")
    if reason == PackageError.SCHEMA:
        return tr("A database in the backup is damaged or holds code, so it was not restored.")
    if reason == PackageError.DESTINATION:
        return tr("The new library folder must be a new folder or an empty one.")
    if reason == PackageError.CANCELLED:
        return tr("Cancelled.")
    return str(exc)


def size_text(size: int) -> str:
    for unit, scale in (("GB", 1 << 30), ("MB", 1 << 20), ("kB", 1 << 10)):
        if size >= scale:
            return f"{size / scale:,.1f} {unit}"
    return f"{size:,} B"


class BackupDialog(QtWidgets.QDialog):
    """A backup as it is written: its file, its progress, Cancel; then what was written."""

    def __init__(self, backup: LibraryBackup, dest: str, feature_cache: bool, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("Back Up Library"))
        self._backup = backup
        self._running = True
        self.lbl_status = QtWidgets.QLabel(tr("Packing the library into:"))
        self.lbl_status.setWordWrap(True)
        self.lbl_path = ElidedLabel(dest, mode=Qt.ElideMiddle)
        self.bar = QtWidgets.QProgressBar()
        self.bar.setRange(0, 100)
        self.btn_cancel = QtWidgets.QPushButton(tr("Cancel"))
        self.btn_cancel.clicked.connect(self._cancel)
        self.btn_close = QtWidgets.QPushButton(tr("Close"))
        self.btn_close.setEnabled(False)
        self.btn_close.clicked.connect(self.accept)
        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self.btn_cancel)
        buttons.addWidget(self.btn_close)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.lbl_status)
        layout.addWidget(self.lbl_path)
        layout.addWidget(self.bar)
        layout.addLayout(buttons)
        self.resize(560, 150)
        backup.progress.connect(self._on_progress)
        backup.backed_up.connect(self._on_done)
        backup.failed.connect(self._on_failed)
        backup.backup(dest, feature_cache)

    def closeEvent(self, event) -> None:
        if self._running:
            event.ignore()
            return
        super().closeEvent(event)

    def reject(self) -> None:
        if not self._running:   # Esc while it runs: kept open (closeEvent's guard is not asked on Esc)
            super().reject()

    def _cancel(self) -> None:
        self.btn_cancel.setEnabled(False)
        self._backup.cancel()

    def _end(self) -> None:
        self._running = False
        for signal, slot in ((self._backup.progress, self._on_progress), (self._backup.backed_up, self._on_done),
                             (self._backup.failed, self._on_failed)):
            signal.disconnect(slot)
        self.btn_cancel.setEnabled(False)
        self.btn_close.setEnabled(True)

    @QtCore.Slot(float)
    def _on_progress(self, share: float) -> None:
        self.bar.setValue(int(100 * share))

    @QtCore.Slot(object)
    def _on_done(self, result) -> None:
        self._end()
        self.bar.setValue(100)
        self.lbl_status.setText(tr_n("Backed up {n} track ({size}) to:", "Backed up {n} tracks ({size}) to:",
                                     result.tracks).format(n=f"{result.tracks:,}", size=size_text(result.size)))
        self.lbl_path.setText(result.path)

    @QtCore.Slot(object)
    def _on_failed(self, exc) -> None:
        self._end()
        if getattr(exc, "reason", None) == PackageError.CANCELLED:
            self.lbl_status.setText(tr("The backup was cancelled; no file was left."))
        else:
            self.lbl_status.setText(tr("The library could not be backed up:\n{error}").format(error=reason_text(exc)))


class RestoreDialog(QtWidgets.QDialog):
    """A backup restored into a new library folder, its music folders joined to this device's. ``finished_restore``
    carries the RestoreResult (the window switches to it); the library in use is never touched."""
    finished_restore = QtCore.Signal(object)

    COL_PACKAGE, COL_TRACKS, COL_HERE, COL_FOUND = range(4)

    def __init__(self, backup: LibraryBackup, info: PackageInfo, libpath: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("Restore Library"))
        self._backup = backup
        self._info = info
        self._libpath = libpath
        self._running = False
        self._asked = 0          # the folders' check asked last (an earlier answer coming late is not shown)
        self._roots = path_roots(p for p, _size in info.paths)

        made = info.created
        try:
            made = datetime.fromisoformat(info.created).astimezone().strftime("%Y-%m-%d %H:%M")
        except ValueError:
            pass
        grp_backup = QtWidgets.QGroupBox(tr("Backup"))
        f_backup = QtWidgets.QFormLayout(grp_backup)
        f_backup.addRow(tr("File"), ElidedLabel(info.path, mode=Qt.ElideMiddle))
        f_backup.addRow(tr("Made"), QtWidgets.QLabel(made))
        f_backup.addRow(tr("On"), QtWidgets.QLabel(os_support.NAMES.get(info.os, info.os)))
        f_backup.addRow(tr("Tracks"), QtWidgets.QLabel(f"{info.tracks:,}"))
        f_backup.addRow(tr("Size"), QtWidgets.QLabel(size_text(info.size)))
        f_backup.addRow(tr("Feature caches"), QtWidgets.QLabel(
            tr("Included") if info.feature_cache else tr("Not included")))

        grp_folder = QtWidgets.QGroupBox(tr("New Library Folder"))
        v_folder = QtWidgets.QVBoxLayout(grp_folder)
        row = QtWidgets.QHBoxLayout()
        self.ed_folder = QtWidgets.QLineEdit(default_folder(libpath))
        btn_folder = QtWidgets.QPushButton(tr("Browse…"))
        btn_folder.clicked.connect(self._browse_folder)
        row.addWidget(self.ed_folder, 1)
        row.addWidget(btn_folder)
        v_folder.addLayout(row)
        note = QtWidgets.QLabel(tr(
            "The backup's library is unpacked into this new folder and Mixlyzer restarts on it. The library in use "
            "now stays as it is in its folder. Your shortcuts, MIDI mapping and retrained models are replaced by the "
            "backup's; the ones here are kept beside them. Audio and file paths stay this device's settings."))
        note.setWordWrap(True)
        v_folder.addWidget(note)

        grp_music = QtWidgets.QGroupBox(tr("Music Folders"))
        v_music = QtWidgets.QVBoxLayout(grp_music)
        note = QtWidgets.QLabel(tr(
            "Where each folder of the backup's music files is on this device. A folder left empty keeps its tracks' "
            "paths as they were."))
        note.setWordWrap(True)
        v_music.addWidget(note)
        self.table = QtWidgets.QTableWidget(len(self._roots), 4)
        self.table.setHorizontalHeaderLabels([tr("In the Backup"), tr("Tracks"), tr("On This Device"), tr("Found")])
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(self.COL_PACKAGE, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(self.COL_HERE, QtWidgets.QHeaderView.Stretch)
        for col in (self.COL_TRACKS, self.COL_FOUND):
            header.setSectionResizeMode(col, QtWidgets.QHeaderView.ResizeToContents)
        for r, root in enumerate(self._roots):
            here = root.path if os.path.isdir(root.path) else ""
            for col, text in ((self.COL_PACKAGE, root.path), (self.COL_TRACKS, f"{root.tracks:,}"),
                              (self.COL_HERE, here), (self.COL_FOUND, "")):
                item = QtWidgets.QTableWidgetItem(text)
                item.setToolTip(text)
                if col == self.COL_TRACKS:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(r, col, item)
        self.table.doubleClicked.connect(lambda _index: self._choose())
        v_music.addWidget(self.table, 1)
        row = QtWidgets.QHBoxLayout()
        self.btn_choose = QtWidgets.QPushButton(tr("Choose Folder…"))
        self.btn_choose.clicked.connect(self._choose)
        self.btn_clear = QtWidgets.QPushButton(tr("Clear"))
        self.btn_clear.clicked.connect(self._clear)
        row.addWidget(self.btn_choose)
        row.addWidget(self.btn_clear)
        row.addStretch(1)
        v_music.addLayout(row)

        self.lbl_status = QtWidgets.QLabel("")
        self.lbl_status.setWordWrap(True)
        self.bar = QtWidgets.QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setVisible(False)
        self.btn_restore = QtWidgets.QPushButton(tr("Restore"))
        self.btn_restore.setDefault(True)
        self.btn_restore.clicked.connect(self._restore)
        self.btn_cancel = QtWidgets.QPushButton(tr("Cancel"))
        self.btn_cancel.clicked.connect(self._cancel)
        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self.btn_restore)
        buttons.addWidget(self.btn_cancel)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(grp_backup)
        layout.addWidget(grp_folder)
        layout.addWidget(grp_music, 1)
        layout.addWidget(self.lbl_status)
        layout.addWidget(self.bar)
        layout.addLayout(buttons)
        self.resize(760, 640)
        self._inputs = (self.ed_folder, btn_folder, self.table, self.btn_choose, self.btn_clear, self.btn_restore)
        self._check()

    # ---- the music folders ----
    def mapping(self) -> dict:
        return {root.path: self.table.item(r, self.COL_HERE).text().strip() for r, root in enumerate(self._roots)}

    def _choose(self) -> None:
        r = self.table.currentRow()
        if r < 0 or self._running:
            return
        start = self.table.item(r, self.COL_HERE).text() or os.path.expanduser("~")
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, tr("Choose the Folder on This Device"), start)
        if folder:
            self._set_here(r, os.path.normpath(folder))

    def _clear(self) -> None:
        r = self.table.currentRow()
        if r >= 0 and not self._running:
            self._set_here(r, "")

    def _set_here(self, r: int, folder: str) -> None:
        item = self.table.item(r, self.COL_HERE)
        item.setText(folder)
        item.setToolTip(folder)
        self._check()

    def _check(self) -> None:
        self._asked += 1
        asked = self._asked
        for r in range(len(self._roots)):
            self.table.item(r, self.COL_FOUND).setText("…")
        self._backup.check(self._info, {k: v for k, v in self.mapping().items() if v},
                           lambda found: self._show_check(asked, found))

    def _show_check(self, asked: int, found: dict) -> None:
        if asked != self._asked or not shiboken_valid(self):
            return
        for r, root in enumerate(self._roots):
            item = self.table.item(r, self.COL_FOUND)
            counts = found.get(root.path)
            if counts is None:
                item.setText(tr("Not joined"))
                item.setToolTip(tr("Its tracks keep their paths as they were."))
                continue
            parts = [tr("{n} found").format(n=f"{counts.found:,}")]
            if counts.differs:
                parts.append(tr("{n} of another size").format(n=f"{counts.differs:,}"))
            if counts.missing:
                parts.append(tr("{n} missing").format(n=f"{counts.missing:,}"))
            item.setText(", ".join(parts))
            item.setToolTip(item.text())

    # ---- restoring ----
    def _restore(self) -> None:
        folder = self.ed_folder.text().strip()
        if not folder:
            return
        target = os.path.abspath(folder)
        if os_support.path_key(target) == os_support.path_key(self._libpath) or (
                os.path.exists(target) and (not os.path.isdir(target) or os.listdir(target))):
            QtWidgets.QMessageBox.warning(self, tr("Restore Library"), reason_text(
                PackageError(PackageError.DESTINATION)))
            return
        self._running = True
        for widget in self._inputs:
            widget.setEnabled(False)
        self.bar.setVisible(True)
        self.bar.setValue(0)
        self.lbl_status.setText(tr("Unpacking the library…"))
        self._backup.progress.connect(self._on_progress)
        self._backup.restored.connect(self._on_restored)
        self._backup.failed.connect(self._on_failed)
        self._backup.restore(self._info, target, {k: v for k, v in self.mapping().items() if v})

    def _cancel(self) -> None:
        if self._running:
            self.btn_cancel.setEnabled(False)
            self._backup.cancel()
        else:
            self.reject()

    def closeEvent(self, event) -> None:
        if self._running:
            event.ignore()
            return
        super().closeEvent(event)

    def reject(self) -> None:
        if not self._running:   # Esc while it runs: kept open (closeEvent's guard is not asked on Esc)
            super().reject()

    def _end(self) -> None:
        self._running = False
        for signal, slot in ((self._backup.progress, self._on_progress), (self._backup.restored, self._on_restored),
                             (self._backup.failed, self._on_failed)):
            signal.disconnect(slot)

    @QtCore.Slot(float)
    def _on_progress(self, share: float) -> None:
        self.bar.setValue(int(100 * share))

    @QtCore.Slot(object)
    def _on_restored(self, result) -> None:
        self._end()
        self.bar.setValue(100)
        self.finished_restore.emit(result)
        self.accept()

    @QtCore.Slot(object)
    def _on_failed(self, exc) -> None:
        self._end()
        self.bar.setVisible(False)
        self.lbl_status.setText(tr("The library could not be restored:\n{error}").format(error=reason_text(exc)))
        for widget in self._inputs:
            widget.setEnabled(True)
        self.btn_cancel.setEnabled(True)

    def _browse_folder(self) -> None:
        start = os.path.dirname(os.path.abspath(self.ed_folder.text().strip() or self._libpath))
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, tr("New Library Folder"), start)
        if folder:
            self.ed_folder.setText(os.path.normpath(folder))


def shiboken_valid(widget) -> bool:
    import shiboken6
    return shiboken6.isValid(widget)
