"""Settings > Library > Backups: the backups Mixlyzer makes before it writes a user's files (the Sync tab's: a file's
tags, core.tags.backup; rekordbox's analysis, third_party.rekordbox_phrases), each listed with when it was made and
its size, the selected ones deleted after a confirmation, their folder opened. A deletion stays inside the backup
folders and never follows a link (a symbolic link or a junction there is left alone, with what it points to)."""
from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from core.tags import backup as tag_backup
from system.i18n import tr, tr_n
from system.os_support import path_key
from third_party import rekordbox_phrases


def _roots() -> list[tuple[str, Path]]:
    return [(tr("File tags"), tag_backup.folder()), ("rekordbox", rekordbox_phrases.backup_folder())]


def _link(path: Path) -> bool:
    return path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for folder, dirs, files in os.walk(path, followlinks=False):
        total += sum((Path(folder) / name).stat().st_size for name in files if not _link(Path(folder) / name))
        dirs[:] = [d for d in dirs if not _link(Path(folder) / d)]
    return total


def file_backups() -> list[tuple[str, Path, float, int]]:
    """(kind, path, when made (s since the epoch), bytes) of every backup, newest first: the entries of each kind's
    day folders (links left out)."""
    out = []
    for kind, root in _roots():
        if not root.is_dir():
            continue
        for day in root.iterdir():
            if not day.is_dir() or _link(day):
                continue
            for entry in day.iterdir():
                if not _link(entry):
                    out.append((kind, entry, entry.stat().st_mtime, _size(entry)))
    return sorted(out, key=lambda row: row[2], reverse=True)


def delete_backup(path: Path) -> None:
    """``path`` (one of file_backups') deleted, and its day folder when that is left empty; OSError for a path that
    is no backup's (outside the backup folders, a link, a folder holding one)."""
    roots = [root for _kind, root in _roots()]
    parent = path_key(path.parent.parent)
    if not any(parent == path_key(root) for root in roots) or _link(path) or _link(path.parent):
        raise OSError(tr("Not a backup of Mixlyzer's: {path}").format(path=path))
    if path.is_dir():
        for folder, dirs, files in os.walk(path, followlinks=False):
            if any(_link(Path(folder) / name) for name in dirs + files):
                raise OSError(tr("Not a backup of Mixlyzer's: {path}").format(path=path))
        shutil.rmtree(path)
    else:
        path.unlink()
    try:
        path.parent.rmdir()             # only when empty
    except OSError:
        pass


def _megabytes(size: int) -> str:
    return f"{size / 1e6:.1f} MB" if size >= 100_000 else f"{size / 1e3:.0f} KB"


def build_backups_group(dialog) -> QtWidgets.QGroupBox:
    """The Backups page's group (``dialog``: the settings dialog, given ``tree_file_backups`` and
    ``refresh_file_backups``)."""
    group = QtWidgets.QGroupBox(tr("Backups Before Writing"))
    group.setToolTip(tr(
        "<p>The copies made before the Sync tab writes a file's tags or rekordbox's analysis (as its confirmation "
        "asks). Restore from Backup there puts one back; delete those you no longer need here.</p>"))
    column = QtWidgets.QVBoxLayout(group)
    tree = QtWidgets.QTreeWidget()
    tree.setRootIsDecorated(False)
    tree.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
    tree.setHeaderLabels([tr("Kind"), tr("Backup"), tr("Made"), tr("Size")])
    tree.setMinimumHeight(260)
    total = QtWidgets.QLabel("")
    delete = QtWidgets.QPushButton(tr("Delete Selected..."))
    open_folder = QtWidgets.QPushButton(tr("Open Folder"))
    buttons = QtWidgets.QHBoxLayout()
    buttons.addWidget(delete)
    buttons.addWidget(open_folder)
    buttons.addStretch(1)
    column.addWidget(tree)
    column.addWidget(total)
    column.addLayout(buttons)

    def refresh() -> None:
        tree.clear()
        rows = file_backups()
        for kind, path, made, size in rows:
            item = QtWidgets.QTreeWidgetItem([kind, path.name, datetime.fromtimestamp(made).strftime("%Y-%m-%d %H:%M"),
                                              _megabytes(size)])
            item.setData(0, QtCore.Qt.UserRole, str(path))
            item.setToolTip(1, str(path))
            item.setTextAlignment(3, QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            tree.addTopLevelItem(item)
        for i in range(4):
            tree.resizeColumnToContents(i)
        total.setText(tr_n("{count} backup, {size} in all", "{count} backups, {size} in all", len(rows)).format(
            count=len(rows), size=_megabytes(sum(row[3] for row in rows))))
        delete.setEnabled(bool(rows))

    def remove() -> None:
        paths = [Path(item.data(0, QtCore.Qt.UserRole)) for item in tree.selectedItems()]
        if not paths:
            return
        answer = QtWidgets.QMessageBox.question(
            dialog, tr("Delete Backups"),
            tr_n("Delete {count} backup? It cannot be put back after this.",
                 "Delete {count} backups? They cannot be put back after this.", len(paths)).format(count=len(paths)),
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
        if answer != QtWidgets.QMessageBox.Yes:
            return
        failed = []
        for path in paths:
            try:
                delete_backup(path)
            except OSError as exc:
                failed.append(str(exc))
        refresh()
        if failed:
            QtWidgets.QMessageBox.warning(dialog, tr("Delete Backups"), "\n".join(failed))

    def reveal() -> None:
        folder = tag_backup.folder().parent           # the app folder's backups: the tags' and rekordbox's
        folder.mkdir(parents=True, exist_ok=True)
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(folder)))

    delete.clicked.connect(remove)
    open_folder.clicked.connect(reveal)
    dialog.tree_file_backups = tree
    dialog.refresh_file_backups = refresh
    return group
