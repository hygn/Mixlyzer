import sys, os
from pathlib import Path
from PySide6 import QtWidgets, QtGui, QtCore
from core.config import load_cfg
from core.library_version import (
    CURRENT_LIBRARY_VERSION,
    ensure_current_version_file,
    read_library_version,
    version_file_path,
)
from migration.migration import get_migration_path, migrate_library
from app.window import AppWindow
from utils.fonts import load_fonts_and_set_global


class _MigrationWorker(QtCore.QObject):
    sig_progress = QtCore.Signal(int)
    sig_finished = QtCore.Signal()
    sig_failed = QtCore.Signal(str)

    def __init__(self, lib_path: Path):
        super().__init__()
        self._lib_path = Path(lib_path)

    @QtCore.Slot()
    def run(self) -> None:
        try:
            migrate_library(self._lib_path, logger=lambda _msg: None, progress_callback=self._emit_progress)
        except Exception as exc:
            self.sig_failed.emit(str(exc))
            return
        self.sig_finished.emit()

    def _emit_progress(self, value: int) -> None:
        self.sig_progress.emit(int(max(0, min(100, value))))


def _apply_dark_theme(app: QtWidgets.QApplication) -> None:
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor(45, 45, 45))
    palette.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor(30, 30, 30))
    palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(45, 45, 45))
    palette.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor(30, 30, 30))
    palette.setColor(QtGui.QPalette.ToolTipText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor(45, 45, 45))
    palette.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(90, 135, 255))
    palette.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.black)
    app.setPalette(palette)

def _is_running_as_admin() -> bool:
    if os.name != "nt":
        return False
    try:
        return bool(__import__("ctypes").windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def main():
    app = QtWidgets.QApplication(sys.argv)
    _apply_dark_theme(app)
    load_fonts_and_set_global(app)
    if _is_running_as_admin():
        QtWidgets.QMessageBox.critical(
            None,
            "Administrator Execution Blocked",
            "Mixlyzer cannot run with administrator privileges.\n"
            "Close it and launch again as a normal user.",
        )
        sys.exit(1)
    cfg = load_cfg()
    lib_path = Path(cfg.libconfig.libpath)
    version_path = version_file_path(lib_path)
    db_path = lib_path / "library.db"
    if not version_path.exists() and not db_path.exists():
        ensure_current_version_file(lib_path)
    current_lib_version = read_library_version(lib_path)
    if current_lib_version != CURRENT_LIBRARY_VERSION:
        try:
            migration_steps = get_migration_path(current_lib_version, CURRENT_LIBRARY_VERSION)
            step_text = " -> ".join([current_lib_version] + [m.TARGET_VERSION for m in migration_steps])
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                None,
                "Library Migration Error",
                "Library version is not compatible and no migration path was found.\n"
                f"Current: {current_lib_version}\n"
                f"Required: {CURRENT_LIBRARY_VERSION}\n"
                f"Error: {exc}",
            )
            sys.exit(1)
        answer = QtWidgets.QMessageBox.question(
            None,
            "Library Migration Required",
            "The library version does not match the app.\n"
            f"Current: {current_lib_version}\n"
            f"Required: {CURRENT_LIBRARY_VERSION}\n"
            f"Migration path: {step_text}\n\n"
            "Run migration now?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.Yes,
        )
        if answer != QtWidgets.QMessageBox.Yes:
            sys.exit(1)
        progress = QtWidgets.QProgressDialog("Running library migration...", None, 0, 100)
        progress.setWindowTitle("Library Migration")
        progress.setCancelButton(None)
        progress.setMinimumDuration(0)
        progress.setWindowModality(QtCore.Qt.ApplicationModal)
        progress.setValue(0)
        progress.show()
        app.processEvents()
        thread = QtCore.QThread()
        worker = _MigrationWorker(lib_path)
        worker.moveToThread(thread)
        loop = QtCore.QEventLoop()
        result = {"error": None, "finished": False}

        worker.sig_progress.connect(progress.setValue)
        worker.sig_finished.connect(lambda: result.__setitem__("finished", True))
        worker.sig_failed.connect(lambda message: result.__setitem__("error", message))
        thread.started.connect(worker.run)
        worker.sig_finished.connect(thread.quit)
        worker.sig_failed.connect(thread.quit)
        thread.finished.connect(loop.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.start()
        loop.exec()
        thread.wait()
        progress.close()
        if result["error"] is not None or not result["finished"]:
            QtWidgets.QMessageBox.critical(
                None,
                "Library Migration Failed",
                f"Migration failed.\n{result['error'] or 'Worker terminated before completion.'}",
            )
            sys.exit(1)
        QtWidgets.QMessageBox.information(
            None,
            "Library Migration Complete",
            f"Library was migrated to {CURRENT_LIBRARY_VERSION}.",
        )
    w = AppWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
