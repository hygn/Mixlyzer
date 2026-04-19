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


def _preferred_style_name() -> str:
    available = {name.lower(): name for name in QtWidgets.QStyleFactory.keys()}
    if os.name == "nt":
        try:
            win_ver = sys.getwindowsversion()
            if win_ver.major >= 10 and win_ver.build >= 22000:
                style = available.get("windows11")
                if style:
                    return style
        except Exception:
            pass
    return available.get("fusion", "Fusion")


def _apply_dark_theme(app: QtWidgets.QApplication) -> None:
    app.setStyle(_preferred_style_name())

    palette = QtGui.QPalette()
    window = QtGui.QColor(45, 45, 45)
    base = QtGui.QColor(30, 30, 30)
    alt_base = QtGui.QColor(45, 45, 45)
    text = QtGui.QColor(255, 255, 255)
    disabled_text = QtGui.QColor(140, 140, 140)
    button = QtGui.QColor(45, 45, 45)
    button_text = QtGui.QColor(255, 255, 255)
    highlight = QtGui.QColor(90, 135, 255)
    highlighted_text = QtGui.QColor(0, 0, 0)
    bright_text = QtGui.QColor(255, 80, 80)
    shadow = QtGui.QColor(20, 20, 20)

    for group in (
        QtGui.QPalette.Active,
        QtGui.QPalette.Inactive,
        QtGui.QPalette.Disabled,
    ):
        is_disabled = group == QtGui.QPalette.Disabled
        fg = disabled_text if is_disabled else text
        btn_fg = disabled_text if is_disabled else button_text
        palette.setColor(group, QtGui.QPalette.Window, window)
        palette.setColor(group, QtGui.QPalette.WindowText, fg)
        palette.setColor(group, QtGui.QPalette.Base, base)
        palette.setColor(group, QtGui.QPalette.AlternateBase, alt_base)
        palette.setColor(group, QtGui.QPalette.ToolTipBase, base)
        palette.setColor(group, QtGui.QPalette.ToolTipText, text)
        palette.setColor(group, QtGui.QPalette.Text, fg)
        palette.setColor(group, QtGui.QPalette.Button, button)
        palette.setColor(group, QtGui.QPalette.ButtonText, btn_fg)
        palette.setColor(group, QtGui.QPalette.BrightText, bright_text)
        palette.setColor(group, QtGui.QPalette.Highlight, highlight)
        palette.setColor(group, QtGui.QPalette.HighlightedText, highlighted_text)
        palette.setColor(group, QtGui.QPalette.Light, window.lighter(130))
        palette.setColor(group, QtGui.QPalette.Midlight, window.lighter(115))
        palette.setColor(group, QtGui.QPalette.Dark, shadow)
        palette.setColor(group, QtGui.QPalette.Mid, QtGui.QColor(60, 60, 60))
        palette.setColor(group, QtGui.QPalette.Shadow, shadow)
        palette.setColor(group, QtGui.QPalette.Link, QtGui.QColor(110, 160, 255))
        palette.setColor(group, QtGui.QPalette.LinkVisited, QtGui.QColor(150, 120, 255))
    app.setPalette(palette)
    app.setStyleSheet(
        """
        QToolTip {
            color: #ffffff;
            background-color: #1e1e1e;
            border: 1px solid #505050;
        }
        """
    )

    style_hints = QtGui.QGuiApplication.styleHints()
    if hasattr(style_hints, "setColorScheme"):
        style_hints.setColorScheme(QtCore.Qt.ColorScheme.Dark)

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
