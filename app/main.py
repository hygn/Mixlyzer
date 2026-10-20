import sys, os
from pathlib import Path
from PySide6 import QtWidgets
from core import i18n
from core.config import load_cfg, ui_language
from core.i18n import tr
from core.library_version import (
    ensure_current_version_file,
    version_file_path,
)
from migration import library_migration
from app.startup import StartupSplash, run_startup, startup_tasks
from ui.theme import apply_theme
from utils.fonts import load_fonts_and_set_global

# the logs carry translated texts and track titles: a console that cannot show a character escapes it instead of
# failing (run in the worker processes too: they import this module again)
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="backslashreplace")
    except (AttributeError, ValueError):
        pass


def _is_running_as_admin() -> bool:
    if os.name != "nt":
        return False
    try:
        return bool(__import__("ctypes").windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def main():
    app = QtWidgets.QApplication(sys.argv)
    # the language before any UI is made (or imported: the startup imports the rest)
    i18n.install(ui_language())
    i18n.install_qt(app)
    apply_theme(app)
    load_fonts_and_set_global(app)
    if _is_running_as_admin():
        QtWidgets.QMessageBox.critical(
            None,
            tr("Administrator Execution Blocked"),
            tr("Mixlyzer cannot run with administrator privileges.\n"
               "Close it and launch again as a normal user."),
        )
        sys.exit(1)
    splash = StartupSplash()
    splash.show()
    app.processEvents()
    cfg = load_cfg()
    lib_path = Path(cfg.libconfig.libpath)
    version_path = version_file_path(lib_path)
    db_path = lib_path / "library.db"
    if not version_path.exists() and not db_path.exists():
        ensure_current_version_file(lib_path)
    migration_status = library_migration.inspect(lib_path)
    if not migration_status.is_current:
        splash.close()
        QtWidgets.QMessageBox.critical(
            None,
            tr("Incompatible Library Version"),
            tr("This library cannot be opened by the current app version.\n"
               "Library: {current}\n"
               "Required: {required}\n\n"
               "Automatic library migration is not available.").format(
                current=migration_status.current_version, required=migration_status.required_version),
        )
        sys.exit(1)
    # the slow work first (imports, JIT, FFmpeg, search readers and keys), then the window
    w = run_startup(splash, startup_tasks(cfg))[-1]
    w.show()
    splash.close()
    sys.exit(app.exec())

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
