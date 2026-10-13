import sys, os
from pathlib import Path
from PySide6 import QtWidgets
from core.config import load_cfg
from core.library_version import (
    ensure_current_version_file,
    version_file_path,
)
from migration import library_migration
from app.window import AppWindow
from ui.theme import apply_theme
from utils.fonts import load_fonts_and_set_global


def _is_running_as_admin() -> bool:
    if os.name != "nt":
        return False
    try:
        return bool(__import__("ctypes").windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def main():
    app = QtWidgets.QApplication(sys.argv)
    apply_theme(app)
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
    migration_status = library_migration.inspect(lib_path)
    if not migration_status.is_current:
        QtWidgets.QMessageBox.critical(
            None,
            "Incompatible Library Version",
            "This library cannot be opened by the current app version.\n"
            f"Library: {migration_status.current_version}\n"
            f"Required: {migration_status.required_version}\n\n"
            "Automatic library migration is not available.",
        )
        sys.exit(1)
    w = AppWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
