import gc
import os
import sys
from pathlib import Path
from PySide6 import QtWidgets
from system import i18n
from system import os_support
from core.settings.config import load_cfg, ui_language, unreadable_settings
from system.resource_paths import app_folder
from system.i18n import tr
from core.library.version import (
    ensure_current_version_file,
    version_file_path,
)
from migration import library_migration
from app.startup import StartupSplash, center_on_screen, one_blas_thread, run_startup, startup_tasks
from app.startup_console import ConsoleLines
from ui.theme import apply_theme
from ui.fonts import load_fonts_and_set_global
from ui.widgets import wheel

# the logs carry translated texts and track titles: a console that cannot show a character escapes it instead of
# failing (run in the worker processes too: they import this module again)
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="backslashreplace")
    except (AttributeError, ValueError):
        pass


def main():
    # config.json, the library and the caches are the app folder's, whatever folder the app was started from
    # (a shortcut's "Start in", a terminal): another folder's config.json, or a program in it named as one the
    # app runs, is never taken
    os.chdir(app_folder())
    # what the start prints (the app's and the audio engine's lines), shown on the splash too
    console = ConsoleLines()
    console.install()
    one_blas_thread()   # before numpy loads (the startup imports it): none of the imports above does
    # The audio thread waits for the GIL whenever this (GUI) thread holds it: handed over after 1 ms instead of
    # Python's 5, so a small output buffer is fed in time (measured: a 20 ms buffer ran dry under GUI work).
    sys.setswitchinterval(0.001)
    app = QtWidgets.QApplication(sys.argv)
    # the language before any UI is made (or imported: the startup imports the rest)
    i18n.install(ui_language())
    i18n.install_qt(app)
    apply_theme(app)
    load_fonts_and_set_global(app)
    wheel.install()     # the wheel scrolls the page, never an input field's value
    print(f"[Startup] OS: {os_support.os_name()} ({os_support.OS})")
    if not os_support.IS_SUPPORTED:
        QtWidgets.QMessageBox.warning(
            None,
            tr("Unsupported Operating System"),
            tr("Mixlyzer supports Windows, Linux and macOS; this system ({system}) is not one of them.\n"
               "It may not work correctly.").format(system=os_support.os_name()),
        )
    if os_support.is_elevated():
        QtWidgets.QMessageBox.critical(
            None,
            tr("Administrator Execution Blocked"),
            tr("Mixlyzer cannot run with administrator privileges.\n"
               "Close it and launch again as a normal user."),
        )
        sys.exit(1)
    splash = StartupSplash(console)
    splash.show()
    app.processEvents()
    cfg = load_cfg()
    for name, kept in unreadable_settings():
        QtWidgets.QMessageBox.warning(
            None,
            tr("Settings Reset"),
            tr("{name} could not be read, so its settings are back to their defaults.\n"
               "The unreadable file was kept as:\n{path}\n\n"
               "To get the settings back, fix that file and put it in place of {name} while Mixlyzer is "
               "closed.").format(name=name, path=kept),
        )
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
    # Everything the startup made (the library, the search readers' tables: ~470k objects) is kept for good: out
    # of the garbage collector's scans, whose full pass over them took ~120 ms with every thread stopped (the
    # audio feed too: a dropout with any buffer under that). Collected once first, so no garbage is kept.
    gc.collect()
    gc.freeze()
    center_on_screen(w, splash.screen())   # on the screen the splash is on
    w.show()
    splash.close()
    console.uninstall()
    sys.exit(app.exec())

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
