"""The smallest screen the window is made for, FHD at 125 % (1536 x 864 logical), on the app's own fonts and theme: the deck
and the library at their minimum and the toolbar's row at its minimum fit it, nothing in the row cut."""
import os
import tempfile
import unittest

from unit.ui_tests.qt_env import APP, drain, isolated

from PySide6 import QtCore, QtWidgets

from core.library.db import LibraryDB
from core.library.service import LibraryService
from core.settings.config import default_cfg
from core.track.model import DataModel
from core.workflows.event_bus import EventBus
from ui.deck.deck_controls import ROW_SPACING, block, fill, row_title, text_room
from ui.deck.deck_switches import DeckSwitches, MasterLevels
from ui.deck.midi_toggle import MidiToggle
from ui.fonts import load_fonts_and_set_global
from ui.pane import MainPane
from ui.theme import apply_theme
from views.timeline import TimelineCoordinator

WIDTH, HEIGHT = 1536, 864
TASKBAR_AND_TITLE, MENU_TOOLBAR_STATUS = 71, 80     # px of the height: the system's, the window's around the pane

_ISOLATION = None
_APP_LOOK = None


def setUpModule():
    global _ISOLATION, _APP_LOOK
    _ISOLATION = isolated()
    _APP_LOOK = (APP.font(), APP.styleSheet(), APP.palette(), APP.style().objectName())
    load_fonts_and_set_global(APP)
    apply_theme(APP)


def tearDownModule():
    font, sheet, palette, style = _APP_LOOK
    APP.setStyleSheet(sheet)
    APP.setStyle(style)
    APP.setPalette(palette)
    APP.setFont(font)
    _ISOLATION.close()


class SmallestScreenTests(unittest.TestCase):
    def test_the_toolbars_row_fits_with_nothing_cut(self):
        bus = EventBus()
        row = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(row)
        lay.setContentsMargins(4, 0, 4, 0)
        lay.setSpacing(ROW_SPACING)
        lay.addWidget(row_title("CH 1"), 0, QtCore.Qt.AlignVCenter)
        lay.addSpacing(2)
        fill(lay, [block("CONTROLLER", [MidiToggle()]), DeckSwitches(bus)])
        lay.addSpacing(4)
        lay.addWidget(row_title("MASTER"), 0, QtCore.Qt.AlignVCenter)
        lay.addSpacing(2)
        fill(lay, [MasterLevels(bus)])
        self.assertLessEqual(row.minimumSizeHint().width(), WIDTH)
        row.resize(row.minimumSizeHint().width(), row.sizeHint().height())
        row.show()
        drain(10)
        for button in row.findChildren(QtWidgets.QToolButton):
            if button.text():
                self.assertGreaterEqual(button.width(), text_room(button, button.text()), button.text())
        row.close()

    def test_the_deck_and_the_library_fit(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            db = LibraryDB(os.path.join(folder, "library.db"))
            db.connect()
            db.close()
            cfg = default_cfg()
            cfg.libconfig.libpath = folder
            bus = EventBus()
            library = LibraryService(folder, settings=lambda: cfg.searchconfig, bus=bus)
            library.refresh()
            for _ in range(4):
                library.wait()
            pane = MainPane(bus, DataModel(), TimelineCoordinator(bus), cfg, library)
            pane.install_default()
            pane.show()
            drain(10)
            size = pane.minimumSizeHint()
            pane.close()
            pane.clear_views()
            pane.deleteLater()
            drain(10)
            library.close()
        self.assertLessEqual(size.width(), WIDTH)
        self.assertLessEqual(size.height(), HEIGHT - TASKBAR_AND_TITLE - MENU_TOOLBAR_STATUS)


if __name__ == "__main__":
    unittest.main()
