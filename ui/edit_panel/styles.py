from PySide6.QtGui import QColor

from ui.theme import current_theme

_T = current_theme()

# Normal buttons follow the app style sheet (ui.theme); setting this clears a
# warning state.
BUTTON_STYLESHEET = ""
# Unsaved edits (SAVE) / tap BPM under a tempo change.
WARNING_BUTTON_STYLESHEET = f"""
QToolButton {{
    color: #1a1a1a;
    background-color: {_T.warning};
    border-color: {_T.warning};
}}
QToolButton:hover {{
    background-color: {QColor(_T.warning).lighter(115).name()};
}}
QToolButton:disabled {{
    color: {_T.text_disabled};
    background-color: #3a3320;
    border-color: #4a4128;
}}
"""
