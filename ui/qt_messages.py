"""Qt's messages the app leaves out of its console (and so of the log and the diagnostics): notes that say nothing
wrong with it; every other message as Qt prints it."""
import sys

from PySide6 import QtCore

# (category, the message's start or a part of it) of each note left out
QUIET = (
    # Qt's font matching notes each family of the fallback chain lacking a script before it reaches one that has it
    # (the bundled Noto fonts have no Bengali, Tamil, Telugu, Sinhala ...: the language picker's names in their own
    # scripts end on the system's font, Nirmala UI): hundreds of lines for text drawn right
    ("qt.text.font.db", "OpenType support missing for"),
    # pyqtgraph's mkQApp (each GraphicsLayoutWidget made) follows the system's dark mode by a plain function
    # connected as a unique connection, which Qt does not check for a function: the app sets its colours itself
    ("qt.core.qobject.connect", "QObject::connect(QStyleHints, QStyleHints): unique connections require"),
    # the media backend's version, said when the audio inputs are first listed (Search by Music)
    ("qt.multimedia.ffmpeg", "Using Qt multimedia with FFmpeg version"),
)
_previous_handler = None


def quiet(category: str, message: str) -> bool:
    return any(category == c and message.startswith(start) for c, start in QUIET)


def _qt_messages(kind, context, message):
    if quiet(context.category or "", message):
        return
    if _previous_handler is not None:
        _previous_handler(kind, context, message)
    else:
        print(QtCore.qFormatLogMessage(kind, context, message), file=sys.stderr)


def install() -> None:
    """The QUIET notes left out from now on (the start installs it once, before the fonts load)."""
    global _previous_handler
    previous = QtCore.qInstallMessageHandler(_qt_messages)
    if previous is not _qt_messages:     # installed again: not passed on to itself
        _previous_handler = previous
