"""Keyboard shortcuts: the actions (``ACTIONS``) and the manager binding their keys (``core.shortcuts``) to what
the main window's parts do for them (``MainPane.shortcut_handlers``).

The shipped keys follow rekordbox's (Performance 1 preset, deck 1) where Mixlyzer has the function, in three tiers:
no modifier (Shift / Alt variants included) for what changes no data (playback, the playhead, the library's
selection, the editor's pages and lock), Ctrl (Command on macOS) for what changes the data in memory (the editor's
edits, kept until saved), Ctrl+Shift for what destroys data (deleting a cue, removing tracks, a full reanalysis).

Every action has its own keys. An operation more than one editor page has, the same on each (a key or phrase
boundary earlier / later, split, merge, reanalysis, the edit modes), is one action done on the current page
(``EditPanel._context``: the page shown, or with pages side by side the one picked or clicked last, marked in the
selector); a page's own tool shows its page first (and the CUE list page holding its slot), which becomes the
current page. Either way the button is worked as the mouse does and flashes; an edit while editing is locked does
nothing. A handler returns a message for the status bar when it does nothing for a reason worth telling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from PySide6 import QtCore, QtGui, QtWidgets

from core import os_support
from core.i18n import tr, tr_mark
from utils.phrases import PHRASE_LABELS, phrase_display_name

# the groups, in the settings' order (translated where shown; Key and CUE as they are)
PLAYBACK = tr_mark("Playback")
CUE = "CUE"
LIBRARY = tr_mark("Library")
EDITOR = tr_mark("Editor")
CURRENT_PAGE = tr_mark("Current page")
BEATGRID = tr_mark("Beatgrid")
KEY = "Key"
PHRASE = tr_mark("Phrase")
GROUPS = (PLAYBACK, CUE, LIBRARY, EDITOR, CURRENT_PAGE, BEATGRID, KEY, PHRASE)

JUMPCUE_LETTERS = "ABCDEFGH"


@dataclass(frozen=True)
class Action:
    id: str
    group: str
    text: str                 # its name, short; translated where shown, filled with ``args``
    repeat: bool = False      # repeats while the key is held (nudges, steps), at the system's key repeat
    hold_rate: float = 0.0    # > 0: repeats while the key is held at this many a second, from the press on (the
                              # playhead's steps: an even pace, not the system's delay and rate)
    momentary: bool = False   # its handler takes pressed (True) and let go (False): CUE, held to play
    args: dict = field(default_factory=dict)
    tip: str = ""             # the details, for its tooltip (translated where shown, filled with ``args``)

    def _fill(self, text: str) -> str:
        if not self.args:
            return text
        return text.format(**{k: (v() if callable(v) else v) for k, v in self.args.items()})

    def label(self) -> str:
        return self._fill(tr(self.text))

    def tooltip(self) -> str:
        """The details as rich text (wraps), or "" without any."""
        return f"<p>{self._fill(tr(self.tip))}</p>" if self.tip else ""


def _page(name: str) -> Callable[[], str]:
    return lambda: tr(name) if name in (BEATGRID, PHRASE) else name


ACTIONS: tuple[Action, ...] = (
    Action("playback.play_pause", PLAYBACK, tr_mark("Play / pause")),
    Action("playback.cue", PLAYBACK, "CUE", momentary=True,
           tip=tr_mark("As rekordbox's CUE: stopped, sets the cue point at the playhead (on the nearest beat); held "
                       "on the cue point, plays until let go (PLAY while held: plays on); playing, back to the cue "
                       "point and stop.")),
    Action("playback.stop", PLAYBACK, tr_mark("Stop"), tip=tr_mark("Stop and go back to the start of the track.")),
    Action("playback.beat_back", PLAYBACK, tr_mark("One beat back"), hold_rate=8.0,
           tip=tr_mark("The playhead one beat back: while playing, as a beat jump (its place in the beat kept); "
                       "stopped, to the beat itself.")),
    Action("playback.beat_forward", PLAYBACK, tr_mark("One beat forward"), hold_rate=8.0,
           tip=tr_mark("The playhead one beat forward: while playing, as a beat jump (its place in the beat "
                       "kept); stopped, to the beat itself.")),
    Action("playback.bar_back", PLAYBACK, tr_mark("One bar back"), hold_rate=4.0,
           tip=tr_mark("The playhead one bar back, as one beat back by downbeats.")),
    Action("playback.bar_forward", PLAYBACK, tr_mark("One bar forward"), hold_rate=4.0,
           tip=tr_mark("The playhead one bar forward, as one beat forward by downbeats.")),
    Action("playback.tempo_down", PLAYBACK, tr_mark("Tempo -1%"), repeat=True),
    Action("playback.tempo_up", PLAYBACK, tr_mark("Tempo +1%"), repeat=True),
    Action("playback.tempo_reset", PLAYBACK, tr_mark("Tempo reset (100%)")),
    Action("playback.tempo_range", PLAYBACK, tr_mark("Tempo range"),
           tip=tr_mark("The tempo fader's range: ±6% / ±10% / ±16% / WIDE (0%-200%), the next one.")),
    Action("playback.master_tempo", PLAYBACK, tr_mark("Master tempo on / off")),
    Action("playback.metronome", PLAYBACK, tr_mark("Metronome on / off")),
    Action("playback.zoom_in", PLAYBACK, tr_mark("Zoom in"), repeat=True),
    Action("playback.zoom_out", PLAYBACK, tr_mark("Zoom out"), repeat=True),
    Action("playback.unload", PLAYBACK, tr_mark("Unload the track")),

    *(Action(f"cue.jump.{c}", CUE, tr_mark("JUMP: JumpCUE {letter}"), args={"letter": c},
             tip=tr_mark("Jump to JumpCUE {letter} on the next beat (stopped: move the playhead there)."))
      for c in JUMPCUE_LETTERS),
    Action("cue.call_previous", CUE, tr_mark("CALL: the previous CUE point"),
           tip=tr_mark("Move the playhead to the CUE point before it (while stopped).")),
    Action("cue.call_next", CUE, tr_mark("CALL: the next CUE point"),
           tip=tr_mark("Move the playhead to the CUE point after it (while stopped).")),
    *(Action(f"cue.set.{c}", CUE, tr_mark("SET: JumpCUE {letter}"), args={"letter": c},
             tip=tr_mark("JumpCUE {letter} at the nearest beat or track edge, when its slot is empty."))
      for c in JUMPCUE_LETTERS),
    Action("cue.memory", CUE, tr_mark("MEMORY: a CUE point"),
           tip=tr_mark("MEMORY: set a CUE point at the nearest beat or track edge")),
    *(Action(f"cue.delete.{c}", CUE, tr_mark("Delete JumpCUE {letter}"), args={"letter": c}) for c in JUMPCUE_LETTERS),
    Action("cue.memory_delete", CUE, tr_mark("Delete the CUE point at the playhead"),
           tip=tr_mark("The CUE point the playhead is on, as after CALL.")),

    Action("library.previous", LIBRARY, tr_mark("Previous track"), repeat=True),
    Action("library.next", LIBRARY, tr_mark("Next track"), repeat=True),
    Action("library.load", LIBRARY, tr_mark("Load the selected track"),
           tip=tr_mark("In the sources panel: the highlighted playlist or folder shown in the library.")),
    Action("library.browse_sources", LIBRARY, tr_mark("Sources panel"),
           tip=tr_mark("To the panel beside the library (playlists, local folders) and back. Up / Down browse, "
                       "Left / Right close / open a folder, the load key shows the highlighted one.")),
    Action("library.search", LIBRARY, tr_mark("Search")),
    Action("library.leave_search", LIBRARY, tr_mark("Back to the tracks"),
           tip=tr_mark("To the track list from the search field (its text kept), the sources panel or the track "
                       "panel; out of another text field: the keys work the shortcuts again.")),
    Action("library.locate_loaded", LIBRARY, tr_mark("Select the loaded track")),
    Action("library.edit_track", LIBRARY, tr_mark("Edit Track"),
           tip=tr_mark("The track panel's Edit tab with the selected track, its first field focused.")),
    Action("library.export_track", LIBRARY, tr_mark("Export Track"),
           tip=tr_mark("The track panel's Export tab with the selected track, focused.")),
    Action("library.write_tags", LIBRARY, tr_mark("Write Tags to File"),
           tip=tr_mark("The selected track's stored values written into its audio file's tags: the fields checked "
                       "in the track panel's Export tab.")),
    Action("library.reveal", LIBRARY, tr_mark("Show in Explorer") if os_support.IS_WINDOWS else
           tr_mark("Show in Finder") if os_support.IS_MACOS else tr_mark("Show in File Manager")),
    Action("library.new_playlist", LIBRARY, tr_mark("New Playlist")),
    Action("library.reanalyze", LIBRARY, tr_mark("Reanalyze Track"),
           tip=tr_mark("A full analysis of the selected tracks: their stored analysis and edits are replaced.")),
    Action("library.remove_from_library", LIBRARY, tr_mark("Remove from Library")),
    Action("library.remove_from_playlist", LIBRARY, tr_mark("Remove from Playlist")),

    Action("editor.lock", EDITOR, tr_mark("Edit lock on / off")),
    *(Action(f"editor.page_{name.lower()}", EDITOR, tr_mark("Show the {page} page"), args={"page": _page(name)},
             tip=tr_mark("It becomes the current page."))
      for name in (BEATGRID, KEY, PHRASE, CUE)),
    Action("editor.undo", EDITOR, tr_mark("Undo Changes"), repeat=True),
    Action("editor.redo", EDITOR, tr_mark("Redo Changes"), repeat=True),
    Action("editor.save", EDITOR, tr_mark("Save Edits")),

    # the same operation on more than one page, done on the current page
    Action("editor.start_earlier", CURRENT_PAGE, tr_mark("Start earlier"), repeat=True,
           tip=tr_mark("Key, Phrase: the current segment's start, by a beat or a bar (the boundary step).")),
    Action("editor.start_later", CURRENT_PAGE, tr_mark("Start later"), repeat=True,
           tip=tr_mark("Key, Phrase: the current segment's start, by a beat or a bar (the boundary step).")),
    Action("editor.end_earlier", CURRENT_PAGE, tr_mark("End earlier"), repeat=True,
           tip=tr_mark("Key, Phrase: the current segment's end, by a beat or a bar (the boundary step).")),
    Action("editor.end_later", CURRENT_PAGE, tr_mark("End later"), repeat=True,
           tip=tr_mark("Key, Phrase: the current segment's end, by a beat or a bar (the boundary step).")),
    Action("editor.split", CURRENT_PAGE, tr_mark("Split at the playhead"),
           tip=tr_mark("Beatgrid, Key, Phrase: the current segment.")),
    Action("editor.merge_previous", CURRENT_PAGE, tr_mark("Merge the previous segment"),
           tip=tr_mark("Beatgrid, Key, Phrase: the previous segment into the current one.")),
    Action("editor.merge_next", CURRENT_PAGE, tr_mark("Merge the next segment"),
           tip=tr_mark("Beatgrid, Key, Phrase: the next segment into the current one.")),
    Action("editor.reanalyze_segment", CURRENT_PAGE, tr_mark("Reanalyze the current segment"),
           tip=tr_mark("Beatgrid, Key.")),
    Action("editor.reanalyze_all", CURRENT_PAGE, tr_mark("Reanalyze the whole track"),
           tip=tr_mark("Key, Phrase; on the CUE page, the JumpCUEs.")),
    Action("editor.scope_all", CURRENT_PAGE, tr_mark("Edit scope: the whole track"), tip=tr_mark("Beatgrid, Key.")),
    Action("editor.scope_segment", CURRENT_PAGE, tr_mark("Edit scope: the current segment"),
           tip=tr_mark("Beatgrid, Key.")),
    Action("editor.step_beat", CURRENT_PAGE, tr_mark("Boundary step: one beat"), tip=tr_mark("Key, Phrase.")),
    Action("editor.step_bar", CURRENT_PAGE, tr_mark("Boundary step: one bar"), tip=tr_mark("Key, Phrase.")),

    Action("beatgrid.shift_half_back", BEATGRID, tr_mark("Shift half a beat earlier"), repeat=True),
    Action("beatgrid.shift_coarse_back", BEATGRID, tr_mark("Shift 10 ms earlier"), repeat=True),
    Action("beatgrid.shift_fine_back", BEATGRID, tr_mark("Shift 1 ms earlier"), repeat=True),
    Action("beatgrid.shift_fine_forward", BEATGRID, tr_mark("Shift 1 ms later"), repeat=True),
    Action("beatgrid.shift_coarse_forward", BEATGRID, tr_mark("Shift 10 ms later"), repeat=True),
    Action("beatgrid.shift_half_forward", BEATGRID, tr_mark("Shift half a beat later"), repeat=True),
    Action("beatgrid.zero", BEATGRID, tr_mark("ZERO"), tip=tr_mark("Clear the reference BPM and the tap history")),
    Action("beatgrid.tap", BEATGRID, tr_mark("TAP"), tip=tr_mark("Tap the beat to set the reference BPM")),
    Action("beatgrid.assign", BEATGRID, tr_mark("Assign the reference BPM"),
           tip=tr_mark("Assign the reference BPM to the current segment (downbeat at its start)")),
    Action("beatgrid.ramp_start", BEATGRID, tr_mark("Ramp: the starting tempo"),
           tip=tr_mark("Make the reference BPM the current segment's starting tempo (a ramp to its end tempo; "
                       "its first beat stays)")),
    Action("beatgrid.ramp_end", BEATGRID, tr_mark("Ramp: the ending tempo"),
           tip=tr_mark("Make the reference BPM the current segment's ending tempo (a ramp from its start tempo; "
                       "its first beat stays)")),
    Action("beatgrid.downbeat", BEATGRID, tr_mark("Downbeat at the playhead"),
           tip=tr_mark("Make the beat nearest the playhead the current segment's downbeat")),
    Action("beatgrid.bpm_half", BEATGRID, tr_mark("BPM x1/2")),
    Action("beatgrid.bpm_double", BEATGRID, tr_mark("BPM x2")),
    *(Action(f"beatgrid.time_signature_{n}", BEATGRID, tr_mark("Time signature {meter}"), args={"meter": f"{n}/4"})
      for n in (3, 4, 5, 7)),

    Action("key.pick", KEY, tr_mark("Pick the key"), tip=tr_mark("Pick the key (Alphanumeric wheel)")),
    Action("key.flat", KEY, tr_mark("A semitone down"), repeat=True),
    Action("key.sharp", KEY, tr_mark("A semitone up"), repeat=True),
    Action("key.fifth_down", KEY, tr_mark("Circle of fifths -1"), repeat=True,
           tip=tr_mark("One step down the circle of fifths (Alphanumeric -1)")),
    Action("key.fifth_up", KEY, tr_mark("Circle of fifths +1"), repeat=True,
           tip=tr_mark("One step up the circle of fifths (Alphanumeric +1)")),
    Action("key.relative", KEY, tr_mark("Relative major / minor"), tip=tr_mark("Relative major / minor (C <-> Am)")),
    Action("key.merge_same", KEY, tr_mark("Merge the same keys"),
           tip=tr_mark("Merge every run of neighboring segments with the same key")),
    Action("key.root", KEY, tr_mark("Listening aid: root")),
    Action("key.drone", KEY, tr_mark("Listening aid: root and fifth")),
    Action("key.chord", KEY, tr_mark("Listening aid: tonic chord")),

    Action("phrase.cue_points", PHRASE, tr_mark("CUE points from the phrases"),
           tip=tr_mark("Replace the CUE points with the ones the CUE rules give the current phrases")),
    Action("phrase.unassign", PHRASE, tr_mark("Unassign the current phrase (UNASSIGNED)")),
    *(Action(f"phrase.label.{label}", PHRASE, tr_mark("Label the current phrase {label}"),
             args={"label": (lambda label=label: phrase_display_name(label))}) for label in PHRASE_LABELS),
)

# plain keys a focused list or value field takes rather than the shortcuts: moving in a list, stepping a value
# (typed characters already go to a text field)
_NAVIGATION_KEYS = {QtCore.Qt.Key_Up, QtCore.Qt.Key_Down, QtCore.Qt.Key_PageUp, QtCore.Qt.Key_PageDown,
                    QtCore.Qt.Key_Home, QtCore.Qt.Key_End}
_COMMAND_MODIFIERS = QtCore.Qt.ControlModifier | QtCore.Qt.AltModifier | QtCore.Qt.MetaModifier

STATUS_MS = 3000


class ShortcutManager(QtCore.QObject):
    """One QShortcut per action on ``window`` (keys set by ``apply``); ``handlers``: action id -> callable
    returning None or a message for ``status(message, ms)``. An action with a hold rate is repeated by a timer
    while its key is held (the key's release, or the app losing the focus, stops it); a momentary one's handler
    is called with True when its key is pressed and False when it is let go (or the app loses the focus)."""

    def __init__(self, window: QtWidgets.QWidget, handlers: dict[str, Callable[[], Optional[str]]],
                 status: Callable[[str, int], None]):
        super().__init__(window)
        self._status = status
        self._shortcuts: dict[str, QtGui.QShortcut] = {}
        self._held: Optional[tuple[str, Callable[[], Optional[str]]]] = None   # (action id, handler) held
        self._held_momentary = False
        self._hold_timer = QtCore.QTimer(self)
        self._hold_timer.setTimerType(QtCore.Qt.PreciseTimer)
        self._hold_timer.timeout.connect(self._hold_step)
        for action in ACTIONS:
            handler = handlers.get(action.id)
            if handler is None:
                print(f"[Shortcuts] no handler for {action.id}")
                continue
            sc = QtGui.QShortcut(window)
            sc.setContext(QtCore.Qt.WindowShortcut)
            sc.setAutoRepeat(action.repeat and action.hold_rate <= 0.0 and not action.momentary)
            if action.momentary:
                sc.activated.connect(lambda action=action, handler=handler: self._press_momentary(action, handler))
            elif action.hold_rate > 0.0:
                sc.activated.connect(lambda action=action, handler=handler: self._press(action, handler))
            else:
                sc.activated.connect(lambda handler=handler: self._run(handler))
            sc.activatedAmbiguously.connect(lambda action=action: self._ambiguous(action))
            self._shortcuts[action.id] = sc
        app = QtWidgets.QApplication.instance()
        app.installEventFilter(self)
        app.applicationStateChanged.connect(
            lambda state: state == QtCore.Qt.ApplicationActive or self._release())

    def apply(self, keymap: dict[str, list[str]]) -> None:
        for action_id, sc in self._shortcuts.items():
            sc.setKeys([QtGui.QKeySequence(k) for k in keymap.get(action_id, [])])
        # keys another program holds never arrive: said once they are set
        held = [(k, action_id) for action_id in self._shortcuts for k in keymap.get(action_id, [])
                if os_support.hotkey_taken(k)]
        if held:
            print("[Shortcuts] held by another program:", ", ".join(f"{k} ({a})" for k, a in held))
            keys = ", ".join(QtGui.QKeySequence(k).toString(QtGui.QKeySequence.NativeText) for k, _a in held)
            self._status(tr("Another program on this computer takes {keys}: these shortcuts do not work "
                            "(Settings > Shortcuts).").format(keys=keys), 10000)

    def _run(self, handler: Callable[[], Optional[str]]) -> None:
        message = handler()
        if message:
            self._status(message, STATUS_MS)

    def _press(self, action: Action, handler: Callable[[], Optional[str]]) -> None:
        self._release()   # a key held before (CUE) is let go first: one held at a time
        self._run(handler)
        self._held = (action.id, handler)
        self._hold_timer.start(max(1, round(1000.0 / action.hold_rate)))

    def _press_momentary(self, action: Action, handler: Callable[[bool], Optional[str]]) -> None:
        if self._held is not None and self._held[0] == action.id:
            return   # the key's own repeat
        self._release()
        message = handler(True)
        if message:
            self._status(message, STATUS_MS)
        self._held = (action.id, handler)
        self._held_momentary = True

    def _hold_step(self) -> None:
        if self._held is None:
            self._hold_timer.stop()
            return
        self._run(self._held[1])

    def _release(self) -> None:
        held, momentary = self._held, self._held_momentary
        self._held = None
        self._held_momentary = False
        self._hold_timer.stop()
        if held is not None and momentary:
            held[1](False)

    def _held_keys(self) -> set[int]:
        """The keys (without modifiers) of the held action's shortcuts."""
        if self._held is None:
            return set()
        return {_key_value(seq[0].key()) for seq in self._shortcuts[self._held[0]].keys() if not seq.isEmpty()}

    def _ambiguous(self, action: Action) -> None:
        keys = ", ".join(k.toString(QtGui.QKeySequence.NativeText) for k in self._shortcuts[action.id].keys())
        self._status(tr("{keys} is used by more than one action (Settings > Shortcuts).").format(keys=keys), STATUS_MS)

    def eventFilter(self, obj, event) -> bool:
        if (self._held is not None and event.type() == QtCore.QEvent.KeyRelease and not event.isAutoRepeat()
                and _key_value(event.key()) in self._held_keys()):
            self._release()
        # a plain navigation key moves in the focused list or steps the focused value instead of a shortcut
        if (event.type() == QtCore.QEvent.ShortcutOverride and not (event.modifiers() & _COMMAND_MODIFIERS)
                and _takes_navigation(obj, event)):
            event.accept()
        return False


def _key_value(key) -> int:
    """A Qt key as an int (QKeyEvent.key() gives one, QKeyCombination.key() an enum)."""
    return int(getattr(key, "value", key))


def _takes_navigation(widget, event) -> bool:
    """Lists, value fields (spin and combo boxes, also through their line edit) and multi-line text: not a
    plain line edit, where Up / Down do nothing (they go to the library's list). A tree also takes plain Left /
    Right (closing / opening a folder); a table edited by typing takes typed characters and F2 (editing a cell),
    and the editor of a cell every plain key (Esc cancels it)."""
    if not isinstance(widget, QtWidgets.QWidget):
        return False
    key = event.key()
    if _cell_editor(widget):
        return True
    if isinstance(widget, QtWidgets.QAbstractItemView):
        triggers = widget.editTriggers()
        if triggers & QtWidgets.QAbstractItemView.AnyKeyPressed and event.text().isprintable() and event.text():
            return True
        if triggers & QtWidgets.QAbstractItemView.EditKeyPressed and key == QtCore.Qt.Key_F2:
            return True
    if isinstance(widget, QtWidgets.QTreeView) and key in (QtCore.Qt.Key_Left, QtCore.Qt.Key_Right):
        return event.modifiers() == QtCore.Qt.NoModifier
    parent = widget.parentWidget()
    return key in _NAVIGATION_KEYS and (
        isinstance(widget, (QtWidgets.QAbstractItemView, QtWidgets.QAbstractSpinBox, QtWidgets.QComboBox,
                            QtWidgets.QTextEdit, QtWidgets.QPlainTextEdit))
        or isinstance(parent, (QtWidgets.QAbstractSpinBox, QtWidgets.QComboBox)))


def _cell_editor(widget: QtWidgets.QWidget) -> bool:
    """A list's or table's cell being edited (its editor sits in the view's viewport)."""
    parent = widget.parentWidget()
    view = parent.parentWidget() if parent is not None else None
    return isinstance(view, QtWidgets.QAbstractItemView) and parent is view.viewport()
