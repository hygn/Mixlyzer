"""A MIDI controller's mapping: the shipped one in ``assets/midi/default_midi.json`` (empty: no controller) and the
user's in ``assets/midi/midi.json``, as the keyboard shortcuts' (``core.shortcuts``): an action -> its MIDI
messages, the user's file holding only the actions changed from the shipped mapping.

The actions are the keyboard shortcuts' (``ui.shortcuts.ACTIONS``) and a controller's own (``CONTROLS``: faders,
a jog wheel). The native audio engine reads the controller (audioengine/native/src/deck.rs, never through the
GUI's Python): the deck's actions (``DECK_ACTIONS``) are done there; the others are passed on to the GUI, which
runs them as their shortcuts. A message is written as rekordbox's MIDI CSV writes it: status and data 1 in hex,
"900B" (a note on channel 1) or "B000" (a CC); "H" after a CC: 14 bits, its LSB at data 1 + 32 ("B000H").

``import_rekordbox_csv`` reads a mapping exported from rekordbox (MIDI settings: DDJ-400 and the like): deck 1's
column and the channel fader of mixer channel 1 (its level meter from the output column); the deck 2 / global
columns are left out.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

from audioengine.native import client as deck
from core.i18n import tr_mark
from core.model_files import is_shipped
from core.resource_paths import resource_path
from utils.atomic_io import atomic_write_json

DEFAULT_FILE = "assets/midi/default_midi.json"
USER_FILE = "assets/midi/midi.json"

Mapping = dict[str, list[str]]

BUTTON, FADER, JOG, TOUCH, METER = "button", "fader", "jog", "touch", "meter"


@dataclass(frozen=True)
class Control:
    id: str
    text: str     # its name (translated where shown)
    kind: str     # FADER (absolute CC), JOG (relative CC), TOUCH (a note), METER (an output CC)
    code: int     # the deck's action
    tip: str = ""


# a controller's own controls (the deck's, done in the engine)
CONTROLS: tuple[Control, ...] = (
    Control("midi.tempo_fader", tr_mark("Tempo fader"), FADER, deck.ACT_TEMPO_FADER,
            tr_mark("The tempo over the tempo range (Tempo range); its middle is 100%.")),
    Control("midi.volume_fader", tr_mark("Volume fader"), FADER, deck.ACT_VOLUME_FADER,
            tr_mark("The deck's volume (the mixer's channel fader).")),
    Control("midi.jog_touch", tr_mark("Jog wheel: touch"), TOUCH, deck.ACT_JOG_TOUCH,
            tr_mark("The jog wheel's top touched: the record held (scratch); let go, it plays on.")),
    Control("midi.jog_scratch", tr_mark("Jog wheel: scratch"), JOG, deck.ACT_JOG_SCRATCH,
            tr_mark("The jog wheel's top turned: touched, the record moves with it (33 1/3 rpm); else a pitch "
                    "bend.")),
    Control("midi.jog_bend", tr_mark("Jog wheel: pitch bend"), JOG, deck.ACT_JOG_BEND,
            tr_mark("The jog wheel turned: the tempo bent while it turns (a turn a second: +10%).")),
    Control("midi.jog_search", tr_mark("Jog wheel: search"), JOG, deck.ACT_JOG_SEARCH,
            tr_mark("The jog wheel turned: the playhead moved through the track fast.")),
    Control("midi.level_meter", tr_mark("Level meter (output)"), METER, deck.ACT_LEVEL_METER,
            tr_mark("The controller's level meter shows the deck's peak meter (a CC sent, 0 - 127 over -24 - 0 dBFS); "
                    "taken from a rekordbox mapping (its ChannelLevel), not learned.")),
)

# the shortcuts' actions the deck does itself
DECK_ACTIONS: dict[str, int] = {
    "playback.play_pause": deck.ACT_PLAY_PAUSE,
    "playback.cue": deck.ACT_CUE,
    "playback.stop": deck.ACT_STOP,
    "playback.beat_back": deck.ACT_BEAT_BACK,
    "playback.beat_forward": deck.ACT_BEAT_FORWARD,
    "playback.bar_back": deck.ACT_BAR_BACK,
    "playback.bar_forward": deck.ACT_BAR_FORWARD,
    "playback.tempo_down": deck.ACT_TEMPO_DOWN,
    "playback.tempo_up": deck.ACT_TEMPO_UP,
    "playback.tempo_reset": deck.ACT_TEMPO_RESET,
    "playback.tempo_range": deck.ACT_TEMPO_RANGE,
    "cue.call_previous": deck.ACT_CALL_PREVIOUS,
    "cue.call_next": deck.ACT_CALL_NEXT,
    **{f"cue.jump.{c}": deck.ACT_JUMP_A + i for i, c in enumerate("ABCDEFGH")},
}

_MESSAGE = re.compile(r"^([89AB])([0-9A-F])([0-7][0-9A-F])(H?)$")


@dataclass(frozen=True)
class Message:
    status: int   # a note off is read as its note on (0x9n)
    data1: int
    hires: bool = False

    @classmethod
    def parse(cls, text: str) -> "Message | None":
        m = _MESSAGE.match(str(text or "").strip().upper())
        if m is None:
            return None
        kind, channel, data1, hires = m.groups()
        status = (0x90 if kind in "89" else int(kind, 16) << 4) | int(channel, 16)
        return cls(status, int(data1, 16), bool(hires) and status & 0xF0 == 0xB0)

    def text(self) -> str:
        return f"{self.status:02X}{self.data1:02X}" + ("H" if self.hires else "")

    def label(self) -> str:
        """As a person reads it: "Ch 1 Note 0x0B", "Ch 1 CC 0x00 (14-bit)"."""
        channel = (self.status & 0x0F) + 1
        kind = {0x90: "Note", 0xA0: "Aftertouch", 0xB0: "CC"}.get(self.status & 0xF0, "?")
        return f"Ch {channel} {kind} 0x{self.data1:02X}" + (" (14-bit)" if self.hires else "")


def _read(path: Path) -> Mapping:
    """The action -> messages entries of a mapping file (malformed ones left out)."""
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"[MIDI] {path.name} not read: {exc}")
        return {}
    if not isinstance(data, dict):
        print(f"[MIDI] {path.name} not read: not an object")
        return {}
    out: Mapping = {}
    for action, messages in data.items():
        if isinstance(messages, list) and all(isinstance(m, str) for m in messages):
            out[str(action)] = [Message.parse(m).text() for m in messages if Message.parse(m) is not None]
        else:
            print(f"[MIDI] {path.name}: {action} left out (not a list of messages)")
    return out


def defaults() -> Mapping:
    return _read(resource_path(DEFAULT_FILE))


def user_changes() -> Mapping:
    return _read(resource_path(USER_FILE))


def mapping() -> Mapping:
    """The mapping in use: the shipped one with the user's changes."""
    return {**defaults(), **user_changes()}


def save(messages: Mapping, path: str | None = None) -> None:
    """Keep ``messages`` (every action's) as the user's: only the actions that differ from the shipped mapping are
    written; with none, the user's file is removed."""
    target = resource_path(path or USER_FILE)
    if is_shipped(target):
        raise ValueError(f"the shipped MIDI mapping ({target.name}) is not written")
    shipped = defaults()
    changes = {action: list(m) for action, m in messages.items() if list(m) != shipped.get(action, [])}
    if not changes:
        target.unlink(missing_ok=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(target, changes)


def action_code(action_id: str, gui_ids: list[str]) -> int | None:
    """The engine's code of an action: the deck's own, or ACT_GUI + its place among ``gui_ids`` (passed on)."""
    if action_id in DECK_ACTIONS:
        return DECK_ACTIONS[action_id]
    control = next((c for c in CONTROLS if c.id == action_id), None)
    if control is not None:
        return control.code
    if action_id in gui_ids:
        return deck.ACT_GUI + gui_ids.index(action_id)
    return None


def lamp(action_id: str) -> int:
    """The controller's light at an action's message (client.LAMP_*; 0 none): PLAY, CUE, a JumpCUE's pads (JUMP
    and its SHIFT layer's delete: lit while it is set)."""
    if action_id == "playback.play_pause":
        return deck.LAMP_PLAY
    if action_id == "playback.cue":
        return deck.LAMP_CUE
    m = re.match(r"^cue\.(jump|delete)\.([A-Z])$", action_id)
    return ord(m.group(2)) - ord("A") + 1 if m else 0


def bindings(messages: Mapping, gui_ids: list[str]) -> list[tuple[int, int, bool, int, int]]:
    """The engine's bindings (status, data1, 14-bit, lamp, action code) of ``messages``."""
    out = []
    for action_id, texts in messages.items():
        code = action_code(action_id, gui_ids)
        if code is None:
            continue
        for text in texts:
            m = Message.parse(text)
            if m is not None:
                out.append((m.status, m.data1, m.hires, lamp(action_id), code))
    return out


# rekordbox's MIDI CSV: its function (the first column) -> the action
_REKORDBOX = {
    "PlayPause": "playback.play_pause",
    "Cue": "playback.cue",
    "JumpToTrackStart": "playback.stop",
    "TempoSlider": "midi.tempo_fader",
    "TempoRange": "playback.tempo_range",
    "ChannelFader": "midi.volume_fader",
    "JogTouch": "midi.jog_touch",
    "JogScratch": "midi.jog_scratch",
    "JogPitchBend": "midi.jog_bend",
    "WheelPitchBend": "midi.jog_bend",
    "JogSearch": "midi.jog_search",
    "Parameter1Left_HotCue": "cue.call_previous",
    "Parameter1Right_HotCue": "cue.call_next",
    "Memory": "cue.memory",
    "Delete": "cue.memory_delete",
    "Load": "library.load",
    # BEAT JUMP pads: 1 beat back / on (pads 1, 2), 4 beats: a bar (pads 5, 6)
    "PAD1_BeatJump": "playback.beat_back",
    "PAD2_BeatJump": "playback.beat_forward",
    "PAD5_BeatJump": "playback.bar_back",
    "PAD6_BeatJump": "playback.bar_forward",
    **{f"PAD{i + 1}_HotCue": f"cue.jump.{c}" for i, c in enumerate("ABCDEFGH")},
    **{f"PAD{i + 1}_HotCue+Shift": f"cue.delete.{c}" for i, c in enumerate("ABCDEFGH")},
}
# outputs: taken from deck 1's output column
_REKORDBOX_OUTPUT = {"ChannelLevel": "midi.level_meter"}
_HEX4 = re.compile(r"^[0-9A-Fa-f]{4}$")


def import_rekordbox_csv(path: str) -> tuple[Mapping, list[str]]:
    """A rekordbox MIDI mapping (CSV: function, controller name, type, input, deck 1-4 inputs, output, deck 1-4
    outputs, options, comment): deck 1's controls (its column: a whole message, "900B", or a channel offset to
    the input column's, "0") as Mixlyzer's actions. Returns (the mapping, the functions not taken)."""
    out: Mapping = {}
    skipped: list[str] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.reader(f):
            if not row or not row[0].strip() or row[0].lstrip().startswith(("@", "#")):
                continue
            row = [c.strip() for c in row] + [""] * (15 - len(row))
            name, kind, base, deck1 = row[0], row[2], row[3], row[4]
            if name in _REKORDBOX_OUTPUT:   # an indicator: deck 1's output (or the output's, offset by it)
                base, deck1 = row[8], row[9]
            action = _REKORDBOX.get(name) or _REKORDBOX_OUTPUT.get(name)
            if action is None:
                skipped.append(name)
                continue
            if _HEX4.match(deck1):
                text = deck1.upper()
            elif deck1.isdigit() and _HEX4.match(base):   # a channel offset to the input's message
                status = (int(base[:2], 16) & 0xF0) | ((int(base[:2], 16) + int(deck1)) & 0x0F)
                text = f"{status:02X}{base[2:].upper()}"
            else:
                skipped.append(name)   # not on deck 1 (deck 2, the browser, the effects ...)
                continue
            if kind == "KnobSliderHiRes":
                text += "H"
            message = Message.parse(text)
            if message is None:
                skipped.append(name)
                continue
            texts = out.setdefault(action, [])
            if message.text() not in texts:
                texts.append(message.text())
    return out, sorted(set(skipped))
