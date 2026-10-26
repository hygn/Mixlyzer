"""A MIDI controller's mapping: the shipped one in ``assets/midi/default_midi.json`` (empty: no controller) and the
user's in ``assets/midi/midi.json``, as the keyboard shortcuts' (``core.shortcuts``): an action -> its MIDI
messages, the user's file holding only the actions changed from the shipped mapping.

The actions (``ACTIONS``) are a DJ controller's, as rekordbox / a CDJ and DJM do them, not the keyboard shortcuts:
the deck (PLAY, CUE, SHIFT + CUE to the track's start, the tempo fader and its range, pitch bend buttons, the jog
wheel), the JumpCUE pads (JUMP; SET on an empty slot; SHIFT: delete), the CUE points (CALL, MEMORY, DELETE), beat
jumps (1 - 32 beats), the browser (its knob turned / pressed), the mixer (a channel's fader, EQ, trim and level
meter; not the crossfader: the engine's act has no action, not shown to the user). The native audio engine reads the controller (audioengine/native/src/deck.rs, never through
the GUI's Python) and does the deck's and the mixer's actions itself; the others (loading, the browser, edits) are
passed on to the GUI, which runs them as its shortcuts do.

Every deck's actions are here for decks 1 - 4 and every mixer channel's for channels 1 - 4 (the engine takes them
all); the settings show deck 1's and channel 1's that are ``shown`` (``ACTIONS_SHOWN``): of the mixer's, the volume
fader, the EQ, the headphones CUE and the level meter, and the headphones' CUE / MASTER mix and level.

A message is written as rekordbox's MIDI CSV writes it, status and data 1 in hex ("900B" a note on channel 1, "B000"
a CC, "E000" a pitch bend: data 1 not read), then its options: "H" a 14-bit CC (its LSB at data 1 + 32), "+" / "-"
a knob's turns one way only (clockwise / counterclockwise, a step an action), "I" inverted (a fader the other way up,
a knob the other way round), and a knob's encoding: "O" 64 + steps, "T" two's complement, "S" sign and magnitude
(none: told apart by the values, offset or two's complement). ``import_rekordbox_csv`` reads a mapping exported from
rekordbox (DDJ-400 and the like): deck 1's column, mixer channel 1, the browser and the headphones' knobs.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from audioengine.native import client as engine
from foundation.i18n import tr_mark
from foundation.model_files import is_shipped
from foundation.resource_paths import resource_path
from utils.atomic_io import atomic_write_json

DEFAULT_FILE = "assets/midi/default_midi.json"
USER_FILE = "assets/midi/midi.json"

Mapping = dict[str, list[str]]

# what a control is (Learn reads it so): a button (a note or a CC: pressed above 0), a fader (absolute: a CC, 7 or 14
# bits, or a pitch bend), a knob turned (a relative CC), a touch (a note or a CC), an output (a CC sent)
BUTTON, FADER, KNOB, TOUCH, METER = "button", "fader", "knob", "touch", "meter"

DECKS = 8   # decks and mixer channels the engine takes (the app shows deck 1)
JUMPCUE_PADS = "ABCDEFGH"

# the groups, in the settings' order (translated where shown; CUE as it is)
DECK = tr_mark("Deck")
JUMPCUE = tr_mark("JumpCUE pads")
CUE_POINTS = tr_mark("CUE points")
BEAT_JUMP = tr_mark("Beat jump")
TEMPO = tr_mark("Tempo")
JOG = tr_mark("Jog wheel")
BROWSER = tr_mark("Browser")
MIXER = tr_mark("Mixer")
EFFECTS = tr_mark("Effects")
GROUPS = (DECK, JUMPCUE, CUE_POINTS, BEAT_JUMP, TEMPO, JOG, BROWSER, MIXER, EFFECTS)


@dataclass(frozen=True)
class Action:
    id: str
    group: str
    text: str            # its name (translated where shown, filled with ``args``)
    kind: str
    code: int = 0        # the engine's action (client.ACT_*, the deck in its upper bits); 0: the GUI's
    gui: tuple = ()      # the GUI's: the shortcut action run (an encoder's: clockwise, counterclockwise)
    lamp: int = 0        # the controller's light at its message (client.LAMP_*)
    shown: bool = True   # in the settings (deck 1 / channel 1's)
    tip: str = ""
    args: dict = field(default_factory=dict)


def _deck_code(code: int, n: int) -> int:
    return code | (n - 1) << engine.ACT_DECK_SHIFT


def _deck_actions(n: int) -> list[Action]:
    """Deck ``n``'s actions (the GUI's for deck 1 only: the app shows one deck)."""
    d, one = f"deck{n}", n == 1
    a = {"deck": n}
    out = [
        Action(f"{d}.play_pause", DECK, tr_mark("Deck {deck}: PLAY / PAUSE"), BUTTON,
               _deck_code(engine.ACT_PLAY_PAUSE, n), lamp=engine.LAMP_PLAY, shown=one, args=a),
        Action(f"{d}.cue", DECK, tr_mark("Deck {deck}: CUE"), BUTTON, _deck_code(engine.ACT_CUE, n),
               lamp=engine.LAMP_CUE, shown=one, args=a,
               tip=tr_mark("As rekordbox's CUE: stopped, sets the cue point (on the nearest beat); held on it, plays "
                           "until let go; playing, back to it and stop.")),
        Action(f"{d}.track_start", DECK, tr_mark("Deck {deck}: back to the track's start"), BUTTON,
               _deck_code(engine.ACT_TRACK_START, n), shown=one, args=a,
               tip=tr_mark("As rekordbox's SHIFT + CUE: the track's start, playing on if it plays.")),
    ]
    if one:
        out += [
            Action("deck1.load", DECK, tr_mark("Deck {deck}: LOAD (the selected track)"), BUTTON,
                   gui=("library.load",), args=a),
            Action("deck1.master_tempo", DECK, tr_mark("Deck {deck}: MASTER TEMPO"), BUTTON,
                   gui=("playback.master_tempo",), args=a, tip=tr_mark("Key lock: the time stretch keeps the pitch.")),
        ]
    out += [
        Action(f"{d}.tempo_fader", TEMPO, tr_mark("Deck {deck}: TEMPO slider"), FADER,
               _deck_code(engine.ACT_TEMPO_FADER, n), shown=one, args=a,
               tip=tr_mark("The tempo over the tempo range; its middle is 100%. A pitch bend message or a CC.")),
        Action(f"{d}.tempo_range", TEMPO, tr_mark("Deck {deck}: TEMPO RANGE"), BUTTON,
               _deck_code(engine.ACT_TEMPO_RANGE, n), shown=one, args=a,
               tip=tr_mark("±6% / ±10% / ±16% / WIDE, the next one.")),
        Action(f"{d}.tempo_reset", TEMPO, tr_mark("Deck {deck}: TEMPO RESET"), BUTTON,
               _deck_code(engine.ACT_TEMPO_RESET, n), shown=one, args=a),
        Action(f"{d}.bend_down", TEMPO, tr_mark("Deck {deck}: PITCH BEND −"), BUTTON,
               _deck_code(engine.ACT_BEND_DOWN, n), shown=one, args=a, tip=tr_mark("Held: the tempo bent down 4%.")),
        Action(f"{d}.bend_up", TEMPO, tr_mark("Deck {deck}: PITCH BEND +"), BUTTON,
               _deck_code(engine.ACT_BEND_UP, n), shown=one, args=a, tip=tr_mark("Held: the tempo bent up 4%.")),
        Action(f"{d}.jog_mode", JOG, tr_mark("Deck {deck}: JOG MODE (VINYL / CDJ)"), BUTTON,
               _deck_code(engine.ACT_JOG_MODE, n), lamp=engine.LAMP_VINYL, shown=one, args=a,
               tip=tr_mark("A press switches the jog wheel's mode: VINYL (its top's touch holds the record) or CDJ "
                           "(a bend only). Lit in VINYL mode.")),
        Action(f"{d}.jog_touch", JOG, tr_mark("Deck {deck}: jog wheel touch"), TOUCH,
               _deck_code(engine.ACT_JOG_TOUCH, n), shown=one, args=a,
               tip=tr_mark("The top touched: the record held (vinyl mode); let go, it plays on (a spin: when it "
                           "stops).")),
        Action(f"{d}.jog_scratch", JOG, tr_mark("Deck {deck}: jog wheel top"), KNOB,
               _deck_code(engine.ACT_JOG_SCRATCH, n), shown=one, args=a,
               tip=tr_mark("Turned: touched, the record moves with it (33 1/3 rpm); else a pitch bend.")),
        Action(f"{d}.jog_bend", JOG, tr_mark("Deck {deck}: jog wheel side (pitch bend)"), KNOB,
               _deck_code(engine.ACT_JOG_BEND, n), shown=one, args=a,
               tip=tr_mark("Turned: the tempo bent while it turns (a turn a second: +10%).")),
        Action(f"{d}.jog_search", JOG, tr_mark("Deck {deck}: jog wheel search (SHIFT)"), KNOB,
               _deck_code(engine.ACT_JOG_SEARCH, n), shown=one, args=a,
               tip=tr_mark("Turned: the playhead moved through the track fast.")),
        Action(f"{d}.memory_previous", CUE_POINTS, tr_mark("Deck {deck}: CUE CALL ◁"), BUTTON,
               _deck_code(engine.ACT_CALL_PREVIOUS, n), shown=one, args=a,
               tip=tr_mark("The CUE point before (stopped); the cue point moves there.")),
        Action(f"{d}.memory_next", CUE_POINTS, tr_mark("Deck {deck}: CUE CALL ▷"), BUTTON,
               _deck_code(engine.ACT_CALL_NEXT, n), shown=one, args=a,
               tip=tr_mark("The CUE point after (stopped); the cue point moves there.")),
    ]
    if one:
        out += [
            Action("deck1.memory_set", CUE_POINTS, tr_mark("Deck {deck}: MEMORY"), BUTTON, gui=("cue.memory",), args=a,
                   tip=tr_mark("A CUE point at the nearest beat (an edit).")),
            Action("deck1.memory_delete", CUE_POINTS, tr_mark("Deck {deck}: CUE point DELETE"), BUTTON,
                   gui=("cue.memory_delete",), args=a, tip=tr_mark("The CUE point the playhead is on (an edit).")),
        ]
    for i, letter in enumerate(JUMPCUE_PADS):
        pad = {"deck": n, "letter": letter}
        out.append(Action(f"{d}.jumpcue.{letter}", JUMPCUE, tr_mark("Deck {deck}: JumpCUE {letter}"), BUTTON,
                          _deck_code(engine.ACT_JUMP_A + i, n), lamp=i + 1, shown=one, args=pad,
                          tip=tr_mark("JUMP on the next beat (stopped: the playhead there); an empty slot: SET (an "
                                      "edit). Lit while set.")))
        if one:
            out.append(Action(f"deck1.jumpcue_delete.{letter}", JUMPCUE, tr_mark("Deck {deck}: JumpCUE {letter} delete"),
                              BUTTON, gui=(f"cue.delete.{letter}",), lamp=i + 1, args=pad,
                              tip=tr_mark("As SHIFT + a pad (an edit).")))
    for k, beats in enumerate(engine.BEAT_JUMP_SIZES):
        b = {"deck": n, "beats": beats}
        out.append(Action(f"{d}.beatjump_back.{beats}", BEAT_JUMP, tr_mark("Deck {deck}: BEAT JUMP ◁ {beats}"), BUTTON,
                          _deck_code(engine.ACT_BEAT_JUMP + 2 * k, n), shown=one, args=b))
        out.append(Action(f"{d}.beatjump_forward.{beats}", BEAT_JUMP, tr_mark("Deck {deck}: BEAT JUMP ▷ {beats}"),
                          BUTTON, _deck_code(engine.ACT_BEAT_JUMP + 2 * k + 1, n), shown=one, args=b))
    return out


def _mixer_actions(n: int) -> list[Action]:
    """Mixer channel ``n``'s (the settings show channel 1's volume fader, EQ, headphones CUE and level meter)."""
    c, one = f"mixer.ch{n}", n == 1
    a = {"channel": n}
    return [
        Action(f"{c}.volume", MIXER, tr_mark("Channel {channel}: channel fader"), FADER,
               _deck_code(engine.ACT_VOLUME_FADER, n), shown=one, args=a, tip=tr_mark("The deck's volume.")),
        Action(f"{c}.eq_high", MIXER, tr_mark("Channel {channel}: EQ HI"), FADER,
               _deck_code(engine.ACT_EQ_HIGH, n), shown=one, args=a,
               tip=tr_mark("An isolator band: the bottom kills it, the middle 0 dB, the top +6 dB.")),
        Action(f"{c}.eq_mid", MIXER, tr_mark("Channel {channel}: EQ MID"), FADER,
               _deck_code(engine.ACT_EQ_MID, n), shown=one, args=a,
               tip=tr_mark("An isolator band: the bottom kills it, the middle 0 dB, the top +6 dB.")),
        Action(f"{c}.eq_low", MIXER, tr_mark("Channel {channel}: EQ LOW"), FADER,
               _deck_code(engine.ACT_EQ_LOW, n), shown=one, args=a,
               tip=tr_mark("An isolator band: the bottom kills it, the middle 0 dB, the top +6 dB.")),
        Action(f"{c}.trim", MIXER, tr_mark("Channel {channel}: TRIM"), FADER,
               _deck_code(engine.ACT_TRIM, n), shown=False, args=a),
        Action(f"{c}.color", EFFECTS, tr_mark("Channel {channel}: COLOR (Sound Color FX)"), FADER,
               _deck_code(engine.ACT_COLOR_KNOB, n), shown=one, args=a,
               tip=tr_mark("The Sound Color FX picked, on the channel: its middle off, its two flavours left and "
                           "right (a DDJ's CFX knob).")),
        Action(f"{c}.cue", MIXER, tr_mark("Channel {channel}: headphones CUE"), BUTTON,
               _deck_code(engine.ACT_CHANNEL_CUE, n), lamp=engine.LAMP_CHANNEL_CUE, shown=one, args=a,
               tip=tr_mark("A press turns it on or off: the channel's sound before its fader in the headphones. Lit "
                           "while on.")),
        Action(f"{c}.level_meter", MIXER, tr_mark("Channel {channel}: level meter (output)"), METER,
               _deck_code(engine.ACT_LEVEL_METER, n), shown=one, args=a,
               tip=tr_mark("The controller's level meter shows the deck's peak meter (a CC sent: its segments lit, "
                           "-30 to +12 dB, as 0 - 127); taken from a rekordbox mapping (its ChannelLevel), not "
                           "learned.")),
    ]


def _effect_actions() -> list[Action]:
    """The mixer's effects: Sound Color FX (each effect's button, PARAMETER). The engine's Beat FX acts
    (engine.ACT_BEAT_FX_*) have no action: Beat FX is not shown to the user (the engine keeps it, off)."""
    return [
        *(Action(f"fx.color_select.{k}", EFFECTS, tr_mark("Sound Color FX: {effect}"), BUTTON,
                 engine.ACT_COLOR_SELECT + k, lamp=engine.LAMP_COLOR + k, args={"effect": name},
                 tip=tr_mark("A press picks it; pressed again, none. Lit while picked."))
          for k, name in enumerate(engine.COLOR_TYPES)),
        Action("fx.color_param", EFFECTS, tr_mark("Sound Color FX: PARAMETER"), FADER, engine.ACT_COLOR_PARAM,
               tip=tr_mark("The effect's quantity (feedback, centre frequency, noise, crush, resonance, JET's feedback).")),
    ]


ACTIONS: tuple[Action, ...] = (
    *(action for n in range(1, DECKS + 1) for action in _deck_actions(n)),
    Action("browser.rotate", BROWSER, tr_mark("Browser: BROWSE knob"), KNOB,
           gui=("library.next", "library.previous"), tip=tr_mark("Turned: the next / previous track.")),
    Action("browser.press", BROWSER, tr_mark("Browser: BROWSE pressed"), BUTTON, gui=("library.browse_sources",),
           tip=tr_mark("To the sources panel (playlists, folders) and back.")),
    Action("browser.back", BROWSER, tr_mark("Browser: BACK"), BUTTON, gui=("library.leave_search",),
           tip=tr_mark("Back to the tracks.")),
    Action("browser.zoom", BROWSER, tr_mark("Browser: waveform zoom (SHIFT + BROWSE)"), KNOB,
           gui=("playback.zoom_in", "playback.zoom_out"), tip=tr_mark("Turned: the waveform zoomed in / out.")),
    *(action for n in range(1, DECKS + 1) for action in _mixer_actions(n)),
    Action("mixer.headphones_mix", MIXER, tr_mark("Headphones: CUE / MASTER"), FADER, engine.ACT_HEADPHONES_MIX,
           tip=tr_mark("The CUE'd channels against the master in the headphones (with the master mixed in: Settings "
                       "> Audio > Headphones).")),
    Action("mixer.headphones_level", MIXER, tr_mark("Headphones: level"), FADER, engine.ACT_HEADPHONES_LEVEL),
    Action("mixer.master_level", MIXER, tr_mark("Master: level"), FADER, engine.ACT_MASTER_LEVEL),
    *_effect_actions(),
)
ACTIONS_SHOWN = tuple(a for a in ACTIONS if a.shown)
BY_ID = {a.id: a for a in ACTIONS}


def action_label(action: Action) -> str:
    from foundation.i18n import tr
    return tr(action.text).format(**action.args) if action.args else tr(action.text)


def gui_code(action: Action) -> int:
    """The engine's code of an action passed on to the GUI: ACT_GUI + its place in ACTIONS."""
    return engine.ACT_GUI + ACTIONS.index(action)


def gui_action(code: int) -> Action | None:
    i = int(code) - engine.ACT_GUI
    return ACTIONS[i] if 0 <= i < len(ACTIONS) else None


_MESSAGE = re.compile(r"^([89ABE])([0-9A-F])([0-7][0-9A-F])([HIOTS+\-]*)$")
_ENCODINGS = {"": engine.BIND_ENC_AUTO, "O": engine.BIND_ENC_OFFSET, "T": engine.BIND_ENC_TWOS,
              "S": engine.BIND_ENC_SIGN}


@dataclass(frozen=True)
class Message:
    status: int            # a note off is read as its note on (0x9n)
    data1: int             # a pitch bend's: 0 (not read)
    hires: bool = False
    direction: int = 0     # a knob's turns: +1 clockwise only, -1 counterclockwise only, 0 any
    invert: bool = False
    encoding: str = ""     # a knob's: "" auto, "O", "T", "S"

    @classmethod
    def parse(cls, text: str) -> "Message | None":
        m = _MESSAGE.match(str(text or "").strip().upper())
        if m is None:
            return None
        kind, channel, data1, options = m.groups()
        if len(set(c for c in options if c in "+-")) > 1 or sum(c in "OTS" for c in options) > 1:
            return None
        status = (0x90 if kind in "89" else int(kind, 16) << 4) | int(channel, 16)
        cc = status & 0xF0 == 0xB0
        direction = 1 if "+" in options else -1 if "-" in options else 0
        encoding = next((c for c in options if c in "OTS"), "")
        return cls(status, 0 if status & 0xF0 == 0xE0 else int(data1, 16), "H" in options and cc,
                   direction if cc else 0, "I" in options, encoding if cc else "")

    def text(self) -> str:
        options = ("H" if self.hires else "") + ("+" if self.direction > 0 else "-" if self.direction < 0 else "") \
            + ("I" if self.invert else "") + self.encoding
        return f"{self.status:02X}{self.data1:02X}" + options

    def flags(self) -> int:
        return (engine.BIND_HIRES * self.hires
                | (engine.BIND_CLOCKWISE if self.direction > 0 else engine.BIND_COUNTERCLOCKWISE if self.direction < 0
                   else 0)
                | engine.BIND_INVERT * self.invert
                | _ENCODINGS[self.encoding] << engine.BIND_ENCODING_SHIFT)

    def label(self) -> str:
        """As a person reads it: "Ch 1 Note 0x0B", "Ch 1 CC 0x00 (14-bit)", "Ch 7 CC 0x40 ↻", "Ch 1 Pitch bend"."""
        channel = (self.status & 0x0F) + 1
        kind = self.status & 0xF0
        if kind == 0xE0:
            text = f"Ch {channel} Pitch bend"
        else:
            name = {0x90: "Note", 0xA0: "Aftertouch", 0xB0: "CC"}.get(kind, "?")
            text = f"Ch {channel} {name} 0x{self.data1:02X}" + (" (14-bit)" if self.hires else "")
        if self.direction:
            text += " ↻" if self.direction > 0 else " ↺"
        if self.invert:
            text += " ⇅"
        if self.encoding:
            text += {"O": " (64±)", "T": " (2's)", "S": " (±)"}[self.encoding]
        return text


def relative_steps(d2: int) -> int:
    """A relative CC's steps (+ clockwise) told apart by the value (deck.rs relative_steps' auto): two's complement
    near 0, else 64 + steps; 0 and 64: none."""
    v = int(d2)
    if 1 <= v <= 31:
        return v
    if 97 <= v <= 127:
        return v - 128
    if 32 <= v <= 96:
        return v - 64
    return 0


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


def bindings(messages: Mapping) -> list[tuple[int, int, int, int, int]]:
    """The engine's bindings (status, data1, flags, lamp, action code) of ``messages``."""
    out = []
    for action_id, texts in messages.items():
        action = BY_ID.get(action_id)
        if action is None:
            continue
        code = action.code or gui_code(action)
        for text in texts:
            m = Message.parse(text)
            if m is not None:
                out.append((m.status, m.data1, m.flags(), action.lamp, code))
    return out


# rekordbox's MIDI CSV: its function (the first column) -> the action; deck 1's column
_REKORDBOX = {
    "PlayPause": "deck1.play_pause",
    "Cue": "deck1.cue",
    "JumpToTrackStart": "deck1.track_start",
    "TempoSlider": "deck1.tempo_fader",
    "TempoRange": "deck1.tempo_range",
    "JogTouch": "deck1.jog_touch",
    "JogScratch": "deck1.jog_scratch",
    "JogPitchBend": "deck1.jog_bend",
    "WheelPitchBend": "deck1.jog_bend",
    "JogSearch": "deck1.jog_search",
    "Parameter1Left_HotCue": "deck1.memory_previous",
    "Parameter1Right_HotCue": "deck1.memory_next",
    "Memory": "deck1.memory_set",
    "Delete": "deck1.memory_delete",
    "Load": "deck1.load",
    "ChannelFader": "mixer.ch1.volume",
    "EQHigh": "mixer.ch1.eq_high",
    "EQMid": "mixer.ch1.eq_mid",
    "EQLow": "mixer.ch1.eq_low",
    "Gain": "mixer.ch1.trim",
    "HeadphoneCue": "mixer.ch1.cue",
    # BEAT JUMP pads, rekordbox's page 2 (the default): 1, 1, 2, 2, 4, 4, 8, 8 beats, back / forward
    **{f"PAD{2 * k + 1}_BeatJump": f"deck1.beatjump_back.{beats}" for k, beats in enumerate((1, 2, 4, 8))},
    **{f"PAD{2 * k + 2}_BeatJump": f"deck1.beatjump_forward.{beats}" for k, beats in enumerate((1, 2, 4, 8))},
    **{f"PAD{i + 1}_HotCue": f"deck1.jumpcue.{c}" for i, c in enumerate(JUMPCUE_PADS)},
    **{f"PAD{i + 1}_HotCue+Shift": f"deck1.jumpcue_delete.{c}" for i, c in enumerate(JUMPCUE_PADS)},
}
# the browser and the headphones: one of each (the input column)
_REKORDBOX_GLOBAL = {
    "Browse": "browser.rotate",
    "Forward": "browser.press",
    "Back": "browser.back",
    "Zoom": "browser.zoom",
    "HeadphonesMix": "mixer.headphones_mix",
    "HeadphonesVolume": "mixer.headphones_level",
    # its CFX knobs (the channel's COLOR); FX1 (Beat FX) is left out: not shown to the user
    "CFXParameterCH1": "mixer.ch1.color",
    "CFXParameterCH2": "mixer.ch2.color",
}
# outputs: deck 1's output column
_REKORDBOX_OUTPUT = {"ChannelLevel": "mixer.ch1.level_meter"}
_HEX4 = re.compile(r"^[0-9A-Fa-f]{4}$")


def import_rekordbox_csv(path: str) -> tuple[Mapping, list[str]]:
    """A rekordbox MIDI mapping (CSV: function, controller name, type, input, deck 1-4 inputs, output, deck 1-4
    outputs, options, comment): deck 1's controls (its column: a whole message, "900B", or a channel offset to
    the input column's, "0"), mixer channel 1's and the browser's as Mixlyzer's actions. Returns
    (the mapping, the functions not taken)."""
    out: Mapping = {}
    skipped: list[str] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.reader(f):
            if not row or not row[0].strip() or row[0].lstrip().startswith(("@", "#")):
                continue
            row = [c.strip() for c in row] + [""] * (15 - len(row))
            name, kind, base, deck1 = row[0], row[2], row[3], row[4]
            if row[14].startswith("No function assigned"):
                # a control rekordbox leaves to the controller (the DDJ-400's CUE / MASTER and headphones level knobs:
                # analog there; bound here they would act twice)
                skipped.append(name)
                continue
            if row[1] == f"{name}+Shift":
                # the same function on the SHIFT layer (the jog's touch: there SHIFT + jog searches, the record not
                # held), not a function of its own (Delete, Memory: CallLeft+Shift ...)
                skipped.append(row[1])
                continue
            if name in _REKORDBOX_OUTPUT:   # an indicator: deck 1's output (or the output's, offset by it)
                base, deck1 = row[8], row[9]
            if name in _REKORDBOX_GLOBAL:   # one for the controller: the input column
                action, text = _REKORDBOX_GLOBAL[name], base.upper() if _HEX4.match(base) else ""
            else:
                action = _REKORDBOX.get(name) or _REKORDBOX_OUTPUT.get(name)
                if _HEX4.match(deck1):
                    text = deck1.upper()
                elif deck1.isdigit() and _HEX4.match(base):   # a channel offset to the input's message
                    status = (int(base[:2], 16) & 0xF0) | ((int(base[:2], 16) + int(deck1)) & 0x0F)
                    text = f"{status:02X}{base[2:].upper()}"
                else:
                    text = ""   # not on deck 1 (deck 2, the effects ...)
            if action is None or not text:
                skipped.append(name)
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
