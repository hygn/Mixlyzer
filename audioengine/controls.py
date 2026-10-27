"""The deck's and the mixer's controls as both engines take them and report them: the deck's actions (ACT_*: the
GUI's buttons and keys, a MIDI controller's messages) and lamps, a MIDI binding's flags and events, the mixer's
settings (MIX_KEYS) with their modes, effects and knob scales, the tempo ranges, the settings' ranges each engine
takes, and DeckReport, what the deck is as the GUI shows it.

The numbers are the native engine's (src/ipc.rs act::, bind::, mix::; src/deck.rs, src/mixer.rs) and the Python
engine's (audioengine/python/deck.py) alike: the GUI and the app's controls name a control here, never in an
engine's module (audioengine.native.client encodes them for its engine, which keeps its own layout)."""
from __future__ import annotations

import math

# ---- the mixer's settings (ipc.rs mix: MIX_KEYS' place is its key) ----
MIX_KEYS = ("eq_mode", "fader_curve", "color_type", "color_param", "hp_mix", "hp_level", "hp_master", "hp_split",
            "master_level", "booth_level", "booth_eq_hi", "booth_eq_low")
MIX_KEY = {name: k for k, name in enumerate(MIX_KEYS)}
LEVEL_UNITY = 0.8   # a level knob's position at 0 dB
BOOTH_EQ_DB = (-12.0, 6.0)   # the booth EQ's knob: its bottom and top (src/mixer.rs booth_eq_db)
LEVEL_DB_MIN = -40.0   # a level shown in dB: its bottom (the knob's 0 is silence)


def level_knob_to_db(knob: float) -> float:
    """A level knob's gain in dB (src/mixer.rs level_gain: (knob / LEVEL_UNITY)^2); LEVEL_DB_MIN at most down."""
    k = min(1.0, max(0.0, float(knob)))
    return max(LEVEL_DB_MIN, 40.0 * math.log10(k / LEVEL_UNITY)) if k > 0.0 else LEVEL_DB_MIN


def level_db_to_knob(db: float) -> float:
    return min(1.0, LEVEL_UNITY * 10.0 ** (float(db) / 40.0)) if db > LEVEL_DB_MIN else 0.0


def booth_eq_knob_to_db(knob: float) -> float:
    k = min(1.0, max(0.0, float(knob)))
    return BOOTH_EQ_DB[0] * (1.0 - 2.0 * k) if k < 0.5 else BOOTH_EQ_DB[1] * (2.0 * k - 1.0)


def booth_eq_db_to_knob(db: float) -> float:
    d = min(BOOTH_EQ_DB[1], max(BOOTH_EQ_DB[0], float(db)))
    return 0.5 * (1.0 - d / BOOTH_EQ_DB[0]) if d < 0.0 else 0.5 + 0.5 * d / BOOTH_EQ_DB[1]


# Sound Color FX (mix color_type; COLOR_OFF none)
COLOR_TYPES = ("SPACE", "DUB ECHO", "SWEEP", "NOISE", "CRUSH", "FILTER", "JET")
COLOR_OFF = 7
# each effect's place in the channel's strip (src/colorfx.rs ColorType::slot, the A9's block diagram): before the EQ,
# after it (before the CUE and the fader), after the fader
COLOR_SLOTS = ("post_fader", "post_fader", "post_eq", "pre_eq", "post_eq", "post_eq", "post_eq")


# ---- the deck's actions (ipc.rs act): the GUI's buttons and keys and the MIDI controller's ----
ACT_PLAY_PAUSE, ACT_CUE, ACT_STOP, ACT_BEAT_BACK, ACT_BEAT_FORWARD, ACT_BAR_BACK, ACT_BAR_FORWARD = 1, 2, 3, 4, 5, 6, 7
ACT_TEMPO_DOWN, ACT_TEMPO_UP, ACT_TEMPO_RESET, ACT_CALL_PREVIOUS, ACT_CALL_NEXT, ACT_TEMPO_RANGE = 8, 9, 10, 11, 12, 13
ACT_TRACK_START = 14   # to the track's start, playing on (rekordbox's SHIFT + CUE)
ACT_JUMP_A = 16   # .. + 7: H
ACT_BEND_DOWN, ACT_BEND_UP = 24, 25   # held: the tempo bent (pitch bend buttons)
ACT_JOG_MODE = 26   # a press: the jog wheel's VINYL / CDJ mode switched
ACT_BEAT_JUMP = 64   # + 2 k back, + 2 k + 1 forward, by BEAT_JUMP_SIZES[k] beats
BEAT_JUMP_SIZES = (1, 2, 4, 8, 16, 32)
# a MIDI controller's alone: a fader, a jog wheel
ACT_TEMPO_FADER, ACT_VOLUME_FADER, ACT_JOG_TOUCH, ACT_JOG_SCRATCH, ACT_JOG_BEND, ACT_JOG_SEARCH = 32, 33, 34, 35, 36, 37
ACT_LEVEL_METER = 38   # an output: the controller's level meter (a CC)
# the mixer (src/mixer.rs): the channel's trim and EQ bands, its headphones CUE
ACT_TRIM, ACT_EQ_LOW, ACT_EQ_MID, ACT_EQ_HIGH, ACT_CHANNEL_CUE = 39, 40, 41, 42, 44
ACT_HEADPHONES_MIX, ACT_HEADPHONES_LEVEL = 45, 46   # the headphones' CUE / MASTER mix, their level
# a DJM's: the channel's COLOR knob, the master, the booth, Sound Color FX, the headphones' MONO SPLIT
ACT_COLOR_KNOB = 47
ACT_MASTER_LEVEL, ACT_BOOTH_LEVEL, ACT_BOOTH_EQ_HI, ACT_BOOTH_EQ_LOW, ACT_COLOR_PARAM = 51, 52, 53, 54, 55
ACT_COLOR_SELECT = 56   # + k: Sound Color FX k picked (COLOR_TYPES)
ACT_HEADPHONES_SPLIT = 94
# a binding's lamp (src/deck.rs): the controller's light at its message; 1 - 26 the JumpCUE A - Z (lit while set)
LAMP_PLAY, LAMP_CUE = 32, 33
LAMP_CHANNEL_CUE = 34   # the mixer channel's headphones CUE: lit while it is on
LAMP_HP_SPLIT = 40   # the headphones' MONO SPLIT on
LAMP_COLOR = 45   # + k: Sound Color FX k picked
LAMP_VINYL = 52   # the jog in VINYL mode
# the mixer's EQ (mix eq_mode) and channel fader curves (fader_curve), by their settings' names
EQ_MODES = ("isolator", "eq")
FADER_CURVES = ("default", "steep", "slow")
ACT_GUI = 0x1000   # the GUI's own actions from here (core.controls.midi: their numbering), passed on to it
# the tempo fader's ranges (+-%; 100: WIDE, 0 - 200 %)
TEMPO_RANGES = (6, 10, 16, 100)

# ---- a MIDI controller's messages handed to the GUI (ipc.rs MIDI_EVENTS: an event's flags) and its bindings ----
MIDI_EVENT_GUI = 1   # the message is bound to one of the GUI's actions
MIDI_EVENT_RELATIVE = 2   # ... of one turning direction: done once per step (the event's steps)
MIDI_EVENT_HOTCUE_SET = 4   # a JumpCUE pad (its action ACT_JUMP_A + k) pressed on an empty slot: the GUI sets it
# a binding's flags (ipc.rs bind): 14 bits, a knob's one direction, inverted, a knob's encoding (<< 4)
BIND_HIRES, BIND_CLOCKWISE, BIND_COUNTERCLOCKWISE, BIND_INVERT = 1, 2, 4, 8
BIND_ENCODING_SHIFT = 4
BIND_ENC_AUTO, BIND_ENC_OFFSET, BIND_ENC_TWOS, BIND_ENC_SIGN = 0, 1, 2, 3

# ---- what each engine's settings take (Settings > Audio) and reports ----
PEAK_FLOOR_DBFS = -45.0                 # a meter's floor: below its lowest segment (-42 dBFS)
NATIVE_BUFFER_MS_RANGE = (2, 500)       # the native engine fills its buffer every 1 ms below 10 ms (engine.rs)
PYTHON_BUFFER_MS_RANGE = (10, 500)
NATIVE_EXCLUSIVE_RATES = (44100, 48000, 88200, 96000, 176400, 192000)


class DeckReport:
    """What the deck is (src/deck.rs), as the GUI shows it: ``cue`` and ``jump_dest`` (the armed JumpCUE's
    destination) in seconds, NaN for none; ``lamps`` bit 0 PLAY lit, bit 1 CUE lit (blinking: as they are now);
    ``midi_state`` 0 closed, 1 open, 2 the port did not open; its mixer channel: ``knobs`` (TRIM, EQ LOW, MID,
    HIGH: 0 .. 1, the middle 0 dB), ``channel_cue`` (its headphones CUE), ``peak`` (its meter, dBFS), ``color`` (its
    COLOR knob); ``mix``, the mixer's settings (MIX_KEYS' order; empty: none, the Python engine)."""

    __slots__ = ("playing", "factor", "volume", "cue", "jump_dest", "lamps", "tempo_range", "midi_state", "knobs",
                 "channel_cue", "peak", "color", "mix")

    def __init__(self, playing: bool, factor: float, volume: float, cue: float, jump_dest: float, lamps: int,
                 tempo_range: int, midi_state: int, knobs: tuple = (0.5, 0.5, 0.5, 0.5), channel_cue: bool = False,
                 peak: float = -45.0, color: float = 0.5, mix: tuple = ()) -> None:
        self.playing, self.factor, self.volume = bool(playing), float(factor), float(volume)
        self.cue, self.jump_dest, self.lamps = float(cue), float(jump_dest), int(lamps)
        self.tempo_range, self.midi_state = int(tempo_range), int(midi_state)
        self.knobs, self.channel_cue = tuple(float(k) for k in knobs), bool(channel_cue)
        self.peak, self.color, self.mix = float(peak), float(color), tuple(float(v) for v in mix)

    @property
    def vinyl(self) -> bool:
        """Its jog wheel in VINYL mode (lamps bit 2; CDJ: off)."""
        return bool(self.lamps & 4)

    def mix_value(self, key: str, default: float = 0.0) -> float:
        """The mixer's setting ``key`` (MIX_KEYS) as reported."""
        k = MIX_KEY.get(key)
        return self.mix[k] if k is not None and k < len(self.mix) else default

    def key(self) -> tuple:
        """What a change of shows (the peak aside: the meters read it at their own pace)."""
        return (self.playing, self.factor, self.volume, self.cue, self.jump_dest, self.lamps, self.tempo_range,
                self.midi_state, self.knobs, self.channel_cue, self.color, self.mix)
