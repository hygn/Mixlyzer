"""The native engine's control block as both sides spell it: audioengine/native/src/ipc.rs (the engine) against
audioengine/native/client.py and audioengine/controls.py (the GUI). The layout is written twice by hand ("keep them
together"); these read the Rust source itself, so a field moved on one side only fails here, without building the
engine. Also the command ring: a record the GUI writes, read back as ipc.rs' next_command reads it."""
import re
import struct
import threading
import unittest
import zlib
from pathlib import Path

from audioengine import controls
from audioengine.native import client

IPC_RS = Path(__file__).resolve().parents[3] / "audioengine" / "native" / "src" / "ipc.rs"


def _strip_comments(text: str) -> str:
    return re.sub(r"//[^\n]*", "", text)


def _blocks(text: str) -> dict[str, str]:
    """The top level ("") and each ``pub mod name { ... }``'s body."""
    out, top, at = {}, [], 0
    for m in re.finditer(r"pub mod (\w+)\s*\{", text):
        if m.start() < at:
            continue
        depth, k = 1, m.end()
        while depth:
            depth += {"{": 1, "}": -1}.get(text[k], 0)
            k += 1
        top.append(text[at:m.start()])
        out[m.group(1)] = text[m.end():k - 1]
        at = k
    top.append(text[at:])
    out[""] = "".join(top)
    return out


def _consts(body: str) -> dict[str, tuple[str, str]]:
    return {m.group(1): (m.group(2).strip(), m.group(3).strip())
            for m in re.finditer(r"pub const (\w+)\s*:\s*([^=]+?)\s*=\s*(.+?);", body, re.S)}


def _evaluate(blocks: dict[str, str]) -> dict[str, dict[str, object]]:
    """Every const's value (integers, floats, arrays of them or of strings), the names resolved within the module
    and through ``super::``."""
    raw = {mod: _consts(body) for mod, body in blocks.items()}
    values: dict[str, dict[str, object]] = {mod: {} for mod in raw}
    special = {'u64::from_le_bytes(*b"MXAUDIC1")': int.from_bytes(b"MXAUDIC1", "little")}
    for _ in range(10):
        progress = False
        for mod, consts in raw.items():
            for name, (_type, expr) in consts.items():
                if name in values[mod]:
                    continue
                if expr in special:
                    values[mod][name] = special[expr]
                    progress = True
                    continue
                code = re.sub(r"super::(\w+)", r"__top__\1", expr).replace("::", ".")
                scope = {**values[mod], **{f"__top__{k}": v for k, v in values[""].items()}}
                try:
                    values[mod][name] = eval(code, {"__builtins__": {}}, scope)   # noqa: S307 (the repo's own source)
                    progress = True
                except NameError:
                    pass
        if not progress:
            break
    missing = [f"{mod}::{name}" for mod, consts in raw.items() for name in consts if name not in values[mod]]
    assert not missing, f"not evaluated: {missing}"
    return values


def _types(blocks):
    return {mod: {name: t for name, (t, _e) in _consts(body).items()} for mod, body in blocks.items()}


def _field_offsets(fmt: str) -> list[int]:
    """The byte offset of each field of a struct format ("<" assumed: no alignment), pad bytes left out."""
    fmt = fmt.replace(" ", "").lstrip("<")
    offsets, at = [], 0
    for count, code in re.findall(r"(\d*)([a-zA-Z?])", fmt):
        n = int(count or 1)
        size = struct.calcsize("<" + code)
        if code != "x":
            offsets.extend(at + size * k for k in range(n))
        at += size * n
    return offsets


_CODES = {"B": "u8", "H": "u16", "I": "u32", "Q": "u64", "i": "i32", "q": "i64", "f": "f32", "d": "f64"}


def _field_types(fmt: str) -> dict[int, str]:
    """Each field's offset -> its Rust type, by its struct code."""
    codes = [code for count, code in re.findall(r"(\d*)([a-zA-Z?])", fmt.replace(" ", "").lstrip("<"))
             for _ in range(int(count or 1)) if code != "x"]
    return dict(zip(_field_offsets(fmt), (_CODES[c] for c in codes), strict=True))


def _comment_types(raw: str) -> dict[str, dict[str, str]]:
    """The field types ipc.rs states in a field's comment (``pub const PEAK: usize = 72; // f64 dBFS ...``), by
    module (a ``pub mod`` opens one, a ``}`` at the line's start closes it)."""
    out: dict[str, dict[str, str]] = {"": {}}
    mod = ""
    for line in raw.splitlines():
        m = re.match(r"pub mod (\w+)", line)
        if m:
            mod = m.group(1)
            out.setdefault(mod, {})
        elif line.startswith("}"):
            mod = ""
        m = re.search(r"pub const (\w+): usize = [^;]+; *// ?(u8|u16|u32|u64|i32|i64|f32|f64)\b", line)
        if m:
            out[mod][m.group(1)] = m.group(2)
    return out


RAW = IPC_RS.read_text(encoding="utf-8")
TEXT = _strip_comments(RAW)
BLOCKS = _blocks(TEXT)
RS = _evaluate(BLOCKS)
TYPES = _types(BLOCKS)
FIELD_TYPES = _comment_types(RAW)


class TopLevelTests(unittest.TestCase):
    def test_offsets_and_versions(self):
        for name in ("MAGIC", "VERSION", "STATE_OFFSET", "RING_OFFSET", "RING_HEADER", "HIST_MAX", "MAX_DECKS",
                     "DEVICES_OFFSET", "OUTPUTS_OFFSET", "MIX_OFFSET", "MIDI_PORTS_OFFSET", "MIDI_EVENTS_OFFSET",
                     "MIDI_EVENTS"):
            with self.subTest(name=name):
                self.assertEqual(getattr(client, name), RS[""][name])
        for name in ("MIDI_EVENT_GUI", "MIDI_EVENT_RELATIVE", "MIDI_EVENT_HOTCUE_SET"):
            self.assertEqual(getattr(controls, name), RS[""][name])

    def test_regions_do_not_overlap(self):
        top, st, mix = RS[""], RS["st"], RS["mix"]
        regions = [
            ("state", top["STATE_OFFSET"], top["STATE_OFFSET"] + st["MIXER_STATE"] + client._MIXER.size),
            ("midi ports", top["MIDI_PORTS_OFFSET"], top["MIDI_PORTS_OFFSET"] + 16 + top["MIDI_PORTS_MAX"]),
            ("midi events", top["MIDI_EVENTS_OFFSET"],
             top["MIDI_EVENTS_OFFSET"] + 16 + top["MIDI_EVENTS"] * client._MIDI_EVENT.size),
            ("devices", top["DEVICES_OFFSET"], top["DEVICES_OFFSET"] + 16 + top["DEVICES_MAX"]),
            ("outputs", top["OUTPUTS_OFFSET"], top["OUTPUTS_OFFSET"] + len(client.AUX_BUSES) * RS["out"]["SIZE"]),
            ("mixer", top["MIX_OFFSET"], top["MIX_OFFSET"] + 8 * (mix["METERS"] + mix["METER_COUNT"])),
            ("ring", top["RING_OFFSET"], client.CONTROL_SIZE),
        ]
        regions.sort(key=lambda r: r[1])
        for (a, _a0, a1), (b, b0, _b1) in zip(regions, regions[1:]):
            self.assertLessEqual(a1, b0, f"{a} runs into {b}")
        self.assertEqual(client._MIDI_EVENT.size, 16)
        self.assertGreaterEqual(mix["METERS"], mix["KEYS"])                     # the meters after the keys


class StateTests(unittest.TestCase):
    def test_state_fields(self):
        st, types = RS["st"], TYPES["st"]
        rust = sorted(v for k, v in st.items() if types[k] == "usize" and v < st["HIST"]
                      and not k.startswith("ROW_") and k != "DECK_STATE_SIZE")
        self.assertEqual(rust, _field_offsets(client._STATE.format))
        self.assertEqual(client._HIST, st["HIST"])
        self._same_types("st", client._STATE.format, 0, lambda k: st[k] < st["HIST"] and not k.startswith("ROW_"))

    def _same_types(self, mod, fmt, base, picked):
        """Each field whose comment states its type is read as that type."""
        python = _field_types(fmt)
        checked = 0
        for name, rust_type in FIELD_TYPES[mod].items():
            if not picked(name):
                continue
            with self.subTest(field=f"{mod}::{name}"):
                self.assertEqual(python.get(RS[mod][name] - base), rust_type)
                checked += 1
        self.assertGreater(checked, 5)

    def test_field_types(self):
        st = RS["st"]
        self._same_types("st", client._DECK.format, st["DECK"],
                         lambda k: st["DECK"] < st[k] < st["DECKS_STATE"] or k == "FACTOR")
        self._same_types("st", client._DECK_ROW.format, 0, lambda k: k.startswith("ROW_"))
        self._same_types("out", client._OUT.format, 0, lambda k: RS["out"][k] <= RS["out"]["NAME"])

    def test_deck_fields(self):
        st = RS["st"]
        self.assertEqual(client._DECK_AT, st["DECK"])
        rust = sorted(v - st["DECK"] for k, v in st.items() if st["DECK"] <= v < st["DECKS_STATE"] and k != "DECK")
        self.assertEqual(rust, _field_offsets(client._DECK.format))
        self.assertEqual(client._DECKS_AT, st["DECKS_STATE"])
        self.assertEqual(client._DECK_ROW.size, st["DECK_STATE_SIZE"])
        self.assertEqual(client._MIXER_AT, st["MIXER_STATE"])
        self.assertEqual(client.MAX_DECKS, RS[""]["MAX_DECKS"])

    def test_deck_rows_by_name(self):
        st = RS["st"]
        python = dict(zip(client._DECK_ROW_KEYS, _field_offsets(client._DECK_ROW.format), strict=True))
        names = {"KNOBS": "trim"}
        rows = {k[4:]: v for k, v in st.items() if k.startswith("ROW_")}
        self.assertEqual(len(rows), len(python) - 3)                             # the knobs: four fields, one row
        for name, offset in rows.items():
            with self.subTest(row=name):
                self.assertEqual(python[names.get(name, name.lower())], offset)

    def test_output_fields(self):
        out, types = RS["out"], TYPES["out"]
        rust = sorted(v for k, v in out.items() if types[k] == "usize" and v <= out["NAME"])   # NAME: its u32 length
        self.assertEqual(rust, _field_offsets(client._OUT.format))
        self.assertEqual((client._OUT_NAME, client._OUT_RATE, client._OUT_HIST, client._OUT_SIZE),
                         (out["NAME"], out["RATE"], out["HIST"], out["SIZE"]))
        self.assertLessEqual(out["NAME"] + 4 + out["NAME_MAX"], out["RATE"])                # the length, the text
        states = {v: k for k, v in out.items() if k.startswith("STATE_") and types[k] == "u32"}
        self.assertEqual(set(states), set(client.AUX_STATES))


class MixerTests(unittest.TestCase):
    def test_keys(self):
        mix, types = RS["mix"], TYPES["mix"]
        self.assertEqual(list(mix["NAMES"]), list(controls.MIX_KEYS))
        self.assertEqual(mix["KEYS"], len(controls.MIX_KEYS))
        self.assertEqual(len(mix["DEFAULTS"]), mix["KEYS"])
        self.assertEqual(mix["LEVEL_UNITY"], controls.LEVEL_UNITY)
        index = {k: v for k, v in mix.items() if types[k] == "usize" and k not in ("KEYS", "METERS")
                 and not k.startswith("METER_")}
        self.assertEqual(len(index), mix["KEYS"])
        for name, k in index.items():
            self.assertEqual(controls.MIX_KEYS[k], name.lower(), name)
        self.assertEqual(client.MIX_METERS_AT, mix["METERS"])
        self.assertEqual(len(client.MIX_METERS), mix["METER_COUNT"])

    def test_defaults_are_neutral_where_the_gui_says(self):
        defaults = dict(zip(controls.MIX_KEYS, RS["mix"]["DEFAULTS"]))
        self.assertIn(int(defaults["color_type"]), range(len(controls.COLOR_TYPES) + 1))
        self.assertEqual(defaults["master_level"], controls.LEVEL_UNITY)
        self.assertEqual(defaults["hp_a_level"], controls.LEVEL_UNITY)
        self.assertEqual(defaults["booth_level"], controls.LEVEL_UNITY)
        self.assertEqual((defaults["booth_eq_hi"], defaults["booth_eq_low"]), (0.5, 0.5))   # the booth EQ flat


class CodeTests(unittest.TestCase):
    def test_ops(self):
        rust = RS["op"]
        python = {k[3:]: v for k, v in vars(client).items() if k.startswith("OP_")}
        self.assertEqual(python, rust)
        self.assertEqual(len(set(rust.values())), len(rust))
        self.assertTrue(client.DECK_OPS <= set(rust.values()))

    def test_acts(self):
        rust = dict(RS["act"])
        sizes = rust.pop("BEAT_JUMP_SIZES")
        self.assertEqual(list(sizes), list(controls.BEAT_JUMP_SIZES))
        self.assertEqual(rust.pop("CODE_MASK"), (1 << controls.ACT_DECK_SHIFT) - 1)
        python = {k[4:]: v for k, v in vars(controls).items() if k.startswith("ACT_") and isinstance(v, int)}
        self.assertEqual(python, rust)

    def test_binding_flags(self):
        bind = RS["bind"]
        for name in ("HIRES", "CLOCKWISE", "COUNTERCLOCKWISE", "INVERT", "ENCODING_SHIFT"):
            self.assertEqual(getattr(controls, f"BIND_{name}"), bind[name], name)
        for name in ("AUTO", "OFFSET", "TWOS", "SIGN"):
            self.assertEqual(getattr(controls, f"BIND_ENC_{name}"), bind[f"ENC_{name}"], name)
        self.assertEqual(bind["ENCODING"], 3 << bind["ENCODING_SHIFT"])
        flags = [bind[n] for n in ("HIRES", "CLOCKWISE", "COUNTERCLOCKWISE", "INVERT")]
        self.assertEqual(sum(flags), 0x0F)                                       # one bit each, under the encoding
        self.assertEqual(bind["ENCODING"] & 0x0F, 0)


def _reader_fields() -> dict[str, tuple[int, str]]:
    """The record fields next_command reads: name -> (offset, Rust type)."""
    found = {m.group(1): (int(m.group(2) or 0), m.group(3)) for m in re.finditer(
        r"let (\w+) = unsafe \{ std::ptr::read_volatile\(rec(?:\.add\((\d+)\))? as \*const (\w+)\) \}", TEXT)}
    assert {"size", "op", "seq", "sum", "length", "sent_at"} <= set(found), found
    return found


def _rust_checksum(op, seq, sent_at, payload) -> int:
    """ipc.rs checksum: CRC-32 of op (u32), seq (u64), sent_at (f64), little-endian, then the payload."""
    return zlib.crc32(struct.pack("<I", op) + struct.pack("<Q", seq) + struct.pack("<d", sent_at) + payload)


class _Engine(client.NativeEngine):
    """A NativeEngine on a bare control block: no process, no shared memory."""

    def __init__(self, write: int = 0) -> None:   # noqa: D107 (the base class starts a process)
        self._buf = bytearray(client.CONTROL_SIZE)
        self._lock = threading.Lock()
        self._seq = 0
        struct.pack_into("<QQ", self._buf, client.RING_OFFSET, write, write)

    def alive(self) -> bool:
        return True

    def records(self):
        """The ring's records as ipc.rs next_command takes them (PAD records skipped)."""
        fields = _reader_fields()
        sizes = {"u32": "<I", "u64": "<Q", "f64": "<d"}
        cap = client.RING_BYTES
        base = client.RING_OFFSET + client.RING_HEADER
        write, read = struct.unpack_from("<QQ", self._buf, client.RING_OFFSET)
        out = []
        while read < write:
            at = read % cap
            def get(name):
                off, kind = fields[name]
                return struct.unpack_from(sizes[kind], self._buf, base + at + off)[0]
            size = get("size")
            assert size >= 8 and size % 8 == 0 and at + size <= cap and read + size <= write, (size, at)
            if get("op") == RS["op"]["PAD"]:                                    # only its size and op are read
                read += size
                continue
            got = {name: get(name) for name in fields}
            assert size >= 32 and got["length"] <= size - 32
            payload = bytes(self._buf[base + at + 32:base + at + 32 + got["length"]])
            assert _rust_checksum(got["op"], got["seq"], got["sent_at"], payload) == got["sum"]
            out.append((got["op"], got["seq"], payload))
            read += size
        return out


class CommandRingTests(unittest.TestCase):
    def test_records_read_back(self):
        engine = _Engine()
        payloads = [b"", b"\x01", bytes(range(7)), bytes(range(8)), bytes(range(256)) * 3]
        for k, payload in enumerate(payloads):
            self.assertTrue(engine.send(client.OP_SEEK + k, payload))
        self.assertEqual(engine.records(), [(client.OP_SEEK + k, k + 1, p) for k, p in enumerate(payloads)])

    def test_a_record_never_wraps(self):
        """Near the ring's end the rest is a PAD record and the command starts at the ring's start."""
        for room in (8, 32, 40, 64):
            with self.subTest(room=room):
                engine = _Engine(write=client.RING_BYTES - room)
                payload = b"x" * 33                                              # a record of 72 bytes
                self.assertTrue(engine.send(client.OP_LOAD, payload))
                self.assertEqual(engine.records(), [(client.OP_LOAD, 1, payload)])

    def test_record_exactly_filling_the_rest(self):
        engine = _Engine(write=client.RING_BYTES - 40)
        self.assertTrue(engine.send(client.OP_PLAY, b"12345678"))               # 32 + 8 = 40
        write, _read = struct.unpack_from("<QQ", engine._buf, client.RING_OFFSET)
        self.assertEqual(write % client.RING_BYTES, 0)                          # no PAD needed
        self.assertEqual(engine.records(), [(client.OP_PLAY, 1, b"12345678")])


if __name__ == "__main__":
    unittest.main()
