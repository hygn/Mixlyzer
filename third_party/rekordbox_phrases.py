"""A track's phrases written into rekordbox's own library, as its phrase analysis (PSSI) reads them: the one thing
the Rekordbox XML cannot carry (its beatgrid, cues, key and BPM go by the XML). rekordbox keeps a track's analysis
in files of its own (ANLZ0000.DAT / .EXT in its share folder, the track's row in master.db naming them): the PSSI is
in the .EXT, its phrases by beat number on the beatgrid of the .DAT (PQTZ). So the phrases are put on rekordbox's
own beats (each phrase from the beat nearest its start), whichever grid it holds (one imported from the XML, or its
own analysis), and the .EXT is written again with the PSSI replaced, its other tags as they are. Local files hold
the PSSI as it is (rekordbox masks it only in a USB export's).

master.db (SQLCipher) is only read, through pyrekordbox (it finds rekordbox's settings and opens the database),
to find the track's row by its file's path; nothing is written while rekordbox runs, and the .EXT is copied first
(app folder's backups/rekordbox). Layout after pyrekordbox's and Deep Symmetry's notes: a file is "PMAI", its
header's length (u32), the file's length (u32), then tags (a fourcc, its header's length, its length); PSSI: its
entries' length (u32, 24), their count (u16), the mood (u16: 1 high, 2 mid, 3 low), 6 bytes, the end beat (u16), 2
bytes, the bank (u8), 1 byte, then each entry: its index (u16, from 1), its first beat (u16, from 1), its kind
(u16) and 18 bytes of variants (0).

Mixlyzer's labels are written in a mood (Settings > Library > Rekordbox: libconfig.rekordbox_phrase_mood), each as
one of its phrases (libconfig.rekordbox_phrase_map, as map_text writes it; a label not named there as its mood's
DEFAULT_MAP says): by default the mid mood (pop's song structure), LEAD_IN / INTRO Intro, VERSE Verse 1-6 (BY_RUN: a
run of verses one number, the next run the next), CHORUS Chorus, BRIDGE / BREAK_CHORUS / INTERLUDE Bridge, OUTRO
Outro, SILENCE going on with the phrase before it ("", as the gaps do). rekordbox has no names of its own choosing:
each mood's kinds have fixed names (KIND_NAMES)."""
from __future__ import annotations

import hashlib
import shutil
import struct
import time
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

import numpy as np

from analyzer.music.marks import Phrase
from system.atomic_io import atomic_output_path
from system.i18n import tr
from system.os_support import path_key
from system.resource_paths import app_folder

MOOD_HIGH, MOOD_MID, MOOD_LOW = 1, 2, 3
KIND_NAMES = {
    MOOD_HIGH: {1: "Intro", 2: "Up", 3: "Down", 5: "Chorus", 6: "Outro"},
    MOOD_MID: {1: "Intro", 2: "Verse 1", 3: "Verse 2", 4: "Verse 3", 5: "Verse 4", 6: "Verse 5", 7: "Verse 6",
               8: "Bridge", 9: "Chorus", 10: "Outro"},
    MOOD_LOW: {1: "Intro", 2: "Verse 1", 3: "Verse 1", 4: "Verse 1", 5: "Verse 2", 6: "Verse 2", 7: "Verse 2",
               8: "Bridge", 9: "Chorus", 10: "Outro"},
}
MOODS = {"high": MOOD_HIGH, "mid": MOOD_MID, "low": MOOD_LOW}
BY_RUN = "verses"      # a run of verses one number, the next run the next (past the mood's last verse: the last)
# what a label can be written as, per mood: its kind (BY_RUN: the verses'); "" goes on with the phrase before it
CHOICES = {
    MOOD_HIGH: {"intro": 1, "up": 2, "down": 3, "chorus": 5, "outro": 6},
    MOOD_MID: {"intro": 1, BY_RUN: 0, **{f"verse{n}": 1 + n for n in range(1, 7)}, "bridge": 8, "chorus": 9,
               "outro": 10},
    MOOD_LOW: {"intro": 1, BY_RUN: 0, "verse1": 2, "verse2": 5, "bridge": 8, "chorus": 9, "outro": 10},
}
_RUN_KINDS = {MOOD_MID: (2, 3, 4, 5, 6, 7), MOOD_LOW: (2, 5)}
_POP = {"LEAD_IN": "intro", "INTRO": "intro", "VERSE": BY_RUN, "CHORUS": "chorus", "BREAK_CHORUS": "bridge",
        "BRIDGE": "bridge", "INTERLUDE": "bridge", "OUTRO": "outro", "SILENCE": ""}
DEFAULT_MAP = {
    MOOD_HIGH: {"LEAD_IN": "intro", "INTRO": "intro", "VERSE": "up", "CHORUS": "chorus", "BREAK_CHORUS": "down",
                "BRIDGE": "down", "INTERLUDE": "down", "OUTRO": "outro", "SILENCE": ""},
    MOOD_MID: _POP,
    MOOD_LOW: _POP,
}
LABELS = tuple(_POP)      # the labels mapped (core.marks.phrases' but UNASSIGNED, which is no phrase)
BACKUPS = Path("backups") / "rekordbox"


class RekordboxError(Exception):
    """The phrases cannot be written (its message for the user); nothing was changed."""


@dataclass(frozen=True)
class PhrasePlan:
    """What a write changes: the track's .EXT (``ext``) as read (``data``) and as it is to be (``new``); its phrases
    now and after (each (name, its first beat's time s)) and where they end (``now_end`` / ``after_end``, s); the
    last beat (``length``, s); what rekordbox cannot hold (``losses``); how far a phrase's start is from the beat it
    is put on at most (``off_grid``, s: well past a decoder's offset, rekordbox's beatgrid is not the track's)."""
    ext: Path
    data: bytes
    new: bytes
    now: list[tuple[str, float]]
    after: list[tuple[str, float]]
    now_end: float
    after_end: float
    length: float
    losses: list[str]
    off_grid: float


def choice_name(mood: int, choice: str) -> str:
    """A choice as rekordbox names it ("" for going on with the phrase before)."""
    if choice == BY_RUN:
        kinds = _RUN_KINDS.get(mood, ())
        return f"Verse 1-{len(kinds)}" if kinds else ""
    kind = CHOICES.get(mood, {}).get(choice)
    return KIND_NAMES[mood].get(kind, "") if kind else ""


def phrase_map(mood: int, text: str = "") -> dict[str, str]:
    """Each label's choice in ``mood``: ``text``'s (map_text's "LABEL=choice,...") where it is one of the mood's,
    else the mood's default."""
    out = dict(DEFAULT_MAP.get(mood, _POP))
    for part in (text or "").split(","):
        label, _, choice = part.partition("=")
        label, choice = label.strip().upper(), choice.strip().lower()
        if label in out and (choice == "" or choice in CHOICES.get(mood, {})):
            out[label] = choice
    return out


def map_text(mapping: dict[str, str], mood: int | None = None) -> str:
    """``mapping`` as the setting holds it ("LABEL=choice,..."); with ``mood``, only the labels off its default ("":
    all the default)."""
    default = DEFAULT_MAP.get(mood, {}) if mood is not None else {}
    return ",".join(f"{label}={mapping.get(label, '')}" for label in LABELS
                    if label in mapping and (mood is None or mapping[label] != default.get(label)))


def _tags(data: bytes) -> list[tuple[bytes, int, int]]:
    """(fourcc, offset, length) of an ANLZ file's tags; RekordboxError for a file that is not one."""
    if len(data) < 12 or data[:4] != b"PMAI":
        raise RekordboxError(tr("rekordbox's analysis file is not one it writes."))
    at, out = int.from_bytes(data[4:8], "big"), []
    while at + 12 <= len(data):
        length = int.from_bytes(data[at + 8:at + 12], "big")
        if length < 12 or at + length > len(data):
            raise RekordboxError(tr("rekordbox's analysis file is damaged."))
        out.append((data[at:at + 4], at, length))
        at += length
    return out


def beat_times(dat: bytes) -> np.ndarray:
    """The beats (s) of an ANLZ .DAT's beatgrid (PQTZ: a header of 24, then each beat's number, BPM x100 and ms)."""
    for tag, at, length in _tags(dat):
        if tag == b"PQTZ":
            header = int.from_bytes(dat[at + 4:at + 8], "big")
            count = int.from_bytes(dat[at + 20:at + 24], "big")
            count = min(count, (length - header) // 8)
            body = np.frombuffer(dat, ">u4", count * 2, at + header).reshape(-1, 2)
            return body[:, 1].astype(np.float64) / 1000.0
    return np.zeros(0)


def read_phrases(ext: bytes) -> tuple[int, int, int, list[tuple[int, int]]]:
    """(mood, bank, end beat, [(first beat, kind)]) of an ANLZ .EXT's PSSI ((0, 0, 0, []) without one)."""
    for tag, at, length in _tags(ext):
        if tag == b"PSSI" and length >= 32:
            count = int.from_bytes(ext[at + 16:at + 18], "big")
            mood = int.from_bytes(ext[at + 18:at + 20], "big")
            end = int.from_bytes(ext[at + 26:at + 28], "big")
            entries = [struct.unpack(">HH", ext[at + 34 + 24 * i:at + 38 + 24 * i])
                       for i in range(min(count, (length - 32) // 24))]
            return mood, ext[at + 30], end, entries
    return 0, 0, 0, []


def phrase_entries(phrases: list[Phrase], beats: np.ndarray, mood: int = MOOD_MID,
                   mapping: dict[str, str] | None = None) -> tuple[list[tuple[int, int]], int, list[str], float]:
    """(first beat from 1, kind) of each phrase in ``mood`` as ``mapping`` (phrase_map's) says, on ``beats``
    (rekordbox's); the end beat; what rekordbox cannot hold ("continued:LABEL": a label going on with the phrase
    before; "verses": verse runs past the mood's last verse); how far a written phrase's start is from its beat at
    most (s; a phrase before the first beat left out: it starts on the first beat by its nature)."""
    mapping = phrase_map(mood) if mapping is None else mapping
    runs = _RUN_KINDS.get(mood, ())
    out: list[tuple[int, int]] = []
    losses: list[str] = []
    verse, last_label, end, off = 0, None, 0, 0.0
    if beats.size == 0:
        return out, 0, losses, off
    for p in sorted(phrases, key=lambda p: p.start):
        choice = mapping.get(p.label, "")
        if not choice:                  # UNASSIGNED (no phrase), or a label going on with the phrase before
            if p.label in mapping and f"continued:{p.label}" not in losses:
                losses.append(f"continued:{p.label}")
            continue
        if choice == BY_RUN:
            if last_label != p.label:
                if verse == len(runs) and "verses" not in losses:
                    losses.append("verses")
                verse = min(verse + 1, len(runs))
            kind = runs[verse - 1] if runs else 1
        else:
            kind = CHOICES[mood][choice]
        beat = int(np.argmin(np.abs(beats - p.start))) + 1
        if out and beat <= out[-1][0]:          # two phrases on one beat: the first kept
            continue
        out.append((beat, kind))
        if p.start >= beats[0]:         # one before the first beat (an intro from the track's start) is put on it
            off = max(off, abs(float(beats[beat - 1]) - p.start))
        last_label = p.label
        end = max(end, int(np.argmin(np.abs(beats - p.end))) + 1)
    return out, max(end, out[-1][0] if out else 0), losses, off


def pssi_tag(entries: list[tuple[int, int]], end_beat: int, mood: int = MOOD_MID, bank: int = 0) -> bytes:
    """A PSSI tag of ``entries`` ((first beat, kind)), unmasked (as rekordbox's own files hold it)."""
    body = b"".join(struct.pack(">HHH", i, beat, kind) + bytes(18) for i, (beat, kind) in enumerate(entries, 1))
    head = struct.pack(">IHH6sH2sB1s", 24, len(entries), mood, bytes(6), end_beat, bytes(2), bank, bytes(1))
    return b"PSSI" + struct.pack(">II", 32, 32 + len(body)) + head + body


def with_phrases(ext: bytes, tag: bytes) -> bytes:
    """The .EXT with its PSSI replaced by ``tag`` (added at the end without one), its file length set."""
    tags = _tags(ext)
    old = next(((at, length) for t, at, length in tags if t == b"PSSI"), None)
    if old is None:
        out = bytearray(ext + tag)
    else:
        out = bytearray(ext[:old[0]] + tag + ext[old[0] + old[1]:])
    struct.pack_into(">I", out, 8, len(out))
    return bytes(out)


def _named(entries: list[tuple[int, int]], mood: int, beats: np.ndarray) -> list[tuple[str, float]]:
    names = KIND_NAMES.get(mood, KIND_NAMES[MOOD_MID])
    return [(names.get(kind, str(kind)), float(beats[beat - 1]) if 0 < beat <= beats.size else 0.0)
            for beat, kind in entries]


def running() -> bool:
    from pyrekordbox.utils import get_rekordbox_pid
    return bool(get_rekordbox_pid())


def _analysis_files(audio_path: str) -> tuple[Path, Path]:
    """The track's .DAT and .EXT in rekordbox's library (master.db only read); RekordboxError when it has none."""
    try:
        from pyrekordbox import MasterDatabase
        db = MasterDatabase()
    except Exception as exc:        # not installed, no library, the database not opened
        raise RekordboxError(tr("rekordbox's library could not be opened: {error}").format(error=exc)) from exc
    try:
        key = path_key(audio_path)
        content = next((c for c in db.get_content()
                        if not c.rb_local_deleted and c.FolderPath and path_key(c.FolderPath) == key), None)
        if content is None:
            raise RekordboxError(tr("The track is not in rekordbox's collection."))
        if not content.AnalysisDataPath:
            raise RekordboxError(tr("rekordbox has not analyzed the track."))
        folder = Path(db.get_anlz_dir(content))
    finally:
        db.close()
    dat, ext = folder / "ANLZ0000.DAT", folder / "ANLZ0000.EXT"
    if not dat.is_file() or not ext.is_file():
        raise RekordboxError(tr("rekordbox's analysis files of the track are missing; analyze it in rekordbox."))
    return dat, ext


def plan(audio_path: str, phrases: list[Phrase], mood: int = MOOD_MID,
         mapping: dict[str, str] | None = None) -> PhrasePlan:
    """What writing ``phrases`` (in ``mood``, as ``mapping`` says) into the track's rekordbox analysis changes (read
    only); RekordboxError when it cannot be written."""
    if running():
        raise RekordboxError(tr("rekordbox is running: close it, then write again."))
    dat_path, ext_path = _analysis_files(audio_path)
    beats = beat_times(dat_path.read_bytes())
    if beats.size == 0:
        raise RekordboxError(tr("rekordbox's analysis of the track has no beatgrid."))
    ext = ext_path.read_bytes()
    now_mood, bank, now_end, now = read_phrases(ext)
    mood = mood if mood in KIND_NAMES else MOOD_MID
    entries, end_beat, losses, off = phrase_entries(phrases, beats, mood, mapping)
    if not entries:
        raise RekordboxError(tr("The track has no phrases to write."))
    new = with_phrases(ext, pssi_tag(entries, end_beat, mood, bank))
    at = lambda beat: float(beats[min(max(beat, 1), beats.size) - 1])   # noqa: E731
    return PhrasePlan(ext_path, ext, new, _named(now, now_mood, beats), _named(entries, mood, beats),
                      at(now_end) if now else 0.0, at(end_beat), float(beats[-1]), losses, off)


def backup_folder() -> Path:
    return app_folder() / BACKUPS


BACKUP_KINDS = ("", "ext", "folder")     # none, the .EXT, the track's analysis folder (its .DAT / .EXT / .2EX / .3EX)


def analysis_folder(audio_path: str) -> Path:
    """The track's analysis folder in rekordbox's library (master.db only read); RekordboxError when it has none."""
    return _analysis_files(audio_path)[0].parent


def backups_of(folder: Path) -> list[Path]:
    """The backups made of an analysis folder's files (write's: a "<folder> <time> ANLZ0000.EXT" file or a
    "<folder> <time>" folder in a day's folder), newest first."""
    root = backup_folder()
    if not root.is_dir():
        return []
    found = [entry for day in root.iterdir() if day.is_dir() and not day.is_symlink()
             for entry in day.iterdir() if entry.name.startswith(folder.name + " ") and not entry.is_symlink()]
    return sorted(found, key=lambda entry: (entry.parent.name, entry.name), reverse=True)


def _anlz_files(backup: Path) -> list[Path]:
    """The analysis files a backup holds (an ANLZ file: "PMAI" first)."""
    files = sorted(backup.iterdir()) if backup.is_dir() else [backup]
    out = []
    for path in files:
        name = path.name.rsplit(" ", 1)[-1]
        if path.is_file() and not path.is_symlink() and name.startswith("ANLZ0000."):
            with open(path, "rb") as f:
                if f.read(4) == b"PMAI":
                    out.append(path)
    return out


def restore(backup: Path, folder: Path) -> list[str]:
    """The analysis folder's files put back as ``backup`` (backups_of's) holds them (the names put back returned);
    RekordboxError while rekordbox runs or for a backup that holds no analysis file (nothing changed then)."""
    if running():
        raise RekordboxError(tr("rekordbox is running: close it, then restore again."))
    files = _anlz_files(backup)
    if not files:
        raise RekordboxError(tr("The backup holds no rekordbox analysis file."))
    put = []
    for path in files:
        name = path.name.rsplit(" ", 1)[-1]
        with atomic_output_path(folder / name) as temporary:
            shutil.copyfile(path, temporary)
        put.append(name)
    return put


def write(p: PhrasePlan, backup: str = "ext") -> Optional[Path]:
    """The plan written, the backup asked for made first (``backup``: BACKUP_KINDS'; its path returned, None for
    none); RekordboxError when rekordbox runs or the file changed since it was read (nothing written then). A write
    that fails leaves no backup behind."""
    if running():
        raise RekordboxError(tr("rekordbox is running: close it, then write again."))
    if hashlib.sha256(p.ext.read_bytes()).digest() != hashlib.sha256(p.data).digest():
        raise RekordboxError(tr("rekordbox's analysis file changed since it was read; write again."))
    made: Optional[Path] = None
    if backup in ("ext", "folder"):
        day = backup_folder() / time.strftime("%Y%m%d")
        day.mkdir(parents=True, exist_ok=True)
        stamp = f"{p.ext.parent.name} {time.strftime('%H%M%S')}"
        if backup == "ext":
            made = day / f"{stamp} {p.ext.name}"
            shutil.copy2(p.ext, made)
        else:
            made = day / stamp
            shutil.copytree(p.ext.parent, made)
    try:
        with atomic_output_path(p.ext) as temporary:
            temporary.write_bytes(p.new)
    except OSError:
        if made is not None:
            shutil.rmtree(made, ignore_errors=True) if made.is_dir() else made.unlink(missing_ok=True)
        raise
    return made
