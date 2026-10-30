"""A track's tags written into its audio file: title, artist, album, comment, BPM and key, each only when asked
for ("" removes the tag). Everything else stays as it is: the audio bytes, the artwork and every other tag
(other programs' data, such as Serato's, included). Own code: the tag libraries for Python (mutagen and the
like) are GPL. The format is told by the file's content, not its name.

A field can be spelled in a few places of a format, and the analysis's reader (``extract_tags``: ``read_fields``
here, the WAV INFO reader) takes it from the first of them holding a value (``LEVELS``, MediaInfo's order; the
places of a level joined: an M4A title in the iTunes list's (c)nam and the 3GPP titl). Most are the field's copies in the format's tag
systems (WAV INFO and its ID3 chunk, MP3's COMM and FFmpeg's TXXX "comment"); some are another field it falls
back to for an empty one (``_FALLBACKS``: the album artist for the artist, a description for the comment).
A value is written where the analysis read it from (its sources, ``tag_sources``, kept in the library: the place
it took and the copies holding the same value), else (no source, or one the file's format lacks, as after the
file was replaced) to the field's copies the file has (none: where the reader takes it first; WAV: INFO and the
ID3 chunk). An emptied field clears its sources and its copies holding a value; a fallback field only when it was
the source. The other places are left as they are.

- FLAC: the Vorbis comment, resized in the padding;
- MP3 / AAC (ADTS): the ID3v2 tag at the start, each of them when there are a few in a row (v2.3 / v2.4 kept, a
  new one v2.3); ID3v2.2 is not written. ID3v1 (30 Latin-1 characters a field) is read, not written: a value read
  from it goes to ID3v2, which readers take first;
- WAV: RIFF INFO (UTF-8, as the analysis reads it) and an ID3 chunk;
- AIFF: an ID3 chunk ("ID3 "); its own NAME / AUTH / ANNO chunks are ASCII, read in the system's code page, so a
  value read from one goes to the ID3 chunk and the chunk is left out. AIFC is not written (MediaInfo, the
  analysis's reader, does not read its ID3);
- MP4 / M4A: the iTunes list (moov/udta/meta/ilst; tmpo a whole BPM, the key the freeform
  com.apple.iTunes:initialkey; a QuickTime-style meta becomes an ISO one) and the 3GPP asset texts in udta;
  when moov is before the audio and grows, the chunk offsets follow. Fragmented files and files with QuickTime
  keys metadata (an mdta meta: FFmpeg's use_metadata_tags, cameras) are not written.

A file is changed in place when the tags fit their room (the padding, a free atom after moov), else written
anew beside it and moved over it. Either way it is read back first: the spellings written hold the values and
every other one is as it was, else the file is left as it was."""

from __future__ import annotations

import math
import os
import shutil
import struct
import tempfile
import zlib
from pathlib import Path
from typing import Callable, Optional, Union

from system.i18n import tr

FIELDS = ("title", "artist", "album", "comment", "bpm", "key")
TEXT_FIELDS = ("title", "artist", "album", "comment")   # the ones the analysis reads from the tags
PADDING = 2048          # room left for later edits when a file is written anew
_COPY_BLOCK = 1 << 20

# where each format keeps each field, as the analysis's reader takes it: the first level holding a value (a
# level's places joined); FLAC's artist: see _levels
LEVELS: dict[str, dict[str, tuple[tuple[str, ...], ...]]] = {
    "flac": {"title": (("vorbis:TITLE",),), "artist": (("vorbis:ARTIST",), ("vorbis:ALBUMARTIST",)),
             "album": (("vorbis:ALBUM",),), "comment": (("vorbis:COMMENT",), ("vorbis:DESCRIPTION",)),
             "bpm": (("vorbis:BPM",),), "key": (("vorbis:INITIALKEY",),)},
    "id3": {"title": (("id3:TIT2",), ("id3v1:title",)), "artist": (("id3:TPE1",), ("id3v1:artist",), ("id3:TPE2",)),
            "album": (("id3:TALB",), ("id3v1:album",)),
            "comment": (("id3:COMM",), ("id3v1:comment",), ("id3:TXXX:comment",)),
            "bpm": (("id3:TBPM",),), "key": (("id3:TKEY",),)},
    "wav": {"title": (("info:INAM",), ("id3:TIT2",)), "artist": (("info:IART",), ("id3:TPE1",), ("id3:TPE2",)),
            "album": (("info:IPRD",), ("id3:TALB",)),
            "comment": (("info:ICMT",), ("id3:COMM",), ("id3:TXXX:comment",)),
            "bpm": (("id3:TBPM",),), "key": (("id3:TKEY",),)},
    "aiff": {"title": (("aiff:NAME",), ("id3:TIT2",)), "artist": (("aiff:AUTH",), ("id3:TPE1",), ("id3:TPE2",)),
             "album": (("id3:TALB",),), "comment": (("aiff:ANNO",), ("id3:COMM",), ("id3:TXXX:comment",)),
             "bpm": (("id3:TBPM",),), "key": (("id3:TKEY",),)},
    "mp4": {"title": (("mp4:\u00a9nam", "mp4:titl"),), "artist": (("mp4:\u00a9ART", "mp4:perf"), ("mp4:aART",)),
            "album": (("mp4:\u00a9alb", "mp4:albm"),), "comment": (("mp4:\u00a9cmt",), ("mp4:desc", "mp4:dscp")),
            "bpm": (("mp4:tmpo",),), "key": (("mp4:initialkey",),)},
}
# another field the reader takes for an empty one: written (and emptied) only when it was the value's source
_FALLBACKS = {"vorbis:ALBUMARTIST", "vorbis:DESCRIPTION", "id3:TPE2", "mp4:aART", "mp4:desc", "mp4:dscp"}
# MediaInfo takes FLAC's PERFORMER for the performer when the file has one (even blank: its ARTIST then is the
# composer, its ALBUMARTIST the album composer)
_FLAC_PERFORMER_LEVELS = (("vorbis:PERFORMER",), ("vorbis:ALBUMARTIST",))
SPELLINGS: dict[str, dict[str, tuple[str, ...]]] = {
    fmt: {field: tuple(dict.fromkeys(s for level in levels for s in level)) for field, levels in fields.items()}
    for fmt, fields in LEVELS.items()}
SPELLINGS["flac"]["artist"] += ("vorbis:PERFORMER",)
# places that cannot hold any value (AIFF's ASCII text chunks, ID3v1's 30 Latin-1 characters): a value read
# from one is written to the ID3v2 field instead (AIFF's chunk then left out; ID3v1 left as it is)
_REDIRECT = {"aiff:NAME": "id3:TIT2", "aiff:AUTH": "id3:TPE1", "aiff:ANNO": "id3:COMM",
             "id3v1:title": "id3:TIT2", "id3v1:artist": "id3:TPE1", "id3v1:album": "id3:TALB",
             "id3v1:comment": "id3:COMM"}

Segment = Union[bytes, tuple[int, int]]   # new bytes, or (offset, length) of the file as it is
Spellings = dict[str, list[str]]          # spelling -> its values in the file (several: repeated, a few tags)


class TagWriteError(Exception):
    """The tags could not be written; the file is as it was."""


def _damaged() -> TagWriteError:
    return TagWriteError(tr("The file's tags are damaged or of a kind not written here; the file is unchanged."))


def _not_read_back() -> TagWriteError:
    return TagWriteError(tr("The written tags did not read back as written; the file is unchanged."))


def bpm_text(bpm) -> str:
    """The BPM as a text tag holds it: up to two decimals, no trailing zeros ("" for none)."""
    try:
        value = float(bpm)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(value) or value <= 0:
        return ""
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _texts(values: dict) -> dict[str, str]:
    out = {}
    for field, value in values.items():
        if field not in FIELDS:
            raise ValueError(f"unknown tag field: {field}")
        text = bpm_text(value) if field == "bpm" else ("" if value is None else str(value))
        out[field] = text.replace("\x00", "")
    return out


# --- the format, the fields' places -------------------------------------------------------------------------

def tag_format(path) -> Optional[str]:
    """The tags this module writes for the file: "flac", "id3" (MP3 / AAC), "wav", "aiff", "mp4", or None (another
    format, AIFC, ID3v2.2, a fragmented MP4 or one with QuickTime keys metadata, unreadable)."""
    try:
        with open(path, "rb") as f:
            head = f.read(12)
            if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
                return "wav"
            if head[:4] == b"FORM" and head[8:12] == b"AIFF":
                return "aiff"
            if head[4:8] == b"ftyp":
                return "mp4" if not _has_keys_metadata(_read_moov(Path(path))) else None
            offset = 0
            while head[:3] == b"ID3":   # one tag, or a few in a row
                f.seek(offset)
                header = f.read(10)
                if header[3] not in (3, 4):
                    return None
                offset += _id3_size(header)
                f.seek(offset)
                head = f.read(12)
            if head[:4] == b"fLaC":
                return "flac"
            return "id3" if _mpeg_audio_at(f, offset) else None
    except (OSError, TagWriteError, struct.error, IndexError, ValueError):
        return None


def _mpeg_audio_at(f, offset: int) -> bool:
    """MPEG audio (MP3, ADTS AAC) frames at ``offset``, after the zeros some taggers leave past a tag."""
    f.seek(offset)
    data = f.read(65536)
    i = 0
    while i < len(data) and data[i] == 0:
        i += 1
    return i + 1 < len(data) and data[i] == 0xFF and data[i + 1] & 0xE0 == 0xE0


def _levels(fmt: str, field: str, found: Spellings) -> tuple[tuple[str, ...], ...]:
    if fmt == "flac" and field == "artist" and "vorbis:PERFORMER" in found:
        return _FLAC_PERFORMER_LEVELS
    return LEVELS[fmt][field]


def targets(fmt: str, fields, sources: Optional[dict] = None, found: Optional[Spellings] = None
            ) -> dict[str, tuple[str, ...]]:
    """Where each field is written in a file of ``fmt`` holding ``found`` (``fields``: names, or field -> text,
    "" an emptied one): its sources the format has, else the field's copies the file has (none: the reader's
    first place; WAV: INFO and the ID3 chunk), and the format's own place for it (LEVELS' first: what DJ software
    reads, whatever the reader took); an emptied field: its sources and its copies holding a value (a
    fallback field only as its source; a blank placeholder stays). A place that cannot hold any value sends it to
    its ID3v2 field."""
    found = found or {}
    out = {}
    for field in fields:
        emptied = isinstance(fields, dict) and not fields[field]
        levels = _levels(fmt, field, found)
        copies = [s for level in levels for s in level if s not in _FALLBACKS]
        sourced = [s for s in (sources or {}).get(field, ()) if s in SPELLINGS[fmt][field]]
        if emptied:   # where it came from, and its copies holding a value (a blank placeholder, as FLAC's
            # PERFORMER, stays: it steers how MediaInfo reads the others)
            held = [s for s in copies if any(v.strip() for v in found.get(s, ()))]
            chosen = [s for s in sourced if s in found] + held
        else:
            chosen = sourced
            if not chosen:
                chosen = [s for s in copies if s in found] or [levels[0][0]]
                if fmt == "wav" and not any(s in found for s in copies) and len(levels) > 1:
                    chosen.append(levels[1][0])
            # the format's own place too, the one DJ software reads (rekordbox an M4A's ©cmt only: a store's file
            # with a blank ©cmt and the comment in desc showed none), unless FLAC's PERFORMER steers the artist
            if not (fmt == "flac" and field == "artist" and "vorbis:PERFORMER" in (*found, *sourced)):
                chosen = [LEVELS[fmt][field][0][0]] + chosen
        chosen += [_REDIRECT[s] for s in chosen if s in _REDIRECT]
        out[field] = tuple(s for s in dict.fromkeys(chosen) if not s.startswith("id3v1:"))
    return out


def read_spellings(path) -> Spellings:
    """Every place the file holds one of the fields, with its values."""
    path = Path(path)
    fmt = tag_format(path)
    if fmt is None:
        raise TagWriteError(tr("Tags cannot be written to this file's format."))
    try:
        return _READERS[fmt](path)
    except (struct.error, IndexError, ValueError, UnicodeError, OverflowError) as exc:
        raise _damaged() from exc


def read_tags(path) -> dict[str, str]:
    """Each field's value as the file has it: its first spelling there (the reader's first; a field missing:
    not in the dict)."""
    path = Path(path)
    found = read_spellings(path)
    fmt = tag_format(path)
    out = {}
    for field in FIELDS:
        value = next((found[s][0] for s in SPELLINGS[fmt][field] if found.get(s)), None)
        if value is not None:
            out[field] = value
    return out


PlannedPlace = tuple[str, str, list[str], list[str], str]


def planned_writes(path, values: dict, sources: Optional[dict] = None) -> dict[str, list[PlannedPlace]]:
    """Where ``write_tags`` would write each of ``values`` (with ``sources``) and what each place holds: field ->
    [(the place, as ``write_tags``' ``places`` take it: "mp4:©cmt"; its name in the format: "©cmt", "desc", "COMM",
    "TXXX:comment"; its values now; its values after; "source" where the field was read from / "own" the format's
    own place, written too: what DJ software reads / "copy")]. A field whose places all hold their values after
    already is as written. TagWriteError as ``write_tags``."""
    path = Path(path)
    fmt = tag_format(path)
    if fmt is None:
        raise TagWriteError(tr("Tags cannot be written to this file's format."))
    texts = _texts(values)
    try:
        found = _READERS[fmt](path)
    except (struct.error, IndexError, ValueError, UnicodeError, OverflowError) as exc:
        raise _damaged() from exc
    out = {}
    for field, places in targets(fmt, texts, sources, found).items():
        own = _REDIRECT.get(LEVELS[fmt][field][0][0], LEVELS[fmt][field][0][0])
        sourced = set((sources or {}).get(field, ()))
        out[field] = [(s, s.split(":", 1)[1], list(found.get(s, [])), _written(fmt, s, texts[field]),
                       "source" if s in sourced else "own" if s == own else "copy") for s in places]
    return out


def read_fields(path) -> dict[str, str]:
    """The text fields as the analysis reads them (``LEVELS``, as MediaInfo did): the first level holding a
    value, its different values joined with " / " (a field without one: not in the dict)."""
    path = Path(path)
    found = read_spellings(path)
    fmt = tag_format(path)
    out = {}
    for field in TEXT_FIELDS:
        for level in _levels(fmt, field, found):
            values = [v.strip() for s in level for v in found.get(s, ()) if v.strip()]
            if values:
                out[field] = " / ".join(dict.fromkeys(values))
                break
    return out


def tag_sources(path, read: dict[str, str]) -> dict[str, list[str]]:
    """Where the values the analysis read (``read``: field -> text) are in the file: the places of the reader's
    level that held it (the first level holding a value; MediaInfo joins a level's different values with " / ");
    if they do not hold it (the reader's way not followed here), every place holding it; [] when none does (a
    title from the file name, a field read empty)."""
    found = read_spellings(path)
    fmt = tag_format(path)
    out: dict[str, list[str]] = {}
    for field in TEXT_FIELDS:
        value = str(read.get(field) or "").strip()
        parts = {p.strip() for p in value.split(" / ")} if value else set()

        def holds(spelling: str) -> bool:
            return any(v.strip() == value or v.strip() in parts for v in found.get(spelling, ()))

        held: list[str] = []
        if value:
            levels = _levels(fmt, field, found)
            for level in levels:
                if any(v.strip() for s in level for v in found.get(s, ())):   # the level the reader took
                    held = [s for s in level if holds(s)]
                    if held and not any(s in _FALLBACKS for s in held):   # and the field's copies holding it
                        held += [s for lv in levels for s in lv
                                 if s not in _FALLBACKS and s not in held and holds(s)]
                    break
            if not held:
                held = [s for s in SPELLINGS[fmt][field] if holds(s)]
        out[field] = held
    return out


# --- writing ------------------------------------------------------------------------------------------------

def write_tags(path, values: dict, sources: Optional[dict] = None,
               about_to_write: Optional[Callable[[Path, list[Segment]], None]] = None,
               places: Optional[dict] = None) -> dict[str, tuple[str, ...]]:
    """``values`` (FIELDS -> text, the BPM a number; "" / None removes the tag) written into the file's tags,
    where ``sources`` (field -> spellings, ``tag_sources``) say, else where the reader takes each first (``targets``),
    or exactly at ``places`` (field -> spellings, as ``planned_writes`` names them: a field without any is not
    written); returns where each was written. TagWriteError when the file cannot take them or what was written does
    not read back; the file is then as it was. ``about_to_write(path, segments)`` is called with the file to come
    (``Segment``s of the file as it is) before anything is written (core.tags.backup's tag block backup): raising,
    nothing is."""
    path = Path(path)
    texts = _texts(values)
    if not texts:
        return {}
    fmt = tag_format(path)
    if fmt is None:
        raise TagWriteError(tr("Tags cannot be written to this file's format."))
    if places is not None:   # the field's own places (ID3v1 never written, as targets has it)
        chosen = {f: tuple(s for s in dict.fromkeys(places.get(f, ())) if s in SPELLINGS[fmt][f]
                           and not s.startswith("id3v1:")) for f in texts}
        texts = {f: t for f, t in texts.items() if chosen[f]}
        if not texts:
            return {}
    try:
        before = _READERS[fmt](path)
        where = targets(fmt, texts, sources, before) if places is None else {f: chosen[f] for f in texts}
        where = {field: spellings for field, spellings in where.items() if spellings}
        segments, extra_check = _PLANS[fmt](path, texts, where)
        if about_to_write is not None:
            about_to_write(path, segments)

        def check(p: Path) -> None:
            _check_spellings(fmt, before, _READERS[fmt](p), where, texts)
            if extra_check is not None:
                extra_check(p)

        _apply(path, segments, check)
    except (struct.error, IndexError, ValueError, UnicodeError, OverflowError) as exc:
        raise _damaged() from exc
    return where


def _written(fmt: str, spelling: str, text: str) -> list[str]:
    """What a spelling holds after ``text`` was written there."""
    if not text or spelling.startswith("aiff:"):
        return []
    if spelling == "mp4:tmpo":
        bpm = int(round(float(text)))
        return [str(bpm)] if 0 < bpm < 1 << 16 else []
    return [text]


def _check_spellings(fmt: str, before: Spellings, after: Spellings, where: dict, texts: dict[str, str]) -> None:
    """The spellings written hold the values (each of their places), every other one is as it was."""
    written = set()
    for field, places in where.items():
        for spelling in places:
            written.add(spelling)
            want = _written(fmt, spelling, texts[field])
            got = after.get(spelling, [])
            if (want and (not got or any(v != want[0] for v in got))) or (not want and got):
                raise _not_read_back()
    for spelling in set(before) | set(after):
        if spelling not in written and before.get(spelling, []) != after.get(spelling, []):
            raise _not_read_back()


def _apply(path: Path, segments: list[Segment], check: Callable[[Path], None]) -> None:
    """The file made of ``segments``: in place when only new bytes of the same length at the same place change
    (the old ones put back if the check fails), else written anew beside it, checked and moved over it."""
    size = path.stat().st_size
    pos, in_place, writes = 0, True, []
    for seg in segments:
        if isinstance(seg, tuple):
            if seg[0] != pos:
                in_place = False
            pos += seg[1]
        else:
            if seg:
                writes.append((pos, seg))
            pos += len(seg)
    if in_place and pos == size:
        with open(path, "r+b") as f:
            old = []
            for offset, data in writes:
                f.seek(offset)
                old.append((offset, f.read(len(data))))
            try:
                for offset, data in writes:
                    f.seek(offset)
                    f.write(data)
                f.flush()
                os.fsync(f.fileno())
                check(path)
            except BaseException:
                for offset, data in old:
                    f.seek(offset)
                    f.write(data)
                f.flush()
                os.fsync(f.fileno())
                raise
        return
    fd, tmp_name = tempfile.mkstemp(prefix=".mixlyzer-tags-", suffix=path.suffix, dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as out, open(path, "rb") as src:
            for seg in segments:
                if isinstance(seg, tuple):
                    src.seek(seg[0])
                    _copy(src, out, seg[1])
                else:
                    out.write(seg)
            out.flush()
            os.fsync(out.fileno())
        shutil.copymode(path, tmp)
        check(tmp)
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def _copy(src, out, length: int) -> None:
    while length > 0:
        block = src.read(min(length, _COPY_BLOCK))
        if not block:
            raise TagWriteError(tr("The file changed while its tags were written; it is unchanged."))
        out.write(block)
        length -= len(block)


def _keys(where: dict, prefix: str) -> dict[str, list[str]]:
    """field -> the names after ``prefix`` of its places written."""
    return {field: [s[len(prefix):] for s in places if s.startswith(prefix)] for field, places in where.items()}


# --- FLAC: the Vorbis comment -------------------------------------------------------------------------------

def _flac_layout(f, size: int) -> tuple[int, list[tuple[int, int, int]]]:
    """(the offset of "fLaC", [(block type, offset of its data, length)]) of a FLAC file's metadata."""
    start = 0
    f.seek(0)
    header = f.read(10)
    while header[:3] == b"ID3":
        start += _id3_size(header)
        f.seek(start)
        header = f.read(10)
    if header[:4] != b"fLaC":
        raise _damaged()
    blocks, pos = [], start + 4
    while True:
        f.seek(pos)
        h = f.read(4)
        if len(h) < 4:
            raise _damaged()
        kind, length = h[0] & 0x7F, int.from_bytes(h[1:4], "big")
        if kind == 127 or pos + 4 + length > size or len(blocks) > 4096:
            raise _damaged()
        blocks.append((kind, pos + 4, length))
        pos += 4 + length
        if h[0] & 0x80:
            break
    if blocks[0][0] != 0:   # STREAMINFO comes first
        raise _damaged()
    return start, blocks


def _vorbis_parse(data: bytes) -> tuple[bytes, list[bytes]]:
    n = struct.unpack_from("<I", data, 0)[0]
    if 4 + n + 4 > len(data):
        raise _damaged()
    vendor, pos = data[4:4 + n], 4 + n
    count = struct.unpack_from("<I", data, pos)[0]
    pos += 4
    entries = []
    for _ in range(count):
        length = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        if pos + length > len(data):
            raise _damaged()
        entries.append(data[pos:pos + length])
        pos += length
    return vendor, entries


def _vorbis_build(vendor: bytes, entries: list[bytes]) -> bytes:
    out = [struct.pack("<I", len(vendor)), vendor, struct.pack("<I", len(entries))]
    for entry in entries:
        out += [struct.pack("<I", len(entry)), entry]
    return b"".join(out)


def _vorbis_key(entry: bytes) -> str:
    return entry.split(b"=", 1)[0].decode("ascii", "replace").upper()


def _plan_flac(path: Path, texts: dict[str, str], where: dict):
    size = path.stat().st_size
    with open(path, "rb") as f:
        start, blocks = _flac_layout(f, size)
        payloads = []
        for kind, offset, length in blocks:
            if kind == 1:   # padding: made again at the end
                continue
            f.seek(offset)
            payloads.append([kind, f.read(length)])
    comment = next((p for p in payloads if p[0] == 4), None)
    if comment is None and not any(texts.values()):
        return [(0, size)], None
    if comment is None:
        comment = [4, _vorbis_build(b"Mixlyzer", [])]
        payloads.insert(1, comment)
    vendor, entries = _vorbis_parse(comment[1])
    keys = _keys(where, "vorbis:")
    drop = {key for names in keys.values() for key in names}
    entries = [e for e in entries if _vorbis_key(e) not in drop]
    for field in FIELDS:
        if texts.get(field):
            entries += [f"{key}={texts[field]}".encode("utf-8") for key in keys.get(field, ())]
    comment[1] = _vorbis_build(vendor, entries)
    if len(comment[1]) >= 1 << 24:
        raise TagWriteError(tr("The tags are too long for the file."))
    used = sum(4 + len(data) for _, data in payloads)
    room = blocks[-1][1] + blocks[-1][2] - (start + 4)
    if used == room:
        padding = None
    elif used + 4 <= room and room - used - 4 < 1 << 24:
        padding = room - used - 4
    else:
        padding = PADDING
    if padding is not None:
        payloads.append([1, bytes(padding)])
    meta = bytearray()
    for i, (kind, data) in enumerate(payloads):
        meta += bytes([(0x80 if i == len(payloads) - 1 else 0) | kind]) + len(data).to_bytes(3, "big") + data
    end = start + 4 + room
    return [(0, start + 4), bytes(meta), (end, size - end)], None


def _read_flac(path: Path) -> Spellings:
    known = {s for places in SPELLINGS["flac"].values() for s in places}
    out: Spellings = {}
    with open(path, "rb") as f:
        _start, blocks = _flac_layout(f, path.stat().st_size)
        for kind, offset, length in blocks:
            if kind == 4:
                f.seek(offset)
                for entry in _vorbis_parse(f.read(length))[1]:
                    spelling = "vorbis:" + _vorbis_key(entry)
                    if spelling in known and b"=" in entry:
                        out.setdefault(spelling, []).append(entry.split(b"=", 1)[1].decode("utf-8", "replace"))
                break
    return out


# --- ID3v2 --------------------------------------------------------------------------------------------------

def _syncsafe(b: bytes) -> int:
    return (b[0] & 0x7F) << 21 | (b[1] & 0x7F) << 14 | (b[2] & 0x7F) << 7 | (b[3] & 0x7F)


def _to_syncsafe(n: int) -> bytes:
    if n >= 1 << 28:
        raise TagWriteError(tr("The tags are too long for the file."))
    return bytes([(n >> 21) & 0x7F, (n >> 14) & 0x7F, (n >> 7) & 0x7F, n & 0x7F])


def _id3_size(header: bytes) -> int:
    """The whole ID3v2 tag's length (its footer included) from its 10-byte header."""
    if len(header) < 10 or header[:3] != b"ID3" or any(b & 0x80 for b in header[6:10]):
        raise _damaged()
    return 10 + _syncsafe(header[6:10]) + (10 if header[3] == 4 and header[5] & 0x10 else 0)


def _id3_frames(tag: bytes) -> tuple[int, list[tuple[bytes, bytes, bytes]]]:
    """(major version, [(frame id, flags, data)]) of an ID3v2.3 / 2.4 tag (its header first)."""
    major, flags = tag[3], tag[5]
    if major not in (3, 4) or len(tag) < 10:
        raise _damaged()
    body = tag[10:10 + _syncsafe(tag[6:10])]
    if len(body) < _syncsafe(tag[6:10]):
        raise _damaged()
    if flags & 0x80 and major == 3:   # the whole tag unsynchronised (v2.4: each frame says so itself)
        body = body.replace(b"\xff\x00", b"\xff")
    pos = 0
    if flags & 0x40:   # an extended header (left out when the tag is written again)
        if len(body) < 4:
            raise _damaged()
        pos = 4 + int.from_bytes(body[:4], "big") if major == 3 else _syncsafe(body[:4])
        if pos > len(body):
            raise _damaged()
    frames = _id3_parse_frames(body, pos, major == 4)
    if frames is None and major == 4:   # sizes written as plain integers (some old taggers)
        frames = _id3_parse_frames(body, pos, False)
    if frames is None:
        raise _damaged()
    return major, frames


def _id3_parse_frames(body: bytes, pos: int, syncsafe: bool):
    frames = []
    while pos + 10 <= len(body):
        fid = body[pos:pos + 4]
        if fid[0] == 0:   # padding
            break
        if not all(48 <= c <= 57 or 65 <= c <= 90 for c in fid):
            return None
        raw = body[pos + 4:pos + 8]
        if syncsafe and any(b & 0x80 for b in raw):
            return None
        size = _syncsafe(raw) if syncsafe else int.from_bytes(raw, "big")
        if pos + 10 + size > len(body):
            return None
        frames.append((fid, body[pos + 8:pos + 10], body[pos + 10:pos + 10 + size]))
        pos += 10 + size
    if any(body[pos:]):   # what follows the frames is padding
        return None
    return frames


_INFLATED_MAX = 32 << 20   # a compressed frame unpacking to more than this is not read (a zip bomb in a tag)


def _inflate(data: bytes) -> Optional[bytes]:
    unpack = zlib.decompressobj()
    out = unpack.decompress(data, _INFLATED_MAX)
    return None if unpack.unconsumed_tail else out


def _id3_frame_data(flags: bytes, data: bytes, major: int) -> Optional[bytes]:
    """A frame's content without its format flags' extras, decompressed (None: encrypted, or not readable)."""
    try:
        if major == 4:
            if flags[1] & 0x04:
                return None
            if flags[1] & 0x40:   # group
                data = data[1:]
            if flags[1] & 0x01:   # data length indicator
                data = data[4:]
            if flags[1] & 0x02:
                data = data.replace(b"\xff\x00", b"\xff")
            return _inflate(data) if flags[1] & 0x08 else data
        if flags[1] & 0x40:
            return None
        compressed = bool(flags[1] & 0x80)
        if compressed:   # the size it decompresses to first
            data = data[4:]
        if flags[1] & 0x20:   # group
            data = data[1:]
        return _inflate(data) if compressed else data
    except zlib.error:
        return None


def _id3_split(data: bytes, enc: int) -> tuple[bytes, bytes]:
    """A string ended by the encoding's zero, and what follows it."""
    if enc in (1, 2):
        for i in range(0, len(data) - 1, 2):
            if data[i:i + 2] == b"\x00\x00":
                return data[:i], data[i + 2:]
        return data, b""
    i = data.find(b"\x00")
    return (data, b"") if i < 0 else (data[:i], data[i + 1:])


def _id3_decode(raw: bytes, enc: int) -> str:
    if enc == 0:
        text = raw.decode("latin-1")
    elif enc == 1:
        text = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-16-le")
    elif enc == 2:
        text = raw.decode("utf-16-be")
    elif enc == 3:
        text = raw.decode("utf-8")
    else:
        raise _damaged()
    return text.split("\x00")[0]


def _id3_pair(data: Optional[bytes], *, lang: bool) -> Optional[tuple[str, str]]:
    """(description, text) of a COMM frame's content (``lang``: after its language) or a TXXX's (None: not
    readable)."""
    if data is None or len(data) < (4 if lang else 1):
        return None
    enc = data[0]
    if enc > 3:
        return None
    desc, text = _id3_split(data[4 if lang else 1:], enc)
    try:
        return _id3_decode(desc, enc), _id3_decode(text, enc)
    except (UnicodeError, TagWriteError):
        return None


def _id3_spelling(fid: bytes, flags: bytes, data: bytes, major: int) -> Optional[tuple[str, str]]:
    """(spelling, value) of a frame holding one of the fields (None: another frame, or not readable). The
    main comment is the COMM without a description; FFmpeg writes it as TXXX "comment"."""
    if fid in (b"TIT2", b"TPE1", b"TPE2", b"TALB", b"TBPM", b"TKEY"):
        content = _id3_frame_data(flags, data, major)
        if not content:
            return None
        try:
            return "id3:" + fid.decode(), _id3_decode(content[1:], content[0])
        except (UnicodeError, TagWriteError):
            return None
    if fid == b"COMM":
        pair = _id3_pair(_id3_frame_data(flags, data, major), lang=True)
        return ("id3:COMM", pair[1]) if pair is not None and pair[0] == "" else None
    if fid == b"TXXX":
        pair = _id3_pair(_id3_frame_data(flags, data, major), lang=False)
        return ("id3:TXXX:comment", pair[1]) if pair is not None and pair[0].lower() == "comment" else None
    return None


def _id3_tag_spellings(tag: bytes, out: Spellings) -> None:
    major, frames = _id3_frames(tag)
    for fid, flags, data in frames:
        found = _id3_spelling(fid, flags, data, major)
        if found is not None:
            out.setdefault(found[0], []).append(found[1])


def _id3_strings(first: str, second: Optional[str], major: int) -> tuple[int, bytes]:
    """(encoding, the string(s) encoded; two: the first ended by the encoding's zero)."""
    texts = [first] if second is None else [first, second]
    if major == 4:
        return 3, b"\x00".join(t.encode("utf-8") for t in texts)
    try:
        return 0, b"\x00".join(t.encode("latin-1") for t in texts)
    except UnicodeEncodeError:
        return 1, b"\x00\x00".join(b"\xff\xfe" + t.encode("utf-16-le") for t in texts)


def _id3_frame(fid: bytes, data: bytes, major: int, flags: bytes = b"\x00\x00") -> bytes:
    size = _to_syncsafe(len(data)) if major == 4 else len(data).to_bytes(4, "big")
    return fid + size + flags + data


def _id3_rebuild(tag: Optional[bytes], places: dict[str, list[str]], texts: dict[str, str]) -> tuple[int, bytes]:
    """(version, frames) of the tag (None: a new v2.3 one) with the ID3 places written (``places``: field -> the
    names after "id3:"); the other frames as they are."""
    major, frames = _id3_frames(tag) if tag else (3, [])
    drop = {name for names in places.values() for name in names}
    txxx_desc = "comment"   # FFmpeg's TXXX comment written again with its own description's case
    out = bytearray()
    for fid, flags, data in frames:
        found = _id3_spelling(fid, flags, data, major)
        name = found[0][4:] if found is not None else None
        if name in drop:
            if name == "TXXX:comment":
                txxx_desc = _id3_pair(_id3_frame_data(flags, data, major), lang=False)[0]
            continue
        out += _id3_frame(fid, data, major, flags)
    for field in FIELDS:
        text = texts.get(field, "")
        if not text:
            continue
        for name in places.get(field, ()):
            if name == "COMM":
                enc, raw = _id3_strings("", text, major)
                out += _id3_frame(b"COMM", bytes([enc]) + b"eng" + raw, major)
            elif name == "TXXX:comment":
                enc, raw = _id3_strings(txxx_desc, text, major)
                out += _id3_frame(b"TXXX", bytes([enc]) + raw, major)
            else:
                enc, raw = _id3_strings(text, None, major)
                out += _id3_frame(name.encode(), bytes([enc]) + raw, major)
    return major, bytes(out)


def _id3_tag(major: int, frames: bytes, padding: int) -> bytes:
    return b"ID3" + bytes([major, 0, 0]) + _to_syncsafe(len(frames) + padding) + frames + bytes(padding)


def _id3_tags(f) -> list[bytes]:
    """The ID3v2 tags at the start of the file, one or a few in a row."""
    tags, offset = [], 0
    f.seek(0)
    header = f.read(10)
    while header[:3] == b"ID3":
        total = _id3_size(header)
        f.seek(offset)
        tag = f.read(total)
        if len(tag) < total:
            raise _damaged()
        tags.append(tag)
        offset += total
        header = f.read(10)
    return tags


def _plan_id3(path: Path, texts: dict[str, str], where: dict):
    size = path.stat().st_size
    with open(path, "rb") as f:
        old_tags = _id3_tags(f)
    places = _keys(where, "id3:")
    if not old_tags:
        major, frames = _id3_rebuild(None, places, texts)
        if not frames:   # no tag, nothing to put in one
            return [(0, size)], None
        return [_id3_tag(major, frames, PADDING), (0, size)], None
    segments: list[Segment] = []
    for i, old in enumerate(old_tags):   # each tag of a row the same (readers take the first or the last)
        major, frames = _id3_rebuild(old, places, texts)
        room = len(old) - 10
        last = i == len(old_tags) - 1
        segments.append(_id3_tag(major, frames, room - len(frames) if len(frames) <= room
                                 else (PADDING if last else 0)))
    end = sum(len(t) for t in old_tags)
    return [*segments, (end, size - end)], None


_V1_FIELDS = (("id3v1:title", 3, 33), ("id3v1:artist", 33, 63), ("id3v1:album", 63, 93), ("id3v1:comment", 97, 127))


def _read_id3_file(path: Path) -> Spellings:
    out: Spellings = {}
    with open(path, "rb") as f:
        for tag in _id3_tags(f):
            _id3_tag_spellings(tag, out)
        if path.stat().st_size >= 128:
            f.seek(-128, os.SEEK_END)
            v1 = f.read(128)
            if v1[:3] == b"TAG":
                for spelling, start, end in _V1_FIELDS:
                    raw = v1[start:end]
                    if spelling == "id3v1:comment" and raw[28] == 0 and raw[29] != 0:   # v1.1: a track number
                        raw = raw[:28]
                    text = raw.split(b"\x00")[0].decode("latin-1").rstrip()
                    if text:
                        out[spelling] = [text]
    return out


# --- WAV / AIFF: chunks -------------------------------------------------------------------------------------

def _iff_layout(f, size: int, big: bool) -> tuple[list[tuple[bytes, int, int]], int]:
    """([(chunk id, header offset, data length)], the end of the chunks) of a RIFF / FORM file."""
    order = "big" if big else "little"
    f.seek(0)
    head = f.read(12)
    end = min(8 + int.from_bytes(head[4:8], order), size)
    chunks, pos = [], 12
    while pos + 8 <= end:
        f.seek(pos)
        h = f.read(8)
        length = int.from_bytes(h[4:8], order)
        if not all(0x20 <= c <= 0x7E for c in h[:4]) or pos + 8 + length > size:
            raise _damaged()
        chunks.append((h[:4], pos, length))
        pos += 8 + length + (length & 1)
    return chunks, min(pos, size)


def _iff_chunk(cid: bytes, data: bytes, big: bool) -> bytes:
    return cid + len(data).to_bytes(4, "big" if big else "little") + data + (b"\x00" if len(data) & 1 else b"")


def _info_parse(data: bytes) -> list[tuple[bytes, bytes]]:
    """The subchunks (id, data) of a LIST INFO chunk's data (after "INFO")."""
    subs, pos = [], 4
    while pos + 8 <= len(data):
        sid, length = data[pos:pos + 4], int.from_bytes(data[pos + 4:pos + 8], "little")
        if pos + 8 + length > len(data):
            raise _damaged()
        subs.append((sid, data[pos + 8:pos + 8 + length]))
        pos += 8 + length + (length & 1)
    return subs


def _info_rebuild(data: Optional[bytes], places: dict[str, list[str]], texts: dict[str, str]) -> bytes:
    """The LIST INFO chunk's data with its places written (b"": nothing left in it)."""
    subs = _info_parse(data) if data else []
    drop = {name.encode() for names in places.values() for name in names}
    kept = [(sid, value) for sid, value in subs if sid not in drop]
    for field in FIELDS:
        if texts.get(field):
            kept += [(name.encode(), texts[field].encode("utf-8") + b"\x00") for name in places.get(field, ())]
    if not kept:
        return b""
    return b"INFO" + b"".join(_iff_chunk(sid, value, False) for sid, value in kept)


def _iff_parts(path: Path, big: bool):
    """(chunks, their end, the first ID3 chunk (offset, id, data) or None, the first LIST INFO (offset, data) or
    None, the FORM / RIFF header)."""
    size = path.stat().st_size
    with open(path, "rb") as f:
        chunks, end = _iff_layout(f, size, big)
        id3 = info = None
        for cid, pos, length in chunks:
            if cid in (b"id3 ", b"ID3 ") and id3 is None:
                f.seek(pos + 8)
                id3 = (pos, cid, f.read(length))
            elif not big and cid == b"LIST" and info is None:
                f.seek(pos + 8)
                data = f.read(length)
                if data[:4] == b"INFO":
                    info = (pos, data)
        f.seek(0)
        form = f.read(12)
    if id3 is not None and id3[2][:3] != b"ID3":
        raise _damaged()
    ids = {cid for cid, _pos, _length in chunks}
    if not (b"COMM" in ids if big else {b"fmt ", b"data"} <= ids):
        raise _damaged()
    return chunks, end, id3, info, form


def _plan_iff(path: Path, texts: dict[str, str], where: dict, big: bool):
    size = path.stat().st_size
    chunks, end, id3, info, form = _iff_parts(path, big)
    id3_places = _keys(where, "id3:")
    major, frames = _id3_rebuild(id3[2] if id3 is not None else None, id3_places, texts)
    new_id3 = _iff_chunk(b"ID3 " if big else (id3[1] if id3 is not None else b"id3 "),
                         _id3_tag(major, frames, 0), big)
    # AIFF's own NAME / AUTH / ANNO written to: left out (their value went to the ID3 chunk)
    native = {name.encode() for names in _keys(where, "aiff:").values() for name in names} if big else set()
    info_places = _keys(where, "info:")
    new_info = b""
    if not big and (info is not None or any(texts.get(f) for f, names in info_places.items() if names)):
        data = _info_rebuild(info[1] if info is not None else None, info_places, texts)
        new_info = _iff_chunk(b"LIST", data, False) if data else b""
    body: list[Segment] = []
    for cid, pos, length in chunks:
        if id3 is not None and pos == id3[0]:
            body.append(new_id3)
        elif info is not None and pos == info[0]:
            body.append(new_info)
        elif cid in native:
            continue
        elif pos + 8 + length + (length & 1) > size:   # its pad byte missing at the end of the file
            body += [(pos, 8 + length), b"\x00"]
        else:
            body.append((pos, 8 + length + (length & 1)))
    if id3 is None and frames:
        body.append(new_id3)
    if info is None and new_info:
        body.append(new_info)
    length = 4 + sum(len(s) if isinstance(s, bytes) else s[1] for s in body)
    header = form[:4] + length.to_bytes(4, "big" if big else "little") + form[8:12]
    return [header, *body, (end, size - end)], None


def _plan_wav(path: Path, texts: dict[str, str], where: dict):
    return _plan_iff(path, texts, where, big=False)


def _plan_aiff(path: Path, texts: dict[str, str], where: dict):
    return _plan_iff(path, texts, where, big=True)


_AIFF_TEXT = {b"NAME": "aiff:NAME", b"AUTH": "aiff:AUTH", b"ANNO": "aiff:ANNO"}


def _read_iff(path: Path, big: bool) -> Spellings:
    out: Spellings = {}
    chunks, _end, id3, info, _form = _iff_parts(path, big)
    if info is not None:
        for sid, value in _info_parse(info[1]):
            spelling = "info:" + sid.decode("latin-1")
            if spelling in SPELLINGS["wav"]["title"] + SPELLINGS["wav"]["artist"] + SPELLINGS["wav"]["album"] \
                    + SPELLINGS["wav"]["comment"]:
                out.setdefault(spelling, []).append(value.split(b"\x00")[0].decode("utf-8", "replace"))
    if id3 is not None:
        _id3_tag_spellings(id3[2][:_id3_size(id3[2][:10])], out)
    if big:
        with open(path, "rb") as f:
            for cid, pos, length in chunks:
                if cid in _AIFF_TEXT:
                    f.seek(pos + 8)
                    out.setdefault(_AIFF_TEXT[cid], []).append(f.read(length).decode("utf-8", "replace"))
    return out


def _read_wav(path: Path) -> Spellings:
    return _read_iff(path, big=False)


def _read_aiff(path: Path) -> Spellings:
    return _read_iff(path, big=True)


# --- MP4: the iTunes list, the 3GPP texts -------------------------------------------------------------------

_ILST_TEXT = {"©nam": b"\xa9nam", "©ART": b"\xa9ART", "©alb": b"\xa9alb", "©cmt": b"\xa9cmt",
              "aART": b"aART", "desc": b"desc"}
_GPP_TEXT = {"titl": b"titl", "perf": b"perf", "albm": b"albm", "dscp": b"dscp"}   # 3GPP asset texts in udta
_KEY_MEAN, _KEY_NAME = "com.apple.iTunes", "initialkey"
_OFFSET_PATHS = ((b"trak", b"mdia", b"minf", b"stbl", b"stco"), (b"trak", b"mdia", b"minf", b"stbl", b"co64"))
_UNDETERMINED = (0x55C4).to_bytes(2, "big")   # ISO 639-2 "und", packed


def _box(kind: bytes, payload: bytes) -> bytes:
    size = 8 + len(payload)
    if size < 1 << 32:
        return size.to_bytes(4, "big") + kind + payload
    return (1).to_bytes(4, "big") + kind + (size + 8).to_bytes(8, "big") + payload


def _atoms(data: bytes, start: int, end: int) -> tuple[list[tuple[bytes, int, int, int]], int]:
    """([(type, offset, header length, length)] of the atoms in data[start:end], where they stop)."""
    out, pos = [], start
    while pos + 8 <= end:
        length, head = int.from_bytes(data[pos:pos + 4], "big"), 8
        if length == 1:
            if pos + 16 > end:
                raise _damaged()
            length, head = int.from_bytes(data[pos + 8:pos + 16], "big"), 16
        elif length == 0:
            length = end - pos
        if length < head or pos + length > end:
            raise _damaged()
        out.append((data[pos + 4:pos + 8], pos, head, length))
        pos += length
    return out, pos


def _head(box: bytes) -> int:
    return 16 if int.from_bytes(box[:4], "big") == 1 else 8


def _meta_skip(meta: bytes) -> int:
    """The bytes before a meta atom's children: an ISO full box's version and flags, none in QuickTime's."""
    head = _head(meta)
    return 0 if meta[head + 4:head + 8] in (b"hdlr", b"ilst", b"keys", b"free") else 4


def _child(box: bytes, kind: bytes, skip: int = 0) -> Optional[bytes]:
    start = _head(box) + skip
    for k, off, _h, length in _atoms(box, start, len(box))[0]:
        if k == kind:
            return box[off:off + length]
    return None


def _with_child(box: bytes, kind: bytes, make: Callable[[Optional[bytes]], bytes], skip: int = 0) -> bytes:
    """The atom with its first child of ``kind`` made again by ``make`` (from None when it has none; b"" drops
    it); the other children, and whatever follows them, as they are."""
    head = _head(box)
    start = head + skip
    kids, stop = _atoms(box, start, len(box))
    parts, found = [box[head:start]], False
    for k, off, _h, length in kids:
        child = box[off:off + length]
        if box[off:off + 4] == b"\x00\x00\x00\x00":   # "to the end of the parent": its length written out
            child = _box(k, child[8:])
        if k == kind and not found:
            found = True
            child = make(child)
        parts.append(child)
    if not found:
        parts.append(make(None))
    parts.append(box[stop:])
    return _box(box[4:8], b"".join(parts))


def _data_atom(kind: int, payload: bytes) -> bytes:
    return _box(b"data", kind.to_bytes(4, "big") + b"\x00\x00\x00\x00" + payload)


def _item_data(item: bytes) -> Optional[bytes]:
    """The first data atom's value of an ilst item."""
    data = _child(item, b"data")
    if data is None or len(data) < _head(data) + 8:
        return None
    return data[_head(data) + 8:]


def _freeform_name(item: bytes) -> tuple[str, str]:
    mean = _child(item, b"mean")
    name = _child(item, b"name")
    decode = lambda box: box[_head(box) + 4:].decode("utf-8", "replace") if box is not None else ""
    return decode(mean), decode(name)


def _is_key_item(kind: bytes, item: bytes) -> bool:
    if kind != b"----":
        return False
    mean, name = _freeform_name(item)
    return mean == _KEY_MEAN and name.lower() == _KEY_NAME


def _ilst_spelling(kind: bytes, item: bytes) -> Optional[str]:
    if kind == b"tmpo":
        return "mp4:tmpo"
    if _is_key_item(kind, item):
        return "mp4:initialkey"
    name = next((n for n, k in _ILST_TEXT.items() if k == kind), None)
    return "mp4:" + name if name is not None else None


def _ilst_rebuild(ilst: Optional[bytes], places: dict[str, list[str]], texts: dict[str, str]) -> bytes:
    drop = {name for names in places.values() for name in names}
    kept, tail = [], b""
    if ilst is not None:
        kids, stop = _atoms(ilst, _head(ilst), len(ilst))
        for k, off, _h, length in kids:
            item = ilst[off:off + length]
            spelling = _ilst_spelling(k, item)
            if spelling is not None and spelling[4:] in drop:
                continue
            kept.append(item)
        tail = ilst[stop:]
    for field in FIELDS:
        text = texts.get(field, "")
        if not text:
            continue
        for name in places.get(field, ()):
            if name in _ILST_TEXT:
                kept.append(_box(_ILST_TEXT[name], _data_atom(1, text.encode("utf-8"))))
            elif name == "tmpo":
                bpm = int(round(float(text)))
                if 0 < bpm < 1 << 16:
                    kept.append(_box(b"tmpo", _data_atom(21, bpm.to_bytes(2, "big"))))
            elif name == "initialkey":
                kept.append(_box(b"----", _box(b"mean", bytes(4) + _KEY_MEAN.encode())
                                 + _box(b"name", bytes(4) + _KEY_NAME.encode())
                                 + _data_atom(1, text.encode("utf-8"))))
    return _box(b"ilst", b"".join(kept) + tail)


def _hdlr_mdir() -> bytes:
    return _box(b"hdlr", bytes(4) + bytes(4) + b"mdir" + b"appl" + bytes(8) + b"\x00")


def _gpp_split(atom: bytes) -> tuple[bytes, str, bytes]:
    """(version / flags and language, the text, what follows its end: albm's track number) of a 3GPP asset
    text atom."""
    head = _head(atom)
    prefix, body = atom[head:head + 6], atom[head + 6:]
    if len(prefix) < 6:
        raise _damaged()
    if body[:2] in (b"\xfe\xff", b"\xff\xfe"):
        raw, rest = _id3_split(body, 1)
        return prefix, raw.decode("utf-16"), rest
    raw, rest = _id3_split(body, 0)
    return prefix, raw.decode("utf-8", "replace"), rest


def _gpp_rebuild(udta: bytes, places: dict[str, list[str]], texts: dict[str, str]) -> bytes:
    """udta with its 3GPP texts written (each atom's own language kept; removed when the value is emptied;
    made, language undetermined, when it has none)."""
    field_of = {_GPP_TEXT[name]: field for field, names in places.items() for name in names if name in _GPP_TEXT}
    head = _head(udta)
    kids, stop = _atoms(udta, head, len(udta))
    parts, seen = [], set()
    for k, off, _h, length in kids:
        atom = udta[off:off + length]
        if k in field_of:
            seen.add(k)
            text = texts[field_of[k]]
            if not text:
                continue
            prefix, _old, rest = _gpp_split(atom)
            atom = _box(k, prefix + text.encode("utf-8") + b"\x00" + (rest if len(rest) == 1 else b""))
        parts.append(atom)
    for k, field in field_of.items():
        if k not in seen and texts[field]:
            parts.append(_box(k, bytes(4) + _UNDETERMINED + texts[field].encode("utf-8") + b"\x00"))
    return _box(udta[4:8], b"".join(parts) + udta[stop:])


def _ilst(moov: bytes) -> Optional[bytes]:
    udta = _child(moov, b"udta")
    meta = _child(udta, b"meta") if udta is not None else None
    return _child(meta, b"ilst", _meta_skip(meta)) if meta is not None else None


def _moov_rebuild(moov: bytes, where: dict, texts: dict[str, str]) -> Optional[bytes]:
    """moov with its places written (None: nothing to change: no list, no 3GPP text, and nothing to add)."""
    places = _keys(where, "mp4:")
    ilst_places = {f: [n for n in names if n not in _GPP_TEXT] for f, names in places.items()}
    gpp_places = {f: [n for n in names if n in _GPP_TEXT] for f, names in places.items()}
    udta = _child(moov, b"udta")
    adds_ilst = any(texts[f] for f, names in ilst_places.items() if names)
    adds_gpp = any(texts[f] for f, names in gpp_places.items() if names)
    has_ilst = _ilst(moov) is not None
    if udta is None and not adds_ilst and not adds_gpp:
        return None

    def meta_make(meta: Optional[bytes]) -> bytes:
        if meta is None:
            return _box(b"meta", bytes(4) + _hdlr_mdir() + _ilst_rebuild(None, ilst_places, texts))
        if _meta_skip(meta) == 0:   # QuickTime-style (no version / flags): made an ISO full box, as iTunes'
            meta = _box(b"meta", bytes(4) + meta[_head(meta):])
        return _with_child(meta, b"ilst", lambda ilst: _ilst_rebuild(ilst, ilst_places, texts), skip=4)

    def udta_make(old: Optional[bytes]) -> bytes:
        new = old if old is not None else _box(b"udta", b"")
        if has_ilst or adds_ilst:
            new = _with_child(new, b"meta", meta_make)
        return _gpp_rebuild(new, gpp_places, texts)

    return _with_child(moov, b"udta", udta_make)


def _has_keys_metadata(moov: bytes) -> bool:
    """QuickTime keys metadata (a meta with an mdta handler) in moov or its udta."""
    udta = _child(moov, b"udta")
    for meta in (_child(moov, b"meta"), _child(udta, b"meta") if udta is not None else None):
        if meta is not None:
            hdlr = _child(meta, b"hdlr", _meta_skip(meta))
            if hdlr is not None and hdlr[_head(hdlr) + 8:_head(hdlr) + 12] == b"mdta":
                return True
    return False


def _find(data, start: int, end: int, path: tuple[bytes, ...]) -> list[tuple[int, int, int]]:
    found = []
    for k, off, head, length in _atoms(data, start, end)[0]:
        if k == path[0]:
            if len(path) == 1:
                found.append((off, head, length))
            else:
                found += _find(data, off + head, off + length, path[1:])
    return found


def _offset_tables(moov) -> list[tuple[int, int, int]]:
    """(where the entries start, how many, bytes each) of every chunk offset table (stco / co64) in moov."""
    tables = []
    for path, width in zip(_OFFSET_PATHS, (4, 8)):
        for off, head, length in _find(moov, _head(moov), len(moov), path):
            count = int.from_bytes(moov[off + head + 4:off + head + 8], "big")
            start = off + head + 8
            if start + count * width > off + length:
                raise _damaged()
            tables.append((start, count, width))
    return tables


def _chunk_offsets(moov) -> list[list[int]]:
    return [[int.from_bytes(moov[start + i * width:start + (i + 1) * width], "big") for i in range(count)]
            for start, count, width in _offset_tables(moov)]


def _shift_offsets(moov: bytearray, after: int, shift: int) -> None:
    """The chunk offsets at or past ``after`` moved by ``shift`` (the audio after a grown moov)."""
    for start, count, width in _offset_tables(moov):
        for i in range(count):
            p = start + i * width
            value = int.from_bytes(moov[p:p + width], "big")
            if value >= after:
                value += shift
                if value >= 1 << (8 * width):
                    raise TagWriteError(tr("The file's chunk offsets cannot take the longer tags; it is unchanged."))
                moov[p:p + width] = value.to_bytes(width, "big")


def _file_atoms(f, size: int) -> list[tuple[bytes, int, int, int]]:
    atoms, pos = [], 0
    while pos + 8 <= size:
        f.seek(pos)
        h = f.read(16)
        length, head = int.from_bytes(h[:4], "big"), 8
        if length == 1:
            length, head = int.from_bytes(h[8:16], "big"), 16
        elif length == 0:
            length = size - pos
        if length < head or pos + length > size or not all(0x20 <= c <= 0x7E for c in h[4:8]):
            raise _damaged()
        atoms.append((h[4:8], pos, head, length))
        pos += length
    return atoms


def _moov_at(path: Path) -> tuple[int, int, list]:
    """(moov's offset, its length, the top-level atoms)."""
    size = path.stat().st_size
    with open(path, "rb") as f:
        atoms = _file_atoms(f, size)
    if any(kind == b"moof" for kind, *_ in atoms):
        raise TagWriteError(tr("Fragmented MP4 files are not written."))
    moovs = [a for a in atoms if a[0] == b"moov"]
    if len(moovs) != 1:
        raise _damaged()
    return moovs[0][1], moovs[0][3], atoms


def _read_moov(path: Path) -> bytes:
    offset, length, _atoms_ = _moov_at(path)
    with open(path, "rb") as f:
        f.seek(offset)
        return f.read(length)


def _plan_mp4(path: Path, texts: dict[str, str], where: dict):
    size = path.stat().st_size
    moov_at, moov_len, atoms = _moov_at(path)
    old_moov = _read_moov(path)
    if _has_keys_metadata(old_moov):
        raise TagWriteError(tr("MP4 files with QuickTime keys metadata are not written."))
    old_offsets = _chunk_offsets(old_moov)
    rebuilt = _moov_rebuild(old_moov, where, texts)
    if rebuilt is None:
        return [(0, size)], None
    new_moov = bytearray(rebuilt)
    after = next((a for a in atoms if a[1] == moov_at + moov_len), None)
    room = moov_len + (after[3] if after is not None and after[0] in (b"free", b"skip") else 0)
    if len(new_moov) == room:
        new = bytes(new_moov)
    elif len(new_moov) + 8 <= room:   # the rest of the room stays padding
        new = bytes(new_moov) + _box(b"free", bytes(room - len(new_moov) - 8))
    else:   # moov grows: the audio after it moves, and its chunk offsets with it
        padding = _box(b"free", bytes(PADDING - 8))
        _shift_offsets(new_moov, moov_at + room, len(new_moov) + len(padding) - room)
        new = bytes(new_moov) + padding

    def check(p: Path) -> None:
        """The audio chunks where the new offsets say (the first and last of each track)."""
        offsets = _chunk_offsets(_read_moov(p))
        if [len(t) for t in offsets] != [len(t) for t in old_offsets]:
            raise _not_read_back()
        with open(path, "rb") as old_file, open(p, "rb") as new_file:
            for old_table, new_table in zip(old_offsets, offsets):
                for i in {0, len(old_table) - 1} if old_table else ():
                    old_file.seek(old_table[i])
                    new_file.seek(new_table[i])
                    if old_file.read(32) != new_file.read(32):
                        raise _not_read_back()

    return [(0, moov_at), new, (moov_at + room, size - moov_at - room)], check


def _read_mp4(path: Path) -> Spellings:
    moov = _read_moov(path)
    out: Spellings = {}
    ilst = _ilst(moov)
    if ilst is not None:
        for k, off, _h, length in _atoms(ilst, _head(ilst), len(ilst))[0]:
            item = ilst[off:off + length]
            spelling = _ilst_spelling(k, item)
            value = _item_data(item) if spelling is not None else None
            if value is not None:
                text = str(int.from_bytes(value, "big")) if spelling == "mp4:tmpo" else value.decode("utf-8", "replace")
                out.setdefault(spelling, []).append(text)
    udta = _child(moov, b"udta")
    if udta is not None:
        for k, off, _h, length in _atoms(udta, _head(udta), len(udta))[0]:
            name = next((n for n, kind in _GPP_TEXT.items() if kind == k), None)
            if name is not None:
                out.setdefault("mp4:" + name, []).append(_gpp_split(udta[off:off + length])[1])
    return out


# --- the cover picture (read only) --------------------------------------------------------------------------

_FRONT_COVER = 3   # the picture type of a front cover (ID3 APIC, FLAC PICTURE)


def _flac_picture(path: Path) -> Optional[bytes]:
    """A FLAC PICTURE block's image: the front cover's, else the first's."""
    pictures = []
    with open(path, "rb") as f:
        _start, blocks = _flac_layout(f, path.stat().st_size)
        for kind, offset, length in blocks:
            if kind != 6:
                continue
            f.seek(offset)
            block = f.read(length)
            ptype, mime_len = struct.unpack_from(">II", block, 0)
            at = 8 + mime_len
            desc_len = struct.unpack_from(">I", block, at)[0]
            at += 4 + desc_len + 16   # the description, then width, height, depth and colours
            data_len = struct.unpack_from(">I", block, at)[0]
            data = block[at + 4:at + 4 + data_len]
            if len(data) != data_len:
                raise _damaged()
            pictures.append((ptype, data))
    return next((d for t, d in pictures if t == _FRONT_COVER), pictures[0][1] if pictures else None)


def _id3_picture(tags: list[bytes]) -> Optional[bytes]:
    """An APIC frame's image in ID3v2 tags: the front cover's, else the first's."""
    pictures = []
    for tag in tags:
        major, frames = _id3_frames(tag)
        for fid, flags, data in frames:
            if fid != b"APIC":
                continue
            content = _id3_frame_data(flags, data, major)
            if not content or content[0] > 3:
                continue
            mime_end = content.find(b"\x00", 1)
            if mime_end < 0 or mime_end + 2 > len(content):
                continue
            ptype = content[mime_end + 1]
            _desc, image = _id3_split(content[mime_end + 2:], content[0])
            if image:
                pictures.append((ptype, image))
    return next((d for t, d in pictures if t == _FRONT_COVER), pictures[0][1] if pictures else None)


def _mp4_picture(path: Path) -> Optional[bytes]:
    ilst = _ilst(_read_moov(path))
    item = _child(ilst, b"covr") if ilst is not None else None
    return _item_data(item) if item is not None else None


def read_picture(path) -> Optional[bytes]:
    """The cover picture's bytes as the file holds them (FLAC PICTURE, ID3 APIC, MP4 covr; the front cover
    first), None when it has none. TagWriteError when the format is not one this module reads, or damaged."""
    path = Path(path)
    fmt = tag_format(path)
    if fmt is None:
        raise TagWriteError(tr("Tags cannot be written to this file's format."))
    try:
        if fmt == "flac":
            return _flac_picture(path)
        if fmt == "mp4":
            return _mp4_picture(path)
        if fmt == "id3":
            with open(path, "rb") as f:
                return _id3_picture(_id3_tags(f))
        id3 = _iff_parts(path, fmt == "aiff")[2]
        return _id3_picture([id3[2][:_id3_size(id3[2][:10])]]) if id3 is not None else None
    except (struct.error, IndexError, ValueError, OverflowError) as exc:
        raise _damaged() from exc


_READERS = {"flac": _read_flac, "id3": _read_id3_file, "wav": _read_wav, "aiff": _read_aiff, "mp4": _read_mp4}
_PLANS = {"flac": _plan_flac, "id3": _plan_id3, "wav": _plan_wav, "aiff": _plan_aiff, "mp4": _plan_mp4}
