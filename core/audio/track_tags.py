"""A track file's tags as the library reads them: title / artist / album / comment (and where the file holds each, so
the Export tab writes them back there), and its cover picture. This app's own reader for the formats it knows
(core.audio.tag_writer), FFmpeg's for the others (analyzer.audio.decoder)."""
import os
import struct
from pathlib import Path
from typing import Optional

from PySide6 import QtGui

from analyzer.audio.decoder import COVER_MAX_BYTES, read_cover, read_metadata
from core.audio.tag_writer import TagWriteError, read_fields, read_picture, tag_format, tag_sources


_RIFF_INFO_FIELDS = {
    b"INAM": "title",
    b"IART": "artist",
    b"IPRD": "album",
    b"ICMT": "comment",
}


def _utf8_wav_info(path: str) -> dict[str, str]:
    """Read UTF-8 RIFF INFO tags without applying the Windows ANSI code page.

    Readers such as MediaInfo interpret RIFF INFO text using the active Windows code page.
    Recordings written by rekordbox/PioneerDJ contain UTF-8 here, which turns
    Japanese text into mojibake on a Korean Windows installation.
    """
    if Path(path).suffix.lower() not in (".wav", ".wave"):
        return {}
    result: dict[str, str] = {}
    try:
        with open(path, "rb") as stream:
            header = stream.read(12)
            if len(header) != 12 or header[:4] != b"RIFF" or header[8:] != b"WAVE":
                return result
            riff_end = min(
                os.fstat(stream.fileno()).st_size,
                8 + struct.unpack("<I", header[4:8])[0],
            )
            while stream.tell() + 8 <= riff_end:
                chunk_id = stream.read(4)
                size_data = stream.read(4)
                if len(size_data) != 4:
                    break
                chunk_size = struct.unpack("<I", size_data)[0]
                chunk_start = stream.tell()
                chunk_end = chunk_start + chunk_size
                if chunk_end > riff_end:
                    break
                if chunk_id == b"LIST" and chunk_size >= 4 and stream.read(4) == b"INFO":
                    while stream.tell() + 8 <= chunk_end:
                        info_id = stream.read(4)
                        info_size_data = stream.read(4)
                        if len(info_size_data) != 4:
                            break
                        info_size = struct.unpack("<I", info_size_data)[0]
                        info_end = stream.tell() + info_size
                        if info_end > chunk_end:
                            break
                        raw = stream.read(info_size).rstrip(b"\x00")
                        field = _RIFF_INFO_FIELDS.get(info_id)
                        if field and raw:
                            try:
                                value = raw.decode("utf-8-sig").strip()
                            except UnicodeDecodeError:
                                value = ""
                            if value:
                                result[field] = value
                        stream.seek(info_end + (info_size & 1))
                stream.seek(chunk_end + (chunk_size & 1))
    except (OSError, struct.error):
        return {}
    return result


def _file_fields(path: str) -> dict[str, str]:
    """Title / artist / album / comment as the file holds them: this app's own reader for the formats it knows
    (core.audio.tag_writer, MediaInfo's order), else FFmpeg's (Ogg, Opus, WMA ...); no native tag library parses
    the file (an outside file: a crafted tag cannot reach memory-unsafe code beyond FFmpeg, which decodes it
    anyway)."""
    if tag_format(path) is not None:
        try:
            return read_fields(path)
        except TagWriteError:
            pass   # damaged for this reader: as FFmpeg reads it
    meta = read_metadata(path)

    def pick(*keys: str) -> str:
        return next((meta[k].strip() for k in keys if meta.get(k, "").strip()), "")

    return {"title": pick("title"), "artist": pick("artist", "performer", "album_artist"),
            "album": pick("album"), "comment": pick("comment", "description")}


def extract_tags(path: str) -> tuple[str, str, str, str]:
    """(title, artist, album, comment) of the file; the title its file name when it has none."""
    title = os.path.splitext(os.path.basename(path))[0]
    try:
        wav_info = _utf8_wav_info(path)
        fields = _file_fields(path)
    except Exception:
        return title, "", "", ""
    return tuple(wav_info.get(f) or fields.get(f) or (title if f == "title" else "")
                 for f in ("title", "artist", "album", "comment"))


def read_tag_sources(path: str, title: str, artist: str, album: str, comment: str) -> dict:
    """Where the file holds the values read (``extract_tags``): field -> tag_writer spellings ({}: a file whose
    tags are not written)."""
    read = {"title": title, "artist": artist, "album": album, "comment": comment}
    try:
        return tag_sources(path, read) if tag_format(path) is not None else {}
    except TagWriteError:
        return {}


# the picture formats a cover is read in (by its first bytes): named to Qt, so no other image reader sees it
_COVER_FORMATS = ((b"\xff\xd8\xff", "JPEG"), (b"\x89PNG\r\n\x1a\n", "PNG"), (b"GIF8", "GIF"), (b"BM", "BMP"))


def cover_data(path) -> Optional[tuple[bytes, str]]:
    """The file's cover picture as stored (its bytes and their format: JPEG, PNG, GIF, BMP or WebP), None when it has
    none: read as the tags are (``_file_fields``), this app's reader first, else FFmpeg's."""
    if not path:
        return None
    try:
        data = read_picture(path)
    except TagWriteError:
        data = read_cover(str(path))
    except OSError:
        return None
    if not data or len(data) > COVER_MAX_BYTES:
        return None
    fmt = next((name for magic, name in _COVER_FORMATS if data.startswith(magic)), None)
    if fmt is None and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        fmt = "WEBP"
    return (bytes(data), fmt) if fmt is not None else None


def getAlbumArt(path) -> Optional[QtGui.QImage]:
    """The file's cover picture (cover_data's, decoded), None when it has none."""
    cover = cover_data(path)
    if cover is None:
        return None
    img = QtGui.QImage.fromData(*cover)
    return None if img.isNull() else img
