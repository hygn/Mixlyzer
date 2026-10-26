"""Open Sound Control 1.0 packets (opensoundcontrol.org): the messages External Sync receives.

Strict, as a packet comes from outside: every length is checked against the packet, a string must end in its
NUL and padding, bundles nest at most ``MAX_DEPTH`` deep, and only the argument types read here are accepted
(i int32, h int64, f float32, d float64, s string, T / F true / false); a packet holding anything else is
refused as a whole (``OscError``). ``message`` builds one (the tests, a sender's reference)."""

from __future__ import annotations

import struct

MAX_PACKET = 8192   # bytes: a deck's state is a few hundred
MAX_DEPTH = 4
_BUNDLE = b"#bundle\x00"


class OscError(ValueError):
    pass


def _string(data: bytes, at: int) -> tuple[str, int]:
    """The NUL-terminated, 4-byte padded string at ``at`` and where it ends."""
    end = data.find(b"\x00", at)
    if end < 0:
        raise OscError("unterminated string")
    stop = (end + 4) & ~3
    if stop > len(data) or any(data[end:stop]):
        raise OscError("bad string padding")
    try:
        return data[at:end].decode("utf-8"), stop
    except UnicodeDecodeError as exc:
        raise OscError("string not UTF-8") from exc


_SIZES = {"i": (">i", 4), "h": (">q", 8), "f": (">f", 4), "d": (">d", 8)}


def _message(data: bytes) -> tuple[str, list]:
    address, at = _string(data, 0)
    if not address.startswith("/"):
        raise OscError("not an OSC address")
    if at == len(data):   # no type tags: no arguments (OSC 1.0 allows it)
        return address, []
    tags, at = _string(data, at)
    if not tags.startswith(","):
        raise OscError("no type tags")
    args: list = []
    for tag in tags[1:]:
        if tag in _SIZES:
            fmt, size = _SIZES[tag]
            if at + size > len(data):
                raise OscError("argument past the packet")
            args.append(struct.unpack_from(fmt, data, at)[0])
            at += size
        elif tag == "s":
            value, at = _string(data, at)
            args.append(value)
        elif tag in "TF":
            args.append(tag == "T")
        else:
            raise OscError(f"argument type {tag!r} not read")
    if at != len(data):
        raise OscError("bytes after the arguments")
    return address, args


def _elements(data: bytes, depth: int, out: list) -> None:
    if len(data) % 4:
        raise OscError("size not a multiple of 4")
    if not data.startswith(_BUNDLE):
        out.append(_message(data))
        return
    if depth >= MAX_DEPTH:
        raise OscError("bundles nested too deep")
    at = len(_BUNDLE) + 8   # the time tag (not used: the newest state is taken as it comes)
    if at > len(data):
        raise OscError("bundle without its time tag")
    while at < len(data):
        if at + 4 > len(data):
            raise OscError("element size past the packet")
        size = struct.unpack_from(">i", data, at)[0]
        at += 4
        if size <= 0 or at + size > len(data):
            raise OscError("element past the packet")
        _elements(data[at:at + size], depth + 1, out)
        at += size


def parse(packet: bytes) -> list[tuple[str, list]]:
    """The messages of a packet, a bundle's in order: [(address, arguments)]."""
    if not packet or len(packet) > MAX_PACKET:
        raise OscError("empty or oversized packet")
    out: list[tuple[str, list]] = []
    _elements(bytes(packet), 0, out)
    return out


def _padded(raw: bytes) -> bytes:
    return raw + b"\x00" * (4 - len(raw) % 4)


def message(address: str, *args) -> bytes:
    """An OSC message: bool as T / F, int as i (h past 32 bits), float as d, str as s."""
    tags, body = ",", b""
    for arg in args:
        if isinstance(arg, bool):
            tags += "T" if arg else "F"
        elif isinstance(arg, int):
            small = -(1 << 31) <= arg < (1 << 31)
            tags += "i" if small else "h"
            body += struct.pack(">i" if small else ">q", arg)
        elif isinstance(arg, float):
            tags += "d"
            body += struct.pack(">d", arg)
        elif isinstance(arg, str):
            tags += "s"
            body += _padded(arg.encode("utf-8"))
        else:
            raise TypeError(f"not an OSC argument: {type(arg).__name__}")
    return _padded(address.encode("utf-8")) + _padded(tags.encode("ascii")) + body
