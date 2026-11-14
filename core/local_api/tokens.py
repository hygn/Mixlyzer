"""The local API's tokens: what an app paired with Mixlyzer sends as its bearer token (``Authorization: Bearer``).

A token carries its permissions itself, bound by its hash: ``mxl1.<payload>.<secret>``, the payload the base64url
(no padding) of the compact JSON ``{"id": ..., "scopes": [...]}``, the secret 32 random bytes in base64url. Mixlyzer
keeps only the SHA-256 of the whole token (core.local_api.pairings), found by its id: a token whose payload is
changed (a scope added) hashes otherwise and is refused, so the scopes a token grants are the ones it was issued with,
and a token is never made from what Mixlyzer keeps. More permissions are a new token, paired again.

Scopes: READ, an analysis asked and answered (a library track's stored analysis included); WRITE, a new analysis
stored in the library."""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
import secrets
from dataclasses import dataclass
from typing import Iterable, Optional

PREFIX = "mxl1"
READ, WRITE = "read", "write"
SCOPES = (READ, WRITE)
MAX_TOKEN_CHARS = 256          # longer: refused unread
SECRET_BYTES = 32


@dataclass(frozen=True)
class Issued:
    """A new token and what Mixlyzer keeps of it."""
    token: str
    token_id: str
    scopes: tuple[str, ...]
    sha256: str


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def normalized_scopes(scopes: Iterable[str]) -> tuple[str, ...]:
    """The scopes in their order (SCOPES), each once; ValueError for one not known."""
    wanted = set()
    for scope in scopes:
        if scope not in SCOPES:
            raise ValueError(f"unknown scope: {scope!r}")
        wanted.add(scope)
    return tuple(s for s in SCOPES if s in wanted)


def digest(token: str) -> str:
    """The SHA-256 (hex) Mixlyzer keeps of a token."""
    return hashlib.sha256(token.encode("ascii", "replace")).hexdigest()


def issue(scopes: Iterable[str]) -> Issued:
    """A new token granting ``scopes``."""
    kept = normalized_scopes(scopes)
    token_id = secrets.token_hex(8)
    payload = json.dumps({"id": token_id, "scopes": list(kept)}, separators=(",", ":"), sort_keys=True)
    token = f"{PREFIX}.{_b64(payload.encode('ascii'))}.{_b64(secrets.token_bytes(SECRET_BYTES))}"
    return Issued(token, token_id, kept, digest(token))


def parse(token: str) -> Optional[tuple[str, tuple[str, ...]]]:
    """A token's id and scopes as it says them (not yet checked against what Mixlyzer keeps: `digest`); None when
    it is not a token of this form."""
    if not isinstance(token, str) or len(token) > MAX_TOKEN_CHARS or not token.isascii():
        return None
    parts = token.split(".")
    if len(parts) != 3 or parts[0] != PREFIX:
        return None
    try:
        payload = json.loads(_unb64(parts[1]).decode("ascii"))
        secret = _unb64(parts[2])
    except (ValueError, UnicodeDecodeError, binascii.Error):
        return None
    if len(secret) != SECRET_BYTES or not isinstance(payload, dict) or set(payload) != {"id", "scopes"}:
        return None
    token_id, scopes = payload["id"], payload["scopes"]
    if not isinstance(token_id, str) or not token_id or len(token_id) > 64 or not token_id.isalnum():
        return None
    if not isinstance(scopes, list) or not all(isinstance(s, str) for s in scopes):
        return None
    try:
        kept = normalized_scopes(scopes)
    except ValueError:
        return None
    if list(kept) != scopes:          # as issued: in their order, each once
        return None
    return token_id, kept
