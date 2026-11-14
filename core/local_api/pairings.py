"""The apps paired with the local API and the pairing itself: one holder (`PairingStore`, its own lock: the server's
threads ask it, the settings page acts on it).

The paired apps are kept in PAIRINGS_FILE in the app's folder (beside config.json: this device's, never in a library
backup): each app's name as it gave it, the scopes its token grants (shown; the token itself is what grants them:
core.local_api.tokens), the token's SHA-256, when it was paired and last used. A file that does not read is moved aside
(``pairings.json.unreadable-<time>``) and the apps pair again.

Pairing, a state machine (`Phase`): CLOSED; the user opens it (Settings > Library > Local Apps: `open`, for
PAIRING_SEC) -> OPEN; an app asks (`ask`: its name and the scopes it wants) and gets a 4-digit code and its token, not
valid yet -> ASKED (one app at a time: another asking meanwhile is refused); the app shows the code, the user types it
into Mixlyzer (`enter`): right, the token is kept and valid; wrong CODE_TRIES times, or refused (`refuse`), or its time
run out, it is not -> CLOSED (one answer per opening). The app asks whether it is paired (`check`, the verify endpoint)
until it is; the outcome of its token is kept OUTCOME_KEPT_SEC for that."""
from __future__ import annotations

import enum
import hmac
import json
import os
import secrets
import threading
import time
import unicodedata
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Callable, Optional

from core.local_api import tokens
from system.atomic_io import atomic_write_json

PAIRINGS_FILE = "pairings.json"
FORMAT = 1
PAIRING_SEC = 120.0          # how long a pairing opened waits for an app, and an app's code for the user
CODE_TRIES = 3               # wrong codes typed before the app is refused
OUTCOME_KEPT_SEC = 600.0     # how long an app asking whether it is paired is told how its pairing ended
LAST_USED_WRITE_SEC = 60.0   # a token's last use written at most this often
MAX_NAME_CHARS = 64


class Phase(enum.Enum):
    CLOSED = "closed"
    OPEN = "open"            # waiting for an app to ask
    ASKED = "asked"          # an app asked: waiting for the user to type its code


class Verdict(enum.Enum):
    """What a token is to the pairing (`check`)."""
    PAIRED = "paired"
    PENDING = "pending"      # its app asked: the user has not typed its code yet
    DENIED = "denied"        # refused, or the code typed wrong CODE_TRIES times
    EXPIRED = "expired"      # the pairing's time ran out
    UNKNOWN = "unknown"      # never paired, unpaired, or not a token


class Refused(Exception):
    """An app's asking refused: ``code`` the API's error code (pairing_closed, pairing_busy, bad_request)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Pairing:
    """A paired app."""
    token_id: str
    name: str
    scopes: tuple[str, ...]      # what its token grants, as shown (the token's own are what is checked)
    sha256: str
    paired: str                  # local time, ISO 8601 to the second
    last_used: str = ""


@dataclass(frozen=True)
class Asking:
    """An app asking to be paired."""
    name: str
    scopes: tuple[str, ...]
    code: str
    issued: tokens.Issued
    until: float                 # the clock's time its code is no longer taken
    tries_left: int = CODE_TRIES


@dataclass(frozen=True)
class Grant:
    """What a valid token grants (its own scopes)."""
    token_id: str
    name: str
    scopes: tuple[str, ...]


@dataclass(frozen=True)
class PairingView:
    """The pairing as the settings page shows it."""
    phase: Phase
    seconds_left: float = 0.0
    asking: Optional[Asking] = None
    last: Optional[tuple[Verdict, str]] = None   # how the last pairing ended, and its app's name


def clean_name(name) -> str:
    """An app's name as kept and shown: its control and format characters left out, its spaces collapsed;
    ValueError when nothing is left or it is longer than MAX_NAME_CHARS."""
    if not isinstance(name, str):
        raise ValueError("name: a string")
    kept = "".join(" " if c.isspace() else c for c in name
                   if c.isspace() or unicodedata.category(c) not in ("Cc", "Cf", "Cs", "Co", "Cn"))
    kept = " ".join(kept.split())
    if not kept or len(kept) > MAX_NAME_CHARS:
        raise ValueError(f"name: 1 to {MAX_NAME_CHARS} characters")
    return kept


class PairingStore:
    def __init__(self, path: str = PAIRINGS_FILE, *, clock: Callable[[], float] = time.monotonic,
                 on_change: Optional[Callable[[], None]] = None) -> None:
        self._path = str(path)
        self._clock = clock
        self.on_change = on_change                               # told of every change (any thread)
        self._lock = threading.RLock()
        self._pairings: dict[str, Pairing] = {}
        self._phase = Phase.CLOSED
        self._until = 0.0
        self._asking: Optional[Asking] = None
        self._outcomes: dict[str, tuple[Verdict, float]] = {}   # an asked token's hash -> how it ended, kept until
        self._last: Optional[tuple[Verdict, str]] = None
        self._last_written = 0.0
        self._unwritten = False                                  # a last use not written yet
        self.problem = ""                                        # the file's, when it did not read

    # ---- the file ----
    def load(self) -> None:
        """The paired apps from the file (none: no file). One that does not read is moved aside (``problem``
        says so) and none are paired."""
        with self._lock:
            self._pairings = {}
            self.problem = ""
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if not isinstance(loaded, dict) or loaded.get("format") != FORMAT \
                        or not isinstance(loaded.get("pairings"), list):
                    raise ValueError("not the paired apps' file")
                for entry in loaded["pairings"]:
                    pairing = _pairing_of(entry)
                    if pairing is not None:
                        self._pairings[pairing.token_id] = pairing
            except FileNotFoundError:
                return
            except (OSError, ValueError, UnicodeDecodeError) as exc:
                aside = f"{self._path}.unreadable-{time.strftime('%Y%m%d-%H%M%S')}"
                try:
                    os.replace(self._path, aside)
                except OSError:
                    aside = ""
                self.problem = f"{exc}" + (f" (kept as {os.path.basename(aside)})" if aside else "")
                print(f"[Local API] the paired apps' file could not be read: {self.problem}")

    def _save(self) -> None:
        payload = {"format": FORMAT, "pairings": [
            {"id": p.token_id, "name": p.name, "scopes": list(p.scopes), "sha256": p.sha256, "paired": p.paired,
             "last_used": p.last_used} for p in sorted(self._pairings.values(), key=lambda p: p.paired)]}
        try:
            atomic_write_json(self._path, payload, ensure_ascii=True, indent=1)
            self._last_written = self._clock()
            self._unwritten = False
        except OSError as exc:
            print(f"[Local API] the paired apps' file could not be written: {exc}")

    def flush(self) -> None:
        """The last uses not written yet written (the app quitting)."""
        with self._lock:
            if self._unwritten:
                self._save()

    # ---- the paired apps ----
    def pairings(self) -> list[Pairing]:
        with self._lock:
            return sorted(self._pairings.values(), key=lambda p: p.paired)

    def unpair(self, token_id: str) -> bool:
        with self._lock:
            if self._pairings.pop(token_id, None) is None:
                return False
            self._save()
        self._changed()
        return True

    def authenticate(self, token: str) -> Optional[Grant]:
        """What a paired app's token grants (its own scopes), its use noted; None for any other token."""
        parsed = tokens.parse(token)
        if parsed is None:
            return None
        token_id, scopes = parsed
        with self._lock:
            pairing = self._pairings.get(token_id)
            if pairing is None or not hmac.compare_digest(tokens.digest(token), pairing.sha256):
                return None
            self._pairings[token_id] = replace(pairing, last_used=_now())
            self._unwritten = True
            if self._clock() - self._last_written >= LAST_USED_WRITE_SEC:
                self._save()
            return Grant(token_id, pairing.name, scopes)

    # ---- pairing ----
    def view(self) -> PairingView:
        with self._lock:
            self._expire()
            left = max(0.0, (self._asking.until if self._asking else self._until) - self._clock())
            return PairingView(self._phase, left if self._phase is not Phase.CLOSED else 0.0, self._asking,
                               self._last)

    def open(self, seconds: float = PAIRING_SEC) -> None:
        """Pairing opened for an app to ask (again: its time started again; an app asking stays asked)."""
        with self._lock:
            self._expire()
            if self._phase is Phase.CLOSED:
                self._phase = Phase.OPEN
                self._last = None
            if self._phase is Phase.OPEN:
                self._until = self._clock() + float(seconds)
        self._changed()

    def close(self) -> None:
        """Pairing closed by the user: an app asking is refused."""
        with self._lock:
            if self._phase is Phase.ASKED:
                self._end(Verdict.DENIED)
            self._phase = Phase.CLOSED
        self._changed()

    def ask(self, name, scopes) -> Asking:
        """An app asking to be paired: its code and its token (valid once the user types the code)."""
        try:
            name = clean_name(name)
            scopes = tokens.normalized_scopes(scopes)
        except (ValueError, TypeError) as exc:
            raise Refused("bad_request", str(exc)) from None
        if tokens.READ not in scopes:
            raise Refused("bad_request", "scopes: read is needed (write alone grants nothing)")
        with self._lock:
            self._expire()
            if self._phase is Phase.CLOSED:
                raise Refused("pairing_closed", "Pairing is not open: open it in Mixlyzer's settings first.")
            if self._phase is Phase.ASKED:
                raise Refused("pairing_busy", "Another app is being paired: try again in a moment.")
            self._asking = Asking(name, scopes, f"{secrets.randbelow(10000):04d}", tokens.issue(scopes),
                                  self._clock() + PAIRING_SEC)
            self._phase = Phase.ASKED
            asking = self._asking
        self._changed()
        return asking

    def enter(self, code: str) -> Verdict:
        """The code the user typed for the app asking: PAIRED (its token kept, valid), PENDING (wrong: tries left),
        DENIED (wrong for the last time), EXPIRED / UNKNOWN (no app asking)."""
        with self._lock:
            self._expire()
            asking = self._asking
            if self._phase is not Phase.ASKED or asking is None:
                return self._last[0] if self._last else Verdict.UNKNOWN
            typed = "".join(str(code or "").split())
            if hmac.compare_digest(typed.encode("ascii", "replace"), asking.code.encode("ascii")):
                issued = asking.issued
                self._pairings[issued.token_id] = Pairing(issued.token_id, asking.name, issued.scopes,
                                                          issued.sha256, _now())
                self._save()
                verdict = self._end(Verdict.PAIRED)
            elif asking.tries_left > 1:
                self._asking = replace(asking, tries_left=asking.tries_left - 1)
                verdict = Verdict.PENDING
            else:
                verdict = self._end(Verdict.DENIED)
        self._changed()
        return verdict

    def refuse(self) -> None:
        """The app asking refused by the user."""
        with self._lock:
            if self._phase is Phase.ASKED:
                self._end(Verdict.DENIED)
        self._changed()

    def check(self, token: str) -> tuple[Verdict, Optional[Grant]]:
        """What ``token`` is to the pairing, and what it grants when PAIRED (the verify endpoint)."""
        grant = self.authenticate(token)
        if grant is not None:
            return Verdict.PAIRED, grant
        with self._lock:
            self._expire()
            hashed = tokens.digest(token) if isinstance(token, str) else ""
            if self._asking is not None and hmac.compare_digest(hashed, self._asking.issued.sha256):
                return Verdict.PENDING, None
            outcome = self._outcomes.get(hashed)
            if outcome is not None and outcome[1] > self._clock():
                return outcome[0], None
        return Verdict.UNKNOWN, None

    def expire_due(self) -> bool:
        """The pairing's time checked (the settings page's timer): True when it just ended."""
        with self._lock:
            ended = self._expire()
        if ended:
            self._changed()
        return ended

    # ---- inside ----
    def _expire(self) -> bool:
        now = self._clock()
        self._outcomes = {k: v for k, v in self._outcomes.items() if v[1] > now}
        if self._phase is Phase.ASKED and self._asking is not None and now >= self._asking.until:
            self._end(Verdict.EXPIRED)
            return True
        if self._phase is Phase.OPEN and now >= self._until:
            self._phase = Phase.CLOSED
            self._last = (Verdict.EXPIRED, "")
            return True
        return False

    def _end(self, verdict: Verdict) -> Verdict:
        """The app asking answered: the pairing closed, its outcome kept for its asking."""
        asking = self._asking
        if asking is not None:
            self._outcomes[asking.issued.sha256] = (verdict, self._clock() + OUTCOME_KEPT_SEC)
            self._last = (verdict, asking.name)
        self._asking = None
        self._phase = Phase.CLOSED
        return verdict

    def _changed(self) -> None:
        if self.on_change is not None:
            try:
                self.on_change()
            except Exception as exc:
                print(f"[Local API] {exc}")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _pairing_of(entry) -> Optional[Pairing]:
    """A paired app as the file keeps it; None for an entry that is not one."""
    if not isinstance(entry, dict):
        return None
    try:
        token_id, sha256 = entry["id"], entry["sha256"]
        if not (isinstance(token_id, str) and token_id.isalnum() and isinstance(sha256, str) and len(sha256) == 64
                and all(c in "0123456789abcdef" for c in sha256)):
            return None
        scopes = tokens.normalized_scopes(entry.get("scopes") or ())
        return Pairing(token_id, clean_name(entry.get("name")), scopes, sha256, str(entry.get("paired") or ""),
                       str(entry.get("last_used") or ""))
    except (KeyError, ValueError, TypeError):
        return None
