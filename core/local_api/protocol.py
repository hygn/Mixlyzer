"""The local API's HTTP as a request and an answer: what the server hands the routes (`Request`), what they answer
(`Response`, or an `ApiError`), every answer one JSON object ``{"status", "code", "message", ...}``: ``status`` the
HTTP status, ``code`` a fixed word a program reads (ok, accepted, in_progress, file_not_found, not_paired, ...),
``message`` a sentence in English for a person (the API's texts are not translated: another program shows them, in
its own language if it likes, by the code)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional

PREFIX = "/api/v1/"
MAX_BODY_BYTES = 64 * 1024
# headers only a browser sends: a request with one is a web page's, refused (never asked to pair, never answered)
BROWSER_HEADERS = ("origin", "sec-fetch-site", "sec-fetch-mode", "sec-fetch-dest", "sec-fetch-user")
LOCAL_HOSTS = ("127.0.0.1", "localhost", "[::1]")


@dataclass(frozen=True)
class Request:
    method: str
    path: str                                 # without the query
    headers: Mapping[str, str]                # names in lower case
    body: Optional[dict] = None               # a POST's JSON object

    def bearer(self) -> Optional[str]:
        """The ``Authorization: Bearer`` token, None when there is none."""
        value = self.headers.get("authorization", "")
        scheme, _, token = value.strip().partition(" ")
        if scheme.lower() != "bearer":
            return None
        return token.strip() or None


@dataclass(frozen=True)
class Response:
    status: int
    payload: dict
    headers: tuple[tuple[str, str], ...] = field(default=())


class ApiError(Exception):
    """An answer refusing the request."""

    def __init__(self, status: int, code: str, message: str, *, headers: tuple = (), **fields) -> None:
        super().__init__(message)
        self.status, self.code, self.message = int(status), code, message
        self.headers = tuple(headers)
        self.fields = fields

    def response(self) -> Response:
        return Response(self.status, answer(self.status, self.code, self.message, **self.fields), self.headers)


def answer(status: int, code: str, message: str, **fields) -> dict:
    return {"status": int(status), "code": code, "message": message, **fields}


def host_allowed(host: str, port: int) -> bool:
    """The Host header names this computer at the server's port (a page of another site whose name is made to point
    here, DNS rebinding, names its own host)."""
    return str(host or "").strip().lower() in {f"{h}:{int(port)}" for h in LOCAL_HOSTS}


def from_browser(headers: Mapping[str, str]) -> bool:
    return any(name in headers for name in BROWSER_HEADERS)


def json_media_type(content_type: str) -> bool:
    """``application/json``, with parameters or not (``; charset=utf-8``)."""
    return str(content_type or "").split(";", 1)[0].strip().lower() == "application/json"
