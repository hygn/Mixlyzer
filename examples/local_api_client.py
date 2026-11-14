"""A client of Mixlyzer's local API (docs/local_api.md), the standard library's alone: it pairs with Mixlyzer, then
asks for a track's analysis and writes its JAMS (JAMS-mxl-sbx1: examples/jams_mxl_sbx1.py reads it).

Mixlyzer first: Settings > Library > Local Apps, "Let apps on this computer ask for analyses" on (and applied), then
Pair an App... before this asks to pair.

    python examples/local_api_client.py pair --name "My Tool" [--write] [--token-file FILE]
    python examples/local_api_client.py analyze TRACK.flac [--save] [--token-file FILE] [--out TRACK.jams]

``pair`` prints the code to type into Mixlyzer and keeps the token in --token-file (default mixlyzer_token.txt; a
secret: whoever holds it asks Mixlyzer as this app). ``analyze`` asks for the file's analysis (the library's at once
when Mixlyzer has the track), polls its progress, and writes the JAMS to --out (else prints it). ``--write`` asks for
the permission to save into Mixlyzer's library, ``--save`` uses it.

As a module: ``Client(url, token).analysis(path)`` returns the JAMS document."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Callable, Optional

DEFAULT_URL = "http://127.0.0.1:48700"
DEFAULT_TOKEN_FILE = "mixlyzer_token.txt"


class ApiError(Exception):
    """Mixlyzer refused a request: its HTTP status, its code (``not_paired``, ``file_not_found`` ...) and message."""

    def __init__(self, status: int, code: str, message: str, payload: Optional[dict] = None) -> None:
        super().__init__(f"{status} {code}: {message}")
        self.status, self.code, self.message, self.payload = status, code, message, payload or {}


class Client:
    def __init__(self, url: str = DEFAULT_URL, token: Optional[str] = None, timeout: float = 30.0) -> None:
        self.url = url.rstrip("/")
        self.token = token
        self.timeout = timeout
        # never through a proxy: Mixlyzer is on this computer
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _request(self, method: str, endpoint: str, body: Optional[dict] = None) -> tuple[int, dict]:
        """(status, answer) of a request; ApiError for a refusal (4xx, 5xx)."""
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(f"{self.url}/api/v1/{endpoint}", data=data, method=method)
        if data is not None:
            request.add_header("Content-Type", "application/json")
        if self.token:
            request.add_header("Authorization", f"Bearer {self.token}")
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            try:
                payload = json.loads(error.read().decode("utf-8"))
            except ValueError:
                payload = {}
            raise ApiError(error.code, str(payload.get("code", "error")), str(payload.get("message", error.reason)),
                           payload) from None

    # ---- pairing ----
    def pair(self, name: str, write: bool = False) -> dict:
        """Ask to be paired: the answer's ``pairing_code`` is shown to the user, who types it into Mixlyzer; its
        ``token`` is this client's from now on (valid once the user has typed the code)."""
        _, answer = self._request("POST", "pair", {"name": name, "scopes": ["read", "write"] if write else ["read"]})
        self.token = answer["token"]
        return answer

    def verify(self) -> dict:
        """The token's pairing: ``state`` "paired" or "pending" (ApiError: refused, its time out, not paired)."""
        return self._request("GET", "verify")[1]

    def wait_paired(self, timeout: float = 130.0, interval: float = 1.0) -> dict:
        """Asked again until the user has typed the code (ApiError when refused or its time is out)."""
        end = time.monotonic() + timeout
        while True:
            answer = self.verify()
            if answer.get("state") == "paired":
                return answer
            if time.monotonic() > end:
                raise TimeoutError("not paired in time")
            time.sleep(interval)

    # ---- analyses ----
    def analyze(self, path: str, save: bool = False) -> dict:
        """The file's analysis asked for: ``state`` "done" with its ``jams``, or an ``analysis_id`` to ask
        `progress` of."""
        return self._request("POST", "analyze", {"path": os.path.abspath(path), "save": bool(save)})[1]

    def progress(self, analysis_id: str) -> dict:
        return self._request("GET", f"progress/{analysis_id}")[1]

    def analysis(self, path: str, save: bool = False, interval: float = 1.0,
                 on_progress: Optional[Callable[[dict], None]] = None) -> dict:
        """The file's JAMS document: the library's at once, else its analysis waited for (ApiError: it failed)."""
        answer = self.analyze(path, save)
        while answer.get("state") != "done":
            if on_progress is not None:
                on_progress(answer)
            time.sleep(interval)
            answer = self.progress(answer["analysis_id"])
        return answer["jams"]


def _write_token(path: str, token: str) -> None:
    """The token kept where only this user reads it (on Linux and macOS; a Windows file takes its folder's)."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="ascii") as f:
        f.write(token + "\n")


def main(argv: Optional[list[str]] = None) -> int:
    for stream in (sys.stdout, sys.stderr):        # a console in another code page prints what it can
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description="A client of Mixlyzer's local API.")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"Mixlyzer's address (default {DEFAULT_URL})")
    parser.add_argument("--token-file", default=DEFAULT_TOKEN_FILE, help="where the token is kept")
    commands = parser.add_subparsers(dest="command", required=True)
    pair = commands.add_parser("pair", help="pair with Mixlyzer")
    pair.add_argument("--name", required=True, help="this app's name as Mixlyzer shows it")
    pair.add_argument("--write", action="store_true", help="ask for the permission to save into the library")
    analyze = commands.add_parser("analyze", help="a track's analysis as JAMS")
    analyze.add_argument("path", help="the audio file")
    analyze.add_argument("--save", action="store_true", help="save a new analysis into Mixlyzer's library")
    analyze.add_argument("--out", help="the JAMS file to write (default: printed)")
    args = parser.parse_args(argv)
    try:
        if args.command == "pair":
            client = Client(args.url)
            answer = client.pair(args.name, write=args.write)
            print(f"Type this code into Mixlyzer: {answer['pairing_code']}", flush=True)
            paired = client.wait_paired(timeout=float(answer.get("expires_in", 120)) + 10)
            _write_token(args.token_file, client.token)
            print(f"Paired as {paired['name']!r} ({', '.join(paired['scopes'])}); token kept in {args.token_file}")
            return 0
        with open(args.token_file, encoding="ascii") as f:
            client = Client(args.url, f.read().strip())
        jams = client.analysis(args.path, save=args.save, on_progress=lambda a: print(
            f"{a.get('state')} {100 * float(a.get('progress') or 0):.0f}%", file=sys.stderr, flush=True))
        text = json.dumps(jams, ensure_ascii=True, indent=2) + "\n"
        if args.out:
            with open(args.out, "w", encoding="ascii") as f:
                f.write(text)
        else:
            sys.stdout.write(text)
        return 0
    except ApiError as error:
        print(f"Mixlyzer refused: {error}", file=sys.stderr)
    except (OSError, TimeoutError) as error:       # not running, not listening, no token file ...
        print(f"error: {error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
