"""The local API's HTTP server: on 127.0.0.1 alone (never another interface), on a thread of its own, a thread per
connection (MAX_CONNECTIONS at once; more are closed unanswered). Before a route sees a request (core.local_api.
protocol): the peer is this computer, the Host header names it at this port (a page whose site's name is made to point
here, DNS rebinding, names its own), no browser's headers (a web page's request), a POST's body a JSON object of
MAX_BODY_BYTES at most with its length given. On Windows the port is the server's alone (SO_EXCLUSIVEADDRUSE: another
program cannot listen on it beside the server and take its requests); a port taken refuses the server (OSError)."""
from __future__ import annotations

import http.server
import ipaddress
import json
import socket
import socketserver
import threading
from typing import Callable, Optional
from urllib.parse import urlsplit

from core.local_api.protocol import (MAX_BODY_BYTES, ApiError, Request, Response, answer, from_browser, host_allowed,
                                     json_media_type)
from system.os_support import IS_WINDOWS

HOST = "127.0.0.1"
MAX_CONNECTIONS = 16
SOCKET_TIMEOUT_SEC = 15.0        # a client silent this long is let go

Route = Callable[[Request], Response]


class _Server(http.server.ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False
    allow_reuse_address = False      # on Windows SO_REUSEADDR would let two programs listen on one port

    def __init__(self, address, route: Route) -> None:
        self.route = route
        self._slots = threading.BoundedSemaphore(MAX_CONNECTIONS)
        super().__init__(address, _Handler)

    def server_bind(self) -> None:
        if IS_WINDOWS and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        socketserver.TCPServer.server_bind(self)     # not HTTPServer's: its host name lookup is not needed
        self.server_name, self.server_port = self.server_address[:2]

    def verify_request(self, request, client_address) -> bool:
        try:
            return ipaddress.ip_address(client_address[0]).is_loopback
        except ValueError:
            return False

    def process_request(self, request, client_address) -> None:
        if not self._slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self._slots.release()
            raise

    def process_request_thread(self, request, client_address) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._slots.release()


class _Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "Mixlyzer"
    sys_version = ""
    timeout = SOCKET_TIMEOUT_SEC

    def do_GET(self) -> None:
        self._handle()

    do_POST = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = do_GET

    def _handle(self) -> None:
        try:
            response = self._respond()
        except ApiError as exc:
            response = exc.response()
        except Exception as exc:
            print(f"[Local API] {self.command} {urlsplit(self.path).path}: {exc!r}")
            response = Response(500, answer(500, "internal_error", "Mixlyzer could not answer the request."))
        self._send(response.status, response.payload, response.headers)

    def _refuse(self, status: int, code: str, message: str) -> ApiError:
        self.close_connection = True       # its body, if any, not read
        return ApiError(status, code, message)

    def _respond(self) -> Response:
        headers = {name.lower(): value for name, value in self.headers.items()}
        raw = self._read_body(headers)     # read before any refusal: a socket closed on unread bytes resets
        if not host_allowed(headers.get("host", ""), self.server.server_address[1]):
            raise self._refuse(403, "forbidden_host", "The Host header must name this computer (127.0.0.1) and "
                                                      "this port.")
        if from_browser(headers):
            raise self._refuse(403, "browser_request", "Requests from web pages are refused.")
        body = self._body(headers, raw)
        return self.server.route(Request(self.command, urlsplit(self.path).path, headers, body))

    def _read_body(self, headers: dict) -> Optional[bytes]:
        """The body as sent (None: none said), MAX_BODY_BYTES at most."""
        if "transfer-encoding" in headers:
            raise self._refuse(411, "length_required", "Send the body with a Content-Length.")
        length_text = headers.get("content-length")
        if length_text is None:
            return None
        try:
            length = int(length_text)
            if length < 0:
                raise ValueError(length_text)
        except ValueError:
            raise self._refuse(400, "bad_request", "Content-Length: a number of bytes.") from None
        if length > MAX_BODY_BYTES:
            raise self._refuse(413, "too_large", f"The body is larger than {MAX_BODY_BYTES} bytes.")
        raw = self.rfile.read(length)
        if len(raw) != length:
            raise self._refuse(400, "bad_request", "The body is shorter than its Content-Length.")
        return raw

    def _body(self, headers: dict, raw: Optional[bytes]) -> Optional[dict]:
        if self.command != "POST":
            if raw:
                raise ApiError(400, "bad_request", "Only a POST takes a body.")
            return None
        if raw is None:
            raise self._refuse(411, "length_required", "Send the body with a Content-Length.")
        if not json_media_type(headers.get("content-type", "")):
            raise ApiError(415, "unsupported_media_type", "The body must be application/json.")
        try:
            body = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError, RecursionError):
            raise ApiError(400, "invalid_json", "The body is not JSON in UTF-8.") from None
        if not isinstance(body, dict):
            raise ApiError(400, "invalid_json", "The body must be a JSON object.")
        return body

    def _send(self, status: int, payload: dict, headers=()) -> None:
        data = (json.dumps(payload, ensure_ascii=True, separators=(",", ":")) + "\n").encode("ascii")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for name, value in headers:
            self.send_header(name, value)
        if self.close_connection:
            self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(data)

    def send_error(self, code, message=None, explain=None) -> None:
        """A request the HTTP layer refuses itself (a malformed line, headers too long): as JSON too."""
        self.close_connection = True
        try:
            short = message or self.responses[code][0]
        except KeyError:
            short = "Error"
        self._send(int(code), answer(int(code), "bad_request" if int(code) < 500 else "internal_error", short))

    def log_message(self, format, *args) -> None:    # nothing logged per request (core.local_api.service says errors)
        pass


class LocalApiServer:
    """The server at ``port`` on 127.0.0.1 (0: one the system picks, for tests), each request to ``route``."""

    def __init__(self, port: int, route: Route) -> None:
        self._httpd = _Server((HOST, int(port)), route)     # OSError: the port taken or not allowed
        self._thread = threading.Thread(target=self._httpd.serve_forever, kwargs={"poll_interval": 0.25},
                                        name="LocalApi", daemon=True)

    @property
    def port(self) -> int:
        return int(self._httpd.server_address[1])

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        """No more requests taken; those being answered finish on their threads."""
        if self._thread.is_alive():
            self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=2.0)
