"""The local API over HTTP on localhost (core.local_api, the server on a port the system picks), with a stand-in for
the analysis workflow and the library: pairing (closed, the code typed, refused), the requests refused before any
route (another Host, a browser's headers, a body not JSON or too large or without its length, another method), the
tokens (none, not paired, a scope added, a scope missing), the files taken (a full local path to a file the app takes:
no network path), a library track's JAMS at once, an analysis's progress and its JAMS (not stored: no uid), a failed
one, the port taken; the example client (examples/local_api_client.py) against it."""
import http.client
import json
import os
import socket
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
from PySide6 import QtCore  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from analyzer.audio.decoder import AudioPacketHash  # noqa: E402
from analyzer.music.marks import CuePoint  # noqa: E402
from core.analysis_library.stored import StoredAnalysis  # noqa: E402
from core.local_api import service as service_module  # noqa: E402
from core.local_api import tokens  # noqa: E402
from core.local_api.pairings import PairingStore  # noqa: E402
from core.local_api.service import LocalApi  # noqa: E402
from core.workflows.analysis import AnalysisState  # noqa: E402
from examples import local_api_client  # noqa: E402

APP = QApplication.instance() or QApplication([])
AUDIO = AudioPacketHash("cd" * 32, 4321, "flac", 44100, "stereo", "n7.1-test")
ANALYSIS = StoredAnalysis(duration_sec=2.0, beats=np.asarray([0.5, 1.0, 1.5]),
                          tempo_segments=np.asarray([[0.0, 2.0, 120.0, 0.5, 4.0, 120.0]]),
                          cue_points=(CuePoint(0.5, "A", "", 0),))


class _Workflow(QtCore.QObject):
    """The analysis workflow's stand-in: each request an analysis id, its state as the test sets it."""
    ended = QtCore.Signal(str, object, str)

    def __init__(self):
        super().__init__()
        self.requests, self.states, self.library_holds = [], {}, set()

    def request(self, path, *, store):
        if path in self.library_holds:
            return None
        analysis_id = f"{len(self.requests) + 1:032x}"
        self.requests.append((path, store, analysis_id))
        self.states[analysis_id] = AnalysisState(analysis_id, path, "waiting", 0.0, 1)
        return analysis_id

    def state(self, analysis_id):
        return self.states.get(analysis_id)


class _Library:
    """The library's stand-in: a snapshot finding the tracks given, their stored analysis by `call`."""

    def __init__(self):
        self.tracks = {}
        self.snapshot = SimpleNamespace(find_file=lambda path: self.tracks.get(os.path.normcase(path)))

    def call(self, job, uid):
        assert threading.current_thread() is not threading.main_thread()
        return ANALYSIS if uid == "uid-1" else None


def forged(token: str, scopes: list) -> str:
    """``token`` with its payload's scopes changed (its secret kept)."""
    prefix, _payload, secret = token.split(".")
    payload = json.dumps({"id": tokens.parse(token)[0], "scopes": scopes}, separators=(",", ":"), sort_keys=True)
    return f"{prefix}.{tokens._b64(payload.encode('ascii'))}.{secret}"


def wait_until(condition, timeout=5.0):
    end = time.monotonic() + timeout
    while not condition():
        APP.processEvents()
        if time.monotonic() > end:
            raise AssertionError("timed out")
        time.sleep(0.005)


class LocalApiTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.audio = os.path.join(self.dir.name, "song.flac")
        with open(self.audio, "wb") as f:
            f.write(b"fLaC" + bytes(64))
        self.text = os.path.join(self.dir.name, "notes.txt")
        with open(self.text, "w") as f:
            f.write("x")
        hashed = mock.patch.object(service_module, "audio_packet_hash", lambda path: AUDIO)
        hashed.start()
        self.addCleanup(hashed.stop)
        self.workflow = _Workflow()
        self.library = _Library()
        self.store = PairingStore(os.path.join(self.dir.name, "pairings.json"))
        self.api = LocalApi(library=self.library, workflow=self.workflow, store=self.store)
        self.api.apply(SimpleNamespace(libconfig=SimpleNamespace(local_api=True, local_api_port=0)))
        self.assertTrue(self.api.status().listening, self.api.status().error)
        self.port = self.api.status().port

    def tearDown(self):
        self.api.shutdown()
        self.dir.cleanup()

    # ---- a request on a thread of its own, the GUI thread's events handled meanwhile ----
    def call(self, method, endpoint, body=None, token=None, headers=None, raw=None, headers_only=False):
        """(status, answer, headers) of a request; ``headers_only``: its headers sent alone (a body it says it has
        never sent: the server refuses it before reading one)."""
        box = {}

        def run():
            con = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
            sent = {"Host": f"127.0.0.1:{self.port}"}
            data = raw
            if body is not None:
                data = json.dumps(body).encode("utf-8")
                sent["Content-Type"] = "application/json"
            if token:
                sent["Authorization"] = f"Bearer {token}"
            sent.update(headers or {})
            url = f"/api/v1/{endpoint}" if not endpoint.startswith("/") else endpoint
            try:
                if headers_only:
                    con.putrequest(method, url, skip_host=True, skip_accept_encoding=True)
                    for name, value in sent.items():
                        con.putheader(name, value)
                    con.endheaders()
                else:
                    con.request(method, url, data, sent)
                response = con.getresponse()
                box["result"] = (response.status, json.loads(response.read()), dict(response.getheaders()))
            except Exception as exc:
                box["error"] = exc
            finally:
                con.close()

        thread = threading.Thread(target=run)
        thread.start()
        wait_until(lambda: not thread.is_alive(), 15)
        if "error" in box:
            raise box["error"]
        return box["result"]

    def paired(self, scopes=("read",)):
        self.api.open_pairing()
        status, answer, _ = self.call("POST", "pair", {"name": "Tool", "scopes": list(scopes)})
        self.assertEqual(status, 200, answer)
        self.store.enter(answer["pairing_code"])
        return answer["token"]

    # ---- pairing ----
    def test_pairing(self):
        status, answer, _ = self.call("POST", "pair", {"name": "Tool"})
        self.assertEqual((status, answer["code"]), (403, "pairing_closed"))
        self.api.open_pairing()
        status, answer, headers = self.call("POST", "pair", {"name": "Tool", "scopes": ["read", "write"]})
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "application/json; charset=utf-8")
        self.assertEqual(headers["Cache-Control"], "no-store")
        token, code = answer["token"], answer["pairing_code"]
        self.assertEqual(self.call("POST", "pair", {"name": "Other"})[0], 409)        # one app at a time
        status, answer, _ = self.call("GET", "verify", token=token)
        self.assertEqual((status, answer["state"]), (202, "pending"))
        self.assertEqual(self.call("POST", "analyze", {"path": self.audio}, token=token)[1]["code"], "not_paired")
        self.store.enter(code)
        status, answer, _ = self.call("GET", "verify", token=token)
        self.assertEqual((status, answer["state"], answer["name"], answer["scopes"]),
                         (200, "paired", "Tool", ["read", "write"]))
        self.assertEqual(self.call("POST", "analyze", {"path": self.audio}, token=token)[0], 202)

    def test_refused_pairing(self):
        self.api.open_pairing()
        token = self.call("POST", "pair", {"name": "Tool"})[1]["token"]
        self.store.refuse()
        status, answer, _ = self.call("GET", "verify", token=token)
        self.assertEqual((status, answer["code"]), (403, "pairing_denied"))
        self.assertEqual(self.call("GET", "verify")[1]["code"], "missing_token")
        self.assertEqual(self.call("GET", "verify", token="mxl1.a.b")[1]["code"], "not_paired")

    # ---- refused before any route ----
    def test_not_this_computer_or_a_browser(self):
        self.api.open_pairing()
        for headers in ({"Host": "evil.example"}, {"Host": f"evil.example:{self.port}"}, {"Host": "127.0.0.1"},
                        {"Origin": "https://evil.example"}, {"Origin": "null"}, {"Sec-Fetch-Site": "cross-site"},
                        {"Sec-Fetch-Mode": "no-cors"}):
            with self.subTest(headers=headers):
                status, answer, sent = self.call("POST", "pair", {"name": "Tool"}, headers=headers)
                self.assertEqual(status, 403)
                self.assertIn(answer["code"], ("forbidden_host", "browser_request"))
                self.assertNotIn("Access-Control-Allow-Origin", sent)
        self.assertEqual(self.store.view().phase.value, "open")       # never asked to pair

    def test_bodies_and_methods(self):
        token = self.paired()
        auth = {"Authorization": f"Bearer {token}", "Host": f"127.0.0.1:{self.port}"}
        cases = [
            ("POST", "analyze", b'{"path": 1}', {"Content-Type": "text/plain"}, 415, "unsupported_media_type"),
            ("POST", "analyze", b"{not json", {"Content-Type": "application/json"}, 400, "invalid_json"),
            ("POST", "analyze", b"[1, 2]", {"Content-Type": "application/json"}, 400, "invalid_json"),
            ("POST", "analyze", b"[" * 60000, {"Content-Type": "application/json"}, 400, "invalid_json"),
            ("POST", "analyze", b"", {"Content-Type": "application/json"}, 400, "invalid_json"),
            ("GET", "verify", b"{}", {"Content-Type": "application/json"}, 400, "bad_request"),
            ("GET", "analyze", None, {}, 405, "method_not_allowed"),
            ("OPTIONS", "pair", None, {}, 405, "method_not_allowed"),
            ("GET", "/api/v1/nothing", None, {}, 404, "not_found"),
            ("GET", "/", None, {}, 404, "not_found"),
        ]
        for method, endpoint, raw, headers, status, code in cases:
            with self.subTest(method=method, endpoint=endpoint, code=code):
                got, answer, sent = self.call(method, endpoint, raw=raw, headers={**auth, **headers})
                self.assertEqual((got, answer["code"]), (status, code))
                self.assertNotIn("Access-Control-Allow-Origin", sent)
        self.assertEqual(self.call("GET", "analyze")[2]["Allow"], "POST")
        # refused before their body is read: too large, chunked, none said
        json_type = {"Content-Type": "application/json"}
        for sent, status, code in (({"Content-Length": str(64 * 1024 + 1)}, 413, "too_large"),
                                   ({"Transfer-Encoding": "chunked"}, 411, "length_required"),
                                   ({}, 411, "length_required"),
                                   ({"Content-Length": "-1"}, 400, "bad_request")):
            with self.subTest(sent=sent):
                got, answer, headers = self.call("POST", "analyze", headers={**auth, **json_type, **sent},
                                                 headers_only=True)
                self.assertEqual((got, answer["code"]), (status, code))
                self.assertEqual(headers.get("Connection"), "close")

    # ---- tokens and files ----
    def test_tokens_and_scopes(self):
        token = self.paired(scopes=("read",))
        self.assertEqual(self.call("POST", "analyze", {"path": self.audio})[1]["code"], "missing_token")
        status, answer, headers = self.call("POST", "analyze", {"path": self.audio}, token=token + "x")
        self.assertEqual((status, answer["code"]), (401, "not_paired"))
        self.assertIn("Bearer", headers["WWW-Authenticate"])
        status, answer, _ = self.call("POST", "analyze", {"path": self.audio, "save": True}, token=token)
        self.assertEqual((status, answer["code"]), (403, "scope_denied"))
        self.assertEqual(self.call("POST", "analyze", {"path": self.audio, "save": True},
                                   token=forged(token, ["read", "write"]))[1]["code"], "not_paired")
        self.store.unpair(self.store.pairings()[0].token_id)
        self.assertEqual(self.call("POST", "analyze", {"path": self.audio}, token=token)[1]["code"], "not_paired")

    def test_files_taken(self):
        token = self.paired()
        cases = [({"path": "song.flac"}, 400, "path_not_absolute"),
                 ({"path": "\\\\server\\share\\song.flac"}, 400, "path_not_local"),
                 ({"path": "//server/share/song.flac"}, 400, "path_not_local"),
                 ({"path": self.text}, 415, "unsupported_file"),
                 ({"path": os.path.join(self.dir.name, "missing.flac")}, 404, "file_not_found"),
                 ({"path": 5}, 400, "bad_request"),
                 ({"path": self.audio, "save": "yes"}, 400, "bad_request")]
        for body, status, code in cases:
            with self.subTest(body=body):
                got, answer, _ = self.call("POST", "analyze", body, token=token)
                self.assertEqual((got, answer["code"]), (status, code))
        self.assertEqual(self.workflow.requests, [])

    # ---- answers ----
    def test_a_library_track_at_once(self):
        token = self.paired()
        self.library.tracks[os.path.normcase(self.audio)] = SimpleNamespace(
            uid="uid-1", path=self.audio, title="Title", artist="Artist", album="Album", total_samples=88200,
            duration=2.0)
        status, answer, _ = self.call("POST", "analyze", {"path": self.audio}, token=token)
        self.assertEqual((status, answer["source"], answer["saved"], answer["analysis_id"]), (200, "library", True, None))
        mixlyzer = answer["jams"]["sandbox"]["mixlyzer"]
        self.assertEqual((mixlyzer["uid"], mixlyzer["file"]["path"]), ("uid-1", self.audio))
        self.assertEqual(mixlyzer["file"]["audio"]["packets_sha256"], AUDIO.sha256)
        self.assertEqual(answer["jams"]["file_metadata"]["title"], "Title")
        self.assertEqual(self.workflow.requests, [])

    def test_an_analysis_and_its_progress(self):
        token = self.paired()
        status, answer, headers = self.call("POST", "analyze", {"path": self.audio}, token=token)
        self.assertEqual((status, answer["state"]), (202, "waiting"))
        analysis_id = answer["analysis_id"]
        self.assertEqual(headers["Location"], f"/api/v1/progress/{analysis_id}")
        self.assertEqual(self.workflow.requests, [(os.path.abspath(self.audio), False, analysis_id)])
        status, answer, _ = self.call("GET", f"progress/{analysis_id}", token=token)
        self.assertEqual((status, answer["state"], answer["position"]), (202, "waiting", 1))
        self.workflow.states[analysis_id] = AnalysisState(analysis_id, self.audio, "running", 0.4)
        status, answer, _ = self.call("GET", f"progress/{analysis_id}", token=token)
        self.assertEqual((status, answer["state"], answer["progress"]), (202, "running", 0.4))
        # the same file asked again joins it
        self.assertEqual(self.call("POST", "analyze", {"path": self.audio}, token=token)[0], 202)
        del self.workflow.states[analysis_id]
        self.workflow.ended.emit(analysis_id, {"analysis": ANALYSIS, "properties": {
            "path": os.path.abspath(self.audio), "title": "New", "artist": "", "album": "", "total_samples": 88200,
            "duration_sec": 2.0}}, "")
        box = {}
        wait_until(lambda: box.setdefault("a", self.call("GET", f"progress/{analysis_id}", token=token))[0] == 200
                   or box.pop("a") is None)
        status, answer, _ = box["a"]
        self.assertEqual((answer["state"], answer["source"], answer["saved"]), ("done", "analysis", False))
        mixlyzer = answer["jams"]["sandbox"]["mixlyzer"]
        self.assertNotIn("uid", mixlyzer)                       # not in the library: no uid
        self.assertEqual(mixlyzer["file"]["path"], os.path.abspath(self.audio))
        self.assertEqual(answer["jams"]["file_metadata"]["title"], "New")
        self.assertEqual(self.call("GET", "progress/" + "0" * 32, token=token)[1]["code"], "unknown_analysis")
        self.assertEqual(self.call("GET", "progress/../verify", token=token)[1]["code"], "unknown_analysis")

    def test_a_failed_analysis(self):
        token = self.paired(scopes=("read", "write"))
        analysis_id = self.call("POST", "analyze", {"path": self.audio, "save": True}, token=token)[1]["analysis_id"]
        self.assertEqual(self.workflow.requests[0][1], True)
        del self.workflow.states[analysis_id]
        self.workflow.ended.emit(analysis_id, None, "decoder said no")
        status, answer, _ = self.call("GET", f"progress/{analysis_id}", token=token)
        self.assertEqual((status, answer["code"], answer["detail"]), (422, "analysis_failed", "decoder said no"))

    def test_the_port_taken(self):
        taken = socket.socket()
        taken.bind(("127.0.0.1", 0))
        taken.listen(1)
        try:
            api = LocalApi(library=self.library, workflow=self.workflow,
                           store=PairingStore(os.path.join(self.dir.name, "other.json")))
            api.apply(SimpleNamespace(libconfig=SimpleNamespace(local_api=True,
                                                                local_api_port=taken.getsockname()[1])))
            status = api.status()
            self.assertFalse(status.listening)
            self.assertTrue(status.error)
            api.shutdown()
        finally:
            taken.close()

    # ---- the example client ----
    def test_the_example_client(self):
        self.api.open_pairing()
        client = local_api_client.Client(f"http://127.0.0.1:{self.port}")
        box = {}

        def pair():
            try:
                box["pair"] = client.pair("Example", write=False)
                box["paired"] = client.wait_paired(timeout=10, interval=0.05)
                box["jams"] = client.analysis(self.audio, interval=0.05)
            except Exception as exc:
                box["error"] = exc

        thread = threading.Thread(target=pair)
        thread.start()
        wait_until(lambda: "pair" in box or "error" in box)
        self.store.enter(box["pair"]["pairing_code"])
        wait_until(lambda: self.workflow.requests or "error" in box)
        analysis_id = self.workflow.requests[0][2]
        del self.workflow.states[analysis_id]
        self.workflow.ended.emit(analysis_id, {"analysis": ANALYSIS, "properties": {"path": self.audio}}, "")
        wait_until(lambda: not thread.is_alive(), 15)
        self.assertNotIn("error", box, box.get("error"))
        self.assertEqual(box["paired"]["name"], "Example")
        self.assertEqual(box["jams"]["sandbox"]["mixlyzer"]["format"], "JAMS-mxl-sbx1")
        with self.assertRaises(local_api_client.ApiError) as caught:
            local_api_client.Client(f"http://127.0.0.1:{self.port}", "mxl1.a.b").analyze(self.audio)
        self.assertEqual((caught.exception.status, caught.exception.code), (401, "not_paired"))


class ExampleClientImportsTests(unittest.TestCase):
    def test_the_standard_library_alone(self):
        """examples/local_api_client.py is for other programs to take: it imports nothing outside Python's own."""
        import ast
        import sys
        source = os.path.join(os.path.dirname(__file__), "..", "examples", "local_api_client.py")
        with open(source, encoding="utf-8") as f:
            tree = ast.parse(f.read())
        names = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import)
                 for alias in node.names}
        names |= {node.module.split(".")[0] for node in ast.walk(tree)
                  if isinstance(node, ast.ImportFrom) and node.module and not node.level}
        self.assertLessEqual(names - {"__future__"}, set(sys.stdlib_module_names))


if __name__ == "__main__":
    unittest.main()
