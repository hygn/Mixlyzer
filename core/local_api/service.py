"""The local API's app side (`LocalApi`, on the GUI thread): the server started and stopped as the settings say
(libconfig.local_api / local_api_port), the routes, the pairing as the settings page drives it, and the analyses other
programs asked for. docs/local_api.md is the API's reference for those programs.

Routes (each answer a JSON object with status / code / message: core.local_api.protocol):
- ``POST /api/v1/pair`` ``{"name", "scopes"}``: while the user has pairing open, a 4-digit code and a token not
  valid yet (core.local_api.pairings); the app shows the code, the user types it into Mixlyzer.
- ``GET /api/v1/verify`` (the token): paired (200), the user not done yet (202), refused (403), its time out (401).
- ``POST /api/v1/analyze`` ``{"path", "save"}`` (``read``; ``save`` true: ``write``): a file on this computer the app
  takes (analyzer.audio.decoder.AUDIO_EXTENSIONS); the library's track of it answered at once (200, its JAMS), else an
  analysis started or the one in flight joined (202, its analysis id: core.workflows.analysis.AnalysisWorkflow.request),
  stored in the library only when ``save``.
- ``GET /api/v1/progress/{analysis_id}`` (``read``): waiting / running / saving / finishing (202, its progress), done
  (200, its JAMS; kept JOB_KEPT_SEC after), failed (422).
The JAMS is third_party.jams' JAMS-mxl-sbx1 of the track: its path always (the app sent it), its uid when the track
is in the library (a track not stored has none: left out).

The server's threads ask the analysis workflow on the GUI thread (`_on_gui`, waited GUI_WAIT_SEC at most: else 503,
the call dropped if it had not run), the library through its owner (`call`), and make the JAMS themselves (the audio's
hash reads the file). An analysis's end comes on the GUI thread (``AnalysisWorkflow.ended``), its JAMS made on a
thread of its own."""
from __future__ import annotations

import concurrent.futures
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Optional

from PySide6 import QtCore

from analyzer.audio.decoder import AUDIO_EXTENSIONS, audio_packet_hash
from core.analysis_library import jobs as analysis_jobs
from core.local_api.pairings import PAIRING_SEC, PairingStore, Phase, Refused, Verdict
from core.local_api.protocol import PREFIX, ApiError, Request, Response, answer
from core.local_api.server import HOST, LocalApiServer
from core.local_api.tokens import READ, WRITE
from system.os_support import network_path, path_too_long
from system.version import APP_VERSION
from third_party.jams import jams_document, marks_of

GUI_WAIT_SEC = 10.0          # a server thread's wait for the GUI thread
JOB_KEPT_SEC = 600.0         # an analysis's answer kept after it ended
MAX_JOBS = 200               # analyses asked for and answers kept, at most
MAX_PATH_CHARS = 4096
_ANALYSIS_ID = re.compile(r"[0-9a-f]{32}")
_WWW_AUTHENTICATE = (("WWW-Authenticate", 'Bearer realm="Mixlyzer"'),)
_RETRY = (("Retry-After", "1"),)

PENDING, FINISHING, DONE, FAILED = "pending", "finishing", "done", "failed"


@dataclass
class _Job:
    """An analysis another program asked for (its fields under LocalApi._jobs_lock)."""
    analysis_id: str
    path: str
    state: str = PENDING
    jams: Optional[dict] = None
    saved: bool = False
    error: str = ""
    ended_at: float = 0.0


@dataclass(frozen=True)
class ApiStatus:
    """The server as the settings page shows it."""
    on: bool              # the settings ask for it
    listening: bool
    port: int
    error: str = ""       # why it is not listening while asked to


class LocalApi(QtCore.QObject):
    status_changed = QtCore.Signal()      # the server started, stopped or refused (status())
    pairing_changed = QtCore.Signal()     # the pairing or the paired apps changed (from any thread)
    _run = QtCore.Signal(object)          # (a call, its future) from a server thread, run on the GUI thread

    def __init__(self, *, library, workflow, store: Optional[PairingStore] = None, parent=None) -> None:
        super().__init__(parent)
        self._library = library            # core.library.service.LibraryService
        self._workflow = workflow          # core.workflows.analysis.AnalysisWorkflow
        self.store = store if store is not None else PairingStore()
        self.store.on_change = self.pairing_changed.emit
        self.store.load()
        self._server: Optional[LocalApiServer] = None
        self._wanted = (False, 0)
        self._error = ""
        self._closing = False
        self._jobs: dict[str, _Job] = {}
        self._jobs_lock = threading.Lock()
        self._run.connect(self._on_run)
        workflow.ended.connect(self._on_ended)
        self._pairing_timer = QtCore.QTimer(self)       # the pairing's time checked while it is open
        self._pairing_timer.setInterval(500)
        self._pairing_timer.timeout.connect(self._check_pairing)

    # ---- the server ----
    def apply(self, cfg) -> None:
        """The server as the settings say: started, stopped, or moved to another port (one that failed tried
        again)."""
        wanted = (bool(cfg.libconfig.local_api), int(cfg.libconfig.local_api_port))
        if wanted == self._wanted and not self._error:
            return
        self._wanted = wanted
        self._stop_server()
        self._error = ""
        if wanted[0]:
            try:
                server = LocalApiServer(wanted[1], self.route)
                server.start()
                self._server = server
                print(f"[Local API] listening on {HOST}:{server.port}")
            except OSError as exc:
                self._error = exc.strerror or str(exc)
                print(f"[Local API] port {wanted[1]} could not be opened: {self._error}")
        self.status_changed.emit()

    @QtCore.Slot(object, object)
    def settings_changed(self, _old, new) -> None:
        self.apply(new)

    def status(self) -> ApiStatus:
        port = self._server.port if self._server is not None else self._wanted[1]
        return ApiStatus(self._wanted[0], self._server is not None, port, self._error)

    def shutdown(self) -> None:
        """The app quitting: the server stopped, the pairing closed, the last uses written."""
        self._closing = True
        self._pairing_timer.stop()
        self._stop_server()
        self.store.flush()

    def _stop_server(self) -> None:
        server, self._server = self._server, None
        if server is not None:
            server.stop()

    # ---- pairing (the settings page) ----
    def open_pairing(self) -> None:
        self.store.open(PAIRING_SEC)
        self._pairing_timer.start()

    def close_pairing(self) -> None:
        self.store.close()
        self._pairing_timer.stop()

    def _check_pairing(self) -> None:
        self.store.expire_due()
        if self.store.view().phase is Phase.CLOSED:
            self._pairing_timer.stop()

    # ---- the GUI thread, from a server thread ----
    def _on_gui(self, call):
        """``call()`` run on the GUI thread, its result returned (its exception raised)."""
        if threading.current_thread() is threading.main_thread():
            return call()
        future: concurrent.futures.Future = concurrent.futures.Future()
        self._run.emit((call, future))
        try:
            return future.result(timeout=GUI_WAIT_SEC)
        except concurrent.futures.TimeoutError:
            future.cancel()            # not run yet: never run
            raise ApiError(503, "busy", "Mixlyzer is busy: try again in a moment.", headers=(("Retry-After", "2"),))

    @QtCore.Slot(object)
    def _on_run(self, item) -> None:
        call, future = item
        if not future.set_running_or_notify_cancel():
            return
        try:
            future.set_result(call())
        except BaseException as exc:
            future.set_exception(exc)

    # ---- routes (the server's threads) ----
    def route(self, request: Request) -> Response:
        if self._closing:
            raise ApiError(503, "shutting_down", "Mixlyzer is quitting.")
        if not request.path.startswith(PREFIX):
            raise ApiError(404, "not_found", "No such endpoint.")
        name = request.path[len(PREFIX):]
        if name.startswith("progress/"):
            endpoint, method = (lambda r: self._progress(r, name[len("progress/"):])), "GET"
        else:
            endpoint, method = {"pair": (self._pair, "POST"), "verify": (self._verify, "GET"),
                                "analyze": (self._analyze, "POST")}.get(name, (None, ""))
        if endpoint is None:
            raise ApiError(404, "not_found", "No such endpoint.")
        if request.method != method:
            raise ApiError(405, "method_not_allowed", f"This endpoint takes {method}.", headers=(("Allow", method),))
        try:
            return endpoint(request)
        except ApiError as exc:
            if exc.status >= 500:
                print(f"[Local API] {request.method} {request.path}: {exc.status} {exc.code}")
            raise

    def _grant(self, request: Request, scope: str):
        token = request.bearer()
        if token is None:
            raise ApiError(401, "missing_token", "Send the token Mixlyzer gave at pairing: Authorization: Bearer "
                                                "<token>.", headers=_WWW_AUTHENTICATE)
        grant = self.store.authenticate(token)
        if grant is None:
            raise ApiError(401, "not_paired", "This token is not paired with Mixlyzer (never paired, or unpaired): "
                                              "pair again.", headers=_WWW_AUTHENTICATE)
        if scope not in grant.scopes:
            raise ApiError(403, "scope_denied", f"This token does not grant {scope!r}: pair again asking for it.")
        return grant

    def _pair(self, request: Request) -> Response:
        body = request.body or {}
        try:
            asking = self.store.ask(body.get("name"), body.get("scopes", [READ]))
        except Refused as exc:
            status = {"pairing_closed": 403, "pairing_busy": 409}.get(exc.code, 400)
            raise ApiError(status, exc.code, str(exc)) from None
        print(f"[Local API] pairing asked by {asking.name!r} ({', '.join(asking.scopes)})")
        return Response(200, answer(
            200, "ok", "Show the pairing code to the user, who types it into Mixlyzer; then ask /api/v1/verify "
                       "with the token until it is paired.",
            pairing_code=asking.code, token=asking.issued.token, scopes=list(asking.scopes),
            expires_in=int(PAIRING_SEC)))

    def _verify(self, request: Request) -> Response:
        token = request.bearer()
        if token is None:
            raise ApiError(401, "missing_token", "Send the token Mixlyzer gave at pairing: Authorization: Bearer "
                                                "<token>.", headers=_WWW_AUTHENTICATE)
        verdict, grant = self.store.check(token)
        if verdict is Verdict.PAIRED:
            return Response(200, answer(200, "ok", "Paired.", state="paired", name=grant.name,
                                        scopes=list(grant.scopes)))
        if verdict is Verdict.PENDING:
            return Response(202, answer(202, "pairing_pending", "Waiting for the user to type the code into "
                                                                "Mixlyzer.", state="pending"), _RETRY)
        if verdict is Verdict.DENIED:
            raise ApiError(403, "pairing_denied", "The pairing was refused (or its code typed wrong): pair again.",
                           state="denied")
        if verdict is Verdict.EXPIRED:
            raise ApiError(401, "pairing_expired", "The pairing's time ran out: pair again.", state="expired",
                           headers=_WWW_AUTHENTICATE)
        raise ApiError(401, "not_paired", "This token is not paired with Mixlyzer: pair again.", state="unknown",
                       headers=_WWW_AUTHENTICATE)

    def _analyze(self, request: Request) -> Response:
        grant = self._grant(request, READ)
        body = request.body or {}
        save = body.get("save", False)
        if not isinstance(save, bool):
            raise ApiError(400, "bad_request", "save: true or false.")
        if save and WRITE not in grant.scopes:
            raise ApiError(403, "scope_denied", "Saving into the library needs the 'write' permission: pair again "
                                                "asking for it.")
        path = _local_file(body.get("path"))
        track = self._library.snapshot.find_file(path)
        if track is not None and track.uid:
            return self._from_library(track)
        self._prune()
        with self._jobs_lock:
            if len(self._jobs) >= MAX_JOBS:
                raise ApiError(429, "too_many_analyses", "Too many analyses asked for: try again once some are "
                                                         "done.", headers=(("Retry-After", "10"),))

        def ask() -> Optional[str]:      # on the GUI thread: the job kept before its analysis can end
            analysis_id = self._workflow.request(path, store=save)
            if analysis_id is not None:
                with self._jobs_lock:
                    self._jobs.setdefault(analysis_id, _Job(analysis_id, path))
            return analysis_id

        analysis_id = self._on_gui(ask)
        if analysis_id is None:
            track = self._library.snapshot.find_file(path)
            if track is not None and track.uid:          # added to the library meanwhile
                return self._from_library(track)
            raise ApiError(500, "not_started", "The analysis could not be started.")
        return Response(202, answer(202, "accepted", "Analyzing: ask /api/v1/progress/<analysis_id> until it is "
                                                     "done.", analysis_id=analysis_id, state="waiting"),
                        (("Location", f"{PREFIX}progress/{analysis_id}"),) + _RETRY)

    def _from_library(self, track) -> Response:
        """The library's track of the file: its stored analysis as JAMS."""
        analysis = self._library.call(analysis_jobs.analysis, track.uid)
        if analysis is None:
            raise ApiError(500, "stored_analysis_unreadable", "The track's stored analysis could not be read: "
                                                              "reanalyze it in Mixlyzer.")
        jams = _jams(analysis, path=track.path, title=track.title, artist=track.artist, album=track.album,
                     uid=track.uid, total_samples=track.total_samples, duration_sec=track.duration)
        return Response(200, answer(200, "ok", "From the library.", analysis_id=None, state="done",
                                    source="library", saved=True, jams=jams))

    def _progress(self, request: Request, analysis_id: str) -> Response:
        self._grant(request, READ)
        self._prune()
        job = self._job(analysis_id)
        if job.state == PENDING:
            state = self._on_gui(lambda: self._workflow.state(analysis_id))
            job = self._job(analysis_id)
            if state is not None and job.state == PENDING:
                return Response(202, answer(202, "in_progress", "Analyzing.", analysis_id=analysis_id,
                                            state=state.state, progress=round(float(state.progress), 3),
                                            position=int(state.position)), _RETRY)
            if job.state == PENDING:      # ended without telling: never left waiting
                self._finish_job(analysis_id, error="The analysis ended without a result.")
                job = self._job(analysis_id)
        if job.state == FINISHING:
            return Response(202, answer(202, "in_progress", "Analyzed: making its JAMS.", analysis_id=analysis_id,
                                        state="finishing", progress=1.0, position=0), _RETRY)
        if job.state == FAILED:
            raise ApiError(422, "analysis_failed", "The analysis failed.", detail=job.error, analysis_id=analysis_id,
                           state="failed")
        return Response(200, answer(200, "ok", "Analyzed.", analysis_id=analysis_id, state="done", source="analysis",
                                    saved=job.saved, jams=job.jams))

    def _job(self, analysis_id: str) -> _Job:
        """A copy of the job's fields as they are now."""
        with self._jobs_lock:
            job = self._jobs.get(analysis_id) if _ANALYSIS_ID.fullmatch(analysis_id or "") else None
            if job is None:
                raise ApiError(404, "unknown_analysis", "No analysis of this id (never asked for here, or ended "
                                                        f"over {int(JOB_KEPT_SEC // 60)} minutes ago).")
            return _Job(**vars(job))

    def _prune(self) -> None:
        cutoff = time.monotonic() - JOB_KEPT_SEC
        with self._jobs_lock:
            for analysis_id in [k for k, j in self._jobs.items() if j.state in (DONE, FAILED) and j.ended_at < cutoff]:
                del self._jobs[analysis_id]

    # ---- an analysis ended (the GUI thread) ----
    @QtCore.Slot(str, object, str)
    def _on_ended(self, analysis_id: str, result, error: str) -> None:
        with self._jobs_lock:
            job = self._jobs.get(analysis_id)
            if job is None or job.state != PENDING:
                return
            if error or result is None:
                job.state, job.error, job.ended_at = FAILED, error or "The analysis failed.", time.monotonic()
                return
            job.state = FINISHING
            path = job.path
        threading.Thread(target=self._make_jams, args=(analysis_id, path, result), name="LocalApiJams",
                         daemon=True).start()

    def _make_jams(self, analysis_id: str, path: str, result: dict) -> None:
        """(a thread of its own) An analysis's JAMS, kept for its asking."""
        try:
            analysis = result.get("analysis")
            if analysis is None:
                raise ApiError(500, "analysis_unreadable", "The analysis could not be read.")
            properties = result.get("properties") or {}
            uid = str(properties.get("uid") or "") or None
            jams = _jams(analysis, path=str(properties.get("path") or path), title=properties.get("title"),
                         artist=properties.get("artist"), album=properties.get("album"), uid=uid,
                         total_samples=properties.get("total_samples"),
                         duration_sec=properties.get("duration_sec"))
        except ApiError as exc:
            self._finish_job(analysis_id, error=exc.message)
        except Exception as exc:
            self._finish_job(analysis_id, error=f"The JAMS could not be made: {exc}")
        else:
            self._finish_job(analysis_id, jams=jams, saved=uid is not None)

    def _finish_job(self, analysis_id: str, *, jams: Optional[dict] = None, saved: bool = False,
                    error: str = "") -> None:
        with self._jobs_lock:
            job = self._jobs.get(analysis_id)
            if job is None:
                return
            job.state = FAILED if error else DONE
            job.jams, job.saved, job.error, job.ended_at = jams, saved, error, time.monotonic()


def _local_file(path) -> str:
    """The audio file asked for: a full path on this computer to a file the app takes; an ApiError otherwise."""
    if not isinstance(path, str) or not path.strip() or len(path) > MAX_PATH_CHARS or "\x00" in path:
        raise ApiError(400, "bad_request", "path: the audio file's full path, as a string.")
    if network_path(path):
        raise ApiError(400, "path_not_local", "Only a file on this computer is taken, not a network path.")
    if not os.path.isabs(path):
        raise ApiError(400, "path_not_absolute", "path: a full path, from the drive or the root.")
    if os.path.splitext(path)[1].lower() not in AUDIO_EXTENSIONS:
        raise ApiError(415, "unsupported_file", "Mixlyzer does not take this kind of file.",
                       accepted=sorted(AUDIO_EXTENSIONS))
    if path_too_long(path):
        raise ApiError(400, "path_too_long", "The path is too long for Windows while its long paths are off.")
    if not os.path.isfile(path):
        raise ApiError(404, "file_not_found", "No file at this path.")
    return os.path.abspath(path)


def _jams(analysis, *, path: str, title, artist, album, uid: Optional[str], total_samples,
          duration_sec=None) -> dict:
    """The track's JAMS (third_party.jams): its marks, the file's identity (its audio's hash read here)."""
    audio = audio_packet_hash(path)
    if audio is None:
        raise ApiError(422, "audio_unreadable", "The audio file could not be read.")
    return jams_document(**marks_of(analysis, float(duration_sec or 0.0)), title=str(title or ""),
                         artist=str(artist or ""), album=str(album or ""), path=path,
                         file_size=os.path.getsize(path), uid=uid, audio=audio,
                         total_samples=int(total_samples) if total_samples else None, app_version=APP_VERSION)
