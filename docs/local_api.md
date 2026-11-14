# Mixlyzer local API

Programs on the same computer can ask Mixlyzer to analyze an audio file and get the track's beats, tempo, keys,
phrases, CUE points and JumpCUEs back as a JAMS document (the `JAMS-mxl-sbx1` extension). The API is HTTP on
`127.0.0.1` and works only while Mixlyzer is running.

A reference client that uses only Python's standard library is in
[`examples/local_api_client.py`](../examples/local_api_client.py). A reference reader for the JAMS it returns is in
[`examples/jams_mxl_sbx1.py`](../examples/jams_mxl_sbx1.py).

## Turning it on

1. In Mixlyzer, open **Settings > Library > Local Apps**.
2. Check **Let apps on this computer ask for analyses** and set the port (default `48700`).
3. Click **Apply**. The page shows `Listening on 127.0.0.1:48700.`
4. Each app pairs once (below). The paired apps are listed on the same page, where you can unpair them.

## Security

- The server listens on `127.0.0.1` only. Other computers cannot reach it.
- Requests from web pages are refused. A request is refused if:
  - it carries `Origin` or any `Sec-Fetch-*` header;
  - its `Host` header is not `127.0.0.1:<port>`, `localhost:<port>` or `[::1]:<port>` (this blocks DNS rebinding).
- The server never sends CORS headers.
- `POST` bodies must be `application/json` with a `Content-Length` of at most 64 KiB.
- Every endpoint except `pair` needs a token, sent as `Authorization: Bearer <token>`. A token is never accepted in
  the URL or the body.
- Mixlyzer keeps only a SHA-256 of each token. The permissions are inside the token and bound by that hash, so a
  token whose permissions were edited is refused. To get more permissions, pair again.
- Only full paths to files on this computer are accepted. Network paths (`\\server\share`, `//host`) and device paths
  are refused.

## Permissions

| Scope | Grants |
|---|---|
| `read` | `analyze` and `progress`. If the file is already in the library, its stored analysis (with the user's edits) is returned. |
| `write` | `analyze` with `"save": true`: a new analysis is stored in the library. |

`read` is always required. `write` is optional.

## Answers

Every answer is one JSON object:

```json
{"status": 200, "code": "ok", "message": "...", "...": "..."}
```

- `status` repeats the HTTP status.
- `code` is a fixed word for programs to act on.
- `message` is an English sentence for people. It is not translated; show your own text by `code` if you need another
  language.

## Endpoints

### `POST /api/v1/pair`

Asks to pair. The user must first click **Pair an App…** in Mixlyzer, which opens pairing for two minutes. One app
can ask at a time.

Request:

```json
{"name": "My Tool", "scopes": ["read", "write"]}
```

- `name`: 1–64 characters. It is shown to the user.
- `scopes`: optional; the default is `["read"]`.

Answer `200`:

```json
{"status": 200, "code": "ok", "message": "...", "pairing_code": "0427", "token": "mxl1....",
 "scopes": ["read", "write"], "expires_in": 120}
```

Show `pairing_code` to the user, who types it into Mixlyzer. The token becomes valid once the code is typed. Keep the
token secret: anyone holding it can use Mixlyzer as your app.

Errors:

| Status | Code | Meaning |
|---|---|---|
| 403 | `pairing_closed` | Pairing is not open in Mixlyzer. |
| 409 | `pairing_busy` | Another app is being paired. |
| 400 | `bad_request` | The name or the scopes are not valid. |

### `GET /api/v1/verify`

Tells whether the token is paired. Ask about once a second after `pair` until the state is `paired`.

| Status | Code | `state` | Meaning |
|---|---|---|---|
| 200 | `ok` | `paired` | Paired. The answer also has `name` and `scopes`. |
| 202 | `pairing_pending` | `pending` | The user has not typed the code yet. |
| 403 | `pairing_denied` | `denied` | The user refused, or typed the code wrong three times. |
| 401 | `pairing_expired` | `expired` | The two minutes ran out. |
| 401 | `not_paired` | `unknown` | Never paired, or unpaired. |

### `POST /api/v1/analyze`

Requires `read`. With `"save": true`, also requires `write`.

Request:

```json
{"path": "C:\\Music\\track.flac", "save": false}
```

- `path`: the full path to an audio file Mixlyzer takes. The accepted extensions are `.aac`, `.aif`, `.aiff`, `.alac`,
  `.flac`, `.m4a`, `.mp3`, `.mp4`, `.ogg`, `.opus`, `.wav` and `.wma`.
- `save`: store a new analysis in Mixlyzer's library. The default is `false`.

The answer depends on whether the library already has the track:

- **In the library:** `200` at once, with no analysis run.

  ```json
  {"status": 200, "code": "ok", "analysis_id": null, "state": "done", "source": "library", "saved": true,
   "jams": {...}}
  ```

- **Not in the library:** `202`. An analysis is started, or the one already running for the same file is joined. The
  answer has a `Location` header pointing at its progress.

  ```json
  {"status": 202, "code": "accepted", "analysis_id": "3f2c...", "state": "waiting"}
  ```

  An analysis that is not saved is still saved if the user, or another request with `"save": true`, asks for the
  same file while it runs.

Errors:

| Status | Code | Meaning |
|---|---|---|
| 400 | `path_not_absolute`, `path_not_local`, `path_too_long`, `bad_request` | The path is not taken. |
| 404 | `file_not_found` | No file at the path. |
| 415 | `unsupported_file` | Not an audio file Mixlyzer takes. The answer lists `accepted`. |
| 401 | `missing_token`, `not_paired` | No token, or a token that is not paired. |
| 403 | `scope_denied` | The token lacks `write` for `"save": true`. |
| 422 | `audio_unreadable` | The audio could not be read to identify it. |
| 429 | `too_many_analyses` | Too many analyses are waiting. Retry later. |

### `GET /api/v1/progress/{analysis_id}`

Requires `read`. Ask about once a second, or as `Retry-After` says.

| Status | Code | `state` | Meaning |
|---|---|---|---|
| 202 | `in_progress` | `waiting` | Queued. `position` is its place in the queue (1 is next). |
| 202 | `in_progress` | `running` | Analyzing. `progress` is between 0 and 1. |
| 202 | `in_progress` | `saving` | Storing in the library. |
| 202 | `in_progress` | `finishing` | Making its JAMS. |
| 200 | `ok` | `done` | Done. `jams` holds the document; `saved` tells whether it is in the library. |
| 422 | `analysis_failed` | `failed` | The analysis failed. `detail` says why. |
| 404 | `unknown_analysis` | | No such analysis, or it ended more than 10 minutes ago. |

A finished answer is kept for 10 minutes.

## The JAMS document

`jams` is the same document the Export tab writes (`JAMS-mxl-sbx1`, described in
[`third_party/jams.py`](../third_party/jams.py)), with these differences:

- `sandbox.mixlyzer.file.path` is always present: it is the file you sent, or the library's spelling of it.
- `sandbox.mixlyzer.uid` is present only when the track is in the library. An analysis that was not saved has none.

`sandbox.mixlyzer.file.audio.packets_sha256` identifies the audio itself and does not change when tags are written.
`examples/jams_mxl_sbx1.py` can check it against a file.

## Other errors

| Status | Code | Meaning |
|---|---|---|
| 403 | `forbidden_host`, `browser_request` | The `Host` header or a browser's headers were refused. |
| 404 | `not_found` | No such endpoint. |
| 405 | `method_not_allowed` | Wrong method. The `Allow` header names the right one. |
| 411 / 413 / 415 | `length_required` / `too_large` / `unsupported_media_type` | The body was refused. |
| 400 | `invalid_json` | The body is not a JSON object in UTF-8. |
| 503 | `busy`, `shutting_down` | Mixlyzer could not answer now. Retry later. |
