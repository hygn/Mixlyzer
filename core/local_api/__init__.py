"""The local API: other programs on this computer ask Mixlyzer for a track's analysis over HTTP on 127.0.0.1, once
paired with it (docs/local_api.md). `tokens` the bearer tokens and their scopes; `pairings` the paired apps' file and
the pairing; `protocol` a request and an answer; `server` the HTTP server and its checks; `service` the app's side
(`LocalApi`): the routes, the analyses asked for, the server started as the settings say."""
