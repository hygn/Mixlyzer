"""Writes a song's ground truth into songs.json from a library where it is annotated (read only), and its audio file's
place into the local audio.json (git-ignored). The song's title and artist go into songs.json (published), so a
runner knows which release to get; nothing else of the library does (no uid, no path).

    uv run python tests/ground_truth/make_fixture.py <id> <uid> [--library <folder>]

Run again for an id already there: its ground truth is replaced, its scores' floors (``expect``) kept."""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SONGS = HERE / "songs.json"
AUDIO = HERE / "audio.json"


def _round(values, places=4):
    return [round(float(v), places) for v in np.asarray(values, dtype=float).ravel()]


def ground_truth(arrays) -> dict:
    from analyzer.music.tempo_rows import with_end_bpm
    from core.analysis_library.stored import StoredAnalysis
    a = StoredAnalysis.from_arrays(arrays)
    return {
        "duration_sec": round(float(a.duration_sec), 3),
        "beats": _round(a.beats),
        "tempo_segments": [_round(r) for r in with_end_bpm(a.tempo_segments)],
        # no keys: a library's keys are mostly the analyzer's own; a song's keys come from a human source alone
        # (songs.json "keys", written by hand with its source)
        "phrases": [[round(p.start, 4), round(p.end, 4), p.label] for p in a.phrases],
    }


def dumps(songs: dict) -> str:
    """songs.json as written: a field a line, each value compact (a diff shows the field that changed)."""
    def one(song):
        return "  {\n" + ",\n".join(f"   {json.dumps(k)}: {json.dumps(v, ensure_ascii=False)}"
                                    for k, v in song.items()) + "\n  }"
    return '{\n "songs": [\n' + ",\n".join(one(s) for s in songs["songs"]) + "\n ]\n}\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("id", help="the song's id in songs.json (a short slug)")
    p.add_argument("uid", help="the track's uid in the library (a unique prefix will do)")
    p.add_argument("--library", default=str(ROOT / "library"))
    args = p.parse_args(argv)

    sys.path.insert(0, str(ROOT))
    from core.analysis_library.store import FeatureNPZStore
    lib = Path(args.library)
    db = sqlite3.connect(f"{(lib / 'library.db').resolve().as_uri()}?mode=ro", uri=True)
    rows = db.execute("SELECT uid, title, artist, album, path FROM tracks WHERE uid LIKE ?", (args.uid + "%",)).fetchall()
    if len(rows) != 1:
        print(f"{len(rows)} tracks match that uid")
        return 1
    uid, title, artist, album, path = rows[0]
    songs = json.loads(SONGS.read_text(encoding="utf-8")) if SONGS.is_file() else {"songs": []}
    old = next((s for s in songs["songs"] if s["id"] == args.id), None)
    song = {"id": args.id, "title": title, "artist": artist, "release": album or "",
            "expect": (old or {}).get("expect", {}), **ground_truth(FeatureNPZStore(str(lib)).load(uid))}
    if old and "keys" in old:
        song["keys"] = old["keys"]                                        # the human keys kept
    songs["songs"] = [s for s in songs["songs"] if s["id"] != args.id] + [song]
    SONGS.write_text(dumps(songs), encoding="utf-8")
    audio = json.loads(AUDIO.read_text(encoding="utf-8")) if AUDIO.is_file() else {}
    audio[args.id] = path
    AUDIO.write_text(json.dumps(audio, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{args.id}: {len(song['beats'])} beats, {len(song['key_segments'])} keys, {len(song['phrases'])} phrases")
    return 0


if __name__ == "__main__":
    sys.exit(main())
