"""A song's keys from human sources alone: a chord chart's Key markers (a ChordWiki page saved from the browser), timed
by the song's synced lyrics (an LRC). Where the two say for sure which key plays is a key segment; the rest is left
out (not scored). The chart and the LRC stay where they are: only the segments and the chart's address go into
songs.json.

    uv run python tests/ground_truth/chart_keys.py <saved chart .htm> <song .lrc> <id> <chart url> [--write]

Each sung chart line is matched to a run of 1 - 3 LRC lines (an LRC splits a chart's line now and then), in order. A
block (the lines after a Key marker) starts at its first line's time snapped to the nearest downbeat of the song's
grid when that line is sung (the key changes with it), else at its first sung line (the key surely holds by then:
the chords before it are the block's too, but their time is not known); the first block starts at 0. A block ends
where the next starts when the two meet on sung lines, else one LRC line's length (the LRC's median) after its last
sung line; the chart's last block ends with the track. A block without a sung line is left out. The keys are the
chart's names (a key signature: the score compares scales)."""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from html.parser import HTMLParser
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TIME = re.compile(r"\[(\d+):(\d+(?:\.\d+)?)\]")
PITCH = {"C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3, "E": 4, "F": 5, "F#": 6, "GB": 6, "G": 7, "G#": 8,
         "AB": 8, "A": 9, "A#": 10, "BB": 10, "B": 11, "CB": 11}
MIN_SIMILARITY = 0.5
RUN = 3


class _Chart(HTMLParser):
    """A ChordWiki page's chart: its key markers, comments and lines (their chords and sung text)."""

    KINDS = {"key": "key", "line": "line", "line comment": "comment"}

    def __init__(self):
        super().__init__()
        self.lines, self._cur, self._span = [], None, None

    def handle_starttag(self, tag, attrs):
        cls = dict(attrs).get("class", "")
        if tag == "p" and cls in self.KINDS:
            self._cur = {"kind": self.KINDS[cls], "chords": [], "text": ""}
        elif self._cur is not None and tag == "span":
            self._span = cls

    def handle_endtag(self, tag):
        if tag == "p" and self._cur is not None:
            self._cur["text"] = self._cur["text"].strip()
            self.lines.append(self._cur)
            self._cur = None
        elif tag == "span":
            self._span = None

    def handle_data(self, data):
        if self._cur is not None:
            if self._span == "chord":
                self._cur["chords"].append(data.strip())
            else:
                self._cur["text"] += data


def chart_lines(path) -> list[dict]:
    parser = _Chart()
    parser.feed(Path(path).read_text(encoding="utf-8", errors="replace"))
    return parser.lines


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[♦♥♠♣◆◇●○★☆]", "", text)        # a chart's marks of who sings
    return "".join(ch for ch in text.casefold() if ch.isalnum())


def lrc_lines(path) -> list[tuple[float, str]]:
    out = []
    for raw in Path(path).read_text(encoding="utf-8-sig", errors="replace").splitlines():
        text = norm(TIME.sub("", raw))
        for m, s in TIME.findall(raw):
            if text:
                out.append((int(m) * 60 + float(s), text))
    return sorted(out)


def align(sung, lrc) -> dict[int, tuple[float, float]]:
    """sung: [(chart line, text)] -> chart line: (its first LRC line's time, similarity), in order."""
    from rapidfuzz import fuzz
    n, m = len(sung), len(lrc)
    best = np.full((n + 1, m + 1), -np.inf)
    best[0, :] = best[:, 0] = 0.0
    move = {}
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            options = [(best[i - 1, j], ("skip_chart",)), (best[i, j - 1], ("skip_lrc",))]
            for r in range(1, min(RUN, j) + 1):
                s = fuzz.ratio(sung[i - 1][1], "".join(t for _x, t in lrc[j - r:j])) / 100.0
                options.append((best[i - 1, j - r] + (s if s >= MIN_SIMILARITY else -1.0), ("match", r, s)))
            best[i, j], move[i, j] = max(options, key=lambda o: o[0])
    pairs, i, j = {}, n, m
    while i and j:
        step = move[i, j]
        if step[0] == "match":
            pairs[sung[i - 1][0]] = (lrc[j - step[1]][0], step[2])
            i, j = i - 1, j - step[1]
        elif step[0] == "skip_chart":
            i -= 1
        else:
            j -= 1
    return pairs


def key_of(name: str) -> tuple[int, int]:
    name = name.strip()
    minor = name.endswith("m")
    return PITCH[(name[:-1] if minor else name).upper().replace("♯", "#").replace("♭", "B")], int(minor)


def timed_keys(chart, lrc, song) -> tuple[list[list[float]], list[str]]:
    from analyzer.music.beat_geometry import downbeat_times
    downs = downbeat_times(np.asarray(song["beats"]), np.asarray(song["tempo_segments"]))
    snap = lambda t: float(downs[np.argmin(np.abs(downs - t))])   # noqa: E731
    sung = [(k, norm(line["text"])) for k, line in enumerate(chart) if line["kind"] == "line" and norm(line["text"])]
    pairs = align(sung, lrc)
    gaps = np.diff([t for t, _x in lrc])
    line_len = float(np.median(gaps[gaps > 0.5]))
    blocks: list[tuple[str, list[int]]] = []
    for k, line in enumerate(chart):
        if line["kind"] == "key":
            blocks.append((line["text"].split(":", 1)[1].strip(), []))
        elif line["kind"] == "line" and blocks:
            blocks[-1][1].append(k)
    timed, notes = [], []
    for b, (key, lines) in enumerate(blocks):
        sung_lines = [k for k in lines if k in pairs]
        if not sung_lines:
            notes.append(f"block {b} (Key {key}): no sung line, left out")
            continue
        first = lines[0] in pairs
        start = 0.0 if b == 0 else (snap(pairs[lines[0]][0]) if first else pairs[sung_lines[0]][0])
        timed.append({"key": key, "start": start, "first_sung": first, "last_sung": lines[-1] in pairs,
                      "final": b == len(blocks) - 1, "last_time": pairs[sung_lines[-1]][0],
                      "score": min(pairs[k][1] for k in sung_lines)})
    segments = []
    for b, blk in enumerate(timed):
        if blk["final"]:
            end = float(song["duration_sec"])
        elif b + 1 == len(timed):
            end = blk["last_time"] + line_len
        elif blk["last_sung"] and timed[b + 1]["first_sung"]:
            end = timed[b + 1]["start"]
        else:
            end = min(blk["last_time"] + line_len, timed[b + 1]["start"])
        segments.append([*key_of(blk["key"]), round(blk["start"], 3), round(end, 3)])
        notes.append(f"Key {blk['key']:>3}: {blk['start'] // 60:.0f}:{blk['start'] % 60:05.2f} - "
                     f"{end // 60:.0f}:{end % 60:05.2f}, its lines matched at {blk['score']:.2f} at least")
    return segments, notes


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("chart", help="the chart page saved from the browser (.htm)")
    p.add_argument("lrc", help="the song's synced lyrics (.lrc) for the same recording")
    p.add_argument("id", help="the song's id in songs.json")
    p.add_argument("source", help="the chart's address (kept as the keys' source)")
    p.add_argument("--write", action="store_true", help="write the keys into songs.json")
    args = p.parse_args(argv)
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(HERE))
    from make_fixture import SONGS, dumps
    songs = json.loads(SONGS.read_text(encoding="utf-8"))
    song = next(s for s in songs["songs"] if s["id"] == args.id)
    segments, notes = timed_keys(chart_lines(args.chart), lrc_lines(args.lrc), song)
    print("\n".join(notes))
    if args.write:
        song["keys"] = {"source": args.source, "timing": "the chart's Key markers timed by the song's LRC "
                        "(tests/ground_truth/chart_keys.py)", "segments": segments}
        SONGS.write_text(dumps(songs), encoding="utf-8")
        print(f"{args.id}: {len(segments)} key segments written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
