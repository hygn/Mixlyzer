"""The key model's chords, to look at and to compare with a chord chart (for developers). The key model reads a
chord per beat on its way to the keys (analyzer/key/chord_key.py, ``beat_chords``). Those chords were never taught:
keys are the only training input. This lists them per bar with the keys the analysis gives, and, given a chart, lines
the chart's chords up with the track's beats and shows both side by side with how often they agree. The comparison
only reports: it feeds nothing back.

    uv run python -m training.key_chords UID [--chart FILE] [--model PATH]

UID: a library track's uid or the start of one (run from the app's folder: config.json names the library and the
key model in use). FILE: a chart as text, its chords in brackets before the syllable they fall on, one line of the
song a line (``[C]歌詞[G/B]歌詞``; lines without a chord are skipped). The chart is lined up by its chords alone: a
DTW over the beats, each chord one beat or more, jumps to any line's start (repeats a chart writes once, a TV size)
and beats matched to none (drums alone). It is also tried transposed: a chart in another key than the recording
(a capo chart) says so."""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from analyzer.key import chord_key as CK

LETTER = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
CHORD_MARK = re.compile(r"\[([^\]]*)\]")


# --- a chart ---------------------------------------------------------------------------------------

def _pitch(text: str) -> tuple[int | None, str]:
    m = re.match(r"([A-G])([#b]?)", text)
    if not m:
        return None, text
    return (LETTER[m.group(1)] + {"#": 1, "b": -1, "": 0}[m.group(2)]) % 12, text[m.end():]


def parse_chord(symbol: str) -> dict | None:
    """A chart's chord ('A#/D', 'Bm7-5', 'Gsus4', 'FM7', 'E7(9)') -> {root, bass, tones: {interval: weight}, quality}
    (quality: the model's name for it, chord_key.QUALITIES, or None for a sus / dim / aug chord); None for N.C. or
    an unreadable one."""
    s = re.sub(r"\s+", "", symbol.replace("♯", "#").replace("♭", "b").replace("（", "(").replace("）", ")"))
    if not s or s.upper().startswith("N.C") or s.upper() == "NC":
        return None
    root, rest = _pitch(s)
    if root is None:
        return None
    bass = root
    if "/" in rest:
        rest, b = rest.split("/", 1)
        bp, _ = _pitch(b)
        bass = root if bp is None else bp
    minor = rest.startswith("m") and not rest.startswith("maj")
    third, fifth = (3 if minor else 4), 7
    if "dim" in rest:
        third, fifth = 3, 6
    if "aug" in rest or re.search(r"(?<![0-9])\+(?![0-9])|\+5|#5", rest):
        fifth = 8
    if re.search(r"-5|b5", rest):
        fifth = 6
    if "sus4" in rest:
        third = 5
    elif "sus2" in rest:
        third = 2
    tones = {0: 1.0, third: 1.0, fifth: 1.0}
    body = re.sub(r"\(.*?\)", "", rest)
    seventh = None
    if re.search(r"M7|maj7|△7|M9|maj9", rest):
        seventh = 11
    elif re.search(r"7|9|11|13", body) and "add" not in body:
        seventh = 9 if "dim7" in rest else 10
    if seventh is not None:
        tones[seventh] = 0.5
    if re.search(r"(?<![0-9])6", rest):
        tones[9] = 0.5
    if "9" in rest:
        tones.setdefault(2, 0.5)
    quality = None
    if fifth == 7 and third in (3, 4):
        quality = {(4, None): "maj", (3, None): "min", (4, 10): "dom7", (4, 11): "maj7", (3, 10): "min7",
                   (3, 11): "min"}.get((third, seventh), "maj" if third == 4 else "min")
    return {"root": root, "bass": bass, "tones": tones, "quality": quality, "symbol": symbol.strip()}


def read_chart(path: str | Path) -> list[list[dict]]:
    """The chart's lines with chords, each its chords in order."""
    lines = []
    for raw in Path(path).read_text(encoding="utf-8-sig", errors="replace").splitlines():
        chords = [c for c in (parse_chord(m) for m in CHORD_MARK.findall(raw)) if c is not None]
        if chords:
            lines.append(chords)
    return lines


def model_index(chord: dict, qualities: int) -> int | None:
    """The chart's chord in the model's vocabulary (12 * quality + root), a seventh read as its triad where the
    model has no sevenths; None for a chord the vocabulary has no place for."""
    q = chord["quality"]
    if q is None:
        return None
    if CK.QUALITIES.index(q) >= qualities:
        q = "min" if q == "min7" else "maj"
    return 12 * CK.QUALITIES.index(q) + chord["root"]


# --- lining a chart up with the beats ---------------------------------------------------------------

JUMP = 3.0             # a jump to another line's start
GAP = 0.75             # a beat matched to no chord


def _templates(chords, shift):
    T = np.zeros((len(chords), 12))
    bass = np.zeros(len(chords), int)
    for i, c in enumerate(chords):
        for iv, w in c["tones"].items():
            T[i, (c["root"] + iv + shift) % 12] = max(T[i, (c["root"] + iv + shift) % 12], w)
        bass[i] = (c["bass"] + shift) % 12
        T[i, bass[i]] = max(T[i, bass[i]], 1.0)
    return T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9), bass


def align(x: np.ndarray, chords: list[dict], line_of: np.ndarray, shift: int = 0) -> tuple[np.ndarray, float]:
    """The chart's chords (in order, each one beat or more, a jump to any line's start, beats matched to none)
    against the beats' chroma ``x`` [nb, 36] (chord_key.pool_beats), the chart transposed by ``shift``.
    -> (chord index per beat, -1 for a gap; the path's mean cost on its matched beats)"""
    up = x[:, 12:24] + x[:, 24:36]
    up = up / (np.linalg.norm(up, axis=1, keepdims=True) + 1e-9)
    bas = x[:, :12] / (x[:, :12].max(axis=1, keepdims=True) + 1e-9)
    T, bass = _templates(chords, shift)
    C = (1.0 - T @ up.T) + 0.5 * (1.0 - bas[:, bass].T)                # [n chords, nb]
    n, nb = C.shape
    starts = np.r_[True, line_of[1:] != line_of[:-1]]
    inf = 1e18
    D = np.where(starts, C[:, 0], inf)
    G = GAP
    back = np.zeros((nb, n), np.int32)
    gback = np.zeros(nb, np.int32)
    back[0] = -2
    gback[0] = -2
    for j in range(1, nb):
        best = int(np.argmin(D))
        cand = np.stack([D, np.r_[inf, D[:-1]], np.where(starts, D[best] + JUMP, inf), np.where(starts, G, inf)])
        k = np.argmin(cand, axis=0)
        back[j] = np.where(k == 0, np.arange(n), np.where(k == 1, np.arange(n) - 1, np.where(k == 2, best, -1)))
        newD = cand[k, np.arange(n)] + C[:, j]
        if G <= D[best]:
            G, gback[j] = G + GAP, -1
        else:
            G, gback[j] = D[best] + GAP, best
        D = newD
    out = np.full(nb, -1)
    end = int(np.argmin(D))
    state = end if D[end] < G else -1
    for j in range(nb - 1, -1, -1):
        out[j] = state
        prev = back[j, state] if state >= 0 else gback[j]
        if prev == -2:
            break
        state = int(prev)
    m = out >= 0
    return out, (float(C[out[m], np.flatnonzero(m)].mean()) if m.any() else float("inf"))


# --- the report ----------------------------------------------------------------------------------------

def mmss(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:04.1f}"


def bars_report(model, grid, x, keys_per_bar) -> list[str]:
    """Per bar: its time, the key the analysis gives, the model's chord per beat (a held chord as '-')."""
    _, top = model.beat_chords(x)
    lines = []
    for b in range(int(grid.bar.max()) + 1):
        beats = np.flatnonzero(grid.bar == b)
        names, last = [], None
        for j in beats:
            names.append("-" if top[j] == last else CK.chord_name(top[j]))
            last = top[j]
        key = _key_name(int(keys_per_bar[b]))
        lines.append(f"{mmss(grid.beat_t[beats[0]]):>7}  {key:>9} | " + " ".join(f"{n:<5}" for n in names))
    return lines


def _key_name(k: int) -> str:
    from analyzer.music.key_names import idx_to_labels
    number, name = idx_to_labels(k)
    return f"{name} ({number})"


def chart_report(model, x, chart: list[list[dict]]) -> list[str]:
    """The chart lined up: the best transposition (0 when the chart is in the recording's key), per chart line its
    chords and the model's chord heard most on each one's beats, and how often they agree on the matched beats."""
    flat = [c for line in chart for c in line]
    line_of = np.asarray([i for i, line in enumerate(chart) for _ in line])
    costs = [align(x, flat, line_of, s)[1] for s in range(12)]
    best = int(np.argmin(costs))
    path, cost = align(x, flat, line_of, 0)
    _, top = model.beat_chords(x)
    out = [f"chart: {len(chart)} lines, {len(flat)} chords; matched beats {np.mean(path >= 0):.0%}; mean cost "
           f"{cost:.3f}; the best transposition {best:+d} semitones"
           + ("" if best == 0 else f" (cost {costs[best]:.3f}: the chart may be in another key than the recording)")]
    heard: dict[int, Counter] = {}
    for j in np.flatnonzero(path >= 0):
        heard.setdefault(int(path[j]), Counter())[int(top[j])] += 1
    k = 0
    for i, line in enumerate(chart):
        cells = []
        for c in line:
            h = heard.get(k)
            cells.append(f"{c['symbol']}={CK.chord_name(h.most_common(1)[0][0]) if h else '?'}")
            k += 1
        out.append(f"  {i + 1:3d}  " + "  ".join(cells))
    exact = root = triad = counted = 0
    for j in np.flatnonzero(path >= 0):
        want = model_index(flat[int(path[j])], model.Q)
        if want is None:
            continue
        got = int(top[j])
        counted += 1
        exact += got == want
        root += got % 12 == want % 12
        triad += got % 12 == want % 12 and (CK.QUALITIES[got // 12] in ("min", "min7")) == (
            CK.QUALITIES[want // 12] in ("min", "min7"))
    if counted:
        out.append(f"on {counted} matched beats (sus / dim / aug chords aside): the same chord {exact / counted:.0%}, "
                   f"the same root and third {triad / counted:.0%}, the same root {root / counted:.0%}")
    return out


def track_inputs(uid: str, libpath: str):
    """(the track's title, its beat grid, beat inputs and duration), read through the library's owner."""
    from analyzer.audio.hpss import hpss_audio_with_spectra
    from analyzer.audio.load import fast_load
    from analyzer.music.beat_geometry import downbeat_times
    from core.analysis_library import jobs as analysis_jobs
    from core.library.service import LibraryService
    library = LibraryService(libpath)
    try:
        def read():
            tracks = [t for t in library.call(analysis_jobs.training_tracks) if str(t["uid"]).startswith(uid)]
            if len(tracks) != 1:
                raise ValueError(f"{len(tracks)} library tracks start with {uid}")
            (annotation, error), = library.call(analysis_jobs.training_annotations, [tracks[0]["uid"]])
            if annotation is None:
                raise ValueError(f"the track's analysis does not read: {error}")
            return tracks[0], annotation
        with ThreadPoolExecutor(1) as pool:       # LibraryService.call waits: never on the main thread
            track, a = pool.submit(read).result()
    finally:
        library.close()
    samp = np.ascontiguousarray(fast_load(str(track["audio_path"]), target_sr=CK.SR, stereo=True).mean(axis=1),
                                dtype=np.float32)
    frames = CK.frame_chroma(hpss_audio_with_spectra(samp, CK.SR)[0], CK.SR)
    duration = float(a.duration_sec)
    beats = np.asarray(a.beats, float)
    segs = np.asarray(a.tempo_segments, float) if a.tempo_segments is not None else np.zeros((0, 5))
    grid = CK.beat_grid(beats, downbeat_times(beats, segs) if segs.ndim == 2 and segs.shape[0] else None,
                        np.asarray(a.phrase_starts if a.phrase_starts is not None else [], float), duration)
    return f"{track.get('title') or ''} / {track.get('artist') or ''}", grid, CK.pool_beats(frames, grid, duration), duration


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m training.key_chords", description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("uid", help="a library track's uid, or the start of one")
    parser.add_argument("--chart", help="a chord chart (text, [C] marks) to compare with")
    parser.add_argument("--model", help="a key model file (default: the one in use)")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    from core.workers.analysis_settings import model_file
    from training.__main__ import project_path, read_settings
    cfg = read_settings()
    model = CK.ChordKeyModel(args.model or model_file(cfg.analysisconfig.key_parameter_path, "key_parameter_path"))
    try:
        title, grid, x, duration = track_inputs(args.uid, project_path(cfg.libconfig.libpath).as_posix())
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    L = model.key_loglik(model.chords(x), grid.pos)
    _, keys_per_bar = model.decode(L, grid, duration)
    print(title)
    print("\n".join(bars_report(model, grid, x, keys_per_bar)))
    if args.chart:
        print("\n".join(chart_report(model, x, read_chart(args.chart))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
