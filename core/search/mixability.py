"""The mix candidates of the loaded track (the library's sources panel, Mix Candidates): the library's tracks to play
after it (or, ``MixOptions.previous``, before it), each checked against it by the criteria turned on (``MixOptions``:
its BPM, its key, its sound; all of them or any) and ranked by how alike the two sound where they would be blended
(analyzer.search.mixability).

A is the track blended out, B the one coming in: the loaded track is A, a candidate B (the next); or a candidate A,
the loaded track B (the previous). Where the two meet: A leaves where the best blend found ends (else at its first
INTERLUDE / OUTRO after its first chorus, else its last phrase); B comes in where that blend starts on it (else its
top, its first phrase after a lead-in). The BPM and the key compared are each track's there. Not candidates: the
loaded track, its versions (the Variants) and the instrumentals (a title saying so: the Variants' instrumental
relation misreads quiet vocals)."""
from __future__ import annotations

import re
from dataclasses import dataclass

from analyzer.music.key_names import ALPHANUMERIC_LABELS
from analyzer.search import mixability as mx

# a blend's score (analyzer.search.mixability.best_overlap) at most this: alike enough for the sound criterion (the
# median of random pairs of a 420-track library: more alike than half of them)
AUDIO_PASS = 1.08
# the rank: the blend's score and the tempo's distance, this many % counting as one component's typical distance
# (the user's recorded sets moved the tempo 1.1 % at the median; on their playlists the track played next ranked 27th
# at the median of ~150 within +-8 %, 46th by the sound alone)
TEMPO_RANK_UNIT = 8.0
TEMPO_RATIOS = (1.0, 2.0, 0.5)    # B at its tempo, or read at double / half of it (a grid an octave apart)
INSTRUMENTAL = re.compile(r"instrumental|off[\s_-]?vocal|karaoke|\binst\b|オフボーカル|カラオケ|インスト", re.I)


@dataclass(frozen=True)
class MixOptions:
    """The criteria (deck.json; set at the end of the library's search bar row)."""
    bpm: bool = True
    key: bool = False
    audio: bool = True
    match_all: bool = True       # every criterion turned on (else any)
    tempo_range: float = 8.0     # +- % of A's BPM
    previous: bool = False       # the tracks to play before the loaded one (else after it)


@dataclass(frozen=True)
class TrackMarks:
    """What the candidates read of a library track: its title, phrases (``mx.Unit``), tempo rows (start, end, bpm,
    bpm_end) and key rows (start, end, key: 0-11 major, 12-23 minor; None unknown)."""
    uid: str
    title: str
    phrases: tuple
    tempo: tuple
    keys: tuple


@dataclass(frozen=True)
class Candidate:
    """A candidate and its blend with the loaded track: A blended out, B coming in (the module's note)."""
    uid: str
    passed: tuple            # (bpm, key, audio): which criteria it meets
    score: float | None      # the blend's (lower: more alike); None: no blend fits the phrases (or no bars)
    a_from: float            # the blend on A: from, to (s); A leaves at a_to
    a_to: float
    b_from: float            # B from (s)
    bars: int                # the blend's length (0: none found)
    bpm_a: float
    bpm_b: float
    tempo_pct: float         # B's BPM against A's (at the nearest of TEMPO_RATIOS), %
    key_a: int | None
    key_b: int | None


def tempo_at(rows, t: float) -> float:
    """The BPM at ``t`` of tempo rows (start, end, bpm, bpm_end: a ramp between them); 0 without rows."""
    if not rows:
        return 0.0
    for start, end, bpm, bpm_end in rows:
        if start - 1e-6 <= t < end:
            if bpm_end and end > start:
                return float(bpm + (bpm_end - bpm) * (t - start) / (end - start))
            return float(bpm)
    return float(rows[-1][2] if t >= rows[-1][0] else rows[0][2])


def key_at(rows, t: float) -> int | None:
    """The key at ``t`` (the nearest row when none holds it)."""
    if not rows:
        return None
    holding = [k for start, end, k in rows if start - 1e-6 <= t < end]
    if holding:
        return holding[0]
    return min(rows, key=lambda r: min(abs(r[0] - t), abs(r[1] - t)))[2]


def key_relation(a: int | None, b: int | None) -> str:
    """How B's key stands to A's on the Alphanumeric wheel: "same", "relative" (the other letter, the same number),
    "adjacent" (+-1, the same letter), "other"; "" when one is unknown."""
    if a is None or b is None:
        return ""
    la, lb = ALPHANUMERIC_LABELS[int(a) % 24], ALPHANUMERIC_LABELS[int(b) % 24]
    na, nb = int(la[:-1]), int(lb[:-1])
    if la == lb:
        return "same"
    if na == nb:
        return "relative"
    if la[-1] == lb[-1] and (na - nb) % 12 in (1, 11):
        return "adjacent"
    return "other"


def tempo_offset(bpm_a: float, bpm_b: float) -> float:
    """B's BPM against A's, %, at the nearest of TEMPO_RATIOS (inf when one is unknown)."""
    if bpm_a <= 0 or bpm_b <= 0:
        return float("inf")
    return min((100.0 * (bpm_b * r / bpm_a - 1.0) for r in TEMPO_RATIOS), key=abs)


def exit_time(units) -> float:
    """Where A leaves without a blend found: its first INTERLUDE / OUTRO after its first chorus, else its last
    phrase."""
    chorus = next((i for i, u in enumerate(units) if u.label == mx.CHORUS), None)
    if chorus is not None:
        for u in units[chorus:]:
            if u.label in mx.EXITS:
                return float(u.start)
    return float(units[-1].start) if units else 0.0


def entry_time(units) -> float:
    return float(units[mx.entry_head(list(units))].start) if units else 0.0


def is_instrumental(title: str) -> bool:
    return bool(INSTRUMENTAL.search(title or ""))


def candidates(loaded: TrackMarks, tracks: list[TrackMarks], bars: dict, excluded: set[str],
               options: MixOptions) -> list[Candidate]:
    """The library's tracks (but ``excluded`` and the instrumentals) that meet the criteria turned on (all of them
    with ``match_all``, else any; none turned on: every track), best first: a blend found, by its score and the
    tempo's distance (TEMPO_RANK_UNIT); then the rest by the tempo's distance."""
    out = []
    for other in tracks:
        if other.uid == loaded.uid or other.uid in excluded or is_instrumental(other.title):
            continue
        a, b = (other, loaded) if options.previous else (loaded, other)
        A, B = bars.get(a.uid), bars.get(b.uid)
        a_units = list(a.phrases)
        found = mx.best_overlap(A, B, a_units, list(b.phrases)) if A is not None and B is not None else None
        if found is not None:
            a_from, a_to = float(A.start[found.a_bar]), float(A.end[found.a_bar + found.bars - 1])
            b_from = float(B.start[found.b_bar])
            bpm_a = sum(A.bpm(found.a_bar + k) for k in range(found.bars)) / found.bars
            bpm_b = sum(B.bpm(found.b_bar + k) for k in range(found.bars)) / found.bars
        else:
            a_from = a_to = exit_time(a_units)
            b_from = entry_time(b.phrases)
            bpm_a, bpm_b = tempo_at(a.tempo, a_from), tempo_at(b.tempo, b_from)
        offset = tempo_offset(bpm_a, bpm_b)
        key_a, key_b = key_at(a.keys, a_from), key_at(b.keys, b_from)
        relation = key_relation(key_a, key_b)
        passed = (abs(offset) <= options.tempo_range, relation in ("same", "relative", "adjacent"),
                  found is not None and found.score <= AUDIO_PASS)
        on = [p for p, wanted in zip(passed, (options.bpm, options.key, options.audio)) if wanted]
        if on and not (all(on) if options.match_all else any(on)):
            continue
        out.append(Candidate(other.uid, passed, found.score if found is not None else None, a_from, a_to, b_from,
                             found.bars if found is not None else 0, bpm_a, bpm_b, offset, key_a, key_b))
    out.sort(key=lambda c: (c.score is None, (c.score or 0.0) + abs(c.tempo_pct) / TEMPO_RANK_UNIT))
    return out
