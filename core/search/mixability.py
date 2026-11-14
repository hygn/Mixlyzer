"""The mix candidates of the loaded track (the library's sources panel, Mix Candidates): the library's tracks to play
after it (or, ``MixOptions.previous``, before it), each checked against it by the criteria turned on (``MixOptions``:
its BPM, its key; both or either), the nearest tempo first.

A is the track blended out, B the one coming in: the loaded track is A, a candidate B (the next); or a candidate A,
the loaded track B (the previous). Where the two meet: A leaves at its first INTERLUDE / OUTRO after its first chorus
(else its last phrase); B comes in at its top (its first phrase after a lead-in). The BPM and the key compared are each
track's there. Not candidates: the loaded track, its versions (the Variants) and the instrumentals (a title saying so:
the Variants' instrumental relation misreads quiet vocals)."""
from __future__ import annotations

from dataclasses import dataclass

from analyzer.music.key_names import ALPHANUMERIC_LABELS
from core.search import titles

TEMPO_RATIOS = (1.0, 2.0, 0.5)    # B at its tempo, or read at double / half of it (a grid an octave apart)
LEAD = ("LEAD_IN", "SILENCE")
EXITS = ("INTERLUDE", "OUTRO")
CHORUS = "CHORUS"


@dataclass(frozen=True)
class MixOptions:
    """The criteria (deck.json; set at the end of the library's search bar row)."""
    bpm: bool = True
    key: bool = False
    match_all: bool = True       # every criterion turned on (else any)
    tempo_range: float = 8.0     # +- % of A's BPM
    previous: bool = False       # the tracks to play before the loaded one (else after it)


@dataclass(frozen=True)
class Unit:
    """A phrase: start / end (s) and its label."""
    start: float
    end: float
    label: str


@dataclass(frozen=True)
class TrackMarks:
    """What the candidates read of a library track: its title, phrases (``Unit``), tempo rows (start, end, bpm,
    bpm_end) and key rows (start, end, key: 0-11 major, 12-23 minor; None unknown)."""
    uid: str
    title: str
    phrases: tuple
    tempo: tuple
    keys: tuple


@dataclass(frozen=True)
class Candidate:
    """A candidate and where it meets the loaded track: A blended out, B coming in (the module's note)."""
    uid: str
    passed: tuple            # (bpm, key): which criteria it meets
    a_from: float            # where A leaves (s)
    b_from: float            # where B comes in (s)
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
    """Where A leaves: its first INTERLUDE / OUTRO after its first chorus, else its last phrase."""
    chorus = next((i for i, u in enumerate(units) if u.label == CHORUS), None)
    if chorus is not None:
        for u in units[chorus:]:
            if u.label in EXITS:
                return float(u.start)
    return float(units[-1].start) if units else 0.0


def entry_time(units) -> float:
    """Where B comes in: its first phrase after a lead-in."""
    return float(next((u.start for u in units if u.label not in LEAD), units[0].start)) if units else 0.0


def is_instrumental(title: str) -> bool:
    return bool(titles.INSTRUMENTAL.search(title or ""))


def candidates(loaded: TrackMarks, tracks: list[TrackMarks], excluded: set[str],
               options: MixOptions) -> list[Candidate]:
    """The library's tracks (but ``excluded`` and the instrumentals) that meet the criteria turned on (all of them
    with ``match_all``, else any; none turned on: every track), the nearest tempo first (an unknown one last)."""
    out = []
    for other in tracks:
        if other.uid == loaded.uid or other.uid in excluded or is_instrumental(other.title):
            continue
        a, b = (other, loaded) if options.previous else (loaded, other)
        a_from, b_from = exit_time(list(a.phrases)), entry_time(list(b.phrases))
        bpm_a, bpm_b = tempo_at(a.tempo, a_from), tempo_at(b.tempo, b_from)
        offset = tempo_offset(bpm_a, bpm_b)
        key_a, key_b = key_at(a.keys, a_from), key_at(b.keys, b_from)
        passed = (abs(offset) <= options.tempo_range, key_relation(key_a, key_b) in ("same", "relative", "adjacent"))
        on = [p for p, wanted in zip(passed, (options.bpm, options.key)) if wanted]
        if on and not (all(on) if options.match_all else any(on)):
            continue
        out.append(Candidate(other.uid, passed, a_from, b_from, bpm_a, bpm_b, offset, key_a, key_b))
    out.sort(key=lambda c: abs(c.tempo_pct))
    return out
