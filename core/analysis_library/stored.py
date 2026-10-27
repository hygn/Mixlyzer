"""A track's stored analysis as the app holds it: `StoredAnalysis`, read from its {uid}.npz's arrays, and the other
ways those arrays are made or read. With store.py (the file format) the one place that knows the stored keys and
blocks; the rest of the app reads the fields.

    StoredAnalysis.from_arrays(arrays)   the file's arrays as the app reads them
    arrays_of(result)                     an analysis's results (analyzer.pipeline.track.TrackAnalysis) as stored
    TrackEdits / apply_edits(arrays, e)   the editor's save, the blocks it replaces
    annotation_of(arrays)                 Reoptimize's annotation of a track (analyzer.training.library_tracks)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import numpy as np

from analyzer.audio.wave_overview import WaveOverview
from analyzer.beat.general.tracker import dominant_tempo
from analyzer.music.marks import CuePoint, JumpCueGroup, Phrase
from analyzer.training.library_tracks import Annotation
from core.analysis_library.cue_points import build_cue_points_np, extract_cue_points
from core.analysis_library.jump_cues import build_jump_cue_block, build_jump_cues_np, extract_jump_cue_groups, \
    jump_cue_point_groups
from core.analysis_library.phrases import build_phrase_segments_np, extract_phrase_segments, stored_phrase_rows
from core.marks.jump_cues import check_slot_letters
from core.marks.phrases import assigned_phrases

# the stored keys: scalars and arrays, then the blocks (a block's fields stored flattened, ``block.field``)
DURATION, SAMPLE_RATE, TEMPO, BEATS, TEMPO_ROWS = "duration_sec", "sr", "tempo_global", "beats_time_sec", "tempo_segments"
KEYS, KEYS_AUTO = "key_segments", "key_segments_auto"
WAVE = ("wave_minmax_np", "wave_minmax_ref", "wave_bands_np", "wave_bands_ref")
WAVE_DETAIL_REF = "wave_detail_band_ref"
PHRASES, CUE_POINTS, JUMP_CUES, JUMP_CUES_AUTO = "phrase_segments_np", "cue_points_np", "jump_cues_np", "jump_cues_auto_np"


def _scalar(value: Any, default: Optional[float] = 0.0) -> Optional[float]:
    """A stored scalar (a 0-d or 1-element array), ``default`` when it is not one."""
    if value is None:
        return default
    arr = np.asarray(value)
    try:
        return float(arr.item())
    except Exception:
        try:
            return float(arr.astype(float).flatten()[0])
        except Exception:
            return default


def _has_block(arrays: Mapping, name: str) -> bool:
    return isinstance(arrays.get(name), dict) or any(isinstance(k, str) and k.startswith(name + ".") for k in arrays)


@dataclass(frozen=True)
class StoredAnalysis:
    """A track's stored analysis (the user's marks as last saved, the automatic results beside them, the overview
    waveform). The arrays are as stored; the marks as their stored forms read them (core.track.marks makes the
    editor's forms of them)."""
    duration_sec: float = 0.0
    sample_rate: int = 0                     # the analysis's (0: not stored)
    main_tempo: float = 0.0
    beats: np.ndarray = field(default_factory=lambda: np.empty(0))
    tempo_segments: np.ndarray = field(default_factory=lambda: np.empty((0, 6)))   # rows [start, end, bpm, inizio, ts, bpm_end]
    key_segments: np.ndarray = field(default_factory=lambda: np.empty((0, 4)))     # rows [pitch, minor, start, end]
    key_segments_auto: Optional[np.ndarray] = None     # the automatic key result (None: not kept)
    phrases: tuple[Phrase, ...] = ()                   # as stored (no UNASSIGNED: where no phrase is)
    cue_points: tuple[CuePoint, ...] = ()              # by time
    jump_cues: tuple[JumpCueGroup, ...] = ()           # the user's, their letters as stored
    jump_cues_auto: Optional[tuple[JumpCueGroup, ...]] = None   # the automatic result (None: not kept)
    wave: Optional[WaveOverview] = None

    @classmethod
    def from_arrays(cls, arrays: Optional[Mapping]) -> "StoredAnalysis":
        """The stored analysis of a {uid}.npz's arrays (flattened keys, or blocks as dicts); a part not stored is
        empty (or None where it says so)."""
        a = dict(arrays or {})
        wave = None
        if all(k in a for k in WAVE):
            wave = WaveOverview(*(np.asarray(a[k]) for k in WAVE),
                                np.asarray(a.get(WAVE_DETAIL_REF, np.zeros(0, np.float32))))
        return cls(
            duration_sec=_scalar(a.get(DURATION)),
            sample_rate=int(_scalar(a.get(SAMPLE_RATE))),
            main_tempo=_scalar(a.get(TEMPO)),
            beats=np.asarray(a[BEATS]) if BEATS in a else np.empty(0),
            tempo_segments=np.asarray(a[TEMPO_ROWS]) if TEMPO_ROWS in a else np.empty((0, 6)),
            key_segments=np.asarray(a[KEYS]) if KEYS in a else np.empty((0, 4)),
            key_segments_auto=np.asarray(a[KEYS_AUTO]) if KEYS_AUTO in a else None,
            phrases=tuple(stored_phrase_rows(a)),
            cue_points=tuple(extract_cue_points(a)),
            jump_cues=tuple(extract_jump_cue_groups(a, JUMP_CUES)),
            jump_cues_auto=tuple(extract_jump_cue_groups(a, JUMP_CUES_AUTO)) if _has_block(a, JUMP_CUES_AUTO) else None,
            wave=wave,
        )


def arrays_of(result) -> dict[str, Any]:
    """An analysis's results (analyzer.pipeline.track.TrackAnalysis) as the library stores them; the automatic results
    kept beside the user's marks (which edits replace): the optimizers learn from where the two differ."""
    wave = result.wave_overview
    jump_cues = build_jump_cues_np(result.jump_pairs, canonicalize_labels=True, merge_coincident=True)
    return {
        **dict(zip(WAVE, (wave.minmax, wave.minmax_ref, wave.bands, wave.bands_ref))),
        WAVE_DETAIL_REF: wave.detail_band_ref,
        DURATION: result.duration_sec,
        TEMPO_ROWS: result.tempo_segments,
        BEATS: result.beats,
        TEMPO: result.main_tempo,
        PHRASES: build_phrase_segments_np(result.phrases),
        CUE_POINTS: build_cue_points_np(result.cue_points),
        JUMP_CUES: jump_cues,
        JUMP_CUES_AUTO: {k: np.array(v, copy=True) for k, v in jump_cues.items()},
        KEYS: result.key_segments,
        KEYS_AUTO: np.array(result.key_segments, dtype=float, copy=True),
        SAMPLE_RATE: result.sample_rate,
    }


@dataclass(frozen=True)
class TrackEdits:
    """The editor's save: the marks as edited (each replaces its stored block whole) and a reanalysis's new automatic
    results (None: as stored). ValueError when a JumpCUE's letter is not a slot's (A-Z, no two alike)."""
    beats: np.ndarray
    tempo_segments: np.ndarray
    key_segments: np.ndarray
    phrases: tuple[Phrase, ...]          # UNASSIGNED left out (it is where no phrase is)
    jump_cues: Optional[tuple[JumpCueGroup, ...]]   # None: the stored block as it is (DataModel.jump_cues_left_out)
    cue_points: tuple[CuePoint, ...]
    key_segments_auto: Optional[np.ndarray] = None
    jump_cues_auto: Optional[tuple[JumpCueGroup, ...]] = None

    def __post_init__(self) -> None:
        if self.jump_cues is not None:
            check_slot_letters([group for group in self.jump_cues if group.cues])

    def main_tempo(self) -> float:
        """The track's tempo as the analysis gives it, of the rows as edited (the library's BPM, the NPZ's)."""
        return dominant_tempo(self.tempo_segments)


def _replace_block(arrays: dict, name: str, block: dict) -> None:
    """Replace a stored block (flattened as ``name.<field>``) as a whole, so no field of the previous block is left
    behind."""
    for key in [k for k in arrays if k.startswith(name + ".")]:
        del arrays[key]
    arrays[name] = block


def apply_edits(arrays: dict, edits: TrackEdits) -> None:
    """``edits`` into a track's stored arrays (in place); the main tempo made from the rows as edited."""
    arrays[BEATS] = np.array(edits.beats, dtype=float)
    arrays[TEMPO_ROWS] = np.array(edits.tempo_segments, dtype=float)
    arrays[TEMPO] = edits.main_tempo()
    arrays[KEYS] = np.array(edits.key_segments, dtype=float)
    if edits.key_segments_auto is not None:
        arrays[KEYS_AUTO] = np.array(edits.key_segments_auto, dtype=float)
    if edits.jump_cues is not None:
        _replace_block(arrays, JUMP_CUES, build_jump_cue_block(list(edits.jump_cues)))
    if edits.jump_cues_auto is not None:
        _replace_block(arrays, JUMP_CUES_AUTO, build_jump_cue_block(list(edits.jump_cues_auto)))
    _replace_block(arrays, PHRASES, build_phrase_segments_np(edits.phrases))
    _replace_block(arrays, CUE_POINTS, build_cue_points_np(list(edits.cue_points)))


# what Reoptimize reads of a track's stored arrays: the grids, keys, phrases and JumpCUEs the user annotated (a few kB
# a track; the overview and the rest are not decompressed)
ANNOTATION_KEYS = (BEATS, SAMPLE_RATE, DURATION, TEMPO_ROWS, KEYS, KEYS_AUTO)
ANNOTATION_BLOCKS = (PHRASES + ".", JUMP_CUES + ".", JUMP_CUES_AUTO + ".")
# what the Variants' fingerprint reads (analyzer.search.variants): the beats, the length and the overview
FINGERPRINT_KEYS = (BEATS, DURATION, *WAVE, WAVE_DETAIL_REF)
# what the mix candidates' bars read (analyzer.search.mixability.bars): those and the tempo rows (the downbeats)
MIX_KEYS = (*FINGERPRINT_KEYS, TEMPO_ROWS)
# what library.db's rows of a track are made from (core.library.segments.track_index): the tempo and key rows and the
# phrases
INDEX_KEYS = (DURATION, TEMPO_ROWS, KEYS)
INDEX_BLOCKS = (PHRASES + ".",)


def index_marks(arrays: Mapping) -> tuple[np.ndarray, np.ndarray, list[Phrase]]:
    """The marks library.db's rows of a track are made from (its arrays as stored, or as an edit left them):
    (tempo rows, key rows, the phrases stored)."""
    a = dict(arrays)

    def rows(key, width):
        return np.asarray(a[key], dtype=float) if key in a else np.empty((0, width))
    return rows(TEMPO_ROWS, 6), rows(KEYS, 4), assigned_phrases(extract_phrase_segments(a))


def annotation_of(arrays: Mapping) -> Annotation:
    """A track's annotation (ANNOTATION_KEYS / ANNOTATION_BLOCKS of its stored arrays), the JumpCUEs as each group's
    cue points (an unreadable block: none)."""
    a = dict(arrays)

    def array(key):
        return np.asarray(a[key]) if key in a else None

    def scalar(key, kind):
        value = _scalar(a.get(key), None)
        return kind(value) if value is not None and math.isfinite(value) else None

    def points(block):
        try:
            return jump_cue_point_groups(a, block)
        except Exception:
            return []

    phrase = {f: array(f"{PHRASES}.{f}") for f in ("start", "end", "label")}
    return Annotation(beats=array(BEATS), sample_rate=scalar(SAMPLE_RATE, int), duration_sec=scalar(DURATION, float),
                      tempo_segments=array(TEMPO_ROWS), key_segments=array(KEYS), key_segments_auto=array(KEYS_AUTO),
                      phrase_starts=phrase["start"], phrase_ends=phrase["end"], phrase_labels=phrase["label"],
                      jump_cue_groups=points(JUMP_CUES), jump_cue_auto_groups=points(JUMP_CUES_AUTO))
