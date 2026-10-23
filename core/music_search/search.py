"""The search by music: one entry for any query, no mode to pick.

- the landmarks first (a few ms): a recording of the song itself (a phone, a speaker); a song with MIN_VOTES of them at
  one offset is it (a voice gives a handful by chance), and nothing else is searched;
- else, at once on two threads: the voice's pitch (a hum, singing, a whistle told from them: core.music_search.query)
  against the songs' reference lines; and, when the query is no single voice (under VOICE_LIKE of its frames with a
  clear period: a mix), the melody extractor's line (core.music_search.melody, as the references are made) within
  +-RECORDING_RANGE: a recording whose landmarks no longer match (played at another tempo or pitch) still carries
  the song's melody, which a voice's pitch tracker cannot follow in a mix.
The two melody searches' songs are joined by their lower cost (the same cost, semitones a frame). The kernels run on
SEARCH_THREADS threads: a search is made every few seconds beside the GUI and the audio, never all the cores.
"""
from __future__ import annotations

import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numba
import numpy as np

from core.music_search import landmarks, melody, query
from core.music_search.matcher import MelodyIndex
from core.music_search.melody import SR

MIN_VOTES = 8
MIN_VOICED_SEC = 2.0       # less pitch than this: no melody to search by
RECORDING_RANGE = 1.0      # semitones: a recording is in the song's key, a DJ's pitch fader at most a semitone off
MUSIC_MAX_COST = 0.25      # the extractor's line: a song's own melody, read as its reference was, matches within a
                           # quarter semitone a frame (recordings 4 % fast: 16 of 17 found under it), anything else
                           # does not (noise, rumble, mains hum: 0.28 and up; the extractor finds lines in anything)
VOICE_LIKE = 0.6           # a single voice has a clear period most of the time (hums, whistles: 0.79 and up); a mix
                           # rarely does (songs: median 0.16)
SEARCH_THREADS = max(1, min(4, (os.cpu_count() or 4) // 4))
_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="MusicSearch")


@dataclass(frozen=True)
class MusicMatch:
    uid: str
    kind: str              # "recording" (landmarks) or "melody"
    score: float           # votes for a recording; the mean frame cost (semitones, lower better) for a melody
    start_sec: float       # where in the song the query matched
    end_sec: float
    transposition: float   # semitones the song is above the query (a melody's); 0 for a recording
    places: tuple[float, ...] = ()   # every place in the song it matched about as well (a song repeats), best first


@dataclass(frozen=True)
class SearchResult:
    kind: str                       # the best match's: "recording", "melody", or "none"
    matches: tuple[MusicMatch, ...]
    voiced_sec: float               # seconds of pitch in the query (the voice's)


def _recording(con, y, sr, top):
    if con is None:
        return []
    span = y.size / float(sr)
    return [MusicMatch(uid, "recording", float(v), max(0.0, st), max(0.0, st) + span, 0.0, places)
            for v, uid, st, places in landmarks.search(con, y, sr, top=top) if v >= MIN_VOTES]


def _voice(index, pitch, top):
    numba.set_num_threads(SEARCH_THREADS)       # per thread
    return index.search(pitch, top=top)


def _music(index, y, top):
    numba.set_num_threads(SEARCH_THREADS)
    pitch = melody.candidates(melody.harmonic_part(y), 1)[0]
    if float(np.isfinite(pitch).sum()) * melody.HOP / SR < MIN_VOICED_SEC:
        return []
    return [m for m in index.search(pitch, top=top, rng=RECORDING_RANGE) if m.cost <= MUSIC_MAX_COST]


def search_audio(y: np.ndarray, sr: int, index: MelodyIndex | None, con: sqlite3.Connection | None,
                 top: int = 10) -> SearchResult:
    """The best songs for a query (mono float at ``sr``) over ``index`` (the reference lines) and ``con`` (the
    landmarks; store.open_reader's)."""
    y = np.asarray(y, dtype=np.float32)
    found = _recording(con, y, sr, top)
    if found:
        return SearchResult("recording", tuple(found), 0.0)
    if sr != SR:
        import soxr
        y = soxr.resample(y, sr, SR).astype(np.float32)
    pitch = query.yin(y, SR, *(query.WHISTLE if query.is_whistle(y, SR) else query.HUM))[0]
    voiced_frac = float(np.isfinite(pitch).mean()) if pitch.size else 0.0
    voiced = float(np.isfinite(pitch).sum()) * query.HOP / SR
    if index is None or not len(index):
        return SearchResult("none", (), voiced)
    voice = _POOL.submit(_voice, index, pitch, top) if voiced >= MIN_VOICED_SEC else None
    music = _POOL.submit(_music, index, y, top) if voiced_frac < VOICE_LIKE else None
    best: dict[str, object] = {}
    for m in (voice.result() if voice else []) + (music.result() if music else []):
        if m.uid not in best or m.cost < best[m.uid].cost:
            best[m.uid] = m
    joined = [MusicMatch(m.uid, "melody", m.cost, m.start_sec, m.end_sec, m.transposition, m.places)
              for m in sorted(best.values(), key=lambda m: m.cost)][:top]
    return SearchResult("melody" if joined else "none", tuple(joined), voiced)
