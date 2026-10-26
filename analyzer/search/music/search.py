"""The search by music: one entry for any query, no mode to pick.

- the voice's pitch first (a hum, singing, a whistle told from them: analyzer.search.music.query; ~0.1 s), which also
  tells a single voice or instrument (VOICE_LIKE of its frames with a clear period) from a mix;
- the landmarks (tens of ms): a recording of the song itself (a phone, a speaker, a room's microphone); a song with
  MIN_VOTES of them at one offset is it (a single voice: at the index's density, VOICE_MIN_VOTES), and nothing else
  is searched;
- else, at once on three threads: for a mix, the landmarks through a DJ's key-locked tempo (MIN_TEMPO_VOTES); the
  voice's pitch against the songs' reference lines; and, for a mix, the melody extractor's line (melody.candidates of melody.harmonic_part,
  read as the references are) within +-RECORDING_RANGE: a recording whose landmarks no longer match (played at
  another tempo or pitch) still carries the song's melody, which a voice's pitch tracker cannot follow in a mix.
A tempo landmark match is the result; else the two melody searches' songs are joined by their lower cost (the same
cost, semitones a frame). The kernels run on SEARCH_THREADS threads: a search is made every few seconds beside the
GUI and the audio, never all the cores.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import numba
import numpy as np

from analyzer.search.music import landmarks, melody, query
from analyzer.search.music.landmarks import LandmarkIndex
from analyzer.search.music.matcher import MelodyIndex
from analyzer.search.music.melody import SR

QUERY_PEAKS_PER_SEC = 30   # the query's landmarks, three times the index's: a crowd's calls and screams take a
QUERY_FAN_OUT = 12         # second's strongest peaks, so a query keeping more (paired with more, the index's pairs
                           # among them) keeps the song's (a DJ set's room microphone: 0.44 -> 0.73 joined)
MIN_VOTES = 10             # a mix's chance votes at that density: hums, whistles, a piano at most 9 over 385 songs
MIN_TEMPO_VOTES = 12       # through the tempo factors: 16 tries, so a wrong song reaches more by chance
VOICE_MIN_VOTES = 8        # a single voice or instrument (VOICE_LIKE) is searched at the index's density only (no
                           # crowd to see through, and its few strong partials pair by chance: at most 5 there)
MIN_VOICED_SEC = 2.0       # less pitch than this: no melody to search by
QUERY_HOP = 2 * melody.HOP  # the voice's pitch every 23 ms (the search's frames are 0.1 s and up): as found as at
                            # 11.6 ms (hum 0.64 / 0.63, whistle 0.65 / 0.67, piano 0.65 / 0.67), 586 -> 369 ms a search
RECORDING_RANGE = 1.0      # semitones: a recording is in the song's key, a DJ's pitch fader at most a semitone off
MUSIC_MAX_COST = 0.25      # the extractor's line: a song's own melody, read as its reference was, matches within a
                           # quarter semitone a frame (recordings 4 % fast: 16 of 17 found under it), anything else
                           # does not (noise, rumble, mains hum: 0.28 and up; the extractor finds lines in anything)
VOICE_LIKE = 0.6           # a single voice has a clear period most of the time (hums, whistles: 0.79 and up); a mix
                           # rarely does (songs: median 0.16)
SEARCH_THREADS = max(1, min(4, (os.cpu_count() or 4) // 4))
_POOL = ThreadPoolExecutor(max_workers=3, thread_name_prefix="MusicSearch")


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


def _recording(landmark_index, y, sr, top, least, dense=True):
    span = y.size / float(sr)
    found = landmarks.search(landmark_index, y, sr, top, *((QUERY_PEAKS_PER_SEC, QUERY_FAN_OUT) if dense else
                                                (landmarks.PEAKS_PER_SEC, landmarks.FAN_OUT)))
    return [MusicMatch(uid, "recording", float(v), max(0.0, st), max(0.0, st) + span, 0.0, places)
            for v, uid, st, places in found if v >= least]


def _recording_tempo(landmark_index, y, sr, top, least):
    span = y.size / float(sr)
    found = landmarks.search_tempo(landmark_index, y, sr, top, per_sec=QUERY_PEAKS_PER_SEC, fan_out=QUERY_FAN_OUT)
    return [MusicMatch(uid, "recording", float(v), max(0.0, st), max(0.0, st) + span * f, 0.0, places)
            for v, uid, st, places, f in found if v >= least]


def _voice(index, pitch, top):
    numba.set_num_threads(SEARCH_THREADS)       # per thread
    return index.search(pitch, top=top, hop_sec=QUERY_HOP / SR)


def _music(index, y, top):
    numba.set_num_threads(SEARCH_THREADS)
    pitch = melody.candidates(melody.harmonic_part(y), 1)[0]     # read as the references are
    hop_sec = melody.HOP / SR
    if float(np.isfinite(pitch).sum()) * hop_sec < MIN_VOICED_SEC:
        return []
    return [m for m in index.search(pitch, top=top, rng=RECORDING_RANGE, hop_sec=hop_sec)
            if m.cost <= MUSIC_MAX_COST]


def search_audio(y: np.ndarray, sr: int, index: MelodyIndex | None, landmark_index: LandmarkIndex | None,
                 top: int = 10) -> SearchResult:
    """The best songs for a query (mono float at ``sr``) over ``index`` (the reference lines) and ``landmark_index`` (the
    landmarks: the app's store lends them)."""
    y = np.asarray(y, dtype=np.float32)
    if sr != SR:
        import soxr
        y = soxr.resample(y, sr, SR).astype(np.float32)
    pitch = query.yin(y, SR, *(query.WHISTLE if query.is_whistle(y, SR) else query.HUM), hop=QUERY_HOP)[0]
    voiced_frac = float(np.isfinite(pitch).mean()) if pitch.size else 0.0
    voiced = float(np.isfinite(pitch).sum()) * QUERY_HOP / SR
    # a single voice (or an instrument) gets more landmark votes by chance than a mix: its few strong partials
    voice_like = voiced_frac >= VOICE_LIKE
    if landmark_index is not None:
        found = (_recording(landmark_index, y, SR, top, VOICE_MIN_VOTES, dense=False) if voice_like else
                 _recording(landmark_index, y, SR, top, MIN_VOTES))
        if found:
            return SearchResult("recording", tuple(found), voiced)
    if index is None or not len(index):
        return SearchResult("none", (), voiced)
    mix = landmark_index is not None and not voice_like
    tempo = _POOL.submit(_recording_tempo, landmark_index, y, SR, top, MIN_TEMPO_VOTES) if mix else None
    voice = _POOL.submit(_voice, index, pitch, top) if voiced >= MIN_VOICED_SEC else None
    music = _POOL.submit(_music, index, y, top) if not voice_like else None
    found = tempo.result() if tempo else []
    if found:
        for job in (voice, music):
            if job:
                job.result()          # one search at a time: the next must not find these still running
        return SearchResult("recording", tuple(found), voiced)
    best: dict[str, object] = {}
    for m in (voice.result() if voice else []) + (music.result() if music else []):
        if m.uid not in best or m.cost < best[m.uid].cost:
            best[m.uid] = m
    joined = [MusicMatch(m.uid, "melody", m.cost, m.start_sec, m.end_sec, m.transposition, m.places)
              for m in sorted(best.values(), key=lambda m: m.cost)][:top]
    return SearchResult("melody" if joined else "none", tuple(joined), voiced)
