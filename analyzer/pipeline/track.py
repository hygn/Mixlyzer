"""The full analysis of a track, every stage in order: the audio decoded once (stereo, its mono mix) -> HPSS -> the
overview waveform (its own 48 kHz decode) -> the frame features and the beat tracker -> phrases and their CUE points
-> JumpCUEs -> keys -> Search by Music's index entry -> the Variants' fingerprints -> the mix candidates' beats'
features -> TrackAnalysis. What reads none of the steps before it but the audio, the HPSS or the beats runs beside
them on threads of its own: the overview waveform, the keys' frame chroma (from the harmonic part alone), the phrase
features' frame-level part, a stacked phrase model's song cues' frame-level part, Search by Music's index entry and
what JumpCUE reads of the audio and the beats; each is
the same function on the same input, so the results are those of the steps in order.

It reads the audio only: where the results are stored, and in what form, is the caller's."""

from __future__ import annotations

import importlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from analyzer.audio.hpss import HPSS_HOP_LENGTH, HPSS_N_FFT, hpss_audio_with_spectra
from analyzer.audio.load import fast_load
from analyzer.audio.wave_overview import WAVE_OVERVIEW_SAMPLE_RATE, WaveOverview, build_wave_overview
from analyzer.beat import melody_contour
from analyzer.beat.frame_features import FRAME_HOP_LENGTH, extract_frame_features
from analyzer.jumpcue.JumpCUE import JumpCueConfig, JumpCueEngine, audio_context as jump_audio_context
from analyzer.jumpcue.autocue import autocue_jump_pair
from analyzer.jumpcue.jump_point_model import jump_point_beat_features
from analyzer.key.chord_key import analyze_keys as analyze_chord_keys, frame_chroma as key_frame_chroma
from analyzer.music.beat_geometry import downbeat_times
from analyzer.music.marks import CuePoint, Phrase
from analyzer.phrase import detect_phrase_segments
from analyzer.phrase.cue_points import autocue_time_sec, phrase_cue_points
from analyzer.phrase.phrase_analyzer import phrase_cue_frames, phrase_song_frames
from analyzer.pipeline import stages
from analyzer.pipeline.stages import Progress, report
from analyzer.runtime import prime_physical_core_count
from analyzer.search import mixability, variants
from analyzer.search.mixability import BeatFeatures
from analyzer.search.music.index import IndexEntry, index_audio
from analyzer.settings import ANALYSIS_SAMPLE_RATE, AnalysisSettings

# sklearn would otherwise launch powershell.exe to count cores in every analysis process.
prime_physical_core_count()

# librosa's modules the analysis reaches, which librosa imports at their first use: imported on one thread before the
# analysis's threads start (two threads importing one package's modules at once can meet in an import deadlock), and
# by the resident analysis processes while they wait (core.workers.analysis).
LAZY_MODULES = (
    "librosa.core.audio", "librosa.core.constantq", "librosa.core.pitch", "librosa.core.spectrum",
    "librosa.feature.spectral", "librosa.feature.utils", "librosa.filters", "librosa.onset", "librosa.segment",
)

@dataclass(frozen=True)
class TrackAnalysis:
    duration_sec: float
    sample_rate: int                       # the analysis's (ANALYSIS_SAMPLE_RATE)
    wave_overview: WaveOverview
    tempo_segments: np.ndarray             # float32 rows [start, end, bpm, inizio, ts, bpm_end] covering the track
    beats: np.ndarray                      # the beats the rows make
    main_tempo: float
    phrases: list[Phrase]                  # as detected ([]: none)
    cue_points: list[CuePoint]             # by the phrases' rules (and AutoCUE's)
    jump_pairs: list[dict]                 # the JumpCUE pairs (AutoCUE's last, with AutoCUE on JumpCUEs)
    key_segments: np.ndarray               # [n, 4] (pitch class, mode, start, end); the track's key is the app's,
                                           # made from them and the phrases (analyzer.music.key_rows.main_key)
    music_index: IndexEntry | None         # Search by Music's entry (None: not asked, or it failed)
    variants: list[Any] | None             # the Variants' fingerprints (search.variants.track_fingerprints; None: failed)
    mix_features: BeatFeatures | None = None   # the mix candidates' beats' features on ``beats`` (None: failed)


def analyze_track(path: str, settings: AnalysisSettings, progress: Progress = None) -> TrackAnalysis:
    for name in LAZY_MODULES:
        importlib.import_module(name)
    # the steps beside the others (the module's docstring); an error ends the analysis once they have ended
    with ThreadPoolExecutor(max_workers=5, thread_name_prefix="analysis-side") as side:
        return _analyze_track(path, settings, progress, side)


def _wave_overview(path: str, settings: AnalysisSettings) -> WaveOverview:
    """The overview waveform: the min / max of each column and the power of 12 log-spaced bands (low / mid / high are
    summed from them when drawn), from a mono decode at WAVE_OVERVIEW_SAMPLE_RATE, not the analysis rate, so the
    bands are not cut at the analysis Nyquist; and the detail waveform's three-band scale (these ranges)."""
    print("[Waveform] Analysis Initalized")
    wave = build_wave_overview(
        fast_load(path, WAVE_OVERVIEW_SAMPLE_RATE),
        WAVE_OVERVIEW_SAMPLE_RATE,
        tuple((float(low), float(high)) for low, high in settings.wave_bands),
        n_fft=HPSS_N_FFT,
        hop_length=HPSS_HOP_LENGTH,
    )
    print("[Waveform] Analysis Finished")
    return wave


def _analyze_track(path: str, settings: AnalysisSettings, progress: Progress, side: ThreadPoolExecutor) -> TrackAnalysis:
    models = settings.models
    report(progress, stages.LOAD, 0.0)
    print("[Analyzer] Analysis Initalized")
    wave_job = side.submit(_wave_overview, path, settings)
    global_sr = ANALYSIS_SAMPLE_RATE
    print("[Analyzer] Loading Track")
    # Decode once: stereo for the structure detector, mono derived for the rest.
    samp_stereo = fast_load(path, target_sr=global_sr, stereo=True)
    samp = np.ascontiguousarray(samp_stereo.mean(axis=1), dtype=np.float32)
    # a stacked phrase model's song cues, their frame-level part (None for a model without them), beside the rest
    phrase_cues_job = side.submit(phrase_cue_frames, samp, int(global_sr), model_path=models.phrase)

    report(progress, stages.HPSS, 0.1)
    print("[Analyzer] Processing HPSS")
    # The spectra are reused by the phrase features (same mono mix, STFT and HPSS).
    y_harm, y_perc, hpss_spectra = hpss_audio_with_spectra(samp, int(global_sr))
    # the keys' frame chroma, from the harmonic part alone (the keys are found on the beats and phrases below)
    key_frames_job = side.submit(key_frame_chroma, y_harm, int(global_sr))
    # the phrase features' frame-level part, from the audio and its HPSS (synchronized to the beats below)
    phrase_frames_job = side.submit(phrase_song_frames, samp_stereo, int(global_sr), model_path=models.phrase,
                                    hpss=hpss_spectra)
    duration_sec = samp.size / float(global_sr)

    report(progress, stages.WAVEFORM, 0.2)
    report(progress, stages.TEMPO, 0.30)
    print("[Tempo] Analysis Initalized")
    # Frame features on the shared frame grid: the phrase features read them below.
    frame_features = extract_frame_features(
        samp, int(global_sr), y_perc, y_harm, hop_length=FRAME_HOP_LENGTH
    )
    # Search by Music's index entry, beside the beats and phrases: from the decoded stereo and the HPSS's harmonic
    # part and mask (its own STFT frames are these: no second HPSS), and the melody contours the frame features have
    # just found in that part (melody_contour.take_contours)
    music_job = None
    if settings.music_index and hpss_spectra is not None:
        music_mask = (hpss_spectra.harmonic / np.maximum(hpss_spectra.magnitude, 1e-12)).astype(np.float32)
        music_job = side.submit(index_audio, samp_stereo, y_harm, music_mask)
        del music_mask
    # The generalized tracker (analyzer.beat.general): a frame TCN (three trainings) ->
    # section decoding with one section cost -> ramps -> bar-supported phase -> metronome critic ->
    # tempo-change boundaries on the local onset -> drifting sections -> junction rules -> bars from
    # two downbeat observers -> the consensus of the three; rows with inizio on a downbeat.
    from analyzer.beat.general import (
        FRAME_HOP_LENGTH as TRACKER_HOP, SAMPLE_RATE as TRACKER_SR, dominant_tempo, load_models,
        track as track_beats,
    )
    from analyzer.beat.learned_onset import LazyRefineOnset

    # Its models were trained on these frames: the HPSS parts at the analysis rate, hop 256.
    assert (int(global_sr), FRAME_HOP_LENGTH) == (TRACKER_SR, TRACKER_HOP)
    tracker_frames = frame_features
    frame_times = np.asarray(tracker_frames.frame_times, dtype=float)
    # the tempo-change onset, made when a tempo change first reads it (a track of one tempo never does)
    refine_onset = (
        LazyRefineOnset(tracker_frames, models.refine_onset),
        float(np.median(np.diff(frame_times))) if frame_times.size > 1 else TRACKER_HOP / TRACKER_SR,
        float(frame_times[0]),
    )
    dynamic = bool(settings.bpm_dynamic)
    tracked = track_beats(
        tracker_frames, float(settings.bpm_min),
        load_models(models.downbeat_ctx, models.beat_tcn, models.beat_critic, models.beat_decoder),
        refine_onset,
        dynamic=dynamic, dynamic_downbeat=dynamic or bool(settings.dynamic_downbeat),
    )
    seg_arr = np.asarray(tracked.segments, dtype=np.float64).copy()
    if seg_arr.size:
        seg_arr[0, 0] = 0.0
        seg_arr[-1, 1] = max(float(seg_arr[-1, 1]), duration_sec)
    seg_arr = seg_arr.astype(np.float32)
    # The stored rows are the grid (the editor and the views rebuild it from them): the saved beats are theirs.
    from analyzer.music.tempo_rows import rebuild_grid_from_segments
    beats_out = rebuild_grid_from_segments(seg_arr)
    beats_out = beats_out[(beats_out >= 0.0) & (beats_out <= duration_sec)]
    main_tempo = dominant_tempo(seg_arr)
    print(
        f"[Tempo] consensus kept seed {tracked.seed}; {seg_arr.shape[0]} tempo rows, "
        f"{beats_out.size} beats, main tempo {main_tempo:.2f}"
    )
    for i, row in enumerate(seg_arr):
        start, end, bpm, inizio, ts = [float(x) for x in row[:5]]
        bpm_end = float(row[5]) if row.size > 5 else bpm
        ramp = f"->{bpm_end:.2f}" if abs(bpm_end - bpm) > 1e-6 else ""
        print(f"[Tempo]   tempo{i}: {start:.3f}-{end:.3f}s bpm={bpm:.2f}{ramp} inizio(downbeat)={inizio:.3f}s ts={int(ts)}/4")
    print("[Tempo] Analysis Finished")

    beats_time_arr = np.asarray(beats_out, dtype=float)
    # what JumpCUE reads of the audio and the beats alone, made beside the phrases (JumpCUE.audio_context)
    jump_config = JumpCueConfig(jump_point_model_path=models.jumpcue)
    if settings.autocue == "jumpcue":
        # one hot cue slot stays free for the AutoCUE JumpCUE (below)
        jump_config = replace(jump_config, hot_cue_slots=jump_config.hot_cue_slots - 1)
    jump_mix_magnitude = (
        hpss_spectra.magnitude
        if hpss_spectra is not None
        and (hpss_spectra.n_fft, hpss_spectra.hop_length) == (jump_config.vocal.n_fft, jump_config.vocal.hop)
        else None
    )
    jump_context_job = None
    if beats_time_arr.size >= 2:
        jump_context_job = side.submit(jump_audio_context, jump_config, y_harm, global_sr, beats_time_arr, y_mix=samp,
                                       mix_magnitude=jump_mix_magnitude)
    report(progress, stages.PHRASE, 0.58)
    print("[Phrase] Analysis Initalized")
    # The phrase boundary probability is a JumpCUE feature (below).
    phrase_boundary_prob = None
    phrase_feature_z = None
    phrase_segments = []
    if beats_time_arr.size >= 17 and seg_arr.shape[0] > 0:
        try:
            # Joint phrase boundary + functional-label detection -> Phrase data.
            # Reuse the already-decoded stereo audio (no second decode).
            phrase_segments, phrase_boundary_prob, phrase_feature_z = detect_phrase_segments(
                samp_stereo,
                int(global_sr),
                beats_time_arr,
                seg_arr,
                model_path=models.phrase,
                hpss=hpss_spectra,
                return_boundary_probability=True,
                return_feature_z=True,
                frames=phrase_frames_job.result(),
                cues=phrase_cues_job.result(),
            )
        except Exception as exc:
            import traceback

            print(f"[Phrase] joint detection skipped: {exc}")
            traceback.print_exc()
            phrase_segments = []
    else:
        print("[Phrase] skipped: too few beats")
    # AutoCUE (analyzer.phrase.cue_points): lead-in end, an early first downbeat, or 0:00 when the track begins
    # before a usable downbeat.
    autocue = settings.autocue
    autocue_target = autocue_time_sec(phrase_segments, beats_time_arr, seg_arr)
    cue_points = phrase_cue_points(phrase_segments, beats_time_arr, seg_arr, settings.cue_rules, autocue)
    labels_summary = ", ".join(str(s.label) for s in phrase_segments)
    print(
        f"[Phrase] segments={len(phrase_segments)} "
        f"cue_points={len(cue_points)} labels=[{labels_summary}]"
    )
    print("[Phrase] Analysis Finished")

    # Learned JumpCUE detection needs phrase-boundary probability.  The simple
    # AutoCUE JumpCUE does not and is still added below when detection is skipped.
    report(progress, stages.JUMPCUE, 0.62)
    jump_pairs = []
    if phrase_boundary_prob is None:
        print("[JumpCUE] learned pairs skipped: no phrase boundary probability")
    elif beats_time_arr.size >= 2:
        # a failure leaves the track without learned JumpCUEs (the AutoCUE one still added), never the analysis undone
        try:
            jump_engine = JumpCueEngine(jump_config)
            jump_result = jump_engine.run(
                y_harm=y_harm,
                sr=global_sr,
                beats_time=beats_time_arr,
                y_mix=samp,
                tempo_segments=seg_arr,
                mix_magnitude=jump_mix_magnitude,
                jump_point_features=jump_point_beat_features(
                    frame_features, beats_time_arr, seg_arr, phrase_boundary_prob,
                    audio=samp_stereo, phrase_feature_z=phrase_feature_z,
                ),
                context=jump_context_job.result(),
            )
            jump_pairs = [pair.as_dict() for pair in jump_result.pairs]
        except Exception as exc:
            import traceback

            print(f"[JumpCUE] detection skipped: {exc}")
            traceback.print_exc()
            jump_pairs = []
        if jump_pairs:
            print("[JumpCUE] pairs", jump_pairs)
        else:
            print("[JumpCUE] no jump-compatible pairs detected")
    if autocue == "jumpcue":
        jump_pairs = [*jump_pairs, autocue_jump_pair(autocue_target)]
    hpss_spectra = None

    report(progress, stages.KEY, 0.7)
    print("[Key] Analysis Initalized")
    try:
        # chord-aware keys on the bars and phrases (analyzer/key/chord_key.py)
        _key_per_beat, key_segments = analyze_chord_keys(
            y_harm,
            int(global_sr),
            beats_time_arr,
            downbeat_times(beats_time_arr, seg_arr),
            [float(s.start) for s in phrase_segments],
            float(duration_sec),
            models.key,
            dynamic=bool(settings.key_dynamic),
            phrases=phrase_segments,
            frames=key_frames_job.result(),
        )
    except Exception as exc:
        import traceback

        print(f"[Key] analysis skipped: {exc}")
        traceback.print_exc()
        key_segments = np.zeros((0, 4), float)
    print("[Key] Analysis Finished")

    # Search by Music's index (made beside the beats and phrases). A failure leaves the track unindexed, never the
    # analysis undone.
    music_entry = None
    if music_job is not None:
        report(progress, stages.MUSIC_INDEX, 0.78)
        try:
            music_entry = music_job.result()
        except Exception as exc:
            print(f"[Music search] indexing skipped: {exc}")
    melody_contour.forget_contours()      # the frame features' contours, kept for the index
    wave = wave_job.result()

    # the Variants' fingerprints (from the beats and the overview's bands, ~20 ms); a failure leaves the track out of
    # the Variants, never the analysis undone
    try:
        fingerprints = variants.track_fingerprints(beats_out, wave, duration_sec)
    except Exception as exc:
        print(f"[Variants] not fingerprinted: {exc}")
        fingerprints = None
    # the mix candidates' beats' features (the frame features' chroma and percussive bands pooled into the beats,
    # ~20 ms); a failure leaves the track out of the candidates, never the analysis undone
    try:
        mix_features = mixability.beat_features(
            beats_out, frame_features.frame_times, frame_features.chroma, frame_features.percussive_bands)
    except Exception as exc:
        print(f"[Mix candidates] beats' features not made: {exc}")
        mix_features = None

    print("[Analyzer] Analysis Finished")
    return TrackAnalysis(
        duration_sec=duration_sec,
        sample_rate=global_sr,
        wave_overview=wave,
        tempo_segments=seg_arr,
        beats=beats_out,
        main_tempo=main_tempo,
        phrases=phrase_segments,
        cue_points=cue_points,
        jump_pairs=jump_pairs,
        key_segments=key_segments,
        music_index=music_entry,
        variants=fingerprints,
        mix_features=mix_features,
    )
