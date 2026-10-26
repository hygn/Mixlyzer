"""The full analysis of a track, every stage in order: the audio decoded once (stereo, its mono mix) -> HPSS -> the
overview waveform (its own 48 kHz decode) -> the frame features and the beat tracker -> phrases and their CUE points
-> JumpCUEs -> keys -> Search by Music's index (given by the caller) -> TrackAnalysis.

It reads the audio only: where the results are stored, and in what form, is the caller's."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Optional

import numpy as np

from analyzer.audio.hpss import HPSS_HOP_LENGTH, HPSS_N_FFT, hpss_audio_with_spectra
from analyzer.audio.load import fast_load
from analyzer.audio.wave_overview import WAVE_OVERVIEW_SAMPLE_RATE, build_wave_overview
from analyzer.beat import melody_contour
from analyzer.beat.frame_features import FRAME_HOP_LENGTH, extract_frame_features
from analyzer.jumpcue.JumpCUE import JumpCueConfig, JumpCueEngine
from analyzer.jumpcue.autocue import autocue_jump_pair
from analyzer.jumpcue.jump_point_model import jump_point_beat_features
from analyzer.key.chord_key import analyze_keys as analyze_chord_keys
from analyzer.music.beat_geometry import downbeat_times
from analyzer.phrase import detect_phrase_segments
from analyzer.phrase.cue_points import autocue_time_sec, phrase_cue_points
from analyzer.pipeline import stages
from analyzer.pipeline.stages import Progress, report
from analyzer.runtime import prime_physical_core_count
from analyzer.settings import ANALYSIS_SAMPLE_RATE, AnalysisSettings

# sklearn would otherwise launch powershell.exe to count cores in every analysis process.
prime_physical_core_count()

# Search by Music's index of the track, made by the caller from the decoded stereo, the HPSS's harmonic part and
# its harmonic mask (None: no index)
IndexMusic = Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]], Any]


@dataclass(frozen=True)
class TrackAnalysis:
    duration_sec: float
    sample_rate: int                       # the analysis's (ANALYSIS_SAMPLE_RATE)
    wave_overview: dict[str, np.ndarray]   # analyzer.audio.wave_overview's arrays
    tempo_segments: np.ndarray             # float32 rows [start, end, bpm, inizio, ts, bpm_end] covering the track
    beats: np.ndarray                      # the beats the rows make
    main_tempo: float
    phrases: list[dict]                    # {start, end, label, ...} as detected ([]: none)
    cue_points: list[dict]                 # {id, time_sec, label, comment}
    jump_pairs: list[dict]                 # the JumpCUE pairs (AutoCUE's last, with AutoCUE on JumpCUEs)
    key_segments: np.ndarray               # [n, 4] (pitch class, mode, start, end)
    main_key: int | None                   # the key of the most beats (pitch class + 12 * minor)
    music_index: Any                       # what index_music made (None: not asked, or it failed)


def analyze_track(path: str, settings: AnalysisSettings, progress: Progress = None,
                  index_music: IndexMusic | None = None) -> TrackAnalysis:
    models = settings.models
    report(progress, stages.LOAD, 0.0)
    print("[Analyzer] Analysis Initalized")
    global_sr = ANALYSIS_SAMPLE_RATE
    print("[Analyzer] Loading Track")
    # Decode once: stereo for the structure detector, mono derived for the rest.
    samp_stereo = fast_load(path, target_sr=global_sr, stereo=True)
    samp = np.ascontiguousarray(samp_stereo.mean(axis=1), dtype=np.float32)

    report(progress, stages.HPSS, 0.1)
    print("[Analyzer] Processing HPSS")
    # The spectra are reused by the phrase features (same mono mix, STFT and HPSS).
    y_harm, y_perc, hpss_spectra = hpss_audio_with_spectra(samp, int(global_sr))

    report(progress, stages.WAVEFORM, 0.2)
    print("[Waveform] Analysis Initalized")
    # Overview waveform: the min / max of each column and the power of 12 log-spaced bands (low / mid / high are
    # summed from them when drawn), from a mono decode at WAVE_OVERVIEW_SAMPLE_RATE, not the analysis rate, so
    # the bands are not cut at the analysis Nyquist; and the detail waveform's three-band scale (these ranges).
    wave = build_wave_overview(
        fast_load(path, WAVE_OVERVIEW_SAMPLE_RATE),
        WAVE_OVERVIEW_SAMPLE_RATE,
        tuple((float(low), float(high)) for low, high in settings.wave_bands),
        n_fft=HPSS_N_FFT,
        hop_length=HPSS_HOP_LENGTH,
    )
    duration_sec = samp.size / float(global_sr)
    print("[Waveform] Analysis Finished")

    report(progress, stages.TEMPO, 0.30)
    print("[Tempo] Analysis Initalized")
    # Frame features on the shared frame grid: the phrase features read them below.
    frame_features = extract_frame_features(
        samp, int(global_sr), y_perc, y_harm, hop_length=FRAME_HOP_LENGTH
    )
    # The generalized tracker (analyzer.beat.general): a frame TCN (three trainings) ->
    # section decoding with one section cost -> ramps -> bar-supported phase -> metronome critic ->
    # tempo-change boundaries on the local onset -> drifting sections -> junction rules -> bars from
    # two downbeat observers -> the consensus of the three; rows with inizio on a downbeat.
    from analyzer.beat.general import (
        FRAME_HOP_LENGTH as TRACKER_HOP, SAMPLE_RATE as TRACKER_SR, dominant_tempo, load_models,
        track as track_beats,
    )
    from analyzer.beat.learned_onset import refine_onset_activation

    # Its models were trained on these frames: the HPSS parts at the analysis rate, hop 256.
    assert (int(global_sr), FRAME_HOP_LENGTH) == (TRACKER_SR, TRACKER_HOP)
    tracker_frames = frame_features
    frame_times = np.asarray(tracker_frames.frame_times, dtype=float)
    refine_onset = None
    try:
        refine_onset = (
            refine_onset_activation(tracker_frames, models.refine_onset).astype(np.float64),
            float(np.median(np.diff(frame_times))) if frame_times.size > 1 else TRACKER_HOP / TRACKER_SR,
            float(frame_times[0]),
        )
    except Exception as exc:
        print(f"[Tempo] boundary onset unavailable ({exc}); tempo changes stay where the decoder put them")
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
    labels_summary = ", ".join(str(s["label"]) for s in phrase_segments)
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
        jump_config = JumpCueConfig(jump_point_model_path=models.jumpcue)
        if autocue == "jumpcue":
            # one hot cue slot stays free for the AutoCUE JumpCUE (below)
            jump_config = replace(jump_config, hot_cue_slots=jump_config.hot_cue_slots - 1)
        jump_engine = JumpCueEngine(jump_config)
        jump_result = jump_engine.run(
            y_harm=y_harm,
            sr=global_sr,
            beats_time=beats_time_arr,
            y_mix=samp,
            tempo_segments=seg_arr,
            mix_magnitude=(
                hpss_spectra.magnitude
                if hpss_spectra is not None
                and (hpss_spectra.n_fft, hpss_spectra.hop_length)
                == (jump_engine.config.vocal.n_fft, jump_engine.config.vocal.hop)
                else None
            ),
            jump_point_features=jump_point_beat_features(
                frame_features, beats_time_arr, seg_arr, phrase_boundary_prob,
                audio=samp_stereo, phrase_feature_z=phrase_feature_z,
            ),
        )
        jump_pairs = [pair.as_dict() for pair in jump_result.pairs]
        if jump_pairs:
            print("[JumpCUE] pairs", jump_pairs)
        else:
            print("[JumpCUE] no jump-compatible pairs detected")
    if autocue == "jumpcue":
        jump_pairs = [*jump_pairs, autocue_jump_pair(autocue_target)]
    # the search by music's index takes the harmonic mask (its own STFT frames are these): no second HPSS
    music_mask = None
    if index_music is not None and hpss_spectra is not None:
        music_mask = (hpss_spectra.harmonic / np.maximum(hpss_spectra.magnitude, 1e-12)).astype(np.float32)
    hpss_spectra = None

    report(progress, stages.KEY, 0.7)
    print("[Key] Analysis Initalized")
    try:
        # chord-aware keys on the bars and phrases (analyzer/key/chord_key.py)
        key, key_segments = analyze_chord_keys(
            y_harm,
            int(global_sr),
            beats_time_arr,
            downbeat_times(beats_time_arr, seg_arr),
            [float(s["start"]) for s in phrase_segments],
            float(duration_sec),
            models.key,
            dynamic=bool(settings.key_dynamic),
            phrases=phrase_segments,
        )
    except Exception as exc:
        import traceback

        print(f"[Key] analysis skipped: {exc}")
        traceback.print_exc()
        key, key_segments = np.zeros(0, int), np.zeros((0, 4), float)
    print("[Key] Analysis Finished")

    # Search by Music's index: from the decoded stereo and the HPSS's harmonic part, mask and melody contours already
    # made. A failure leaves the track unindexed, never the analysis undone.
    music_entry = None
    if index_music is not None:
        report(progress, stages.MUSIC_INDEX, 0.78)
        try:
            music_entry = index_music(samp_stereo, y_harm, music_mask)
        except Exception as exc:
            print(f"[Music search] indexing skipped: {exc}")
        music_mask = None
    melody_contour.forget_contours()      # the frame features' contours, kept for the index

    values, counts = np.unique(key, return_counts=True)
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
        main_key=int(values[np.argmax(counts)]) if values.size else None,
        music_index=music_entry,
    )
