from analyzer_core.self_correlation.JumpCUE import JumpCueConfig, JumpCueEngine
from analyzer_core.self_correlation.jump_point_model import jump_point_beat_features
from typing import Callable, Optional
import numpy as np
from analyzer_core.beat.frame_features import extract_frame_features
from analyzer_core.cue_and_phrase import detect_phrase_segments
from analyzer_core.global_analyzer import fast_load
from analyzer_core.hpss import hpss_audio_with_spectra
from core.config import config
from utils.jump_cues import build_jump_cues_np, extract_jump_cue_pairs

def reanalyze_jumpCUE(path:str, cfg:config, beats_time_arr:np.ndarray, *,
    tempo_segments: Optional[np.ndarray] = None,
    progress_cb: Optional[Callable[[str, float], None]] = None,):
    """JumpCUE on the current beat grid, computed as in the full analysis: the
    parts it needs (HPSS, frame features, phrase boundary probability) are
    analyzed first. Needs the phrase analysis enabled."""
    gcf = cfg.analysisconfig
    if not bool(getattr(gcf, "phrase_analysis_enabled", True)):
        raise ValueError("JumpCUE needs the phrase analysis; enable it in the analysis settings")
    if tempo_segments is None or np.asarray(tempo_segments).size == 0:
        raise ValueError("tempo segments are required for JumpCUE reanalysis")
    tempo_segments = np.asarray(tempo_segments, dtype=float)
    if progress_cb:
        progress_cb("Loading audio", 0.0)
    sr = int(getattr(gcf, "analysis_samp_rate", 44100))
    audio_stereo = fast_load(path, sr, stereo=True)
    audio_full = np.ascontiguousarray(audio_stereo.mean(axis=1), dtype=np.float32)
    if progress_cb:
        progress_cb("Processing HPSS", 0.1)
    hpss_spectra = None
    if getattr(gcf, "use_hpss", False):
        y_harm, y_perc, hpss_spectra = hpss_audio_with_spectra(audio_full, sr)
    else:
        y_harm = y_perc = audio_full
    if progress_cb:
        progress_cb("Frame features", 0.25)
    frames = extract_frame_features(audio_full, sr, y_perc, y_harm, hop_length=int(gcf.bpm_hop_length))
    if progress_cb:
        progress_cb("Phrase boundaries", 0.4)
    _segments, boundary_prob, phrase_feature_z = detect_phrase_segments(
        audio_stereo,
        sr,
        beats_time_arr,
        tempo_segments,
        model_path=str(gcf.phrase_parameter_path).strip(),
        hpss=hpss_spectra,
        return_boundary_probability=True,
        return_feature_z=True,
    )
    if boundary_prob is None:
        raise ValueError("too few beats for the phrase analysis JumpCUE needs")
    if progress_cb:
        progress_cb("Analyzing JumpCUE", 0.55)
    jump_engine = JumpCueEngine(JumpCueConfig(jump_point_model_path=str(gcf.jumpcue_parameter_path)))
    jump_result = jump_engine.run(
        y_harm=y_harm,
        sr=sr,
        beats_time=beats_time_arr,
        y_mix=audio_full,
        tempo_segments=tempo_segments,
        mix_magnitude=(
            hpss_spectra.magnitude
            if hpss_spectra is not None
            and (hpss_spectra.n_fft, hpss_spectra.hop_length)
            == (jump_engine.config.vocal.n_fft, jump_engine.config.vocal.hop)
            else None
        ),
        jump_point_features=jump_point_beat_features(
            frames, beats_time_arr, tempo_segments, boundary_prob, audio=audio_stereo,
            phrase_feature_z=phrase_feature_z,
        ),
    )
    jump_pairs = [pair.as_dict() for pair in jump_result.pairs]
    if jump_pairs:
        print("[JumpCUE] pairs", jump_pairs)
    else:
        print("[JumpCUE] no jump-compatible pairs detected")
    jump_cues_np = build_jump_cues_np(
        jump_pairs,
        canonicalize_labels=True,
        merge_coincident=True,
    )

    jump_cues_extracted = extract_jump_cue_pairs({"jump_cues_np":jump_cues_np})
    if progress_cb:
        progress_cb("JumpCUE Updated", 0.9)
    return {
        "jump_cues_np": jump_cues_np,
        "jump_cues_auto_np": jump_cues_np,
        "jump_cues_extracted": jump_cues_extracted,
    }
