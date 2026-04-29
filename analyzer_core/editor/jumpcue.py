from analyzer_core.self_correlation.JumpCUE import JumpCueEngine
from typing import Callable, Optional
import numpy as np
import librosa
from analyzer_core.global_analyzer import fast_load
from core.config import config
from utils.jump_cues import build_jump_cues_np, extract_jump_cue_pairs

def reanalyze_jumpCUE(path:str, cfg:config, beats_time_arr:np.ndarray, *,
    progress_cb: Optional[Callable[[str, float], None]] = None,):
    if progress_cb:
        progress_cb("Loading audio", 0.0)
    gcf = cfg.analysisconfig
    sr = int(getattr(gcf, "analysis_samp_rate", 44100))
    audio_full = fast_load(path, sr)
    if progress_cb:
        progress_cb("Preparing analyzer", 0.1)
    use_hpss = getattr(gcf, "use_hpss", False)
    if use_hpss:
        try:
            y_harm, _ = librosa.effects.hpss(audio_full)
        except Exception:
            y_harm = audio_full
    else:
        y_harm = audio_full
    if progress_cb:
        progress_cb("Analyzing JumpCUE", 0.5)
    if beats_time_arr.size >= 2:
        jump_engine = JumpCueEngine()
        jump_result = jump_engine.run(
            y_harm=y_harm,
            sr=sr,
            beats_time=beats_time_arr,
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
    else:
        jump_cues_np = build_jump_cues_np(
            [],
            canonicalize_labels=True,
            merge_coincident=True,
        )
    
    jump_cues_extracted = extract_jump_cue_pairs({"jump_cues_np":jump_cues_np})
    if progress_cb:
        progress_cb("JumpCUE Updated", 0.9)
    return {
        "jump_cues_np": jump_cues_np,
        "jump_cues_extracted": jump_cues_extracted,
    }
