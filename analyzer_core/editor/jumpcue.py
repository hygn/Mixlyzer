from analyzer_core.self_correlation.JumpCUE import JumpCueEngine
from typing import Callable, Optional
import numpy as np
import librosa
from analyzer_core.global_analyzer import fast_load
from core.config import config
from utils.jump_cues import extract_jump_cue_pairs

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
        # Flatten arrays for NPZ persistence
        if jump_pairs:
            labels_fwd = [str(p["forward"].get("label", "")) for p in jump_pairs]
            starts_fwd = [float(p["forward"].get("start", 0.0)) for p in jump_pairs]
            ends_fwd = [float(p["forward"].get("end", 0.0)) for p in jump_pairs]
            labels_bwd = [str(p["backward"].get("label", "")) for p in jump_pairs]
            starts_bwd = [float(p["backward"].get("start", 0.0)) for p in jump_pairs]
            ends_bwd = [float(p["backward"].get("end", 0.0)) for p in jump_pairs]
            lag_beats = [float(p.get("lag_beats", 0.0)) for p in jump_pairs]
            lag_sec = [float(p.get("lag_sec", 0.0)) for p in jump_pairs]
            scores = [float(p.get("score", 0.0)) for p in jump_pairs]
            confs = [float(p.get("confidence", 0.0)) for p in jump_pairs]
            points_fwd = [float(p["forward"].get("point", 0.0)) for p in jump_pairs]
            points_bwd = [float(p["backward"].get("point", 0.0)) for p in jump_pairs]
            label_width_f = max(1, max(len(s) for s in labels_fwd))
            label_width_b = max(1, max(len(s) for s in labels_bwd))
            jump_cues_np = {
                "forward_label": np.asarray(labels_fwd, dtype=f"U{label_width_f}"),
                "forward_start": np.asarray(starts_fwd, dtype=np.float32),
                "forward_end": np.asarray(ends_fwd, dtype=np.float32),
                "forward_point": np.asarray(points_fwd, dtype=np.float32),
                "backward_label": np.asarray(labels_bwd, dtype=f"U{label_width_b}"),
                "backward_start": np.asarray(starts_bwd, dtype=np.float32),
                "backward_end": np.asarray(ends_bwd, dtype=np.float32),
                "backward_point": np.asarray(points_bwd, dtype=np.float32),
                "lag_beats": np.asarray(lag_beats, dtype=np.float32),
                "lag_sec": np.asarray(lag_sec, dtype=np.float32),
                "score": np.asarray(scores, dtype=np.float32),
                "confidence": np.asarray(confs, dtype=np.float32),
            }
        else:
            jump_cues_np = {
                "forward_label": np.asarray([], dtype="U1"),
                "forward_start": np.asarray([], dtype=np.float32),
                "forward_end": np.asarray([], dtype=np.float32),
                "forward_point": np.asarray([], dtype=np.float32),
                "backward_label": np.asarray([], dtype="U1"),
                "backward_start": np.asarray([], dtype=np.float32),
                "backward_end": np.asarray([], dtype=np.float32),
                "backward_point": np.asarray([], dtype=np.float32),
                "lag_beats": np.asarray([], dtype=np.float32),
                "lag_sec": np.asarray([], dtype=np.float32),
                "score": np.asarray([], dtype=np.float32),
                "confidence": np.asarray([], dtype=np.float32),
            }
    else:
        jump_cues_np = {
            "forward_label": np.asarray([], dtype="U1"),
            "forward_start": np.asarray([], dtype=np.float32),
            "forward_end": np.asarray([], dtype=np.float32),
            "forward_point": np.asarray([], dtype=np.float32),
            "backward_label": np.asarray([], dtype="U1"),
            "backward_start": np.asarray([], dtype=np.float32),
            "backward_end": np.asarray([], dtype=np.float32),
            "backward_point": np.asarray([], dtype=np.float32),
            "lag_beats": np.asarray([], dtype=np.float32),
            "lag_sec": np.asarray([], dtype=np.float32),
            "score": np.asarray([], dtype=np.float32),
            "confidence": np.asarray([], dtype=np.float32),
        }
    
    jump_cues_extracted = extract_jump_cue_pairs({"jump_cues_np":jump_cues_np})
    if progress_cb:
        progress_cb("JumpCUE Updated", 0.9)
    return {
        "jump_cues_np": jump_cues_np,
        "jump_cues_extracted": jump_cues_extracted,
    }
