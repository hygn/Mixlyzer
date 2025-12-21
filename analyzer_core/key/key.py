import numpy as np
import librosa
from analyzer_core.key.viterbi_key import *
from analyzer_core.key.viterbi_mode import *
from analyzer_core.utils import *
from analyzer_core.key.key_cqt import keydetect as KeyCQTDetector
from typing import List, Tuple
from core.config import config
from scipy.special import logsumexp
from collections import Counter
from utils.labels import KEY_DISPLAY_LABELS

circle_of_fifths = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]
circle_pos = {k: i for i, k in enumerate(circle_of_fifths)}
mode_viterbi = False

NOTE_NAME_TO_PC = {
    "C": 0,
    "C#": 1,
    "DB": 1,
    "D": 2,
    "D#": 3,
    "EB": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "GB": 6,
    "G": 7,
    "G#": 8,
    "AB": 8,
    "A": 9,
    "A#": 10,
    "BB": 10,
    "B": 11,
}


def _constant_segments(path: np.ndarray) -> List[Tuple[int, int, int]]:
    arr = np.asarray(path, dtype=int).ravel()
    n = arr.size
    if n == 0:
        return []
    change_idx = np.where(np.diff(arr) != 0)[0] + 1
    boundaries = np.concatenate(([0], change_idx, [n]))
    segments: List[Tuple[int, int, int]] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if start >= end:
            continue
        segments.append((int(start), int(end), int(arr[start])))
    return segments


def _parse_cqt_key_name(name: str | None) -> Tuple[int, int] | None:
    if not name or ":" not in name:
        return None
    root, mode = name.split(":", 1)
    root = root.strip().upper()
    mode = mode.strip().lower()
    pc = NOTE_NAME_TO_PC.get(root)
    if pc is None:
        return None
    if mode == "major":
        return pc, 0
    if mode == "minor":
        return pc, 1
    return None


def fifth_distance(k1, k2):
    """Return minimal distance on the circle of fifths."""
    i1, i2 = circle_pos[k1], circle_pos[k2]
    diff = abs(i1 - i2)
    return min(diff, 12 - diff)

def flatten_by_fifths(key_array, max_dist=1):
    """Segment by Circle-of-Fifths proximity and flatten by mode."""
    key_array = np.asarray(key_array)
    if len(key_array) == 0:
        return np.array([])
    
    segments = []
    start = 0

    for i in range(1, len(key_array)):
        if fifth_distance(int(key_array[i-1]), int(key_array[i])) > max_dist:
            segments.append(key_array[start:i])
            start = i
    segments.append(key_array[start:])

    # Flatten by mode of each segment
    flattened = []
    for seg in segments:
        mode_val = Counter(seg).most_common(1)[0][0]
        flattened.extend([mode_val] * len(seg))
    return np.array(flattened)


def keyanalyzer(
    beat_synced_chromagram: np.ndarray,
    syncgrid: np.array,
    config: config,
    key_flatten_iters: int = 0,
    *,
    y_harm: np.ndarray,
    sample_rate: int | float,
    beat_times: np.ndarray | None = None,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[tuple[int | None, int | None, float, float]],
]:
    # generate key mask for calculate emission
    key_mask = gen_key_mask(minor=False, min_offset=config.keyconfig.min_offset)
    # calculate emission
    key_emission = calc_emission(beat_synced_chromagram, key_mask)
    # calculate log-likelihood for key
    log_B_key = to_log_B_from_scores(key_emission, temp=5)
    log_A_key = build_transition_12(p_self=config.keyconfig.pitch_self,
                                    p_semi=config.keyconfig.pitch_semitone,
                                    p_fifth=config.keyconfig.pitch_fifth,
                                    p_other=config.keyconfig.pitch_others)
    log_pi_key = build_log_pi(n=12)
    key_path = viterbi_log(log_pi_key, log_A_key, log_B_key)
    for i in range(key_flatten_iters):
        key_path = flatten_by_fifths(key_path)
    if mode_viterbi:
        # calculate log-likelihood for mode
        log_B_mode = emission_mode_logB(beat_synced_chromagram, key_path)
        log_A_mode = build_mode_logA(self_bias=20)
        log_pi_mode = build_log_pi(n=2)
        # run viterbi algorithm
        mode_path = viterbi_log(log_pi_mode, log_A_mode, log_B_mode)
        
        # combine key and mode
        full_path = fuse_key_mode_to_24_path(key_path, mode_path)
    
        combined_path = np.asarray(full_path, dtype=int).copy()
    else:
        combined_path = np.asarray(key_path, dtype=int).copy()

    try:
        sr_int = int(sample_rate)
    except (TypeError, ValueError) as exc:
        raise ValueError("sample_rate must be an integer-convertible value.") from exc
    if sr_int <= 0:
        raise ValueError("sample_rate must be positive.")

    y_harm_vec = np.asarray(y_harm, dtype=np.float32).reshape(-1)
    if y_harm_vec.size == 0:
        raise ValueError("y_harm must contain harmonic audio samples.")

    detector = KeyCQTDetector(y_harm_vec, sr_int)
    duration = y_harm_vec.size / float(sr_int)

    centers = np.asarray(syncgrid, dtype=float).ravel()
    if centers.size != len(key_path):
        raise ValueError("syncgrid length must match key path length.")
    if centers.size == 0:
        return combined_path, key_path, log_B_key, syncgrid, log_pi_key, log_A_key, []

    if centers.size > 1:
        midpoints = 0.5 * (centers[:-1] + centers[1:])
        edges = np.concatenate(([0.0], midpoints, [duration]))
    else:
        edges = np.array([0.0, duration], dtype=float)
    edges = np.clip(edges, 0.0, duration)

    # Snap edges to beat grid (and duration endpoints) to avoid drift/quantization.
    if beat_times is not None:
        grid = np.asarray(beat_times, dtype=float).ravel()
        grid = grid[np.isfinite(grid)]
        if grid.size:
            grid = np.clip(grid, 0.0, duration)
            grid = np.unique(np.concatenate(([0.0], grid, [duration])))

            def _snap(val: float) -> float:
                idx = np.searchsorted(grid, val)
                if idx <= 0:
                    return grid[0]
                if idx >= grid.size:
                    return grid[-1]
                left = grid[idx - 1]
                right = grid[idx]
                return left if (val - left) <= (right - val) else right

            edges = np.array([_snap(v) for v in edges], dtype=float)
            edges = np.maximum.accumulate(edges)
            edges = np.clip(edges, 0.0, duration)

    print(f"[key] segments={len(_constant_segments(key_path))} | centers={len(centers)} | duration={duration:.3f}s")

    key_segments: list[tuple[int | None, int | None, float, float]] = []
    seg_mode_score = 0
    frame_modes = np.zeros_like(combined_path, dtype=int)
    for seg_start, seg_end, _ in _constant_segments(key_path):
        t0 = edges[seg_start]
        t1 = edges[seg_end]
        if t1 <= t0:
            continue

        print(f"[key]   seg frames=({seg_start},{seg_end}) -> times=({t0:.3f},{t1:.3f})")
        cqt_out = detector.keydetect(songrange=(t0, t1), beat_times=beat_times, verbose=True)
        top_candidate = None
        for cand in cqt_out.get("keys", []):
            parsed = _parse_cqt_key_name(cand.get("key"))
            if parsed is not None:
                top_candidate = parsed
                break
        if top_candidate is None:
            raise ValueError(f"CQT key detection failed for segment {seg_start}:{seg_end}.")

        pitch_class, mode_flag = top_candidate
        seg_mode_score += (t1 - t0) * (1 if mode_flag == 0 else -1)
        combined_path[seg_start:seg_end] = pitch_class
        frame_modes[seg_start:seg_end] = mode_flag
        key_segments.append((pitch_class, mode_flag, t0, t1))
        label_idx = pitch_class + (12 if mode_flag else 0)
        print(f"[key]     detected {KEY_DISPLAY_LABELS[label_idx]} over {t1 - t0:.3f}s")

    overall_mode_flag = 0 if seg_mode_score >= 0 else 1  # 0=Major, 1=Minor
    frame_pc = np.asarray(combined_path, dtype=int) % 12
    frame_modes = np.asarray(frame_modes, dtype=int) % 2

    if overall_mode_flag == 0:
        mask = frame_modes == 1
        frame_pc[mask] = (frame_pc[mask] + 3) % 12
    else:
        mask = frame_modes == 0
        frame_pc[mask] = (frame_pc[mask] + 9) % 12

    combined_path = frame_pc + (12 * overall_mode_flag)
    print(f"[key] overall_mode={'Major' if overall_mode_flag == 0 else 'Minor'} (score={seg_mode_score:.3f})")

    # Normalize segment metadata to the unified mode
    unified_segments: list[tuple[int | None, int | None, float, float]] = []
    for pc, mode_flag, t0, t1 in key_segments:
        if pc is None or mode_flag is None:
            unified_segments.append((pc, overall_mode_flag, t0, t1))
            continue
        pc_int = int(pc) % 12
        mode_int = int(mode_flag) % 2
        if overall_mode_flag == 0 and mode_int == 1:
            pc_int = (pc_int + 3) % 12
        elif overall_mode_flag == 1 and mode_int == 0:
            pc_int = (pc_int + 9) % 12
        unified_segments.append((pc_int, overall_mode_flag, t0, t1))
    key_segments = unified_segments

    return combined_path, key_path, log_B_key, syncgrid, log_pi_key, log_A_key, key_segments

def chroma_to_subdiv_grid(y_harm, beat_frames, sr=22050, hop_length=512, bins_per_octave=36, n_octaves=6, mode="cens", n_fft = 4096, subdiv=1):
    if mode == "cqt":
        chroma = librosa.feature.chroma_cqt(
            y=y_harm, sr=sr,
            bins_per_octave=bins_per_octave,
            hop_length=hop_length,
            n_octaves=n_octaves,
            tuning=None)
    elif mode == "cens":
        chroma = librosa.feature.chroma_cens(
            y=y_harm, sr=sr,
            bins_per_octave=bins_per_octave,
            hop_length=hop_length,
            n_octaves=n_octaves,
            tuning=None)
    elif mode == "hcpc":
        chroma = librosa.feature.chroma_stft(
            y=y_harm, sr=sr,
            hop_length=hop_length,
            n_fft=n_fft,
            tuning=None)

    t_frames = librosa.frames_to_time(np.arange(chroma.shape[1]),
                                      sr=sr, hop_length=hop_length)

    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)

    subdiv_times = []
    for i in range(len(beat_times) - 1):
        b0, b1 = beat_times[i], beat_times[i+1]
        for k in range(subdiv):
            t = b0 + (b1 - b0) * (k / subdiv)
            subdiv_times.append(t)
    subdiv_times.append(beat_times[-1])

    subdiv_times = np.array(subdiv_times, dtype=np.float64)
    subdiv_times = np.unique(np.clip(subdiv_times, t_frames[0], t_frames[-1]))

    chroma_subdiv = np.empty((12, len(subdiv_times)), dtype=np.float32)
    subdiv_frames = librosa.time_to_frames(subdiv_times, sr=sr, hop_length=hop_length)
    chroma_subdiv = librosa.util.sync(chroma, subdiv_frames, aggregate=np.median, pad=False)

    t_edges   = librosa.frames_to_time(subdiv_frames, sr=sr, hop_length=hop_length)
    t_centers = 0.5*(t_edges[:-1] + t_edges[1:])

    return {
        "chroma_full": np.abs(chroma),
        "t_full": t_frames,
        "beat_times": beat_times,
        "t_subdiv": t_centers,
        "t_beatgrid": t_edges[:-1],
        "chroma_subdiv": np.abs(chroma_subdiv)
    }
