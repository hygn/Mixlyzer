import numpy as np
import librosa
from scipy.signal import butter, filtfilt
import time
import os
import base64
from pymediainfo import MediaInfo
import soundfile as sf
from pathlib import Path
from PySide6 import QtGui
from typing import Optional
from analyzer_core.key.key import *
from analyzer_core.beat.beat import *
from analyzer_core.beat.beat import _compute_odf
from analyzer_core.self_correlation.JumpCUE import JumpCueEngine
from utils.jump_cues import build_jump_cues_np
from core.audio.decoder import decode_to_memmap, get_total_samples
from core.library_handler import LibraryDB
from core.analysis_lib_handler import FeatureNPZStore
from core.config import config
from core.adapters import normalize_gui_buffers
from core.model import GlobalParams
from core.taskmanager import taskmanager
from core.linear_segments import build_bpm_segments, build_key_segments
from analyzer_core.utils import offset_beats_and_segments

def fast_load(path: str, target_sr) -> np.ndarray:
    path = Path(path)
    sr_req = int(target_sr) if target_sr else 44100
    try:
        pcm = decode_to_memmap(path.as_posix(), sr_req, ch=1)
        y = np.array(pcm, dtype=np.float32, copy=False).reshape(-1)
        sr = sr_req
    except Exception:
        try:
            y, sr = sf.read(path.as_posix(), dtype="float32", always_2d=False)
        except Exception:
            y, sr = librosa.load(path.as_posix(), sr=target_sr, mono=True)
        if y.ndim > 1:
            y = y.mean(axis=1).astype(np.float32)
    if target_sr and sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr, res_type="kaiser_fast")
    return np.ascontiguousarray(y, dtype=np.float32)

def resample_array(arr: np.ndarray, n: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 1:
        raise ValueError

    x_old = np.linspace(0, 1, len(arr))
    x_new = np.linspace(0, 1, n)
    y_new = np.interp(x_new, x_old, arr)
    return y_new

def _butter_bandpass(low, high, fs, order=4):
    nyq = 0.5 * fs
    low_n = max(low / nyq, 1e-6)
    high_n = min(high / nyq, 0.999999)
    b, a = butter(order, [low_n, high_n], btype="band")
    return b, a

def _band_envelope_rms(y, sr, low, high, frame_length, hop_length, order=4):
    b, a = _butter_bandpass(low, high, sr, order=order)
    yb = filtfilt(b, a, y).astype(np.float32)
    fl = int(max(16, frame_length))
    if fl % 2 == 0:
        fl += 1
    env = librosa.feature.rms(
        y=yb, frame_length=fl, hop_length=int(max(1, hop_length)), center=True
    )[0]
    return env

def frame_minmax(y: np.ndarray, hop: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(y)
    n_frames = int(np.ceil(n / hop))
    mins, maxs = np.empty(n_frames, np.float32), np.empty(n_frames, np.float32)

    for i in range(n_frames):
        start = i * hop
        end = min(start + hop, n)
        frame = y[start:end]
        mins[i] = np.min(frame)
        maxs[i] = np.max(frame)

    return mins, maxs

def _file_stats(path: str) -> tuple[int, float]:
    try:
        st = os.stat(path)
        return int(st.st_size), float(st.st_mtime)
    except FileNotFoundError:
        return 0, 0.0

def extract_tags(path: str) -> tuple[str, str, str, str]:
    title = os.path.splitext(os.path.basename(path))[0]
    artist = ""
    album = ""
    comment = ""
    try:
        mi = MediaInfo.parse(path, cover_data=False)
        tracks = mi.tracks if mi else []
        audio_track = next((t for t in tracks if t.track_type == "Audio"), None)
        general_track = next((t for t in tracks if t.track_type == "General"), None)

        def _pick_from_track(track, *names):
            if not track:
                return ""
            track_data = track.to_data() if hasattr(track, "to_data") else None
            for name in names:
                v = getattr(track, name, None)
                if v is None and isinstance(track_data, dict):
                    v = track_data.get(name)
                if isinstance(v, list):
                    for item in v:
                        if item and str(item).strip():
                            return str(item).strip()
                elif v:
                    v_str = str(v).strip()
                    if v_str:
                        return v_str
            return ""

        t = (
            _pick_from_track(general_track, "title", "track_name")
            or _pick_from_track(audio_track, "title", "track_name")
            or title
        )
        ar = (
            _pick_from_track(general_track, "artist", "performer")
            or _pick_from_track(audio_track, "artist", "performer")
            or _pick_from_track(general_track, "album_artist", "album_performer", "album_composer")
            or _pick_from_track(audio_track, "album_artist", "album_performer", "album_composer")
            or ""
        )
        al = (
            _pick_from_track(general_track, "album")
            or _pick_from_track(audio_track, "album")
            or ""
        )
        cm = (
            _pick_from_track(general_track, "comment", "description")
            or _pick_from_track(audio_track, "comment", "description")
            or ""
        )
        return t, ar, al, cm
    except Exception:
        return title, artist, album, comment


def _merge_track_properties(properties: dict, track, *, preserve_existing: bool) -> dict:
    if not (preserve_existing and track):
        return properties
    merged = dict(properties)
    for key in ("album", "artist", "rating", "title", "comment"):
        merged[key] = getattr(track, key, merged.get(key))
    return merged


def _persist_analysis_result(
    library: LibraryDB,
    store: FeatureNPZStore,
    features: dict,
    properties: dict,
    *,
    bpm_segments=None,
    key_segments=None,
):
    uid = library.upsert_meta(properties)
    library.conn.commit()
    properties["uid"] = uid
    if bpm_segments is not None:
        library.replace_bpm_segments(uid, bpm_segments)
        library.conn.commit()
    if key_segments is not None:
        library.replace_key_segments(uid, key_segments)
        library.conn.commit()
    store.save(uid, dict(features))
    return library.list_all()

def getAlbumArt(path) -> Optional[QtGui.QImage]:
    if not path:
        return None
    try:
        mi = MediaInfo.parse(path, cover_data=True)
        if not mi:
            return None
        data: Optional[bytes] = None
        for tr in mi.tracks:
            blob = getattr(tr, "cover_data", None)
            if not blob:
                continue
            if isinstance(blob, (bytes, bytearray)):
                data = bytes(blob)
            elif isinstance(blob, list):
                try:
                    data = bytes(blob)
                except Exception:
                    data = None
            elif isinstance(blob, str):
                try:
                    data = base64.b64decode(blob, validate=False)
                except Exception:
                    data = None
            if data:
                break
        if data:
            img = QtGui.QImage.fromData(data)
            return img if not img.isNull() else None
    except Exception:
        pass
    return None

def normalize_y(y: np.ndarray, peak:float=1) -> np.ndarray:
    nd = np.max(np.abs(y))
    return y*peak / nd

def timesig_exp(lags):
    n = np.abs(np.asarray(lags, int))
    F = np.array([3, 2, 5, 7])
    numer = np.array([3, 4, 5, 7])

    V = np.zeros((4, n.size), int)
    for k, f in enumerate(F):
        x = n.copy()
        while True:
            m = (x % f == 0) & (x > 0)
            if not m.any(): break
            V[k] += m
            x[m] //= f

    score = V.sum(1).astype(float)
    score[1] *= 0.5

    print("valuation(V) per factor  [3,2,5,7]:\n", V)
    print("sum(V) per factor:", V.sum(1))
    print("final score:", score)

    best = int(numer[score.argmax()])
    return best, dict(zip(numer, score))


def _apply_beatgrid_offset(synced_bpm: dict, offset_sec: float, track_duration: float) -> dict:
    out = dict(synced_bpm)
    beats, _ = offset_beats_and_segments(
        out.get("beats_time", ()),
        (),
        offset_sec,
        track_duration,
    )
    out["beats_time"] = beats

    shifted_segments = []
    for seg in out.get("tempo_segments") or ():
        if not isinstance(seg, dict):
            continue
        row = np.asarray(
            [[
                float(seg.get("segment_start", seg.get("start", 0.0))),
                float(seg.get("segment_end", seg.get("end", 0.0))),
                float(seg.get("bpm", 0.0)),
                float(seg.get("inizio", seg.get("segment_start", seg.get("start", 0.0)))),
            ]],
            dtype=float,
        )
        _, shifted = offset_beats_and_segments((), row, offset_sec, track_duration)
        seg_shifted = dict(seg)
        if shifted.size:
            seg_shifted["segment_start"] = float(shifted[0, 0])
            seg_shifted["segment_end"] = float(shifted[0, 1])
            if "start" in seg_shifted:
                seg_shifted["start"] = float(shifted[0, 0])
            if "end" in seg_shifted:
                seg_shifted["end"] = float(shifted[0, 1])
            seg_shifted["inizio"] = float(shifted[0, 3])
        shifted_segments.append(seg_shifted)
    out["tempo_segments"] = shifted_segments
    return out


def precompute_features(path: str, config: config, taskmgr: taskmanager, taskid:int, force_analyze: bool = False):
    l = LibraryDB(os.path.join(config.libconfig.libpath, "library.db"))
    l.connect()
    track = l.get(path)
    feature_store = FeatureNPZStore(base_dir=config.libconfig.libpath, compressed=True)
    if (track is not None) and track.uid and (not force_analyze):
        try:
            feat = feature_store.load(track.uid)
            l.close()
            metadata = track.to_meta()
            features_properties = {"features": feat, "properties": metadata, "update_db": False, "taskid":taskid}
            yield features_properties
            return
        except (FileNotFoundError, ValueError):
            pass
    
    title, artist, album, comment = extract_tags(path)
    yield {"status": "Loading"}
    print("[Analyzer] Analysis Initalized")
    gcf = config.analysisconfig
    global_sr = gcf.analysis_samp_rate
    print("[Analyzer] Loading Track")
    samp = fast_load(path, target_sr=global_sr)
    base_sig = normalize_y(samp, 0.99)
    taskmgr.updatetask(taskid, "Processing HPSS", 0.1)

    yield {"status": "HPSS"}
    print("[Analyzer] Processing HPSS")
    # HPSS
    if gcf.use_hpss:
        y_harm, y_perc = librosa.effects.hpss(samp)
    else:
        y_harm = y_perc = samp.astype(np.float32)
    features = {}

    yield {"status": "env_filter"}
    print("[Analyzer] Filtering")
    taskmgr.updatetask(taskid, "Filtering", 0.2)
    lo_low, lo_high   = gcf.env_lo
    mid_low, mid_high = gcf.env_mid
    hi_low,  hi_high  = gcf.env_hi

    nyq = 0.5 * global_sr
    lo_high  = min(lo_high,  nyq * 0.98)
    mid_high = min(mid_high, nyq * 0.98)
    hi_high  = min(hi_high,  nyq * 0.98)

    frame_ms = float(getattr(gcf, "env_frame_ms", 20))
    frame_ms = frame_ms/4
    env_frame_len = int(round(frame_ms * 1e-3 * global_sr))
    env_hop = max(1, env_frame_len)

    yield {"status": "env_rms"}
    print("[Env] Analysis Initalized")
    taskmgr.updatetask(taskid, "Processing Envelope", 0.23)
    lo_env  = _band_envelope_rms(base_sig, global_sr, lo_low,  lo_high,  env_frame_len, env_hop, gcf.env_order)
    mid_env = _band_envelope_rms(base_sig, global_sr, mid_low, mid_high, env_frame_len, env_hop, gcf.env_order)
    hi_env  = _band_envelope_rms(base_sig, global_sr, hi_low,  hi_high,  env_frame_len, env_hop, gcf.env_order)
    min_env, max_env = frame_minmax(base_sig, env_hop)
    features["min_env"] = min_env
    features["max_env"] = max_env

    def _norm(x):
        m = float(np.max(x)) if x.size else 0.0
        return (x / m) if m > 1e-12 else x

    lo_env_n  = _norm(lo_env)
    mid_env_n = _norm(mid_env)
    hi_env_n  = _norm(hi_env)

    env_times = librosa.times_like(lo_env_n, sr=global_sr, hop_length=env_hop)
    features["lo_env"] = lo_env_n
    features["mid_env"] = mid_env_n 
    features["hi_env"]  = hi_env_n
    features["env_times"] = env_times
    features["env_hop_length"] = int(env_hop)
    features["duration_sec"] = librosa.get_duration(y=samp, sr=global_sr)
    print("[Env] Analysis Finished")

    yield {"status": "tempo"}
    print("[Tempo] Analysis Initalized")
    taskmgr.updatetask(taskid, "Tempo Analyzing", 0.30)
    b, a = _butter_bandpass(20, 200, global_sr, order=4)
    lp_y = filtfilt(b, a, y_perc).astype(np.float32)
    odf_cached, hop_t_cached = _compute_odf(y_perc, global_sr, gcf.bpm_hop_length)
    odf_lp_cached, _ = _compute_odf(lp_y, global_sr, gcf.bpm_hop_length)
    if gcf.bpm_dynamic:
        cur_score = 0
        synced_bpm_best = {}
        w_mpls = [1, 2, 4, None] if gcf.bpm_adaptive_window else [1]
        for win_multiplier in w_mpls:
            if win_multiplier != None:
                synced_bpm = bpm_dynamic_phase_sync(
                    global_sr,
                    gcf.bpm_hop_length,
                    gcf.bpm_hop_length,
                    audio=y_perc,
                    audio_lp=lp_y,
                    win_s=gcf.bpm_win_length*win_multiplier/1000,
                    step_s=0.25,
                    bpm_bounds=(gcf.bpm_min,gcf.bpm_max),
                    odf_precomputed=odf_cached,
                    odf_lp_precomputed=odf_lp_cached,
                    hop_t_precomputed=hop_t_cached,
                )
            else:
                synced_bpm = bpm_phase_sync(
                    global_sr,
                    gcf.bpm_hop_length,
                    gcf.bpm_hop_length,
                    audio=y_perc,
                    audio_lp=lp_y,
                    win_s=gcf.bpm_win_length/1000,
                    step_s=0.1,
                    bpm_bounds=(gcf.bpm_min,gcf.bpm_max),
                    odf_precomputed=odf_cached,
                    odf_lp_precomputed=odf_lp_cached,
                    hop_t_precomputed=hop_t_cached,
                )
            if cur_score <= synced_bpm["score"]:
                synced_bpm_best = synced_bpm
                cur_score = synced_bpm["score"]
        synced_bpm = synced_bpm_best
    else:
        synced_bpm = bpm_phase_sync(
            global_sr,
            gcf.bpm_hop_length,
            gcf.bpm_hop_length,
            audio=y_perc,
            audio_lp=lp_y,
            win_s=gcf.bpm_win_length/1000,
            step_s=0.1,
            bpm_bounds=(gcf.bpm_min,gcf.bpm_max),
            odf_precomputed=odf_cached,
            odf_lp_precomputed=odf_lp_cached,
            hop_t_precomputed=hop_t_cached,
        )
    synced_bpm = _apply_beatgrid_offset(
        synced_bpm,
        float(gcf.beatgrid_offset_msec) / 1000.0,
        float(features.get("duration_sec", 0.0)),
    )
    features["tempo_global"] = synced_bpm["tempo_global"]
    features["beats_time_sec"] = synced_bpm["beats_time"]
    tempo_segments = synced_bpm["tempo_segments"]
    # tempo_segments columns: [start, end, bpm, inizio, ts_num] (ts_num defaults to 4)
    _DEFAULT_TS = 4.0
    seg_arr: np.ndarray
    if isinstance(tempo_segments, list):
        rows = []
        for seg in tempo_segments:
            start = float(seg.get("segment_start", 0.0))
            inizio = float(seg.get("inizio", seg.get("segment_start", 0.0)))
            end = float(seg.get("end", seg.get("segment_end", start)))
            bpm = float(seg.get("bpm", 0.0))
            ts_num = float(seg.get("time_signature", _DEFAULT_TS) or _DEFAULT_TS)
            rows.append((start, end, bpm, inizio, ts_num))
        seg_arr = np.asarray(rows, dtype=np.float32) if rows else np.empty((0, 5), dtype=np.float32)
    elif tempo_segments is not None:
        arr = np.asarray(tempo_segments, dtype=float)
        if arr.ndim == 1:
            for width in (5, 4, 3):
                if arr.size and arr.size % width == 0:
                    arr = arr.reshape((-1, width))
                    break
            else:
                arr = np.empty((0, 3), dtype=float)
        if arr.ndim == 2 and arr.shape[1] >= 3:
            n = arr.shape[0]
            seg_arr = np.empty((n, 5), dtype=np.float32)
            seg_arr[:, :3] = arr[:, :3]
            seg_arr[:, 3] = arr[:, 3] if arr.shape[1] >= 4 else arr[:, 0]
            seg_arr[:, 4] = arr[:, 4] if arr.shape[1] >= 5 else _DEFAULT_TS
        else:
            seg_arr = np.empty((0, 5), dtype=np.float32)
    else:
        seg_arr = np.empty((0, 5), dtype=np.float32)
    features["tempo_segments"] = seg_arr
    print("[Tempo] Analysis Finished")

    # Downbeat-offset realignment: assign the bar-start beat (downbeat) to each
    # segment where the detected downbeat phase changes. Beat timings are kept.
    yield {"status": "downbeat"}
    print("[Downbeat] Analysis Initalized")
    taskmgr.updatetask(taskid, "Downbeat Analyzing", 0.45)
    try:
        from analyzer_core.beat.downbeat_offset import (
            detect_downbeat_offset_segments,
            apply_downbeat_offset_segments,
        )
        beats_for_db = np.asarray(features.get("beats_time_sec"), dtype=float)
        if seg_arr.shape[0] > 0 and beats_for_db.size >= 2:
            downbeat_method = "dynamic" if bool(getattr(gcf, "dynamic_downbeat", False)) else "global"
            db_segments = detect_downbeat_offset_segments(
                samp, int(global_sr), beats_for_db, method=downbeat_method
            )
            print(f"[Downbeat] method={downbeat_method} beats={beats_for_db.size} detected offset segments={len(db_segments)}")
            _head = np.round(beats_for_db[:6], 3).tolist()
            print(f"[Downbeat] first beats_time_sec: {_head} (beat0={float(beats_for_db[0]):.3f}s)")
            prev_phase = None
            for i, ds in enumerate(db_segments):
                db_time = ds.first_downbeat_time_sec
                changed = "" if prev_phase is None else (" <PHASE CHANGE>" if int(ds.downbeat_phase) != int(prev_phase) else "")
                db_text = "n/a" if db_time is None else f"{db_time:.3f}s"
                print(
                    f"[Downbeat]   seg{i}: {ds.start_sec:.3f}-{ds.end_sec:.3f}s "
                    f"beats[{ds.start_beat_index}:{ds.end_beat_index}] "
                    f"phase={ds.downbeat_phase} offset_beats={ds.downbeat_offset_beats} "
                    f"first_downbeat=beat#{ds.first_downbeat_beat_index}@{db_text} "
                    f"conf={ds.confidence:.3f}{changed}"
                )
                prev_phase = ds.downbeat_phase

            before = int(seg_arr.shape[0])
            seg_arr = apply_downbeat_offset_segments(seg_arr, db_segments, beats_for_db)
            features["tempo_segments"] = seg_arr
            after = int(seg_arr.shape[0])
            print(f"[Downbeat] tempo segments: {before} -> {after} ({after - before} split(s) added)")
            for i, row in enumerate(seg_arr):
                start, end, bpm, inizio, ts = [float(x) for x in row[:5]]
                print(
                    f"[Downbeat]   tempo{i}: {start:.3f}-{end:.3f}s bpm={bpm:.2f} "
                    f"inizio(downbeat)={inizio:.3f}s ts={int(ts)}/4"
                )
        else:
            print("[Downbeat] skipped: no tempo segments or too few beats")
        print("[Downbeat] Analysis Finished")
    except Exception as exc:
        import traceback
        print(f"[Downbeat] offset detection skipped: {exc}")
        traceback.print_exc()

    jump_result = None
    beats_time_arr = np.asarray(features.get("beats_time_sec"), dtype=float)
    taskmgr.updatetask(taskid, "JumpCUE Analyzing", 0.50)
    if beats_time_arr.size >= 2:
        jump_engine = JumpCueEngine()
        jump_result = jump_engine.run(
            y_harm=y_harm,
            sr=global_sr,
            beats_time=beats_time_arr,
        )
        jump_pairs = [pair.as_dict() for pair in jump_result.pairs]
        if jump_pairs:
            print("[JumpCUE] pairs", jump_pairs)
        else:
            print("[JumpCUE] no jump-compatible pairs detected")
        features["jump_cues_np"] = build_jump_cues_np(
            jump_pairs,
            canonicalize_labels=True,
            merge_coincident=True,
        )
    else:
        features["jump_cues_np"] = build_jump_cues_np(
            [],
            canonicalize_labels=True,
            merge_coincident=True,
        )

    #best, scores= timesig_exp(jump_result.report.beat_ssm.peak_indices)
    features["timesignature"] = 4

    yield {"status": "chroma"}
    print("[Chroma] Analysis Initalized")
    taskmgr.updatetask(taskid, "Processing Chromagram", 0.65)
    chroma = chroma_to_subdiv_grid(
        y_harm, np.divide(synced_bpm["beats"], gcf.chroma_hop_length/gcf.bpm_hop_length).astype(int), sr=global_sr,
        hop_length=gcf.chroma_hop_length,
        bins_per_octave=gcf.chroma_cqt_bins_per_octave,
        n_octaves=gcf.chroma_cqt_octaves,
        mode=gcf.chroma_method,
        subdiv=1
    )
    bs = np.asarray(chroma["chroma_subdiv"])  # (12, T)

    features["beatsync_chroma"] = bs.astype(np.float32)

    features["chroma_beatsync_center"] = chroma["t_subdiv"]
    features["chroma_beatsync_grid"] = chroma["t_beatgrid"]
    features["full_chroma"] = chroma["chroma_full"]
    features["chroma_grid"] = chroma["t_full"]
    features["chroma_hop"] = gcf.chroma_hop_length
    print("[Chroma] Analysis Finished")

    yield {"status": "key"}
    print("[Key] Analysis Initalized")
    taskmgr.updatetask(taskid, "Analyzing Keys", 0.75)
    key, key_12, logB, _, logPi, logA, key_segments = keyanalyzer(
        features["beatsync_chroma"],
        features["chroma_beatsync_center"],
        config=config,
        y_harm=y_harm,
        sample_rate=global_sr,
        beat_times=features.get("beats_time_sec")
    )
    print("[Key] Analysis Finished")
    features["key_segments"] = key_segments
    features["bpm_hop"] = gcf.bpm_hop_length
    features["sr"] = global_sr

    # DB Features
    taskmgr.updatetask(taskid, "Processing Metadata", 0.85)
    duration_out = float(features.get("duration_sec", 0.0))

    bpm_out = features.get("tempo_global", 0.0)

    values, counts = np.unique(key, return_counts=True)
    key_int = int(values[np.argmax(counts)])

    fsize, fmtime = _file_stats(path)
    gp = GlobalParams(
            analysis_samp_rate=int(gcf.analysis_samp_rate),
            bpm_hop_length=int(gcf.bpm_hop_length),
            chroma_hop_length=int(gcf.chroma_hop_length),
        )
    taskmgr.updatetask(taskid, "Rendering/Normalizing", 0.88)
    features = normalize_gui_buffers(features, gp)

    features_denylist = ["key_12", "logB", "full_chroma", "lo_env", "mid_env", "hi_env", "min_env", "max_env", "env_times", "beatsync_chroma"]
    for i in features_denylist:
        try:
            features.pop(i)
        except:
            pass

    properties = {
        "path": path,
        "title": title,
        "artist": artist,
        "album": album,
        "bpm": bpm_out,
        "key": key_int,
        "duration_sec": duration_out,
        "total_samples": int(get_total_samples(path) or 0),
        "rating": 0,
        "added_ts": int(time.time()),
        "comment": comment,
        "file_mtime": float(fmtime),
        "file_size": int(fsize),
    }
    properties = _merge_track_properties(properties, track, preserve_existing=bool(force_analyze and track is not None))
    if track and track.uid:
        properties["uid"] = track.uid
    bpm_segments = build_bpm_segments(features.get("tempo_segments"))
    key_segments_db = build_key_segments(features.get("key_segments"))
    features_properties = {"features": features, "properties": properties, "update_db": True, "taskid":taskid}
    yield {"status": "Saving"}
    print("[Analyzer] Saving Results")
    taskmgr.updatetask(taskid, "Saving Result", 0.92)
    try:
        lib_full = _persist_analysis_result(
            l,
            feature_store,
            features,
            properties,
            bpm_segments=bpm_segments,
            key_segments=key_segments_db,
        )
    finally:
        l.close()
    features_properties["library"] = lib_full
    print("[Analyzer] Analysis Finished")
    taskmgr.updatetask(taskid, "Finished", 1)

    yield features_properties
