import numpy as np
import librosa
from concurrent.futures import ThreadPoolExecutor
from threadpoolctl import threadpool_limits
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
from analyzer_core.beat.beat import collapse_short_sandwiched_tempo_segments
from analyzer_core.beat.frame_features import extract_frame_features
from analyzer_core.beat.learned_onset import compute_beat_odf
from analyzer_core.hpss import HPSS_HOP_LENGTH, HPSS_N_FFT, hpss_audio_with_spectra
from utils.wave import build_wave_preview
from analyzer_core.self_correlation.JumpCUE import JumpCueEngine
from analyzer_core.cue_and_phrase import detect_phrase_segments
from utils.jump_cues import build_jump_cues_np
from utils.cue_points import (
    build_cue_points_np,
    build_phrase_cue_points,
    empty_cue_points_np,
)
from utils.phrases import build_phrase_segments_np
from core.audio.decoder import decode_to_memmap, get_total_samples
from core.library_handler import LibraryDB
from core.analysis_lib_handler import FeatureNPZStore
from core.config import config
from core.adapters import normalize_gui_buffers
from core.taskmanager import taskmanager
from core.linear_segments import build_bpm_segments, build_key_segments
from analyzer_core.utils import offset_beats_and_segments, prime_physical_core_count

# sklearn would otherwise launch powershell.exe to count cores in every analysis process.
prime_physical_core_count()

def fast_load(path: str, target_sr, stereo: bool = False) -> np.ndarray:
    """Decode audio to float32. Mono ``(N,)`` by default, stereo ``(N, 2)`` when
    ``stereo=True`` (channel-duplicated if the source is mono)."""
    path = Path(path)
    sr_req = int(target_sr) if target_sr else 44100
    ch = 2 if stereo else 1
    try:
        pcm = decode_to_memmap(path.as_posix(), sr_req, ch=ch)
        arr = np.array(pcm, dtype=np.float32, copy=False)
        y = arr.reshape(-1, 2) if stereo else arr.reshape(-1)
        sr = sr_req
    except Exception:
        try:
            y, sr = sf.read(path.as_posix(), dtype="float32", always_2d=False)
        except Exception as exc:
            raise RuntimeError(f"Audio decode failed via FFmpeg and SoundFile: {path}") from exc
        if stereo:
            if y.ndim == 1:
                y = np.stack([y, y], axis=1)
            elif y.shape[1] == 1:
                y = np.repeat(y, 2, axis=1)
            else:
                y = y[:, :2]
        elif y.ndim > 1:
            y = y.mean(axis=1).astype(np.float32)
    if target_sr and sr != target_sr:
        src = y.T if stereo else y  # resample along last axis
        src = librosa.resample(np.ascontiguousarray(src), orig_sr=sr, target_sr=target_sr, res_type="kaiser_fast")
        y = src.T if stereo else src
    return np.ascontiguousarray(y, dtype=np.float32)

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
    # Decode once: stereo for the structure detector, mono derived for the rest.
    samp_stereo = fast_load(path, target_sr=global_sr, stereo=True)
    samp = np.ascontiguousarray(samp_stereo.mean(axis=1), dtype=np.float32)
    base_sig = normalize_y(samp, 0.99)
    taskmgr.updatetask(taskid, "Processing HPSS", 0.1)

    yield {"status": "HPSS"}
    print("[Analyzer] Processing HPSS")
    # HPSS
    # The spectra are reused by the phrase features (same mono mix, STFT and HPSS).
    hpss_spectra = None
    if gcf.use_hpss:
        y_harm, y_perc, hpss_spectra = hpss_audio_with_spectra(samp, int(global_sr))
    else:
        y_harm = y_perc = samp.astype(np.float32)
    features = {}

    yield {"status": "waveform"}
    print("[Waveform] Analysis Initalized")
    taskmgr.updatetask(taskid, "Processing Waveform", 0.2)
    # Overview waveform, drawn directly at its display resolution. Band colors
    # come from the HPSS STFT of the same mix when it is available.
    nyq = 0.5 * global_sr
    wave_bands = tuple((float(low), min(float(high), nyq * 0.98)) for low, high in (gcf.env_lo, gcf.env_mid, gcf.env_hi))
    # Bars of env_frame_ms / 4 (the resolution of the former full-size image).
    wave_frame_hop = max(1, int(round(float(getattr(gcf, "env_frame_ms", 20)) / 4 * 1e-3 * global_sr)))
    features["wave_img_np_preview"] = build_wave_preview(
        base_sig,
        int(global_sr),
        wave_bands,
        wave_frame_hop,
        hpss_spectra.magnitude if hpss_spectra is not None else None,
        n_fft=hpss_spectra.n_fft if hpss_spectra is not None else HPSS_N_FFT,
        hop_length=hpss_spectra.hop_length if hpss_spectra is not None else HPSS_HOP_LENGTH,
    )
    features["duration_sec"] = librosa.get_duration(y=samp, sr=global_sr)
    print("[Waveform] Analysis Finished")

    yield {"status": "tempo"}
    print("[Tempo] Analysis Initalized")
    taskmgr.updatetask(taskid, "Tempo Analyzing", 0.30)
    # Frame features are analyzed once: the learned onset (beat tracking) and the
    # downbeat model (after pooling to beats) both use them.
    frame_features = extract_frame_features(samp, int(global_sr), y_perc, y_harm)
    odf_cached, hop_t_cached = compute_beat_odf(
        str(gcf.onset_source),
        frame_features,
        y_perc,
        int(global_sr),
        gcf.bpm_hop_length,
        str(gcf.onset_parameter_path).strip(),
    )
    if gcf.bpm_dynamic:
        w_mpls = [1, 2, 4, None] if gcf.bpm_adaptive_window else [1]

        def track_window(win_multiplier):
            if win_multiplier is None:
                return bpm_phase_sync(
                    global_sr,
                    gcf.bpm_hop_length,
                    gcf.bpm_hop_length,
                    audio=y_perc,
                    win_s=gcf.bpm_win_length/1000,
                    step_s=0.1,
                    bpm_bounds=(gcf.bpm_min,gcf.bpm_max),
                    odf_precomputed=odf_cached,
                    hop_t_precomputed=hop_t_cached,
                )
            return bpm_dynamic_phase_sync(
                global_sr,
                gcf.bpm_hop_length,
                gcf.bpm_hop_length,
                audio=y_perc,
                win_s=gcf.bpm_win_length*win_multiplier/1000,
                step_s=0.25,
                bpm_bounds=(gcf.bpm_min,gcf.bpm_max),
                odf_precomputed=odf_cached,
                hop_t_precomputed=hop_t_cached,
            )

        # The window candidates are independent (numpy / scipy release the GIL).
        # KMeans limits BLAS to one thread while it runs and then restores the
        # previous count; concurrent calls would restore each other's limit and
        # leave BLAS single-threaded for the rest of the process. Holding the
        # limit here makes their restores no-ops, and this one restores it.
        with threadpool_limits(limits=1, user_api="blas"), ThreadPoolExecutor(max_workers=len(w_mpls)) as pool:
            candidates = list(pool.map(track_window, w_mpls))
        cur_score = 0
        synced_bpm_best = {}
        for synced_bpm in candidates:
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
            win_s=gcf.bpm_win_length/1000,
            step_s=0.1,
            bpm_bounds=(gcf.bpm_min,gcf.bpm_max),
            odf_precomputed=odf_cached,
            hop_t_precomputed=hop_t_cached,
        )
    synced_bpm = _apply_beatgrid_offset(
        synced_bpm,
        float(gcf.beatgrid_offset_msec) / 1000.0,
        float(features.get("duration_sec", 0.0)),
    )
    bpm_window_sec = float(synced_bpm.get("window_s", float(gcf.bpm_win_length) / 1000.0))
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

    # Beat-phase correction: move the grid by 0, 1/4, 1/2 or 3/4 beat to the phase
    # the learned 16th-cell model scores highest over the whole track.
    if bool(getattr(gcf, "beat_phase_correction", True)) and seg_arr.shape[0] > 0:
        yield {"status": "beat_phase"}
        taskmgr.updatetask(taskid, "Beat Phase Analyzing", 0.42)
        try:
            from analyzer_core.beat.beat_phase import detect_beat_phase, shift_beat_grid

            beats_now = np.asarray(features.get("beats_time_sec"), dtype=float)
            if beats_now.size >= 16:
                # getattr: a config object from an app started before this setting existed lacks it.
                weight_path = str(
                    getattr(gcf, "beat_phase_parameter_path", "") or "assets/weights/beat_phase_weights.json"
                ).strip()
                decision = detect_beat_phase(frame_features, beats_now, weight_path)
                print(
                    f"[BeatPhase] shift {decision.shift_beats:.2f} beat, logit sums "
                    f"{np.round(decision.logit_sums, 1).tolist()}, margin/beat {decision.margin:.3f}"
                )
                if decision.shift_index:
                    new_beats, seg_arr = shift_beat_grid(
                        beats_now, seg_arr, decision.shift_beats, float(features["duration_sec"])
                    )
                    seg_arr = seg_arr.astype(np.float32)
                    features["beats_time_sec"] = new_beats
                    features["tempo_segments"] = seg_arr
                    # Beat-synchronous chroma reads the beats as ODF frames (before the grid offset).
                    odf_len = len(synced_bpm["odf"])
                    synced_bpm["beats"] = np.clip(
                        np.round(
                            (new_beats - float(gcf.beatgrid_offset_msec) / 1000.0) / float(synced_bpm["hop_t"])
                        ).astype(np.int32),
                        0,
                        max(0, odf_len - 1),
                    )
        except Exception as exc:
            import traceback
            print(f"[BeatPhase] correction skipped: {exc}")
            traceback.print_exc()

    # Downbeat-offset realignment: assign the bar-start beat (downbeat) to each
    # segment where the detected downbeat phase changes. Beat timings are kept.
    yield {"status": "downbeat"}
    print("[Downbeat] Analysis Initalized")
    taskmgr.updatetask(taskid, "Downbeat Analyzing", 0.45)
    db_segments: list = []
    try:
        from analyzer_core.beat.downbeat_offset import (
            detect_downbeat_offset_segments,
            apply_downbeat_offset_segments,
        )
        beats_for_db = np.asarray(features.get("beats_time_sec"), dtype=float)
        if seg_arr.shape[0] > 0 and beats_for_db.size >= 2:
            downbeat_method = "dynamic" if bool(getattr(gcf, "dynamic_downbeat", False)) else "global"
            downbeat_parameter_path = str(gcf.downbeat_parameter_path).strip()
            db_segments = detect_downbeat_offset_segments(
                samp,
                int(global_sr),
                beats_for_db,
                method=downbeat_method,
                weight_path=downbeat_parameter_path,
                frames=frame_features,
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

    cleanup_max_sec = 0.5 * bpm_window_sec
    cleanup_before = int(seg_arr.shape[0]) if getattr(seg_arr, "ndim", 0) == 2 else 0
    seg_arr, removed_sandwiched = collapse_short_sandwiched_tempo_segments(
        seg_arr,
        max_duration_sec=cleanup_max_sec,
    )
    features["tempo_segments"] = seg_arr
    if removed_sandwiched:
        cleanup_after = int(seg_arr.shape[0])
        print(
            f"[Tempo] removed {removed_sandwiched} short sandwiched segment(s) "
            f"<= {cleanup_max_sec:.3f}s: {cleanup_before} -> {cleanup_after}"
        )
        for i, row in enumerate(seg_arr):
            start, end, bpm, inizio, ts = [float(x) for x in row[:5]]
            print(
                f"[Tempo]   tempo{i}: {start:.3f}-{end:.3f}s bpm={bpm:.2f} "
                f"inizio(downbeat)={inizio:.3f}s ts={int(ts)}/4"
            )

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
    yield {"status": "phrase"}
    print("[Phrase] Analysis Initalized")
    taskmgr.updatetask(taskid, "Phrase Boundary Analyzing", 0.58)
    phrase_enabled = bool(getattr(gcf, "phrase_analysis_enabled", True))
    phrase_parameter_path = str(gcf.phrase_parameter_path).strip()
    if not phrase_enabled:
        print("[Phrase] skipped: disabled in analysis settings")
        features["phrase_segments_np"] = build_phrase_segments_np([])
        features["cue_points_np"] = empty_cue_points_np()
    elif beats_time_arr.size >= 17 and seg_arr.shape[0] > 0:
        try:
            # Joint phrase boundary + functional-label detection -> Phrase data.
            # Reuse the already-decoded stereo audio (no second decode).
            phrase_segments = detect_phrase_segments(
                samp_stereo,
                int(global_sr),
                beats_time_arr,
                seg_arr,
                model_path=phrase_parameter_path,
                hpss=hpss_spectra,
            )
            features["phrase_segments_np"] = build_phrase_segments_np(phrase_segments)
            phrase_cue_points = build_phrase_cue_points(phrase_segments)
            features["cue_points_np"] = build_cue_points_np(phrase_cue_points)
            labels_summary = ", ".join(str(s["label"]) for s in phrase_segments)
            print(
                f"[Phrase] segments={len(phrase_segments)} "
                f"cue_points={len(phrase_cue_points)} labels=[{labels_summary}]"
            )
        except Exception as exc:
            import traceback

            print(f"[Phrase] joint detection skipped: {exc}")
            traceback.print_exc()
            features["phrase_segments_np"] = build_phrase_segments_np([])
            features["cue_points_np"] = empty_cue_points_np()
    else:
        print("[Phrase] skipped: too few beats")
        features["phrase_segments_np"] = build_phrase_segments_np([])
        features["cue_points_np"] = empty_cue_points_np()
    print("[Phrase] Analysis Finished")
    hpss_spectra = None

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
    taskmgr.updatetask(taskid, "Rendering/Normalizing", 0.88)
    features = normalize_gui_buffers(features)

    features_denylist = ["key_12", "logB", "full_chroma", "beatsync_chroma"]
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
