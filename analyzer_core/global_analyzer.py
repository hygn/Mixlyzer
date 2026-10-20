import numpy as np
import time
import os
import base64
import struct
from pymediainfo import MediaInfo
import soundfile as sf
from dataclasses import replace
from pathlib import Path
from PySide6 import QtGui
from typing import Optional
from analyzer_core.key.chord_key import analyze_keys as analyze_chord_keys
from core.beat_geometry import downbeat_times
from analyzer_core.beat.frame_features import FRAME_HOP_LENGTH, extract_frame_features
from analyzer_core.hpss import HPSS_HOP_LENGTH, HPSS_N_FFT, hpss_audio_with_spectra
from analyzer_core.dsp.resample import resample_audio
from utils.wave import WAVE_OVERVIEW_SAMPLE_RATE, build_wave_overview
from analyzer_core.self_correlation.JumpCUE import JumpCueConfig, JumpCueEngine
from analyzer_core.self_correlation.jump_point_model import jump_point_beat_features
from analyzer_core.cue_and_phrase import detect_phrase_segments
from utils.jump_cues import autocue_jump_pair, build_jump_cues_np
from utils.cue_points import (
    add_autocue_cue_point,
    autocue_time_sec,
    build_cue_points_np,
    build_phrase_cue_points,
    lead_in_end_sec,
)
from utils.phrases import build_phrase_segments_np
from core.audio.decoder import decode_to_memmap, get_total_samples
from core.library_handler import LibraryDB
from core.analysis_lib_handler import FeatureNPZStore
from core.thumbnails import save_thumbnail
from core.config import ANALYSIS_SAMPLE_RATE, config
from core.i18n import tr
from core.taskmanager import taskmanager
from core.linear_segments import build_bpm_segments, build_key_segments, build_phrase_index_rows
from analyzer_core.utils import prime_physical_core_count

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
            raise RuntimeError(tr("Audio decode failed via FFmpeg and SoundFile: {path}").format(path=path)) from exc
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
        y = resample_audio(
            y,
            sr,
            target_sr,
            axis=0 if stereo else -1,
            quality="HQ",
        )
    return np.ascontiguousarray(y, dtype=np.float32)

def _file_stats(path: str) -> tuple[int, float]:
    try:
        st = os.stat(path)
        return int(st.st_size), float(st.st_mtime)
    except FileNotFoundError:
        return 0, 0.0


_RIFF_INFO_FIELDS = {
    b"INAM": "title",
    b"IART": "artist",
    b"IPRD": "album",
    b"ICMT": "comment",
}


def _utf8_wav_info(path: str) -> dict[str, str]:
    """Read UTF-8 RIFF INFO tags without applying the Windows ANSI code page.

    MediaInfo interprets RIFF INFO text using the active Windows code page.
    Recordings written by rekordbox/PioneerDJ contain UTF-8 here, which turns
    Japanese text into mojibake on a Korean Windows installation.
    """
    if Path(path).suffix.lower() not in (".wav", ".wave"):
        return {}
    result: dict[str, str] = {}
    try:
        with open(path, "rb") as stream:
            header = stream.read(12)
            if len(header) != 12 or header[:4] != b"RIFF" or header[8:] != b"WAVE":
                return result
            riff_end = min(
                os.fstat(stream.fileno()).st_size,
                8 + struct.unpack("<I", header[4:8])[0],
            )
            while stream.tell() + 8 <= riff_end:
                chunk_id = stream.read(4)
                size_data = stream.read(4)
                if len(size_data) != 4:
                    break
                chunk_size = struct.unpack("<I", size_data)[0]
                chunk_start = stream.tell()
                chunk_end = chunk_start + chunk_size
                if chunk_end > riff_end:
                    break
                if chunk_id == b"LIST" and chunk_size >= 4 and stream.read(4) == b"INFO":
                    while stream.tell() + 8 <= chunk_end:
                        info_id = stream.read(4)
                        info_size_data = stream.read(4)
                        if len(info_size_data) != 4:
                            break
                        info_size = struct.unpack("<I", info_size_data)[0]
                        info_end = stream.tell() + info_size
                        if info_end > chunk_end:
                            break
                        raw = stream.read(info_size).rstrip(b"\x00")
                        field = _RIFF_INFO_FIELDS.get(info_id)
                        if field and raw:
                            try:
                                value = raw.decode("utf-8-sig").strip()
                            except UnicodeDecodeError:
                                value = ""
                            if value:
                                result[field] = value
                        stream.seek(info_end + (info_size & 1))
                stream.seek(chunk_end + (chunk_size & 1))
    except (OSError, struct.error):
        return {}
    return result


def extract_tags(path: str) -> tuple[str, str, str, str]:
    title = os.path.splitext(os.path.basename(path))[0]
    artist = ""
    album = ""
    comment = ""
    try:
        wav_info = _utf8_wav_info(path)
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
            wav_info.get("title")
            or _pick_from_track(general_track, "title", "track_name")
            or _pick_from_track(audio_track, "title", "track_name")
            or title
        )
        ar = (
            wav_info.get("artist")
            or _pick_from_track(general_track, "artist", "performer")
            or _pick_from_track(audio_track, "artist", "performer")
            or _pick_from_track(general_track, "album_artist", "album_performer", "album_composer")
            or _pick_from_track(audio_track, "album_artist", "album_performer", "album_composer")
            or ""
        )
        al = (
            wav_info.get("album")
            or _pick_from_track(general_track, "album")
            or _pick_from_track(audio_track, "album")
            or ""
        )
        cm = (
            wav_info.get("comment")
            or _pick_from_track(general_track, "comment", "description")
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
    phrase_sections=None,
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
    if phrase_sections is not None:
        library.replace_phrase_index(uid, phrase_sections)
        library.conn.commit()
    store.save(uid, dict(features))
    save_thumbnail(store.base_dir, uid, getAlbumArt(properties.get("path")))
    return library.list_all()

def getAlbumArt(path) -> Optional[QtGui.QImage]:
    if not path:
        return None
    try:
        mi = MediaInfo.parse(path, cover_data=True)
        if not mi:
            return None
        data: Optional[bytes] = None
        for media_track in mi.tracks:
            blob = getattr(media_track, "cover_data", None)
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

def precompute_features(path: str, config: config, taskmgr: taskmanager, taskid:int, force_analyze: bool = False):
    l = LibraryDB(os.path.join(config.libconfig.libpath, "library.db"),
                  reading_languages=config.searchconfig.db_languages, han_run=config.searchconfig.han_run())
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
    yield {"status": tr("Loading")}
    print("[Analyzer] Analysis Initalized")
    gcf = config.analysisconfig
    global_sr = ANALYSIS_SAMPLE_RATE
    print("[Analyzer] Loading Track")
    # Decode once: stereo for the structure detector, mono derived for the rest.
    samp_stereo = fast_load(path, target_sr=global_sr, stereo=True)
    samp = np.ascontiguousarray(samp_stereo.mean(axis=1), dtype=np.float32)
    taskmgr.updatetask(taskid, tr("Processing HPSS"), 0.1)

    yield {"status": "HPSS"}
    print("[Analyzer] Processing HPSS")
    # HPSS
    # The spectra are reused by the phrase features (same mono mix, STFT and HPSS).
    y_harm, y_perc, hpss_spectra = hpss_audio_with_spectra(samp, int(global_sr))
    features = {}

    yield {"status": tr("Waveform")}
    print("[Waveform] Analysis Initalized")
    taskmgr.updatetask(taskid, tr("Processing Waveform"), 0.2)
    # Overview waveform: the min / max of each column and the power of 12 log-spaced bands (low / mid / high are
    # summed from them when drawn), from a mono decode at WAVE_OVERVIEW_SAMPLE_RATE, not the analysis rate, so
    # the bands are not cut at the analysis Nyquist; and the detail waveform's three-band scale (these ranges).
    wave_bands = tuple((float(low), float(high)) for low, high in (config.viewconfig.env_lo, config.viewconfig.env_mid, config.viewconfig.env_hi))
    features.update(build_wave_overview(
        fast_load(path, WAVE_OVERVIEW_SAMPLE_RATE),
        WAVE_OVERVIEW_SAMPLE_RATE,
        wave_bands,
        n_fft=HPSS_N_FFT,
        hop_length=HPSS_HOP_LENGTH,
    ))
    features["duration_sec"] = samp.size / float(global_sr)
    print("[Waveform] Analysis Finished")

    yield {"status": tr("Tempo")}
    print("[Tempo] Analysis Initalized")
    taskmgr.updatetask(taskid, tr("Tempo Analyzing"), 0.30)
    # Frame features on the shared frame grid: the phrase features read them below.
    frame_features = extract_frame_features(
        samp, int(global_sr), y_perc, y_harm, hop_length=FRAME_HOP_LENGTH
    )
    # The generalized tracker (analyzer_core.beat.general): a frame TCN (three trainings) ->
    # section decoding with one section cost -> ramps -> bar-supported phase -> metronome critic ->
    # tempo-change boundaries on the local onset -> drifting sections -> junction rules -> bars from
    # two downbeat observers -> the consensus of the three; rows with inizio on a downbeat.
    from analyzer_core.beat.general import (
        FRAME_HOP_LENGTH as TRACKER_HOP, SAMPLE_RATE as TRACKER_SR, dominant_tempo, load_models,
        track as track_beats,
    )
    from analyzer_core.beat.learned_onset import REFINE_ONSET_PATH, refine_onset_activation

    # Its models were trained on these frames: the HPSS parts at the analysis rate, hop 256.
    assert (int(global_sr), FRAME_HOP_LENGTH) == (TRACKER_SR, TRACKER_HOP)
    tracker_frames = frame_features
    frame_times = np.asarray(tracker_frames.frame_times, dtype=float)
    refine_onset = None
    try:
        refine_path = str(getattr(gcf, "refine_onset_parameter_path", "") or REFINE_ONSET_PATH).strip()
        refine_onset = (
            refine_onset_activation(tracker_frames, refine_path).astype(np.float64),
            float(np.median(np.diff(frame_times))) if frame_times.size > 1 else TRACKER_HOP / TRACKER_SR,
            float(frame_times[0]),
        )
    except Exception as exc:
        print(f"[Tempo] boundary onset unavailable ({exc}); tempo changes stay where the decoder put them")
    dynamic = bool(getattr(gcf, "bpm_dynamic", True))
    tracked = track_beats(
        tracker_frames, float(gcf.bpm_min),
        load_models(
            str(getattr(gcf, "downbeat_ctx_parameter_path", "") or "").strip() or None,
            str(getattr(gcf, "beat_tcn_parameter_path", "") or "").strip() or None,
            str(getattr(gcf, "beat_critic_parameter_path", "") or "").strip() or None,
            str(getattr(gcf, "beat_decoder_parameter_path", "") or "").strip() or None,
        ),
        refine_onset,
        dynamic=dynamic, dynamic_downbeat=dynamic or bool(getattr(gcf, "dynamic_downbeat", True)),
    )
    duration_sec = float(features["duration_sec"])
    seg_arr = np.asarray(tracked.segments, dtype=np.float64).copy()
    if seg_arr.size:
        seg_arr[0, 0] = 0.0
        seg_arr[-1, 1] = max(float(seg_arr[-1, 1]), duration_sec)
    seg_arr = seg_arr.astype(np.float32)
    # The stored rows are the grid (the editor and the views rebuild it from them): the saved beats are theirs.
    from core.tempo_rows import rebuild_grid_from_segments
    beats_out = rebuild_grid_from_segments(seg_arr)
    beats_out = beats_out[(beats_out >= 0.0) & (beats_out <= duration_sec)]
    features["tempo_segments"] = seg_arr
    features["beats_time_sec"] = beats_out
    features["tempo_global"] = dominant_tempo(seg_arr)
    print(
        f"[Tempo] consensus kept seed {tracked.seed}; {seg_arr.shape[0]} tempo rows, "
        f"{beats_out.size} beats, main tempo {features['tempo_global']:.2f}"
    )
    for i, row in enumerate(seg_arr):
        start, end, bpm, inizio, ts = [float(x) for x in row[:5]]
        bpm_end = float(row[5]) if row.size > 5 else bpm
        ramp = f"->{bpm_end:.2f}" if abs(bpm_end - bpm) > 1e-6 else ""
        print(f"[Tempo]   tempo{i}: {start:.3f}-{end:.3f}s bpm={bpm:.2f}{ramp} inizio(downbeat)={inizio:.3f}s ts={int(ts)}/4")
    print("[Tempo] Analysis Finished")

    beats_time_arr = np.asarray(features.get("beats_time_sec"), dtype=float)
    yield {"status": tr("Phrase")}
    print("[Phrase] Analysis Initalized")
    taskmgr.updatetask(taskid, tr("Phrase Boundary Analyzing"), 0.58)
    phrase_enabled = bool(getattr(gcf, "phrase_analysis_enabled", True))
    phrase_parameter_path = str(gcf.phrase_parameter_path).strip()
    # The phrase boundary probability is a JumpCUE feature (below).
    phrase_boundary_prob = None
    phrase_feature_z = None
    # AutoCUE (Settings > Analysis > CUE): lead-in end, an early first
    # downbeat, or 0:00 when the track begins before a usable downbeat.
    autocue = gcf.autocue
    phrase_segments = []
    phrase_cue_points = []
    if not phrase_enabled:
        print("[Phrase] skipped: disabled in analysis settings")
    elif beats_time_arr.size >= 17 and seg_arr.shape[0] > 0:
        try:
            # Joint phrase boundary + functional-label detection -> Phrase data.
            # Reuse the already-decoded stereo audio (no second decode).
            phrase_segments, phrase_boundary_prob, phrase_feature_z = detect_phrase_segments(
                samp_stereo,
                int(global_sr),
                beats_time_arr,
                seg_arr,
                model_path=phrase_parameter_path,
                hpss=hpss_spectra,
                return_boundary_probability=True,
                return_feature_z=True,
            )
            phrase_cue_points = build_phrase_cue_points(phrase_segments, config.cuepointconfig.by_label())
        except Exception as exc:
            import traceback

            print(f"[Phrase] joint detection skipped: {exc}")
            traceback.print_exc()
            phrase_segments = []
            phrase_cue_points = []
    else:
        print("[Phrase] skipped: too few beats")
    features["phrase_segments_np"] = build_phrase_segments_np(phrase_segments)
    lead_in_end = lead_in_end_sec(phrase_segments)
    autocue_target = autocue_time_sec(phrase_segments, beats_time_arr, seg_arr)
    if autocue == "cue_point":
        phrase_cue_points = add_autocue_cue_point(
            phrase_cue_points,
            autocue_target,
            lead_in_end=lead_in_end is not None,
        )
    features["cue_points_np"] = build_cue_points_np(phrase_cue_points)
    labels_summary = ", ".join(str(s["label"]) for s in phrase_segments)
    print(
        f"[Phrase] segments={len(phrase_segments)} "
        f"cue_points={len(phrase_cue_points)} labels=[{labels_summary}]"
    )
    print("[Phrase] Analysis Finished")

    # Learned JumpCUE detection needs phrase-boundary probability.  The simple
    # AutoCUE JumpCUE does not and is still added below when detection is skipped.
    yield {"status": "JumpCUE"}
    taskmgr.updatetask(taskid, tr("JumpCUE Analyzing"), 0.62)
    jump_pairs = []
    if not phrase_enabled:
        print("[JumpCUE] learned pairs skipped: phrase analysis is disabled")
    elif phrase_boundary_prob is None:
        print("[JumpCUE] learned pairs skipped: no phrase boundary probability")
    elif beats_time_arr.size >= 2:
        jump_config = JumpCueConfig(jump_point_model_path=str(gcf.jumpcue_parameter_path))
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
    features["jump_cues_np"] = build_jump_cues_np(
        jump_pairs,
        canonicalize_labels=True,
        merge_coincident=True,
    )
    # The automatic result is kept beside the user's JumpCUEs (which edits
    # replace): the JumpCUE optimizer learns from where the two differ.
    features["jump_cues_auto_np"] = {k: np.array(v, copy=True) for k, v in features["jump_cues_np"].items()}
    hpss_spectra = None

    yield {"status": tr("Key")}
    print("[Key] Analysis Initalized")
    taskmgr.updatetask(taskid, tr("Analyzing Keys"), 0.7)
    try:
        # chord-aware keys on the bars and phrases (analyzer_core/key/chord_key.py)
        key, key_segments = analyze_chord_keys(
            y_harm,
            int(global_sr),
            beats_time_arr,
            downbeat_times(beats_time_arr, seg_arr),
            [float(s["start"]) for s in phrase_segments],
            float(features.get("duration_sec", 0.0)),
            str(getattr(gcf, "key_parameter_path", "") or "").strip() or None,
            dynamic=bool(getattr(gcf, "key_dynamic", True)),
            phrases=phrase_segments,
        )
    except Exception as exc:
        import traceback

        print(f"[Key] analysis skipped: {exc}")
        traceback.print_exc()
        key, key_segments = np.zeros(0, int), np.zeros((0, 4), float)
    print("[Key] Analysis Finished")
    features["key_segments"] = key_segments
    # the automatic result kept beside the user's keys (which edits replace): the key optimizer learns from
    # the tracks whose keys differ from it
    features["key_segments_auto"] = np.array(key_segments, dtype=float, copy=True)
    features["sr"] = global_sr

    # DB Features
    taskmgr.updatetask(taskid, tr("Processing Metadata"), 0.85)
    duration_out = float(features.get("duration_sec", 0.0))

    bpm_out = features.get("tempo_global", 0.0)

    values, counts = np.unique(key, return_counts=True)
    key_int = int(values[np.argmax(counts)]) if values.size else None

    fsize, fmtime = _file_stats(path)
    taskmgr.updatetask(taskid, tr("Rendering/Normalizing"), 0.88)

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
    phrase_sections = build_phrase_index_rows(phrase_segments, bpm_segments, key_segments_db)
    features_properties = {"features": features, "properties": properties, "update_db": True, "taskid":taskid}
    yield {"status": tr("Saving")}
    print("[Analyzer] Saving Results")
    taskmgr.updatetask(taskid, tr("Saving Result"), 0.92)
    try:
        lib_full = _persist_analysis_result(
            l,
            feature_store,
            features,
            properties,
            bpm_segments=bpm_segments,
            key_segments=key_segments_db,
            phrase_sections=phrase_sections,
        )
    finally:
        l.close()
    features_properties["library"] = lib_full
    print("[Analyzer] Analysis Finished")
    taskmgr.updatetask(taskid, tr("Finished"), 1)

    yield features_properties
