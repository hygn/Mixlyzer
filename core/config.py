from dataclasses import asdict, dataclass, fields, is_dataclass
import json
from pathlib import Path
from typing import Literal, Tuple, get_args, get_origin

from core import os_support
from utils.atomic_io import atomic_write_json


@dataclass
class libconfig:
    libpath: str
    # An empty list means the operating system's default Music location.
    local_music_directories: list[str]
    write_log: bool
    logpath: str
    rekordbox_sync_enabled: bool
    rekordbox_xml_path: str
    # How far (ms) the XML's steps of steady tempo standing in for a tempo ramp may put a beat from the
    # ramp's (0: a step per beat).
    rekordbox_ramp_tolerance_ms: float


@dataclass
class phraselabelname:
    alias: str   # shown in place of the label ("" = the label)
    abbrev: str  # the overview's short form ("" = the first letter)


@dataclass
class phraselabelnames:
    """How each phrase label is shown (utils.phrases.PHRASE_LABELS, lowercased)."""
    lead_in: phraselabelname
    intro: phraselabelname
    verse: phraselabelname
    chorus: phraselabelname
    break_chorus: phraselabelname
    bridge: phraselabelname
    outro: phraselabelname
    interlude: phraselabelname
    silence: phraselabelname

    def by_label(self) -> dict[str, phraselabelname]:
        """Names keyed by phrase label (uppercase)."""
        return {f.name.upper(): getattr(self, f.name) for f in fields(self)}


@dataclass
class viewconfig:
    language: str  # the UI's (core.i18n.LANGUAGES; "" = the operating system's), applied at the next start
    display_waveform: bool
    display_beatgrid: bool
    display_keystrip: bool
    display_JumpCUE: bool
    display_phrase: bool
    waveform_style: Literal["rgb", "3band"]  # RGB: colored by the band mix; 3band: rekordbox-style layers
    # low / mid / high ranges (Hz) of the waveforms' colours: the overview and the preview sum them from the stored
    # bands when drawn, the main view waveform filters its audio with them
    env_lo: Tuple[float, float]
    env_mid: Tuple[float, float]
    env_hi: Tuple[float, float]
    # main view waveform: drawing frame (bar) length (ms) of the short views (up to
    # core.timeline.SHORT_WINDOW_MAX_SEC wide) and of the longer ones
    mainview_wave_frame_ms_short: float
    mainview_wave_frame_ms_long: float
    mainview_wave_filter_order: int  # main view waveform: band filter order
    use_output_volume_as_peak_meter_input: bool
    fps: int
    reduce_fps_when_occluded: bool
    record_img_path: str
    phrase_labels: phraselabelnames


@dataclass
class playbackconfig:
    enable_metronome: bool
    metronome_wav_path: str
    metronome_offset_msec: float
    volume_trim_dbfs: float
    default_volume_percent: int
    use_timestretch: bool  # tempo changes keep the pitch (time stretch) instead of varispeed
    metronome_downbeat_volume_percent: int
    metronome_downbeat_pitch_semitones: float  # the click is played faster (higher) / slower
    metronome_beat_volume_percent: int
    metronome_beat_pitch_semitones: float
    metronome_ducking: bool  # lower the music under each click
    soft_clip: bool  # tanh soft clip of the output above SOFT_CLIP_KNEE


@dataclass
class memoryvalueconfig:
    offsets: str
    value_type: Literal["float", "bool", "str", "int"]
    length: int
    encoding: str
    bit_pos: int
    multiplier: float


@dataclass
class memorydeckconfig:
    time: memoryvalueconfig
    sample_index: memoryvalueconfig
    path: memoryvalueconfig
    active: memoryvalueconfig
    loaded: memoryvalueconfig


@dataclass
class externalsyncconfig:
    enabled: bool
    mode: Literal["time", "sample_index"]
    total_sample_count_source: Literal["reference_sample_rate", "file"]
    reference_sample_rate: int
    memory_process_name: str
    memory_process_pid: int
    memory_deck1: memorydeckconfig
    memory_deck2: memorydeckconfig


# Every analysis runs at this sample rate (the learned models -- beat, phrase, JumpCUE, key -- were trained on it).
ANALYSIS_SAMPLE_RATE = 22050


@dataclass
class analysisconfig:
    # The beat tracker reads every tempo in the octave from here (bpm_min .. 2 x bpm_min).
    bpm_min: int
    # Dynamic analysis: tempo changes within a track (off: one steady grid for the whole track).
    bpm_dynamic: bool
    # Dynamic downbeats of the static analysis: odd bars and meter changes (off: one bar length and
    # phase for the whole track); the dynamic analysis always has them.
    dynamic_downbeat: bool
    # The onset model that places tempo changes (retrained by Reoptimize) and its feature cache.
    refine_onset_parameter_path: str
    onset_feature_cache_path: str
    # The beat tracker's frame TCN, metronome critic and contextual downbeat model (Reoptimize
    # retrains them) and the frame-feature cache the beat tracker's optimizers share.
    beat_tcn_parameter_path: str
    beat_critic_parameter_path: str
    # the decoder's costs (JSON), refitted with the beat model they belong to
    beat_decoder_parameter_path: str
    downbeat_ctx_parameter_path: str
    beat_feature_cache_path: str
    phrase_parameter_path: str
    phrase_feature_cache_path: str
    jumpcue_parameter_path: str
    jumpcue_feature_cache_path: str
    # Dynamic key analysis: key changes within a track (off: one key for the whole track, the dynamic
    # analysis's key in the intro / outro / interludes).
    key_dynamic: bool
    # The chord-aware key model (retrained by Reoptimize) and the frame chroma cache it is retrained on.
    key_parameter_path: str
    key_feature_cache_path: str
    # AutoCUE: what the analysis puts at the LEAD_IN end, an early first
    # downbeat, or 0:00: a CUE point, a JumpCUE of its own, or nothing.
    autocue: Literal["off", "cue_point", "jumpcue"]


@dataclass
class cuepointrule:
    at_start: bool               # a CUE point where a phrase of the label starts
    at_end: bool                 # ... and where it ends
    same_label_boundaries: bool  # also between two phrases of the label (else only next to another label)


@dataclass
class cuepointconfig:
    """CUE points the analysis sets from the phrases, by phrase label (the
    predefined labels, utils.phrases.PHRASE_LABELS, lowercased). LEAD_IN has no
    rule: where it ends is analysisconfig.autocue."""
    intro: cuepointrule
    verse: cuepointrule
    chorus: cuepointrule
    break_chorus: cuepointrule
    bridge: cuepointrule
    outro: cuepointrule
    interlude: cuepointrule
    silence: cuepointrule

    def by_label(self) -> dict[str, cuepointrule]:
        """Rules keyed by phrase label (uppercase)."""
        return {f.name.upper(): getattr(self, f.name) for f in fields(self)}


@dataclass
class searchconfig:
    """The library search by how titles / artists / albums read (core.reading)."""
    # the languages Han characters (without kana) are read in for the library's stored readings
    # (core.reading.HAN_LANGUAGES); a change reads the library again
    db_languages: list[str]
    # Han mixed with other characters found by its Korean Hanja reading too (when Korean Hanja is a
    # language): beside Latin letters / Hangul all of it (the per-language readings), beside kana its
    # runs of han_run_min (2 or more) Han characters, the rest in Japanese (Re:ゼロから始める異世界生活
    # ... isegyesaenghwal); for the stored readings (a change reads the library again) and the query
    han_runs: bool
    han_run_min: int
    # the languages a query's Han characters are read in
    query_languages: list[str]
    query_hangul_korean: bool    # a query's Hangul read as Korean (사회 sahoe: Korean titles, Korean Hanja readings)
    query_hangul_foreign: bool   # ... as Korean spells a foreign word (도쿄 tokyo: Japanese / English titles)
    # a query typed on the 2-set keyboard without the 한/영 key also read as what was meant (tkghl 사회, 무ㅐㅜㄷ anone)
    query_keyboard_layout: bool

    def han_run(self) -> int:
        """core.reading's ``han_run``: the Han run length read in Korean Hanja beside kana, 0: off."""
        return max(2, int(self.han_run_min)) if self.han_runs else 0

    def stored_readings(self) -> tuple:
        """What the library's stored readings depend on: a change reads the library again."""
        return tuple(self.db_languages), self.han_run()


@dataclass
class config:
    analysisconfig: analysisconfig
    libconfig: libconfig
    viewconfig: viewconfig
    playbackconfig: playbackconfig
    externalsyncconfig: externalsyncconfig
    cuepointconfig: cuepointconfig
    searchconfig: searchconfig

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict):
        return _coerce_dataclass(cls, _normalize_config_payload(payload))


def search_settings(payload: dict | None) -> searchconfig:
    """The search settings of a config.json dict, the defaults filling what it lacks."""
    return _coerce_dataclass(searchconfig, {**asdict(default_cfg().searchconfig), **(payload or {})})


def _normalize_config_payload(payload: dict) -> dict:
    """Translate configuration keys from older Mixlyzer releases."""
    normalized = dict(payload) if isinstance(payload, dict) else {}
    raw_lib = normalized.get("libconfig")
    if not isinstance(raw_lib, dict):
        return normalized
    lib = dict(raw_lib)
    if "local_music_directories" not in lib and "local_music_directory" in lib:
        legacy = str(lib.get("local_music_directory") or "").strip()
        # ~/Music was the old hard-coded default. An empty list now asks the
        # OS for its Music location, including a redirected Windows folder.
        lib["local_music_directories"] = [] if legacy in ("", "~/Music") else [legacy]
    lib.pop("local_music_directory", None)
    normalized["libconfig"] = lib
    # the waveform band ranges moved from the analysis settings to the view settings
    analysis, view = normalized.get("analysisconfig"), normalized.get("viewconfig")
    if isinstance(analysis, dict) and isinstance(view, dict):
        analysis, view = dict(analysis), dict(view)
        for key in ("env_lo", "env_mid", "env_hi"):
            if key in analysis:
                view.setdefault(key, analysis.pop(key))
        normalized["analysisconfig"], normalized["viewconfig"] = analysis, view
    return normalized


def _coerce_dataclass(cls, payload: dict):
    src = payload if isinstance(payload, dict) else {}
    kwargs = {}
    for field in fields(cls):
        kwargs[field.name] = _coerce_value(field.type, src.get(field.name))
    return cls(**kwargs)


def _coerce_value(type_hint, value):
    origin = get_origin(type_hint)
    if is_dataclass(type_hint):
        return _coerce_dataclass(type_hint, value)
    if origin is Literal:
        options = get_args(type_hint)
        if value in options:
            return value
        return options[0] if options else value
    if origin in (tuple, Tuple):
        args = get_args(type_hint)
        raw_values = list(value) if isinstance(value, (list, tuple)) else []
        coerced = []
        for idx, arg in enumerate(args):
            raw = raw_values[idx] if idx < len(raw_values) else None
            coerced.append(_coerce_scalar(arg, raw))
        return tuple(coerced)
    if origin is list:
        (item_type,) = get_args(type_hint) or (object,)
        raw_values = value if isinstance(value, (list, tuple)) else []
        return [_coerce_scalar(item_type, item) for item in raw_values]
    return _coerce_scalar(type_hint, value)


def _coerce_scalar(type_hint, value):
    if type_hint is bool:
        return bool(value)
    if type_hint is int:
        try:
            return int(value)
        except Exception:
            return 0
    if type_hint is float:
        try:
            return float(value)
        except Exception:
            return 0.0
    if type_hint is str:
        return "" if value is None else str(value)
    return value


def _default_memory_value(
    *,
    offsets: str,
    value_type: Literal["float", "bool", "str", "int"],
    length: int = 0,
    encoding: str = "utf-8",
    bit_pos: int = 0,
    multiplier: float = 1.0,
) -> memoryvalueconfig:
    return memoryvalueconfig(
        offsets=offsets,
        value_type=value_type,
        length=length,
        encoding=encoding,
        bit_pos=bit_pos,
        multiplier=multiplier,
    )


def default_cfg():
    lcfg = libconfig(
        libpath="library",
        local_music_directories=[],
        logpath="mixlyzer.log",
        write_log=False,
        rekordbox_sync_enabled=True,
        rekordbox_xml_path="./rekordbox.xml",
        rekordbox_ramp_tolerance_ms=1.0,
    )
    acfg = analysisconfig(
        bpm_min=100,
        bpm_dynamic=True,
        dynamic_downbeat=True,
        refine_onset_parameter_path="assets/weights/refine_onset_hgb.npz",
        onset_feature_cache_path="featurecache/onset",
        beat_tcn_parameter_path="assets/weights/beat_tcn.npz",
        beat_critic_parameter_path="assets/weights/beat_critic.npz",
        beat_decoder_parameter_path="assets/weights/beat_decoder.json",
        downbeat_ctx_parameter_path="assets/weights/downbeat_ctx_hgb.npz",
        beat_feature_cache_path="featurecache/beat",
        phrase_parameter_path="assets/weights/phrase_analyzer.npz",
        phrase_feature_cache_path="featurecache/phrase",
        jumpcue_parameter_path="assets/weights/jumpcue_jump_point.json",
        jumpcue_feature_cache_path="featurecache/jumpcue/train",
        key_dynamic=True,
        key_parameter_path="assets/weights/key_chord_cnn.npz",
        key_feature_cache_path="featurecache/key",
        autocue="cue_point",
    )
    vcfg = viewconfig(
        language="",
        display_waveform=True,
        display_beatgrid=True,
        display_keystrip=True,
        display_JumpCUE=True,
        display_phrase=True,
        waveform_style="rgb",
        env_lo=(20.0, 200.0),
        env_mid=(250.0, 1500.0),
        env_hi=(1600.0, 11025.0),
        mainview_wave_frame_ms_short=1.0,
        mainview_wave_frame_ms_long=2.0,
        mainview_wave_filter_order=4,
        use_output_volume_as_peak_meter_input=False,
        fps=60,
        reduce_fps_when_occluded=True,
        record_img_path="assets/images/vinyl.png",
        phrase_labels=phraselabelnames(**{
            f.name: phraselabelname(alias="", abbrev="") for f in fields(phraselabelnames)
        }),
    )
    pcfg = playbackconfig(
        enable_metronome=False,
        metronome_wav_path="assets/sound/click.wav",
        metronome_offset_msec=0.0,
        volume_trim_dbfs=-9.0,
        default_volume_percent=100,
        use_timestretch=True,
        metronome_downbeat_volume_percent=60,
        metronome_downbeat_pitch_semitones=7.0,
        metronome_beat_volume_percent=60,
        metronome_beat_pitch_semitones=0.0,
        metronome_ducking=True,
        soft_clip=True,
    )
    xcfg = externalsyncconfig(
        enabled=False,
        mode="sample_index",
        total_sample_count_source="reference_sample_rate",
        reference_sample_rate=44100,
        memory_process_name="",
        memory_process_pid=0,
        memory_deck1=memorydeckconfig(
            time=_default_memory_value(offsets="0", value_type="float"),
            sample_index=_default_memory_value(offsets="0", value_type="int"),
            path=_default_memory_value(offsets="0", value_type="str", length=2048),
            active=_default_memory_value(offsets="0", value_type="bool", bit_pos=0),
            loaded=_default_memory_value(offsets="0", value_type="bool", bit_pos=0),
        ),
        memory_deck2=memorydeckconfig(
            time=_default_memory_value(offsets="0", value_type="float"),
            sample_index=_default_memory_value(offsets="0", value_type="int"),
            path=_default_memory_value(offsets="0", value_type="str", length=2048),
            active=_default_memory_value(offsets="0", value_type="bool", bit_pos=0),
            loaded=_default_memory_value(offsets="0", value_type="bool", bit_pos=0),
        ),
    )
    # CUE points at the start of an interlude and of the outro (not between two
    # of them)
    cue_rules = {
        f.name: cuepointrule(at_start=f.name in ("interlude", "outro"), at_end=False, same_label_boundaries=False)
        for f in fields(cuepointconfig)
    }
    # every language the readers have (core.reading.HAN_LANGUAGES)
    han_languages = ["ja", "zh", "yue", "ko-hanja", "vi", "zh-tw"]
    scfg = searchconfig(
        db_languages=list(han_languages),
        han_runs=True,
        han_run_min=3,
        query_languages=list(han_languages),
        query_hangul_korean=True,
        query_hangul_foreign=True,
        query_keyboard_layout=True,
    )
    return config(
        libconfig=lcfg,
        analysisconfig=acfg,
        viewconfig=vcfg,
        playbackconfig=pcfg,
        externalsyncconfig=xcfg,
        cuepointconfig=cuepointconfig(**cue_rules),
        searchconfig=scfg,
    )


def ui_language() -> str:
    """config.json's UI language (viewconfig.language), only read: set before anything else is done."""
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            value = json.load(f).get("viewconfig", {}).get("language", "")
    except (OSError, ValueError, AttributeError):
        return ""
    return value if isinstance(value, str) else ""


def load_cfg() -> config:
    def _merge_defaults(default_dict: dict, loaded_dict: dict) -> dict:
        merged = dict(default_dict)
        for key, val in loaded_dict.items():
            if isinstance(val, dict) and isinstance(merged.get(key), dict):
                merged[key] = _merge_defaults(merged[key], val)
            else:
                merged[key] = val
        return merged

    def _ensure_library_dir(cfg: config) -> None:
        lib_path = Path(cfg.libconfig.libpath)
        if not lib_path.exists():
            lib_path.mkdir(parents=True, exist_ok=True)

    def _ensure_feature_cache_dirs(cfg: config) -> None:
        for path_text in (
            cfg.analysisconfig.onset_feature_cache_path,
            cfg.analysisconfig.beat_feature_cache_path,
            cfg.analysisconfig.phrase_feature_cache_path,
            cfg.analysisconfig.jumpcue_feature_cache_path,
            cfg.analysisconfig.key_feature_cache_path,
        ):
            text = str(path_text or "").strip()
            if text:
                Path(text).mkdir(parents=True, exist_ok=True)

    cfg = default_cfg()
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            loaded = json.load(f)
            loaded = _normalize_config_payload(loaded)
            merged = _merge_defaults(cfg.to_dict(), loaded)
            cfg = config.from_dict(merged)
    except (TypeError, FileNotFoundError, json.JSONDecodeError):
        atomic_write_json("config.json", cfg.to_dict(), ensure_ascii=True, indent=None)
    _ensure_library_dir(cfg)
    _ensure_feature_cache_dirs(cfg)
    if not os_support.MEMORY_SYNC:
        cfg.externalsyncconfig.enabled = False   # no memory reads on this system (macOS)
    return cfg
