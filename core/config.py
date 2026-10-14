from dataclasses import asdict, dataclass, fields, is_dataclass
import json
from pathlib import Path
from typing import Literal, Tuple, get_args, get_origin

from utils.atomic_io import atomic_write_json


@dataclass
class libconfig:
    libpath: str
    write_log: bool
    logpath: str
    rekordbox_sync_enabled: bool
    rekordbox_xml_path: str


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
    display_waveform: bool
    display_beatgrid: bool
    display_keystrip: bool
    display_JumpCUE: bool
    display_phrase: bool
    waveform_style: Literal["rgb", "3band"]  # RGB: colored by the band mix; 3band: rekordbox-style layers
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


@dataclass
class analysisconfig:
    use_hpss: bool
    analysis_samp_rate: int
    chroma_method: Literal["cqt", "cens"]
    chroma_hop_length: int
    chroma_cqt_bins_per_octave: int
    chroma_cqt_octaves: int
    bpm_hop_length: int
    bpm_win_length: int
    bpm_min: int
    bpm_max: int
    bpm_dynamic: bool
    bpm_adaptive_window: bool
    onset_source: Literal["librosa", "optimized"]
    onset_parameter_path: str
    onset_feature_cache_path: str
    beat_phase_correction: bool
    beat_phase_parameter_path: str
    beat_phase_feature_cache_path: str
    dynamic_downbeat: bool
    downbeat_parameter_path: str
    downbeat_feature_cache_path: str
    beatgrid_offset_msec: float
    # low / mid / high bands (Hz) of the stored overview levels; the main view
    # waveform uses them too
    env_lo: Tuple[float, float]
    env_mid: Tuple[float, float]
    env_hi: Tuple[float, float]
    phrase_analysis_enabled: bool
    phrase_parameter_path: str
    phrase_feature_cache_path: str
    jumpcue_parameter_path: str
    jumpcue_feature_cache_path: str
    # AutoCUE: what the analysis puts where the LEAD_IN ends (the first phrase
    # after it): a CUE point, a JumpCUE of its own, or nothing.
    autocue: Literal["off", "cue_point", "jumpcue"]


@dataclass
class keyconfig:
    min_offset: float
    pitch_self: float
    pitch_semitone: float
    pitch_fifth: float
    pitch_others: float


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
class config:
    analysisconfig: analysisconfig
    keyconfig: keyconfig
    libconfig: libconfig
    viewconfig: viewconfig
    playbackconfig: playbackconfig
    externalsyncconfig: externalsyncconfig
    cuepointconfig: cuepointconfig

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict):
        return _coerce_dataclass(cls, payload)


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
        logpath="mixlyzer.log",
        write_log=False,
        rekordbox_sync_enabled=True,
        rekordbox_xml_path="./rekordbox.xml",
    )
    acfg = analysisconfig(
        analysis_samp_rate=22050,
        chroma_method="cens",
        chroma_hop_length=512,
        chroma_cqt_bins_per_octave=36,
        chroma_cqt_octaves=4,
        use_hpss=True,
        bpm_hop_length=256,
        bpm_win_length=5000,
        bpm_max=200,
        bpm_min=100,
        bpm_dynamic=True,
        bpm_adaptive_window=True,
        onset_source="optimized",
        onset_parameter_path="assets/weights/onset_feature_weights.json",
        onset_feature_cache_path="featurecache/onset",
        beat_phase_correction=True,
        beat_phase_parameter_path="assets/weights/beat_phase_weights.json",
        beat_phase_feature_cache_path="featurecache/beat_phase",
        dynamic_downbeat=True,
        downbeat_parameter_path="assets/weights/downbeat_feature_weights.json",
        downbeat_feature_cache_path="featurecache/downbeat",
        beatgrid_offset_msec=10.0,
        env_lo=(20.0, 200.0),
        env_mid=(250.0, 1500.0),
        env_hi=(1600.0, 11025.0),
        phrase_analysis_enabled=True,
        phrase_parameter_path="assets/weights/phrase_analyzer.npz",
        phrase_feature_cache_path="featurecache/phrase",
        jumpcue_parameter_path="assets/weights/jumpcue_jump_point.json",
        jumpcue_feature_cache_path="featurecache/jumpcue/train",
        autocue="cue_point",
    )
    kcfg = keyconfig(
        min_offset=0.2,
        pitch_self=0.9,
        pitch_semitone=0.02,
        pitch_fifth=0.0,
        pitch_others=0.01,
    )
    vcfg = viewconfig(
        display_waveform=True,
        display_beatgrid=True,
        display_keystrip=True,
        display_JumpCUE=True,
        display_phrase=True,
        waveform_style="rgb",
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
    return config(
        libconfig=lcfg,
        analysisconfig=acfg,
        keyconfig=kcfg,
        viewconfig=vcfg,
        playbackconfig=pcfg,
        externalsyncconfig=xcfg,
        cuepointconfig=cuepointconfig(**cue_rules),
    )


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
            cfg.analysisconfig.beat_phase_feature_cache_path,
            cfg.analysisconfig.downbeat_feature_cache_path,
            cfg.analysisconfig.phrase_feature_cache_path,
            cfg.analysisconfig.jumpcue_feature_cache_path,
        ):
            text = str(path_text or "").strip()
            if text:
                Path(text).mkdir(parents=True, exist_ok=True)

    cfg = default_cfg()
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            loaded = json.load(f)
            merged = _merge_defaults(cfg.to_dict(), loaded if isinstance(loaded, dict) else {})
            cfg = config.from_dict(merged)
    except (TypeError, FileNotFoundError, json.JSONDecodeError):
        atomic_write_json("config.json", cfg.to_dict(), ensure_ascii=True, indent=None)
    _ensure_library_dir(cfg)
    _ensure_feature_cache_dirs(cfg)
    return cfg
