from dataclasses import asdict, dataclass, fields, is_dataclass
import json
from pathlib import Path
from typing import Literal, Tuple, get_args, get_origin


@dataclass
class libconfig:
    libpath: str
    write_log: bool
    logpath: str


@dataclass
class viewconfig:
    display_waveform: bool
    display_beatgrid: bool
    display_keystrip: bool
    display_JumpCUE: bool
    fps: int
    enable_metronome: bool
    record_img_path: str
    metronome_wav_path: str


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
    beatgrid_offset_msec: float
    env_frame_ms: int
    env_lo: Tuple[float, float]
    env_mid: Tuple[float, float]
    env_hi: Tuple[float, float]
    env_order: int


@dataclass
class keyconfig:
    min_offset: float
    pitch_self: float
    pitch_semitone: float
    pitch_fifth: float
    pitch_others: float


@dataclass
class config:
    analysisconfig: analysisconfig
    keyconfig: keyconfig
    libconfig: libconfig
    viewconfig: viewconfig
    externalsyncconfig: externalsyncconfig

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
    lcfg = libconfig(libpath="library", logpath="mixlyzer.log", write_log=False)
    acfg = analysisconfig(
        analysis_samp_rate=22050,
        chroma_method="cens",
        chroma_hop_length=512,
        chroma_cqt_bins_per_octave=36,
        chroma_cqt_octaves=6,
        use_hpss=True,
        bpm_hop_length=128,
        bpm_win_length=5000,
        bpm_max=220,
        bpm_min=110,
        bpm_dynamic=True,
        bpm_adaptive_window=True,
        beatgrid_offset_msec=0.0,
        env_frame_ms=4,
        env_lo=(20.0, 200.0),
        env_mid=(200.0, 3000.0),
        env_hi=(3000.0, 11025.0),
        env_order=4,
    )
    kcfg = keyconfig(
        min_offset=0.4,
        pitch_self=0.9,
        pitch_semitone=0.02,
        pitch_fifth=0.001,
        pitch_others=0.01,
    )
    vcfg = viewconfig(
        display_waveform=True,
        display_beatgrid=True,
        display_keystrip=True,
        display_JumpCUE=True,
        fps=60,
        metronome_wav_path="assets/sound/click.wav",
        record_img_path="assets/images/vinyl.png",
        enable_metronome=False,
    )
    xcfg = externalsyncconfig(
        enabled=False,
        mode="time",
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
    return config(
        libconfig=lcfg,
        analysisconfig=acfg,
        keyconfig=kcfg,
        viewconfig=vcfg,
        externalsyncconfig=xcfg,
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

    cfg = default_cfg()
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            loaded = json.load(f)
            merged = _merge_defaults(cfg.to_dict(), loaded if isinstance(loaded, dict) else {})
            cfg = config.from_dict(merged)
    except (TypeError, FileNotFoundError, json.JSONDecodeError):
        with open("config.json", "w", encoding="utf-8") as f:
            json.dump(cfg.to_dict(), f)
    _ensure_library_dir(cfg)
    return cfg
