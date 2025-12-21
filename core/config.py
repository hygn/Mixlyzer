from dataclasses import dataclass
from typing import Literal, Tuple, Type
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing_extensions import TypeForm

@dataclass
class libconfig:
    libpath: str

@dataclass
class viewconfig:
    display_waveform: bool
    display_beatgrid: bool
    display_keystrip: bool
    display_JumpCUE: bool
    enable_metronome: bool
    record_img_path: str
    metronome_wav_path: str

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
    bpm_adaptive_window:bool

    env_frame_ms: int
    env_lo: Tuple[float, float]
    env_mid: Tuple[float, float]
    env_hi: Tuple[float, float]
    env_order: int

@dataclass
class keyconfig:
    min_offset: float
    # Transitions (pitch class)
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

    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(self, d: dict):
        """dict → config dataclass"""
        return self(
            analysisconfig=analysisconfig(**d["analysisconfig"]),
            keyconfig=keyconfig(**d["keyconfig"]),
            libconfig=libconfig(**d["libconfig"]),
            viewconfig=viewconfig(**d["viewconfig"]),
        )

def default_cfg():
    lcfg = libconfig(libpath="library")
    acfg = analysisconfig(analysis_samp_rate=22050,
                          chroma_method="cens",
                          chroma_hop_length=512,
                          chroma_cqt_bins_per_octave=36,
                          chroma_cqt_octaves=6,
                          use_hpss=True,
                          bpm_hop_length=128,
                          bpm_win_length=5000,
                          bpm_max=240,
                          bpm_min=110,
                          bpm_dynamic=True,
                          bpm_adaptive_window=True,
                          env_frame_ms=4,
                          env_lo=(20.0, 200.0),
                          env_mid=(200.0, 3000.0),
                          env_hi=(3000.0, 11025.0),
                          env_order=4)
    kcfg = keyconfig(min_offset=0.4,
                     pitch_self=0.9,
                     pitch_semitone=0.02,
                     pitch_fifth=0.001,
                     pitch_others=0.01)
    vcfg = viewconfig(display_waveform=True,
                      display_beatgrid=True,
                      display_keystrip=True,
                      display_JumpCUE=True,
                      metronome_wav_path="assets/sound/click.wav",
                      record_img_path="assets/images/vinyl.png",
                      enable_metronome=False)
    return config(libconfig=lcfg, analysisconfig=acfg, keyconfig=kcfg, viewconfig=vcfg)

def load_cfg() -> config:
    def _merge_defaults(default_dict: dict, loaded_dict: dict) -> dict:
        """Recursively merge loaded values onto defaults, preserving defaults when missing."""
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
