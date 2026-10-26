"""The analyzer's settings from the app's (config.json): what an analysis is asked for, each model the file it reads
(the user's retrained file where there is one, else the shipped one: foundation.model_files)."""

from __future__ import annotations

from functools import lru_cache

from analyzer.settings import AnalysisSettings, CueRule, ModelPaths
from core.config import config, default_cfg
from foundation.model_files import resolve_model


@lru_cache(maxsize=1)
def _defaults():
    return default_cfg().analysisconfig


def model_file(path: str | None, setting: str) -> str:
    """The file read for a model setting (an analysisconfig ``*_parameter_path`` field) set to ``path`` (empty: the
    setting's default)."""
    text = str(path or "").strip() or getattr(_defaults(), setting)
    return str(resolve_model(text))


def model_paths(cfg: config) -> ModelPaths:
    a = cfg.analysisconfig
    return ModelPaths(
        beat_tcn=model_file(a.beat_tcn_parameter_path, "beat_tcn_parameter_path"),
        beat_critic=model_file(a.beat_critic_parameter_path, "beat_critic_parameter_path"),
        beat_decoder=model_file(a.beat_decoder_parameter_path, "beat_decoder_parameter_path"),
        downbeat_ctx=model_file(a.downbeat_ctx_parameter_path, "downbeat_ctx_parameter_path"),
        refine_onset=model_file(a.refine_onset_parameter_path, "refine_onset_parameter_path"),
        phrase=model_file(a.phrase_parameter_path, "phrase_parameter_path"),
        jumpcue=model_file(a.jumpcue_parameter_path, "jumpcue_parameter_path"),
        key=model_file(a.key_parameter_path, "key_parameter_path"),
    )


def cue_rules(cfg: config) -> dict[str, CueRule]:
    """Where the analysis sets CUE points, by phrase label (Settings > Analysis > CUE)."""
    return {label: CueRule(bool(rule.at_start), bool(rule.at_end), bool(rule.same_label_boundaries))
            for label, rule in cfg.cuepointconfig.by_label().items()}


def analysis_settings(cfg: config) -> AnalysisSettings:
    a, view = cfg.analysisconfig, cfg.viewconfig
    return AnalysisSettings(
        models=model_paths(cfg),
        bpm_min=float(a.bpm_min),
        wave_bands=tuple((float(low), float(high)) for low, high in (view.env_lo, view.env_mid, view.env_hi)),
        bpm_dynamic=bool(a.bpm_dynamic),
        dynamic_downbeat=bool(a.dynamic_downbeat),
        key_dynamic=bool(a.key_dynamic),
        autocue=a.autocue,
        cue_rules=cue_rules(cfg),
        music_index=bool(cfg.searchconfig.music_index),
    )
