"""The analyzer's settings from the app's (config.json): what an analysis is asked for, each model the file it reads
(the user's retrained file where there is one, else the shipped one: system.model_files)."""

from __future__ import annotations

import json
from functools import lru_cache

from analyzer.settings import AnalysisProfile, AnalysisSettings, CueRule, JumpCueRules, ModelPaths
from core.settings.config import config, default_cfg
from system.model_files import resolve_model
from system.resource_paths import resource_path

PROFILES_FILE = "assets/analysis/profiles.json"
CQT_GAINS_FILE = "assets/analysis/stft_cqt_gains.npz"


@lru_cache(maxsize=4)
def analysis_profile(name: str) -> AnalysisProfile:
    """An analysis profile (``full``, ``balanced``, ``fast``) as the shipped PROFILES_FILE sets it (a name it lacks:
    Full)."""
    with open(resource_path(PROFILES_FILE), encoding="utf-8") as f:
        p = json.load(f).get(str(name)) or {}
    return AnalysisProfile(
        beat_trainings=max(0, int(p.get("beat_trainings", 0))),
        metronome_critic=bool(p.get("metronome_critic", True)),
        key_tuning_estimate=bool(p.get("key_tuning_estimate", True)),
        cqt_gains=str(resource_path(CQT_GAINS_FILE)) if p.get("approximate_cqt") else "",
    )


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
        jumpcue=model_file(a.jumpcue_pairs_parameter_path, "jumpcue_pairs_parameter_path"),
        key=model_file(a.key_parameter_path, "key_parameter_path"),
    )


def cue_rules(cfg: config) -> dict[str, CueRule]:
    """Where the analysis sets CUE points, by phrase label (Settings > Analysis > CUE)."""
    return {label: CueRule(bool(rule.at_start), bool(rule.at_end), bool(rule.same_label_boundaries))
            for label, rule in cfg.cuepointconfig.by_label().items()}


def analysis_settings(cfg: config) -> AnalysisSettings:
    a = cfg.analysisconfig
    return AnalysisSettings(
        models=model_paths(cfg),
        bpm_min=float(a.bpm_min),
        bpm_dynamic=bool(a.bpm_dynamic),
        dynamic_downbeat=bool(a.dynamic_downbeat),
        key_dynamic=bool(a.key_dynamic),
        autocue=a.autocue,
        cue_rules=cue_rules(cfg),
        jumpcue_rules=JumpCueRules(earlier=a.jumpcue_earlier, later=a.jumpcue_later,
                                   section_margin_beats=max(0, int(a.jumpcue_section_margin_beats)),
                                   min_score=float(a.jumpcue_min_score),
                                   min_gap_beats=int(a.jumpcue_min_gap_beats),
                                   first_sung_beats=max(0, int(a.jumpcue_first_sung_beats)),
                                   avoid_cue_points=bool(a.jumpcue_avoid_cue_points)),
        music_index=bool(cfg.searchconfig.music_index),
        profile=analysis_profile(a.analysis_profile),
    )
