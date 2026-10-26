"""Reoptimize's model files and words on the app's side: which file each training reads and writes (the user's
retrained model where there is one, else the shipped one: foundation.model_files), the shipped model never written,
the warnings the dialogs show, and the training's texts in the UI's language (analyzer.training.text)."""

from __future__ import annotations

from pathlib import Path

from analyzer.training.model_start import (
    BEAT_SONGS, CURRENT, KEY_SONGS, MIN_TRACKS, SCRATCH, SHIPPED, SHIPPED_TRAINING_TRACKS, kind_of,
)
from analyzer.training.text import render
from core.workers.analysis_settings import model_file
from foundation.i18n import tr
from foundation.model_files import has_user_model, is_shipped, resolve_model, shipped_path

__all__ = ["BEAT_SONGS", "CURRENT", "KEY_SONGS", "SCRATCH", "SHIPPED", "base_model", "check_output", "kind_of",
           "prepare", "warnings", "words"]

# the analysis's model files in use a beat run is given (the training's ModelSet fields) -> their settings
_CURRENT_MODELS = {"downbeat_ctx": "downbeat_ctx_parameter_path", "tcn": "beat_tcn_parameter_path",
                   "critic": "beat_critic_parameter_path", "decoder": "beat_decoder_parameter_path",
                   "refine_onset": "refine_onset_parameter_path"}


def words(text) -> str:
    """The training's text (analyzer.training.text) in the UI's language."""
    return render(text, tr)


def base_model(output_path: str | Path, start: str) -> Path | None:
    """The model a training starts from (None: from scratch)."""
    if start == SCRATCH:
        return None
    return shipped_path(output_path) if start == SHIPPED else resolve_model(output_path)


def check_output(output_path: str | Path) -> None:
    if is_shipped(output_path):
        raise ValueError(tr("A shipped model is never overwritten; choose another path than {name}")
                         .format(name=Path(output_path).name))


def prepare(optimizer_name: str, values: dict) -> dict:
    """A run's request with the files the training reads: the output checked (never a shipped model), the model in
    use (``current_path``), the one the training starts from (``start_path``, None from scratch) and every model
    setting it names resolved."""
    values = dict(values)
    output = values.get("output_path")
    if output:
        check_output(output)
    start = values.get("start") or CURRENT
    if optimizer_name in ("beat_tcn", "beat_critic", "key_model", "beat_decoder", "jumpcue"):
        values["current_path"] = str(resolve_model(output))
    if optimizer_name in ("beat_tcn", "beat_critic", "key_model"):
        given = values.get("base_path")
        start_path = resolve_model(given) if given else base_model(output, start)
        values["start_path"] = None if start_path is None else str(start_path)
    if optimizer_name == "beat_decoder":
        values["start_path"] = str(base_model(output, SHIPPED if start == SHIPPED else CURRENT))
        for key, setting in (("tcn_path", "beat_tcn_parameter_path"), ("critic_path", "beat_critic_parameter_path"),
                             ("downbeat_ctx_path", "downbeat_ctx_parameter_path"),
                             ("refine_onset_path", "refine_onset_parameter_path")):
            values[key] = model_file(values.get(key), setting)
    if optimizer_name == "beat_critic":
        values["tcn_path"] = model_file(values.get("tcn_path"), "beat_tcn_parameter_path")
    if values.get("current_models") is not None:
        given = dict(values["current_models"])
        values["current_models"] = {key: model_file(given.get(key), setting) for key, setting in _CURRENT_MODELS.items()}
    if optimizer_name == "jumpcue":
        values["phrase_model_file"] = model_file(values.get("phrase_parameter_path"), "phrase_parameter_path")
        values["jump_point_model_path"] = model_file(None, "jumpcue_parameter_path")
    return values


def warnings(count: int, kind: str, start: str | None, output_path: str | Path | None,
             songs=BEAT_SONGS, scratch_takes_hours: bool = True) -> tuple[list[str], bool]:
    """(messages, blocked) for a planned training on ``count`` songs (``scratch_takes_hours``: a training from
    scratch takes one to two hours, said in its warning)."""
    blocked_below, warn_below = MIN_TRACKS[kind]
    out = []
    if count < blocked_below:
        message = (tr("Training from scratch needs at least {needed} {songs}; the library has {count}.")
                   if kind == "scratch" else tr("This needs at least {needed} {songs}; the library has {count}."))
        return [message.format(needed=blocked_below, songs=words(songs), count=count)], True
    if count < warn_below:
        if kind == "scratch":
            out.append(tr("{count} songs: fewer than the shipped model was trained on (about {shipped}); "
                          "a model trained from scratch on them is likely to be worse than the shipped one.")
                       .format(count=count, shipped=SHIPPED_TRAINING_TRACKS))
        else:
            out.append(tr("{count} songs: risk of overfitting to them, and the cross-validation check rests on few "
                          "songs.").format(count=count))
    if start == SCRATCH:
        out.append(tr("Training from scratch takes one to two hours and discards what the shipped model learned.")
                   if scratch_takes_hours else tr("Training from scratch discards what the shipped model learned."))
    elif start == CURRENT and output_path is not None and has_user_model(output_path):
        out.append(tr("The current model is already retrained: fine-tuning it again accumulates changes. "
                      "Fine-tune the shipped model to start over."))
    elif start == SHIPPED and output_path is not None and has_user_model(output_path):
        out.append(tr("Fine-tuning the shipped model replaces the earlier retrained model."))
    return out, False
