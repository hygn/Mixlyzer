"""Where a retrained model starts, how many songs each kind of training needs, and what to warn about.

- shipped: improve the shipped model (``default_<name>``) on the library -- earlier retrainings are discarded;
- current: improve the model in use -- repeated retrainings build on each other;
- scratch: train a new model on the library alone -- needs at least as many songs as a model of its size can
  learn from (the shipped beat models were trained on about 340 songs).
The gradient-boosted models (the downbeat model, the tempo-change onset) are always trained anew (minutes).
"""

from __future__ import annotations

from pathlib import Path

from core.i18n import tr, tr_mark
from core.model_files import has_user_model, is_shipped, resolve_model, shipped_path

SHIPPED, CURRENT, SCRATCH = "shipped", "current", "scratch"
SHIPPED_TRAINING_TRACKS = 340
# (blocked below, warned below) songs with beat grids
MIN_TRACKS = {"fine_tune": (20, 60), "scratch": (200, SHIPPED_TRAINING_TRACKS), "gbm": (50, 150), "search": (20, 60)}
# what the songs counted are (translated where shown)
BEAT_SONGS = tr_mark("songs with beat grids")
KEY_SONGS = tr_mark("songs with checked key annotations")


def kind_of(start: str) -> str:
    return "scratch" if start == SCRATCH else "fine_tune"


def base_model(output_path: str | Path, start: str) -> Path | None:
    """The model a training starts from (None: from scratch)."""
    if start == SCRATCH:
        return None
    return shipped_path(output_path) if start == SHIPPED else resolve_model(output_path)


def check_output(output_path: str | Path) -> None:
    if is_shipped(output_path):
        raise ValueError(tr("A shipped model is never overwritten; choose another path than {name}")
                         .format(name=Path(output_path).name))


def require_tracks(count: int, kind: str, songs: str = BEAT_SONGS) -> None:
    blocked, _ = MIN_TRACKS[kind]
    if count < blocked:
        message = {"scratch": tr("Training from scratch needs at least {needed} {songs} (the library has {count})"),
                   "fine_tune": tr("Retraining needs at least {needed} {songs} (the library has {count})"),
                   "gbm": tr("Retraining needs at least {needed} {songs} (the library has {count})"),
                   "search": tr("Refitting needs at least {needed} {songs} (the library has {count})")}[kind]
        raise RuntimeError(message.format(needed=blocked, songs=tr(songs), count=count))


def warnings(count: int, kind: str, start: str | None, output_path: str | Path | None,
             songs: str = BEAT_SONGS, scratch_takes_hours: bool = True) -> tuple[list[str], bool]:
    """(messages, blocked) for a planned training on ``count`` songs (``scratch_takes_hours``: a training from
    scratch takes one to two hours, said in its warning)."""
    blocked_below, warn_below = MIN_TRACKS[kind]
    out = []
    if count < blocked_below:
        message = (tr("Training from scratch needs at least {needed} {songs}; the library has {count}.")
                   if kind == "scratch" else tr("This needs at least {needed} {songs}; the library has {count}."))
        return [message.format(needed=blocked_below, songs=tr(songs), count=count)], True
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
