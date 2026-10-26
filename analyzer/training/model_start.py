"""Where a retrained model starts and how many songs each kind of training needs.

- shipped: improve the shipped model on the library -- earlier retrainings are discarded;
- current: improve the model in use -- repeated retrainings build on each other;
- scratch: train a new model on the library alone -- needs at least as many songs as a model of its size can
  learn from (the shipped beat models were trained on about 340 songs).
The gradient-boosted models (the downbeat model, the tempo-change onset) are always trained anew (minutes). Which
file each start reads is the app's (core.workers.training_models: it gives the request ``current_path``, the model
in use, and ``start_path``, the model the training starts from, None from scratch).
"""

from __future__ import annotations

from analyzer.training.text import TrainingError, msg

SHIPPED, CURRENT, SCRATCH = "shipped", "current", "scratch"
SHIPPED_TRAINING_TRACKS = 340
# (blocked below, warned below) songs with beat grids
MIN_TRACKS = {"fine_tune": (20, 60), "scratch": (200, SHIPPED_TRAINING_TRACKS), "gbm": (50, 150), "search": (20, 60)}
# what the songs counted are
BEAT_SONGS = msg("songs with beat grids")
KEY_SONGS = msg("songs with checked key annotations")


def kind_of(start: str) -> str:
    return "scratch" if start == SCRATCH else "fine_tune"


def require_tracks(count: int, kind: str, songs=BEAT_SONGS) -> None:
    blocked, _ = MIN_TRACKS[kind]
    if count < blocked:
        values = {"needed": blocked, "songs": songs, "count": count}
        raise TrainingError({
            "scratch": msg("Training from scratch needs at least {needed} {songs} (the library has {count})", **values),
            "fine_tune": msg("Retraining needs at least {needed} {songs} (the library has {count})", **values),
            "gbm": msg("Retraining needs at least {needed} {songs} (the library has {count})", **values),
            "search": msg("Refitting needs at least {needed} {songs} (the library has {count})", **values),
        }[kind])
