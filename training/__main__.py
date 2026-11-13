"""The training's command line, for developers (the app retrains nothing): one optimizer run on the library's
annotations, read through the library's owner (core.library.service), its request given the files it reads as the
analysis reads them (the user's model where there is one, else the shipped one: system.model_files); a shipped model
is never an output, an existing output is copied to backups/ first.

    uv run python -m training NAME [--output PATH] [--cache DIR] [--start shipped|current] [--cv-folds N]
                                   [--set FIELD=VALUE ...] [--no-backup]

NAME: one of OPTIMIZERS. Run from the app's folder (config.json's settings: the library, the models in use, the
lowest BPM): the output is the model setting's file unless --output, the feature cache featurecache/<kind> unless
--cache. --set gives any other field of the optimizer's request (a JSON value, else the text): ``--set steps=4000``;
the key model trains on a key GT file instead of the library's key annotations with ``--set key_gt=PATH``
(training/key_model_optimizer.py); the JumpCUE model takes every group of the tracks added since one, or up to one,
as judged with ``--set full_review_from=UID`` / ``full_review_upto=UID`` (training/jumpcue_optimizer.py).
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from core.analysis_library import jobs as analysis_jobs
from core.workers.analysis_settings import model_file
from system.model_files import is_shipped, resolve_model, shipped_path
from system.resource_paths import project_root
from training import library_tracks
from training.model_start import CURRENT, SCRATCH, SHIPPED
from training.text import TrainingError, render


@dataclasses.dataclass(frozen=True)
class Optimizer:
    module: str      # training.<module>
    request: str     # its request type
    optimize: str    # its function: optimize(request, feature_progress=, optimize_progress=) -> result
    setting: str     # the analysisconfig setting naming the model it writes
    cache: str       # its feature cache's folder


BEAT_CACHE = "featurecache/beat"     # the frame features the beat models share
OPTIMIZERS = {
    "refine_onset": Optimizer("refine_onset_optimizer", "RefineOnsetOptimizationRequest", "optimize_refine_onset",
                              "refine_onset_parameter_path", "featurecache/onset"),
    "downbeat_ctx": Optimizer("downbeat_ctx_optimizer", "DownbeatCtxOptimizationRequest", "optimize_downbeat_ctx",
                              "downbeat_ctx_parameter_path", BEAT_CACHE),
    "beat_tcn": Optimizer("beat_tcn_optimizer", "BeatTcnOptimizationRequest", "optimize_beat_tcn",
                          "beat_tcn_parameter_path", BEAT_CACHE),
    "beat_critic": Optimizer("beat_critic_optimizer", "BeatCriticOptimizationRequest", "optimize_beat_critic",
                             "beat_critic_parameter_path", BEAT_CACHE),
    "beat_decoder": Optimizer("beat_decoder_optimizer", "BeatDecoderOptimizationRequest", "optimize_beat_decoder",
                              "beat_decoder_parameter_path", BEAT_CACHE),
    "phrase": Optimizer("phrase_parameter_optimizer", "PhraseOptimizationRequest", "optimize_phrase_parameters",
                        "phrase_parameter_path", "featurecache/phrase"),
    "phrase_stack": Optimizer("phrase_stack_optimizer", "PhraseStackOptimizationRequest", "optimize_phrase_stack",
                              "phrase_parameter_path", "featurecache/phrase"),
    "key_model": Optimizer("key_model_optimizer", "KeyModelOptimizationRequest", "optimize_key_model",
                           "key_parameter_path", "featurecache/key"),
    "jumpcue": Optimizer("jumpcue_optimizer", "JumpCueOptimizationRequest", "optimize_jumpcue",
                         "jumpcue_pairs_parameter_path", "featurecache/jumpcue"),
}
BEAT_OPTIMIZERS = ("refine_onset", "downbeat_ctx", "beat_tcn", "beat_critic", "beat_decoder")
# the networks a run starts from the shipped model, the one in use or nothing (--start; the others train anew)
START_MODELS = ("beat_critic", "beat_tcn", "key_model")
# the analysis's model files in use a beat run is given (the training's ModelSet fields) -> their settings
_CURRENT_MODELS = {"downbeat_ctx": "downbeat_ctx_parameter_path", "tcn": "beat_tcn_parameter_path",
                   "critic": "beat_critic_parameter_path", "decoder": "beat_decoder_parameter_path",
                   "refine_onset": "refine_onset_parameter_path"}


def project_path(text: str | Path) -> Path:
    path = Path(str(text).strip())
    return path if path.is_absolute() else project_root() / path


def check_output(output_path: str | Path) -> None:
    if is_shipped(output_path):
        raise ValueError(f"a shipped model is never overwritten; choose another path than {Path(output_path).name}")


def prepare(name: str, values: dict) -> dict:
    """A run's request with the files the training reads: the output checked (never a shipped model), the model in
    use (``current_path``), the one the training starts from (``start_path``) and every model setting it names
    resolved."""
    values = dict(values)
    output = values.get("output_path")
    if output:
        check_output(output)
    if name == "beat_decoder":            # the costs refitted around the shipped ones or the ones in use
        values["current_path"] = str(resolve_model(output))
        start = values.get("start") or CURRENT
        values["start_path"] = str(shipped_path(output) if start == SHIPPED else resolve_model(output))
        for key, setting in (("tcn_path", "beat_tcn_parameter_path"), ("critic_path", "beat_critic_parameter_path"),
                             ("downbeat_ctx_path", "downbeat_ctx_parameter_path"),
                             ("refine_onset_path", "refine_onset_parameter_path")):
            values[key] = model_file(values.get(key), setting)
    if name in START_MODELS:              # compared with the model in use; started from the shipped, it or none
        start = values.get("start") or CURRENT
        in_use = resolve_model(values.get("current_path") or output)
        values["current_path"] = str(in_use)
        values["start_path"] = None if start == SCRATCH else str(shipped_path(in_use) if start == SHIPPED else in_use)
    if name == "beat_decoder" and values.get("start") == SCRATCH:
        raise ValueError("the decoder's costs are refitted around the shipped ones or the ones in use "
                         "(--start shipped or current)")
    if values.get("current_models") is not None:
        given = dict(values["current_models"])
        values["current_models"] = {key: model_file(given.get(key), setting) for key, setting in _CURRENT_MODELS.items()}
    return values


def request_values(name: str, analysis, libpath: str, args) -> dict:
    """The request's fields as the settings give them (``analysis``: config.json's analysisconfig), then ``args``'."""
    from analyzer.beat.frame_features import FRAME_HOP_LENGTH
    spec = OPTIMIZERS[name]
    values = {"library_dir": project_path(libpath), "cache_dir": project_path(args.cache or spec.cache),
              "output_path": project_path(args.output or getattr(analysis, spec.setting))}
    if args.start:
        values["start"] = args.start
    if args.cv_folds is not None:
        values["cv_folds"] = args.cv_folds
    if name in BEAT_OPTIMIZERS:
        values["bpm_min"] = float(analysis.bpm_min)
        models = {key: project_path(getattr(analysis, setting)) for key, setting in _CURRENT_MODELS.items()}
        if name == "beat_decoder":          # the costs are fitted for the models in use
            values.update(tcn_path=models["tcn"], critic_path=models["critic"],
                          downbeat_ctx_path=models["downbeat_ctx"], refine_onset_path=models["refine_onset"])
        else:                               # the end-to-end check runs the analysis with them, the new model in
            values["current_models"] = models
        if name == "refine_onset":
            values.update(frame_hop_length=int(FRAME_HOP_LENGTH), frames_cache_dir=project_path(BEAT_CACHE))
    if name in START_MODELS:              # the model in use: the setting's file, whatever the output
        values["current_path"] = project_path(getattr(analysis, spec.setting))
    for item in args.set or ():
        key, _, text = item.partition("=")
        try:
            values[key.strip()] = json.loads(text)
        except ValueError:
            values[key.strip()] = text
    if name == "key_model" and values.get("key_gt"):     # a key GT file, from the app's folder unless absolute
        values["key_gt"] = str(project_path(values["key_gt"]))
    return values


def read_settings():
    """config.json's settings: the training's command lines' one read (this one, training.key_chords), as the app's
    start reads them."""
    from core.settings.config import load_cfg
    return load_cfg()


def read_tracks(libpath: str) -> list[dict]:
    """The library's tracks with their annotations, read by its owner (``TRAINING_BATCH`` tracks a job), each with the
    song it is a version of (``song``: its Variants group's first uid, "" for a track with no other version)."""
    from core.library import jobs as library_jobs
    from core.library.service import LibraryService
    if not (Path(libpath) / "library.db").is_file():
        raise ValueError(f"no library at {libpath}")
    library = LibraryService(libpath)

    def read() -> list[dict]:
        tracks = library.call(analysis_jobs.training_tracks)
        song = {uid: group[0] for group in library.call(library_jobs.song_versions) for uid in group}
        for track in tracks:
            track["song"] = song.get(track["uid"], "")
        for start in range(0, len(tracks), analysis_jobs.TRAINING_BATCH):
            batch = tracks[start:start + analysis_jobs.TRAINING_BATCH]
            read = library.call(analysis_jobs.training_annotations, [t["uid"] for t in batch])
            for track, (annotation, error) in zip(batch, read):
                track["annotation"] = annotation
                if error:
                    track["error"] = error
            print(f"\rreading the stored analyses: {start + len(batch)} / {len(tracks)} tracks", end="", flush=True)
        print()
        return tracks
    try:
        with ThreadPoolExecutor(1) as pool:       # LibraryService.call waits: never on the main thread
            return pool.submit(read).result()
    finally:
        library.close()


def backup(path: Path) -> Path:
    """``path`` copied to backups/<stem>.bak_<time><suffix>."""
    folder = project_root() / "backups"
    folder.mkdir(parents=True, exist_ok=True)
    copy = folder / f"{path.stem}.bak_{datetime.now():%Y%m%d_%H%M%S}{path.suffix}"
    shutil.copy2(path, copy)
    return copy


def progress(stage: str):
    last = [None]

    def report(value: int, text) -> None:
        line = f"[{stage} {int(value):3d}%] {render(text)}".replace("\n", " | ")
        if line != last[0]:
            last[0] = line
            print(line, flush=True)
    return report


def show(value) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    if dataclasses.is_dataclass(value):
        return "{" + ", ".join(f"{f.name}: {show(getattr(value, f.name))}" for f in dataclasses.fields(value)) + "}"
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):      # a report's measures, each shown
        return "{" + ", ".join(f"{k}: {show(v)}" for k, v in value.items()) + "}"
    text = repr(value)
    return text if len(text) <= 200 else text[:200] + " ..."


def report(result) -> None:
    if not dataclasses.is_dataclass(result):
        print(result)
        return
    for f in dataclasses.fields(result):
        if f.name != "skipped_tracks":
            print(f"{f.name}: {show(getattr(result, f.name))}")
    skipped = tuple(getattr(result, "skipped_tracks", ()) or ())
    if skipped:
        print(f"skipped tracks ({len(skipped)}):")
        for track in skipped:
            name = render(track.display_name).replace("\n", " / ")
            print(f"  {track.uid}  {name}: {render(track.reason)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m training", description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name", choices=sorted(OPTIMIZERS))
    parser.add_argument("--output", help="the model file written (default: the model setting's file)")
    parser.add_argument("--cache", help="the feature cache's folder (default: featurecache/<kind>)")
    parser.add_argument("--start", choices=(SHIPPED, CURRENT, SCRATCH),
                        help="what a network starts from (beat_critic, key_model): the shipped model, the one in use, "
                             "or nothing; the decoder's costs are refitted around the shipped or the current ones")
    parser.add_argument("--cv-folds", type=int, help=">= 2: a track-level cross-validation before the final fit")
    parser.add_argument("--set", action="append", metavar="FIELD=VALUE", help="another field of the request")
    parser.add_argument("--no-backup", action="store_true", help="overwrite an existing output without a copy")
    args = parser.parse_args(argv)

    cfg = read_settings()                # the CLI's one read of the settings, as the app's start
    spec = OPTIMIZERS[args.name]
    module = importlib.import_module(f"training.{spec.module}")
    request_type, optimize = getattr(module, spec.request), getattr(module, spec.optimize)
    try:
        values = prepare(args.name, request_values(args.name, cfg.analysisconfig, cfg.libconfig.libpath, args))
        fields = {f.name for f in dataclasses.fields(request_type)}
        unknown = sorted(set(values) - fields)
        if unknown:
            raise ValueError(f"{spec.request} has no field {', '.join(unknown)} (its fields: {', '.join(sorted(fields))})")
        output = Path(values["output_path"])
        library_tracks.use(read_tracks(project_path(cfg.libconfig.libpath).as_posix()))
    except ValueError as exc:             # a request refused before the run
        print(exc, file=sys.stderr)
        return 2
    if output.is_file() and not args.no_backup:
        print(f"the existing {output.name} copied to {backup(output)}")
    try:
        result = optimize(request_type(**values), feature_progress=progress("features"),
                          optimize_progress=progress("training"))
    except TrainingError as exc:          # a refusal (too few songs): its words
        print(render(exc.text), file=sys.stderr)
        return 2
    report(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
