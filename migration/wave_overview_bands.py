"""One-off library migration: the overview waveform's new format (utils/wave.py: the min / max of 32768 columns and
the power of 12 log-spaced bands in 8192 columns) for the tracks analyzed before it, measured again from their audio
(decoded at 48 kHz, as the analysis does). The detail waveform's three-band scale is taken over from the old format
unchanged, so the detail waveform draws exactly as before. Nothing else in a file changes; each changed NPZ is
backed up first and replaced atomically after it reads back. A track whose audio is missing keeps the old format
(still drawn). Run by hand (the app never migrates by itself):

    uv run python -m migration.wave_overview_bands LIBRARY_DIR            # dry run: what would change
    uv run python -m migration.wave_overview_bands LIBRARY_DIR --apply [--workers N]
"""

from __future__ import annotations

import argparse
from concurrent.futures import as_completed
from datetime import datetime
import os
from pathlib import Path
import shutil
import sqlite3

import numpy as np

OLD_KEYS = ("wave_overview_np", "wave_overview_ref")


def _measure(job: tuple[str, str, list[float]]) -> tuple[str, dict[str, np.ndarray] | None, str]:
    """Process-worker entry point: the new overview arrays of one track from its audio."""
    uid, audio_path, detail_ref = job
    try:
        from threadpoolctl import threadpool_limits
        from analyzer_core.global_analyzer import fast_load
        from analyzer_core.hpss import HPSS_HOP_LENGTH, HPSS_N_FFT
        from utils.wave import WAVE_OVERVIEW_SAMPLE_RATE, build_wave_overview

        with threadpool_limits(limits=1):
            y = fast_load(audio_path, WAVE_OVERVIEW_SAMPLE_RATE)
            # the ranges only set the detail scale, which is taken over below
            arrays = build_wave_overview(y, WAVE_OVERVIEW_SAMPLE_RATE, ((20.0, 200.0), (250.0, 1500.0), (1600.0, 11025.0)),
                                         n_fft=HPSS_N_FFT, hop_length=HPSS_HOP_LENGTH)
        arrays["wave_detail_band_ref"] = np.asarray(detail_ref, np.float32)
        return uid, arrays, ""
    except Exception as exc:  # reported, the track keeps its old format
        return uid, None, repr(exc)


def _plan(library_dir: Path) -> tuple[list[tuple[str, str, list[float]]], list[str]]:
    con = sqlite3.connect(f"file:{(library_dir / 'library.db').as_posix()}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT uid, path FROM tracks").fetchall()
    finally:
        con.close()
    jobs, notes = [], []
    for uid, audio_path in rows:
        npz = library_dir / f"{uid}.npz"
        if not uid or not npz.is_file():
            continue
        with np.load(npz, allow_pickle=False) as z:
            if "wave_minmax_np" in z.files or "wave_overview_ref" not in z.files:
                continue
            detail_ref = np.asarray(z["wave_overview_ref"], np.float64)[2:5].tolist()
        if not audio_path or not Path(audio_path).is_file():
            notes.append(f"{uid}: audio file missing, kept in the old format")
            continue
        jobs.append((str(uid), str(audio_path), detail_ref))
    return jobs, notes


def _write(npz: Path, arrays: dict[str, np.ndarray], backup_dir: Path) -> None:
    with np.load(npz, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files if k not in OLD_KEYS}
    data.update(arrays)
    shutil.copy2(npz, backup_dir / npz.name)
    tmp = npz.with_name(npz.stem + ".migrating.npz")
    np.savez_compressed(tmp, **data)
    with np.load(tmp, allow_pickle=False) as z:
        if set(z.files) != set(data) or not all(np.array_equal(z[k], data[k]) for k in data):
            raise RuntimeError(f"{npz.name}: the written file does not read back")
    os.replace(tmp, npz)


def migrate(library_dir: str | Path, *, apply: bool = False, workers: int | None = None) -> dict[str, object]:
    from core.concurrency.process_pool import create_spawn_process_pool

    library_dir = Path(library_dir).resolve()
    jobs, notes = _plan(library_dir)
    report: dict[str, object] = {"to_migrate": len(jobs), "notes": notes, "migrated": 0, "failed": []}
    if not apply or not jobs:
        return report
    backup_dir = library_dir / "backups" / f"wave_overview_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    report["backup_dir"] = str(backup_dir)
    n_workers = max(1, min(int(workers or max(1, (os.cpu_count() or 4) // 2)), len(jobs)))
    with create_spawn_process_pool(n_workers) as executor:
        futures = [executor.submit(_measure, job) for job in jobs]
        for done, future in enumerate(as_completed(futures), start=1):
            uid, arrays, error = future.result()
            if arrays is None:
                report["failed"].append(f"{uid}: {error}")
            else:
                _write(library_dir / f"{uid}.npz", arrays, backup_dir)
                report["migrated"] += 1
            if done % 20 == 0 or done == len(jobs):
                print(f"{done}/{len(jobs)} done ({report['migrated']} written, {len(report['failed'])} failed)", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("library_dir")
    parser.add_argument("--apply", action="store_true", help="write (default: a dry run)")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    result = migrate(args.library_dir, apply=args.apply, workers=args.workers)
    for note in result["notes"]:
        print("note:", note)
    for failure in result["failed"]:
        print("failed:", failure)
    print({k: v for k, v in result.items() if k not in ("notes", "failed")})
