# AGENTS (Mixlyzer)

A quick brief for AI/automation agents working on the Mixlyzer codebase.

## Run & Env
- Python 3.12+; use uv. Install deps: `uv sync`
- Launch: `uv run python -m app.main`
- Keep root `ffmpeg.exe` (or FFmpeg on PATH) and `config.json` in place.

## Repo Map
- `app/`: Qt entrypoint, `AppWindow`, dark theme, metronome.
- `core/`: Infrastructure/state (`EventBus`, `DataModel`/`GlobalParams`, `TimelineCoordinator`, `PlayerController`, `taskmanager`), settings (`config.py`), audio decode (`audio/decoder.py`), library/feature store (`library_handler.py`, `analysis_lib_handler.py`), reanalysis workers.
- `analyzer_core/`: DSP algorithms. Beatgrid (`beat/beat.py`), key (`key/`), JumpCUE (`self_correlation/JumpCUE.py`), full pipeline (`global_analyzer.py`).
- `ui/`: Dialogs and panels (library, track edit, beatgrid edit, settings, export, etc.).
- `views/`: Graphics views (waveform, beatgrid, keystrip, JumpCUE, overview, etc.).
- `utils/`: Color/font/jump cue label helpers.
- `assets/`: Icons and sounds; `config.json` points here.
- `library/`: Analysis cache (.npz) and `library.db`. User data - do not delete or commit.
- `build/`, `dist/`, `installer/`: PyInstaller spec (`build/mixlyzer.spec`) and Inno Setup (`installer/mixlyzer.iss`).
- `README*.md`: User docs (EN/JA/KR).

## Analysis Flow (key files)
- `app/window.py` -> `AnalysisWorker` (multiprocessing spawn) -> `analyzer_core/global_analyzer.precompute_features`.
- Steps: load audio via `decoder.py` -> HPSS -> band envelopes/RMS -> tempo (`bpm_phase_sync`/dynamic) -> JumpCUE (`JumpCueEngine`) -> chroma+key (`keyanalyzer`) -> `normalize_gui_buffers` -> persist to `FeatureNPZStore`/`LibraryDB` -> load into UI.
- Segment reanalysis lives in `core/segment_reanalysis_manager.py` / `segment_reanalysis_worker.py` and uses the same config dataclasses (`core/config.py`).

## UI Flow
- Signals: Qt signals in `core/event_bus.py` drive play/pause/stop, feature load, window/center changes, settings save, reanalysis requests. Feature/property/albumart/beatgrid/key/jumpcue signals are notification-only (no payload); listeners read from the shared `DataModel`.
- `DataModel` holds features/metadata/album art; `MainPane` and views (`views/`) render them. Playback is handled by `PlayerController`; timeline by `TimelineCoordinator`. Overview and other views pull state directly from the shared model.
- Export dialog prefers the in-memory model for the currently loaded track when available; otherwise falls back to NPZ.

## Coding Guidelines
- Multiprocessing safety: keep worker-callable functions at module scope; do not pass Qt objects across processes.
- Config changes: update `core/config.py` dataclasses/`default_cfg()`, `config.json` schema, and `ui/cfgwindow.py` together.
- Heavy DSP/IO must stay off the GUI thread (AnalysisWorker or reanalysis workers). Use EventBus signals to reflect results in UI.
- Preserve NPZ schema keys (`normalize_gui_buffers`, `DataModel`). When adding features, update adapters and views accordingly.
- Keep NumPy arrays contiguous when possible; preserve type hints.
- Treat `library/` contents as user assets; do not delete/commit. Keep root `ffmpeg.exe` intact for `decoder.py`.

## Build/Packaging
- PyInstaller: `pyinstaller -y build/mixlyzer.spec` (includes assets/config/ffmpeg).
- Windows installer: build PyInstaller output, then run `installer/mixlyzer.iss` with Inno Setup (see `installer/README.md`).
