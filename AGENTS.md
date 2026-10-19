# AGENTS (Mixlyzer)

Working notes for AI/automation agents editing the Mixlyzer codebase.

## Run & Environment
- Python 3.12+.
- Dependency management uses `uv`.
- Install deps with `uv sync`.
- Launch the app with `uv run python -m app.main`.
- Keep root-level `config.json` available. `load_cfg()` reads and rewrites it.
- Keep root-level `ffmpeg.exe` available on Windows, or provide FFmpeg on `PATH` for `core/audio/decoder.py`.
- Do not run the app as administrator on Windows; `app/main.py` explicitly blocks it.

## Repo Map
- `app/`: Qt entrypoint and top-level window wiring.
  - `main.py`: app bootstrap, dark palette, font loading, library-version migration flow.
  - `window.py`: `AppWindow`, workflow wiring, EventBus wiring, External Sync, metronome, model updates.
- `core/`: shared runtime/state/integration layer.
  - `config.py`: dataclass config schema, defaults, `load_cfg()`.
  - `model.py`: shared `DataModel` and `GlobalParams`.
  - `event_bus.py`: Qt signal hub.
  - `workers/`: analysis, segment reanalysis, optimizer, and Rekordbox worker implementations.
  - `workflows/`: Qt-facing worker lifecycle and queue orchestration.
  - `library_handler.py`: SQLite metadata store plus BPM/key segment tables and transition search.
  - `analysis_lib_handler.py`: NPZ feature store keyed by track UID.
  - `external_sync.py`: memory-based external deck sync / transport lockout.
  - `player.py`, `timeline.py`, `taskmanager.py`: playback, viewport timeline, worker/task UI state.
  - `tempo_rows.py`: tempo-segment rows and the beats they give (NumPy only; `rebuild_grid_from_segments`, ramp helpers), shared by the analyzer, editor, views and exporters.
  - `audio/`: decode + low-level audio helpers.
- `analyzer_core/`: DSP and editor-side reanalysis algorithms.
  - `global_analyzer.py`: full analysis pipeline and persistence handoff.
  - `beat/`: BPM / beatgrid analysis.
    - `general/`: the full analysis's beat tracker (NumPy only; shipped weights pinned as `assets/weights/default_beat_tcn.npz` (the three trainings in one file), `default_beat_critic.npz`, `default_downbeat_ctx_hgb.npz`, `default_beat_decoder.json`, `default_refine_onset_hgb.npz`; the settings name the user's files (same names without `default_`), and a missing user file means the shipped one (`core/model_files.py`); Reoptimize never writes a `default_` file): `tcn.py` frame TCN, `decode.py` section decoding, `gradual.py` ramps and drifting sections, `critic.py` metronome critic, `junctions.py` junction rules, `bars.py` bar check / bars / production rows, `tracker.py` the pipeline and the consensus of the three trainings.
    - The editor's segment reanalysis (`analyzer_core/editor/beatgrid.py`) uses the same models (the three TCN trainings averaged) on the segment plus 4 s of context: the steady grid with the most evidence (with a reference BPM: within 20 % of it), or a linear ramp when it explains the onsets better by one beat of evidence and moves a beat audibly (`gradual.ramp_or_steady`; guesses from the neighbours' tempos and the segment's halves), then static bars for its downbeat. The full analysis uses only `refine_boundaries` / `hold_slow_tail` from `grid_refine.py` and the refine onset from `learned_onset.py`; `beat.py` keeps grid helpers the editor uses. The rule-based tracker in `beat.py`, the candidate selector, beat phase, meter sections, downbeat offsets (except the feature matrix the bars use), the rest of `grid_refine.py` and their optimizers are unused (removal pending). Settings > Analysis > Beat: BPM min (the octave), Dynamic Analysis (off: one steady grid for the whole track) and Dynamic Downbeat (static analysis only: odd bars / meter changes, off = one bar length and phase; always on with dynamic analysis), and in Advanced the model paths (tempo-change onset, beat TCN, metronome critic, downbeat model, decoder costs `assets/weights/beat_decoder.json`: a new section / a same-tempo half-beat reset / an odd bar, in evidence units of the activation) with their feature caches; Reoptimize Beat Models retrains any of them on the library's grids (`optimizer/`): the decoder costs (`beat_decoder_optimizer.py`: every song tracked with each candidate combination, the costs with the least wrong time -- unusable beats + missed bars -- kept; refit them after the beat model changes, about an hour), the downbeat GBM (sklearn, minutes), the TCN and the critic fine-tuned from the current model in NumPy (`beat_tcn_optimizer.py`, `beat_critic_optimizer.py`: their own forward / backward passes, trainings in parallel processes on a memory-mapped copy of the frames; tens of minutes), all on the tracker's own frames cached by `optimizer/beat_frames.py` (22050 Hz, HPSS, hop 256, every FrameFeatures field in float16). The networks start from the current model, the shipped one, or from scratch, and the decoder costs are refitted around the current or the shipped costs (Quick: the half-beat reset and odd-bar costs only); `optimizer/model_start.py` holds the minimum song counts (blocked below 200 songs from scratch, 20 for fine-tuning, 50 for the GBMs) and the warnings the dialog shows. With CV a group of songs is held out first and the current and new models are compared on it. There is no beatgrid offset: grids are stored as the tracker gives them.
  - `key/`: chroma/key analysis.
  - `self_correlation/`: JumpCUE detection.
  - `editor/`: beatgrid, key, JumpCUE reanalysis helpers used by worker subprocesses.
- `ui/`: Qt widgets/dialogs.
  - `pane.py`: `MainPane` composition.
  - `library.py`: library table, search, transition search, export/reanalyze/edit actions.
  - `edit_panel/`: track editor panel (`panel.py` + per-section mixins: beatgrid, key, phrase, jumpcue, CUE points, beat counter, edit history, save worker; key and phrase are edited by the segment under the playhead; `cue_slots.py` is the CUE page's 4 x 2 slot list).
  - `theme/`: app look on every platform: Fusion + palette + the app-wide style sheet `assets/styles/app.qss` with the design tokens (`tokens.py`) filled in (`qss.py`); colours for new UI come from `current_theme()`.
  - `edit_song.py`, `export.py`, `config/`, `workers.py`: editing/export/settings/task dialogs.
- `views/`: pyqtgraph view plugins registered through `views.base.REGISTRY`.
- `utils/`: labels, colors, fonts, JumpCUE extraction/render helpers, keystrip utilities, waveform helpers.
- `third_party/`: export adapters such as Rekordbox XML generation.
- `migration/`: read-only library-version compatibility adapter, plus one-off development migrations run by hand (`phrase_index_v040.py`, `tempo_ramps_v050.py`); the app never migrates by itself.
- `build/`: PyInstaller spec.
- `docs/`: localized docs (`README.ko.md`, `README.ja.md`).
- `assets/`: icons, fonts, sounds.
- `library/`: user library data directory created/used at runtime. Treat as user data.
- `debug/`: generated debug artifacts; do not assume they are source assets.

## App Flow
- Entry is `app.main:main()`.
- On startup the app:
  - loads `config.json`,
  - ensures the library directory exists,
  - checks `library/VERSION`,
  - checks the on-disk library version through the migration adapter and stops on a mismatch,
  - creates and shows `AppWindow`.
- `AppWindow` creates the shared `EventBus`, `DataModel`, `TimelineCoordinator`, `PlayerController`, `ExternalSyncController`, `SegmentReanalysisManager`, and `MainPane`.
- UI state is mostly signal-driven. Feature/property/album-art signals are notifications only; listeners read current values from the shared `DataModel`.

## Analysis Flow
- Track load requests enter through `EventBus.sig_request_load_track` and end up in `AppWindow.analyze_file()`.
- Full-track analysis runs in a spawned subprocess via `core.analysis_worker.AnalysisWorker`.
- The worker calls `analyzer_core.global_analyzer.precompute_features(...)`.
- Current pipeline in `global_analyzer.py`:
  - decode/load audio,
  - optional HPSS split,
  - overview waveform levels (per-column peak and low/mid/high band RMS),
  - beat tracking (`analyzer_core.beat.general.track` on the frame features; frames on the fixed grid `frame_features.FRAME_HOP_LENGTH` (256); the models were trained on 22050 Hz HPSS frames, another analysis sample rate or HPSS off gives the tracker frames of its own): per training seed, tempo candidates from the activation's periodicity -> section decoding into steady (tempo, phase) grids with one section cost -> exact tempo / phase fit -> sudden vs gradual tempo changes (linear ramps, continuous or with a jump at one end) -> bar-supported half-beat phase (bars of 2, 3 or 4 beats) -> metronome critic (3:2 / 4:3 misreads) -> tempo-change boundaries on the refine onset -> slow tail held -> drifting sections -> tiny re-syncs merged -> junction rules (no too-close / too-far beats at grid changes) -> bars (a Viterbi over 3- and 4-beat bars on two downbeat observers; an odd bar costs, a new section starts free); then the seed whose grid agrees most with the other two. Every tempo is read in the octave from BPM min. The rows have `inizio` on a downbeat and `ts` the bar length (`bars.to_rows` splits a row where the bar length changes), so the segments always rebuild the beats and downbeats,
  - JumpCUE detection,
  - beat-synchronous chroma extraction (used by the key analysis, not stored),
  - key analysis,
  - persistence to SQLite (`LibraryDB`) and NPZ (`FeatureNPZStore`).
- Cached-track loads can skip analysis and load metadata/features directly from `library.db` + NPZ.
- Reanalyze-track requests (`sig_reanalyze_requested`) force a fresh full analysis and overwrite stored results.

## Segment Reanalysis
- Partial reanalysis is orchestrated by `core.segment_reanalysis_manager.SegmentReanalysisManager`.
- Work is offloaded to a subprocess via `SegmentReanalysisWorker`.
- Supported modes currently include:
  - beat segment reanalysis,
  - static key selection reanalysis,
  - dynamic full key reanalysis,
  - JumpCUE reanalysis (with AutoCUE on JumpCUEs, the AutoCUE JumpCUE goes where the editor's current phrases' lead-in ends; with AutoCUE on CUE points, the Phrase page's CUE button places it),
  - phrase reanalysis (the whole track on the current beatgrid; the CUE points stay).
- Results are pushed back into the in-memory `DataModel` and editor widgets; they are not a separate analysis system.

## Data & Persistence
- Track metadata lives in `library/library.db` in the `tracks` table.
- BPM and key transitions are also persisted in:
  - `track_bpm_segments`
  - `track_key_segments`
- Tempo segments are rows `[start, end, bpm, inizio, ts_num, bpm_end]` (`core.tempo_rows`). A `bpm_end` that differs from `bpm` is a linear tempo ramp (tempo linear in time from `start` to `end`; its beats come from the closed-form phase); rows without it (5 columns, older data) or with it equal are steady. `inizio` is a downbeat on the row's own grid. `track_bpm_segments.bpm_end` mirrors it (NULL = steady): transition search reads a ramp as a change from its start to its end tempo, and a phrase section inside a ramp gets the tempo at its midpoint. The editor keeps ramp rows (cut / merge / shift follow the tempo line; Assign makes the row steady at the reference BPM; the ramp start / end buttons make the reference BPM the row's start or end tempo, its first beat kept), and the Rekordbox export writes one TEMPO entry per beat of a ramp.
- Numeric feature arrays live in `{libpath}/{uid}.npz` through `FeatureNPZStore`. Only what is read is stored: beats, tempo/key segments, the overview waveform levels (`wave_overview_np` / `wave_overview_ref`), duration, sr, tempo_global, and the JumpCUE / CUE point / phrase blocks. Saving replaces a block as a whole (`save_worker._replace_block`).
- Track identity across DB and NPZ is the `uid` field stored in SQLite.
- `tracks.path` is stored as given, never normalized (case / separators can name different files on other file systems). A path that may spell a library file differently (a drop, External Sync's deck path) is looked up by the file it opens (`LibraryDB.find_file`) and then used as stored.
- Album art thumbnails (downsampled, JPEG) live in `{libpath}/thumbs/{uid}.jpg` (`core.thumbnails`): written by the full analysis, removed with the track; the library table shows the top of each as its first column.
- The key strip is rendered just in time from `key_segments` (`utils.keystrip.render_keystrip`); no image is stored.
- The overview waveform is rendered just in time from its stored levels (`utils.wave.render_wave_overview`) in the style of `viewconfig.waveform_style` (RGB, or rekordbox-style 3Band); no image is stored. The detail waveform is rendered from the playback PCM.
- JumpCUEs are kept in two states in the NPZ: `jump_cues_np` (the user's, replaced by edits) and `jump_cues_auto_np` (the automatic result, replaced only by full analysis or JumpCUE reanalysis). The JumpCUE optimizer (settings → Analysis → CUE) learns from where they differ; do not write edits into `jump_cues_auto_np`.
- A JumpCUE group's colour is a preset index (`graph_color_idx`, presets in `utils/jumpcue_colors.py`), independent of its group number (`graph_id`); no RGB is stored. Edits carry it on each cue as `color_idx`.
- CUE points (`cue_points_np`, with a colour preset `cue_color_idx`) are the user's: the full analysis makes them from the phrases by the per-label rules in `config.cuepointconfig` (Settings > Analysis > CUE; LEAD_IN has no rule), after that only the CUE page edits them (phrase edits do not regenerate them), or the Phrase page's CUE button replaces them with what the same rules give the current phrases.
- Phrases cover the track when read: stretches without a stored phrase (all of a track analyzed without phrases) are `UNASSIGNED` (`utils.phrases.fill_unassigned`). It is shown and edited like a label but never stored (`assigned_phrases` on save), so it never reaches training data.
- Phrase labels are the fixed `utils.phrases.PHRASE_LABELS` (no custom labels). How each is shown is a setting (Settings > View > Phrase labels: an alias and an overview abbreviation, empty = the label / its first letter), applied through `utils.phrases.set_phrase_label_names`; stored data always keeps the label itself.
- JumpCUE letters are slots (rekordbox hot cue letters A-Z): the analysis letters them by point; edits keep each cue in the slot the user gave it (no relettering by time). A JumpCUE's user name is its `cue_comment` (equal to its letter = no name). CUE points are rekordbox memory cues: always in time order, no letters; a CUE point's name is its `cue_label`.
- `DataModel` holds the live in-memory track state:
  - `features`
  - `properties`
  - `duration_sec`
  - `gp`
  - `album_art`
  - optional predecoded audio buffers

## UI Notes
- `ui.pane.MainPane` owns the top-level plot stack, overview, track info, transport controls, editor panel, and library widget.
- Analysis views are plugin-like classes from `views/` instantiated via `REGISTRY`.
- Enabled views come from `viewconfig`:
  - waveform
  - beatgrid
  - keystrip
  - JumpCUE
- `ui.library.LibraryWidget` is not just a browser:
  - it performs text search,
  - transition search using DB segment tables,
  - remove/reanalyze/edit/export actions.
- `ui.export.ExportTrackDialog` prefers the in-memory `DataModel` for the currently loaded track and falls back to NPZ when needed.

## External Sync
- `core.external_sync.ExternalSyncController` can follow an external process via memory reads using `PyMemoryEditor`.
- When enabled, local playback/load controls are intentionally restricted.
- Process denylist lives in `process_denylist.json`.
- If you touch External Sync config or behavior, keep `core/config.py`, `ui/config/`, and `app/window.py` aligned.

## Change Guidelines
- Keep heavy DSP, file IO, and long-running work off the GUI thread.
- Worker subprocess entry points must remain module-level pickleable callables. Do not pass Qt objects into multiprocessing workers.
- When changing config shape/defaults:
  - update dataclasses in `core/config.py`,
  - update `default_cfg()`,
  - keep `config.json` compatibility in mind,
  - update `ui/config/`,
  - update any call sites assuming the old fields.
- When changing persisted analysis outputs:
  - update readers in `DataModel`/UI/views,
  - update DB segment adapters in `core.linear_segments` if segment schema changes,
  - consider migration needs for existing libraries.
- `library/` is user data. Do not delete, reset, or commit its contents.
- Preserve `uid` linkage between SQLite and NPZ when editing library code.
- Prefer contiguous NumPy arrays where practical; several rendering and DSP paths assume NumPy-friendly numeric buffers.
- Keep Qt signal semantics stable. Many slots expect notification-only signals and then read shared model state.
- If you change view registration or naming, update `views.base.REGISTRY` consumers accordingly.

## Build / Packaging
- PyInstaller spec is `build/mixlyzer.spec`.
- The spec does not bundle `config.json` (it holds personal settings such as External Sync memory addresses); the app writes one from `default_cfg()` on first launch. Keep `default_cfg()` free of personal values.
- Asset and `ffmpeg.exe` bundling are presently commented out in the spec, so do not assume packaging already includes them.
- There is no active `installer/` directory in this repo snapshot; do not document or depend on one unless you add it.

## Safe Assumptions
- The codebase is Windows-first, but much of the Python code is portable.
- The library format is versioned; old docs that ignore migration are outdated.
- Root `README.md` exists, but localized docs currently live under `docs/`.
