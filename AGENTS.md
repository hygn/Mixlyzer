# AGENTS (Mixlyzer)

Working notes for AI/automation agents editing the Mixlyzer codebase.

## Run & Environment
- Python 3.12+.
- Dependency management uses `uv`.
- Install deps with `uv sync` (with the development group: Babel, for `build/translations.py`).
- Launch the app with `uv run python -m app.main`.
- Keep root-level `config.json` available. `load_cfg()` reads and rewrites it.
- Keep a root-level `ffmpeg.exe` (Windows) / executable `ffmpeg` (Linux, macOS), or provide FFmpeg on `PATH` for `core/audio/decoder.py`.
- Do not run the app as administrator (Windows) or root (Linux, macOS); `app/main.py` explicitly blocks it.
- Windows, Linux and macOS are supported (`core.os_support`); on any other system the app warns at the start and runs on the fallbacks.

## Repo Map
- `app/`: Qt entrypoint and top-level window wiring.
  - `main.py`: app bootstrap (the UI language first, before the UI is imported), dark palette, font loading, library-version migration flow. Kept light: a spawned worker process imports the main module again, so `app.window` is imported by the startup, not at the top.
  - `startup.py`: the startup splash (the task, what it is doing, a Task and an Overall progress bar) and the slow work done before the window opens, in order: the heavy imports, the playback's numba kernels (JIT, from numba's disk cache), FFmpeg's first run, the search readers (`reading.warm_up`), every stored reading's search keys (`reading.keys`' LRU cache), then `AppWindow(progress=)`. Each task's weight is its measured time; events are taken between updates except while the window is built (only the splash repainted there, so no timer reaches a half-built window).
  - `window.py`: `AppWindow`, workflow wiring, EventBus wiring, External Sync, metronome, model updates.
- `core/`: shared runtime/state/integration layer.
  - `config.py`: dataclass config schema, defaults, `load_cfg()`.
  - `os_support.py`: the operating system, detected once per process (its first import: `app.main` at the start, each spawned worker) into flags (`OS`, `IS_WINDOWS` / `IS_LINUX` / `IS_MACOS`, `IS_SUPPORTED`, `MEMORY_SYNC`), and the operations that differ by it: elevation (IsUserAnAdmin / uid 0), the taskbar identity and 1 ms timer (Windows only), hidden console windows of child programs (Windows only), the running processes (tasklist / ps), revealing a file (Explorer / Finder / the folder via xdg-open), whether another program holds a key as a global hotkey (`hotkey_taken`: RegisterHotKey refused on Windows; not checked elsewhere). The other Windows API users branch on the flags with their fallbacks: the window occlusion check (`utils/window_visibility.py`: user32 + DWM / Qt's exposure), the scratch cursor poller (`core/audio/cursor_poller.py`: Win32 / the GUI's mouse events), the physical core count for sklearn (`analyzer_core/utils.py`: Win32 / loky's own).
  - `i18n.py`: the UI's language (Settings > View > Language, `viewconfig.language`, "" = the operating system's; applied at the next start). The languages are `assets/i18n/languages.json` (BCP 47 tags, language-REGION: `en-US`, `ko-KR`, `ja-JP` -> the language's name in itself), each one's gettext catalog `assets/i18n/languages/<tag>/mixlyzer.po`, read as it is (no compiled .mo; fuzzy entries left out). `tr()` / `tr_n()` / `tr_ctx()` translate, `tr_mark()` marks a table's texts (translated where shown); Qt's own texts (standard buttons, file dialogs) come from its `qtbase` catalogs (`install_qt`). Worker processes take the language from the environment (`MIXLYZER_LANGUAGE`). English (US) is the source.
  - `shortcuts.py`: the keyboard shortcuts' files: the shipped `assets/shortcuts/default_shortcuts.json` (only read; `save` refuses it) and the user's `assets/shortcuts/shortcuts.json` (gitignored; only the actions changed from the shipped keys, a missing file = the shipped keys), both action id -> key sequences in Qt's portable text.
  - `model.py`: shared `DataModel` and `GlobalParams`.
  - `event_bus.py`: Qt signal hub.
  - `workers/`: analysis, segment reanalysis, optimizer, and Rekordbox worker implementations.
  - `workflows/`: Qt-facing worker lifecycle and queue orchestration.
  - `library_handler.py`: SQLite metadata store plus BPM/key segment tables and transition search.
  - `analysis_lib_handler.py`: NPZ feature store keyed by track UID.
  - `external_sync.py`: memory-based external deck sync / transport lockout.
  - `player.py`, `timeline.py`, `taskmanager.py`: playback, viewport timeline, worker/task UI state.
  - `tempo_rows.py`: tempo-segment rows and the beats they give (NumPy only; `rebuild_grid_from_segments`, ramp helpers), shared by the analyzer, editor, views and exporters.
  - `reading.py`: readings for search, in three layers. (1) `readings(text)`: romaji by script -- Japanese by Sudachi (SudachiDict core) + a kana table, Hangul by Revised Romanization with its sound changes (own code), Latin kept, other scripts by anyascii; Han without kana once per language (`HAN_LANGUAGES`): Japanese, Mandarin (pypinyin), Cantonese (pycantonese Jyutping; a form it lacks, 会 学, read as its Unihan variant: `unihan_cantonese_variants.tsv`), Korean Hanja (Unihan kHangul: `assets/readings/unihan_korean.tsv`, spelled as a word -- the initial sound law, 렬/률 as 열/율 after a vowel or ㄴ, 不 as 부 before ㄷ/ㅈ -- then as Hangul), Sino-Vietnamese (Unihan kVietnamese, which mixes in Nom readings: `unihan_vietnamese.tsv`), Taiwan Mandarin (only where Unihan kMandarin's Taiwan value differs: `unihan_taiwan_mandarin.tsv`); the Unihan tables are Unicode License, a character missing from them is read in Mandarin; at most `READING_SLOTS` (7). Han mixed with other characters is found by its Korean Hanja reading too (Settings > Search's Mixed Text, with Korean Hanja a language): beside Latin letters or Hangul all of it (the per-language readings above); a text with kana is read in Japanese, and also in Korean Hanja for its runs of `han_run_min` (2 or more, default 3) Han characters, the rest in Japanese (`han_runs` on/off; `searchconfig.han_run()` -> `readings(..., han_run=)`, 0 = off; Sino-Japanese words: 異世界生活 isegyesaenghwal; shorter runs are mostly native words: 君, 始める). Korean only for now: other languages' readings of these runs are not made. The languages are Settings > Search (`config.searchconfig`): `db_languages` for the stored readings (`LibraryDB(..., reading_languages=)`: the analysis, Edit Track), `query_languages` and Hangul as Korean / as a foreign word for the query (`query_keys`), and `query_keyboard_layout`: a query typed on the 2-set (Dubeolsik) keyboard without the 한/영 key also read as meant (`keys_to_hangul`: tkghl 사회, word by word, only words composing into 2+ whole syllables with no loose jamo; `hangul_to_keys`: 무ㅐㅜㄷ anone, only with loose jamo). 2-set only (no 3-set layouts). (2) `normalize`: the search form where spellings of one sound agree (shi/si, tsu/츠, long vowels and doubled letters, voicing, l/r; a query's Hangul read as Korean and with Korean's spelling of foreign words). (3) `find`: contained, or nearly (RapidFuzz partial ratio of the query within a reading not much shorter than it, from `SIMILAR_MIN` letters; the library's Fuzzy toggle). `warm_up()` loads the readers (about 2 s; `app.startup` does it before the window opens). Libraries were chosen for LGPL-compatible licences (GPL / CC BY-SA ones -- pykakasi, unidecode, korean-romanizer, ko-pron (a Wiktionary port), taibun's data -- are not used).
  - `audio/`: decode + low-level audio helpers.
- `analyzer_core/`: DSP and editor-side reanalysis algorithms.
  - `global_analyzer.py`: full analysis pipeline and persistence handoff.
  - `beat/`: BPM / beatgrid analysis.
    - `general/`: the full analysis's beat tracker (NumPy only; shipped weights pinned as `assets/weights/default_beat_tcn.npz` (the three trainings in one file), `default_beat_critic.npz`, `default_downbeat_ctx_hgb.npz`, `default_beat_decoder.json`, `default_refine_onset_hgb.npz`; the settings name the user's files (same names without `default_`), and a missing user file means the shipped one (`core/model_files.py`); Reoptimize never writes a `default_` file): `tcn.py` frame TCN, `decode.py` section decoding, `gradual.py` ramps and drifting sections, `critic.py` metronome critic, `junctions.py` junction rules, `bars.py` bar check / bars / production rows, `tracker.py` the pipeline and the consensus of the three trainings.
    - The editor's segment reanalysis (`analyzer_core/editor/beatgrid.py`) uses the same models (the three TCN trainings averaged) on the segment plus 4 s of context: the steady grid with the most evidence (with a reference BPM: within 20 % of it), or a linear ramp when it explains the onsets better by one beat of evidence and moves a beat audibly (`gradual.ramp_or_steady`; guesses from the neighbours' tempos and the segment's halves), then static bars for its downbeat. Beside `general/`, the beat package holds what the tracker reads: `frame_features.py` (the frame features, also used by JumpCUE), `learned_onset.py` (the tempo-change onset, a GBM), `grid_refine.py` (`refine_boundaries` on that onset, `hold_slow_tail`), `downbeat_features.py` (the per-beat features of the contextual downbeat model) with `periodicity.py` and `melody_contour.py`; `optimizer/onset_training.py` is the onset models' training data (library tracks, onset channel cache, samples). Settings > Analysis > Beat: BPM min (the octave), Dynamic Analysis (off: one steady grid for the whole track) and Dynamic Downbeat (static analysis only: odd bars / meter changes, off = one bar length and phase; always on with dynamic analysis), and in Advanced the model paths (tempo-change onset, beat TCN, metronome critic, downbeat model, decoder costs `assets/weights/beat_decoder.json`: a new section / a same-tempo half-beat reset / an odd bar, in evidence units of the activation) with their feature caches; Reoptimize Beat Models retrains any of them on the library's grids (`optimizer/`): the decoder costs (`beat_decoder_optimizer.py`: every song tracked with each candidate combination, the costs with the least wrong time -- unusable beats + missed bars -- kept; refit them after the beat model changes, about an hour), the downbeat GBM (sklearn, minutes), the TCN and the critic fine-tuned from the current model in NumPy (`beat_tcn_optimizer.py`, `beat_critic_optimizer.py`: their own forward / backward passes, trainings in parallel processes on a memory-mapped copy of the frames; tens of minutes), all on the tracker's own frames cached by `optimizer/beat_frames.py` (22050 Hz, HPSS, hop 256, every FrameFeatures field in float16). The networks start from the current model, the shipped one, or from scratch, and the decoder costs are refitted around the current or the shipped costs (Quick: the half-beat reset and odd-bar costs only); `optimizer/model_start.py` holds the minimum song counts (blocked below 200 songs from scratch, 20 for fine-tuning, 50 for the GBMs) and the warnings the dialog shows. With CV a group of songs is held out first and the current and new models are compared on it. There is no beatgrid offset: grids are stored as the tracker gives them.
  - `key/`: key analysis. `chord_key.py` is the full analysis's key analyzer (NumPy only; weights `assets/weights/key_chord_cnn.npz`, a missing file = the shipped `default_key_chord_cnn.npz`): a root-equivariant chord CNN reads every beat (bass / treble CQT chroma and STFT chroma of the harmonic part, 22050 Hz) as P(chord | beat) over 24 triads, the key evidence is log sum_c P(c | beat) P(c | key, phrase position), and a bar Viterbi over the 24 keys sets the keys (a change costs J at a phrase start, J + JX elsewhere, half of it between relative keys, plus an interval prior from the annotated key changes: a minor third / half step / whole step is cheap, a fifth dear; excursions shorter than MINEX bars are removed). Keys are DJ / DAW keys: the scale (key signature, Alphanumeric number), major / minor by the tonic chord. The chord layer was trained through the key loss only (no chord labels). The editor's key reanalysis (`analyzer_core/editor/keystrip.py`) uses the same model: the whole track (the decoder) or a selected span (the key with the most evidence over its beats, read on the whole track). Settings > Analysis > Key: Dynamic Analysis (off: one key covering the whole track -- the dynamic analysis is run as it is and the key holding the most time in its INTRO / OUTRO / INTERLUDE phrases is taken, else the key holding the most time; for the analysis and the editor's whole-track reanalysis), the model path, its feature cache and Reoptimize Key Model (`optimizer/key_model_optimizer.py`: the network fine-tuned from the current / shipped model or trained from scratch in NumPy, its own forward / backward, on the frame chroma cached per track; training songs are the tracks whose keys differ from the kept automatic result, and tracks analyzed before it was kept whose annotation does not have the old analyzer's raw shape; with CV a group of songs is held out and the whole key analysis, current vs new, is compared on it before saving; the decoder settings are kept). There is no other key analyzer.
  - `self_correlation/`: JumpCUE detection.
  - `editor/`: beatgrid, key, JumpCUE reanalysis helpers used by worker subprocesses.
- `ui/`: Qt widgets/dialogs.
  - `pane.py`: `MainPane` composition.
  - `library.py`: library table, search, transition search, export/reanalyze/edit actions.
  - `edit_panel/`: track editor panel (`panel.py` + per-section mixins: beatgrid, key, phrase, jumpcue, CUE points, beat counter, edit history, save worker; key and phrase are edited by the segment under the playhead; `cue_slots.py` is the CUE page's 4 x 2 slot list). Pages: Beatgrid, Key, Phrase, CUE, picked in the selector row after the edit lock (on at every start, not saved): locked, the pages switch but nothing is edited or saved (`EditPanel._apply_edit_enabled`: editing needs the lock open and no segment reanalysis running); what never changes the track stays usable (reference BPM / TAP, beat counter, listening aids, metronome, CUE CALL / JUMP). The Beatgrid / Key / Phrase pages are blocks three button rows high, each a full rectangle, laid out alike so a function sits in the same place on each (`blocks.py`): the edit modes that modify the tools' actions, all at the far left (a mode's two buttons fill a column), a rule, the tools (row 1 moves in time and reanalysis, row 2 values, row 3 split / merge first), a rule, a group of the page's own (time signatures / listening aids / labels); a wide icon takes two cells. A tool whose action follows a mode shows the mode in its icon (`icons.badged_icon`: the scope's three blocks, or one / four dots for a beat / bar step) and its tooltip.
  - `theme/`: app look on every platform: Fusion + palette + the app-wide style sheet `assets/styles/app.qss` with the design tokens (`tokens.py`) filled in (`qss.py`); colours for new UI come from `current_theme()`.
  - `shortcuts.py`: the keyboard shortcuts' actions (`ACTIONS`: id, group, name, auto-repeat) and `ShortcutManager` (a window QShortcut per action, its keys from `core.shortcuts`, Settings > Shortcuts' saves applied through `EventBus.sig_shortcuts_changed`; the handlers come from `MainPane.shortcut_handlers` with the edit panel's and the library's). Settings > Shortcuts is `config/shortcuts.py` (a main and an alternative key per action, a key taken from another action only once agreed, Revert / Revert All to the shipped keys).
  - `edit_song.py`, `export.py`, `config/`, `workers.py`: editing/export/settings/task dialogs. Settings > Analysis > Global shows the analyses' order and dependencies (`config/pipeline_map.py`: `assets/images/analysis_pipeline.svg`, drawn at the screen's pixel size, + its block rectangles `analysis_pipeline.json`; a click on a block opens its settings tab); the SVG has no words: they are in the JSON (`texts`, with position, size, colour and room) and drawn translated over it (`pipeline_map.WORDS` lists them for the catalogs); redraw both when the pipeline changes.
- `views/`: pyqtgraph view plugins registered through `views.base.REGISTRY`.
- `utils/`: labels, colors, fonts, JumpCUE extraction/render helpers, keystrip utilities, waveform helpers.
- `third_party/`: export adapters such as Rekordbox XML generation.
- `migration/`: read-only library-version compatibility adapter, plus one-off development migrations run by hand (`phrase_index_v040.py`, `tempo_ramps_v050.py`, `wave_overview_bands.py`, `uid_keys_v060.py`); the app never migrates by itself.
- `build/`: PyInstaller spec; `translations.py` updates the translation catalogs (Babel).
- `docs/`: localized docs (`README.ko.md`, `README.ja.md`).
- `assets/`: icons, fonts, sounds; `i18n/`: the UI's languages and their catalogs (`core.i18n`); `shortcuts/`: the keyboard shortcuts (`core.shortcuts`).
- `library/`: user library data directory created/used at runtime. Treat as user data.
- `debug/`: generated debug artifacts; do not assume they are source assets.

## App Flow
- Entry is `app.main:main()`.
- On startup the app:
  - sets the UI language (`config.json`'s `viewconfig.language`, only read),
  - shows the startup splash (`app.startup`),
  - loads `config.json`,
  - ensures the library directory exists,
  - checks `library/VERSION`,
  - checks the on-disk library version through the migration adapter and stops on a mismatch,
  - runs the startup tasks (imports, JIT, FFmpeg, search readers and keys) and builds `AppWindow`, then shows it and closes the splash.
- `AppWindow` creates the shared `EventBus`, `DataModel`, `TimelineCoordinator`, `PlayerController`, `ExternalSyncController`, `SegmentReanalysisManager`, and `MainPane`.
- UI state is mostly signal-driven. Feature/property/album-art signals are notifications only; listeners read current values from the shared `DataModel`.

## Analysis Flow
- Track load requests enter through `EventBus.sig_request_load_track` and end up in `AppWindow.analyze_file()`.
- Full-track analysis runs in a spawned subprocess via `core.analysis_worker.AnalysisWorker`.
- The worker calls `analyzer_core.global_analyzer.precompute_features(...)`.
- Current pipeline in `global_analyzer.py`:
  - decode/load audio,
  - HPSS split (always on, not a setting),
  - overview waveform (its own 48 kHz decode: min / max of 32768 columns, the power of 12 log-spaced bands in 8192 columns, and the detail waveform's three-band scale),
  - beat tracking (`analyzer_core.beat.general.track` on the frame features; frames on the fixed grid `frame_features.FRAME_HOP_LENGTH` (256); the models were trained on 22050 Hz HPSS frames; every analysis runs at 22050 Hz (`core.config.ANALYSIS_SAMPLE_RATE`, not a setting) with HPSS, so the tracker reads the analysis's own frames): per training seed, tempo candidates from the activation's periodicity -> section decoding into steady (tempo, phase) grids with one section cost -> exact tempo / phase fit -> sudden vs gradual tempo changes (linear ramps, continuous or with a jump at one end) -> bar-supported half-beat phase (bars of 2, 3 or 4 beats) -> metronome critic (3:2 / 4:3 misreads) -> tempo-change boundaries on the refine onset -> slow tail held -> drifting sections -> tiny re-syncs merged -> junction rules (no too-close / too-far beats at grid changes) -> bars (a Viterbi over 3- and 4-beat bars on two downbeat observers; an odd bar costs, a new section starts free); then the seed whose grid agrees most with the other two. Every tempo is read in the octave from BPM min. The rows have `inizio` on a downbeat and `ts` the bar length (`bars.to_rows` splits a row where the bar length changes), so the segments always rebuild the beats and downbeats,
  - JumpCUE detection,
  - key analysis (`analyzer_core.key.chord_key.analyze_keys` on the beats, the downbeats and the phrase starts; no keys when it fails),
  - persistence to SQLite (`LibraryDB`) and NPZ (`FeatureNPZStore`).
- Cached-track loads can skip analysis and load metadata/features directly from `library.db` + NPZ.
- Reanalyze-track requests (`sig_reanalyze_requested`) force a fresh full analysis and overwrite stored results.

## Segment Reanalysis
- Partial reanalysis is orchestrated by `core.segment_reanalysis_manager.SegmentReanalysisManager`.
- Work is offloaded to a subprocess via `SegmentReanalysisWorker`.
- Supported modes currently include:
  - beat segment reanalysis,
  - static key selection reanalysis (`chord_key`: the key with the most evidence over the selection),
  - dynamic full key reanalysis (`chord_key` on the editor's current beatgrid, bars and phrases),
  - JumpCUE reanalysis (with AutoCUE on JumpCUEs, the AutoCUE JumpCUE goes where the editor's current phrases' lead-in ends; with AutoCUE on CUE points, the Phrase page's CUE button places it),
  - phrase reanalysis (the whole track on the current beatgrid; the CUE points stay).
- Results are pushed back into the in-memory `DataModel` and editor widgets; they are not a separate analysis system.

## Data & Persistence
- Track metadata lives in `library/library.db` in the `tracks` table, keyed by `uid` (PRIMARY KEY, canonical UUIDv4).
- `track_readings` (uid, field, slot) holds the romanized readings of each track's title / artist / album (`core.reading`): the search's intermediate representation, up to 7 a field, each with its source (ja, zh, yue, ko-hanja, vi, zh-tw, ko, text, or user). The readings the user added (Edit Track's Readings grid: only the cells typed there; source `user`) are kept and come first; the automatic ones after them are made anew in the stored languages whenever the track is written (`LibraryDB.upsert` -> `refresh_readings`, `library_handler.field_readings`), so a language unchecked removes only automatic readings. An automatic reading the user removed (emptied, Delete, or typed over in the grid) is kept in `track_reading_removals` (uid, field, reading) and left out from then on, whatever the languages; typed again, it is the user's. Settings > Search's Re-read Library (also started when what the stored readings depend on changes: `searchconfig.stored_readings()`) reads every track again in a spawned process (`core/workers/reading_refresh.py`, `core/workflows/reading_refresh.py`; 50 tracks a transaction, the library reloaded when done). The library's text search matches Title / Artist / Album by these too.
- Playlists (`playlists`, keyed by their own UUIDv4 `uid`) hold one `playlist_tracks` row per entry, (`playlist_uid`, `position`) -> `track_uid`: a track may be in a playlist more than once, so the UI addresses an entry by its position (reorder / remove), never by the track. Removing a track removes its entries and closes the playlists up.
- BPM and key transitions are also persisted in:
  - `track_bpm_segments`
  - `track_key_segments`
- Tempo segments are rows `[start, end, bpm, inizio, ts_num, bpm_end]` (`core.tempo_rows`). A `bpm_end` that differs from `bpm` is a linear tempo ramp (tempo linear in time from `start` to `end`; its beats come from the closed-form phase); rows without it (5 columns, older data) or with it equal are steady. `inizio` is a downbeat on the row's own grid. `track_bpm_segments.bpm_end` mirrors it (NULL = steady): transition search reads a ramp as a change from its start to its end tempo, and a phrase section inside a ramp gets the tempo at its midpoint. The editor keeps ramp rows (cut / merge / shift follow the tempo line; Assign makes the row steady at the reference BPM; the ramp start / end buttons make the reference BPM the row's start or end tempo, its first beat kept), and the Rekordbox export writes one TEMPO entry per beat of a ramp.
- Numeric feature arrays live in `{libpath}/{uid}.npz` through `FeatureNPZStore`. Only what is read is stored: beats, tempo/key segments, the overview waveform (`wave_minmax_np` / `wave_minmax_ref`, `wave_bands_np` / `wave_bands_ref`, `wave_detail_band_ref`), duration, sr, tempo_global, and the JumpCUE / CUE point / phrase blocks. Saving replaces a block as a whole (`save_worker._replace_block`).
- Track identity across DB, NPZ, thumbnails and playlists is the `uid`; the file path is an editable attribute (`LibraryDB.set_path`; the Edit Track dialog). `tracks.path` stays UNIQUE (one file, one track).
- `tracks.path` is stored as given, never normalized (case / separators can name different files on other file systems). A path that may spell a library file differently (a drop, External Sync's deck path) is looked up by the file it opens (`LibraryDB.find_file`) and then used as stored.
- A library track whose file is missing is not loaded: the app says so and asks the user for the file's new location (no guessing by name or content).
- Album art thumbnails (downsampled, JPEG) live in `{libpath}/thumbs/{uid}.jpg` (`core.thumbnails`): written by the full analysis, removed with the track; the library table shows the top of each as its first column.
- The key strip is rendered just in time from `key_segments` (`utils.keystrip.render_keystrip`); no image is stored.
- The overview waveform and the main view's preview (its stand-in until the detail chunks are drawn) are rendered just in time from the stored overview (`utils.wave.render_wave_overview` / `render_wave_detail_preview`) in the style of `viewconfig.waveform_style` (RGB, or rekordbox-style 3Band): low / mid / high are summed from the 12 stored bands with the ranges of Settings > View > Waveform bands (`viewconfig.env_*`, weighted by log-frequency overlap), so a change of the ranges redraws them at once without re-analysis; no image is stored. The detail waveform is rendered from the playback PCM (its three-band scale is the stored `wave_detail_band_ref`). Libraries analyzed before this format are converted by hand with `migration/wave_overview_bands.py`.
- JumpCUEs are kept in two states in the NPZ: `jump_cues_np` (the user's, replaced by edits) and `jump_cues_auto_np` (the automatic result, replaced only by full analysis or JumpCUE reanalysis). The JumpCUE optimizer (settings → Analysis → CUE) learns from where they differ; do not write edits into `jump_cues_auto_np`.
- A JumpCUE group's colour is a preset index (`graph_color_idx`, presets in `utils/jumpcue_colors.py`), independent of its group number (`graph_id`); no RGB is stored. Edits carry it on each cue as `color_idx`.
- Keys are kept in two states in the NPZ like the JumpCUEs: `key_segments` (the user's, replaced by edits) and `key_segments_auto` (the automatic result, replaced by the full analysis and the editor's whole-track key reanalysis, saved with the editor's next save). The key optimizer learns from the tracks where they differ; do not write edits into `key_segments_auto`.
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
- Edit focus: while the edit panel is unlocked, the main view and the overview emphasize the visualizations of the pages shown (lines thicker, markers larger) and dim the other pages' (`EventBus.sig_edit_focus_changed`, a frozenset of `views.base.EDIT_PAGES`; a view declares its page as `ViewPlugin.EDIT_PAGE` and styles its items in `_apply_emphasis`, also after creating items; the overview in `OverviewWidget._apply_emphasis`). Shapes and colours stay; the waveform and the playhead belong to no page.
- Enabled views come from `viewconfig`:
  - waveform
  - beatgrid
  - keystrip
  - JumpCUE
- `ui.library.LibraryWidget` is not just a browser:
  - its table rows are made when the source changes; a library update (`sig_lib_updated`, several in a row applied once after `LIBRARY_UPDATE_DELAY_MS`) changes only the rows of the tracks added / changed / removed (up to `INCREMENTAL_ROWS_MAX`, else the table is rebuilt), and the search and filters hide rows (applied once they pause, `SEARCH_DEBOUNCE_MS`); actions use `_selected_rows()`, which leaves hidden rows out (Ctrl+A selects them too),
  - it performs text search,
  - transition search using DB segment tables,
  - remove/reanalyze/edit/export actions.
- Keyboard shortcuts (`ui.shortcuts`): the shipped keys follow rekordbox's (Performance 1 preset, deck 1: Z play, Shift+A back to the start, R / T call the previous / next memory cue, 1-8 hot cues, Ctrl+Shift+1-8 clear them, Shift+Left load, Alt+F search, Shift+5 master tempo, = / - zoom) where Mixlyzer has the function, in three tiers: no modifier (Shift / Alt variants included) changes no data (playback, the playhead, the library's selection, the editor's pages (F1-F4) and lock (L)), Ctrl changes the data in memory (the editor's edits; Ctrl+Z / Y / S; a new playlist), Ctrl+Shift destroys data (deleting a cue, removing tracks, a full reanalysis). Every action has its own keys. An operation several editor pages have and that is the same on each (a key / phrase boundary earlier / later, split, merge, reanalysis of the current segment / the whole track, the edit scope, the boundary step) is one "Current page" action done on the current page (`EditPanel._context`: the page shown; with pages side by side the one picked (F1-F4) or clicked last, marked in the selector: its title lit (4 shown) or its name lit on its pair's button and the other's dimmed (2 shown)); one operation per key, never a different operation by page (the beatgrid shifts, BPM x2 / x1/2 and the key's semitones are their pages' own). A page's own tool shows its page (and the CUE list page holding its slot), which becomes the current page. Either way the button is worked as the mouse does (`EditPanel._shortcut_click`) and flashes (`ui/edit_panel/flash.py`: an accent glow drawn over the button, its style sheet untouched); an edit while editing is locked does nothing and the status bar says why. Plain navigation keys (Up / Down / Page / Home / End; plain Left / Right in a tree) go to a focused list, spin or combo box instead (typed characters already go to a text field); Up / Down from the search field move the library's selection. The sources panel beside the library (`_LibrarySourceTree`): Alt+S goes there and back; moving with the keys only highlights a source (a click still opens it), the load key (Shift+Left) shows the highlighted one and returns to its tracks; Esc leaves the search field (its text kept) or the sources panel for the tracks. Settings > Shortcuts shows each action by a short name, its details in the tooltip (`Action.tip`). A key another program holds as a global hotkey never reaches the app (graphics card overlays: NVIDIA / AMD hold Alt+R, Alt+Z, Alt+F1 / F9 / F10, Ctrl+Alt+R / M, F12 ...): Settings > Shortcuts marks such keys and asks before assigning one, the start says which assigned keys are held (`os_support.hotkey_taken`), and the shipped keys avoid them.
- `ui.export.ExportTrackDialog` prefers the in-memory `DataModel` for the currently loaded track and falls back to NPZ when needed.

## External Sync
- `core.external_sync.ExternalSyncController` can follow an external process via memory reads using `PyMemoryEditor`.
- Memory reads are Windows and Linux only (`os_support.MEMORY_SYNC`): on macOS PyMemoryEditor is neither installed (a dependency marker) nor imported (it is imported where a process is opened), the settings tab is not shown and `load_cfg()` keeps the setting off.
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
- A frequent report from another thread to the GUI is the newest value with one queued notification at a time (`core.player._Latest`: the playhead time, the output peak), never a queued signal per value: queued signals are not merged, so they pile up while the GUI is busy and replay stale values after. Progress from a loop is reported when its percent changes. The output buffer stats go every 0.25 s.
- If you change view registration or naming, update `views.base.REGISTRY` consumers accordingly.
- A new tool button or action gets a shortcut action: its entry in `ui.shortcuts.ACTIONS`, its handler (`shortcut_handlers` of the part that owns it), and its shipped keys in `assets/shortcuts/default_shortcuts.json` (every action has one; unique, in the tier of what it does, not a system's own: Ctrl+Alt+Del / arrows / F-keys, Ctrl+Alt+L / T (Linux lock screen, terminal), Alt+F2 / F4, Ctrl+Q / W / H (macOS quit, close, hide), Ctrl+Space, Ctrl+Esc, nor a graphics card overlay's: Alt+R / Z, Alt+F1 / F3 / F9 / F10, Ctrl+Alt+R / M, Alt+Shift+R, Shift+F12, F12).
- Every text shown to the user goes through `core.i18n.tr()`: placeholders with `.format()` (no f-string inside `tr`), a whole sentence per case rather than words spliced into one (`tr_n` for counts), `tr_mark()` for a module-level table (translate it where it is shown). Logic never reads a displayed text (combo items carry their value as data). `tr` is never a local name (a function binding it cannot call it). After changing texts run `uv run python build/translations.py` and translate the new / fuzzy entries of each catalog.
- Not translated: the main screen's DJ terms (Tempo, Volume, BPM, Key, Bars, the CUE buttons' SET / DEL / MEMORY / JUMP / CALL), stored data (phrase labels, cue names, Rekordbox XML), logs, product names. Terms: JumpCUE and CUE point stay as they are (never hot cue / memory cue); the key notation 1A..12B is Alphanumeric (never Camelot); Japanese follows the rekordbox Japanese manual's terms.

## Build / Packaging
- PyInstaller spec is `build/mixlyzer.spec`.
- The spec does not bundle `config.json` (it holds personal settings such as External Sync memory addresses); the app writes one from `default_cfg()` on first launch. Keep `default_cfg()` free of personal values.
- Asset and `ffmpeg.exe` bundling are presently commented out in the spec, so do not assume packaging already includes them.
- There is no active `installer/` directory in this repo snapshot; do not document or depend on one unless you add it.

## Safe Assumptions
- Windows, Linux and macOS: code branches on `core.os_support`'s flags, never on `sys.platform` / `os.name` / `platform.system()` itself; a Windows API call has a fallback for the other systems.
- The library format is versioned; old docs that ignore migration are outdated.
- Root `README.md` exists, but localized docs currently live under `docs/`.
