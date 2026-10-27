# AGENTS: app/ (Mixlyzer)

The details of `app/`: the start, the splash and the window's wiring. The root `AGENTS.md`'s rules (run, layers, owners, change guidelines) apply here too; read it first.

## Map
- `app/`: Qt entrypoint and top-level window wiring.
  - `main.py`: app bootstrap (the UI language first, before the UI is imported), dark palette, font loading, library-version migration flow. Kept light: a spawned worker process imports the main module again, so `app.window` is imported by the startup, not at the top.
  - `startup.py`: the startup splash (the task, what it is doing, a Task and an Overall progress bar, and under them the console's last lines: `startup_console.py`, `ConsoleLines` tapping sys.stdout / stderr from `main()`'s start (the print hook, installed later, writes through it; taken out when the splash closes), drawn as a terminal's: the time since the start, a coloured tag per source (`[Startup]`, `[AudioEngine]`, ...), a finished task's time right-aligned, a failure in the danger colour; the engine's lines, printed on its relay thread, drawn at the next repaint) and the slow work done before the window opens, in order: the heavy imports, the Python audio engine's numba kernels (JIT, from numba's disk cache; skipped when the native engine is built), FFmpeg's first run, the search readers (`reading.warm_up`), the library's stores checked against library.db (`jobs.check_stores`, the owner's first job), every stored reading's search keys (`reading.keys`' LRU cache), then `AppWindow(progress=)`. numpy's and scipy's OpenBLAS load with one thread in this process (`one_blas_thread()`, first in `main()`: the GUI does no heavy linear algebra, and each OpenBLAS sets aside buffers per thread, about 1 GB committed on 16), the variable taken back once they have, so the spawned worker processes keep a thread per core. Each task's weight is its measured time; events are taken between updates except while the window is built (only the splash repainted there, so no timer reaches a half-built window).
  - `audio_report.py`: the audio setup printed at the start (`[AudioEngine]` lines: the engine that plays and why, the master's device / API / mode / buffer, the headphones' and the booth's outputs, the process's priority, the mixer's switches, MIDI); what the engine then opened follows as its own lines.
  - `window.py`: `AppWindow`, workflow wiring, EventBus wiring, External Sync, metronome, model updates.

## App Flow
- Entry is `app.main:main()`.
- On startup the app:
  - makes the app folder the current folder (`system.resource_paths.app_folder`: the package folder in a build, else the project root), whatever folder it was started from: `config.json`, the library and the caches are always the app's, and no program or `config.json` of another folder is taken,
  - sets the UI language (`config.json`'s `viewconfig.language`, only read),
  - shows the startup splash (`app.startup`),
  - loads `config.json` and `deck.json` (warning for each that could not be read and was kept aside),
  - ensures the library directory exists,
  - checks `library/VERSION`,
  - checks the on-disk library version through the migration adapter and stops on a mismatch,
  - runs the startup tasks (imports, JIT, FFmpeg, search readers and keys) and builds `AppWindow`, then shows it and closes the splash.
- `AppWindow` creates the shared `EventBus`, `DataModel`, `TimelineCoordinator`, `PlayerController`, `ExternalSyncController`, `SegmentReanalysisWorkflow`, and `MainPane`.
- UI state is mostly signal-driven. The loaded track's signals are the model's (`DataModel.sig_loaded`, `sig_beatgrid_changed`, ...), notifications only; listeners read current values from the shared `DataModel`.
