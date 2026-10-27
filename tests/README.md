# Tests

The suites are grouped by what they need. All are `unittest`. Run them from the repository root.

To run every suite at once (the gated ones skip):

    uv run python -m unittest discover -s tests -p "test_*.py"

| Suite | Needs | Run by default | Command |
|---|---|---|---|
| `tests/test_*.py` | nothing beyond the dependencies; some tests use offscreen Qt, a temporary library, or localhost UDP | yes | `uv run python -m unittest discover -s tests -p "test_*.py"` |
| `tests/unit/` | nothing: synthetic data only | yes | `uv run python -m unittest discover -s tests/unit -t tests -p "test_*.py"` |
| `tests/library_data/` | a Mixlyzer library (read only) | no: `MIXLYZER_LIBRARY_TESTS=1` | `uv run python -m unittest discover -s tests/library_data -t tests -p "test_*.py"` |
| `tests/ground_truth/` | the songs' audio, which you get yourself | no: `MIXLYZER_GT_TESTS=1` | `uv run python -m unittest discover -s tests/ground_truth -t tests -p "test_*.py"` |

Some of the `tests/test_*.py` tests skip when something optional is missing:

- the native engine (`build/audio_engine.py`): `test_native_engine.py`, `test_native_mixer.py`
- FFmpeg: `test_tag_writer.py`'s MP4 cases
- Babel (`uv sync --group dev`): `test_i18n.py`'s catalog check

The mixer's checks on the user's tracks and the engine live on the user's devices (`native_mixer/`) stay local:
they read files and play on outputs only the user has.

## `test_*.py`

These are the app's tests, each on a library or files of its own. They cover:

- the layer boundaries and the library's one owner, read from the source;
- the stored analysis, the track model, and the editor's save;
- tags, readings, tie-ups, and Variants;
- External Sync's OSC;
- the waveform's renderer;
- the native engine: its scratch path against the Python one's, its time stretch, its playback and malformed
  commands, and its mixer by the PCM it renders (the channel strip, each Sound Color FX and its place, the master,
  the booth, the headphones).

## `unit/`

These tests run anywhere. They need no library, no audio, no network and no audio device.

They are grouped by the app's layer: `system_tests`, `analyzer_tests`, `audioengine_tests` and `core_tests`. The
folder names differ from the app's packages, so a test package never shadows an app package.

Edge cases are covered in three ways:

- **Boundaries:** each type's limits, empty input, NaN, reversed spans, track ends.
- **Seeded random inputs:** each property is checked over thousands of cases, with fixed seeds so a failure
  reproduces.
- **Source cross-checks:** `test_ipc_layout.py` reads the native engine's `src/ipc.rs` and compares it with
  `client.py` and `controls.py`. It checks every offset, type and code.

A test marked `@unittest.expectedFailure` records a known bug. When the bug is fixed, the test reports an
unexpected success: remove the marker then.

## `library_data/`

These tests check that a real library is consistent:

- every uid has its stored analysis;
- the stored files read without pickle;
- the index tables match the stored analyses;
- the marks lie within the track;
- a save without edits changes nothing.

The library is only read: SQLite is opened read-only and nothing is written. Reports name a track only by its
place, as `track #k`. The uid appears only with `MIXLYZER_TEST_SHOW_UIDS=1`. Never paste such a report
anywhere public. `MIXLYZER_LIBRARY=<folder>` picks the library; the default is `config.json`'s.

## `ground_truth/`

`songs.json` holds a few songs' annotations: beats, tempo rows and phrases. It also holds each song's title,
artist and release, so you can find and buy the same release. The audio is not included.

Keys are taken only from a human source, never from a library: a library's keys are mostly the analyzer's own
output. A song's `keys` are written by hand with their `source`, and its key score is checked only where they exist.

Tell the tests where your files are, in one of two ways:

- **`tests/ground_truth/audio.json`:** `{"<id>": "<path>"}`. This file is git-ignored.
- **`MIXLYZER_GT_AUDIO=<folder>`:** a folder holding files named `<id>.<ext>`.

A file whose length differs from the annotated one is a different edit, and it is skipped. A song without audio is
also skipped. A constant offset of up to 60 ms is removed before scoring; an MP3's encoder delay is such an offset.

To add a song from your library, run `tests/ground_truth/make_fixture.py <id> <uid>`. Then set its floors
(`expect`) from a run with `MIXLYZER_GT_REPORT=1`.
