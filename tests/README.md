# Tests

The suites are grouped by what they need. All are `unittest`. Run them from the repository root.

To run every suite at once (the gated ones skip):

    uv run python -m unittest discover -s tests -p "test_*.py"

| Suite | Needs | Run by default | Command |
|---|---|---|---|
| `tests/test_*.py` | nothing beyond the dependencies; some tests use offscreen Qt, a temporary library, or localhost UDP | yes | `uv run python -m unittest discover -s tests -p "test_*.py"` |
| `tests/unit/` | nothing: synthetic data only | yes | `uv run python -m unittest discover -s tests/unit -t tests -p "test_*.py"` |
| `tests/library_data/` | a Mixlyzer library (read only) | no: `MIXLYZER_LIBRARY_TESTS=1` | `uv run python -m unittest discover -s tests/library_data -t tests -p "test_*.py"` |
| `tests/ground_truth/` | the songs' audio, which you buy yourself (below: Tests on real songs) | no: `MIXLYZER_GT_TESTS=1` | `uv run python -m unittest discover -s tests/ground_truth -t tests -p "test_*.py"` |

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

## Tests on real songs

Two tests need commercial recordings. The audio is never in the repository: buy the releases named below, then
point the tests to your files (`tests/ground_truth/audio.json` or `MIXLYZER_GT_AUDIO`, as above). A song without
audio is skipped, so you can run with only some of them.

| Test | What it checks | Songs (`id`: title / artist / release) |
|---|---|---|
| `ground_truth/test_analyzer.py` | the full analysis against hand annotations (`songs.json`): beats, downbeats, keys, phrases; Search by Music on the songs' own audio | `go-go-maniac`: GO! GO! MANIAC / 放課後ティータイム / けいおん!はいれぞ!「Come with Me!!」セット<br>`gabriel-dropkick`: ガヴリールドロップキック / ガヴリール(CV:富田美憂) ほか / TVアニメ「ガヴリールドロップアウト」オープニングテーマ「ガヴリールドロップキック」<br>`mainichi-koharubiyori`: 毎日コハルビヨリ / あおい(CV：井口裕香) ひなた(CV：阿澄佳奈) / 毎日コハルビヨリ<br>`fx-senshi-kurumi`: FX戦士くるみちゃん / 福賀くるみ(CV：鈴木愛奈) / TVアニメ「FX戦士くるみちゃん」オープニングテーマ&エンディングテーマ<br>`ashita-de-ii-kara`: 明日でいいから / Happy Clover / 明日でいいから<br>`shukusei-loli-kami`: 粛聖!! ロリ神レクイエム☆ / しぐれうい / まだ雨はやまない<br>`yumeji-labyrinth`: 夢路らびりんす / らびりんず / 夢路らびりんす<br>`menimeri-memories`: メニメリ・メモリーズ！ / シネフォト部 / メニメリ・メモリーズ！<br>`se-no`: せーのっ! / 情報処理部 / TVアニメ「ゆゆ式」オープニングテーマ「せーのっ!」<br>`yoimachi-cantare`: よいまちカンターレ / コーロまちカド / よいまちカンターレ - 町かどタンジェント |
| `ground_truth/test_variants.py` | the Variants on real versions (`variants.json`): each song's versions one group and no other song in it; two purchases of one master the same mix; the instrumental, the other vocals and the TV size told apart | 夢路らびりんす (TV anime "Urara Meirochou" opening; single 夢路らびりんす): the original, 千矢 / 紺 / 小梅 / ノノ Version, Instrumental, and a second purchase (ALAC) of the original and the instrumental<br>アイデン貞貞メルトダウン (TV anime "Oniichan wa Oshimai!" opening; single アイデン貞貞メルトダウン, and the single アイデン貞貞メルトダウン (まひろ&みはりver.)): the original, TV size, えなこパートver., P丸様。パートver., Instrumental, まひろ&みはりver.<br>ひめごと*クライシスターズ (its ending; single ひめごと*クライシスターズ): the original, TV size ver., まひろ / みはり / かえで / もみじ ver., Instrumental<br>Not a version of any: うららか日和の四重奏 (single 夢路らびりんす) |

**Where you buy them matters.** The tests were set with copies bought from OTOTOY and mora. Another store's copy of
the same release, or another product of one store (another format or resolution), can differ in two ways:

- **Time alignment:** the audio may start a little earlier or later (leading silence, an encoder's delay, another
  cut of the master), and the length may differ by a fraction of a second.
- **Mastering:** the level, the limiting and the high end may differ (another master for another store, a
  hi-res master beside a CD one).

How the tests take that:

- `test_analyzer.py` removes a constant offset of up to 60 ms before scoring and skips a file whose length differs
  from the annotated one by more than 1 s (another edit). A copy of another mastering scores against the same
  annotations, but a floor (`expect`) set on one master may sit a little high for another.
- `test_variants.py` needs no alignment: the fingerprint follows the beats. It skips a file whose length differs
  from `variants.json`'s by more than 2 s. A copy of another mastering may no longer be "the same mix" of another
  purchase (the two-purchases check); every other check only needs the versions' own differences to stand out.

The full analysis of a file takes about 20 s of CPU, at a lower priority than the other programs. `test_variants.py`
keeps each file's beats and overview in `cache/test_variants/` (git-ignored) until the file or the fingerprint's
method changes; delete that folder to analyze again.

The mixer's checks on the user's own tracks (`native_mixer/verify_mix_real.py`) also play real songs; they are
local only and name no songs.
