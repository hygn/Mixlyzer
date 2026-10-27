"""Checks on a real library: the user's library.db and {uid}.npz files, read only (SQLite opened read-only, the NPZ
files without pickle; nothing is written). The code holds no library data and the reports name no track: a
failing track is "track #k" (its place among the uids), its uid only when asked for locally.

Off by default (they read the whole library):

    MIXLYZER_LIBRARY_TESTS=1 uv run python -m unittest discover -s tests/library_data -t tests -p "test_*.py"

    MIXLYZER_LIBRARY=<folder>     the library to read (default: config.json's libconfig.libpath, else library/)
    MIXLYZER_TEST_SHOW_UIDS=1     name a failing track by its uid (local runs only: never paste these reports)
"""
