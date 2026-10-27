"""Unit tests that run anywhere: synthetic data only (no library, no audio files, no network, no audio device), so
they can be published with the code. A package per layer of the app, named apart from the app's packages (a test
package named ``system`` would shadow the app's when this folder is the start of discovery).

    uv run python -m unittest discover -s tests/unit -t tests -p "test_*.py"

The tests that need the user's library are ``tests/library_data``; the analyzer's ground truth on songs whose audio
the runner provides is ``tests/ground_truth``."""
