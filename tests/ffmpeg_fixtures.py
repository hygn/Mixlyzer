"""The FFmpegs the tests run: the app's (the repository root's, built by build/ffmpeg.py to decode only) to read
files as the app does, and one that makes the test files (encoders, lavfi's sources, a PNG): MIXLYZER_TEST_FFMPEG,
else an FFmpeg on PATH that has them (None: the tests that need one are skipped)."""
import os
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP_FFMPEG = next((p for p in (ROOT / "ffmpeg.exe", ROOT / "ffmpeg") if p.is_file()), None) or shutil.which("ffmpeg")
NEEDED = {"encoders": (" aac ", " libmp3lame ", " png "), "filters": (" sine ", " color ")}


def _makes_files(exe: str) -> bool:
    for listing, names in NEEDED.items():
        try:
            out = subprocess.run([exe, "-hide_banner", f"-{listing}"], capture_output=True, timeout=30).stdout
        except (OSError, subprocess.TimeoutExpired):
            return False
        text = out.decode("utf-8", "replace")
        if not all(name in text for name in names):
            return False
    return True


@lru_cache(maxsize=1)
def maker() -> str | None:
    """The FFmpeg that makes the test files, or None."""
    named = os.environ.get("MIXLYZER_TEST_FFMPEG")
    for exe in (named, shutil.which("ffmpeg"), str(APP_FFMPEG) if APP_FFMPEG else None):
        if exe and _makes_files(exe):
            return exe
    return None


MAKER = maker()
