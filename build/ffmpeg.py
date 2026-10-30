"""Builds the FFmpeg the package carries (the repository root's ffmpeg.exe / ffmpeg, which analyzer/audio/decoder.py
runs): FFmpeg's own code alone, under the GNU LGPL v2.1 or later, with nothing of another project linked into it
(--disable-autodetect, no external library: build/notices.py refuses a build that has one), and only what the app
asks of it (decoder.py): the audio formats the library takes (MP3, AAC / ALAC in MP4, FLAC, WAV / W64, AIFF,
Ogg Vorbis / Opus, WMA) decoded to 32-bit float PCM at another rate or channel count, their tags as ffmetadata, a
cover copied out as it is, and the start's warm-up (a short silence from lavfi to null).

    uv run python build/ffmpeg.py

FFmpeg's source is the commit FFMPEG_COMMIT (the one the app's decoder offsets were measured on), downloaded from
GitHub into build/_ffmpeg (git-ignored). Also needed:
- nasm (https://www.nasm.us: the x86 SIMD code, as the builds the decoder was checked with have it), its folder on
  PATH or named by NASM;
- Windows: llvm-mingw (as build/audio_engine.py: its bin folder on PATH or named by LLVM_MINGW; clang, the UCRT)
  and Git for Windows' bash and tools (configure is a shell script), make being llvm-mingw's mingw32-make;
- Linux / macOS: a C compiler and make.
The result is written over the repository root's FFmpeg; then run build/notices.py (its notice names this build)."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from system import os_support  # noqa: E402

FFMPEG_COMMIT = "dac2a1116d72423b577311e84c65cacf516e6aed"
# its version as git describe gives it (release/7.1), written into the source's VERSION: an archive has no history,
# and FFmpeg's version.sh would read the repository around it (this one) instead
FFMPEG_VERSION = "n7.1.3-19-gdac2a1116d"
SOURCE_URL = f"https://github.com/FFmpeg/FFmpeg/archive/{FFMPEG_COMMIT}.tar.gz"
WORK = ROOT / "build" / "_ffmpeg"
EXE = "ffmpeg.exe" if os_support.IS_WINDOWS else "ffmpeg"
EXTRA_VERSION = "mixlyzer"      # build/notices.py knows this build (its recipe: this file) by it

DEMUXERS = ("aac", "aiff", "asf", "flac", "mov", "mp3", "ogg", "w64", "wav")
# each format's decoder as a full build picks it (MP1 / MP2's fixed-point first, MP3's float), so the PCM is the same
DECODERS = ("aac", "alac", "flac", "mp1", "mp1float", "mp2", "mp2float", "mp3float", "opus", "vorbis",
            "wmav1", "wmav2", "wmapro", "wmalossless",
            "pcm_s8", "pcm_u8", "pcm_s16le", "pcm_s16be", "pcm_s24le", "pcm_s24be", "pcm_s32le", "pcm_s32be",
            "pcm_f32le", "pcm_f32be", "pcm_f64le", "pcm_f64be", "pcm_alaw", "pcm_mulaw",
            "adpcm_ima_wav", "adpcm_ms")
PARSERS = ("aac", "flac", "mpegaudio", "opus", "vorbis", "mjpeg", "png")
ENCODERS = ("pcm_f32le", "pcm_s16le")          # the PCM read; null's default (the warm-up)
MUXERS = ("pcm_f32le", "ffmetadata", "image2pipe", "null")     # pcm_f32le: the format f32le
FILTERS = ("aresample", "aformat", "anull", "atrim", "anullsrc")
PROTOCOLS = ("file", "pipe")
INDEVS = ("lavfi",)


def run(cmd: list, env: dict | None = None, cwd: Path | None = None) -> None:
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], env=env, cwd=cwd, check=True)


def configure_options() -> list[str]:
    def each(kind: str, names) -> list[str]:
        return [f"--enable-{kind}={n}" for n in names]
    return ["--disable-everything", "--disable-autodetect", "--disable-network", "--disable-doc",
            "--disable-debug", "--disable-ffplay", "--disable-ffprobe", "--disable-swscale",
            f"--extra-version={EXTRA_VERSION}",
            *each("demuxer", DEMUXERS), *each("decoder", DECODERS), *each("parser", PARSERS),
            *each("encoder", ENCODERS), *each("muxer", MUXERS), *each("filter", FILTERS),
            *each("protocol", PROTOCOLS), *each("indev", INDEVS)]


def tool_folder(variable: str, exe: str, hint: str) -> Path:
    named = os.environ.get(variable)
    for folder in ((Path(named), Path(named) / "bin") if named else ()):
        if (folder / exe).is_file():
            return folder
    found = shutil.which(exe)
    if found:
        return Path(found).parent
    sys.exit(f"{exe} not found: put its folder on PATH or set {variable} ({hint})")


def git_bash() -> Path:
    """Git for Windows' bash (not System32's, WSL's)."""
    git = shutil.which("git")
    if git:
        for up in (Path(git).parent.parent, Path(git).parent.parent.parent):
            if (up / "bin" / "bash.exe").is_file():
                return up / "bin" / "bash.exe"
    sys.exit("Git for Windows' bash not found (configure is a shell script)")


def source() -> Path:
    """FFmpeg's source at FFMPEG_COMMIT, downloaded once."""
    src = WORK / f"FFmpeg-{FFMPEG_COMMIT}"
    if not (src / "configure").is_file():
        WORK.mkdir(parents=True, exist_ok=True)
        archive = WORK / f"{FFMPEG_COMMIT}.tar.gz"
        print(f"downloading {SOURCE_URL}", flush=True)
        with urllib.request.urlopen(SOURCE_URL, timeout=120) as resp, open(archive, "wb") as out:
            shutil.copyfileobj(resp, out)
        with tarfile.open(archive) as tar:
            tar.extractall(WORK, filter="data")
        archive.unlink()
    (src / "VERSION").write_text(FFMPEG_VERSION + "\n", encoding="ascii")
    return src


def main() -> None:
    src = source()
    nasm = tool_folder("NASM", "nasm.exe" if os_support.IS_WINDOWS else "nasm", "https://www.nasm.us")
    env = dict(os.environ)
    options = configure_options()
    jobs = str(max(1, (os.cpu_count() or 2) - 2))
    if os_support.IS_WINDOWS:
        llvm = tool_folder("LLVM_MINGW", "x86_64-w64-mingw32-clang.exe", "llvm-mingw")
        bash = git_bash()
        env["PATH"] = os.pathsep.join((str(llvm), str(nasm), str(bash.parent.parent / "usr" / "bin"), env["PATH"]))
        options += ["--cc=clang", "--cxx=clang++", "--ar=llvm-ar", "--nm=llvm-nm", "--ranlib=llvm-ranlib",
                    "--strip=llvm-strip", "--windres=llvm-windres", "--target-os=mingw32", "--arch=x86_64"]
        quoted = " ".join(f"'{o}'" for o in options)
        run([bash, "-c", f"./configure {quoted} && mingw32-make -j{jobs}"], env, cwd=src)
    else:
        env["PATH"] = str(nasm) + os.pathsep + env["PATH"]
        run(["./configure", *options], env, cwd=src)
        run(["make", f"-j{jobs}"], env, cwd=src)
    built = src / EXE
    shutil.copy2(built, ROOT / EXE)
    print(f"built {ROOT / EXE} ({(ROOT / EXE).stat().st_size / 1e6:.1f} MB); run build/notices.py next")


if __name__ == "__main__":
    main()
