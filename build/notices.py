"""The third-party notices shipped with Mixlyzer (assets/notices/THIRD_PARTY_NOTICES.txt, shown in Help > About):
every work the package carries that is not Mixlyzer's own, with its licence and the texts its licence asks to go
with it. Run it again when a dependency changes (as build/translations.py when the texts do):

    uv run python build/notices.py

Three parts, each in a fixed order so a run without changes writes the same file:
  1. what is bundled beside the packages: the Python runtime, Qt (in PySide6), libsndfile (in soundfile), the native
     engine's VOLK with cpu_features, its Rust standard library and its toolchain's runtime (llvm-mingw), FFmpeg
     (the repository root's, an LGPL build: one built with --enable-gpl or --enable-nonfree is refused; its version,
     source, build recipe and configuration from its -version), the fonts, the Unicode data the readings tables come
     from, the Cantonese data pycantonese reads (the spec ships only those);
  2. the Python packages the app runs on (uv export, no development groups), by pip-licenses with their licence and
     notice files;
  3. the native engine's Rust crates on Windows, Linux and macOS (cargo metadata of each, offline: the crates as
     fetched for the last build), each with the licence files it ships; a text seen before is named, not repeated.
     A crate shipping none gets its licence's standard text (the Rust toolchain's) with its authors.
Needs the Rust toolchain stable-x86_64-pc-windows-gnullvm (build/audio_engine.py) and, for the toolchain's runtime
notice, llvm-mingw (LLVM_MINGW or its bin folder on PATH)."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from system import os_support  # noqa: E402

OUT = ROOT / "assets" / "notices" / "THIRD_PARTY_NOTICES.txt"
# the FFmpeg the package carries (the spec bundles it into _internal, where analyzer/audio/decoder.py looks): an LGPL
# build only
FFMPEG = ROOT / ("ffmpeg.exe" if os_support.IS_WINDOWS else "ffmpeg")
FFMPEG_REFUSED = ("--enable-gpl", "--enable-nonfree")
# configure options that enable no library of their own (a licence, the threads, the compiler, the system's TLS)
FFMPEG_NOT_LIBRARIES = {"version3", "pthreads", "cuda-llvm", "schannel"}
ENGINE = ROOT / "audioengine" / "native"
TOOLCHAIN = "stable-x86_64-pc-windows-gnullvm"
TARGETS = ("x86_64-pc-windows-gnullvm", "x86_64-unknown-linux-gnu", "aarch64-apple-darwin", "x86_64-apple-darwin")
LICENSE_FILES = ("LICENSE", "LICENCE", "COPYING", "NOTICE", "COPYRIGHT", "UNLICENSE")
RULE = "=" * 79


def section(title: str) -> str:
    return f"\n{RULE}\n{title}\n{RULE}\n"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace").strip() + "\n"


def run(cmd: list[str], cwd: Path | None = None) -> str:
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True, encoding="utf-8").stdout


def site_packages() -> Path:
    import sysconfig
    return Path(sysconfig.get_paths()["purelib"])


def rust_licenses() -> Path:
    rustc = run(["rustup", "which", "rustc", "--toolchain", TOOLCHAIN]).strip()
    return Path(rustc).parent.parent / "share" / "doc" / "rust" / "licenses"


def llvm_mingw() -> Path | None:
    named = os.environ.get("LLVM_MINGW")
    if named and (Path(named) / "LICENSE.TXT").is_file():
        return Path(named)
    found = shutil.which("x86_64-w64-mingw32-clang")
    return Path(found).parent.parent if found else None


def ffmpeg_build(version_text: str) -> tuple[str, str, list[str]]:
    """An FFmpeg's version, compiler and configure options from its ``-version`` output; refused (SystemExit) when
    it was built with --enable-gpl or --enable-nonfree: the package ships an LGPL build only."""
    lines = version_text.splitlines()
    version = lines[0].split()[2]                        # ffmpeg version <version> Copyright ...
    compiler = next((l[len("built with "):] for l in lines if l.startswith("built with ")), "")
    options = next(l for l in lines if l.startswith("configuration:")).split()[1:]
    refused = [o for o in options if o in FFMPEG_REFUSED]
    if refused:
        sys.exit(f"FFmpeg {version} is built with {' '.join(refused)}: the package ships an LGPL build only")
    return version, compiler, options


def ffmpeg_version_text(path: Path = FFMPEG) -> str:
    return run([str(path), "-hide_banner", "-version"])


def ffmpeg_entry() -> tuple[str, str, str, list[Path]] | None:
    """The FFmpeg at the repository's root as a part 1 entry (None when there is none): its source the commit its
    version names, its recipe the build's (BtbN's FFmpeg-Builds, known by its prefix; else FFMPEG_RECIPE), the
    libraries its configuration enables named."""
    if not FFMPEG.is_file():
        print(f"no {FFMPEG.name} at the repository's root: FFmpeg's notice left out", file=sys.stderr)
        return None
    version, compiler, options = ffmpeg_build(ffmpeg_version_text())
    commit = re.search(r"-g([0-9a-f]{7,40})", version)
    source = (f"https://github.com/FFmpeg/FFmpeg/commit/{commit.group(1)}" if commit
              else f"https://github.com/FFmpeg/FFmpeg/tree/{version}")
    recipe = os.environ.get("FFMPEG_RECIPE")
    if not recipe and "--prefix=/ffbuild/prefix" in options:
        recipe = "https://github.com/BtbN/FFmpeg-Builds"
    if not recipe:
        sys.exit(f"{FFMPEG.name}: its build's recipe is not known: name it in FFMPEG_RECIPE (a URL)")
    v3 = "--enable-version3" in options
    licence = "LGPL-3.0-or-later" if v3 else "LGPL-2.1-or-later"
    text = ("the GNU LGPL v3 given with VOLK above, and the GNU GPL v3 it supplements at this part's end" if v3
            else "the GNU LGPL v2.1 given with libsndfile above")
    libraries = sorted({o[len("--enable-"):] for o in options if o.startswith("--enable-")} - FFMPEG_NOT_LIBRARIES)
    note = (f"The audio decoder ({FFMPEG.name} beside the program): a program of its own the app runs, which can "
            f"be replaced by the user. Built without --enable-gpl and --enable-nonfree; its licence's text: {text}. "
            f"Its source: {source}; its build's recipe: {recipe}"
            + (f" (compiled with {compiler})" if compiler else "") + ".\n"
            f"Linked into it, each under its own licence (their sources and licences: the build's recipe): "
            f"{', '.join(libraries)}.\n"
            f"Its configuration: {' '.join(options)}")
    return f"FFmpeg {version}", licence, note, []


def bundled() -> str:
    """Part 1: what the package carries beside its Python packages and crates."""
    import PySide6
    import soundfile
    site = site_packages()
    volk = ENGINE / "third_party" / "volk-src"
    entries = [
        ("Python", "PSF-2.0", "The Python runtime the package is built with.",
         [Path(sys.base_prefix) / "LICENSE.txt"]),
        (f"Qt {PySide6.__version__}, PySide6 and Shiboken6", "LGPL-3.0-only",
         "Used under the GNU LGPL v3 (its text with VOLK's below; the GNU GPL v3 it supplements at this part's end): "
         "their wheels carry only a reference to Qt's commercial licence, not the LGPL's text. The libraries are "
         "loaded as shared libraries (Qt*.dll / .so / .framework beside the program) and can be replaced by the "
         "user. Qt's source: https://download.qt.io/official_releases/qt/; PySide6's and Shiboken6's: "
         "https://code.qt.io/cgit/pyside/pyside-setup.git", []),
        (f"libsndfile {soundfile.__libsndfile_version__}", "LGPL-2.1-or-later",
         "In soundfile (_soundfile_data), loaded as a shared library that can be replaced. Its source: "
         "https://github.com/libsndfile/libsndfile", [site / "_soundfile_data" / "COPYING"]),
        ("VOLK 3.3.0 (Vector-Optimized Library of Kernels)", "LGPL-3.0-or-later",
         "In the native audio engine (audioengine/native/bin), linked as a shared library (volk.dll / libvolk.so / "
         "libvolk.dylib beside the engine) that can be replaced. Its source: https://github.com/gnuradio/volk "
         "(tag v3.3.0); the engine's source and its build (build/audio_engine.py): "
         "https://github.com/hygn/Mixlyzer", [volk / "COPYING"]),
        ("cpu_features", "Apache-2.0", "In VOLK.", [volk / "cpu_features" / "LICENSE"]),
        ("The Rust standard library", "MIT OR Apache-2.0", "In the native audio engine.",
         [rust_licenses() / "MIT.txt", rust_licenses() / "Apache-2.0.txt"]),
    ]
    ffmpeg = ffmpeg_entry()
    if ffmpeg is not None:
        entries.append(ffmpeg)
    entries += [
        ("Noto Sans, Noto Sans JP, Noto Sans KR, Noto Sans SC, Noto Sans TC", "OFL-1.1",
         "The interface's fonts (assets/fonts).", [ROOT / "assets" / "fonts" / "OFL.txt"]),
        ("Unicode Character Database (Unihan)", "Unicode-3.0",
         "The readings tables (assets/readings) are made from it.",
         [ROOT / "assets" / "readings" / "UNICODE-LICENSE.txt"]),
        ("Hong Kong Cantonese Corpus (HKCanCor)", "CC BY 4.0",
         "Data in pycantonese, read for Cantonese readings. Luke, Kang-Kwong and May L.Y. Wong. 2015. The Hong Kong "
         "Cantonese Corpus: Design and Uses. Journal of Chinese Linguistics.",
         [site / "pycantonese" / "data" / "hkcancor" / "LICENSE.txt"]),
        ("rime-cantonese", "CC BY 4.0", "Data in pycantonese, read for Cantonese readings. "
         "https://github.com/rime/rime-cantonese", [site / "pycantonese" / "data" / "rime_cantonese" / "LICENSE.txt"]),
    ]
    lgpl3 = "Qt, PySide6, Shiboken6, VOLK" + (", FFmpeg" if ffmpeg is not None and "LGPL-3" in ffmpeg[1] else "")
    entries.append(("The GNU General Public License v3", "GPL-3.0",
                    f"The licence the GNU LGPL v3 ({lgpl3}; and Mixlyzer itself) supplements.",
                    [rust_licenses() / "GPL-3.0-or-later.txt"]))
    toolchain = llvm_mingw()
    if toolchain is not None:
        entries.append(("LLVM and mingw-w64 runtimes (llvm-mingw)", "Apache-2.0 WITH LLVM-exception, and others",
                        "The Windows native engine's runtime, from its toolchain.", [toolchain / "LICENSE.TXT"]))
    else:
        print("llvm-mingw not found: its runtime's notice left out", file=sys.stderr)
    out = [section("1. Bundled with the program")]
    for name, licence, note, files in entries:
        out.append(f"\n--- {name}\nLicence: {licence}\n{note}\n")
        for f in files:
            if not f.is_file():
                sys.exit(f"missing: {f}")
            out.append("\n" + read(f))
    return "".join(out)


def python_packages() -> str:
    """Part 2: the packages the app runs on, by pip-licenses."""
    lines = run(["uv", "export", "--no-dev", "--no-hashes", "--no-emit-project", "--no-header"], cwd=ROOT)
    names = sorted({line.split("==")[0].strip() for line in lines.splitlines()
                    if "==" in line and not line.startswith(("#", " "))}, key=str.lower)
    found = json.loads(run([sys.executable, "-m", "piplicenses", "--packages", *names, "--with-license-file",
                            "--with-notice-file", "--no-license-path", "--with-urls", "--format=json"], cwd=ROOT))
    out = [section("2. Python packages")]
    for p in sorted(found, key=lambda p: p["Name"].lower()):
        out.append(f"\n--- {p['Name']} {p['Version']}\nLicence: {p['License']}\n")
        if p.get("URL") and p["URL"] != "UNKNOWN":
            out.append(f"Source: {p['URL']}\n")
        for kind in ("LicenseText", "NoticeText"):
            text = (p.get(kind) or "").strip()
            if text and text != "UNKNOWN":
                out.append(f"\n{text}\n")
    return "".join(out)


def crates() -> str:
    """Part 3: the native engine's Rust crates."""
    found: dict[str, dict] = {}
    for target in TARGETS:
        meta = json.loads(run(["cargo", f"+{TOOLCHAIN}", "metadata", "--format-version", "1", "--locked", "--offline",
                               "--filter-platform", target], cwd=ENGINE))
        ws = set(meta["workspace_members"])
        used = {n["id"] for n in meta["resolve"]["nodes"]}
        for p in meta["packages"]:
            if p["id"] in used and p["id"] not in ws:
                found[p["id"]] = p
    standard = rust_licenses()
    seen: dict[str, str] = {}      # a text's hash -> the crate it was first given with
    out = [section("3. Rust crates (the native audio engine)")]
    for p in sorted(found.values(), key=lambda p: (p["name"], p["version"])):
        name = f"{p['name']} {p['version']}"
        out.append(f"\n--- {name}\nLicence: {p.get('license') or 'see below'}\n")
        if p.get("repository"):
            out.append(f"Source: {p['repository']}\n")
        folder = Path(p["manifest_path"]).parent
        files = sorted(f for f in folder.iterdir() if f.is_file() and f.name.upper().startswith(LICENSE_FILES))
        if not files:
            authors = ", ".join(p.get("authors") or []) or "its authors"
            out.append(f"No licence file in the crate. Copyright (c) {authors}.\n")
            files = [standard / "MIT.txt", standard / "Apache-2.0.txt"] if "MIT" in (p.get("license") or "") \
                or "Apache" in (p.get("license") or "") else []
        for f in files:
            text = read(f)
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if digest in seen:
                out.append(f"{f.name}: the same text as given with {seen[digest]} above.\n")
            else:
                seen[digest] = name
                out.append(f"\n{f.name}:\n\n{text}")
    return "".join(out)


def main() -> None:
    head = ("Mixlyzer: third-party notices\n\n"
            "Mixlyzer is distributed under the GNU Lesser General Public License v3.0 (its own licence: Help > About). "
            "It carries the works below, each under its own licence; their texts follow. Made by build/notices.py.\n")
    text = head + bundled() + python_packages() + crates()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"{OUT.relative_to(ROOT)}: {len(text):,} characters")


if __name__ == "__main__":
    main()
