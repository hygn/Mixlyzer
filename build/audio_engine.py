"""Builds the native audio engine (audioengine/native) for this system into audioengine/native/bin: VOLK
(LGPL-3.0-or-later: the SIMD kernels, picked at run time for the CPU) from its source, then the engine.

    uv sync --group native        (cmake, ninja, mako for VOLK's generated sources, pkgconf)
    uv run python build/audio_engine.py

Also needed: git, Rust (rustup), and per system
- Windows: llvm-mingw (clang + the UCRT, no Visual Studio; https://github.com/mstorsjo/llvm-mingw), its bin
  folder on PATH or named by LLVM_MINGW, and the Rust toolchain stable-x86_64-pc-windows-gnullvm
  (rustup toolchain install stable-x86_64-pc-windows-gnullvm). MinGW's GCC miscompiles AVX code (it does not
  align the stack for it), hence clang.
- Linux: gcc or clang, and ALSA's headers for the audio output (libasound2-dev / alsa-lib-devel).
- macOS: Xcode's command line tools.
The engine finds its VOLK beside it (bin/: Windows' DLL search, an rpath of $ORIGIN / @executable_path).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core import os_support  # noqa: E402

NATIVE = ROOT / "audioengine" / "native"
THIRD = NATIVE / "third_party"
BIN = NATIVE / "bin"
VOLK_URL = "https://github.com/gnuradio/volk.git"
VOLK_TAG = "v3.3.0"
WINDOWS_TOOLCHAIN = "stable-x86_64-pc-windows-gnullvm"


def run(cmd: list, env: dict | None = None, cwd: Path | None = None) -> None:
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], env=env, cwd=cwd, check=True)


def python_tool(name: str) -> str:
    """A tool installed with the build's Python (the native dependency group), else from PATH."""
    scripts = Path(sys.executable).parent
    for candidate in (scripts / name, scripts / f"{name}.exe"):
        if candidate.is_file():
            return str(candidate)
    found = shutil.which(name)
    if not found:
        sys.exit(f"{name} not found: uv sync --group native")
    return found


def llvm_mingw_bin() -> Path:
    named = os.environ.get("LLVM_MINGW")
    if named:
        path = Path(named) / "bin"
        if (path / "x86_64-w64-mingw32-clang.exe").is_file():
            return path
    found = shutil.which("x86_64-w64-mingw32-clang")
    if found:
        return Path(found).parent
    sys.exit("llvm-mingw not found: put its bin folder on PATH or set LLVM_MINGW to its folder")


def build_env() -> dict:
    env = dict(os.environ)
    if os_support.IS_WINDOWS:
        env["PATH"] = str(llvm_mingw_bin()) + os.pathsep + env["PATH"]
    tools = Path(python_tool("cmake")).parent
    env["PATH"] = str(tools) + os.pathsep + env["PATH"]
    return env


def build_volk(env: dict) -> Path:
    """VOLK's source (git, at VOLK_TAG, with its cpu_features), built and installed under third_party."""
    src, build, install = THIRD / "volk-src", THIRD / "volk-build", THIRD / "volk-install"
    stamp = install / "VERSION"
    if stamp.is_file() and stamp.read_text().strip() == VOLK_TAG:
        return install
    if not (src / "CMakeLists.txt").is_file():
        THIRD.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--depth", "1", "--branch", VOLK_TAG, "--recurse-submodules", "--shallow-submodules",
             VOLK_URL, src])
    configure = [
        python_tool("cmake"), "-S", src, "-B", build, "-G", "Ninja",
        f"-DCMAKE_MAKE_PROGRAM={python_tool('ninja')}",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_INSTALL_PREFIX={install}",
        f"-DPYTHON_EXECUTABLE={sys.executable}",
        "-DENABLE_TESTING=OFF", "-DENABLE_MODTOOL=OFF", "-DENABLE_STATIC_LIBS=OFF",
    ]
    if os_support.IS_WINDOWS:
        configure += ["-DCMAKE_C_COMPILER=clang", "-DCMAKE_CXX_COMPILER=clang++"]
    if os_support.IS_MACOS:
        configure += ["-DCMAKE_INSTALL_NAME_DIR=@rpath", "-DCMAKE_MACOSX_RPATH=ON"]
    run(configure, env)
    run([python_tool("cmake"), "--build", build, "--parallel", str(max(1, (os.cpu_count() or 2) - 2))], env)
    run([python_tool("cmake"), "--install", build], env)
    stamp.write_text(VOLK_TAG)
    return install


def build_engine(env: dict, volk: Path) -> Path:
    pkgconfig = [str(p) for p in (volk / "lib" / "pkgconfig", volk / "lib64" / "pkgconfig") if p.is_dir()]
    env = dict(env)
    env["PKG_CONFIG"] = python_tool("pkgconf")
    env["PKG_CONFIG_PATH"] = os.pathsep.join(pkgconfig)
    env["CARGO_TARGET_DIR"] = str(NATIVE / "target")
    cargo = ["cargo"] + ([f"+{WINDOWS_TOOLCHAIN}"] if os_support.IS_WINDOWS else [])
    # from the crate's folder: cargo reads .cargo/config.toml (the linker, crt-static, the rpath) from the current
    # folder, not the manifest's
    run(cargo + ["build", "--release"], env, cwd=NATIVE)
    name = "mixlyzer-audio.exe" if os_support.IS_WINDOWS else "mixlyzer-audio"
    return NATIVE / "target" / "release" / name


def volk_library(volk: Path) -> Path:
    """The shared library the engine loads, by the name it loads it by."""
    if os_support.IS_WINDOWS:
        return volk / "bin" / "libvolk.dll"
    libs = [p for d in ("lib", "lib64") for p in (volk / d).glob("libvolk*") if p.is_file() or p.is_symlink()]
    pattern = r"libvolk\.\d+\.\d+\.dylib$" if os_support.IS_MACOS else r"libvolk\.so\.\d+\.\d+$"
    for p in libs:
        if re.match(pattern, p.name):
            return p
    sys.exit(f"VOLK's shared library not found in {volk}")


def main() -> None:
    env = build_env()
    volk = build_volk(env)
    exe = build_engine(env, volk)
    BIN.mkdir(parents=True, exist_ok=True)
    shutil.copy2(exe, BIN / exe.name)
    lib = volk_library(volk)
    shutil.copy2(lib, BIN / lib.name, follow_symlinks=True)
    # VOLK's licence beside the library it covers (LGPL-3.0: the library is replaceable, as it is a file)
    shutil.copy2(THIRD / "volk-src" / "COPYING", BIN / "volk-COPYING.txt")
    print(f"built into {BIN}: {', '.join(sorted(p.name for p in BIN.iterdir()))}")


if __name__ == "__main__":
    main()
