# Mixlyzer

Languages: [한국어](docs/README.ko.md) | [日本語](docs/README.ja.md)

## About

Mixlyzer is a track-analysis and editing companion built for DJs and producers who want precise control over beatgrids, keys, and cues. It scans each song, highlights tempo changes, detects key modulations, proposes JumpCUE pairs for seamless jumps, and lets you polish everything through an interactive editor before exporting to external platforms.

## Features

- **Static Beatgrid Analysis** - Detect a track’s global BPM and phase to generate a solid starting grid.  
- **Dynamic Beatgrid Analysis** - Split songs into multiple segments so shifting BPM and phase changes are mapped accurately.  
- **Dynamic Key Analysis** - Trace musical key changes over time so mixing stays in key.  
- **JumpCUE Analysis** - Find compatible cue pairs for clean forward/backward jumps or creative mashups.  
- **Fast, Interactive Editing** - Segment-based beatgrid tools, partial reanalysis, and undo/redo so you can move quickly.  
- **Export** - Rekordbox XML export with corrected tempo segments, downbeats, and JumpCUE entries.

## Made For

- DJs who play energetic, rhythmically complex tracks without wanting drifting grids.  
- DJs from animation OST, doujin, game-music, Vocaloid, VTuber, and other otaku scenes that often involve tempo/key shifts and jumps.  
- Producers/remixers who need reliable analysis data before rebuilding a track.

## Getting Started

### Option A: Using uv (recommended for development)
1. [Install uv](https://github.com/astral-sh/uv) if it’s not already on your system.  
2. Clone this repository and open the project folder.  
3. Run `uv sync` to install dependencies and set up the virtual environment (Python 3.12+).  
4. Launch Mixlyzer with `uv run python -m app.main`.

#### Building the native audio engine (optional)
Without it Mixlyzer plays on its Python engine. The native engine is developed and tested on Windows; on Linux and macOS it is experimental (it builds there, but has not been tested on them yet). To build it into `audioengine/native/bin`:
1. Install git and [Rust](https://rustup.rs), plus a C compiler:
   - **Windows**: [llvm-mingw](https://github.com/mstorsjo/llvm-mingw) (no Visual Studio needed; its `bin` folder on `PATH`, or the folder named by `LLVM_MINGW`) and the Rust toolchain `rustup toolchain install stable-x86_64-pc-windows-gnullvm`.
   - **Linux**: gcc or clang, and the ALSA and JACK headers (`libasound2-dev libjack-jackd2-dev` / `alsa-lib-devel pipewire-jack-audio-connection-kit-devel`).
   - **macOS**: Xcode's command line tools.
2. Run `uv sync --group native`, then `uv run --group native python build/audio_engine.py` (it builds VOLK from its source, then the engine). Close Mixlyzer first when rebuilding: a running app holds the engine.

Settings > Playback picks the engine (**Native first**, which falls back to Python, or **Python**; applied at the next start), the output device, and each engine's buffer size. The native engine's audio API: **WASAPI** (Windows: a low-latency shared stream, or exclusive mode with a sample rate of its own, applied from the next track loaded), **WASAPI (cpal)**, **ALSA** / **JACK** (Linux) or **CoreAudio** (macOS). Under the buffer meter, the measured response (a command to its sound) and the device's latency. The native engine's priority (High by default; Windows: the process's priority class, Linux: its threads' real-time scheduling or nice, through rtkit where the user may not set them, macOS: a real-time thread) and power throttling (off by default; Windows: EcoQoS, Linux: timer slack and uclamp, macOS: App Nap and timer coalescing) can be set there too; what the engine got is shown under the meter.

### Option B: Using a released binary
1. Download the latest Mixlyzer build from the Releases page.  
2. Extract it to your preferred location.  
3. Start the executable (e.g., `Mixlyzer.exe` on Windows).  
4. Set your music library path on first launch, then start analyzing and exporting right away.  

## Typical Workflow

1. **Load a track** - Drag a file into Mixlyzer or pick one from the library pane.  
2. **Let the analyzer work** - Tempo segments, keys, and JumpCUE pairs are detected automatically.  
3. **Fine-tune the grid** - Use the Beatgrid editor to adjust BPM, shift segments, or reanalyze specific sections.  
4. **Export** - Save your edits and export a Rekordbox XML so the refined beatgrid travels with you.  
