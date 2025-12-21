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
