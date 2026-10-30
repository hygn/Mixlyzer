# -*- mode: python ; coding: utf-8 -*-
import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

app_name = 'Mixlyzer'

block_cipher = None

icon_path = os.path.abspath(os.path.join(os.path.dirname(vars().get('__file__', 'build/mixlyzer.spec')), '..', 'assets', 'images', 'mixlyzer.ico'))
REPO = os.path.abspath(os.path.join(SPECPATH, '..'))


def shipped(folder, keep=lambda name: True):
    """(file, its folder in the app) for the files under ``folder`` (repo-relative) that ``keep`` accepts."""
    out = []
    for root, _dirs, files in os.walk(os.path.join(REPO, folder)):
        for name in files:
            if keep(name):
                out.append((os.path.join(root, name), os.path.relpath(root, REPO)))
    return out


def ffmpeg():
    """The repository root's FFmpeg into _internal (analyzer/audio/decoder.py looks there): FFmpeg's own code under
    the LGPL (build/ffmpeg.py; one built with --enable-gpl or --enable-nonfree, or with another project's library, is
    refused), the one the notices name (build/notices.py)."""
    sys.path.insert(0, SPECPATH)
    import notices
    if not notices.FFMPEG.is_file():
        print(f"no {notices.FFMPEG.name} at the repository's root: the package leaves FFmpeg out (the decoder then "
              "takes the one on PATH)")
        return []
    version, _compiler, _options = notices.ffmpeg_build(notices.ffmpeg_version_text())
    if f"--- FFmpeg {version}\n" not in notices.OUT.read_text(encoding='utf-8'):
        raise SystemExit(f"the notices do not name FFmpeg {version}: run build/notices.py again")
    return [(str(notices.FFMPEG), '.')]


a = Analysis(
    ['../app/main.py'],
    pathex=['.'],
    binaries=[],
    # The shipped assets, found through system.resource_paths: not the user's files beside them (a retrained model
    # without default_, assets/shortcuts/shortcuts.json, assets/midi/midi.json). config.json is not bundled: the local one holds personal
    # settings, and load_cfg() writes one from default_cfg() on first launch.
    datas=shipped('assets/weights', lambda name: name.startswith('default_'))
    # the images the app shows (not the design sources: the Photoshop file, the large render)
    + shipped('assets/icons') + shipped('assets/images', lambda name: not name.endswith('.psd')
                                        and name != 'mixlyzer_large.png')
    + shipped('assets/fonts') + shipped('assets/sound')
    + shipped('assets/styles') + shipped('assets/readings')
    # the third-party notices Help > About shows (build/notices.py)
    + shipped('assets/notices')
    # the UI's languages (system.i18n): the list and each one's catalog
    + shipped('assets/i18n', lambda name: name in ('languages.json', 'mixlyzer.po'))
    + shipped('assets/shortcuts', lambda name: name == 'default_shortcuts.json')
    + shipped('assets/midi', lambda name: name == 'default_midi.json')
    # the artist names kept whole (core.tieups.web), not the user's list beside them
    + shipped('assets/artists', lambda name: name.startswith('default_'))
    # the native audio engine and its VOLK (build/audio_engine.py), as built: copied, not analysed or packed
    + shipped('audioengine/native/bin')
    + ffmpeg()
    # the title readings (core.search.reading): the readers' dictionaries and tables
    + collect_data_files('sudachipy')
    + collect_data_files('sudachidict_core')
    + collect_data_files('anyascii')
    + collect_data_files('pypinyin')
    # pycantonese's data the readings read (HKCanCor, rime-cantonese), not its other corpora (CantoMap: GPL-3.0,
    # Common Voice: MPL-2.0, CTCPC), which nothing reads: the notices name what is shipped
    + collect_data_files('pycantonese', excludes=['data/cantomap', 'data/common_voice', 'data/ctcpc']),
    # Sudachi imports its dictionary package by name (dict="core")
    hiddenimports=['sudachidict_core'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

# Include assets/ recursively
#from PyInstaller.building.datastruct import Tree
#a.datas += Tree('assets', prefix='assets')

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=app_name,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    # windowed app; MIXLYZER_CONSOLE=1 builds one with a console (its output and errors readable, for testing)
    console=os.environ.get('MIXLYZER_CONSOLE') == '1',
    icon=icon_path,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name=app_name,
)

