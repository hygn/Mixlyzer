# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

app_name = 'Mixlyzer'

block_cipher = None

icon_path = os.path.abspath(os.path.join(os.path.dirname(vars().get('__file__', 'build/mixlyzer.spec')), '..', 'assets', 'images', 'mixlyzer.ico'))

a = Analysis(
    ['../app/main.py'],
    pathex=['.'],
    binaries=[],
    # config.json is not bundled: the local one holds personal settings, and
    # load_cfg() writes one from default_cfg() on first launch.
    datas=[
        (
            '../assets/weights/phrase_analyzer.npz',
            'assets/weights',
        ),
        (
            '../assets/weights/default_refine_onset_hgb.npz',
            'assets/weights',
        ),
        (
            '../assets/weights/default_beat_tcn.npz',
            'assets/weights',
        ),
        (
            '../assets/weights/default_beat_critic.npz',
            'assets/weights',
        ),
        (
            '../assets/weights/default_beat_decoder.json',
            'assets/weights',
        ),
        (
            '../assets/weights/default_downbeat_ctx_hgb.npz',
            'assets/weights',
        ),
        (
            '../assets/styles/app.qss',
            'assets/styles',
        ),
        (
            '../assets/readings/unihan_taiwan_mandarin.tsv',
            'assets/readings',
        ),
        (
            '../assets/readings/unihan_korean.tsv',
            'assets/readings',
        ),
        (
            '../assets/readings/unihan_vietnamese.tsv',
            'assets/readings',
        ),
        (
            '../assets/readings/unihan_cantonese_variants.tsv',
            'assets/readings',
        ),
        (
            '../assets/readings/UNICODE-LICENSE.txt',
            'assets/readings',
        ),
    ]
    # the title readings (core.reading): the readers' dictionaries and tables
    + collect_data_files('sudachipy')
    + collect_data_files('sudachidict_core')
    + collect_data_files('anyascii')
    + collect_data_files('pypinyin')
    + collect_data_files('pycantonese'),
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

# Optionally include ffmpeg.exe if present
#if os.path.exists('../ffmpeg.exe'):
#    a.datas.append(('../ffmpeg.exe', '.'))

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
    console=False,  # windowed app
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

