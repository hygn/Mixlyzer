# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_submodules

app_name = 'Mixlyzer'

block_cipher = None

icon_path = os.path.abspath(os.path.join(os.path.dirname(vars().get('__file__', 'build/mixlyzer.spec')), '..', 'assets', 'images', 'mixlyzer.ico'))

a = Analysis(
    ['../app/main.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        ('../config.json', '.'),
    ],
    hiddenimports=[],
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

