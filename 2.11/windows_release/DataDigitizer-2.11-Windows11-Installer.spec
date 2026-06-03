# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path


block_cipher = None
root = Path(SPECPATH).parent

datas = [
    (str(root / "dist" / "digitizer.exe"), "payload"),
    (str(root / "dist" / "accuracytester.exe"), "payload"),
    (str(root / "dist" / "DataDigitizer-2.11.exe"), "payload"),
]

a = Analysis(
    [str(root / "windows_release" / "gui_installer.py")],
    pathex=[str(root)],
    binaries=[],
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="DataDigitizer-2.11-Windows11-Installer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(root / "assets" / "digitizer.ico"),
)
