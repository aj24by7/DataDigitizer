# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path


block_cipher = None
root = Path(SPECPATH)
repo_root = root.parent

datas = [(str(root / "version.json"), ".")]

vendor_tesseract = repo_root / "vendor" / "tesseract"
if vendor_tesseract.exists() and any(path.is_file() for path in vendor_tesseract.rglob("*")):
    datas.append((str(vendor_tesseract), "vendor/tesseract"))

hiddenimports = [
    "openpyxl",
    "pytesseract",
    "PIL.Image",
    "PIL.ImageEnhance",
    "PIL.ImageFilter",
    "PIL.ImageOps",
]

a = Analysis(
    ["digitizer_2_11.py"],
    pathex=[str(root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
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
    name="DataDigitizer-2.11",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
