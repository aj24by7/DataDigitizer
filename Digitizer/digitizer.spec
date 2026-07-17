# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path


block_cipher = None
root = Path(SPECPATH)

datas = [(str(root / "version.json"), ".")]

# Tesseract OCR runtime bundled from Digitizer\vendor\tesseract (self-contained OCR).
# vendor/ is gitignored, so a fresh clone does not have it. Skipping it silently produced an
# exe with no OCR that still LOOKED fine to whoever built it -- _resolve_tesseract_cmd falls
# back to an installed C:\Program Files\Tesseract-OCR, so the gap only surfaced for whoever
# they handed the exe to. Say so loudly at build time instead.
vendor_tesseract = root / "vendor" / "tesseract"
if vendor_tesseract.exists() and any(path.is_file() for path in vendor_tesseract.rglob("*")):
    datas.append((str(vendor_tesseract), "vendor/tesseract"))
else:
    import warnings

    warnings.warn(
        "\n"
        "    ****************************************************************\n"
        "    WARNING: vendor/tesseract not found -- building an exe with NO OCR.\n"
        "    Axis Scale Detection will not work for anyone you give this exe to.\n"
        "    It may still appear to work on THIS machine if Tesseract is installed\n"
        "    separately. See 'Before you build from a fresh clone' in the README.\n"
        "    ****************************************************************\n",
        stacklevel=1,
    )

hiddenimports = [
    "openpyxl",
    "pytesseract",
    "PIL.Image",
    "PIL.ImageEnhance",
    "PIL.ImageFilter",
    "PIL.ImageOps",
]

a = Analysis(
    ["digitizer_desktop.py"],
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
    name="Digitizer",
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
