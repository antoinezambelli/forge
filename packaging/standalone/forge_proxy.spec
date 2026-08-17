"""Single PyInstaller definition for all ruled native targets and both forms."""

import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, copy_metadata


ROOT = Path(SPECPATH).parents[1]
sys.path.insert(0, str(ROOT))

from scripts.standalone.inputs import COLLECT_PACKAGES, EXCLUDED_MODULES


form = os.environ["FORGE_STANDALONE_FORM"]
if form not in {"onedir", "onefile"}:
    raise ValueError(f"unsupported standalone form: {form}")

datas = copy_metadata("forge-guardrails")
binaries = []
hiddenimports = []
for package in COLLECT_PACKAGES:
    package_datas, package_binaries, package_hidden = collect_all(package)
    datas.extend(package_datas)
    binaries.extend(package_binaries)
    hiddenimports.extend(package_hidden)

analysis = Analysis(
    [str(ROOT / "src" / "forge" / "proxy" / "__main__.py")],
    pathex=[str(ROOT / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=list(EXCLUDED_MODULES),
    noarchive=False,
    optimize=0,
)
pyz = PYZ(analysis.pure)

if form == "onedir":
    executable = EXE(
        pyz,
        analysis.scripts,
        [],
        exclude_binaries=True,
        name="forge-proxy",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=True,
    )
    bundle = COLLECT(
        executable,
        analysis.binaries,
        analysis.datas,
        strip=False,
        upx=False,
        name="forge-proxy",
    )
else:
    executable = EXE(
        pyz,
        analysis.scripts,
        analysis.binaries,
        analysis.datas,
        [],
        name="forge-proxy",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=True,
    )
