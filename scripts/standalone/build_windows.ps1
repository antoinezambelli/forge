$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$BuildEnv = Join-Path $RepoRoot ".standalone-build-env"

py -3.14 -m venv --clear $BuildEnv
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& (Join-Path $BuildEnv "Scripts\python.exe") -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& (Join-Path $BuildEnv "Scripts\python.exe") -m pip install "$RepoRoot[anthropic]" pyinstaller
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& (Join-Path $BuildEnv "Scripts\python.exe") -m scripts.standalone.build `
    --target windows-x86_64 --form all
exit $LASTEXITCODE
