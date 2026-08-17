#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
build_env="$repo_root/.standalone-build-env"

python3.14 -m venv "$build_env"
"$build_env/bin/python" -m pip install --upgrade pip
"$build_env/bin/python" -m pip install "$repo_root[anthropic]" pyinstaller
"$build_env/bin/python" -m scripts.standalone.build \
  --target macos-arm64 --form all
