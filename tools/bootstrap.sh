#!/bin/bash

set -e

# OS-dependent bootstrap script for initializing the build system.
# All it does is pulling a python interpreter, and then we can use python to do workspace tasks like building, testing, formatting, etc.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS="$ROOT/_tools"
BIN="$TOOLS/bin"
PYS="$TOOLS/python"
CACHE="$TOOLS/cache/uv"
VENV="$TOOLS/venv"

mkdir -p "$BIN" "$PYS" "$CACHE"

UV="$BIN/uv"
if [[ ! -f "$UV" ]]; then
    export UV_INSTALL_DIR="$BIN"
    export UV_NO_MODIFY_PATH="1"
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

export UV_CACHE_DIR="$CACHE"
export UV_PYTHON_INSTALL_DIR="$PYS"
export UV_MANAGED_PYTHON="1"

"$UV" python install
if [[ ! -d "$VENV" ]]; then
    "$UV" venv "$VENV"
fi

PY="$VENV/bin/python"
REQUIREMENTS="$ROOT/tools/requirements.txt"
"$UV" pip install --python "$PY" -r "$REQUIREMENTS"

echo "OK: $VENV"
