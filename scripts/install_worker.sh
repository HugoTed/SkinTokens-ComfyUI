#!/usr/bin/env bash
set -euo pipefail

PLUGIN_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PLUGIN_ROOT"

BOOTSTRAP="$PLUGIN_ROOT/bootstrap.py"
PYTHON=""

for candidate in "${TOKENRIG_PYTHON:-}" python3.11 python3 python; do
  if [[ -z "$candidate" ]]; then
    continue
  fi
  if command -v "$candidate" >/dev/null 2>&1; then
    PYTHON="$candidate"
    break
  fi
done

if [[ -z "$PYTHON" ]]; then
  echo "[TokenRig] No Python interpreter found. Install Python 3.11+ or set TOKENRIG_PYTHON." >&2
  exit 1
fi

echo "[TokenRig] Installing worker in $PLUGIN_ROOT using $PYTHON"
"$PYTHON" "$BOOTSTRAP"
echo "[TokenRig] Worker install complete."
