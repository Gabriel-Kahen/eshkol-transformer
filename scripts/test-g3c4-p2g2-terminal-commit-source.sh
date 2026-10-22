#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/test-g3c4-p2g2-terminal-precommit-frame-source.sh
python3 -B scripts/check-g3c4-p2g2-terminal-commit-source.py
if [[ -n "${ESHKOL_SOURCE_REPO:-}" ]]; then
  python3 -B scripts/check-g3c4-p2g2-terminal-null-ffi-source.py \
    --eshkol-repo "$ESHKOL_SOURCE_REPO"
fi
