#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/test-g3c4-p2g2-terminal-second-source.sh
python3 -B scripts/check-g3c4-p2g2-terminal-text-preflight-source.py
