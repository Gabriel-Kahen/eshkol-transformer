#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -B scripts/check-g3c4-p2g2-terminal-anchor.py
python3 -B scripts/check-g3c4-p2g2-terminal-first-source.py
