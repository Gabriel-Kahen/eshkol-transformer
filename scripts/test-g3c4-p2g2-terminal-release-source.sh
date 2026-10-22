#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python3 "$root/scripts/check-g3c4-p2g2-terminal-release-source.py"
cd "$root"
GUILE_AUTO_COMPILE=0 guile --no-auto-compile -s \
  scripts/test-g3c4-p2g2-terminal-release-dead-authority.scm
