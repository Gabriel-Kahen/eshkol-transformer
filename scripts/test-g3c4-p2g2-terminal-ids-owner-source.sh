#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python3 "$root/scripts/check-g3c4-p2g2-terminal-ids-owner-source.py"
