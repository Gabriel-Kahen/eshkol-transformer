#!/usr/bin/env bash
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONDONTWRITEBYTECODE=1
python3 "$root/tests/g3r/test_tr3_held_o2_snapshot_recheck_source.py"
bash "$root/scripts/test-g3r-o2-held-save-native.sh"
bash "$root/scripts/test-g3r-current-history-held-owner.sh"
bash "$root/scripts/test-g3r-trainer-g3-save-host.sh"
printf 'G3R held-O2 TR3 source union host PASS\n'
