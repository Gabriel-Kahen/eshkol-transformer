#!/usr/bin/env bash
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONDONTWRITEBYTECODE=1
python3 "$root/tests/g3r/test_original_owner_coordinator_source.py"
python3 "$root/scripts/check-g3r-held-o2-adopter.py"
python3 "$root/tests/g3r/test_original_owner_preclaim_boundary_source.py"
bash "$root/scripts/test-g3r-o2-tr3-recheck-union-host.sh"
printf 'G3R original-owner preclaim host PASS\n'
