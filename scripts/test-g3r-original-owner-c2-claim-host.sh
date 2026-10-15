#!/usr/bin/env bash
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONDONTWRITEBYTECODE=1
bash "$root/scripts/test-g3r-original-owner-preclaim-host.sh"
python3 "$root/tests/g3r/test_original_owner_c2_claim_union_source.py"
(cd "$root" && python3 -m unittest -v tests.c2.test_handler_order)
printf 'G3R original-owner C2 claim host source PASS\n'
