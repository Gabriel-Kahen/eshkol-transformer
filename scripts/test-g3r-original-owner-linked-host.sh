#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONDONTWRITEBYTECODE=1
bash "$root/scripts/test-g3r-original-owner-c2-claim-host.sh"
python3 "$root/tests/g3r/test_original_owner_linked_source.py"
bash -n "$root/scripts/test-g3r-original-owner-linked.sh"
printf 'G3R linked owner host source PASS; compiled gate pending\n'
