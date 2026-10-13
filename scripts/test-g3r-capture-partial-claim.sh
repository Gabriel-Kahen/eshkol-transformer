#!/usr/bin/env bash
# Source preparation only. No compiled gate is admitted while E3 holds /tmp.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
case "${1:-}" in
  --check)
    [[ $# -eq 1 ]] || exit 64
    PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v \
      tests.g3r.test_capture_partial_claim_cut_source
    ;;
  --prepare)
    [[ $# -eq 2 ]] || exit 64
    PYTHONDONTWRITEBYTECODE=1 python3 -m \
      tests.g3r.prepare_capture_partial_claim_cut --output "$2"
    ;;
  --run)
    echo "HOLD: independent source review and an approved exact native union are required before compiled execution" >&2
    exit 74
    ;;
  *)
    echo "usage: $0 --check | --prepare FRESH_ABSOLUTE_OUTPUT | --run" >&2
    exit 64
    ;;
esac
