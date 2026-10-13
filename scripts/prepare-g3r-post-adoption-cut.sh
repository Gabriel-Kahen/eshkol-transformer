#!/usr/bin/env bash
# Source-only fixture launcher. Compiled execution remains a separate HOLD gate.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
usage() {
  echo "usage: $0 --check | --prepare FRESH_ABSOLUTE_OUTPUT" >&2
  exit 74
}
[[ $# == 1 && $1 == --check || $# == 2 && $1 == --prepare ]] || usage
cd "$root"
export PYTHONDONTWRITEBYTECODE=1
if [[ $1 == --check ]]; then
  python3 -c 'from tests.g3r.prepare_save_post_adoption_cut import validate_source; validate_source()'
  echo 'G3-R post-adoption cut source checks PASS; compiled runtime HOLD'
  exit 0
fi
[[ $2 = /* && ! -e $2 && ! -L $2 ]] || usage
python3 -m tests.g3r.prepare_save_post_adoption_cut --output "$2"
echo "G3-R post-adoption cut prepared at $2; compiled runtime HOLD"
