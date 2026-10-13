#!/usr/bin/env bash
# Source-only fixture launcher. Compiled execution remains a separate HOLD gate.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
usage() {
  echo "usage: $0 --check | --prepare FRESH_ABSOLUTE_OUTPUT | --verify-runtime PREPARED_DIR COMPILED_BINARY CORPUS_DIR FRESH_EVIDENCE_DIR (HOLD)" >&2
  exit 74
}
[[ $# == 1 && $1 == --check || $# == 2 && $1 == --prepare ||
   $# == 5 && $1 == --verify-runtime ]] || usage
cd "$root"
export PYTHONDONTWRITEBYTECODE=1
if [[ $1 == --check ]]; then
  python3 -c 'from tests.g3r.prepare_save_post_adoption_cut import validate_source; validate_source()'
  echo 'G3-R post-adoption cut source checks PASS; compiled runtime HOLD'
  exit 0
fi
if [[ $1 == --prepare ]]; then
  [[ $2 = /* && ! -e $2 && ! -L $2 ]] || usage
  python3 -m tests.g3r.prepare_save_post_adoption_cut --output "$2"
  echo "G3-R post-adoption cut prepared at $2; compiled runtime HOLD"
  exit 0
fi

# The accepted source fixture has no reviewed builder or trusted receipt.
# In particular, caller-selected executable/corpus paths cannot establish
# that either process came from the prepared 96-leaf source bundle.
echo 'HOLD: no reviewed capture build, corpus contract, or trusted receipt; runtime verification disabled' >&2
exit 74
