#!/usr/bin/env bash
# Source-only fixture launcher. Compiled execution remains a separate HOLD gate.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
usage() {
  echo "usage: $0 --check | --prepare FRESH_ABSOLUTE_OUTPUT | --verify-runtime PREPARED_DIR COMPILED_BINARY CORPUS_DIR FRESH_EVIDENCE_DIR" >&2
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

# Only a separately reviewed compiler may supply the binary. This verifier
# runs two isolated processes and checks the production abandon exit status.
[[ "${G3R_POST_ADOPTION_RUNTIME_RELEASE:-0}" == 1 ]] || {
  echo 'HOLD: explicit post-adoption runtime release is required' >&2; exit 74;
}
[[ "${ESHKOL_ALLOW_UNSUPPORTED_HOST:-0}" != 1 ]] || {
  echo 'HOLD: unsupported-host compatibility override is not a gate' >&2; exit 74;
}
source "$root/scripts/common.sh"
verify_toolchain
for required in sha256sum timeout grep; do require_command "$required"; done
prepared=$2 binary=$3 corpus=$4 evidence=$5
[[ $prepared = /* && -f $prepared/PREPARED.tsv &&
   -f $prepared/SOURCE-SHA256SUMS && $binary = /* && -x $binary &&
   $corpus = /* && -d $corpus && $evidence = /* &&
   ! -e $evidence && ! -L $evidence ]] || usage
python3 -c 'from tests.g3r.prepare_save_post_adoption_cut import validate_source; validate_source()'
(cd "$prepared" && sha256sum --check --status SOURCE-SHA256SUMS)
[[ "$(tsv_value "$prepared/PREPARED.tsv" source_commit)" == "$(git rev-parse HEAD)" &&
   "$(tsv_value "$prepared/PREPARED.tsv" source_tree)" == "$(git rev-parse 'HEAD^{tree}')" &&
   "$(tsv_value "$prepared/PREPARED.tsv" status)" == 'source-prepared; compiled gate not run' ]] || {
  echo 'prepared fixture does not match this source commit/tree' >&2; exit 74;
}
mkdir -p "$evidence"
printf '%s\n' "$(sha256sum "$binary")" >"$evidence/binary.sha256"
for mode in success fail-stop; do
  args=("$corpus")
  [[ $mode == success ]] || args+=(fail-stop)
  status=0
  ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM --kill-after=5s \
    120s "$binary" "${args[@]}" >"$evidence/$mode.stdout" \
    2>"$evidence/$mode.stderr" || status=$?
  printf '%s\n' "$status" >"$evidence/$mode.status"
done
[[ "$(cat "$evidence/success.status")" == 0 &&
   "$(cat "$evidence/fail-stop.status")" == 134 &&
   ! -s "$evidence/success.stderr" && ! -s "$evidence/fail-stop.stderr" ]] || {
  echo 'post-adoption runtime status/stderr check failed' >&2; exit 1;
}
grep -Eq '^G3-R post-adoption cut PASS: checks=[1-9][0-9]*$' \
  "$evidence/success.stdout"
grep -Fxq 'G3-R post-adoption O2 release fault injected' \
  "$evidence/fail-stop.stdout"
if grep -Eq 'RECOVERED ERROR|RETURNED|cut PASS' "$evidence/fail-stop.stdout"; then
  echo 'post-adoption release failure recovered or returned' >&2; exit 1;
fi
echo "G3-R post-adoption isolated runtime checks PASS: $evidence"
