#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
eshkol_source=${ESHKOL_SOURCE_DIR:-"$root/.deps/eshkol-src"}
cc=${CC:-clang}
test -f "$eshkol_source/inc/eshkol/eshkol.h"

# Production CLI3 owner flags, then the trained-pair union's private delta.
base=(
  ET_CLI3_E3_G3T_AGGREGATE_BUILD
  ET_G3T_PREFILL_SAMPLE_PRIVATE
  ET_G3T_OUTPUT_TEXT_PRIVATE
  ET_G3T_FINAL_PUBLICATION_PRIVATE
  ET_G3T_FULL_REQUEST_PREFLIGHT_PRIVATE
  ET_G3T_OUTPUT_IDS_CLONE_PRIVATE
  ET_G3T_OUTPUT_LENGTHS_CLONE_PRIVATE
  ET_G3T_OUTPUT_CACHE_LENGTHS_CLONE_PRIVATE
  ET_G3T_OUTPUT_RNG_CLONE_PRIVATE
  ET_G3T_GENERATOR_RNG_PRIVATE
  ET_G3T_INPUT_FROM_T1_PRIVATE
  ET_G3T_OWNED_TOKEN_INPUT_PRIVATE
  ET_I64_TENSOR_STORAGE_QUERY_PRIVATE
  ET_A2_KV_CACHE_STORAGE_QUERY_PRIVATE
)
delta=(
  ET_G3T_ZERO_BUDGET_PRIVATE
  ET_G3T_P2_ZERO_BUDGET_PRIVATE
  ET_G3T_MANUAL_LOGITS_PRIVATE
  ET_G3T_MANUAL_P1_PREFILL_PRIVATE
  ET_G3T_MANUAL_P2_PREFILL_PRIVATE
  ET_G3T_MANUAL_DECODE_PRIVATE
  ET_G3T_MANUAL_LOGITS_MATERIALIZE_PRIVATE
  ET_G3R_COMMITTED_HISTORY_PRIVATE
  ET_G3R_COMMITTED_SAVE_SNAPSHOT_PRIVATE
  ET_F32_TENSOR_STORAGE_QUERY_PRIVATE
)
test "${#base[@]}" -eq 14
test "${#delta[@]}" -eq 10

common=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic
  -fstack-protector-all -fPIC -fvisibility=hidden -fno-common
  -ffp-contract=off -fexcess-precision=standard -frounding-math -fno-fast-math
  -I "$root/include" -I "$root/native" -I "$root/src"
  -I "$eshkol_source/inc" -fsyntax-only
)
owner="$root/native/cli3_generate_native_owner.c"
compile() {
  local flags=() flag
  for flag in "$@"; do flags+=("-D$flag"); done
  "$cc" "${common[@]}" "${flags[@]}" "$owner"
}

compile "${base[@]}"
echo 'PASS: production CLI3 owner syntax'
compile "${base[@]}" "${delta[@]}"
echo 'PASS: trained-pair owner without replay flag'
compile "${base[@]}" "${delta[@]}" ET_G3R_REPLAYED_P1_G1_PRIVATE
echo 'PASS: trained-pair owner with replay flag'

without_p2=()
for flag in "${delta[@]}"; do
  if [[ "$flag" != ET_G3T_MANUAL_P2_PREFILL_PRIVATE ]]; then
    without_p2+=("$flag")
  fi
done
stderr=$(mktemp)
trap 'rm -f "$stderr"' EXIT
if compile "${base[@]}" "${without_p2[@]}" 2>"$stderr"; then
  echo 'FAIL: missing manual P2 dependency compiled' >&2
  exit 1
fi
grep -Fq 'G3-R history requires final G3-T P1/P2 and manual decode commits' "$stderr"
echo 'PASS: missing manual P2 dependency rejected'
