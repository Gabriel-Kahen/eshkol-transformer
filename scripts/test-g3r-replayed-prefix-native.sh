#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
out="$(mktemp -d "${TMPDIR:-/tmp}/g3r-replayed-prefix.XXXXXX")"
trap 'rm -rf -- "$out"' EXIT
cc="${HOST_CC:-cc}"
common=(-std=c11 -Wall -Wextra -Werror -Wpedantic
  -Wno-misleading-indentation -Wno-unused-parameter
  -I "$root/include" -I "$root/native" -I "$root/src"
  -I "$root/src/eshkol_transformer")
features=(-DET_G3R_REPLAYED_PREFIX_PRIVATE
  -DET_G3R_COMMITTED_HISTORY_PRIVATE
  -DET_G3T_MANUAL_DECODE_PRIVATE -DET_G3T_MANUAL_P2_PREFILL_PRIVATE
  -DET_G3T_MANUAL_P1_PREFILL_PRIVATE -DET_G3T_MANUAL_LOGITS_PRIVATE
  -DET_G3T_OWNED_TOKEN_INPUT_PRIVATE -DET_G3T_P2_ZERO_BUDGET_PRIVATE
  -DET_G3T_ZERO_BUDGET_PRIVATE -DET_G3T_OUTPUT_TEXT_PRIVATE
  -DET_G3T_FINAL_PUBLICATION_PRIVATE -DET_G3T_INPUT_FROM_T1_PRIVATE
  -DET_G3T_PREFILL_SAMPLE_PRIVATE
  -DET_I64_TENSOR_STORAGE_QUERY_PRIVATE
  -DET_A2_KV_CACHE_STORAGE_QUERY_PRIVATE)
"$cc" "${common[@]}" -c "$root/src/eshkol_transformer/g3t_transport.c" \
  -o "$out/off.o"
if nm -g --defined-only "$out/off.o" | \
    rg -q 'et_g3t_private_replayed_prefix_call_acquire_v1'; then
  echo 'G3-R replayed-prefix ABI escaped feature-off object' >&2
  exit 1
fi
"$cc" "${common[@]}" "${features[@]}" \
  -c "$root/src/eshkol_transformer/g3t_transport.c" -o "$out/on.o"
nm -g --defined-only "$out/on.o" | \
  rg -q 'et_g3t_private_replayed_prefix_call_acquire_v1'
for mode in normal sanitize; do
  flags=(-O0)
  runtime=(env)
  if [[ "$mode" == sanitize ]]; then
    flags=(-O0 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
    runtime=(env ASAN_OPTIONS=detect_leaks="${G3R_LSAN:-0}":halt_on_error=1
      UBSAN_OPTIONS=halt_on_error=1)
  fi
  "$cc" "${common[@]}" "${flags[@]}" -ffunction-sections \
    -fdata-sections "$root/tests/g3r/replayed_prefix_transition_test.c" \
    -Wl,--gc-sections -o "$out/$mode"
  "${runtime[@]}" "$out/$mode" >"$out/$mode.stdout"
done
cmp "$out/normal.stdout" "$out/sanitize.stdout"
cat "$out/normal.stdout"
echo 'G3-R replayed-prefix native feature on/off and sanitizer PASS'
