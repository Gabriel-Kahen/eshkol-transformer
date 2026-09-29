#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
for command in cmp git grep nm python3 sha256sum; do require_command "$command"; done
cc= cxx=
resolve_provenance_compilers cc cxx \
  "${CC:-$(lock_value supported_cc)}" "${CXX:-$(lock_value supported_cxx)}"
evidence="${1:-$(project_build_dir)/g3c4-p2g2-first-frame-test}"
mkdir -p "$evidence"
python3 "$PROJECT_ROOT/scripts/check-g3c4-p2g2-first-frame.py" \
  >"$evidence/static.stdout"

flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -fstack-protector-all
       -ffp-contract=off -fexcess-precision=standard -fno-fast-math
       -I "$PROJECT_ROOT/include" -I "$PROJECT_ROOT/native"
       -I "$PROJECT_ROOT/src" -I "$PROJECT_ROOT/src/eshkol_transformer")
test_macros=(-DET_I64_TENSOR_TESTING -DET_I64_TENSOR_STORAGE_QUERY_PRIVATE
             -DET_A2_KV_CACHE_TESTING -DET_A2_KV_CACHE_STORAGE_QUERY_PRIVATE)
owner_macros=(-DET_G3C4_CONTEXT_PRIVATE -DET_G3C4_NATIVE_PINS_PRIVATE
  -DET_G3C4_ACTIVE_CALL_PRIVATE -DET_G3C4_GENERATOR_PRIVATE
  -DET_G3C4_PROMPT_T1_BORROW_PRIVATE -DET_G3C4_PROVIDER_ROUTES_PRIVATE
  -DET_G3C4_FULL_PREFIX_FORWARD_PRIVATE -DET_G3C4_SAMPLER_TRANSPORT_PRIVATE
  -DET_G3C4_TOKEN_FRAME_PRIVATE -DET_G3C4_TOKEN_FORWARD_PRIVATE
  -DET_G3C4_PREFILL3_PRIVATE -DET_G3C4_PREFILL1_PRIVATE
  -DET_G3C4_PREFILL2_PRIVATE -DET_G3C4_PROMPT_PREFILL_PRIVATE
  -DET_G3C4_OUTPUT_RESERVATION_PRIVATE -DET_G3C4_LAST_LOGIT_FRAME_PRIVATE
  -DET_G3C4_OUTPUT_PREPARE_PRIVATE -DET_I64_TENSOR_STORAGE_QUERY_PRIVATE
  -DET_A2_KV_CACHE_STORAGE_QUERY_PRIVATE -DET_G3C4_OUTPUT_DECODE_IDS_PRIVATE
  -DET_G3C4_OUTPUT_TEXT_PRIVATE -DET_G3C4_P2_G1_PENDING_PRIVATE)
native_sources=(
  "$PROJECT_ROOT/src/eshkol_transformer/m3_i64_integration.c"
  "$PROJECT_ROOT/tests/m3t/kernel_fail_allocator.c"
  "$PROJECT_ROOT/native/t1_i64_shell.c"
  "$PROJECT_ROOT/native/n3k_primitives_provider.c"
  "$PROJECT_ROOT/native/g3c4_primitives_provider.c"
  "$PROJECT_ROOT/native/g3s_sampling_provider.c"
  "$PROJECT_ROOT/native/g3n_primitives_provider.c"
  "$PROJECT_ROOT/native/n2_primitives_provider.c"
  "$PROJECT_ROOT/native/a2_kv_cache.c")

tmp="$(mktemp -d "$evidence/build.XXXXXX")"
cleanup() {
  local status=$?
  if (( status == 0 )); then rm -rf -- "$tmp"; else
    printf 'G3-C4 P2/G2 first-frame failure retained: %s\n' "$tmp" >&2
  fi
}
trap cleanup EXIT

for mode in off on; do
  macros=("${owner_macros[@]}")
  if [[ "$mode" == on ]]; then
    macros+=(-DET_G3C4_P2_G2_FIRST_FRAME_PRIVATE)
  fi
  "$cc" "${flags[@]}" -O2 "${macros[@]}" \
    -c "$PROJECT_ROOT/src/eshkol_transformer/g3c4_model_owner.c" \
    -o "$tmp/$mode.o"
  nm -g --defined-only --format=posix "$tmp/$mode.o" |
    awk 'NF >= 2 {print $1}' | LC_ALL=C sort -u >"$tmp/$mode.symbols"
done
cmp "$tmp/off.symbols" "$tmp/on.symbols"
if "$cc" "${flags[@]}" -O2 -DET_G3C4_P2_G2_FIRST_FRAME_PRIVATE \
    -c "$PROJECT_ROOT/src/eshkol_transformer/g3c4_model_owner.c" \
    -o "$tmp/invalid.o" >"$tmp/invalid.stdout" \
    2>"$tmp/invalid.stderr"; then
  printf 'isolated P2/G2 feature macro compiled\n' >&2; exit 1
fi
grep -F 'C4 P2/G2 first frame requires pending output and token forward' \
  "$tmp/invalid.stderr" >/dev/null
printf 'G3-C4 P2/G2 first-frame symbols and feature closure PASS\n' \
  >"$evidence/symbols.stdout"

"$cc" "${flags[@]}" -O2 "${test_macros[@]}" \
  "$PROJECT_ROOT/tests/g3c4/test_p2_g1_pending.c" \
  "${native_sources[@]}" -Wl,--wrap=et_kernel_runtime_dispatch \
  -Wl,--wrap=et_i64_tensor_borrow_view_v1 -lm -o "$tmp/feature-off"
"$tmp/feature-off" >"$evidence/feature-off.stdout" \
  2>"$evidence/feature-off.stderr"
test ! -s "$evidence/feature-off.stderr"
grep -E '^G3-C4 P2/G1 pending PASS: checks=[1-9][0-9]*$' \
  "$evidence/feature-off.stdout" >/dev/null

for mode in normal sanitize; do
  optimization=(-O2)
  runtime=(env)
  if [[ "$mode" == sanitize ]]; then
    optimization=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
    runtime=(env ASAN_OPTIONS=detect_leaks=1:halt_on_error=1
                 UBSAN_OPTIONS=halt_on_error=1)
  fi
  "$cc" "${flags[@]}" "${optimization[@]}" "${test_macros[@]}" \
    "$PROJECT_ROOT/tests/g3c4/test_p2_g2_first_frame.c" \
    "${native_sources[@]}" -Wl,--wrap=et_kernel_runtime_dispatch \
    -Wl,--wrap=et_i64_tensor_borrow_view_v1 -lm -o "$tmp/$mode"
  "${runtime[@]}" "$tmp/$mode" >"$evidence/$mode.stdout" \
    2>"$evidence/$mode.stderr"
  test ! -s "$evidence/$mode.stderr"
  grep -E '^G3-C4 P2/G2 first frame PASS: checks=[1-9][0-9]*$' \
    "$evidence/$mode.stdout" >/dev/null
done
python3 "$PROJECT_ROOT/tests/g3c4/check_p2_g1_oracle.py" \
  "$evidence/normal.stdout" >"$evidence/oracle.stdout"
"$tmp/normal" >"$evidence/repeat.stdout" 2>"$evidence/repeat.stderr"
test ! -s "$evidence/repeat.stderr"
cmp "$evidence/normal.stdout" "$evidence/repeat.stdout"
cmp "$evidence/normal.stdout" "$evidence/sanitize.stdout"
printf 'G3-C4 P2/G2 first-frame normal/repeat/sanitizer PASS\n' \
  >"$evidence/gate.stdout"
