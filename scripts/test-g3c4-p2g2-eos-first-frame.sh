#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
cc= cxx=
resolve_provenance_compilers cc cxx \
  "${CC:-$(lock_value supported_cc)}" "${CXX:-$(lock_value supported_cxx)}"
evidence="${1:-$(project_build_dir)/g3c4-p2g2-eos-first-frame-test}"
mkdir -p "$evidence"
tmp="$(mktemp -d "$evidence/build.XXXXXX")"
python3 "$PROJECT_ROOT/scripts/check-g3c4-p2g2-eos-first-frame.py" \
  --baseline-dir "$tmp/baseline" >"$evidence/static.stdout"
bash "$PROJECT_ROOT/scripts/test-g3c4-p2g2-carrier-bridge.sh" \
  "$evidence/carrier-predecessor" >"$evidence/predecessor.stdout"

flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -fstack-protector-all
  -ffp-contract=off -fexcess-precision=standard -fno-fast-math
  -I "$PROJECT_ROOT/include" -I "$PROJECT_ROOT/native"
  -I "$PROJECT_ROOT/src" -I "$PROJECT_ROOT/src/eshkol_transformer"
  -DET_I64_TENSOR_TESTING -DET_I64_TENSOR_STORAGE_QUERY_PRIVATE
  -DET_F32_TENSOR_STORAGE_QUERY_PRIVATE
  -DET_A2_KV_CACHE_TESTING -DET_A2_KV_CACHE_STORAGE_QUERY_PRIVATE)
owner_macros=(-DET_G3C4_CONTEXT_PRIVATE -DET_G3C4_NATIVE_PINS_PRIVATE
  -DET_G3C4_ACTIVE_CALL_PRIVATE -DET_G3C4_GENERATOR_PRIVATE
  -DET_G3C4_PROMPT_T1_BORROW_PRIVATE -DET_G3C4_PROVIDER_ROUTES_PRIVATE
  -DET_G3C4_FULL_PREFIX_FORWARD_PRIVATE -DET_G3C4_SAMPLER_TRANSPORT_PRIVATE
  -DET_G3C4_TOKEN_FRAME_PRIVATE -DET_G3C4_TOKEN_FORWARD_PRIVATE
  -DET_G3C4_PREFILL3_PRIVATE -DET_G3C4_PREFILL1_PRIVATE
  -DET_G3C4_PREFILL2_PRIVATE -DET_G3C4_PROMPT_PREFILL_PRIVATE
  -DET_G3C4_OUTPUT_RESERVATION_PRIVATE -DET_G3C4_LAST_LOGIT_FRAME_PRIVATE
  -DET_G3C4_OUTPUT_PREPARE_PRIVATE -DET_G3C4_OUTPUT_DECODE_IDS_PRIVATE
  -DET_G3C4_OUTPUT_TEXT_PRIVATE -DET_G3C4_P2_G1_PENDING_PRIVATE
  -DET_G3C4_P2_G2_FIRST_FRAME_PRIVATE -DET_G3C4_P2_G2_PREFIX_COMMIT_PRIVATE
  -DET_G3C4_P2_G2_CARRIER_BRIDGE_PRIVATE)
sources=("$PROJECT_ROOT/tests/g3c4/test_p2_g2_eos_first_frame.c"
  "$PROJECT_ROOT/src/eshkol_transformer/m3_i64_integration.c"
  "$PROJECT_ROOT/tests/m3t/kernel_fail_allocator.c"
  "$PROJECT_ROOT/native/t1_i64_shell.c"
  "$PROJECT_ROOT/native/n3k_primitives_provider.c"
  "$PROJECT_ROOT/native/g3c4_primitives_provider.c"
  "$PROJECT_ROOT/native/g3s_sampling_provider.c"
  "$PROJECT_ROOT/native/g3n_primitives_provider.c"
  "$PROJECT_ROOT/native/n2_primitives_provider.c"
  "$PROJECT_ROOT/native/a2_kv_cache.c")
for mode in off on; do
  macros=("${owner_macros[@]}")
  if [[ "$mode" == on ]]; then
    macros+=(-DET_G3C4_P2_G2_EOS_FIRST_FRAME_PRIVATE)
  fi
  "$cc" "${flags[@]}" -O2 "${macros[@]}" -c \
    "$PROJECT_ROOT/src/eshkol_transformer/g3c4_model_owner.c" -o "$tmp/$mode.o"
  nm -g --defined-only --format=posix "$tmp/$mode.o" |
    awk 'NF >= 2 {print $1}' | LC_ALL=C sort -u >"$tmp/$mode.symbols"
done
comm -3 "$tmp/off.symbols" "$tmp/on.symbols" >"$tmp/symbols.diff"
printf '\tet_g3c4_private_p2g2_eos_first_frame_carrier_v1\n' \
  >"$tmp/expected.symbols.diff"
cmp "$tmp/expected.symbols.diff" "$tmp/symbols.diff"
"$cc" "${flags[@]}" "${owner_macros[@]}" -E -P \
  "$PROJECT_ROOT/src/eshkol_transformer/g3c4_model_owner.c" >"$tmp/off.preprocessed"
"$cc" "${flags[@]}" "${owner_macros[@]}" -E -P \
  "$tmp/baseline/g3c4_model_owner.c" >"$tmp/baseline.preprocessed"
cmp "$tmp/off.preprocessed" "$tmp/baseline.preprocessed"
if "$cc" "${flags[@]}" -DET_G3C4_P2_G2_EOS_FIRST_FRAME_PRIVATE -c \
    "$PROJECT_ROOT/src/eshkol_transformer/g3c4_model_owner.c" \
    -o "$tmp/isolated.o" >"$tmp/isolated.stdout" 2>"$tmp/isolated.stderr"; then
  printf 'isolated EOS first-frame feature compiled\n' >&2; exit 1
fi
grep -F 'C4 P2/G2 EOS first frame requires the authenticated carrier bridge' \
  "$tmp/isolated.stderr" >/dev/null
printf 'EOS private symbol, macro closure and feature-off preprocessor identity PASS\n' \
  >"$evidence/boundary.stdout"
for mode in normal sanitize; do
  optimization=(-O2)
  runtime=(env)
  if [[ "$mode" == sanitize ]]; then
    optimization=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
    runtime=(env ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1)
  fi
  "$cc" "${flags[@]}" "${optimization[@]}" "${sources[@]}" \
    -Wl,--wrap=et_kernel_runtime_dispatch -Wl,--wrap=et_i64_tensor_borrow_view_v1 \
    -lm -o "$tmp/$mode" >"$evidence/$mode.compile.stdout" \
    2>"$evidence/$mode.compile.stderr"
  test ! -s "$evidence/$mode.compile.stderr"
  "${runtime[@]}" "$tmp/$mode" >"$evidence/$mode.stdout" 2>"$evidence/$mode.stderr"
  test ! -s "$evidence/$mode.stderr"
  grep -E '^G3-C4 P2/G2 EOS first frame PASS: checks=[1-9][0-9]* pending-only=1$' \
    "$evidence/$mode.stdout" >/dev/null
done
"$tmp/normal" >"$evidence/repeat.stdout" 2>"$evidence/repeat.stderr"
test ! -s "$evidence/repeat.stderr"
cmp "$evidence/normal.stdout" "$evidence/repeat.stdout"
cmp "$evidence/normal.stdout" "$evidence/sanitize.stdout"
for pair in 0 1; do
  awk -v pair="$pair" '/^EOS-ORACLE-PAIR / {active=($2==pair)}
    active && /^ORACLE / {print}' "$evidence/normal.stdout" \
    >"$evidence/pair-$pair.oracle-input"
  python3 "$PROJECT_ROOT/tests/g3c4/check_p2_g1_oracle.py" \
    "$evidence/pair-$pair.oracle-input" >"$evidence/pair-$pair.oracle.stdout"
done
printf 'EOS first frame normal/repeat/sanitizer and independent oracle PASS\n' \
  >"$evidence/gate.stdout"
