#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
if [[ "${ESHKOL_HOLDER_SOURCE_PROBE:-0}" == 1 ]]; then
  cc="${CC:-cc}"
  command -v "$cc" >/dev/null 2>&1 || die "source-probe C compiler unavailable"
  printf 'unsupported-host source-only C probe: %s\n' "$cc" >&2
  leak_detection=0
else
  verify_toolchain
  cc= cxx=
  resolve_provenance_compilers cc cxx \
    "${CC:-$(lock_value supported_cc)}" "${CXX:-$(lock_value supported_cxx)}"
  leak_detection=1
fi
evidence="${1:-$(project_build_dir)/g3c4-p2g2-held-holder-test}"
mkdir -p "$evidence"
tmp="$(mktemp -d "$evidence/build.XXXXXX")"
export TMPDIR="$tmp"
trap 'status=$?; if (( status == 0 )); then rm -rf -- "$tmp"; else
  printf "P2/G2 held-holder failure retained: %s\n" "$tmp" >&2; fi' EXIT
flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -fstack-protector-all
  -ffp-contract=off -fexcess-precision=standard -fno-fast-math
  -I "$PROJECT_ROOT/include" -I "$PROJECT_ROOT/native"
  -I "$PROJECT_ROOT/src" -I "$PROJECT_ROOT/src/eshkol_transformer"
  -DET_I64_TENSOR_TESTING -DET_I64_TENSOR_STORAGE_QUERY_PRIVATE
  -DET_F32_TENSOR_STORAGE_QUERY_PRIVATE
  -DET_A2_KV_CACHE_TESTING -DET_A2_KV_CACHE_STORAGE_QUERY_PRIVATE
  -DET_A2_KV_CACHE_TERMINAL_WITNESS_PRIVATE)
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
  -DET_G3C4_P2_G2_CARRIER_BRIDGE_PRIVATE
  -DET_G3C4_P2_G2_EOS_FIRST_FRAME_PRIVATE
  -DET_G3C4_P2_G2_SECOND_FRAME_PRIVATE
  -DET_G3C4_P2_G2_TERMINAL_PRIVATE)
for mode in off on; do
  macros=("${owner_macros[@]}")
  [[ "$mode" == on ]] && macros+=(-DET_G3C4_P2_G2_HELD_HOLDER_PRIVATE)
  "$cc" "${flags[@]}" -O2 "${macros[@]}" -c \
    "$PROJECT_ROOT/src/eshkol_transformer/g3c4_model_owner.c" \
    -o "$tmp/owner-$mode.o"
  nm -g --defined-only --format=posix "$tmp/owner-$mode.o" |
    awk 'NF >= 2 {print $1}' | LC_ALL=C sort -u \
    >"$tmp/owner-$mode.symbols"
done
comm -3 "$tmp/owner-off.symbols" "$tmp/owner-on.symbols" \
  >"$tmp/owner-symbols.diff"
printf '\tet_g3c4_private_p2g2_holder_abort_attempt_v1\n\tet_g3c4_private_p2g2_holder_begin_capture_v1\n\tet_g3c4_private_p2g2_holder_bind_v1\n\tet_g3c4_private_p2g2_holder_dispose_v1\n\tet_g3c4_private_p2g2_holder_end_after_busy_v1\n\tet_g3c4_private_p2g2_holder_generation_v1\n\tet_g3c4_private_p2g2_holder_new_v1\n\tet_g3c4_private_p2g2_holder_normal_close_v1\n\tet_g3c4_private_p2g2_holder_witness_v1\n' \
  >"$tmp/owner-expected.diff"
cmp "$tmp/owner-expected.diff" "$tmp/owner-symbols.diff"
if "$cc" "${flags[@]}" -DET_G3C4_P2_G2_HELD_HOLDER_PRIVATE -c \
  "$PROJECT_ROOT/src/eshkol_transformer/g3c4_model_owner.c" \
  -o "$tmp/isolated.o" 2>"$tmp/isolated.stderr"; then
  printf 'isolated holder feature compiled\n' >&2; exit 1
fi
rg -F 'held holder requires terminal, pending I1 and A2 witness' \
  "$tmp/isolated.stderr" >/dev/null
printf 'private holder feature-on/off symbols and macro closure PASS\n' \
  >"$evidence/boundary.stdout"

sources=("$PROJECT_ROOT/src/eshkol_transformer/m3_i64_integration.c"
  "$PROJECT_ROOT/tests/m3t/kernel_fail_allocator.c"
  "$PROJECT_ROOT/native/t1_i64_shell.c"
  "$PROJECT_ROOT/native/n3k_primitives_provider.c"
  "$PROJECT_ROOT/native/g3c4_primitives_provider.c"
  "$PROJECT_ROOT/native/g3s_sampling_provider.c"
  "$PROJECT_ROOT/native/g3n_primitives_provider.c"
  "$PROJECT_ROOT/native/n2_primitives_provider.c"
  "$PROJECT_ROOT/native/a2_kv_cache.c")
wraps=(-Wl,--wrap=et_kernel_runtime_dispatch
  -Wl,--wrap=et_i64_tensor_borrow_view_v1
  -Wl,--wrap=et_a2_kv_cache_transaction_view_end_v1
  -Wl,--wrap=et_a2_kv_cache_transaction_abort_v1
  -Wl,--wrap=et_i64_tensor_copy_from_v1
  -Wl,--wrap=et_a2_kv_cache_transaction_commit_v1)
for mode in normal sanitize; do
  optimization=(-O2)
  runtime=(env)
  if [[ "$mode" == sanitize ]]; then
    optimization=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
    runtime=(env ASAN_OPTIONS=detect_leaks="$leak_detection":halt_on_error=1
      UBSAN_OPTIONS=halt_on_error=1)
  fi
  "$cc" "${flags[@]}" "${optimization[@]}" \
    -DET_G3C4_HOLDER_TOKEN_CAPACITY=32 \
    "$PROJECT_ROOT/tests/g3c4/test_p2_g2_held_holder.c" \
    "${sources[@]}" "${wraps[@]}" -lm -o "$tmp/$mode" \
    >"$evidence/$mode.compile.stdout" 2>"$evidence/$mode.compile.stderr"
  test ! -s "$evidence/$mode.compile.stderr"
  "${runtime[@]}" "$tmp/$mode" >"$evidence/$mode.stdout" \
    2>"$evidence/$mode.stderr"
  test ! -s "$evidence/$mode.stderr"
  rg '^G3-C4 P2/G2 private holder PASS: checks=[1-9][0-9]*$' \
    "$evidence/$mode.stdout" >/dev/null
done
"$tmp/normal" >"$evidence/repeat.stdout" 2>"$evidence/repeat.stderr"
test ! -s "$evidence/repeat.stderr"
cmp "$evidence/normal.stdout" "$evidence/repeat.stdout"
cmp "$evidence/normal.stdout" "$evidence/sanitize.stdout"
printf 'focused normal/repeat/sanitizer and feature-off PASS\n' \
  >"$evidence/gate.stdout"
