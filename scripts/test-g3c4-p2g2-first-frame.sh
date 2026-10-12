#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

evidence="${1:-$(project_build_dir)/g3c4-p2g2-first-frame-test}"
mkdir -p "$evidence"
source "$PROJECT_ROOT/scripts/g3c4_terminal_command_receipt.sh"
terminal_command_index=0
terminal_commands_root=$(mktemp -d "$evidence/commands.XXXXXX")
printf '%s\n' "$terminal_commands_root" >"$evidence/commands-path.txt"
terminal_record_command toolchain-verify - - -- \
  bash -c 'source "$1"; verify_toolchain' bash "$PROJECT_ROOT/scripts/common.sh"
terminal_restore_toolchain_export
for command in cmp git grep nm python3 sha256sum sort awk; do require_command "$command"; done
cc= cxx=
resolve_provenance_compilers cc cxx \
  "${CC:-$(lock_value supported_cc)}" "${CXX:-$(lock_value supported_cxx)}"
terminal_record_command static static.stdout static.stderr -- \
  python3 "$PROJECT_ROOT/scripts/check-g3c4-p2g2-first-frame.py"

first_receipt_stdout() {
  printf '%s/%04d-%s/stdout.raw\n' \
    "$terminal_commands_root" "$terminal_command_index" "$1"
}

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
  terminal_record_command "owner-$mode-compile" - - -- \
    "$cc" "${flags[@]}" -O2 "${macros[@]}" \
    -c "$PROJECT_ROOT/src/eshkol_transformer/g3c4_model_owner.c" \
    -o "$tmp/$mode.o"
  terminal_record_command "owner-$mode-nm" - - -- \
    nm -g --defined-only --format=posix "$tmp/$mode.o"
  nm_stdout=$(first_receipt_stdout "owner-$mode-nm")
  terminal_record_command "owner-$mode-awk" - - -- \
    awk 'NF >= 2 {print $1}' "$nm_stdout"
  awk_stdout=$(first_receipt_stdout "owner-$mode-awk")
  terminal_record_command "owner-$mode-sort" "$mode.symbols" - -- \
    env LC_ALL=C sort -u "$awk_stdout"
done
terminal_record_command symbol-parity - - -- \
  cmp "$evidence/off.symbols" "$evidence/on.symbols"
invalid_status=0
terminal_record_command invalid-compile invalid.stdout invalid.stderr -- \
  "$cc" "${flags[@]}" -O2 -DET_G3C4_P2_G2_FIRST_FRAME_PRIVATE \
    -c "$PROJECT_ROOT/src/eshkol_transformer/g3c4_model_owner.c" \
    -o "$tmp/invalid.o" || invalid_status=$?
if (( invalid_status == 0 || invalid_status >= 124 )); then
  printf 'isolated P2/G2 compile lacked ordinary diagnostic rejection: %s\n' \
    "$invalid_status" >&2
  exit 1
fi
terminal_record_command invalid-diagnostic - - -- \
  grep -F 'C4 P2/G2 first frame requires pending output and token forward' \
    "$evidence/invalid.stderr"
printf 'G3-C4 P2/G2 first-frame symbols and feature closure PASS\n' \
  >"$evidence/symbols.stdout"

terminal_record_command feature-off-compile - - -- \
  "$cc" "${flags[@]}" -O2 "${test_macros[@]}" \
  "$PROJECT_ROOT/tests/g3c4/test_p2_g1_pending.c" \
  "${native_sources[@]}" -Wl,--wrap=et_kernel_runtime_dispatch \
  -Wl,--wrap=et_i64_tensor_borrow_view_v1 -lm -o "$tmp/feature-off"
terminal_record_command feature-off-run feature-off.stdout feature-off.stderr -- \
  "$tmp/feature-off"
terminal_record_command feature-off-stderr-empty - - -- \
  test ! -s "$evidence/feature-off.stderr"
terminal_record_command feature-off-pass - - -- \
  grep -E '^G3-C4 P2/G1 pending PASS: checks=[1-9][0-9]*$' \
    "$evidence/feature-off.stdout"

for mode in normal sanitize; do
  optimization=(-O2)
  runtime=(env)
  if [[ "$mode" == sanitize ]]; then
    optimization=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
    runtime=(env ASAN_OPTIONS=detect_leaks=1:halt_on_error=1
                 UBSAN_OPTIONS=halt_on_error=1)
  fi
  terminal_record_command "$mode-compile" - - -- \
    "$cc" "${flags[@]}" "${optimization[@]}" "${test_macros[@]}" \
    "$PROJECT_ROOT/tests/g3c4/test_p2_g2_first_frame.c" \
    "${native_sources[@]}" -Wl,--wrap=et_kernel_runtime_dispatch \
    -Wl,--wrap=et_i64_tensor_borrow_view_v1 -lm -o "$tmp/$mode"
  terminal_record_command "$mode-run" "$mode.stdout" "$mode.stderr" -- \
    "${runtime[@]}" "$tmp/$mode"
  terminal_record_command "$mode-stderr-empty" - - -- \
    test ! -s "$evidence/$mode.stderr"
  terminal_record_command "$mode-pass" - - -- \
    grep -E '^G3-C4 P2/G2 first frame PASS: checks=[1-9][0-9]*$' \
      "$evidence/$mode.stdout"
done
terminal_record_command oracle oracle.stdout oracle.stderr -- \
  python3 "$PROJECT_ROOT/tests/g3c4/check_p2_g1_oracle.py" \
    "$evidence/normal.stdout"
terminal_record_command repeat-run repeat.stdout repeat.stderr -- \
  "$tmp/normal"
terminal_record_command repeat-stderr-empty - - -- \
  test ! -s "$evidence/repeat.stderr"
terminal_record_command repeat-parity - - -- \
  cmp "$evidence/normal.stdout" "$evidence/repeat.stdout"
terminal_record_command sanitize-parity - - -- \
  cmp "$evidence/normal.stdout" "$evidence/sanitize.stdout"
terminal_record_command receipts-audit receipts-audit.stdout receipts-audit.stderr -- \
  python3 "$PROJECT_ROOT/tests/g3c4/check_first_frame_command_receipts.py" \
    "$evidence" --seal-current
printf 'G3-C4 P2/G2 first-frame normal/repeat/sanitizer PASS\n' \
  >"$evidence/gate.stdout"
