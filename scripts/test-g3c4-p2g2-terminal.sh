#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/g3c4_proof_command.sh"
if [[ -n ${G3C4_PROOF_RECEIPTS:-} ]]; then
  export G3C4_PROOF_SCRIPT="${BASH_SOURCE[0]##*/}"
fi
verify_toolchain
cc= cxx=
resolve_provenance_compilers cc cxx \
  "${CC:-$(lock_value supported_cc)}" "${CXX:-$(lock_value supported_cxx)}"
evidence="${1:-$(project_build_dir)/g3c4-p2g2-terminal-test}"
proof_command mkdir -p "$evidence"
tmp="$(proof_command mktemp -d "$evidence/build.XXXXXX")"
proof_command python3 "$PROJECT_ROOT/scripts/check-g3c4-p2g2-terminal.py" \
  --cc "$cc" >"$evidence/static.stdout"

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
  -DET_G3C4_P2_G2_CARRIER_BRIDGE_PRIVATE
  -DET_G3C4_P2_G2_EOS_FIRST_FRAME_PRIVATE
  -DET_G3C4_P2_G2_SECOND_FRAME_PRIVATE)
for mode in off on; do
  macros=("${owner_macros[@]}")
  a2_macros=()
  if [[ "$mode" == on ]]; then
    macros+=(-DET_G3C4_P2_G2_TERMINAL_PRIVATE
      -DET_A2_KV_CACHE_TERMINAL_WITNESS_PRIVATE)
    a2_macros+=(-DET_A2_KV_CACHE_TERMINAL_WITNESS_PRIVATE)
  fi
  proof_command "$cc" "${flags[@]}" -O2 "${macros[@]}" -c \
    "$PROJECT_ROOT/src/eshkol_transformer/g3c4_model_owner.c" \
    -o "$tmp/owner-$mode.o"
  proof_command "$cc" "${flags[@]}" -O2 "${a2_macros[@]}" -c \
    "$PROJECT_ROOT/native/a2_kv_cache.c" -o "$tmp/a2-$mode.o"
  for unit in owner a2; do
    proof_command nm -g --defined-only --format=posix "$tmp/$unit-$mode.o" |
      proof_command awk 'NF >= 2 {print $1}' | LC_ALL=C proof_command sort -u >"$tmp/$unit-$mode.symbols"
  done
done
proof_command comm -3 "$tmp/owner-off.symbols" "$tmp/owner-on.symbols" \
  >"$tmp/owner-symbols.diff"
printf '\tet_g3c4_private_p2g2_terminal_commit_v1\n\tet_g3c4_private_p2g2_terminal_snapshot_v1\n' \
  >"$tmp/owner-expected.diff"
proof_command cmp "$tmp/owner-expected.diff" "$tmp/owner-symbols.diff"
proof_command comm -3 "$tmp/a2-off.symbols" "$tmp/a2-on.symbols" \
  >"$tmp/a2-symbols.diff"
printf '\tet_a2_kv_cache_private_terminal_transaction_witness_v1\n' \
  >"$tmp/a2-expected.diff"
proof_command cmp "$tmp/a2-expected.diff" "$tmp/a2-symbols.diff"
if proof_command "$cc" "${flags[@]}" -DET_G3C4_P2_G2_TERMINAL_PRIVATE -c \
  "$PROJECT_ROOT/src/eshkol_transformer/g3c4_model_owner.c" \
  -o "$tmp/isolated.o" 2>"$tmp/isolated.stderr"; then
  printf 'isolated terminal feature compiled\n' >&2; exit 1
fi
proof_command grep -F 'terminal requires second/EOS frames' "$tmp/isolated.stderr" >/dev/null
if proof_command "$cc" "${flags[@]}" "${owner_macros[@]}" \
  -DET_G3C4_P2_G2_TERMINAL_PRIVATE -c \
  "$PROJECT_ROOT/src/eshkol_transformer/g3c4_model_owner.c" \
  -o "$tmp/no-witness.o" 2>"$tmp/no-witness.stderr"; then
  printf 'terminal feature compiled without A2 witness\n' >&2; exit 1
fi
proof_command grep -F 'terminal requires second/EOS frames' "$tmp/no-witness.stderr" >/dev/null
printf 'private symbols, macro closure and feature-off preprocessing PASS\n' \
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
terminal_test="$PROJECT_ROOT/tests/g3c4/test_p2_g2_native_terminal.c"
terminal_wrap=(-Wl,--wrap=et_kernel_runtime_dispatch
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
    runtime=(env ASAN_OPTIONS=detect_leaks=1:halt_on_error=1
      UBSAN_OPTIONS=halt_on_error=1)
  fi
  proof_command "$cc" "${flags[@]}" -DET_A2_KV_CACHE_TERMINAL_WITNESS_PRIVATE \
    "${optimization[@]}" "$terminal_test" "${sources[@]}" \
    "${terminal_wrap[@]}" -lm -o "$tmp/$mode" \
    >"$evidence/$mode.compile.stdout" 2>"$evidence/$mode.compile.stderr"
  test ! -s "$evidence/$mode.compile.stderr"
  proof_command "${runtime[@]}" "$tmp/$mode" >"$evidence/$mode.stdout" \
    2>"$evidence/$mode.stderr"
  test ! -s "$evidence/$mode.stderr"
  proof_command grep -E '^G3-C4 P2/G2 terminal PASS: checks=[1-9][0-9]*$' \
    "$evidence/$mode.stdout" >/dev/null
  proof_command python3 "$PROJECT_ROOT/tests/g3c4/check_p2_g2_second_oracle.py" \
    "$evidence/$mode.stdout" >"$evidence/$mode.oracle.stdout"
  proof_command grep -Fx 'G3-C4 P2/G2 independent second-draw oracle PASS: rows=17 carry/final/singleton=1' \
    "$evidence/$mode.oracle.stdout" >/dev/null
done
proof_command "$tmp/normal" >"$evidence/repeat.stdout" 2>"$evidence/repeat.stderr"
test ! -s "$evidence/repeat.stderr"
proof_command cmp "$evidence/normal.stdout" "$evidence/repeat.stdout"
proof_command cmp "$evidence/normal.stdout" "$evidence/sanitize.stdout"

coexist="$PROJECT_ROOT/tests/g3c4/test_p2_g2_terminal_p2g1_coexist.c"
proof_command "$cc" "${flags[@]}" -DET_A2_KV_CACHE_TERMINAL_WITNESS_PRIVATE \
  -O2 "$coexist" "${sources[@]}" \
  -Wl,--wrap=et_kernel_runtime_dispatch \
  -Wl,--wrap=et_i64_tensor_borrow_view_v1 -lm \
  -o "$tmp/coexist" >"$evidence/coexist.compile.stdout" \
  2>"$evidence/coexist.compile.stderr"
test ! -s "$evidence/coexist.compile.stderr"
proof_command "$tmp/coexist" >"$evidence/coexist.stdout" 2>"$evidence/coexist.stderr"
test ! -s "$evidence/coexist.stderr"
proof_command grep -E '^G3-C4 P2/G1 with terminal feature PASS: checks=[1-9][0-9]*$' \
  "$evidence/coexist.stdout" >/dev/null

for predecessor in prefix_commit eos_first_frame second_frame; do
  wraps=(-Wl,--wrap=et_kernel_runtime_dispatch
    -Wl,--wrap=et_i64_tensor_borrow_view_v1)
  if [[ "$predecessor" == prefix_commit ]]; then
    wraps+=(-Wl,--wrap=et_a2_kv_cache_transaction_commit_v1)
  elif [[ "$predecessor" == second_frame ]]; then
    wraps+=(-Wl,--wrap=et_a2_kv_cache_transaction_abort_v1)
  fi
  proof_command "$cc" "${flags[@]}" -O2 \
    "$PROJECT_ROOT/tests/g3c4/test_p2_g2_$predecessor.c" \
    "${sources[@]}" "${wraps[@]}" -lm -o "$tmp/$predecessor" \
    >"$evidence/$predecessor.compile.stdout" \
    2>"$evidence/$predecessor.compile.stderr"
  test ! -s "$evidence/$predecessor.compile.stderr"
  proof_command "$tmp/$predecessor" >"$evidence/$predecessor.stdout" \
    2>"$evidence/$predecessor.stderr"
  test ! -s "$evidence/$predecessor.stderr"
  proof_command grep -E '^G3-C4 P2/G2 .* PASS: checks=[1-9][0-9]*' \
    "$evidence/$predecessor.stdout" >/dev/null
done
proof_command python3 "$PROJECT_ROOT/tests/g3c4/check_p2_g2_second_oracle.py" \
  "$evidence/second_frame.stdout" >"$evidence/second_frame.oracle.stdout"

printf 'Terminal normal/repeat/sanitizer/oracle, coexistence, predecessors and boundaries PASS\n' \
  >"$evidence/gate.stdout"
