#!/usr/bin/env bash
# Private source-root compile probe. Numerical/fault acceptance is separate.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
for command in ar cmp env grep nm python3 timeout; do require_command "$command"; done
cc= cxx=
resolve_provenance_compilers cc cxx \
  "${CC:-$(lock_value supported_cc)}" "${CXX:-$(lock_value supported_cxx)}"
evidence="${1:?an empty evidence directory is required}"
mkdir -p "$evidence"
[[ -z "$(find "$evidence" -mindepth 1 -maxdepth 1 -print -quit)" ]] || \
  die 'typed aggregate smoke evidence directory is not empty'
tmp=$(mktemp -d "$evidence/build.XXXXXX")

PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -q \
  tests.g3c4.test_p2_g2_terminal_source_prerequisites \
  tests.g3c4.test_p2_g2_terminal_close_source \
  tests.g3c4.test_p2_g2_terminal_detached_source \
  tests.g3c4.test_p2_g2_terminal_typed_composition_source \
  tests.g3c4.test_p2_g2_typed_aggregate_package_source \
  >"$evidence/static.stdout" 2>"$evidence/static.stderr"
if env -u PYTHONOPTIMIZE python3 \
    "$PROJECT_ROOT/tests/g3c4/check_p2_g2_typed_aggregate_oracle.py" \
    /dev/null >"$evidence/oracle-empty.stdout" \
    2>"$evidence/oracle-empty.stderr"; then
  die 'empty independent-oracle transcript unexpectedly passed'
fi
CC="$cc" "$PROJECT_ROOT/scripts/test-g3c4-p2g2-typed-result.sh" \
  "$evidence/native-typed" >"$evidence/native-typed.stdout" \
  2>"$evidence/native-typed.stderr"
test ! -s "$evidence/native-typed.stderr"

common=(-std=c11 -Wall -Wextra -Werror -Wpedantic -fPIC
  -fvisibility=hidden -fno-common -fstack-protector-all
  -ffp-contract=off -fexcess-precision=standard -fno-fast-math
  -I "$PROJECT_ROOT/include" -I "$PROJECT_ROOT/native"
  -I "$PROJECT_ROOT/src" -I "$PROJECT_ROOT/src/eshkol_transformer"
  -I "$(eshkol_source_dir)/inc")
owner=(-DET_G3C4_CONTEXT_PRIVATE -DET_G3C4_NATIVE_PINS_PRIVATE
  -DET_G3C4_ACTIVE_CALL_PRIVATE -DET_G3C4_GENERATOR_PRIVATE
  -DET_G3C4_PROMPT_T1_BORROW_PRIVATE -DET_G3C4_PROVIDER_ROUTES_PRIVATE
  -DET_G3C4_FULL_PREFIX_FORWARD_PRIVATE -DET_G3C4_SAMPLER_TRANSPORT_PRIVATE
  -DET_G3C4_TOKEN_FRAME_PRIVATE -DET_G3C4_TOKEN_FORWARD_PRIVATE
  -DET_G3C4_PREFILL3_PRIVATE -DET_G3C4_PREFILL1_PRIVATE
  -DET_G3C4_PREFILL2_PRIVATE -DET_G3C4_PROMPT_PREFILL_PRIVATE
  -DET_G3C4_OUTPUT_RESERVATION_PRIVATE -DET_G3C4_LAST_LOGIT_FRAME_PRIVATE
  -DET_G3C4_OUTPUT_PREPARE_PRIVATE -DET_G3C4_OUTPUT_DECODE_IDS_PRIVATE
  -DET_G3C4_OUTPUT_TEXT_PRIVATE -DET_G3C4_P2_G1_PENDING_PRIVATE
  -DET_G3C4_P2_G1_PUBLICATION_PRIVATE -DET_G3C4_P2_G1_COPYOUT_PRIVATE
  -DET_G3C4_P2_G1_COORDINATOR_PRIVATE -DET_G3C4_P2_G1_PUBLIC_RESULT_PRIVATE
  -DET_G3C4_P2_G2_FIRST_FRAME_PRIVATE -DET_G3C4_P2_G2_PREFIX_COMMIT_PRIVATE
  -DET_G3C4_P2_G2_CARRIER_BRIDGE_PRIVATE
  -DET_G3C4_P2_G2_EOS_FIRST_FRAME_PRIVATE
  -DET_G3C4_P2_G2_SECOND_FRAME_PRIVATE
  -DET_G3C4_P2_G2_TERMINAL_PRIVATE
  -DET_G3C4_P2_G2_TERMINAL_IDS_CLONE_PRIVATE
  -DET_G3C4_P2_G2_TYPED_RESULT_PRIVATE
  -DET_G3C4_T1_I1_EXACT_PAIR_PRIVATE
  -DET_I64_TENSOR_STORAGE_QUERY_PRIVATE
  -DET_F32_TENSOR_STORAGE_QUERY_PRIVATE
  -DET_A2_KV_CACHE_STORAGE_QUERY_PRIVATE
  -DET_A2_KV_CACHE_TERMINAL_WITNESS_PRIVATE)
transport=(-DET_G3T_TESTING -DET_F32_TENSOR_TESTING
  -DET_A2_KV_CACHE_TESTING -DET_I64_TENSOR_TESTING
  -DET_G3T_PREFILL_SAMPLE_PRIVATE -DET_G3T_OUTPUT_TEXT_PRIVATE
  -DET_G3T_FINAL_PUBLICATION_PRIVATE -DET_G3T_ZERO_BUDGET_PRIVATE
  -DET_G3T_P2_ZERO_BUDGET_PRIVATE
  -DET_G3T_FULL_REQUEST_PREFLIGHT_PRIVATE
  -DET_G3T_MANUAL_LOGITS_PRIVATE
  -DET_G3T_MANUAL_LOGITS_MATERIALIZE_PRIVATE
  -DET_G3T_MANUAL_P1_PREFILL_PRIVATE -DET_G3T_MANUAL_P2_PREFILL_PRIVATE
  -DET_G3T_MANUAL_DECODE_PRIVATE -DET_G3T_OUTPUT_IDS_CLONE_PRIVATE
  -DET_G3T_OUTPUT_LENGTHS_CLONE_PRIVATE
  -DET_G3T_OUTPUT_CACHE_LENGTHS_CLONE_PRIVATE
  -DET_G3T_OUTPUT_RNG_CLONE_PRIVATE -DET_G3T_GENERATOR_RNG_PRIVATE
  -DET_G3T_INPUT_FROM_T1_PRIVATE -DET_G3T_OWNED_TOKEN_INPUT_PRIVATE
  -DET_I64_TENSOR_STORAGE_QUERY_PRIVATE
  -DET_A2_KV_CACHE_STORAGE_QUERY_PRIVATE)
sources=(native/data_io.c native/checkpoint_io.c native/kernel_abi.c
  tests/g3c4/p2_g2_typed_aggregate_shape_native.c
  native/n3k_primitives_provider.c native/g3n_primitives_provider.c
  native/n2_primitives_provider.c native/g3c4_primitives_provider.c
  native/g3s_sampling_provider.c native/a2_attention_provider.c
  native/a2_kv_cache.c native/p1_identity.c native/t1_i64_shell.c
  native/i2_wave2_package_bridge.c
  src/eshkol_transformer/m3_i64_integration.c
  src/eshkol_transformer/m3_call_f32_integration.c
  src/eshkol_transformer/g3t_transport.c
  tests/g3c4/p2_g2_typed_aggregate_oracle_owner.c)

build_mode() {
  local mode=$1 dir="$tmp/$1" source stem path
  local -a optimization=(-O2) runtime=(env) extra=()
  if [[ "$mode" == sanitize ]]; then
    optimization=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
    runtime=(env ASAN_OPTIONS=detect_leaks=1:halt_on_error=1
                 UBSAN_OPTIONS=halt_on_error=1)
  fi
  mkdir -p "$dir/objects" "$dir/cache"
  for source in "${sources[@]}"; do
    stem=$(basename "$source" .c)
    extra=()
    case "$stem" in
      p2_g2_typed_aggregate_shape_native) extra=(
        -DET_G3C4_P2_G2_TERMINAL_PRIVATE
        -DET_G3C4_P2_G2_TERMINAL_IDS_CLONE_PRIVATE
        -DET_G3C4_P2_G2_TYPED_RESULT_PRIVATE
        -DET_I64_TENSOR_TESTING) ;;
      p2_g2_typed_aggregate_oracle_owner) extra=("${owner[@]}") ;;
      g3t_transport) extra=("${transport[@]}") ;;
      m3_i64_integration) extra=(-DET_M3_TESTING -DET_I64_TENSOR_TESTING
        -DET_I64_TENSOR_STORAGE_QUERY_PRIVATE
        -DET_G3C4_T1_I1_EXACT_PAIR_PRIVATE) ;;
      m3_call_f32_integration) extra=(-DET_F32_TENSOR_TESTING
        -DET_G3C4_NATIVE_PINS_PRIVATE
        -DET_G3C4_NATIVE_OWNER_PRIVATE -DET_F32_TENSOR_STORAGE_QUERY_PRIVATE) ;;
      t1_i64_shell) extra=(-DET_G3C4_T1_I1_EXACT_PAIR_PRIVATE
        -DET_T1_I64_SHELL_TESTING) ;;
      a2_kv_cache) extra=(-DET_A2_KV_CACHE_TESTING
        -DET_A2_KV_CACHE_STORAGE_QUERY_PRIVATE
        -DET_A2_KV_CACHE_TERMINAL_WITNESS_PRIVATE) ;;
      i2_wave2_package_bridge) extra=(-DET_I2_NATIVE_HELPERS_ONLY
        -DET_G3C4_I2_CONSTRUCTION_PRIVATE -DET_G3C4_NATIVE_OWNER_PRIVATE) ;;
      p1_identity) extra=(-DET_P1_TRUSTED_BUILD=1) ;;
    esac
    path="$dir/objects/$stem.o"
    "$cc" "${common[@]}" "${optimization[@]}" "${extra[@]}" \
      -MMD -MF "$dir/objects/$stem.d" \
      -c "$PROJECT_ROOT/$source" -o "$path" \
      >"$dir/$stem.compile.stdout" 2>"$dir/$stem.compile.stderr"
    test ! -s "$dir/$stem.compile.stderr"
  done
  ar rcsD "$dir/libg3c4_p2g2_typed_aggregate.a" "$dir"/objects/*.o
  ar t "$dir/libg3c4_p2g2_typed_aggregate.a" \
    >"$evidence/$mode.archive-members.txt"
  nm -A --defined-only "$dir/libg3c4_p2g2_typed_aggregate.a" \
    >"$evidence/$mode.archive-defined.txt"
  nm -A -u "$dir/libg3c4_p2g2_typed_aggregate.a" \
    >"$evidence/$mode.archive-undefined.txt"
  test "$(wc -l <"$evidence/$mode.archive-members.txt")" -eq "${#sources[@]}"
  ! grep -Fx f32_tensor.o "$evidence/$mode.archive-members.txt" >/dev/null
  for stem in m3_call_f32_integration g3t_transport p2_g2_typed_aggregate_oracle_owner \
              p2_g2_typed_aggregate_shape_native; do
    grep -Fx "$stem.o" "$evidence/$mode.archive-members.txt" >/dev/null
  done
  nm -g --defined-only --format=posix \
    "$dir/objects/m3_call_f32_integration.o" | awk '{print $1}' \
    >"$evidence/$mode.f32-testing-symbols.txt"
  for symbol in et_f32_tensor_test_metadata_snapshot_v1 \
                et_f32_tensor_test_metadata_corrupt_v1 \
                et_f32_tensor_test_metadata_restore_v1 \
                et_f32_tensor_test_fail_alloc_after_v1 \
                et_f32_test_live_counts_snapshot_v1; do
    grep -Fx "$symbol" "$evidence/$mode.f32-testing-symbols.txt" >/dev/null
  done
  local sanitizer_link=
  [[ "$mode" == sanitize ]] && \
    sanitizer_link='-fsanitize=address,undefined -fno-omit-frame-pointer'
  cat >"$dir/cxx-wrap" <<WRAPPER
#!/usr/bin/env bash
exec $(printf '%q' "$cxx") $sanitizer_link "\$@" \\
  -Wl,-Map=$(printf '%q' "$dir/link.map") \\
  -Wl,--wrap=et_g3c4_private_p2g2_terminal_snapshot_v1 \\
  -Wl,--wrap=et_g3c4_private_p2g2_terminal_ids_clone_v1 \\
  -Wl,--wrap=et_g3c4_private_p2g2_terminal_ids_clone_copy_v1 \\
  -Wl,--wrap=et_g3c4_private_p2g2_typed_ids_create_v1 \\
  -Wl,--wrap=et_g3c4_private_p2g2_typed_scalar_create_v1 \\
  -Wl,--wrap=et_g3c4_private_p2g2_eos_first_frame_carrier_v1 \\
  -Wl,--wrap=et_g3c4_private_p2g2_prefix_commit_v1 \\
  -Wl,--wrap=et_g3c4_private_p2g2_second_frame_carrier_v1 \\
  -Wl,--wrap=et_g3c4_private_p2g2_terminal_commit_v1 \\
  -Wl,--wrap=et_a2_kv_cache_transaction_abort_v1
WRAPPER
  chmod 0500 "$dir/cxx-wrap"
  env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
    XDG_CACHE_HOME="$dir/cache" ESHKOL_CXX_COMPILER="$dir/cxx-wrap" \
    ESHKOL_LIB_DIR="$PROJECT_ROOT/lib" \
    timeout --foreground --signal=TERM --kill-after=5s 600s \
    "$(eshkol_build_dir)/eshkol-run" --strict-types --optimize 0 --no-stdlib \
    -I "$PROJECT_ROOT/internal/p1/lib" -I "$PROJECT_ROOT/internal/c1/lib" \
    -I "$PROJECT_ROOT/internal/t1/lib" -I "$PROJECT_ROOT/src" \
    -I "$PROJECT_ROOT/lib" -I "$PROJECT_ROOT/native" \
    -L "$dir" --lib g3c4_p2g2_typed_aggregate \
    "$PROJECT_ROOT/tests/g3c4/p2_g2_terminal_typed_same_aggregate.esk" \
    -o "$dir/fixture" >"$dir/fixture.compile.stdout" \
    2>"$dir/fixture.compile.stderr"
  test ! -s "$dir/fixture.compile.stderr"
  test -s "$dir/link.map"
  grep -F "libg3c4_p2g2_typed_aggregate.a(g3t_transport.o)" \
    "$dir/link.map" >/dev/null
  grep -F "libg3c4_p2g2_typed_aggregate.a(m3_call_f32_integration.o)" \
    "$dir/link.map" >/dev/null
  grep -F "libg3c4_p2g2_typed_aggregate.a(p2_g2_typed_aggregate_oracle_owner.o)" \
    "$dir/link.map" >/dev/null
  ! grep -F 'libg3g_manual_private.a' "$dir/link.map" >/dev/null
  cp "$dir/link.map" "$evidence/$mode.link.map"
  "${runtime[@]}" ESHKOL_ARENA_POISON=1 \
    timeout --foreground --signal=TERM --kill-after=5s 120s \
    "$dir/fixture" >"$evidence/$mode.stdout" \
    2>"$evidence/$mode.stderr"
  test ! -s "$evidence/$mode.stderr"
  grep -E '^P2/G2 typed same-aggregate smoke PASS: checks=[1-9][0-9]*$' \
    "$evidence/$mode.stdout" >/dev/null
  env -u PYTHONOPTIMIZE python3 \
    "$PROJECT_ROOT/tests/g3c4/check_p2_g2_typed_aggregate_oracle.py" \
    "$evidence/$mode.stdout" >"$evidence/$mode.oracle.stdout"
  cp "$dir/fixture.compile.stdout" "$evidence/$mode.compile.stdout"
  cp "$dir/fixture.compile.stderr" "$evidence/$mode.compile.stderr"
}

build_mode normal
ESHKOL_ARENA_POISON=1 "$tmp/normal/fixture" \
  >"$evidence/repeat.stdout" 2>"$evidence/repeat.stderr"
test ! -s "$evidence/repeat.stderr"
build_mode sanitize
cmp "$evidence/normal.stdout" "$evidence/repeat.stdout"
cmp "$evidence/normal.stdout" "$evidence/sanitize.stdout"
printf 'Private P2/G2 typed aggregate smoke only PASS\n'
