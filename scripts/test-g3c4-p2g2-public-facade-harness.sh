#!/usr/bin/env bash
# Candidate compiled C4 same-registry facade gate. It does not build/install G3-G.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
cc= cxx=
resolve_provenance_compilers cc cxx \
  "${CC:-$(lock_value supported_cc)}" "${CXX:-$(lock_value supported_cxx)}"
runner="$(eshkol_build_dir)/eshkol-run"
evidence="${1:-$(project_build_dir)/g3c4-p2g2-public-facade-harness}"
mkdir -p "$evidence"
python3 "$PROJECT_ROOT/scripts/check-g3c4-p2g2-public-facade-contract.py" \
  >"$evidence/static.stdout"
tmp="$(mktemp -d "$evidence/build.XXXXXX")"
trap 'status=$?; if (( status == 0 )); then rm -rf -- "$tmp"; else
  printf "Public G2 harness failure retained: %s\n" "$tmp" >&2; fi' EXIT

compile_mode() {
  local mode=$1 directory="$tmp/$1" source
  local -a optimization=(-O2) runtime=(env)
  if [[ "$mode" == sanitize ]]; then
    optimization=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
    runtime=(env ASAN_OPTIONS=detect_leaks=1:halt_on_error=1
                 UBSAN_OPTIONS=halt_on_error=1)
  fi
  mkdir -p "$directory/objects" "$directory/cache"
  local -a common=(-std=c11 -Wall -Wextra -Werror -Wpedantic -fPIC
    -fvisibility=hidden -fno-common -fstack-protector-all -ffp-contract=off
    -fexcess-precision=standard -fno-fast-math
    -I "$PROJECT_ROOT/include" -I "$PROJECT_ROOT/native"
    -I "$PROJECT_ROOT/src" -I "$PROJECT_ROOT/src/eshkol_transformer"
    -I "$(eshkol_source_dir)/inc")
  "$cc" "${common[@]}" "${optimization[@]}" \
    -DET_A2_KV_CACHE_TESTING -DET_A2_KV_CACHE_TERMINAL_WITNESS_PRIVATE \
    -DET_G3C4_P2_G1_PENDING_PRIVATE \
    -DET_G3C4_P2_G2_FIRST_FRAME_PRIVATE \
    -DET_G3C4_P2_G2_PREFIX_COMMIT_PRIVATE \
    -DET_G3C4_P2_G2_CARRIER_BRIDGE_PRIVATE \
    -DET_G3C4_T1_I1_EXACT_PAIR_PRIVATE \
    -DET_F32_TENSOR_STORAGE_QUERY_PRIVATE \
    -DET_I64_TENSOR_STORAGE_QUERY_PRIVATE \
    -c "$PROJECT_ROOT/tests/g3c4/p2_g2_public_facade_native.c" \
    -o "$directory/objects/facade_native.o"
  "$cc" "${common[@]}" "${optimization[@]}" \
    -c "$PROJECT_ROOT/tests/m3t/kernel_fail_allocator.c" \
    -o "$directory/objects/kernel_fail_allocator.o"
  "$cc" "${common[@]}" "${optimization[@]}" \
    -DET_I2_NATIVE_HELPERS_ONLY -DET_G3C4_I2_CONSTRUCTION_PRIVATE \
    -DET_G3C4_NATIVE_OWNER_PRIVATE \
    -c "$PROJECT_ROOT/native/i2_wave2_package_bridge.c" \
    -o "$directory/objects/i2_native_helpers.o"
  "$cxx" -std=c++17 -Wall -Wextra -Werror -Wpedantic -fPIC \
    -fvisibility=hidden -fno-common -fstack-protector-all \
    "${optimization[@]}" -I "$(eshkol_source_dir)/inc" \
    -I "$(eshkol_source_dir)/lib/core" \
    -c "$PROJECT_ROOT/tests/g3c4/p2_g2_terminal_close_allocation_shim.cpp" \
    -o "$directory/objects/call_entry_allocation_shim.o"
  "$cc" "${common[@]}" "${optimization[@]}" -DET_P1_TRUSTED_BUILD=1 \
    -c "$PROJECT_ROOT/native/p1_identity.c" \
    -o "$directory/objects/p1_identity.o"
  for source in data_io checkpoint_io n3k_primitives_provider \
      g3n_primitives_provider n2_primitives_provider \
      g3c4_primitives_provider g3s_sampling_provider \
      a2_attention_provider a2_kv_cache; do
    "$cc" "${common[@]}" "${optimization[@]}" \
      -DET_A2_KV_CACHE_TESTING -DET_A2_KV_CACHE_STORAGE_QUERY_PRIVATE \
      -DET_A2_KV_CACHE_TERMINAL_WITNESS_PRIVATE \
      -c "$PROJECT_ROOT/native/$source.c" \
      -o "$directory/objects/$source.o"
  done
  "$cc" "${common[@]}" "${optimization[@]}" \
    -DET_G3C4_T1_I1_EXACT_PAIR_PRIVATE -DET_T1_I64_SHELL_TESTING \
    -c "$PROJECT_ROOT/native/t1_i64_shell.c" \
    -o "$directory/objects/t1_i64_shell.o"
  "$cc" "${common[@]}" "${optimization[@]}" -DET_I64_TENSOR_TESTING \
    -DET_G3C4_T1_I1_EXACT_PAIR_PRIVATE \
    -DET_I64_TENSOR_STORAGE_QUERY_PRIVATE -DET_M3_TESTING \
    -c "$PROJECT_ROOT/src/eshkol_transformer/m3_i64_integration.c" \
    -o "$directory/objects/m3_i64_integration.o"
  "$cc" "${common[@]}" "${optimization[@]}" \
    -c "$PROJECT_ROOT/src/eshkol_transformer/m3_model.c" \
    -o "$directory/objects/m3_model.o"
  ar rcsD "$directory/libg3c4_p2g2_public_facade_harness.a" \
    "$directory"/objects/*.o
  local sanitizer_link=
  [[ "$mode" == sanitize ]] && \
    sanitizer_link='-fsanitize=address,undefined -fno-omit-frame-pointer'
  cat >"$directory/cxx-wrap" <<WRAPPER
#!/usr/bin/env bash
exec $(printf '%q' "$cxx") $sanitizer_link "\$@" \\
  -Wl,--wrap=arena_allocate_vector_with_header \\
  -Wl,--wrap=arena_allocate_with_header \\
  -Wl,--wrap=arena_allocate_cons_with_header \\
  -Wl,--wrap=malloc -Wl,--wrap=eshkol_push_exception_handler \\
  -Wl,--wrap=et_kernel_runtime_dispatch \\
  -Wl,--wrap=et_i64_tensor_borrow_view_v1 \\
  -Wl,--wrap=et_g3c4_private_p2g2_first_frame_carrier_v1 \\
  -Wl,--wrap=et_g3c4_private_p2g2_eos_first_frame_carrier_v1 \\
  -Wl,--wrap=et_g3c4_private_p2g2_second_frame_carrier_v1 \\
  -Wl,--wrap=et_g3c4_private_p2g2_prefix_commit_v1 \\
  -Wl,--wrap=et_g3c4_private_p2g2_terminal_commit_v1 \\
  -Wl,--wrap=et_g3c4_private_p2g2_terminal_snapshot_v1 \\
  -Wl,--wrap=et_g3c4_private_p2g2_terminal_ids_clone_copy_v1 \\
  -Wl,--wrap=et_a2_kv_cache_transaction_abort_v1 \\
  -Wl,--wrap=et_a2_kv_cache_transaction_commit_v1
WRAPPER
  chmod 0500 "$directory/cxx-wrap"
  env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
    XDG_CACHE_HOME="$directory/cache" \
    ESHKOL_CXX_COMPILER="$directory/cxx-wrap" \
    ESHKOL_LIB_DIR="$PROJECT_ROOT/lib" \
    timeout --foreground --signal=TERM --kill-after=5s 600s \
    "$runner" --strict-types --optimize 0 --no-stdlib \
    -I "$PROJECT_ROOT/internal/p1/lib" -I "$PROJECT_ROOT/internal/c1/lib" \
    -I "$PROJECT_ROOT/internal/t1/lib" -I "$PROJECT_ROOT/src" \
    -I "$PROJECT_ROOT/lib" -I "$PROJECT_ROOT/native" \
    -L "$directory" --lib g3c4_p2g2_public_facade_harness \
    "$PROJECT_ROOT/tests/g3c4/p2_g2_public_facade_harness_test.esk" \
    -o "$directory/public-g2-harness" \
    >"$evidence/$mode-compile.stdout" \
    2>"$evidence/$mode-compile.stderr"
  "${runtime[@]}" ESHKOL_ARENA_POISON=1 \
    timeout --foreground --signal=TERM --kill-after=5s 120s \
    "$directory/public-g2-harness" \
    >"$evidence/$mode-run.stdout" 2>"$evidence/$mode-run.stderr"
  test ! -s "$evidence/$mode-run.stderr"
  grep -E '^G3-C4 PUBLIC G2 HARNESS PASS: checks=[1-9][0-9]*$' \
    "$evidence/$mode-run.stdout" >/dev/null
  python3 "$PROJECT_ROOT/tests/g3c4/check_p2_g2_public_facade_harness_oracle.py" \
    "$evidence/$mode-run.stdout" >"$evidence/$mode-oracle.stdout"
}

compile_mode normal
ESHKOL_ARENA_POISON=1 \
  timeout --foreground --signal=TERM --kill-after=5s 120s \
  "$tmp/normal/public-g2-harness" \
  >"$evidence/repeat-run.stdout" 2>"$evidence/repeat-run.stderr"
test ! -s "$evidence/repeat-run.stderr"
cmp "$evidence/normal-run.stdout" "$evidence/repeat-run.stdout"
compile_mode sanitize
cmp "$evidence/normal-run.stdout" "$evidence/sanitize-run.stdout"
git -C "$PROJECT_ROOT" diff --check
cat "$evidence/normal-oracle.stdout"
grep -E '^G3-C4 PUBLIC G2 HARNESS PASS:' "$evidence/normal-run.stdout"
