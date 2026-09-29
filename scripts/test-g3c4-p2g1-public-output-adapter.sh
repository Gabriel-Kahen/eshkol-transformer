#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
for command in ar cmp env git grep nm python3 sha256sum timeout; do require_command "$command"; done
cc= cxx=
resolve_provenance_compilers cc cxx \
  "${CC:-$(lock_value supported_cc)}" "${CXX:-$(lock_value supported_cxx)}"
eshkol_runner="$(eshkol_build_dir)/eshkol-run"
evidence="${1:-$(project_build_dir)/g3c4-p2g1-public-output-adapter-test}"
mkdir -p "$evidence"
python3 "$PROJECT_ROOT/scripts/check-g3c4-p2g1-shell.py" >"$evidence/static.stdout"
tmp="$(mktemp -d "$evidence/build.XXXXXX")"
cleanup() {
  local status=$?
  if (( status == 0 )); then rm -rf -- "$tmp"; else
    printf 'G3-C4 P2/G1 public adapter failure evidence retained: %s\n' "$tmp" >&2
  fi
}
trap cleanup EXIT
flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -fstack-protector-all
       -ffp-contract=off -fexcess-precision=standard -fno-fast-math
       -I "$PROJECT_ROOT/include" -I "$PROJECT_ROOT/native"
       -I "$PROJECT_ROOT/src" -I "$PROJECT_ROOT/src/eshkol_transformer")
test_macros=(-DET_G3C4_T1_I1_EXACT_PAIR_PRIVATE
             -DET_I64_TENSOR_TESTING -DET_I64_TENSOR_STORAGE_QUERY_PRIVATE
             -DET_A2_KV_CACHE_TESTING -DET_A2_KV_CACHE_STORAGE_QUERY_PRIVATE)
native_sources=(
  "$PROJECT_ROOT/src/eshkol_transformer/m3_i64_integration.c"
  "$PROJECT_ROOT/tests/m3t/kernel_fail_allocator.c"
  "$PROJECT_ROOT/native/t1_i64_shell.c"
  "$PROJECT_ROOT/native/n3k_primitives_provider.c"
  "$PROJECT_ROOT/native/g3c4_primitives_provider.c"
  "$PROJECT_ROOT/native/g3s_sampling_provider.c"
  "$PROJECT_ROOT/native/g3n_primitives_provider.c"
  "$PROJECT_ROOT/native/n2_primitives_provider.c"
  "$PROJECT_ROOT/native/a2_kv_cache.c"
)
compile_mode() {
  local mode=$1 directory="$tmp/$1" source
  local -a optimization=(-O2) runtime=(env)
  if [[ "$mode" == sanitize ]]; then
    optimization=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
    runtime=(env ASAN_OPTIONS=detect_leaks=1:halt_on_error=1
                 UBSAN_OPTIONS=halt_on_error=1)
  fi
  mkdir -p "$directory/objects" "$directory/cache"
  "$cc" "${flags[@]}" "${optimization[@]}" "${test_macros[@]}" \
    "$PROJECT_ROOT/tests/g3c4/test_p2_g1_copyout.c" \
    "${native_sources[@]}" -Wl,--wrap=et_kernel_runtime_dispatch \
    -Wl,--wrap=et_i64_tensor_borrow_view_v1 -lm \
    -o "$directory/p2g1-native"
  "${runtime[@]}" "$directory/p2g1-native" \
    >"$directory/native.stdout" 2>"$directory/native.stderr"
  test ! -s "$directory/native.stderr"
  grep -E '^G3-C4 P2/G1 private copy-out PASS: checks=[1-9][0-9]*$' \
    "$directory/native.stdout" >/dev/null
  grep '^ORACLE ' "$directory/native.stdout" | tail -n 2 \
    >"$directory/oracle.rows"
  python3 "$PROJECT_ROOT/tests/g3c4/check_p2_g1_oracle.py" \
    "$directory/oracle.rows" >"$directory/oracle.stdout"

  if [[ "$mode" == normal ]]; then
    "$cc" "${flags[@]}" "${optimization[@]}" "${test_macros[@]}" \
      "$PROJECT_ROOT/tests/g3c4/test_p2_g1_feature_off.c" \
      "${native_sources[@]}" -Wl,--wrap=et_kernel_runtime_dispatch -lm \
      -o "$directory/p2g1-feature-off"
    "$directory/p2g1-feature-off" >"$directory/feature-off.stdout" \
      2>"$directory/feature-off.stderr"
    test ! -s "$directory/feature-off.stderr"
    grep -E '^G3-C4 P2/G1 feature-off PASS: checks=[1-9][0-9]*$' \
      "$directory/feature-off.stdout" >/dev/null
  fi

  local -a common=("${flags[@]}" -DET_G3C4_T1_I1_EXACT_PAIR_PRIVATE
                   -fPIC -fvisibility=hidden -fno-common
                   -I "$(eshkol_source_dir)/inc")
  "$cc" "${common[@]}" "${optimization[@]}" \
    -DET_G3C4_P2_G1_PUBLICATION_PRIVATE -DET_G3C4_P2_G1_COPYOUT_PRIVATE \
    -DET_G3C4_P2_G1_COORDINATOR_PRIVATE \
    -DET_G3C4_P2_G1_PUBLIC_RESULT_PRIVATE \
    -DET_G3C4_P2_G1_PUBLIC_RESULT_TESTING \
    -DET_F32_TENSOR_STORAGE_QUERY_PRIVATE -DET_A2_KV_CACHE_TESTING \
    -c "$PROJECT_ROOT/tests/g3c4/p2_g1_shell_native.c" \
    -o "$directory/objects/p2_g1_shell_native.o"
  if [[ "$mode" == normal ]]; then
    "$cc" "${common[@]}" "${optimization[@]}" \
      -DET_G3C4_P2_G1_PUBLICATION_PRIVATE \
      -DET_F32_TENSOR_STORAGE_QUERY_PRIVATE -DET_A2_KV_CACHE_TESTING \
      -c "$PROJECT_ROOT/tests/g3c4/p2_g1_shell_native.c" \
      -o "$directory/feature-off.o"
    nm -g --defined-only --format=posix "$directory/feature-off.o" |
      awk 'NF >= 2 {print $1}' | LC_ALL=C sort -u \
      >"$directory/feature-off-symbols"
    nm -g --defined-only --format=posix \
      "$directory/objects/p2_g1_shell_native.o" |
      awk 'NF >= 2 {print $1}' | LC_ALL=C sort -u \
      >"$directory/feature-on-symbols"
    comm -13 "$directory/feature-off-symbols" \
      "$directory/feature-on-symbols" >"$evidence/coordinator-symbols.txt"
    grep -Fx et_g3c4_private_output_copy_snapshot_v1 \
      "$evidence/coordinator-symbols.txt" >/dev/null
    grep -Fx et_g3c4_private_p2g1_preflight_v1 \
      "$evidence/coordinator-symbols.txt" >/dev/null
    grep -Fx et_g3c4_private_p2g1_run_v1 \
      "$evidence/coordinator-symbols.txt" >/dev/null
    for symbol in et_g3c4_private_result_tensor_create_v1 \
        et_g3c4_private_result_tensor_release_v1 \
        et_g3c4_private_result_rng_create_v1 \
        et_g3c4_private_output_device_state_v1; do
      grep -Fx "$symbol" "$evidence/coordinator-symbols.txt" >/dev/null
      if grep -Fx "$symbol" "$directory/feature-off-symbols"; then
        die "C4 result feature-off object exports $symbol"
      fi
    done
  fi
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
    -c "$PROJECT_ROOT/tests/g3c4/call_entry_allocation_shim.cpp" \
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
      -c "$PROJECT_ROOT/native/$source.c" \
      -o "$directory/objects/$source.o"
  done
  "$cc" "${common[@]}" "${optimization[@]}" -DET_T1_I64_SHELL_TESTING \
    -c "$PROJECT_ROOT/native/t1_i64_shell.c" \
    -o "$directory/objects/t1_i64_shell.o"
  "$cc" "${common[@]}" "${optimization[@]}" -DET_I64_TENSOR_TESTING \
    -DET_I64_TENSOR_STORAGE_QUERY_PRIVATE -DET_M3_TESTING \
    -c "$PROJECT_ROOT/src/eshkol_transformer/m3_i64_integration.c" \
    -o "$directory/objects/m3_i64_integration.o"
  "$cc" "${common[@]}" "${optimization[@]}" \
    -c "$PROJECT_ROOT/src/eshkol_transformer/m3_model.c" \
    -o "$directory/objects/m3_model.o"
  ar rcsD "$directory/libg3c4_p2g1_shell.a" "$directory"/objects/*.o
  local sanitizer_link=
  [[ "$mode" == sanitize ]] && sanitizer_link='-fsanitize=address,undefined -fno-omit-frame-pointer'
  cat >"$directory/cxx-wrap" <<WRAPPER
#!/usr/bin/env bash
exec $(printf '%q' "$cxx") $sanitizer_link "\$@" \\
  -Wl,--wrap=arena_allocate_vector_with_header \\
  -Wl,--wrap=arena_allocate_with_header \\
  -Wl,--wrap=arena_allocate_cons_with_header \\
  -Wl,--wrap=malloc -Wl,--wrap=eshkol_push_exception_handler \\
  -Wl,--wrap=et_kernel_runtime_dispatch \
  -Wl,--wrap=et_g3c4_private_output_copy_decode_ids_v1 \
  -Wl,--wrap=et_g3c4_private_output_accept_text_v1 \
  -Wl,--wrap=et_g3c4_private_output_copy_snapshot_v1 \
  -Wl,--wrap=et_g3c4_private_result_tensor_create_v1 \
  -Wl,--wrap=et_i64_tensor_copy_from_v1
WRAPPER
  chmod 0500 "$directory/cxx-wrap"
  local test expected
  for test in copyout coordinator public_output_adapter; do
    env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
      XDG_CACHE_HOME="$directory/cache" ESHKOL_CXX_COMPILER="$directory/cxx-wrap" \
      ESHKOL_LIB_DIR="$PROJECT_ROOT/lib" \
      timeout --foreground --signal=TERM --kill-after=5s 600s \
      "$eshkol_runner" --strict-types --optimize 0 --no-stdlib \
      -I "$PROJECT_ROOT/internal/p1/lib" -I "$PROJECT_ROOT/internal/c1/lib" \
      -I "$PROJECT_ROOT/internal/t1/lib" -I "$PROJECT_ROOT/src" \
      -I "$PROJECT_ROOT/lib" -I "$PROJECT_ROOT/native" \
      -I "$PROJECT_ROOT/tests/g3c4" \
      -L "$directory" --lib g3c4_p2g1_shell \
      "$PROJECT_ROOT/tests/g3c4/p2_g1_${test}_test.esk" \
      -o "$directory/p2g1-$test" \
      >"$directory/$test-compile.stdout" \
      2>"$directory/$test-compile.stderr"
    "${runtime[@]}" ESHKOL_ARENA_POISON=1 \
      timeout --foreground --signal=TERM --kill-after=5s 120s \
      "$directory/p2g1-$test" \
      >"$directory/$test.stdout" 2>"$directory/$test.stderr"
    test ! -s "$directory/$test.stderr"
    if [[ "$test" == copyout ]]; then
      expected='greedy=1 categorical=1 bytevector-cuts=2 vector-cuts=2 decode-cut=1'
      grep -E "^G3-C4 P2/G1 copy-out PASS: checks=[1-9][0-9]* $expected$" \
        "$directory/$test.stdout" >/dev/null
    elif [[ "$test" == coordinator ]]; then
      grep -E '^G3-C4 P2/G1 coordinator PASS: checks=[1-9][0-9]* prompts=2 retry=1$' \
        "$directory/$test.stdout" >/dev/null
    else
      grep -E '^G3-C4 P2/G1 public adapter source-composed PASS: checks=[1-9][0-9]*$' \
        "$directory/$test.stdout" >/dev/null
    fi
  done
}

compile_mode normal
for test in copyout coordinator public_output_adapter; do
  ESHKOL_ARENA_POISON=1 "$tmp/normal/p2g1-$test" \
    >"$tmp/normal/$test-repeat.stdout" \
    2>"$tmp/normal/$test-repeat.stderr"
  cmp "$tmp/normal/$test.stderr" "$tmp/normal/$test-repeat.stderr"
done
compile_mode sanitize
for test in copyout coordinator public_output_adapter; do
  cmp "$tmp/normal/$test.stdout" "$tmp/normal/$test-repeat.stdout"
  cmp "$tmp/normal/$test.stdout" "$tmp/sanitize/$test.stdout"
done
for mode in normal sanitize; do
  for log in native.stdout native.stderr oracle.stdout copyout.stdout \
      copyout.stderr copyout-compile.stdout copyout-compile.stderr \
      coordinator.stdout coordinator.stderr coordinator-compile.stdout \
      coordinator-compile.stderr public_output_adapter.stdout \
      public_output_adapter.stderr public_output_adapter-compile.stdout \
      public_output_adapter-compile.stderr; do
    cp "$tmp/$mode/$log" "$evidence/$mode-$log"
  done
done
for test in copyout coordinator public_output_adapter; do
  cp "$tmp/normal/$test-repeat.stdout" "$evidence/$test-repeat.stdout"
done
if git -C "$PROJECT_ROOT" rev-parse --is-inside-work-tree \
    >/dev/null 2>&1; then
  git -C "$PROJECT_ROOT" diff --check
else
  printf 'Docker worktree git metadata inaccessible; host-side clean and diff check required\n' \
    >"$evidence/git-metadata.stdout"
fi
(
  cd "$evidence"
  sha256sum -- *.stdout *.stderr >SHA256SUMS
)
cat "$evidence/static.stdout" "$evidence/normal-coordinator.stdout"
printf 'G3-C4 P2/G1 public adapter evidence: %s\n' "$evidence"
