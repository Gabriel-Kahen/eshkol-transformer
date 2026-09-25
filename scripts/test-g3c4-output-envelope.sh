#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
for command in ar cmp env git grep python3 sha256sum timeout; do require_command "$command"; done
cc= cxx=
resolve_provenance_compilers cc cxx "${CC:-$(lock_value supported_cc)}" "${CXX:-$(lock_value supported_cxx)}"
eshkol_runner="$(eshkol_build_dir)/eshkol-run"
evidence="${1:-$(project_build_dir)/g3c4-output-envelope-test}"
mkdir -p "$evidence"
python3 "${PROJECT_ROOT}/scripts/check-g3c4-output-envelope.py" >"$evidence/static.stdout"
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v tests.q0.test_python_isolation \
  >"$evidence/python-isolation.stdout" 2>"$evidence/python-isolation.stderr"
if [[ "${G3C4_SKIP_OUTPUT_ENVELOPE_REGRESSIONS:-0}" == 1 ]]; then
  printf 'G3-C4 output text regression: SKIPPED\n' >"$evidence/output-text-regression.stdout"
  printf 'G3-C4 call entry regression: SKIPPED\n' >"$evidence/call-entry-regression.stdout"
else
  CC="$cc" "${PROJECT_ROOT}/scripts/test-g3c4-output-text.sh" \
    "$evidence/output-text-regression" >"$evidence/output-text-regression.stdout" 2>&1
  CC="$cc" "${PROJECT_ROOT}/scripts/test-g3c4-call-entry.sh" \
    "$evidence/call-entry-regression" >"$evidence/call-entry-regression.stdout" 2>&1
fi

tmp="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-g3c4-output-envelope.XXXXXX")"
cleanup() {
  status=$?
  if (( status == 0 )); then rm -rf -- "$tmp"; else
    printf 'G3-C4 output envelope failure evidence retained at %s\n' "$tmp" >&2
    find "$tmp" -type f \( -name '*.stderr' -o -name '*.stdout' \) -print -exec tail -200 {} \; >&2 || true
  fi
}
trap cleanup EXIT

compile_mode() {
  mode=$1 directory="$tmp/$1"
  mode_flags=(-O2); runtime=(env)
  if [[ "$mode" == sanitize ]]; then
    mode_flags=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
    runtime=(env ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1)
  fi
  mkdir -p "$directory/objects" "$directory/cache"
  common=(-std=c11 -Wall -Wextra -Werror -Wpedantic -fPIC -fvisibility=hidden
          -fno-common -fstack-protector-all -ffp-contract=off
          -fexcess-precision=standard -fno-fast-math
          -I "${PROJECT_ROOT}/include" -I "${PROJECT_ROOT}/native"
          -I "${PROJECT_ROOT}/src" -I "${PROJECT_ROOT}/src/eshkol_transformer"
          -I "$(eshkol_source_dir)/inc")
  "$cc" "${common[@]}" "${mode_flags[@]}" -DET_A2_KV_CACHE_TESTING \
    -c "${PROJECT_ROOT}/tests/g3c4/output_envelope_native.c" \
    -o "$directory/objects/output_envelope_native.o"
  "$cc" "${common[@]}" "${mode_flags[@]}" \
    -c "${PROJECT_ROOT}/tests/m3t/kernel_fail_allocator.c" \
    -o "$directory/objects/kernel_fail_allocator.o"
  "$cc" "${common[@]}" "${mode_flags[@]}" -DET_I2_NATIVE_HELPERS_ONLY \
    -DET_G3C4_I2_CONSTRUCTION_PRIVATE -DET_G3C4_NATIVE_OWNER_PRIVATE \
    -c "${PROJECT_ROOT}/native/i2_wave2_package_bridge.c" \
    -o "$directory/objects/i2_native_helpers.o"
  "$cxx" -std=c++17 -Wall -Wextra -Werror -Wpedantic -fPIC \
    -fvisibility=hidden -fno-common -fstack-protector-all "${mode_flags[@]}" \
    -I "$(eshkol_source_dir)/inc" -I "$(eshkol_source_dir)/lib/core" \
    -c "${PROJECT_ROOT}/tests/g3c4/call_entry_allocation_shim.cpp" \
    -o "$directory/objects/call_entry_allocation_shim.o"
  "$cc" "${common[@]}" "${mode_flags[@]}" -DET_P1_TRUSTED_BUILD=1 \
    -c "${PROJECT_ROOT}/native/p1_identity.c" -o "$directory/objects/p1_identity.o"
  for source in data_io checkpoint_io n3k_primitives_provider \
                g3n_primitives_provider n2_primitives_provider \
                g3c4_primitives_provider g3s_sampling_provider \
                a2_attention_provider a2_kv_cache; do
    "$cc" "${common[@]}" "${mode_flags[@]}" -DET_A2_KV_CACHE_TESTING \
      -DET_A2_KV_CACHE_STORAGE_QUERY_PRIVATE \
      -c "${PROJECT_ROOT}/native/${source}.c" -o "$directory/objects/${source}.o"
  done
  "$cc" "${common[@]}" "${mode_flags[@]}" -DET_T1_I64_SHELL_TESTING \
    -c "${PROJECT_ROOT}/native/t1_i64_shell.c" -o "$directory/objects/t1_i64_shell.o"
  "$cc" "${common[@]}" "${mode_flags[@]}" -DET_I64_TENSOR_TESTING \
    -DET_I64_TENSOR_STORAGE_QUERY_PRIVATE -DET_M3_TESTING \
    -c "${PROJECT_ROOT}/src/eshkol_transformer/m3_i64_integration.c" \
    -o "$directory/objects/m3_i64_integration.o"
  "$cc" "${common[@]}" "${mode_flags[@]}" \
    -c "${PROJECT_ROOT}/src/eshkol_transformer/m3_model.c" -o "$directory/objects/m3_model.o"
  ar rcsD "$directory/libg3c4_output_envelope.a" "$directory"/objects/*.o
  sanitize_link=
  [[ "$mode" == sanitize ]] && sanitize_link='-fsanitize=address,undefined -fno-omit-frame-pointer'
  cat >"$directory/cxx-wrap" <<WRAPPER
#!/usr/bin/env bash
exec $(printf '%q' "$cxx") $sanitize_link "\$@" \\
  -Wl,--wrap=arena_allocate_vector_with_header \\
  -Wl,--wrap=arena_allocate_cons_with_header \\
  -Wl,--wrap=malloc -Wl,--wrap=eshkol_push_exception_handler \\
  -Wl,--wrap=et_kernel_runtime_dispatch
WRAPPER
  chmod 0500 "$directory/cxx-wrap"
  for test in output_envelope_test output_envelope_allocation_test; do
    env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
      XDG_CACHE_HOME="$directory/cache" ESHKOL_CXX_COMPILER="$directory/cxx-wrap" \
      ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" timeout --foreground --signal=TERM --kill-after=5s 600s \
      "$eshkol_runner" --strict-types --optimize 0 --no-stdlib \
      -I "${PROJECT_ROOT}/internal/p1/lib" -I "${PROJECT_ROOT}/internal/c1/lib" \
      -I "${PROJECT_ROOT}/internal/t1/lib" -I "${PROJECT_ROOT}/src" \
      -I "${PROJECT_ROOT}/lib" -I "${PROJECT_ROOT}/native" \
      -L "$directory" --lib g3c4_output_envelope \
      "${PROJECT_ROOT}/tests/g3c4/${test}.esk" -o "$directory/$test" \
      >"$directory/${test}-compile.stdout" 2>"$directory/${test}-compile.stderr"
    "${runtime[@]}" ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM --kill-after=5s 120s \
      "$directory/$test" >"$directory/${test}.stdout" 2>"$directory/${test}.stderr"
    test ! -s "$directory/${test}.stderr"
  done
  grep -E '^G3-C4 output envelope PASS: checks=[1-9][0-9]* routes=2 linkage-cuts=3 native-root-cut=1 repeat-cuts=2$' \
    "$directory/output_envelope_test.stdout" >/dev/null
  grep -E '^G3-C4 output envelope allocation PASS: vector=[1-9][0-9]* cons=[1-9][0-9]* checks=[1-9][0-9]*$' \
    "$directory/output_envelope_allocation_test.stdout" >/dev/null
}
compile_mode normal
for test in output_envelope_test output_envelope_allocation_test; do
  ESHKOL_ARENA_POISON=1 "$tmp/normal/$test" >"$tmp/normal/${test}-runtime-repeat.stdout" \
    2>"$tmp/normal/${test}-runtime-repeat.stderr"
  test ! -s "$tmp/normal/${test}-runtime-repeat.stderr"
done
compile_mode sanitize
for test in output_envelope_test output_envelope_allocation_test; do
  cmp "$tmp/normal/${test}.stdout" "$tmp/normal/${test}-runtime-repeat.stdout"
  cmp "$tmp/normal/${test}.stdout" "$tmp/sanitize/${test}.stdout"
  for mode in normal sanitize; do
    cp "$tmp/$mode/${test}.stdout" "$evidence/$mode-${test}.stdout"
    cp "$tmp/$mode/${test}.stderr" "$evidence/$mode-${test}.stderr"
    cp "$tmp/$mode/${test}-compile.stdout" "$evidence/$mode-${test}-compile.stdout"
    cp "$tmp/$mode/${test}-compile.stderr" "$evidence/$mode-${test}-compile.stderr"
  done
  cp "$tmp/normal/${test}-runtime-repeat.stdout" "$evidence/${test}-runtime-repeat.stdout"
done
git -C "$PROJECT_ROOT" diff --check
sha256sum "${PROJECT_ROOT}/native/g3c4_output_envelope_source_closure.txt" >"$evidence/closure.sha256"
cat "$evidence/static.stdout"
cat "$evidence/normal-output_envelope_test.stdout"
cat "$evidence/normal-output_envelope_allocation_test.stdout"
printf 'G3-C4 private output envelope evidence: %s\n' "$evidence"
