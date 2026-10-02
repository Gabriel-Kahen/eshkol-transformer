#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
for command in ar cmp nm python3 sha256sum timeout; do require_command "$command"; done
cc= cxx=
resolve_provenance_compilers cc cxx "${CC:-$(lock_value supported_cc)}" "${CXX:-$(lock_value supported_cxx)}"
evidence="${1:-$(project_build_dir)/g3r-rng-import}"
mkdir -p "$evidence"
python3 "$PROJECT_ROOT/scripts/check-g3r-rng-import.py" >"$evidence/static.stdout"
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v tests.q0.test_python_isolation \
  >"$evidence/q0.stdout" 2>"$evidence/q0.stderr"
temporary="$(mktemp -d "$evidence/tmp.XXXXXX")"
trap 'status=$?; if (( status == 0 )); then rm -rf -- "$temporary"; else echo "G3-R typed RNG import build retained: $temporary" >&2; fi' EXIT
sources=(native/data_io.c native/checkpoint_io.c native/kernel_abi.c
  src/eshkol_transformer/m3_i64_integration.c native/t1_i64_shell.c
  src/eshkol_transformer/m3_call_f32_integration.c
  src/eshkol_transformer/g3t_transport.c native/a2_kv_cache.c
  native/n2_primitives_provider.c native/n3k_primitives_provider.c
  native/a2_attention_provider.c native/g3n_primitives_provider.c
  native/g3s_sampling_provider.c)
build_mode() {
  local mode="$1" directory="$temporary/$1" source stem
  local -a flags=(-O2) runtime=(env)
  if [[ "$mode" == sanitize ]]; then
    flags=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
    runtime=(env ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1)
  fi
  mkdir -p "$directory/objects" "$directory/cache"
  local -a common=(-std=c11 -Wall -Wextra -Werror -Wpedantic -fPIC
    -fvisibility=hidden -fno-common -fstack-protector-all -ffp-contract=off
    -fexcess-precision=standard -fno-fast-math -I "$PROJECT_ROOT/include"
    -I "$PROJECT_ROOT/native" -I "$PROJECT_ROOT/src"
    -I "$PROJECT_ROOT/src/eshkol_transformer" -I "$(eshkol_source_dir)/inc")
  "$cc" "${common[@]}" "${flags[@]}" -DET_I2_NATIVE_HELPERS_ONLY \
    -DET_M3T_PACKAGE_BUILD -c "$PROJECT_ROOT/native/i2_wave2_package_bridge.c" \
    -o "$directory/objects/i2_helpers.o"
  "$cc" "${common[@]}" "${flags[@]}" -DET_P1_TRUSTED_BUILD=1 \
    -c "$PROJECT_ROOT/native/p1_identity.c" -o "$directory/objects/p1_identity.o"
  for source in "${sources[@]}"; do
    stem="$(basename "$source" .c)"
    extra=()
    if [[ "$stem" == g3t_transport ]]; then
      extra=(-DET_G3T_TESTING -DET_F32_TENSOR_TESTING
        -DET_A2_KV_CACHE_TESTING -DET_I64_TENSOR_TESTING
        -DET_G3T_PREFILL_SAMPLE_PRIVATE -DET_G3T_OUTPUT_TEXT_PRIVATE
        -DET_G3T_FINAL_PUBLICATION_PRIVATE -DET_G3T_ZERO_BUDGET_PRIVATE
        -DET_G3T_P2_ZERO_BUDGET_PRIVATE
        -DET_G3T_FULL_REQUEST_PREFLIGHT_PRIVATE
        -DET_G3T_MANUAL_LOGITS_PRIVATE -DET_G3T_MANUAL_P1_PREFILL_PRIVATE
        -DET_G3T_MANUAL_P2_PREFILL_PRIVATE -DET_G3T_MANUAL_DECODE_PRIVATE
        -DET_G3T_MANUAL_LOGITS_MATERIALIZE_PRIVATE
        -DET_G3T_OUTPUT_RNG_CLONE_PRIVATE -DET_G3T_GENERATOR_RNG_PRIVATE
        -DET_G3T_INPUT_FROM_T1_PRIVATE
        -DET_G3T_OWNED_TOKEN_INPUT_PRIVATE
        -DET_I64_TENSOR_STORAGE_QUERY_PRIVATE
        -DET_A2_KV_CACHE_STORAGE_QUERY_PRIVATE)
      [[ "$mode" != off ]] && extra+=(-DET_G3T_RECORD_RNG_IMPORT_PRIVATE)
    fi
    [[ "$stem" == m3_i64_integration ]] && extra=(-DET_M3_TESTING -DET_I64_TENSOR_TESTING -DET_I64_TENSOR_STORAGE_QUERY_PRIVATE)
    [[ "$stem" == m3_call_f32_integration ]] && extra=(-DET_F32_TENSOR_TESTING)
    [[ "$stem" == a2_kv_cache ]] && extra=(-DET_A2_KV_CACHE_TESTING -DET_A2_KV_CACHE_STORAGE_QUERY_PRIVATE)
    "$cc" "${common[@]}" "${flags[@]}" "${extra[@]}" \
      -MMD -MF "$directory/objects/$stem.d" \
      -c "$PROJECT_ROOT/$source" -o "$directory/objects/$stem.o"
  done
  "$cxx" -std=c++17 -Wall -Wextra -Werror -Wpedantic -fPIC \
    -fvisibility=hidden -fno-common -fstack-protector-all \
    -I "$(eshkol_source_dir)/inc" -I "$(eshkol_source_dir)/lib/core" \
    "${flags[@]}" -c "$PROJECT_ROOT/tests/g3r/rng_import_allocation_shim.cpp" \
    -o "$directory/objects/rng_import_allocation_shim.o"
  ar rcsD "$directory/libg3r_rng_import.a" "$directory"/objects/*.o
  if [[ "$mode" == off ]]; then
    if nm -g --defined-only "$directory/objects/g3t_transport.o" | \
        grep 'et_g3t_private_rng_words_create_v1'; then
      die "typed RNG import symbol escaped feature-off object"
    fi
  else
    nm -g --defined-only "$directory/objects/g3t_transport.o" | \
      grep 'et_g3t_private_rng_words_create_v1' >/dev/null
    "$cc" "${common[@]}" "${flags[@]}" \
      -DET_G3T_TESTING -DET_G3T_PREFILL_SAMPLE_PRIVATE \
      -DET_G3T_OUTPUT_RNG_CLONE_PRIVATE \
      -DET_G3T_RECORD_RNG_IMPORT_PRIVATE \
      "$PROJECT_ROOT/tests/g3r/rng_import_native_test.c" \
      -L "$directory" -lg3r_rng_import -lm -o "$directory/native-test"
    "${runtime[@]}" timeout --foreground --signal=TERM --kill-after=5s 120s \
      "$directory/native-test" >"$directory/native.stdout" 2>"$directory/native.stderr"
    test ! -s "$directory/native.stderr"
    grep -Fx 'G3-R native typed RNG import PASS' "$directory/native.stdout" >/dev/null
  fi
  local sanitizer_link=
  [[ "$mode" == sanitize ]] && sanitizer_link='-fsanitize=address,undefined -fno-omit-frame-pointer'
  cat >"$directory/cxx-wrap" <<WRAPPER
#!/usr/bin/env bash
exec $(printf '%q' "$cxx") $sanitizer_link "\$@" \
  -Wl,--wrap=arena_allocate_vector_with_header \
  -Wl,--wrap=arena_allocate_cons_with_header
WRAPPER
  chmod 0500 "$directory/cxx-wrap"
  local fixture=rng_import_on.esk
  [[ "$mode" == off ]] && fixture=rng_import_off.esk
  env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
    XDG_CACHE_HOME="$directory/cache" ESHKOL_CXX_COMPILER="$directory/cxx-wrap" \
    ESHKOL_LIB_DIR="$PROJECT_ROOT/lib" \
    timeout --foreground --signal=TERM --kill-after=5s 600s \
    "$(eshkol_build_dir)/eshkol-run" --strict-types --optimize 0 --no-stdlib \
    -I "$PROJECT_ROOT/internal/p1/lib" -I "$PROJECT_ROOT/internal/c1/lib" \
    -I "$PROJECT_ROOT/internal/t1/lib" -I "$PROJECT_ROOT/src" \
    -I "$PROJECT_ROOT/lib" -I "$PROJECT_ROOT/native" \
    -L "$directory" --lib g3r_rng_import \
    "$PROJECT_ROOT/tests/g3r/$fixture" -o "$directory/caller" \
    >"$directory/compile.stdout" 2>"$directory/compile.stderr"
  "${runtime[@]}" ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM \
    --kill-after=5s 120s "$directory/caller" \
    >"$directory/stdout" 2>"$directory/stderr"
  test ! -s "$directory/stderr"
  if [[ "$mode" == off ]]; then
    grep -Fx 'G3-R typed RNG import feature-off PASS: checks=1' "$directory/stdout" >/dev/null
  else
    grep -E '^G3-R typed RNG import PASS: checks=[1-9][0-9]*$' \
      "$directory/stdout" >/dev/null
    PYTHONDONTWRITEBYTECODE=1 python3 "$PROJECT_ROOT/tests/g3r/rng_import_oracle.py" \
      "$directory/stdout" >"$directory/oracle.stdout"
  fi
}
build_mode off
build_mode normal
ESHKOL_ARENA_POISON=1 "$temporary/normal/caller" \
  >"$temporary/normal/repeat.stdout" 2>"$temporary/normal/repeat.stderr"
test ! -s "$temporary/normal/repeat.stderr"
PYTHONDONTWRITEBYTECODE=1 python3 "$PROJECT_ROOT/tests/g3r/rng_import_oracle.py" \
  "$temporary/normal/repeat.stdout" >"$temporary/normal/repeat.oracle.stdout"
build_mode sanitize
cmp "$temporary/normal/stdout" "$temporary/normal/repeat.stdout"
cmp "$temporary/normal/stdout" "$temporary/sanitize/stdout"
cmp "$temporary/normal/oracle.stdout" "$temporary/sanitize/oracle.stdout"
for mode in off normal sanitize; do
  cp "$temporary/$mode/stdout" "$evidence/$mode.stdout"
  cp "$temporary/$mode/stderr" "$evidence/$mode.stderr"
  cp "$temporary/$mode/compile.stderr" "$evidence/$mode.compile.stderr"
  cp "$temporary/$mode/objects/g3t_transport.d" "$evidence/$mode.g3t_transport.d"
done
cp "$temporary/normal/oracle.stdout" "$evidence/oracle.stdout"
cp "$temporary/normal/native.stdout" "$evidence/native.stdout"
if nm -g --defined-only "$temporary/off/objects/g3t_transport.o" | \
    grep 'et_g3t_private_rng_words_create_v1'; then
  die "feature-off import symbol present"
fi
sha256sum "$PROJECT_ROOT/native/g3r_rng_import_source_closure.txt" \
  >"$evidence/closure.sha256"
git -C "$PROJECT_ROOT" diff --check
cat "$evidence/static.stdout" "$evidence/oracle.stdout"
printf 'G3-R typed RNG import evidence: %s\n' "$evidence"
