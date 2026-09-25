#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
for command in ar cmp nm python3 sha256sum timeout; do require_command "$command"; done
cc= cxx=
resolve_provenance_compilers cc cxx "${CC:-$(lock_value supported_cc)}" "${CXX:-$(lock_value supported_cxx)}"
evidence="${1:-$(project_build_dir)/g3t-p2-zero-budget}"
mkdir -p "$evidence"
python3 "$PROJECT_ROOT/scripts/check-g3t-generator-constructor.py" >"$evidence/static.stdout"
python3 "$PROJECT_ROOT/scripts/check-g3t-prefill-sample.py" >>"$evidence/static.stdout"
python3 "$PROJECT_ROOT/scripts/check-g3t-output-text.py" >>"$evidence/static.stdout"
python3 "$PROJECT_ROOT/scripts/check-g3t-final-publication.py" >>"$evidence/static.stdout"
python3 "$PROJECT_ROOT/scripts/check-g3t-zero-budget.py" >>"$evidence/static.stdout"
python3 "$PROJECT_ROOT/scripts/check-g3t-p2-zero-budget.py" >>"$evidence/static.stdout"
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v tests.q0.test_python_isolation \
  >"$evidence/q0.stdout" 2>"$evidence/q0.stderr"
temporary="$(mktemp -d "$evidence/tmp.XXXXXX")"
trap 'status=$?; if (( status == 0 )); then rm -rf -- "$temporary"; else echo "G3-T P2 zero budget build retained: $temporary" >&2; fi' EXIT
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
    [[ "$stem" == g3t_transport ]] && extra=(-DET_G3T_TESTING
      -DET_A2_KV_CACHE_TESTING -DET_I64_TENSOR_TESTING
      -DET_G3T_PREFILL_SAMPLE_PRIVATE -DET_G3T_OUTPUT_TEXT_PRIVATE
      -DET_G3T_FINAL_PUBLICATION_PRIVATE -DET_G3T_ZERO_BUDGET_PRIVATE
      -DET_G3T_P2_ZERO_BUDGET_PRIVATE
      -DET_G3T_OUTPUT_IDS_CLONE_PRIVATE
      -DET_I64_TENSOR_STORAGE_QUERY_PRIVATE -DET_A2_KV_CACHE_STORAGE_QUERY_PRIVATE)
    [[ "$stem" == m3_i64_integration ]] && extra=(-DET_I64_TENSOR_TESTING -DET_I64_TENSOR_STORAGE_QUERY_PRIVATE)
    [[ "$stem" == a2_kv_cache ]] && extra=(-DET_A2_KV_CACHE_TESTING -DET_A2_KV_CACHE_STORAGE_QUERY_PRIVATE)
    "$cc" "${common[@]}" "${flags[@]}" "${extra[@]}" \
      -MMD -MF "$directory/objects/$stem.d" \
      -c "$PROJECT_ROOT/$source" -o "$directory/objects/$stem.o"
  done
  ar rcsD "$directory/libg3t_p2_zero_budget.a" "$directory"/objects/*.o
  local sanitizer_link=
  [[ "$mode" == sanitize ]] && sanitizer_link='-fsanitize=address,undefined -fno-omit-frame-pointer'
  cat >"$directory/cxx-wrap" <<WRAPPER
#!/usr/bin/env bash
exec $(printf '%q' "$cxx") $sanitizer_link "\$@"
WRAPPER
  chmod 0500 "$directory/cxx-wrap"
  env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
    XDG_CACHE_HOME="$directory/cache" ESHKOL_CXX_COMPILER="$directory/cxx-wrap" \
    ESHKOL_LIB_DIR="$PROJECT_ROOT/lib" \
    timeout --foreground --signal=TERM --kill-after=5s 600s \
    "$(eshkol_build_dir)/eshkol-run" --strict-types --optimize 0 --no-stdlib \
    -I "$PROJECT_ROOT/internal/p1/lib" -I "$PROJECT_ROOT/internal/c1/lib" \
    -I "$PROJECT_ROOT/internal/t1/lib" -I "$PROJECT_ROOT/src" \
    -I "$PROJECT_ROOT/lib" -I "$PROJECT_ROOT/native" \
    -L "$directory" --lib g3t_p2_zero_budget \
    "$PROJECT_ROOT/tests/g3t/p2_zero_budget_test.esk" -o "$directory/caller" \
    >"$directory/compile.stdout" 2>"$directory/compile.stderr"
  "${runtime[@]}" ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM \
    --kill-after=5s 120s "$directory/caller" \
    >"$directory/stdout" 2>"$directory/stderr"
  test ! -s "$directory/stderr"
  grep -E '^G3-T P2 zero budget PASS: checks=[1-9][0-9]*$' \
    "$directory/stdout" >/dev/null
}
build_mode normal
if [[ "${G3T_P2_ONLY_NORMAL:-0}" == 1 ]]; then
  cp "$temporary/normal/stdout" "$evidence/normal.stdout"
  cat "$evidence/normal.stdout"
  exit 0
fi
ESHKOL_ARENA_POISON=1 "$temporary/normal/caller" \
  >"$temporary/normal/repeat.stdout" 2>"$temporary/normal/repeat.stderr"
test ! -s "$temporary/normal/repeat.stderr"
build_mode sanitize
cmp "$temporary/normal/stdout" "$temporary/normal/repeat.stdout"
cmp "$temporary/normal/stdout" "$temporary/sanitize/stdout"
launches="${G3T_P2_SANITIZER_LAUNCHES:-1}"
[[ "$launches" =~ ^[1-9][0-9]*$ ]] && (( launches <= 300 )) || \
  die "G3T_P2_SANITIZER_LAUNCHES must be 1..300"
for ((attempt = 2; attempt <= launches; ++attempt)); do
  env ASAN_OPTIONS=detect_leaks=1:halt_on_error=1:handle_segv=2:abort_on_error=1:print_stacktrace=1 \
    UBSAN_OPTIONS=halt_on_error=1 ESHKOL_ARENA_POISON=1 \
    timeout --foreground --signal=TERM --kill-after=5s 120s \
    "$temporary/sanitize/caller" \
    >"$temporary/sanitize/extra.stdout" \
    2>"$temporary/sanitize/extra.stderr"
  test ! -s "$temporary/sanitize/extra.stderr"
  cmp "$temporary/normal/stdout" "$temporary/sanitize/extra.stdout"
done
printf '%s\n' "$launches" >"$evidence/sanitizer-launches.txt"
for mode in normal sanitize; do
  cp "$temporary/$mode/stdout" "$evidence/$mode.stdout"
  cp "$temporary/$mode/stderr" "$evidence/$mode.stderr"
  cp "$temporary/$mode/compile.stderr" "$evidence/$mode.compile.stderr"
  cp "$temporary/$mode/objects/g3t_transport.d" "$evidence/$mode.g3t_transport.d"
done
"$cc" -std=c11 -Wall -Wextra -Werror -Wpedantic -fPIC \
  -I "$PROJECT_ROOT/include" -I "$PROJECT_ROOT/native" -I "$PROJECT_ROOT/src" \
  -I "$PROJECT_ROOT/src/eshkol_transformer" \
  -DET_G3T_PREFILL_SAMPLE_PRIVATE -DET_G3T_OUTPUT_TEXT_PRIVATE \
  -DET_G3T_FINAL_PUBLICATION_PRIVATE -DET_G3T_ZERO_BUDGET_PRIVATE \
  -DET_G3T_P2_ZERO_BUDGET_PRIVATE \
  -DET_G3T_OUTPUT_IDS_CLONE_PRIVATE \
  -DET_I64_TENSOR_STORAGE_QUERY_PRIVATE -DET_A2_KV_CACHE_STORAGE_QUERY_PRIVATE \
  -c "$PROJECT_ROOT/src/eshkol_transformer/g3t_transport.c" \
  -o "$temporary/production.o"
if nm -g --defined-only "$temporary/production.o" | grep 'et_g3t_test_'; then
  die "test hook escaped production native object"
fi
git -C "$PROJECT_ROOT" diff --check
sha256sum "$PROJECT_ROOT/native/g3t_p2_zero_budget_source_closure.txt" >"$evidence/closure.sha256"
cat "$evidence/static.stdout"
cat "$evidence/normal.stdout"
printf 'G3-T P2 zero budget evidence: %s\n' "$evidence"
