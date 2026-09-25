#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
for command in ar cmp python3 sha256sum timeout; do require_command "$command"; done
cc= cxx=
resolve_provenance_compilers cc cxx "${CC:-$(lock_value supported_cc)}" "${CXX:-$(lock_value supported_cxx)}"
evidence="${1:-$(project_build_dir)/g3t-model-admission}"
mkdir -p "$evidence"
python3 "$PROJECT_ROOT/scripts/check-g3t-model-admission.py" >"$evidence/static.stdout"
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v tests.q0.test_python_isolation \
  >"$evidence/q0.stdout" 2>"$evidence/q0.stderr"
temp="$(mktemp -d "${TMPDIR:-/tmp}/g3t-model-admission.XXXXXX")"
trap 'status=$?; if (( status == 0 )); then rm -rf -- "$temp"; else echo "G3-T admission build retained: $temp" >&2; fi' EXIT
sources=(native/data_io.c native/checkpoint_io.c native/kernel_abi.c
  src/eshkol_transformer/m3_i64_integration.c native/t1_i64_shell.c
  src/eshkol_transformer/m3_call_f32_integration.c src/eshkol_transformer/m3_model.c
  native/n2_primitives_provider.c native/n3k_primitives_provider.c
  native/a2_attention_provider.c)
build_mode() {
  local mode="$1" dir="$temp/$1" source stem
  local -a flags=(-O2) runtime=(env)
  if [[ "$mode" == sanitize ]]; then
    flags=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
    runtime=(env ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1)
  fi
  mkdir -p "$dir/objects" "$dir/cache"
  local -a common=(-std=c11 -Wall -Wextra -Werror -Wpedantic -fPIC
    -fvisibility=hidden -fno-common -fstack-protector-all -ffp-contract=off
    -fexcess-precision=standard -fno-fast-math -I "$PROJECT_ROOT/include"
    -I "$PROJECT_ROOT/native" -I "$PROJECT_ROOT/src"
    -I "$PROJECT_ROOT/src/eshkol_transformer" -I "$(eshkol_source_dir)/inc")
  "$cc" "${common[@]}" "${flags[@]}" -DET_I2_NATIVE_HELPERS_ONLY \
    -DET_M3T_PACKAGE_BUILD -c "$PROJECT_ROOT/native/i2_wave2_package_bridge.c" \
    -o "$dir/objects/i2_helpers.o"
  "$cc" "${common[@]}" "${flags[@]}" -DET_P1_TRUSTED_BUILD=1 \
    -c "$PROJECT_ROOT/native/p1_identity.c" -o "$dir/objects/p1_identity.o"
  for source in "${sources[@]}"; do
    stem="$(basename "$source" .c)"
    "$cc" "${common[@]}" "${flags[@]}" -c "$PROJECT_ROOT/$source" \
      -o "$dir/objects/$stem.o"
  done
  ar rcsD "$dir/libg3t_model_admission.a" "$dir"/objects/*.o
  local sanitizer_link=
  [[ "$mode" == sanitize ]] && sanitizer_link='-fsanitize=address,undefined -fno-omit-frame-pointer'
  cat >"$dir/cxx-wrap" <<WRAPPER
#!/usr/bin/env bash
exec $(printf '%q' "$cxx") $sanitizer_link "\$@"
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
    -L "$dir" --lib g3t_model_admission \
    "$PROJECT_ROOT/tests/g3t/model_admission_test.esk" -o "$dir/caller" \
    >"$dir/compile.stdout" 2>"$dir/compile.stderr"
  "${runtime[@]}" ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM \
    --kill-after=5s 120s "$dir/caller" >"$dir/stdout" 2>"$dir/stderr"
  test ! -s "$dir/stderr"
  grep -E '^G3-T M3T/C2 model admission PASS: checks=[1-9][0-9]*$' "$dir/stdout" >/dev/null
}
build_mode normal
ESHKOL_ARENA_POISON=1 "$temp/normal/caller" >"$temp/normal/repeat.stdout" 2>"$temp/normal/repeat.stderr"
test ! -s "$temp/normal/repeat.stderr"
build_mode sanitize
cmp "$temp/normal/stdout" "$temp/normal/repeat.stdout"
cmp "$temp/normal/stdout" "$temp/sanitize/stdout"
for mode in normal sanitize; do
  cp "$temp/$mode/stdout" "$evidence/$mode.stdout"
  cp "$temp/$mode/stderr" "$evidence/$mode.stderr"
  cp "$temp/$mode/compile.stderr" "$evidence/$mode.compile.stderr"
done
git -C "$PROJECT_ROOT" diff --check
sha256sum "$PROJECT_ROOT/native/g3t_model_admission_source_closure.txt" >"$evidence/closure.sha256"
cat "$evidence/static.stdout"
cat "$evidence/normal.stdout"
printf 'G3-T model admission evidence: %s\n' "$evidence"
