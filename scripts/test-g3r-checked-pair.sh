#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
for command in cmp env nm python3 timeout; do require_command "$command"; done
cc= cxx=
resolve_provenance_compilers cc cxx \
  "${CC:-$(lock_value supported_cc)}" "${CXX:-$(lock_value supported_cxx)}"
out="${1:-$(project_build_dir)/g3r-checked-pair-test}"
mkdir -p "$out"
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v \
  tests.g3r.test_checked_pair_static >"$out/static.stdout" 2>"$out/static.stderr"
runner="$(eshkol_build_dir)/eshkol-run"
for mode in normal sanitize; do
  directory="$out/$mode"
  mkdir -p "$directory/cache"
  sanitizer_link=
  runtime=(env)
  if [[ "$mode" == sanitize ]]; then
    sanitizer_link='-fsanitize=address,undefined -fno-omit-frame-pointer'
    runtime=(env ASAN_OPTIONS=detect_leaks=1:halt_on_error=1
                 UBSAN_OPTIONS=halt_on_error=1)
  fi
  cat >"$directory/cxx-wrap" <<WRAPPER
#!/usr/bin/env bash
exec $(printf '%q' "$cxx") $sanitizer_link "\$@"
WRAPPER
  chmod 0500 "$directory/cxx-wrap"
  env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
    XDG_CACHE_HOME="$directory/cache" \
    ESHKOL_CXX_COMPILER="$directory/cxx-wrap" \
    ESHKOL_LIB_DIR="$PROJECT_ROOT/lib" \
    timeout --foreground --signal=TERM --kill-after=5s 600s \
    "$runner" --strict-types --optimize 0 --no-stdlib \
    -I "$PROJECT_ROOT/internal/c1/lib" -I "$PROJECT_ROOT/native" \
    "$PROJECT_ROOT/tests/g3r/checked_pair_test.esk" -o "$directory/test" \
    >"$directory/compile.stdout" 2>"$directory/compile.stderr"
  nm "$directory/test" >"$directory/symbols"
  grep -F 'g3r-checked-pair-admit-internal' "$directory/symbols" >/dev/null
  "${runtime[@]}" ESHKOL_ARENA_POISON=1 "$directory/test" \
    >"$directory/run.stdout" 2>"$directory/run.stderr"
  "${runtime[@]}" ESHKOL_ARENA_POISON=1 "$directory/test" \
    >"$directory/repeat.stdout" 2>"$directory/repeat.stderr"
  cmp "$directory/run.stdout" "$directory/repeat.stdout"
  test ! -s "$directory/compile.stderr"
  test ! -s "$directory/run.stderr"
  test ! -s "$directory/repeat.stderr"
  grep -Eq '^G3-R checked pair PASS: checks=[1-9][0-9]*$' \
    "$directory/run.stdout"
done
cmp "$out/normal/run.stdout" "$out/sanitize/run.stdout"
printf 'G3-R checked pair host gate PASS\n'
