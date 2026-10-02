#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
for command in cmp cp env grep nm python3 timeout; do require_command "$command"; done
cc= cxx=
resolve_provenance_compilers cc cxx \
  "${CC:-$(lock_value supported_cc)}" "${CXX:-$(lock_value supported_cxx)}"
eshkol_runner="$(eshkol_build_dir)/eshkol-run"
evidence="${1:-$(project_build_dir)/g3r-record-codec-test}"
mkdir -p "$evidence"
python3 "$PROJECT_ROOT/scripts/check-g3r-record-codec.py" \
  >"$evidence/static.stdout"
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v \
  tests.q0.test_python_isolation \
  >"$evidence/q0.stdout" 2>"$evidence/q0.stderr"

temporary="$(mktemp -d "$evidence/build.XXXXXX")"
trap 'status=$?; if (( status == 0 )); then rm -rf -- "$temporary"; else
  printf "G3-R codec build retained: %s\n" "$temporary" >&2; fi' EXIT

compile_mode() {
  local mode="$1" directory="$temporary/$1" fixture source prefix
  local -a runtime=(env)
  local sanitizer_link=
  if [[ "$mode" == sanitize ]]; then
    sanitizer_link='-fsanitize=address,undefined -fno-omit-frame-pointer'
    runtime=(env ASAN_OPTIONS=detect_leaks=1:halt_on_error=1
                 UBSAN_OPTIONS=halt_on_error=1)
  fi
  mkdir -p "$directory/cache"
  cat >"$directory/cxx-wrap" <<WRAPPER
#!/usr/bin/env bash
exec $(printf '%q' "$cxx") $sanitizer_link "\$@"
WRAPPER
  chmod 0500 "$directory/cxx-wrap"
  for fixture in off on; do
    prefix="$fixture"
    source="$PROJECT_ROOT/tests/g3r/record_codec_test.esk"
    if [[ "$fixture" == off ]]; then
      source="$PROJECT_ROOT/tests/g3r/record_codec_feature_off_test.esk"
    fi
    mkdir -p "$directory/cache/$fixture"
    env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
      XDG_CACHE_HOME="$directory/cache/$fixture" \
      ESHKOL_CXX_COMPILER="$directory/cxx-wrap" \
      ESHKOL_LIB_DIR="$PROJECT_ROOT/lib" \
      timeout --foreground --signal=TERM --kill-after=5s 600s \
      "$eshkol_runner" --strict-types --optimize 0 --no-stdlib \
      -I "$PROJECT_ROOT/internal/c1/lib" -I "$PROJECT_ROOT/native" \
      "$source" -o "$directory/$fixture" \
      >"$directory/$prefix-compile.stdout" \
      2>"$directory/$prefix-compile.stderr"
    nm "$directory/$fixture" >"$directory/$prefix.symbols"
    "${runtime[@]}" ESHKOL_ARENA_POISON=1 \
      timeout --foreground --signal=TERM --kill-after=5s 120s \
      "$directory/$fixture" >"$directory/$prefix-run.stdout" \
      2>"$directory/$prefix-run.stderr"
    "${runtime[@]}" ESHKOL_ARENA_POISON=1 \
      timeout --foreground --signal=TERM --kill-after=5s 120s \
      "$directory/$fixture" >"$directory/$prefix-repeat.stdout" \
      2>"$directory/$prefix-repeat.stderr"
    cmp "$directory/$prefix-run.stdout" "$directory/$prefix-repeat.stdout"
    test ! -s "$directory/$prefix-compile.stderr"
    test ! -s "$directory/$prefix-run.stderr"
    test ! -s "$directory/$prefix-repeat.stderr"
  done
  grep -F 'g3r-record-encode!' "$directory/on.symbols" >/dev/null
  grep -F 'g3r-record-decode!' "$directory/on.symbols" >/dev/null
  if grep -F 'g3r-record-' "$directory/off.symbols" >/dev/null; then
    printf 'feature-off G3-R private symbol leaked\n' >&2
    return 1
  fi
  python3 "$PROJECT_ROOT/tests/g3r/check_record_codec_oracle.py" \
    "$directory/on-run.stdout" >"$directory/oracle.stdout"
  grep -Eq '^G3-R record codec off PASS: checks=2$' \
    "$directory/off-run.stdout"
  grep -Eq '^G3-R record codec PASS: checks=[1-9][0-9]*$' \
    "$directory/on-run.stdout"
  mkdir -p "$evidence/$mode"
  cp "$directory"/{off,on}-{compile,run,repeat}.{stdout,stderr} \
    "$evidence/$mode/"
  cp "$directory"/{off,on}.symbols "$directory/oracle.stdout" \
    "$evidence/$mode/"
}

compile_mode normal
compile_mode sanitize
for fixture in off on; do
  cmp "$evidence/normal/$fixture-run.stdout" \
    "$evidence/sanitize/$fixture-run.stdout"
done
cmp "$evidence/normal/oracle.stdout" "$evidence/sanitize/oracle.stdout"
printf 'G3-R record codec supported gate PASS\n'
