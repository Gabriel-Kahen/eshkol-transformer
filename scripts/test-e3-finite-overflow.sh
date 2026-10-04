#!/usr/bin/env bash
# Mechanical evidence only: full IR, imports and diagnostics need independent review.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
(( $# == 1 )) || die "usage: $0 FRESH_ABSOLUTE_EVIDENCE_DIRECTORY"
out=$1
[[ "$out" == /* && ! -e "$out" && ! -L "$out" ]] || \
  die "E3 overflow evidence directory must be fresh and absolute"
mkdir -- "$out"
cd -- "$PROJECT_ROOT"
run_checked() {
  local label=$1 status
  shift
  set +e
  "$@" >"$out/$label.stdout" 2>"$out/$label.stderr"
  status=$?
  set -e
  printf '%d\n' "$status" >"$out/$label.exit"
  (( status == 0 )) || die "$label exited $status"
}
run_mode() {
  local label=$1 mode=$2
  shift 2
  run_checked "$label" env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR \
    ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="$out/$mode.cache" \
    ESHKOL_CXX_COMPILER="$out/$mode/cxx-wrap" \
    ESHKOL_LIB_DIR="$PROJECT_ROOT/lib" \
    timeout --foreground --signal=TERM --kill-after=5s 900s "$runner" "$@"
}
verify_toolchain
for command in ar cmp nm python3 sha256sum stdbuf timeout; do require_command "$command"; done
cc= cxx=
resolve_provenance_compilers cc cxx "${CC:-$(lock_value supported_cc)}" \
  "${CXX:-$(lock_value supported_cxx)}"
runner="$(eshkol_build_dir)/eshkol-run"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || \
  die "E3 overflow source must be committed and clean"
git rev-parse HEAD >"$out/source.commit"
git rev-parse HEAD^{tree} >"$out/source.tree"
cp "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" \
  "$out/compiler-provenance.tsv"
run_checked toolchain.cc "$cc" --version
run_checked toolchain.cxx "$cxx" --version
run_checked toolchain.runner sha256sum "$runner"
run_checked toolchain.stdbuf sha256sum "$(command -v stdbuf)"
run_checked toolchain.timeout sha256sum "$(command -v timeout)"
run_checked toolchain.stdbuf-preload stdbuf -o0 bash -c \
  'printf "%s\n" "${LD_PRELOAD:-}"'
[[ ! -s "$out/toolchain.stdbuf-preload.stderr" ]] || \
  die "stdbuf preload discovery emitted diagnostics"
preload="$(cat "$out/toolchain.stdbuf-preload.stdout")"
[[ "$preload" == /* && -f "$preload" && ! -L "$preload" ]] || \
  die "stdbuf preload library is not a regular absolute file"
run_checked toolchain.stdbuf-library sha256sum "$preload"
run_checked static env PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest -v \
  tests.e3_private.test_overflow_tuple
run_checked fixture env PYTHONDONTWRITEBYTECODE=1 python3 -B -m \
  tests.e3_reference.finite_overflow "$out/fixture"
mkdir "$out/package-builds"

for mode in normal repeat sanitize; do
  mkdir "$out/$mode" "$out/$mode.cache"
  link_flags=
  if [[ "$mode" == sanitize ]]; then
    link_flags='-fsanitize=address,undefined -fno-omit-frame-pointer'
  fi
  cat >"$out/$mode/cxx-wrap" <<WRAPPER
#!/usr/bin/env bash
exec $(printf '%q' "$cxx") $link_flags "\$@"
WRAPPER
  chmod 0500 "$out/$mode/cxx-wrap"
  package_mode=normal
  [[ "$mode" != sanitize ]] || package_mode=sanitize
  run_checked "$mode.package" bash \
    "$PROJECT_ROOT/scripts/build-e3-overflow-test.sh" \
    "$out/package-builds/$mode" "$package_mode"
  arena_flags=(-O2)
  if [[ "$mode" == sanitize ]]; then
    arena_flags=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
  fi
  run_checked "$mode.arena" "$cxx" -std=c++17 -Wall -Wextra -Werror \
    -fstack-protector-all -fPIC -fvisibility=hidden -fno-common \
    "${arena_flags[@]}" -I "$(eshkol_source_dir)/lib/core" \
    -I "$(eshkol_source_dir)/inc" -MMD -MF "$out/$mode/arena.d" \
    -c "$PROJECT_ROOT/tests/e3_private/retention_arena.cpp" \
    -o "$out/$mode/arena.o"
  run_checked "$mode.archive" ar rcsD \
    "$out/$mode/libeshkol_transformer_e3_overflow_test.a" \
    "$out/package-builds/$mode/e3_overflow_package.o" "$out/$mode/arena.o"
  run_mode "$mode.object" "$mode" --strict-types --no-stdlib -O 2 \
    -L "$out/$mode" --lib eshkol_transformer_e3_overflow_test \
    --emit-object --dump-ir --emit-depfile "$out/$mode/runtime.d" \
    "$PROJECT_ROOT/tests/e3_private/overflow_runtime.esk" \
    -o "$out/$mode/runtime.raw.o"
  [[ -s "$out/$mode/runtime.raw.o" && -s "$out/$mode/runtime.raw.o.ll" && \
     -s "$out/$mode/runtime.raw.o.bc" && -s "$out/$mode/runtime.d" ]] || \
    die "$mode raw IR/bitcode/depfile missing"
  object="$out/$mode/runtime.raw.o"
  if [[ "$mode" == sanitize ]]; then
    run_checked "$mode.asan-prepare" env PYTHONDONTWRITEBYTECODE=1 python3 -B \
      "$PROJECT_ROOT/tests/shared/prepare_eshkol_asan_ir.py" \
      "$out/$mode/runtime.raw.o.ll" "$out/$mode/runtime.asan.ll" \
      "$out/$mode/runtime.asan.json"
    run_checked "$mode.asan-compile" "$cc" -O1 -g \
      -fsanitize=address,undefined -fno-omit-frame-pointer -x ir -c \
      "$out/$mode/runtime.asan.ll" -o "$out/$mode/runtime.asan.o"
    run_checked "$mode.asan-symbols" nm -u --format=posix \
      "$out/$mode/runtime.asan.o"
    grep -E '^__asan_report_(load|store)' "$out/$mode.asan-symbols.stdout" \
      >/dev/null || die "generated Eshkol code lacks ASan memory reporters"
    object="$out/$mode/runtime.asan.o"
  fi
  run_mode "$mode.link" "$mode" --no-stdlib -L "$out/$mode" \
    --lib eshkol_transformer_e3_overflow_test "$object" \
    -o "$out/$mode/runtime"
  for gradient in absent present; do
    for case_name in finite overflow; do
      run_checked "$mode.$gradient.$case_name.run" env \
        ESHKOL_ARENA_POISON=1 ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
        UBSAN_OPTIONS=halt_on_error=1 \
        timeout --foreground --signal=TERM --kill-after=5s 120s \
        "$out/$mode/runtime" "$case_name" \
        "$out/fixture/$case_name.etcp" "$out/fixture/finite-overflow" \
        "$gradient"
      [[ ! -s "$out/$mode.$gradient.$case_name.run.stderr" ]] || \
        die "$mode.$gradient.$case_name emitted runtime diagnostics"
      [[ "$(grep -Fc "E3-FINITE-OVERFLOW-PASS gradient=$gradient" \
          "$out/$mode.$gradient.$case_name.run.stdout")" == 1 ]] || \
        die "$mode.$gradient.$case_name did not pass"
      run_checked "$mode.$gradient.$case_name.trace" env \
        PYTHONDONTWRITEBYTECODE=1 python3 -B \
        -m tests.e3_private.parse_overflow_observation \
        --stdout "$out/$mode.$gradient.$case_name.run.stdout" \
        --case "$case_name" --gradient "$gradient"
    done
  done
done
cmp "$out/package-builds/normal/e3_overflow_package.o" \
    "$out/package-builds/repeat/e3_overflow_package.o"
cmp "$out/normal/runtime" "$out/repeat/runtime"

# The clamp is an isolated noninstalled package; the exact same ordinary E3
# runtime object is relinked with that provider. No new Eshkol source is used.
mkdir "$out/mutant" "$out/mutant.cache"
cp "$out/normal/cxx-wrap" "$out/mutant/cxx-wrap"
run_checked mutant.package bash "$PROJECT_ROOT/scripts/build-e3-overflow-test.sh" \
  "$out/package-builds/mutant" normal clamp
[[ -s "$out/package-builds/mutant/e3_overflow_package.o.evidence/e3_overflow_clamp_provider.c" && \
   -s "$out/package-builds/mutant/e3_overflow_package.o.evidence/e3_overflow_clamp_receipt.json" ]] || \
  die "mutant overlay provenance was not retained"
run_checked mutant.archive ar rcsD \
  "$out/mutant/libeshkol_transformer_e3_overflow_test.a" \
  "$out/package-builds/mutant/e3_overflow_package.o" "$out/normal/arena.o"
run_mode mutant.link mutant --no-stdlib -L "$out/mutant" \
  --lib eshkol_transformer_e3_overflow_test "$out/normal/runtime.raw.o" \
  -o "$out/mutant/runtime"
for gradient in absent present; do
  run_checked "mutant.$gradient.finite.run" env ESHKOL_ARENA_POISON=1 \
    timeout --foreground --signal=TERM --kill-after=5s 120s \
    "$out/mutant/runtime" finite "$out/fixture/finite.etcp" \
    "$out/fixture/finite-overflow" "$gradient"
  [[ ! -s "$out/mutant.$gradient.finite.run.stderr" ]] || \
    die "mutant finite control emitted diagnostics"
  cmp "$out/normal.$gradient.finite.run.stdout" \
      "$out/mutant.$gradient.finite.run.stdout" || \
    die "mutant changed finite control trace or published bytes"
done

# The fixture's expected rejection check prints a labeled failure and calls
# native abort if the mutant publishes. Validate that stdbuf actually retains
# an unflushed C stdout line before classifying any mutant exit.
cat >"$out/mutant/unbuffered-control.c" <<'C'
#include <stdio.h>
#include <stdlib.h>
int main(void) { fputs("E3-UNBUFFERED-CONTROL\n", stdout); abort(); }
C
run_checked mutant.unbuffered-compile "$cc" -std=c11 -Wall -Wextra -Werror \
  "$out/mutant/unbuffered-control.c" -o "$out/mutant/unbuffered-control"
ulimit -c 0
run_abort() {
  local label=$1
  shift
  run_checked "$label.capture" env PYTHONDONTWRITEBYTECODE=1 python3 -B \
    -m tests.e3_private.capture_abort --prefix "$out/$label" -- "$@"
  [[ "$(cat "$out/$label.exit")" == 134 ]] || \
    die "$label did not exit through SIGABRT"
}
printf 'timeout: the monitored command dumped core\n' \
  >"$out/mutant/expected-abort.stderr"
run_abort mutant.unbuffered-control env LC_ALL=C \
  timeout --foreground --signal=TERM --kill-after=5s 120s \
  stdbuf -o0 -e0 "$out/mutant/unbuffered-control"
printf 'E3-UNBUFFERED-CONTROL\n' >"$out/mutant/expected-control.stdout"
cmp "$out/mutant/expected-control.stdout" "$out/mutant.unbuffered-control.stdout"
cmp "$out/mutant/expected-abort.stderr" "$out/mutant.unbuffered-control.stderr"
for gradient in absent present; do
  run_abort "mutant.$gradient.overflow.run" env LC_ALL=C ESHKOL_ARENA_POISON=1 \
    timeout --foreground --signal=TERM --kill-after=5s 120s \
    stdbuf -o0 -e0 "$out/mutant/runtime" overflow \
    "$out/fixture/overflow.etcp" "$out/fixture/finite-overflow" "$gradient"
  cmp "$out/mutant/expected-abort.stderr" \
      "$out/mutant.$gradient.overflow.run.stderr" || \
    die "mutant runtime diagnostic differed from the pinned abort control"
  run_checked "mutant.$gradient.overflow.trace" env PYTHONDONTWRITEBYTECODE=1 \
    python3 -B -m tests.e3_private.parse_overflow_observation \
    --stdout "$out/mutant.$gradient.overflow.run.stdout" \
    --case overflow --gradient "$gradient" --mutant clamp
done
printf 'E3 overflow mechanical gate complete; manual IR/diagnostics review pending\n'
