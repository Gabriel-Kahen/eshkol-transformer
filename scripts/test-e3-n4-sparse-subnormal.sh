#!/usr/bin/env bash
# One-use mechanical gate; independent source, IR, and result review follows.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
(( $# == 1 )) || die "usage: $0 FRESH_ABSOLUTE_EVIDENCE_DIRECTORY"
out=$1
[[ "$out" == /* && ! -e "$out" && ! -L "$out" ]] || \
  die "N4 evidence directory must be fresh and absolute"
mkdir -- "$out"
finish() {
  local status=$?
  trap - EXIT
  printf '%d\n' "$status" >"$out/gate.exit"
  (cd -- "$out" && find . -type f ! -name SHA256SUMS -print0 | \
    LC_ALL=C sort -z | xargs -0 -r sha256sum >SHA256SUMS)
  exit "$status"
}
trap finish EXIT
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
for command in ar cmp nm python3 sha256sum timeout find sort xargs; do
  require_command "$command"
done
cc= cxx=
resolve_provenance_compilers cc cxx "${CC:-$(lock_value supported_cc)}" \
  "${CXX:-$(lock_value supported_cxx)}"
runner="$(eshkol_build_dir)/eshkol-run"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || \
  die "N4 source must be committed and clean"
git rev-parse HEAD >"$out/source.commit"
git rev-parse HEAD^{tree} >"$out/source.tree"
git status --porcelain=v1 --untracked-files=all >"$out/source.status"
git ls-files -z tests/e3_reference/n4_sparse_bits.py \
  tests/e3_private/overflow_driver.esk tests/e3_private/overflow_runtime.esk \
  tests/e3_private/parse_n4_sparse_observation.py \
  src/eshkol_transformer/e3_frame.c scripts/build-e1b-consumer.sh \
  scripts/test-e3-n4-sparse-subnormal.sh scripts/build-e3-overflow-test.sh | \
  xargs -0 -r sha256sum >"$out/source-inputs.sha256"
cp "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" \
  "$out/compiler-provenance.tsv"
run_checked toolchain.cc "$cc" --version
run_checked toolchain.cxx "$cxx" --version
run_checked toolchain.runner sha256sum "$runner"
run_checked static env PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest -v \
  tests.e3_reference.test_n4_sparse_bits tests.e3_private.test_n4_sparse_tuple \
  tests.e3_private.test_overflow_tuple
run_checked fixture env PYTHONDONTWRITEBYTECODE=1 python3 -B -m \
  tests.e3_reference.n4_sparse_bits "$out/fixture"
mkdir "$out/package-builds"

for mode in normal repeat sanitize; do
  mkdir "$out/$mode" "$out/$mode.cache"
  link_flags=
  [[ "$mode" != sanitize ]] || \
    link_flags='-fsanitize=address,undefined -fno-omit-frame-pointer'
  cat >"$out/$mode/cxx-wrap" <<WRAPPER
#!/usr/bin/env bash
exec $(printf '%q' "$cxx") $link_flags "\$@"
WRAPPER
  chmod 0500 "$out/$mode/cxx-wrap"
  package_mode=normal
  [[ "$mode" != sanitize ]] || package_mode=sanitize
  run_checked "$mode.package" env E3_N4_SPARSE_TESTING=1 bash \
    "$PROJECT_ROOT/scripts/build-e3-overflow-test.sh" \
    "$out/package-builds/$mode" "$package_mode"
  arena_flags=(-O2)
  [[ "$mode" != sanitize ]] || \
    arena_flags=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
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
    for run in ordinary poison; do
      poison=0
      [[ "$run" != poison ]] || poison=1
      label="$mode.$gradient.$run"
      run_checked "$label.run" env ESHKOL_ARENA_POISON="$poison" \
        ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
        UBSAN_OPTIONS=halt_on_error=1 \
        timeout --foreground --signal=TERM --kill-after=5s 120s \
        "$out/$mode/runtime" sparse "$out/fixture/sparse.etcp" \
        "$out/fixture/sparse-corpus" "$gradient"
      [[ ! -s "$out/$label.run.stderr" ]] || \
        die "$label emitted runtime diagnostics"
      run_checked "$label.trace" env PYTHONDONTWRITEBYTECODE=1 python3 -B \
        -m tests.e3_private.parse_n4_sparse_observation \
        --stdout "$out/$label.run.stdout" --gradient "$gradient"
    done
  done
done
cmp "$out/package-builds/normal/e3_overflow_package.o" \
    "$out/package-builds/repeat/e3_overflow_package.o"
cmp "$out/normal/runtime" "$out/repeat/runtime"
printf 'N4 sparse mechanical gate complete; independent result review pending\n'
