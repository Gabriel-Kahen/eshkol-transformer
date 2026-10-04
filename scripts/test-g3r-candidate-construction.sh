#!/usr/bin/env bash
# Mechanical evidence only. Full generated IR and diagnostics require review.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
cd -- "$PROJECT_ROOT"

(( $# == 1 )) || die "usage: $0 FRESH_ABSOLUTE_EVIDENCE_DIRECTORY"
evidence=$1
[[ "$evidence" == /* && ! -e "$evidence" && ! -L "$evidence" ]] || \
  die "evidence directory must be fresh and absolute"
mkdir -- "$evidence"

# Every compiler, linker and runtime invocation has a raw stream pair and
# numeric exit, including a failing invocation. The caller retains the tree.
run_checked() {
  local label=$1 status
  shift
  set +e
  "$@" >"$evidence/$label.stdout" 2>"$evidence/$label.stderr"
  status=$?
  set -e
  printf '%d\n' "$status" >"$evidence/$label.exit"
  (( status == 0 )) || die "$label exited $status"
}
run_mode() {
  local label=$1 mode=$2
  shift 2
  run_checked "$label" env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR \
    ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="$evidence/$mode.cache" \
    ESHKOL_CXX_COMPILER="$evidence/$mode/cxx-wrap" \
    ESHKOL_LIB_DIR="$PROJECT_ROOT/lib" \
    timeout --foreground --signal=TERM --kill-after=5s 900s "$runner" "$@"
}

run_checked q0 env PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v \
  tests.q0.test_g3r_candidate_construction_static
run_checked toolchain bash -c 'source "$1"; verify_toolchain' _ \
  "$PROJECT_ROOT/scripts/common.sh"
for command in ar awk cmp comm find git grep nm python3 sed sha256sum sort timeout xargs; do
  require_command "$command"
done
cc= cxx=
resolve_provenance_compilers cc cxx "${CC:-$(lock_value supported_cc)}" \
  "${CXX:-$(lock_value supported_cxx)}"
runner="$(eshkol_build_dir)/eshkol-run"
baseline=31188a6b163c0350e00b6ddd1e455bca510221b7
[[ "$(git -C "$PROJECT_ROOT" rev-parse "${baseline}^{tree}")" == \
   13fcb396163ab60baec6fea4278201022ab53bf3 ]] || die "baseline tree changed"
[[ -z "$(git -C "$PROJECT_ROOT" status --porcelain --untracked-files=all)" ]] || \
  die "source checkout is not clean"
git -C "$PROJECT_ROOT" rev-parse HEAD >"$evidence/source.commit"
git -C "$PROJECT_ROOT" rev-parse HEAD^{tree} >"$evidence/source.tree"
git -C "$PROJECT_ROOT" rev-parse "$baseline" >"$evidence/baseline.commit"
git -C "$PROJECT_ROOT" rev-parse "${baseline}^{tree}" >"$evidence/baseline.tree"
cp "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" \
  "$evidence/compiler-provenance.tsv"
run_checked compiler.cc-version "$cc" --version
run_checked compiler.cxx-version "$cxx" --version
run_checked compiler.runner-version "$runner" --version
printf 'cc\t%s\ncxx\t%s\nrunner\t%s\n' "$cc" "$cxx" "$runner" \
  >"$evidence/compiler.paths.tsv"

overlay="$evidence/overlay"
run_checked overlay env PYTHONDONTWRITEBYTECODE=1 python3 \
  "$PROJECT_ROOT/scripts/build-g3r-candidate-p1-overlay.py" "$overlay"
mkdir "$evidence/baseline"
mkdir "$evidence/baseline/transformer"
git -C "$PROJECT_ROOT" show \
  "$baseline:internal/p1/lib/transformer/module.esk" \
  >"$evidence/baseline/transformer/module.esk"
cmp "$PROJECT_ROOT/internal/p1/lib/transformer/module.esk" \
  "$evidence/baseline/transformer/module.esk" \
  >"$evidence/ordinary-source-cmp.stdout" \
  2>"$evidence/ordinary-source-cmp.stderr"
git -C "$PROJECT_ROOT" show "$baseline:native/p1_identity.c" \
  >"$evidence/baseline/p1_identity.c"
cmp "$PROJECT_ROOT/native/p1_identity.c" "$evidence/baseline/p1_identity.c" \
  >"$evidence/native-source-cmp.stdout" \
  2>"$evidence/native-source-cmp.stderr"

common_c=(-std=c11 -Wall -Wextra -Werror -Wpedantic -fPIC
  -fvisibility=hidden -fno-common -ffp-contract=off
  -fexcess-precision=standard -frounding-math
  -I "$PROJECT_ROOT/include" -I "$PROJECT_ROOT/native"
  -I "$PROJECT_ROOT/src" -DET_F32_TENSOR_TESTING)
compile_c() {
  local mode=$1 label=$2 source=$3
  shift 3
  run_checked "$mode.$label.compile" "$cc" "${common_c[@]}" \
    "${mode_flags[@]}" "$@" -MMD -MF "$evidence/$mode/$label.d" \
    -c "$source" -o "$evidence/$mode/$label.o"
}
common_esk=(--strict-types --optimize 0 --no-stdlib
  -I "$PROJECT_ROOT/internal/p1/lib" -I "$PROJECT_ROOT/internal/c1/lib"
  -I "$PROJECT_ROOT/internal/t1/lib" -I "$PROJECT_ROOT/src"
  -I "$PROJECT_ROOT/lib" -I "$PROJECT_ROOT/native")
fixture="$PROJECT_ROOT/tests/g3r/candidate_construction_test.esk"
ordinary_fixture="$PROJECT_ROOT/tests/p1/i2_construction_test.esk"

for mode in normal sanitize; do
  mkdir "$evidence/$mode" "$evidence/$mode.cache"
  mode_flags=(-O2)
  sanitizer_link=
  if [[ "$mode" == sanitize ]]; then
    mode_flags=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
    sanitizer_link='-fsanitize=address,undefined -fno-omit-frame-pointer'
  fi
  cat >"$evidence/$mode/cxx-wrap" <<WRAPPER
#!/usr/bin/env bash
exec $(printf '%q' "$cxx") $sanitizer_link "\$@"
WRAPPER
  chmod 0500 "$evidence/$mode/cxx-wrap"
  compile_c "$mode" i2_bridge "$PROJECT_ROOT/native/i2_wave2_package_bridge.c" \
    -DET_I2_NATIVE_HELPERS_ONLY -DET_M3T_PACKAGE_BUILD
  for source in data_io checkpoint_io kernel_abi i64_tensor t1_i64_shell \
                n3k_primitives_provider n2_primitives_provider a2_attention_provider; do
    compile_c "$mode" "$source" "$PROJECT_ROOT/native/$source.c"
  done
  for source in m3t_transport m3t_f32_integration; do
    compile_c "$mode" "$source" \
      "$PROJECT_ROOT/src/eshkol_transformer/$source.c"
  done
  compile_c "$mode" candidate_bridge \
    "$PROJECT_ROOT/tests/g3r/candidate_construction_bridge.c"
  run_checked "$mode.archive" ar rcsD \
    "$evidence/$mode/libg3r_candidate_construction.a" "$evidence/$mode"/*.o
  run_mode "$mode.object" "$mode" -I "$overlay" "${common_esk[@]}" \
    -L "$evidence/$mode" --lib g3r_candidate_construction \
    --emit-object --dump-ir --emit-depfile "$evidence/$mode/candidate.d" \
    "$fixture" -o "$evidence/$mode/candidate.raw.o"
  [[ -s "$evidence/$mode/candidate.raw.o.ll" && \
     -s "$evidence/$mode/candidate.raw.o.bc" && \
     -s "$evidence/$mode/candidate.d" ]] || die "$mode raw IR/bitcode/depfile missing"
  if [[ "$mode" == sanitize ]]; then
    run_checked "$mode.ir-instrument" "$cc" -O1 -g -fsanitize=address,undefined \
      -fno-omit-frame-pointer -x ir -c \
      "$evidence/$mode/candidate.raw.o.ll" \
      -o "$evidence/$mode/candidate.instrumented.o"
    candidate_object="$evidence/$mode/candidate.instrumented.o"
  else
    candidate_object="$evidence/$mode/candidate.raw.o"
  fi
  run_mode "$mode.link" "$mode" --no-stdlib -L "$evidence/$mode" \
    --lib g3r_candidate_construction "$candidate_object" \
    -o "$evidence/$mode/candidate"
  run_checked "$mode.run" env ESHKOL_ARENA_POISON=1 \
    ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
    UBSAN_OPTIONS=halt_on_error=1 \
    timeout --foreground --signal=TERM --kill-after=5s 120s \
    "$evidence/$mode/candidate"
  [[ ! -s "$evidence/$mode.run.stderr" ]] || die "$mode runtime stderr nonempty"
  grep -Eq '^G3-R candidate construction PASS: checks=[1-9][0-9]*$' \
    "$evidence/$mode.run.stdout" || die "$mode fixture did not pass"
  [[ "$(grep -Fxc 'G3-R owner and armed-row probes PASS' \
       "$evidence/$mode.run.stdout")" -eq 1 ]] || die "$mode armed-row marker"
  [[ "$(grep -Fxc 'G3-R consumed-active replay PASS' \
       "$evidence/$mode.run.stdout")" -eq 1 ]] || die "$mode replay marker"
  grep '^G3-R candidate fault stage PASS: ' "$evidence/$mode.run.stdout" \
    >"$evidence/$mode.fault-stages"
  printf '%s\n' 1 2 3 4 5 7 8 | \
    sed 's/^/G3-R candidate fault stage PASS: /' \
    >"$evidence/$mode.expected-fault-stages"
  cmp "$evidence/$mode.expected-fault-stages" \
    "$evidence/$mode.fault-stages" \
    >"$evidence/$mode.fault-cmp.stdout" \
    2>"$evidence/$mode.fault-cmp.stderr"
  nm -g --defined-only --format=posix "$evidence/$mode/candidate_bridge.o" \
    >"$evidence/$mode/candidate_bridge.defined.raw"
  nm -u --format=posix "$candidate_object" \
    >"$evidence/$mode/candidate.undefined.raw"
done

# One repeat of the same normal binary, plus an independently built ASan/UBSan
# binary whose Eshkol IR and native objects are instrumented by pinned Clang.
run_checked normal.repeat env ESHKOL_ARENA_POISON=1 \
  timeout --foreground --signal=TERM --kill-after=5s 120s \
  "$evidence/normal/candidate"
[[ ! -s "$evidence/normal.repeat.stderr" ]] || die "repeat stderr nonempty"
cmp "$evidence/normal.run.stdout" "$evidence/normal.repeat.stdout" \
  >"$evidence/repeat-cmp.stdout" 2>"$evidence/repeat-cmp.stderr"
cmp "$evidence/normal.run.stdout" "$evidence/sanitize.run.stdout" \
  >"$evidence/sanitize-cmp.stdout" 2>"$evidence/sanitize-cmp.stderr"

# Feature-off ordinary Eshkol compilation from the accepted source and the
# current installed source, with the same compiler/options. Preserve both raw
# objects and require exact byte parity. A path-sensitive difference fails.
mkdir "$evidence/off" "$evidence/off.cache"
mode_flags=(-O2)
compile_c off ordinary_bridge \
  "$PROJECT_ROOT/tests/p1/i2_construction_test_bridge.c"
for variant in baseline current; do
  include="$PROJECT_ROOT/internal/p1/lib"
  [[ "$variant" == baseline ]] && include="$evidence/baseline"
  run_checked "off.$variant.object" env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR \
    ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="$evidence/off.cache" \
    ESHKOL_CXX_COMPILER="$cxx" ESHKOL_LIB_DIR="$PROJECT_ROOT/lib" \
    timeout --foreground --signal=TERM --kill-after=5s 900s "$runner" \
    --strict-types --optimize 0 --no-stdlib -I "$include" \
    -I "$PROJECT_ROOT/internal/c1/lib" -I "$PROJECT_ROOT/internal/t1/lib" \
    -I "$PROJECT_ROOT/src" -I "$PROJECT_ROOT/lib" -I "$PROJECT_ROOT/native" \
    --emit-object --dump-ir \
    --emit-depfile "$evidence/off/$variant.d" "$ordinary_fixture" \
    -o "$evidence/off/$variant.o"
  nm -u --format=posix "$evidence/off/$variant.o" \
    >"$evidence/off/$variant.undefined.raw"
  if grep -Eq 'et_p1_private_candidate_|et_p1_test_candidate_' \
      "$evidence/off/$variant.undefined.raw"; then
    die "private candidate symbol escaped $variant ordinary object"
  fi
done
cmp "$evidence/off/baseline.o" "$evidence/off/current.o" \
  >"$evidence/off/object-cmp.stdout" \
  2>"$evidence/off/object-cmp.stderr"
cmp "$evidence/off/baseline.undefined.raw" \
  "$evidence/off/current.undefined.raw" \
  >"$evidence/off/symbol-cmp.stdout" \
  2>"$evidence/off/symbol-cmp.stderr"

for variant in baseline current; do
  source="$PROJECT_ROOT/native/p1_identity.c"
  [[ "$variant" == baseline ]] && source="$evidence/baseline/p1_identity.c"
  run_checked "off.$variant.native" "$cc" "${common_c[@]}" -O2 \
    -DET_P1_TRUSTED_BUILD=1 -MMD -MF "$evidence/off/$variant.native.d" \
    -c "$source" -o "$evidence/off/$variant.native.o"
done
cmp "$evidence/off/baseline.native.o" "$evidence/off/current.native.o" \
  >"$evidence/off/native-cmp.stdout" \
  2>"$evidence/off/native-cmp.stderr"

# The native candidate macro is forbidden in the public P1 compilation role.
# Plain diagnostics make the single intended preprocessor error an exact byte
# comparison; any warning, note, second error or infrastructure failure fails.
set +e
"$cc" "${common_c[@]}" -O2 \
  -fno-caret-diagnostics -fno-show-column \
  -fno-diagnostics-show-option -fno-color-diagnostics \
  -DET_G3R_CANDIDATE_RETIRE_PRIVATE=1 \
  -c "$PROJECT_ROOT/native/p1_identity.c" \
  -o "$evidence/off/forbidden-public.o" \
  >"$evidence/off.forbidden-public.stdout" \
  2>"$evidence/off.forbidden-public.stderr"
public_status=$?
set -e
printf '%d\n' "$public_status" >"$evidence/off.forbidden-public.exit"
[[ "$public_status" -eq 1 && ! -s "$evidence/off.forbidden-public.stdout" && \
   ! -e "$evidence/off/forbidden-public.o" ]] || \
  die "public-role candidate compile did not fail exactly before object emission"
printf '%s\n' "$PROJECT_ROOT/native/p1_identity.c:5: error: \"G3-R candidate P1 roster requires the trusted P1 build\"" \
  >"$evidence/off.forbidden-public.expected-stderr"
cmp "$evidence/off.forbidden-public.expected-stderr" \
  "$evidence/off.forbidden-public.stderr" \
  >"$evidence/off.forbidden-public.diagnostic-cmp.stdout" \
  2>"$evidence/off.forbidden-public.diagnostic-cmp.stderr"

nm -g --defined-only --format=posix "$evidence/normal/candidate_bridge.o" | \
  awk 'NF >= 2 { print $1 }' | LC_ALL=C sort -u \
  >"$evidence/private.defined"
nm -g --defined-only --format=posix "$evidence/off/ordinary_bridge.o" | \
  awk 'NF >= 2 { print $1 }' | LC_ALL=C sort -u \
  >"$evidence/off.native.defined"
comm -13 "$evidence/off.native.defined" "$evidence/private.defined" \
  >"$evidence/private.added"
comm -23 "$evidence/off.native.defined" "$evidence/private.defined" \
  >"$evidence/private.removed"
[[ ! -s "$evidence/private.removed" ]] || die "private bridge removed native symbols"
printf '%s\n' et_p1_private_candidate_construction_begin_v1 \
  et_p1_private_candidate_graph_preflight_v1 \
  et_p1_private_candidate_graph_revoke_v1 \
  et_p1_test_candidate_roster_bytes_v1 \
  et_p1_test_candidate_roster_entries_v1 | LC_ALL=C sort \
  >"$evidence/private.expected"
cmp "$evidence/private.expected" "$evidence/private.added" \
  >"$evidence/private-symbol-cmp.stdout" \
  2>"$evidence/private-symbol-cmp.stderr"
for mode in normal sanitize; do
  for symbol in et_p1_private_candidate_construction_begin_v1 \
                et_p1_test_candidate_roster_bytes_v1; do
    grep -Eq "^${symbol}[[:space:]]" \
      "$evidence/$mode/candidate.undefined.raw" || \
      die "$mode Eshkol object omits $symbol"
  done
done

run_checked inputs env PYTHONDONTWRITEBYTECODE=1 python3 \
  "$PROJECT_ROOT/tests/g3r/check_candidate_construction_inputs.py" \
  "$PROJECT_ROOT" "$evidence" "$(eshkol_source_dir)"
run_checked toolchain-post bash -c 'source "$1"; verify_toolchain' _ \
  "$PROJECT_ROOT/scripts/common.sh"
printf 'PENDING: Manual classification of every compiler/linker diagnostic and\n' \
  >"$evidence/MANUAL_REVIEW_PENDING.txt"
printf 'full raw candidate IR/ABI/ownership review is required. Exit zero\n' \
  >>"$evidence/MANUAL_REVIEW_PENDING.txt"
printf 'means only mechanical checks passed; it is not source or gate ACCEPT.\n' \
  >>"$evidence/MANUAL_REVIEW_PENDING.txt"
[[ "$(git -C "$PROJECT_ROOT" rev-parse HEAD)" == \
   "$(cat "$evidence/source.commit")" && \
   "$(git -C "$PROJECT_ROOT" rev-parse HEAD^{tree})" == \
   "$(cat "$evidence/source.tree")" && \
   -z "$(git -C "$PROJECT_ROOT" status --porcelain --untracked-files=all)" ]] || \
  die "source changed during gate"
find "$evidence" -type f ! -name SHA256SUMS -print0 | LC_ALL=C sort -z | \
  xargs -0 sha256sum >"$evidence/SHA256SUMS"
printf 'G3-R candidate construction mechanical evidence complete; manual IR/diagnostic review PENDING: %s\n' \
  "$evidence"
