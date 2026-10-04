#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

(( $# == 1 )) || die "usage: $0 FRESH_ABSOLUTE_EVIDENCE_DIRECTORY"
evidence=$1
[[ "$evidence" == /* && ! -e "$evidence" && ! -L "$evidence" ]] || \
  die "evidence directory must be fresh and absolute"
mkdir -- "$evidence"
(cd "$PROJECT_ROOT" && PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v \
  tests.q0.test_g3r_candidate_construction_static) \
  >"$evidence/static.stdout" 2>"$evidence/static.stderr"
verify_toolchain >"$evidence/toolchain.stdout" 2>"$evidence/toolchain.stderr"
for command in ar awk cmp comm find grep nm python3 sha256sum sort timeout xargs; do
  require_command "$command"
done
cc= cxx=
resolve_provenance_compilers cc cxx "${CC:-$(lock_value supported_cc)}" \
  "${CXX:-$(lock_value supported_cxx)}"
runner="$(eshkol_build_dir)/eshkol-run"
overlay="$evidence/overlay"
PYTHONDONTWRITEBYTECODE=1 python3 \
  "$PROJECT_ROOT/scripts/build-g3r-candidate-p1-overlay.py" "$overlay" \
  >"$evidence/overlay.stdout" 2>"$evidence/overlay.stderr"

flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -fPIC -fvisibility=hidden
       -fno-common -ffp-contract=off -fexcess-precision=standard -frounding-math
       -I "$PROJECT_ROOT/include" -I "$PROJECT_ROOT/native" -I "$PROJECT_ROOT/src"
       -DET_F32_TENSOR_TESTING)
"$cc" "${flags[@]}" -DET_I2_NATIVE_HELPERS_ONLY -DET_M3T_PACKAGE_BUILD \
  -c "$PROJECT_ROOT/native/i2_wave2_package_bridge.c" \
  -o "$evidence/i2_bridge.o" \
  >"$evidence/i2_bridge.compile.stdout" \
  2>"$evidence/i2_bridge.compile.stderr"
for source in data_io checkpoint_io kernel_abi i64_tensor t1_i64_shell \
              n3k_primitives_provider n2_primitives_provider a2_attention_provider; do
  "$cc" "${flags[@]}" -c "$PROJECT_ROOT/native/$source.c" \
    -o "$evidence/$source.o" \
    >"$evidence/$source.compile.stdout" \
    2>"$evidence/$source.compile.stderr"
done
for source in m3t_transport m3t_f32_integration; do
  "$cc" "${flags[@]}" -c "$PROJECT_ROOT/src/eshkol_transformer/$source.c" \
    -o "$evidence/$source.o" \
    >"$evidence/$source.compile.stdout" \
    2>"$evidence/$source.compile.stderr"
done
"$cc" "${flags[@]}" -c \
  "$PROJECT_ROOT/tests/g3r/candidate_construction_bridge.c" \
  -o "$evidence/candidate_bridge.o" \
  >"$evidence/candidate_bridge.compile.stdout" \
  2>"$evidence/candidate_bridge.compile.stderr"
mkdir "$evidence/off"
"$cc" "${flags[@]}" -c \
  "$PROJECT_ROOT/tests/p1/i2_construction_test_bridge.c" \
  -o "$evidence/off/ordinary_bridge.o" \
  >"$evidence/ordinary_bridge.compile.stdout" \
  2>"$evidence/ordinary_bridge.compile.stderr"
nm -g --defined-only "$evidence/candidate_bridge.o" \
  >"$evidence/candidate_bridge.defined.txt"
for symbol in et_p1_private_candidate_construction_begin_v1 \
              et_p1_private_candidate_graph_preflight_v1 \
              et_p1_private_candidate_graph_revoke_v1; do
  grep -Eq "[[:space:]]${symbol}$" "$evidence/candidate_bridge.defined.txt" || \
    die "private native P1 symbol missing: $symbol"
done
nm -g --defined-only --format=posix "$evidence/off/ordinary_bridge.o" |
  awk 'NF >= 2 { print $1 }' | LC_ALL=C sort -u \
  >"$evidence/ordinary_bridge.symbols"
nm -g --defined-only --format=posix "$evidence/candidate_bridge.o" |
  awk 'NF >= 2 { print $1 }' | LC_ALL=C sort -u \
  >"$evidence/candidate_bridge.symbols"
comm -13 "$evidence/ordinary_bridge.symbols" \
  "$evidence/candidate_bridge.symbols" >"$evidence/private-added.symbols"
comm -23 "$evidence/ordinary_bridge.symbols" \
  "$evidence/candidate_bridge.symbols" >"$evidence/private-removed.symbols"
test ! -s "$evidence/private-removed.symbols"
printf '%s\n' et_p1_private_candidate_construction_begin_v1 \
  et_p1_private_candidate_graph_preflight_v1 \
  et_p1_private_candidate_graph_revoke_v1 \
  et_p1_test_candidate_roster_bytes_v1 \
  et_p1_test_candidate_roster_entries_v1 | LC_ALL=C sort \
  >"$evidence/private-expected.symbols"
cmp "$evidence/private-expected.symbols" "$evidence/private-added.symbols"
ar rcsD "$evidence/libg3r_candidate_construction.a" "$evidence"/*.o

mkdir "$evidence/cache"
common_compile=(--strict-types --optimize 0 --no-stdlib
  -I "$overlay" -I "$PROJECT_ROOT/internal/p1/lib"
  -I "$PROJECT_ROOT/internal/c1/lib" -I "$PROJECT_ROOT/internal/t1/lib"
  -I "$PROJECT_ROOT/src" -I "$PROJECT_ROOT/lib" -I "$PROJECT_ROOT/native"
  -L "$evidence" --lib g3r_candidate_construction)
fixture="$PROJECT_ROOT/tests/g3r/candidate_construction_test.esk"
env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
  XDG_CACHE_HOME="$evidence/cache" ESHKOL_CXX_COMPILER="$cxx" \
  ESHKOL_LIB_DIR="$PROJECT_ROOT/lib" \
  timeout --foreground --signal=TERM --kill-after=5s 600s "$runner" \
  "${common_compile[@]}" --compile-only --emit-depfile "$evidence/candidate.d" \
  "$fixture" -o "$evidence/candidate.o" \
  >"$evidence/object.compile.stdout" 2>"$evidence/object.compile.stderr"
grep -F "$overlay/transformer/module.esk" "$evidence/candidate.d" \
  >"$evidence/overlay-dependency.txt" || die "compiled closure omitted private P1 overlay"
if grep -F "$PROJECT_ROOT/internal/p1/lib/transformer/module.esk" \
  "$evidence/candidate.d" >"$evidence/ordinary-module-dependency.txt"; then
  die "private aggregate loaded the ordinary P1 module"
fi
env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
  XDG_CACHE_HOME="$evidence/cache" ESHKOL_CXX_COMPILER="$cxx" \
  ESHKOL_LIB_DIR="$PROJECT_ROOT/lib" \
  timeout --foreground --signal=TERM --kill-after=5s 600s "$runner" \
  "${common_compile[@]}" "$fixture" -o "$evidence/candidate" \
  >"$evidence/executable.compile.stdout" \
  2>"$evidence/executable.compile.stderr"
for pass in normal repeat; do
  ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM --kill-after=5s \
    120s "$evidence/candidate" >"$evidence/$pass.stdout" \
    2>"$evidence/$pass.stderr"
  grep -Eq '^G3-R candidate construction PASS: checks=[1-9][0-9]*$' \
    "$evidence/$pass.stdout"
  test "$(grep -Fxc 'G3-R owner and armed-row probes PASS' \
            "$evidence/$pass.stdout")" -eq 1
  test ! -s "$evidence/$pass.stderr"
  grep '^G3-R candidate fault stage PASS: ' "$evidence/$pass.stdout" \
    >"$evidence/$pass.fault-stages"
  printf '%s\n' \
    'G3-R candidate fault stage PASS: 1' \
    'G3-R candidate fault stage PASS: 2' \
    'G3-R candidate fault stage PASS: 3' \
    'G3-R candidate fault stage PASS: 4' \
    'G3-R candidate fault stage PASS: 5' \
    'G3-R candidate fault stage PASS: 7' \
    'G3-R candidate fault stage PASS: 8' \
    >"$evidence/$pass.expected-fault-stages"
  cmp "$evidence/$pass.expected-fault-stages" "$evidence/$pass.fault-stages"
done
cmp "$evidence/normal.stdout" "$evidence/repeat.stdout"
find "$evidence" -type f ! -name SHA256SUMS -print0 | LC_ALL=C sort -z | \
  xargs -0 sha256sum \
  >"$evidence/SHA256SUMS"
printf 'G3-R candidate construction development fixture PASS: %s\n' "$evidence"
