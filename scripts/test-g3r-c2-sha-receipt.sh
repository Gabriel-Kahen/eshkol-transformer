#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
for command in ar cmp cp nm python3 realpath rg sed sha256sum stat timeout tr; do
  require_command "$command"
done
[[ $# -eq 1 ]] || die "usage: $0 FRESH_OUTPUT_DIRECTORY"
out="$(realpath -m -- "$1")"
[[ ! -e "$out" && ! -L "$out" ]] || die "G3-R C2 receipt output must be fresh"
mkdir -p "$out"
cc= cxx=
resolve_provenance_compilers cc cxx \
  "${CC:-$(lock_value supported_cc)}" "${CXX:-$(lock_value supported_cxx)}"
runner="$(eshkol_build_dir)/eshkol-run"
{
  printf 'cc\t%s\n' "$cc"
  printf 'cxx\t%s\n' "$cxx"
  printf 'cc_version\t%s\n' "$("$cc" --version | sed -n '1p')"
  printf 'cxx_version\t%s\n' "$("$cxx" --version | sed -n '1p')"
  printf 'eshkol_runner_sha256\t%s\n' "$(sha256sum "$runner" | cut -d ' ' -f 1)"
} >"$out/compiler-selected.tsv"
cp "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" \
  "$out/compiler-provenance.tsv"
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v \
  tests.g3r.test_c2_sha_receipt_static tests.q0.test_python_isolation \
  >"$out/static.stdout" 2>"$out/static.stderr"
/usr/bin/bash "$PROJECT_ROOT/scripts/test-g3r-c2-sha-stage.sh" \
  "$out/stage-predecessor" \
  >"$out/stage-predecessor.stdout" 2>"$out/stage-predecessor.stderr"
rg -x 'G3-R C2 SHA private stage PASS: normal/repeat/sanitize' \
  "$out/stage-predecessor.stdout" >/dev/null
test ! -s "$out/stage-predecessor.stderr"

python3 "$PROJECT_ROOT/tests/c2/prepare_checkpoint_load_fixtures.py" \
  "$out/fixtures"
python3 - "$out/fixtures/valid.c2" "$out/fixtures/truncated.c2" <<'PY'
from pathlib import Path
import sys
Path(sys.argv[2]).write_bytes(Path(sys.argv[1]).read_bytes()[:-1])
PY
sha256sum "$out/fixtures/valid.c2" "$out/fixtures/truncated.c2" \
  >"$out/fixture-sha256.txt"
test "$(stat -c %s "$out/fixtures/larger-valid.c2")" = \
     "$(stat -c %s "$out/fixtures/exact-valid.c2")"
sha256sum "$out/fixtures/larger-valid.c2" \
  "$out/fixtures/exact-valid.c2" >"$out/race-fixture-sha256.txt"

common=(-std=c11 -Wall -Wextra -Werror -Wpedantic -Wconversion
  -Wsign-conversion -Wshadow -ffp-contract=off -fexcess-precision=standard
  -frounding-math -fPIC -fvisibility=hidden -fno-common -fstack-protector-all
  -I "$PROJECT_ROOT/include" -I "$PROJECT_ROOT/native")
test_flags=(-DET_C2_CHECKPOINT_LOAD_TESTING -DET_C2_CHECKPOINT_CORE_TESTING
  -DET_C2_CHECKPOINT_READER_TESTING -DET_CHECKPOINT_IO_TESTING)

build_mode() {
  local mode="$1" directory="$out/$1" name cxx_quoted
  local -a flags=(-O2) runtime=(env)
  local -a eshkol_inputs=(
    -I "$PROJECT_ROOT/internal/p1/lib"
    -I "$PROJECT_ROOT/internal/c1/lib"
    -I "$PROJECT_ROOT/internal/t2/lib"
    -I "$PROJECT_ROOT/internal/t1/lib"
    -I "$PROJECT_ROOT/internal/d2/lib"
    -I "$PROJECT_ROOT/src" -I "$PROJECT_ROOT/lib"
    -I "$PROJECT_ROOT/native" -L "$directory"
    --lib g3r_c2_sha_stage
    "$PROJECT_ROOT/tests/g3r/c2_sha_receipt_test.esk")
  local sanitizer_link=
  if [[ "$mode" == sanitize ]]; then
    flags=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
    sanitizer_link='-fsanitize=address,undefined -fno-omit-frame-pointer'
    runtime=(env ASAN_OPTIONS=detect_leaks=1:halt_on_error=1
                 UBSAN_OPTIONS=halt_on_error=1
                 LSAN_OPTIONS=exitcode=23:report_objects=1)
  fi
  mkdir -p "$directory/objects" "$directory/cache"
  "$cc" "${common[@]}" "${flags[@]}" -c \
    "$PROJECT_ROOT/native/c2_checkpoint_load_bridge.c" \
    -o "$directory/load-off.o"
  if nm -g --defined-only "$directory/load-off.o" | \
      rg 'et_c2_private_checkpoint_load_stage_with_sha_v1'; then
    die "C2 SHA stage leaked into feature-off native object"
  fi
  "$cc" "${common[@]}" "${flags[@]}" "${test_flags[@]}" \
    -DET_I2_NATIVE_HELPERS_ONLY -MMD -MF "$directory/objects/i2.d" \
    -c "$PROJECT_ROOT/native/i2_wave2_package_bridge.c" \
    -o "$directory/objects/i2.o"
  "$cc" "${common[@]}" "${flags[@]}" "${test_flags[@]}" \
    -DET_O2_NATIVE_HELPERS_ONLY -DET_C2_O2_RECONSTRUCT_BRIDGE \
    -MMD -MF "$directory/objects/o2.d" \
    -c "$PROJECT_ROOT/native/o2_wave2_package_bridge.c" \
    -o "$directory/objects/o2.o"
  "$cc" "${common[@]}" "${flags[@]}" "${test_flags[@]}" \
    -DET_P1_TRUSTED_BUILD=1 -MMD -MF "$directory/objects/p1.d" \
    -c "$PROJECT_ROOT/native/p1_identity.c" \
    -o "$directory/objects/p1.o"
  for name in data_io kernel_abi t1_i64_shell f32_tensor i64_tensor \
              d2_native o2_optimizer c2_x1_canonical c2_checkpoint_format; do
    "$cc" "${common[@]}" "${flags[@]}" "${test_flags[@]}" \
      -MMD -MF "$directory/objects/$name.d" \
      -c "$PROJECT_ROOT/native/$name.c" -o "$directory/objects/$name.o"
  done
  "$cc" "${common[@]}" "${flags[@]}" "${test_flags[@]}" \
    -DET_C2_CARRIER_FACTORIES -DET_K2_TESTING \
    -MMD -MF "$directory/objects/k2.d" \
    -c "$PROJECT_ROOT/native/k2_capabilities.c" \
    -o "$directory/objects/k2.o"
  "$cc" "${common[@]}" "${flags[@]}" \
    -MMD -MF "$directory/objects/receipt-test-bridge.d" \
    -c "$PROJECT_ROOT/tests/g3r/c2_sha_receipt_test_bridge.c" \
    -o "$directory/objects/receipt-test-bridge.o"
  "$cc" "${common[@]}" "${flags[@]}" "${test_flags[@]}" \
    -MMD -MF "$directory/objects/io.d" \
    -c "$PROJECT_ROOT/native/checkpoint_io.c" -o "$directory/objects/io.o"
  "$cc" "${common[@]}" "${flags[@]}" "${test_flags[@]}" \
    -MMD -MF "$directory/objects/reader.d" \
    -c "$PROJECT_ROOT/native/c2_checkpoint_reader.c" \
    -o "$directory/objects/reader.o"
  "$cc" "${common[@]}" "${flags[@]}" "${test_flags[@]}" \
    -MMD -MF "$directory/objects/core.d" \
    -c "$PROJECT_ROOT/native/c2_checkpoint_core.c" \
    -o "$directory/objects/core.o"
  "$cc" "${common[@]}" "${flags[@]}" "${test_flags[@]}" \
    -DET_G3R_C2_STAGE_SHA_PRIVATE -MMD -MF "$directory/objects/load.d" \
    -c "$PROJECT_ROOT/native/c2_checkpoint_load_bridge.c" \
    -o "$directory/objects/load.o"
  nm -g --defined-only "$directory/objects/load.o" | \
    rg 'et_c2_private_checkpoint_load_stage_with_sha_v1' \
    >"$directory/feature-on-symbol.txt"
  nm -g --defined-only "$directory/load-off.o" \
    >"$directory/feature-off-symbols.txt"
  ar rcsD "$directory/libg3r_c2_sha_stage.a" "$directory/objects"/*.o
  nm -u --format=posix "$directory/libg3r_c2_sha_stage.a" \
    >"$directory/native-undefined-symbols.txt"
  nm -g --defined-only --format=posix \
    "$directory/libg3r_c2_sha_stage.a" \
    >"$directory/native-defined-symbols.txt"

  printf -v cxx_quoted '%q' "$cxx"
  cat >"$directory/cxx-wrap" <<WRAPPER
#!/usr/bin/env bash
exec $cxx_quoted $sanitizer_link "\$@"
WRAPPER
  chmod 0500 "$directory/cxx-wrap"
  (cd "$directory" &&
    env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
      XDG_CACHE_HOME="$directory/cache" ESHKOL_LIB_DIR="$PROJECT_ROOT/lib" \
      ESHKOL_CXX_COMPILER="$directory/cxx-wrap" \
      timeout --foreground --signal=TERM --kill-after=5s 900s "$runner" \
        --strict-types --optimize 0 --no-stdlib --dump-ir \
        "${eshkol_inputs[@]}" \
        -o "$directory/receipt-test" \
        >"$directory/build.stdout" 2>"$directory/build.stderr")
  test ! -s "$directory/build.stderr"
  test -s "$directory/receipt-test.ll"
  test -s "$directory/receipt-test"

  # Executable mode does not emit a depfile with this pinned compiler.
  # Use the exact same private root in object mode for closure evidence.
  (cd "$directory" &&
    env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
      XDG_CACHE_HOME="$directory/cache" ESHKOL_LIB_DIR="$PROJECT_ROOT/lib" \
      ESHKOL_CXX_COMPILER="$directory/cxx-wrap" \
      timeout --foreground --signal=TERM --kill-after=5s 900s "$runner" \
        --strict-types --optimize 0 --no-stdlib --emit-object \
        --emit-depfile "$directory/stage.d" \
        "${eshkol_inputs[@]}" \
        -o "$directory/stage-dep.o" \
        >"$directory/object-build.stdout" \
        2>"$directory/object-build.stderr")
  test ! -s "$directory/object-build.stderr"
  test -s "$directory/stage.d"
  test -s "$directory/stage-dep.o"
  nm -u --format=posix "$directory/stage-dep.o" \
    >"$directory/dep-object-undefined-symbols.txt"
  nm -g --defined-only --format=posix "$directory/stage-dep.o" \
    >"$directory/dep-object-defined-symbols.txt"
  "$cc" "${flags[@]}" -c -x ir "$directory/receipt-test.ll" \
    -o "$directory/receipt-test.o" \
    >"$directory/ir-object-build.stdout" \
    2>"$directory/ir-object-build.stderr"
  test ! -s "$directory/ir-object-build.stdout"
  python3 - "$directory/ir-object-build.stderr" <<'PY'
from pathlib import Path
import sys

warning = ("warning: <unknown>:0:0: loop not vectorized: the optimizer was "
           "unable to perform the requested transformation; the transformation "
           "might be disabled or specified as part of an unsupported "
           "transformation ordering [-Wpass-failed=transform-warning]\n")
diagnostics = Path(sys.argv[1]).read_text()
count = diagnostics.count(warning)
summary = f"{count} warning{'s' if count != 1 else ''} generated.\n" if count else ""
expected = warning * count + summary
if diagnostics != expected:
    raise SystemExit("unexpected LLVM IR object compiler diagnostics")
PY
  nm -u --format=posix "$directory/receipt-test.o" \
    >"$directory/eshkol-object-undefined-symbols.txt"
  nm -g --defined-only --format=posix "$directory/receipt-test.o" \
    >"$directory/eshkol-object-defined-symbols.txt"
  rg -F 'et_c2_private_checkpoint_load_stage_with_sha_v1' \
    "$directory/eshkol-object-undefined-symbols.txt" >/dev/null
  sed -e 's/^[^:]*://' -e 's/\\//g' "$directory/stage.d" | \
    tr -s '[:space:]' '\n' | rg "^$PROJECT_ROOT/" | \
    sed "s#^$PROJECT_ROOT/##" | \
    rg -v '^tests/g3r/c2_sha_receipt_test\.esk$' \
    >"$directory/source-closure.txt"
  cmp "$PROJECT_ROOT/native/g3r_c2_sha_receipt_source_closure.txt" \
      "$directory/source-closure.txt"
  rg -F 'et_c2_private_checkpoint_load_stage_with_sha_v1' \
    "$directory/receipt-test.ll" >/dev/null
  nm --format=posix "$directory/receipt-test" >"$directory/linked-symbols.txt"
  nm -u --format=posix "$directory/receipt-test" \
    >"$directory/linked-undefined-symbols.txt"
  cp "$out/fixtures/larger-valid.c2" "$directory/mutable.c2"
  cp "$out/fixtures/larger-valid.c2" "$directory/late-replacement.c2"
  cp "$out/fixtures/exact-valid.c2" "$directory/replacement.c2"
  "${runtime[@]}" ESHKOL_ARENA_POISON=1 \
    timeout --foreground --signal=TERM --kill-after=5s 360s \
    "$directory/receipt-test" "$out/fixtures/valid.c2" \
    "$out/fixtures/truncated.c2" "$out/fixtures/larger-valid.c2" \
    "$directory/replacement.c2" "$directory/mutable.c2" \
    "$directory/late-replacement.c2" \
    >"$directory/run.stdout" 2>"$directory/run.stderr"
  test ! -s "$directory/run.stderr"
  rg -x 'G3-R C2 SHA owning receipt PASS: [1-9][0-9]* checks' \
    "$directory/run.stdout" >/dev/null
}

build_mode normal
build_mode repeat
build_mode sanitize
cmp "$out/normal/run.stdout" "$out/repeat/run.stdout"
cmp "$out/normal/run.stdout" "$out/sanitize/run.stdout"
cmp "$out/normal/source-closure.txt" "$out/repeat/source-closure.txt"
cmp "$out/normal/source-closure.txt" "$out/sanitize/source-closure.txt"
printf 'G3-R C2 SHA owning receipt PASS: normal/repeat/sanitize\n'
