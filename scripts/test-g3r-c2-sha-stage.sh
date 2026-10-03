#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
for command in ar cmp nm python3 realpath rg sed timeout tr; do
  require_command "$command"
done
[[ $# -eq 1 ]] || die "usage: $0 FRESH_OUTPUT_DIRECTORY"
out="$(realpath -m -- "$1")"
[[ ! -e "$out" && ! -L "$out" ]] || die "G3-R C2 SHA output must be fresh"
mkdir -p "$out/objects" "$out/cache"
cc= cxx=
resolve_provenance_compilers cc cxx \
  "${CC:-$(lock_value supported_cc)}" "${CXX:-$(lock_value supported_cxx)}"
runner="$(eshkol_build_dir)/eshkol-run"
python3 "$PROJECT_ROOT/tests/c2/prepare_checkpoint_load_fixtures.py" \
  "$out/fixtures"
python3 - "$out/fixtures/valid.c2" "$out/fixtures/truncated.c2" <<'PY'
from pathlib import Path
import sys
Path(sys.argv[2]).write_bytes(Path(sys.argv[1]).read_bytes()[:-1])
PY

common=(-std=c11 -Wall -Wextra -Werror -Wpedantic -Wconversion
  -Wsign-conversion -Wshadow -ffp-contract=off -fexcess-precision=standard
  -frounding-math -fPIC -fvisibility=hidden -fno-common -fstack-protector-all
  -I "$PROJECT_ROOT/include" -I "$PROJECT_ROOT/native")
test_flags=(-DET_C2_CHECKPOINT_LOAD_TESTING -DET_C2_CHECKPOINT_CORE_TESTING
  -DET_C2_CHECKPOINT_READER_TESTING -DET_CHECKPOINT_IO_TESTING)
"$cc" "${common[@]}" -c "$PROJECT_ROOT/native/c2_checkpoint_load_bridge.c" \
  -o "$out/load-off.o"
if nm -g --defined-only "$out/load-off.o" | \
    rg 'et_c2_private_checkpoint_load_stage_with_sha_v1'; then
  die "C2 SHA stage leaked into feature-off native object"
fi

"$cc" "${common[@]}" "${test_flags[@]}" -DET_I2_NATIVE_HELPERS_ONLY \
  -c "$PROJECT_ROOT/native/i2_wave2_package_bridge.c" -o "$out/objects/i2.o"
"$cc" "${common[@]}" "${test_flags[@]}" -DET_O2_NATIVE_HELPERS_ONLY \
  -DET_C2_O2_RECONSTRUCT_BRIDGE \
  -c "$PROJECT_ROOT/native/o2_wave2_package_bridge.c" -o "$out/objects/o2.o"
"$cc" "${common[@]}" "${test_flags[@]}" -DET_P1_TRUSTED_BUILD=1 \
  -c "$PROJECT_ROOT/native/p1_identity.c" -o "$out/objects/p1.o"
for name in data_io kernel_abi t1_i64_shell f32_tensor i64_tensor \
            d2_native o2_optimizer c2_x1_canonical c2_checkpoint_format; do
  "$cc" "${common[@]}" "${test_flags[@]}" \
    -c "$PROJECT_ROOT/native/$name.c" -o "$out/objects/$name.o"
done
"$cc" "${common[@]}" "${test_flags[@]}" -DET_C2_CARRIER_FACTORIES \
  -c "$PROJECT_ROOT/native/k2_capabilities.c" -o "$out/objects/k2.o"
"$cc" "${common[@]}" "${test_flags[@]}" \
  -c "$PROJECT_ROOT/native/checkpoint_io.c" -o "$out/objects/io.o"
"$cc" "${common[@]}" "${test_flags[@]}" \
  -c "$PROJECT_ROOT/native/c2_checkpoint_reader.c" -o "$out/objects/reader.o"
"$cc" "${common[@]}" "${test_flags[@]}" \
  -c "$PROJECT_ROOT/native/c2_checkpoint_core.c" -o "$out/objects/core.o"
"$cc" "${common[@]}" "${test_flags[@]}" \
  -DET_G3R_C2_STAGE_SHA_PRIVATE \
  -c "$PROJECT_ROOT/native/c2_checkpoint_load_bridge.c" \
  -o "$out/objects/load.o"
nm -g --defined-only "$out/objects/load.o" | \
  rg 'et_c2_private_checkpoint_load_stage_with_sha_v1' >/dev/null
ar rcsD "$out/libg3r_c2_sha_stage.a" "$out/objects"/*.o

env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
  XDG_CACHE_HOME="$out/cache" ESHKOL_LIB_DIR="$PROJECT_ROOT/lib" \
  ESHKOL_CXX_COMPILER="$cxx" \
  timeout --foreground --signal=TERM --kill-after=5s 900s "$runner" \
    --strict-types --optimize 0 --no-stdlib --emit-depfile "$out/stage.d" \
    -I "$PROJECT_ROOT/internal/p1/lib" \
    -I "$PROJECT_ROOT/internal/c1/lib" \
    -I "$PROJECT_ROOT/internal/t2/lib" \
    -I "$PROJECT_ROOT/internal/t1/lib" \
    -I "$PROJECT_ROOT/internal/d2/lib" \
    -I "$PROJECT_ROOT/src" -I "$PROJECT_ROOT/lib" -I "$PROJECT_ROOT/native" \
    -L "$out" --lib g3r_c2_sha_stage \
    "$PROJECT_ROOT/tests/g3r/c2_sha_stage_test.esk" -o "$out/stage-test" \
    >"$out/build.stdout" 2>"$out/build.stderr"
test ! -s "$out/build.stderr"
sed -e 's/^[^:]*://' -e 's/\\//g' "$out/stage.d" | \
  tr -s '[:space:]' '\n' | rg "^$PROJECT_ROOT/" | \
  sed "s#^$PROJECT_ROOT/##" | \
  rg -v '^tests/g3r/c2_sha_stage_test\.esk$' >"$out/source-closure.txt"
cmp "$PROJECT_ROOT/native/g3r_c2_sha_stage_source_closure.txt" \
    "$out/source-closure.txt"
ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM --kill-after=5s \
  360s "$out/stage-test" "$out/fixtures/valid.c2" \
  "$out/fixtures/truncated.c2" >"$out/run.stdout" 2>"$out/run.stderr"
test ! -s "$out/run.stderr"
rg -x 'G3-R C2 SHA Eshkol stage PASS: [0-9]+ checks' "$out/run.stdout" >/dev/null
printf 'G3-R C2 SHA private stage PASS: %s\n' "$out"
