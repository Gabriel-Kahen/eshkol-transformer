#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
for command in ar cmp python3 timeout; do require_command "$command"; done
cc= cxx=
resolve_provenance_compilers cc cxx \
  "${CC:-$(lock_value supported_cc)}" "${CXX:-$(lock_value supported_cxx)}"
runner="$(eshkol_build_dir)/eshkol-run"
out="${1:-$(project_build_dir)/mod5-v256-c2-stage-policy}"
[[ ! -e "$out" && ! -L "$out" ]] || die "MOD5 staged LOAD evidence directory must be fresh"
mkdir -p "$out"
out="$(realpath "$out")"
cd "$PROJECT_ROOT"
export LC_ALL=C PYTHONDONTWRITEBYTECODE=1

python3 -B - "$out" <<'PY'
from dataclasses import replace
from pathlib import Path
import sys

from tests.c2.test_checkpoint_format import make_fixture
from tests.eval5.mod5_reference_oracle import safe
from tests.eval5.mod5_v256_fixture import model_state

out = Path(sys.argv[1])
(out / "ordinary.c2").write_bytes(make_fixture()[0])
state = model_state()
(out / "untied.etcp").write_bytes(safe.encode_c1(state))
tensors = list(state.tensors)
tensors[11] = replace(tensors[11], data=tensors[9].data)
tied = replace(state, tensors=tuple(tensors), aliases=((9, 11),))
(out / "tied.etcp").write_bytes(safe.encode_c1(tied))
PY

mkdir "$out/objects" "$out/cache"
flags=(-std=c11 -O2 -Wall -Wextra -Werror -Wpedantic -Wconversion -Wshadow
       -Wsign-conversion -ffp-contract=off -fexcess-precision=standard
       -frounding-math -fno-fast-math -fPIC -fvisibility=hidden
       -fno-common -fstack-protector-all -I include -I native)
compile() {
  local source_path=$1
  shift
  "$cc" "${flags[@]}" "$@" -c "$source_path" \
    -o "$out/objects/$(basename "${source_path%.c}").o"
}
compile native/i2_wave2_package_bridge.c -DET_I2_NATIVE_HELPERS_ONLY -DET_C2_I2_MODEL_COPY
compile native/o2_wave2_package_bridge.c -DET_O2_NATIVE_HELPERS_ONLY -DET_C2_O2_RECONSTRUCT_BRIDGE
compile native/p1_identity.c -DET_P1_TRUSTED_BUILD=1
compile native/k2_capabilities.c -DET_C2_CARRIER_FACTORIES
for source in data_io checkpoint_io kernel_abi i64_tensor t1_i64_shell f32_tensor \
  d2_native o2_optimizer c2_x1_canonical c2_checkpoint_codec c2_checkpoint_format \
  c2_checkpoint_save_bridge c2_checkpoint_reader c2_checkpoint_core \
  c2_checkpoint_load_bridge c2_checkpoint_inspect_bridge; do
  compile "native/$source.c"
done
ar rcsD "$out/libmod5_v256_stage_policy.a" "$out"/objects/*.o

env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR -u ESHKOL_AOT_MODULE_CACHE_DIR \
  ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="$out/cache" ESHKOL_LIB_DIR="$PROJECT_ROOT/lib" \
  ESHKOL_CXX_COMPILER="$cxx" timeout --foreground --kill-after=5s 900s \
  "$runner" --strict-types --no-stdlib -O 0 -I native -I internal/p1/lib \
  -I internal/c1/lib -I internal/t2/lib -I internal/t1/lib -I internal/d2/lib -I src \
  -L "$out" --lib mod5_v256_stage_policy tests/c2/mod5_v256_stage_policy_runtime.esk \
  -o "$out/witness" >"$out/compile.stdout" 2>"$out/compile.stderr"
[[ ! -s "$out/compile.stderr" ]] || die "MOD5 staged LOAD compile diagnostics"
for variant in a b; do
  timeout --foreground --kill-after=5s 90s "$out/witness" "$out/ordinary.c2" \
    "$out/untied.etcp" "$out/tied.etcp" >"$out/$variant.stdout" 2>"$out/$variant.stderr"
  [[ ! -s "$out/$variant.stderr" ]] || die "MOD5 staged LOAD runtime diagnostics"
done
cmp "$out/a.stdout" "$out/b.stdout"
grep -Fx 'MOD5 V256 staged LOAD PASS' "$out/a.stdout"
