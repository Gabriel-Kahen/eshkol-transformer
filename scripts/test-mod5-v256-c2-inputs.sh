#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
for command in ar cmp python3 timeout; do require_command "$command"; done
cc= cxx=
resolve_provenance_compilers cc cxx \
  "${CC:-$(lock_value supported_cc)}" "${CXX:-$(lock_value supported_cxx)}"
runner="$(eshkol_build_dir)/eshkol-run"
out="${1:-$(project_build_dir)/mod5-v256-c2-inputs}"
[[ ! -e "$out" && ! -L "$out" ]] || die "MOD5 V256 inputs evidence directory must be fresh"
mkdir -p "$out"
out="$(realpath "$out")"
cd "$PROJECT_ROOT"
export LC_ALL=C PYTHONDONTWRITEBYTECODE=1

python3 -B -m tests.mod5.mod5_v256_c2_fixture build "$out/fixture" >"$out/fixture-build.json"
python3 -B -m tests.mod5.mod5_v256_c2_fixture verify "$out/fixture" >"$out/fixture-verify.json"
cmp "$out/fixture-build.json" "$out/fixture-verify.json"

mkdir "$out/objects" "$out/cache"
flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -fPIC -fvisibility=hidden
       -fno-common -fstack-protector-all -I include -I native)
"$cc" "${flags[@]}" -DET_I2_NATIVE_HELPERS_ONLY \
  -c native/i2_wave2_package_bridge.c -o "$out/objects/i2_bridge.o"
"$cc" "${flags[@]}" -DET_P1_TRUSTED_BUILD=1 \
  -c native/p1_identity.c -o "$out/objects/p1_identity.o"
for source in data_io checkpoint_io kernel_abi t1_i64_shell f32_tensor i64_tensor d2_native; do
  "$cc" "${flags[@]}" -c "native/$source.c" -o "$out/objects/$source.o"
done
"$cc" "${flags[@]}" -c tests/d2/view_probe.c -o "$out/objects/view_probe.o"
ar rcsD "$out/libmod5_v256_c2_inputs.a" "$out"/objects/*.o

env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR -u ESHKOL_AOT_MODULE_CACHE_DIR \
  ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="$out/cache" ESHKOL_LIB_DIR="$PROJECT_ROOT/lib" \
  ESHKOL_CXX_COMPILER="$cxx" timeout --foreground --kill-after=5s 600s \
  "$runner" --strict-types --no-stdlib -O 0 -I native -I internal/p1/lib \
  -I internal/c1/lib -I internal/t2/lib -I internal/t1/lib -I internal/d2/lib -I src \
  -L "$out" --lib mod5_v256_c2_inputs tests/mod5/mod5_v256_c2_inputs_runtime.esk \
  -o "$out/witness" >"$out/compile.stdout" 2>"$out/compile.stderr"
[[ ! -s "$out/compile.stderr" ]] || die "MOD5 V256 inputs compilation emitted diagnostics"
for variant in a b; do
  mkdir "$out/$variant"
  timeout --foreground --kill-after=5s 90s "$out/witness" "$out/fixture" "$out/$variant" \
    >"$out/$variant.stdout" 2>"$out/$variant.stderr"
  [[ ! -s "$out/$variant.stderr" ]] || die "MOD5 V256 inputs witness emitted diagnostics"
done
cmp "$out/a.stdout" "$out/b.stdout"
diff -qr -- "$out/a" "$out/b"
python3 -B - "$out" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
from tests.x1.manifest_format import fingerprint

out = Path(sys.argv[1])
expected = json.loads((out / "fixture-verify.json").read_text())
observed = out / "a"
assert fingerprint((observed / "resolved-x1.json").read_bytes()) == expected["x1_fingerprint"]
assert (observed / "tokenizer-fingerprint.txt").read_text() == expected["tokenizer_fingerprint"]
actual = [hashlib.sha256((observed / f"cursor-{n}.bin").read_bytes()).hexdigest()
          for n in range(expected["total_updates"] + 1)]
assert actual == expected["oracle_cursor_sha256_by_next_ordinal"]
PY
grep -Fx 'MOD5 V256 X1/D2 PASS rows=6 split=2 suffix=4' "$out/a.stdout"
