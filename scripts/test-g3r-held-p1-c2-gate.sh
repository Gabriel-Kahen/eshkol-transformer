#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
for command in ar nm python3 timeout; do require_command "$command"; done
cc= cxx=
resolve_provenance_compilers cc cxx \
  "${CC:-$(lock_value supported_cc)}" "${CXX:-$(lock_value supported_cxx)}"
evidence="${1:?pass a writable evidence directory outside the source checkout}"
mkdir -p "$evidence/native" "$evidence/off" "$evidence/cache"
evidence="$(readlink -f "$evidence")"
[[ "$evidence" != "$PROJECT_ROOT" && "$evidence" != "$PROJECT_ROOT/"* ]] || \
  die "evidence directory must be outside the source checkout"

PYTHONDONTWRITEBYTECODE=1 python3 \
  "$PROJECT_ROOT/scripts/check-g3r-held-p1-c2-gate.py" \
  >"$evidence/source.stdout"
(cd "$PROJECT_ROOT" && PYTHONDONTWRITEBYTECODE=1 \
  python3 -m tests.d2.prepare_public_resources --output "$evidence/corpus")

common=(-std=c11 -Wall -Wextra -Werror -Wpedantic -fPIC
  -fvisibility=hidden -fno-common -fstack-protector-all -ffp-contract=off
  -fexcess-precision=standard -fno-fast-math -I "$PROJECT_ROOT/include"
  -I "$PROJECT_ROOT/native" -I "$PROJECT_ROOT/src"
  -I "$PROJECT_ROOT/src/eshkol_transformer" -I "$(eshkol_source_dir)/inc")
"$cc" "${common[@]}" -O2 -c "$PROJECT_ROOT/native/o2_optimizer.c" \
  -o "$evidence/off/o2_optimizer.o"
"$cc" "${common[@]}" -O2 -c \
  "$PROJECT_ROOT/src/eshkol_transformer/g3t_transport.c" \
  -o "$evidence/off/g3t_transport.o"
if nm -g --defined-only "$evidence/off/o2_optimizer.o" | \
    grep -Fq et_g3r_private_o2_save_begin_v1; then
  die "G3-R O2 hold escaped the feature-off object"
fi
if nm -g --defined-only "$evidence/off/g3t_transport.o" | \
    grep -Fq et_g3r_private_g3t_save_begin_v1; then
  die "G3-R G3 hold escaped the feature-off object"
fi
compile() {
  local source="$1" stem="$2"
  shift 2
  "$cc" "${common[@]}" -O2 "$@" -c "$PROJECT_ROOT/$source" \
    -o "$evidence/native/$stem.o"
}
compile native/i2_wave2_package_bridge.c i2_bridge \
  -DET_I2_NATIVE_HELPERS_ONLY -DET_M3T_PACKAGE_BUILD \
  -DET_C2_I2_MODEL_COPY
compile native/o2_wave2_package_bridge.c o2_bridge \
  -DET_O2_NATIVE_HELPERS_ONLY -DET_C2_O2_RECONSTRUCT_BRIDGE \
  -DET_G3R_SAME_OWNER_SAVE_PRIVATE
compile native/p1_identity.c p1_identity -DET_P1_TRUSTED_BUILD=1
for source in data_io checkpoint_io kernel_abi t1_i64_shell d2_native \
              c2_x1_canonical n2_primitives_provider \
              n3k_primitives_provider a2_attention_provider \
              g3n_primitives_provider g3s_sampling_provider; do
  compile "native/$source.c" "$source"
done
compile src/eshkol_transformer/m3_i64_integration.c m3_i64_integration \
  -DET_I64_TENSOR_TESTING -DET_I64_TENSOR_STORAGE_QUERY_PRIVATE
# M3 call includes M3T F32, which includes native/f32_tensor.c. Compile it
# once so public F32 symbols and the private registry have one linked owner.
compile src/eshkol_transformer/m3_call_f32_integration.c m3_call_f32_integration \
  -DET_F32_TENSOR_TESTING -DET_F32_TENSOR_STORAGE_QUERY_PRIVATE \
  -DET_G3R_SAME_OWNER_SAVE_PRIVATE -DET_G3R_CANDIDATE_RETIRE_PRIVATE
"$PROJECT_ROOT/scripts/check-g3r-f32-allocator-symbol.sh" \
  "$evidence/native/m3_call_f32_integration.o" \
  "$evidence/native/m3_call_f32_integration.defined-symbols.txt" || \
  die "G3-R M3/F32 owner lacks the linked F32 test allocator"
compile native/o2_optimizer.c o2_optimizer \
  -DET_G3R_SAME_OWNER_SAVE_PRIVATE -DET_G3R_CANDIDATE_RETIRE_PRIVATE
compile native/a2_kv_cache.c a2_kv_cache \
  -DET_A2_KV_CACHE_TESTING -DET_A2_KV_CACHE_STORAGE_QUERY_PRIVATE
compile src/eshkol_transformer/g3t_transport.c g3t_transport \
  -DET_G3T_TESTING -DET_F32_TENSOR_TESTING \
  -DET_A2_KV_CACHE_TESTING -DET_I64_TENSOR_TESTING \
  -DET_G3T_PREFILL_SAMPLE_PRIVATE -DET_G3T_OUTPUT_TEXT_PRIVATE \
  -DET_G3T_FINAL_PUBLICATION_PRIVATE -DET_G3T_ZERO_BUDGET_PRIVATE \
  -DET_G3T_P2_ZERO_BUDGET_PRIVATE -DET_G3T_FULL_REQUEST_PREFLIGHT_PRIVATE \
  -DET_G3T_MANUAL_LOGITS_PRIVATE -DET_G3T_MANUAL_P1_PREFILL_PRIVATE \
  -DET_G3T_MANUAL_P2_PREFILL_PRIVATE -DET_G3T_MANUAL_DECODE_PRIVATE \
  -DET_G3T_MANUAL_LOGITS_MATERIALIZE_PRIVATE \
  -DET_G3T_INPUT_FROM_T1_PRIVATE -DET_G3T_OWNED_TOKEN_INPUT_PRIVATE \
  -DET_G3T_OUTPUT_RNG_CLONE_PRIVATE -DET_G3T_GENERATOR_RNG_PRIVATE \
  -DET_G3R_COMMITTED_HISTORY_PRIVATE \
  -DET_G3R_COMMITTED_SAVE_SNAPSHOT_PRIVATE \
  -DET_G3R_SAME_OWNER_SAVE_PRIVATE \
  -DET_I64_TENSOR_STORAGE_QUERY_PRIVATE \
  -DET_A2_KV_CACHE_STORAGE_QUERY_PRIVATE \
  -DET_F32_TENSOR_STORAGE_QUERY_PRIVATE
compile native/g3r_joint_hold.c g3r_joint_hold \
  -DET_G3R_COMMITTED_HISTORY_PRIVATE -DET_G3R_SAME_OWNER_SAVE_PRIVATE
compile tests/g3r/held_p1_c2_bit_bridge.c held_p1_c2_bit_bridge
ar rcsD "$evidence/native/libg3r_held_p1_c2_gate.a" \
  "$evidence/native"/*.o
# A separate native/f32_tensor.o would duplicate M3's included F32 registry.
archive="$evidence/native/libg3r_held_p1_c2_gate.a"
if ar t "$archive" | grep -Fxq f32_tensor.o; then
  die "G3-R archive contains a second F32 owner"
fi
LC_ALL=C nm -g --defined-only --format=posix "$archive" > \
  "$evidence/native/archive-defined-symbols.txt"
PYTHONDONTWRITEBYTECODE=1 python3 - \
  "$evidence/native/archive-defined-symbols.txt" <<'PY_ARCHIVE'
from collections import Counter
from pathlib import Path
import sys

symbols = [line.split()[0] for line in Path(sys.argv[1]).read_text().splitlines()
           if line.startswith("et_f32_")]
counts = Counter(symbols)
if counts["et_f32_tensor_test_fail_alloc_after_v1"] != 1:
    raise SystemExit("G3-R archive lacks exactly one linked F32 test allocator")
repeated = sorted(symbol for symbol, count in counts.items() if count != 1)
if repeated:
    raise SystemExit("G3-R archive has duplicate F32 definitions: " +
                     ", ".join(repeated))
PY_ARCHIVE

env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
  XDG_CACHE_HOME="$evidence/cache" ESHKOL_CXX_COMPILER="$cxx" \
  ESHKOL_LIB_DIR="$PROJECT_ROOT/lib" \
  timeout --foreground --signal=TERM --kill-after=5s 1200s \
  "$(eshkol_build_dir)/eshkol-run" --strict-types --optimize 0 --no-stdlib \
  -I "$PROJECT_ROOT/internal/p1/lib" -I "$PROJECT_ROOT/internal/c1/lib" \
  -I "$PROJECT_ROOT/internal/t2/lib" -I "$PROJECT_ROOT/internal/t1/lib" \
  -I "$PROJECT_ROOT/internal/d2/lib" -I "$PROJECT_ROOT/src" \
  -I "$PROJECT_ROOT/lib" -I "$PROJECT_ROOT/native" \
  -L "$evidence/native" --lib g3r_held_p1_c2_gate \
  "$PROJECT_ROOT/tests/g3r/held_p1_c2_gate.esk" \
  -o "$evidence/held-p1-c2" \
  >"$evidence/compile.stdout" 2>"$evidence/compile.stderr"
test ! -s "$evidence/compile.stderr"
ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM \
  --kill-after=5s 300s "$evidence/held-p1-c2" \
  "$evidence/corpus/small" \
  >"$evidence/runtime.stdout" 2>"$evidence/runtime.stderr"
PYTHONDONTWRITEBYTECODE=1 python3 - "$evidence/runtime.stderr" <<'PY_PHASES'
from pathlib import Path
import re
import sys

lines = Path(sys.argv[1]).read_text().splitlines()
phases = []
for line in lines:
    match = re.fullmatch(r"G3R_PHASE ([0-9]+)", line)
    if match is None:
        raise SystemExit(f"G3-R runtime emitted non-phase stderr: {line}")
    phases.append(int(match.group(1)))
if phases[:30] != list(range(30)) or not phases or phases[-1] != 999:
    raise SystemExit(f"G3-R runtime phase sequence incomplete: {phases}")
PY_PHASES
grep -E '^G3-R held P1/C2 source aggregate PASS: checks=[1-9][0-9]*$' \
  "$evidence/runtime.stdout" >/dev/null
cat "$evidence/source.stdout"
cat "$evidence/runtime.stdout"
