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
compile native/f32_tensor.c f32_tensor \
  -DET_F32_TENSOR_TESTING -DET_F32_TENSOR_STORAGE_QUERY_PRIVATE \
  -DET_G3R_SAME_OWNER_SAVE_PRIVATE
nm -g --defined-only "$evidence/native/f32_tensor.o" | \
  grep -Fq et_f32_tensor_test_fail_alloc_after_v1 || \
  die "G3-R provider cut lacks the linked F32 test allocator"
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
compile src/eshkol_transformer/m3_call_f32_integration.c m3_call_f32_integration \
  -DET_F32_TENSOR_TESTING -DET_F32_TENSOR_STORAGE_QUERY_PRIVATE \
  -DET_G3R_SAME_OWNER_SAVE_PRIVATE
compile native/o2_optimizer.c o2_optimizer \
  -DET_G3R_SAME_OWNER_SAVE_PRIVATE
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
test ! -s "$evidence/runtime.stderr"
grep -E '^G3-R held P1/C2 source aggregate PASS: checks=[1-9][0-9]*$' \
  "$evidence/runtime.stdout" >/dev/null
cat "$evidence/source.stdout"
cat "$evidence/runtime.stdout"
