#!/usr/bin/env bash
# Supported local linked owner gate; writes evidence outside the checkout.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
for command in ar cmp nm python3 realpath rg sha256sum timeout; do
  require_command "$command"
done
[[ $# == 1 ]] || die "usage: $0 FRESH_EVIDENCE_DIRECTORY"
out="$(realpath -m -- "$1")"
[[ ! -e "$out" && ! -L "$out" ]] || die "linked evidence directory must be fresh"
[[ "$out" != "$PROJECT_ROOT" && "$out" != "$PROJECT_ROOT/"* ]] || \
  die "linked evidence must remain outside the checkout"
mkdir -p "$out/objects" "$out/cache"
cc= cxx=
resolve_provenance_compilers cc cxx \
  "${CC:-$(lock_value supported_cc)}" "${CXX:-$(lock_value supported_cxx)}"
runner="$(eshkol_build_dir)/eshkol-run"
[[ -x "$runner" ]] || die "pinned Eshkol runner is unavailable"
PYTHONDONTWRITEBYTECODE=1 python3 -m tests.tr3b.prepare_fixture \
  --output "$out/corpus"

flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -fPIC
  -fvisibility=hidden -fno-common -fstack-protector-all -ffp-contract=off
  -fexcess-precision=standard -frounding-math
  -I "$PROJECT_ROOT/include" -I "$PROJECT_ROOT/native"
  -I "$PROJECT_ROOT/src" -I "$PROJECT_ROOT/src/eshkol_transformer")
compile() {
  local source=$1 object=$2
  shift 2
  "$cc" "${flags[@]}" "$@" -c "$PROJECT_ROOT/$source" \
    -o "$out/objects/$object.o"
}
{
  compile native/data_io.c data_io
  compile native/checkpoint_io.c checkpoint_io
  compile native/kernel_abi.c kernel_abi
  compile native/t1_i64_shell.c t1_i64_shell
  compile src/eshkol_transformer/m3_i64_integration.c m3_i64_integration \
    -DET_I64_TENSOR_TESTING -DET_I64_TENSOR_STORAGE_QUERY_PRIVATE
  compile src/eshkol_transformer/m3_call_f32_integration.c m3_call_f32_integration \
    -DET_F32_TENSOR_TESTING -DET_F32_TENSOR_STORAGE_QUERY_PRIVATE \
    -DET_G3R_CANDIDATE_RETIRE_PRIVATE \
    -DET_I2_PRIVATE_OWNED_CLONE_MATCH
  compile native/d2_native.c d2_native \
    -DET_TR3_C_D2_RESTORE -DET_D2_NATIVE_TESTING \
    -DET_I64_TENSOR_TESTING
  compile native/a2_kv_cache.c a2_kv_cache \
    -DET_A2_KV_CACHE_TESTING -DET_A2_KV_CACHE_STORAGE_QUERY_PRIVATE
  for name in n2_primitives_provider n3k_primitives_provider \
      a2_attention_provider g3n_primitives_provider \
      g3s_sampling_provider indexed_cross_entropy \
      l3s_masked_objective_provider tr3b_objective_bridge \
      c2_x1_canonical; do
    compile "native/$name.c" "$name"
  done
  compile native/o2_optimizer.c o2_optimizer \
    -DET_O2_TESTING -DET_G3R_SAME_OWNER_SAVE_PRIVATE \
    -DET_G3R_CANDIDATE_RETIRE_PRIVATE \
    -DET_I2_PRIVATE_OWNED_CLONE_MATCH -DET_TR3_O2_STEP_CLEAR_NATIVE
  compile native/p1_identity.c p1_identity -DET_P1_TRUSTED_BUILD=1
  compile native/i2_wave2_package_bridge.c i2_bridge \
    -DET_I2_NATIVE_HELPERS_ONLY -DET_M3T_PACKAGE_BUILD \
    -DET_C2_I2_MODEL_COPY -DET_I2_PRIVATE_OWNED_CLONE_MATCH
  compile native/o2_wave2_package_bridge.c o2_bridge \
    -DET_O2_NATIVE_HELPERS_ONLY -DET_C2_O2_RECONSTRUCT_BRIDGE \
    -DET_TR3_O2_STEP_CLEAR_BRIDGE -DET_G3R_SAME_OWNER_SAVE_PRIVATE \
    -DET_G3R_CANDIDATE_RETIRE_PRIVATE
  compile src/eshkol_transformer/g3t_transport.c g3t_transport \
    -DET_G3R_COMMITTED_HISTORY_PRIVATE -DET_G3R_SAME_OWNER_SAVE_PRIVATE \
    -DET_G3R_COMMITTED_SAVE_SNAPSHOT_PRIVATE \
    -DET_G3T_PREFILL_SAMPLE_PRIVATE -DET_G3T_OUTPUT_TEXT_PRIVATE \
    -DET_G3T_FINAL_PUBLICATION_PRIVATE -DET_G3T_ZERO_BUDGET_PRIVATE \
    -DET_G3T_P2_ZERO_BUDGET_PRIVATE -DET_G3T_FULL_REQUEST_PREFLIGHT_PRIVATE \
    -DET_G3T_MANUAL_LOGITS_PRIVATE -DET_G3T_MANUAL_P1_PREFILL_PRIVATE \
    -DET_G3T_MANUAL_P2_PREFILL_PRIVATE -DET_G3T_MANUAL_DECODE_PRIVATE \
    -DET_G3T_MANUAL_LOGITS_MATERIALIZE_PRIVATE \
    -DET_G3T_INPUT_FROM_T1_PRIVATE -DET_G3T_OWNED_TOKEN_INPUT_PRIVATE \
    -DET_G3T_OUTPUT_RNG_CLONE_PRIVATE -DET_G3T_GENERATOR_RNG_PRIVATE \
    -DET_F32_TENSOR_TESTING -DET_A2_KV_CACHE_TESTING \
    -DET_I64_TENSOR_TESTING -DET_I64_TENSOR_STORAGE_QUERY_PRIVATE \
    -DET_A2_KV_CACHE_STORAGE_QUERY_PRIVATE \
    -DET_F32_TENSOR_STORAGE_QUERY_PRIVATE
  compile native/g3r_trainer_g3_save_bridge.c g3_bridge \
    -DET_G3R_COMMITTED_HISTORY_PRIVATE -DET_G3R_SAME_OWNER_SAVE_PRIVATE
  ar rcsD "$out/libg3r_linked_owner.a" "$out/objects"/*.o
} >"$out/native-build.stdout" 2>"$out/native-build.stderr"
nm -g --defined-only "$out/libg3r_linked_owner.a" \
  >"$out/native-defined.txt"
for symbol in et_g3r_private_trainer_g3_save_begin_v1 \
    et_g3r_private_trainer_o2_save_begin_v1 \
    et_g3r_private_o2_adopter_issue_v1; do
  rg -q "[[:space:]]${symbol}$" "$out/native-defined.txt" || \
    die "linked native archive lacks $symbol"
done

env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
  XDG_CACHE_HOME="$out/cache" ESHKOL_LIB_DIR="$PROJECT_ROOT/lib" \
  ESHKOL_CXX_COMPILER="$cxx" \
  timeout --foreground --signal=TERM --kill-after=5s 1200s "$runner" \
    --strict-types --no-stdlib -O 0 \
    -I "$PROJECT_ROOT/native" -I "$PROJECT_ROOT/internal/p1/lib" \
    -I "$PROJECT_ROOT/internal/c1/lib" -I "$PROJECT_ROOT/internal/t2/lib" \
    -I "$PROJECT_ROOT/internal/t1/lib" -I "$PROJECT_ROOT/internal/d2/lib" \
    -I "$PROJECT_ROOT/src" -I "$PROJECT_ROOT/lib" \
    -L "$out" --lib g3r_linked_owner \
    "$PROJECT_ROOT/tests/g3r/original_owner_linked_runtime.esk" \
    -o "$out/linked-owner" \
    >"$out/compile.stdout" 2>"$out/compile.stderr"
[[ -s "$out/linked-owner" && ! -s "$out/compile.stderr" ]] || \
  die "linked owner compilation failed"
for mode in normal typed false post-cleanup-typed post-cleanup-false; do
  ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM --kill-after=5s 180s \
    "$out/linked-owner" "$out/corpus" "$mode" \
    >"$out/$mode.stdout" 2>"$out/$mode.stderr"
  [[ ! -s "$out/$mode.stderr" ]] || die "$mode wrote stderr"
  rg -q "^G3R LINKED OWNER PASS: $mode [0-9]+ checks$" \
    "$out/$mode.stdout" || die "$mode missed its pass marker"
done
if ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM --kill-after=5s 180s \
    "$out/linked-owner" "$out/corpus" post-forgery \
    >"$out/post-forgery.stdout" 2>"$out/post-forgery.stderr"; then
  status=0
else
  status=$?
fi
[[ "$status" == 134 ]] || die "post-compose forgery did not abort: $status"
rg -q '^G3R LINKED OWNER POST-COMPOSE FORGERY$' \
  "$out/post-forgery.stderr" || die "forgery did not reach the claim check"
! rg -q 'G3R LINKED OWNER (FORGERY ESCAPED|PASS)' \
  "$out/post-forgery.stdout" || die "forgery escaped fail-stop"
{
  printf 'commit\t%s\n' "$(git -C "$PROJECT_ROOT" rev-parse HEAD)"
  printf 'tree\t%s\n' "$(git -C "$PROJECT_ROOT" rev-parse HEAD^{tree})"
  printf 'runner_sha256\t%s\n' "$(sha256sum "$runner" | cut -d ' ' -f 1)"
  printf 'native_archive_sha256\t%s\n' \
    "$(sha256sum "$out/libg3r_linked_owner.a" | cut -d ' ' -f 1)"
  printf 'result\tPASS\n'
} >"$out/manifest.tsv"
printf 'G3R linked owner runtime PASS: %s\n' "$out"
