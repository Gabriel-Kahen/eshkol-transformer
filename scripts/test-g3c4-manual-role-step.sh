#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
for command in cmp comm env git grep nm python3 sha256sum timeout; do require_command "$command"; done
cc= cxx=
resolve_provenance_compilers cc cxx "${CC:-$(lock_value supported_cc)}" "${CXX:-$(lock_value supported_cxx)}"
evidence="${1:-$(project_build_dir)/g3c4-manual-role-step-test}"
mkdir -p "$evidence"
python3 "${PROJECT_ROOT}/scripts/check-g3c4-manual-role-step.py" >"$evidence/static.stdout"
for manifest in \
  tests/g3c4/expected/predecessor_sources.sha256 \
  tests/g3n/expected/predecessor_sources.sha256 \
  tests/g3s/expected/predecessor_sources.sha256 \
  tests/n3k/expected/predecessor_sources.sha256 \
  tests/m3cg/predecessor_sources.sha256 \
  tests/e3_metrics/expected/predecessor_sources.sha256 \
  native/l3s_predecessor_sources.sha256; do
  (cd "$PROJECT_ROOT" && sha256sum -c "$manifest") \
    >>"$evidence/predecessor-hashes.stdout"
done
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v tests.q0.test_python_isolation \
  >"$evidence/python-isolation.stdout" 2>"$evidence/python-isolation.stderr"
if [[ "${G3C4_SKIP_ROLE_STEP_PREDECESSOR:-0}" == 1 ]]; then
  printf 'G3-C4 manual frame commit regression: SKIPPED\n' >"$evidence/frame-commit-regression.stdout"
else
  CC="$cc" "${PROJECT_ROOT}/scripts/test-g3c4-manual-frame-commit.sh" \
    "$evidence/frame-commit-regression" >"$evidence/frame-commit-regression.stdout" 2>&1
fi
flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic
       -ffp-contract=off -fexcess-precision=standard -fno-fast-math
       -fstack-protector-all -I "${PROJECT_ROOT}/include"
       -I "${PROJECT_ROOT}/native" -I "${PROJECT_ROOT}/src"
       -I "${PROJECT_ROOT}/src/eshkol_transformer")
base_macros=(
  -DET_G3C4_CONTEXT_PRIVATE -DET_G3C4_NATIVE_PINS_PRIVATE
  -DET_G3C4_ACTIVE_CALL_PRIVATE -DET_G3C4_GENERATOR_PRIVATE
  -DET_G3C4_PROMPT_T1_BORROW_PRIVATE
  -DET_G3C4_PROVIDER_ROUTES_PRIVATE -DET_G3C4_FULL_PREFIX_FORWARD_PRIVATE
  -DET_G3C4_SAMPLER_TRANSPORT_PRIVATE -DET_G3C4_TOKEN_FRAME_PRIVATE
  -DET_G3C4_TOKEN_FORWARD_PRIVATE -DET_G3C4_PREFILL3_PRIVATE
  -DET_G3C4_PREFILL1_PRIVATE -DET_G3C4_PREFILL2_PRIVATE
  -DET_G3C4_PROMPT_PREFILL_PRIVATE -DET_G3C4_OUTPUT_RESERVATION_PRIVATE
  -DET_G3C4_LAST_LOGIT_FRAME_PRIVATE -DET_G3C4_OUTPUT_PREPARE_PRIVATE
  -DET_G3C4_OUTPUT_DECODE_IDS_PRIVATE -DET_G3C4_OUTPUT_TEXT_PRIVATE
  -DET_I64_TENSOR_STORAGE_QUERY_PRIVATE -DET_A2_KV_CACHE_STORAGE_QUERY_PRIVATE
)
"$cc" "${flags[@]}" -O2 "${base_macros[@]}" \
  -DET_G3C4_LOGITS_RESERVATION_PRIVATE -DET_G3C4_MANUAL_FRAME_BEGIN_PRIVATE \
  -DET_G3C4_MANUAL_ROLE0_PRIVATE -DET_G3C4_MANUAL_PRE_A2_PRIVATE -DET_G3C4_MANUAL_A2_PRIVATE -DET_G3C4_MANUAL_AT_PRIVATE -DET_G3C4_MANUAL_AO_PRIVATE -DET_G3C4_MANUAL_R_PRIVATE -DET_G3C4_MANUAL_TAIL_PRIVATE -DET_G3C4_MANUAL_FRAME_COMMIT_PRIVATE -c \
  "${PROJECT_ROOT}/src/eshkol_transformer/g3c4_model_owner.c" -o "$evidence/frame-commit-owner.o"
"$cc" "${flags[@]}" -O2 "${base_macros[@]}" \
  -DET_G3C4_LOGITS_RESERVATION_PRIVATE -DET_G3C4_MANUAL_FRAME_BEGIN_PRIVATE \
  -DET_G3C4_MANUAL_ROLE0_PRIVATE -DET_G3C4_MANUAL_PRE_A2_PRIVATE -DET_G3C4_MANUAL_A2_PRIVATE -DET_G3C4_MANUAL_AT_PRIVATE -DET_G3C4_MANUAL_AO_PRIVATE -DET_G3C4_MANUAL_R_PRIVATE -DET_G3C4_MANUAL_TAIL_PRIVATE -DET_G3C4_MANUAL_FRAME_COMMIT_PRIVATE -DET_G3C4_MANUAL_ROLE_STEP_PRIVATE -c \
  "${PROJECT_ROOT}/src/eshkol_transformer/g3c4_model_owner.c" -o "$evidence/role-step-owner.o"
for object in frame-commit-owner role-step-owner; do
  nm -g --defined-only --format=posix "$evidence/$object.o" |
    awk 'NF >= 2 { print $1 }' | LC_ALL=C sort -u >"$evidence/$object-defined.txt"
done
comm -13 "$evidence/frame-commit-owner-defined.txt" "$evidence/role-step-owner-defined.txt" \
  >"$evidence/added-defined.txt"
grep -Fx 'et_g3c4_private_role_step_v1' "$evidence/added-defined.txt" >/dev/null
test "$(wc -l <"$evidence/added-defined.txt")" -eq 1
if "$cc" "${flags[@]}" -O2 -DET_G3C4_MANUAL_ROLE_STEP_PRIVATE -c \
   "${PROJECT_ROOT}/src/eshkol_transformer/g3c4_model_owner.c" \
   -o "$evidence/invalid-role-step-only.o" \
   >"$evidence/invalid-role-step-only.stdout" 2>"$evidence/invalid-role-step-only.stderr"; then
  printf 'invalid role-step-only macro tuple compiled\n' >&2; exit 1
fi
grep -F 'ET_G3C4_MANUAL_ROLE_STEP_PRIVATE requires manual frame publication' \
  "$evidence/invalid-role-step-only.stderr" >/dev/null
native_sources=(
  "${PROJECT_ROOT}/tests/g3c4/test_manual_role_step.c"
  "${PROJECT_ROOT}/src/eshkol_transformer/m3_i64_integration.c"
  "${PROJECT_ROOT}/tests/m3t/kernel_fail_allocator.c"
  "${PROJECT_ROOT}/native/t1_i64_shell.c"
  "${PROJECT_ROOT}/native/n3k_primitives_provider.c"
  "${PROJECT_ROOT}/native/g3c4_primitives_provider.c"
  "${PROJECT_ROOT}/native/g3s_sampling_provider.c"
  "${PROJECT_ROOT}/native/g3n_primitives_provider.c"
  "${PROJECT_ROOT}/native/n2_primitives_provider.c"
  "${PROJECT_ROOT}/native/a2_kv_cache.c"
  "${PROJECT_ROOT}/native/a2_attention_provider.c"
)
compile_mode() {
  local mode=$1
  local -a mode_flags=(-O2) runtime=(env)
  if [[ "$mode" == sanitize ]]; then
    mode_flags=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
    runtime=(env ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1)
  fi
  "$cc" "${flags[@]}" "${mode_flags[@]}" \
    -DET_I64_TENSOR_TESTING -DET_I64_TENSOR_STORAGE_QUERY_PRIVATE \
    -DET_A2_KV_CACHE_TESTING -DET_A2_KV_CACHE_STORAGE_QUERY_PRIVATE \
    "${native_sources[@]}" -Wl,--wrap=et_kernel_runtime_dispatch -lm \
    -o "$evidence/$mode-manual-role-step"
  "${runtime[@]}" timeout --foreground --signal=TERM --kill-after=5s 120s \
    "$evidence/$mode-manual-role-step" \
    >"$evidence/$mode.stdout" 2>"$evidence/$mode.stderr"
  test ! -s "$evidence/$mode.stderr"
  grep -E '^G3-C4 private manual role step PASS: checks=[1-9][0-9]* routes=3 retries=21$' \
    "$evidence/$mode.stdout" >/dev/null
}
compile_mode normal
"$evidence/normal-manual-role-step" >"$evidence/runtime-repeat.stdout" \
  2>"$evidence/runtime-repeat.stderr"
test ! -s "$evidence/runtime-repeat.stderr"
compile_mode sanitize
cmp "$evidence/normal.stdout" "$evidence/runtime-repeat.stdout"
cmp "$evidence/normal.stdout" "$evidence/sanitize.stdout"
git -C "$PROJECT_ROOT" diff --check
sha256sum "${PROJECT_ROOT}/native/g3c4_manual_role_step_source_closure.txt" \
  >"$evidence/closure.sha256"
sha256sum "$evidence/static.stdout" "$evidence/normal.stdout" \
  "$evidence/runtime-repeat.stdout" "$evidence/sanitize.stdout" \
  >"$evidence/result-seal.sha256"
cat "$evidence/static.stdout"
cat "$evidence/normal.stdout"
printf 'G3-C4 private manual role step evidence: %s\n' "$evidence"
