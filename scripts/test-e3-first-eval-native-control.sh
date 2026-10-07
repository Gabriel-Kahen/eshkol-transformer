#!/usr/bin/env bash
# Future supported native control. Source review does not execute this script.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_supported_host
verify_toolchain
for command in nm cmp grep; do require_command "${command}"; done
[[ $# == 1 && "$1" == /* && ! -e "$1" ]] ||
  die "first-eval native control needs one fresh absolute output directory"
out=$1
mkdir -m 0700 -- "${out}"
provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
cc="$(tsv_value "${provenance}" cc_path)"
[[ "${cc}" == /* && -x "${cc}" ]] || die "selected C compiler is unavailable"
flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic
  -ffp-contract=off -fexcess-precision=standard -frounding-math
  -fno-fast-math -fstack-protector-all -ffunction-sections -fdata-sections
  -fPIC -fvisibility=hidden
  -DET_F32_TENSOR_TESTING -DET_I64_TENSOR_TESTING
  -DET_E3_RETENTION_TESTING -DET_D2_NATIVE_TESTING
  -I "${PROJECT_ROOT}/include" -I "${PROJECT_ROOT}/native"
  -I "${PROJECT_ROOT}/src/eshkol_transformer")
wrapper="${PROJECT_ROOT}/tests/e3_private/retention_first_eval_frame.c"
sources=("${PROJECT_ROOT}/tests/e3_private/retention_first_eval_native_control.c"
  "${PROJECT_ROOT}/src/eshkol_transformer/m3_call_f32_integration.c"
  "${PROJECT_ROOT}/src/eshkol_transformer/m3_i64_integration.c"
  "${PROJECT_ROOT}/native/kernel_abi.c"
  "${PROJECT_ROOT}/native/t1_i64_shell.c"
  "${PROJECT_ROOT}/native/n3k_primitives_provider.c"
  "${PROJECT_ROOT}/native/n2_primitives_provider.c"
  "${PROJECT_ROOT}/native/a2_attention_provider.c"
  "${PROJECT_ROOT}/native/indexed_cross_entropy.c"
  "${PROJECT_ROOT}/native/l3s_masked_objective_provider.c"
  "${PROJECT_ROOT}/native/e3_evaluation_metrics_provider.c")

# This is the same pinned wrapper with its mutation define absent. No helper
# definition may survive even in a non-exported symbol table.
"${cc}" "${flags[@]}" -O2 -c "${wrapper}" -o "${out}/feature-off.o" \
  >"${out}/feature-off.compile.stdout" 2>"${out}/feature-off.compile.stderr"
if nm -a --defined-only "${out}/feature-off.o" | \
    grep -E '[[:space:]]et_e3_test_first_eval_corrupt_v1$' >/dev/null; then
  die "first-eval mutation helper exists with its feature disabled"
fi

for mode in normal sanitize; do
  extra=(-O2)
  if [[ "${mode}" == sanitize ]]; then
    extra=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
  fi
  "${cc}" "${flags[@]}" -DET_E3_FIRST_EVAL_MUTATION_TESTING \
    "${extra[@]}" "${sources[@]}" -Wl,--gc-sections -lm \
    -o "${out}/native-${mode}" \
    >"${out}/${mode}.compile.stdout" 2>"${out}/${mode}.compile.stderr"
  nm -a --defined-only "${out}/native-${mode}" | \
    grep -E '[[:space:]]et_e3_test_first_eval_corrupt_v1$' >/dev/null ||
    die "first-eval native helper is absent from enabled control"
  for selector in 0 1 2; do
    ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
      UBSAN_OPTIONS=halt_on_error=1 \
      "${out}/native-${mode}" "${selector}" \
      >"${out}/${mode}-${selector}.stdout" \
      2>"${out}/${mode}-${selector}.stderr"
    grep -E "^FIRST-EVAL-NATIVE-CONTROL[[:space:]]${selector}[[:space:]][1-9][0-9]*$" \
      "${out}/${mode}-${selector}.stdout" >/dev/null ||
      die "first-eval native control did not finish exact selector"
    [[ ! -s "${out}/${mode}-${selector}.stderr" ]] ||
      die "first-eval native control emitted unexpected diagnostics"
  done
done
for selector in 0 1 2; do
  "${out}/native-normal" "${selector}" \
    >"${out}/repeat-${selector}.stdout" \
    2>"${out}/repeat-${selector}.stderr"
  cmp "${out}/normal-${selector}.stdout" "${out}/repeat-${selector}.stdout"
  [[ ! -s "${out}/repeat-${selector}.stderr" ]] ||
    die "first-eval native repeat emitted unexpected diagnostics"
done
