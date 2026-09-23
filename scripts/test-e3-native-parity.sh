#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

mode="${1:-all}"
output="${2:-}"
case "${mode}" in
  all|prepare|native|verify) ;;
  *) die 'usage: test-e3-native-parity.sh [all|prepare|native|verify] [OUTPUT]' ;;
esac

if [[ "${mode}" == all ]]; then
  verify_supported_host
  parity_temporary="$(mktemp -d)"
  output="${parity_temporary}/run"
  trap 'rm -rf -- "${parity_temporary}"' EXIT
elif [[ -z "${output}" || "${output}" != /* ]]; then
  die 'split parity modes require an absolute OUTPUT directory'
fi

parity_cases=(packed-single packed-sharded packed-suffix unpacked-three unpacked-pairs)
q0_python="${Q0_PYTHON:-}"

require_q0() {
  [[ "${q0_python}" == /* && -x "${q0_python}" ]] ||
    die 'E3 native parity requires absolute Q0_PYTHON for pinned Python 3.14.6 / PyTorch 2.13.0+cpu'
  local versions
  versions="$(ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE "${q0_python}" - <<'PY'
import platform
import torch
print(platform.python_version())
print(torch.__version__)
PY
)"
  [[ "${versions}" == $'3.14.6\n2.13.0+cpu' ]] ||
    die "unexpected pinned oracle versions: ${versions//$'\n'/ }"
}

prepare() {
  require_q0
  [[ ! -e "${output}" ]] || die 'parity preparation output must be fresh'
  mkdir -p "${output}"
  for generation in a b; do
    "${q0_python}" -m tests.e3_reference.corpus \
      --output "${output}/corpus-${generation}"
    ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE "${q0_python}" \
      -m tests.e3_reference.reference \
      --corpus-reference "${output}/corpus-${generation}/corpus-reference.json" \
      --output "${output}/reference-${generation}.json"
  done
  cmp "${output}/corpus-a/corpus-reference.json" \
      "${output}/corpus-b/corpus-reference.json"
  cmp "${output}/reference-a.json" "${output}/reference-b.json"
  "${q0_python}" -m tests.e3_native_parity.generate_fixture_header \
    --reference "${output}/reference-a.json" \
    --output "${output}/e3_native_parity_fixture.h"
}

native() {
  verify_supported_host
  local parity_cc="${CC:-clang}"
  local parity_cxx="${CXX:-clang++}"
  require_command "${parity_cc}"
  require_command "${parity_cxx}"
  local expected="$(lock_value clang_version)"
  local cc_version="$("${parity_cc}" --version | sed -n '1s/.*version \([0-9][0-9.]*\).*/\1/p')"
  local cxx_version="$("${parity_cxx}" --version | sed -n '1s/.*version \([0-9][0-9.]*\).*/\1/p')"
  check_supported_version Clang "${cc_version}" "${expected}"
  check_supported_version Clang++ "${cxx_version}" "${expected}"
  [[ -f "${output}/e3_native_parity_fixture.h" ]] ||
    die 'prepared native parity fixture header is missing'

  local flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -ffp-contract=off
    -fexcess-precision=standard -frounding-math -fno-fast-math
    -fstack-protector-all -DET_F32_TENSOR_TESTING -DET_I64_TENSOR_TESTING
    -I "${output}" -I "${PROJECT_ROOT}/include" -I "${PROJECT_ROOT}/native"
    -I "${PROJECT_ROOT}/src/eshkol_transformer")
  local sources=("${PROJECT_ROOT}/tests/e3_native_parity/test_frame_parity.c"
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

  mkdir -p "${output}/native"
  for build_mode in normal sanitize; do
    local extra=(-O2)
    if [[ "${build_mode}" == sanitize ]]; then
      extra=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
    fi
    "${parity_cc}" "${flags[@]}" "${extra[@]}" "${sources[@]}" -lm \
      -o "${output}/native/frame-${build_mode}"
    for case_name in "${parity_cases[@]}"; do
      local prefix="${output}/native/${build_mode}-${case_name}"
      ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
        "${output}/native/frame-${build_mode}" "${case_name}" \
        "${prefix}.roles" >"${prefix}.json" 2>"${prefix}.log"
    done
  done
  for case_name in "${parity_cases[@]}"; do
    local prefix="${output}/native/repeat-${case_name}"
    "${output}/native/frame-normal" "${case_name}" "${prefix}.roles" \
      >"${prefix}.json" 2>"${prefix}.log"
    cmp "${output}/native/normal-${case_name}.json" "${prefix}.json"
    cmp "${output}/native/normal-${case_name}.roles" "${prefix}.roles"
  done
}

verify() {
  require_q0
  mkdir -p "${output}/transcripts"
  for build_mode in normal sanitize repeat; do
    for case_name in "${parity_cases[@]}"; do
      ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE "${q0_python}" \
        -m tests.e3_native_parity.verify_observation \
        --reference "${output}/reference-a.json" --case "${case_name}" \
        --observed "${output}/native/${build_mode}-${case_name}.json" \
        --roles "${output}/native/${build_mode}-${case_name}.roles" \
        --transcript "${output}/transcripts/${build_mode}-${case_name}.json"
    done
  done
  for case_name in "${parity_cases[@]}"; do
    cmp "${output}/transcripts/normal-${case_name}.json" \
        "${output}/transcripts/repeat-${case_name}.json"
  done
  cmp "${output}/transcripts/normal-packed-single.json" \
      "${output}/transcripts/normal-packed-sharded.json"
  cmp "${output}/native/normal-packed-single.roles" \
      "${output}/native/normal-packed-sharded.roles"

  "${q0_python}" - "${output}/native/normal-packed-single.roles" \
      "${output}/native/mutated-packed-single.roles" <<'PY'
from pathlib import Path
import sys

source, destination = map(Path, sys.argv[1:])
lines = source.read_text(encoding="ascii").splitlines()
fields = lines[0].split("\t")
words = fields[3].split(",")
words[0] = "7f7fffff"
fields[3] = ",".join(words)
lines[0] = "\t".join(fields)
destination.write_text("\n".join(lines) + "\n", encoding="ascii", newline="\n")
PY
  if ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE "${q0_python}" \
      -m tests.e3_native_parity.verify_observation \
      --reference "${output}/reference-a.json" --case packed-single \
      --observed "${output}/native/normal-packed-single.json" \
      --roles "${output}/native/mutated-packed-single.roles" \
      --transcript "${output}/transcripts/mutated-packed-single.json" \
      >"${output}/mutated-verifier.stdout" 2>"${output}/mutated-verifier.stderr"; then
    die 'role verifier accepted a deliberately corrupted native observation'
  fi
  [[ ! -e "${output}/transcripts/mutated-packed-single.json" ]] ||
    die 'failed role verification published a transcript'
}

if [[ "${mode}" == prepare || "${mode}" == all ]]; then prepare; fi
if [[ "${mode}" == native || "${mode}" == all ]]; then native; fi
if [[ "${mode}" == verify || "${mode}" == all ]]; then verify; fi

case "${mode}" in
  all)
    printf 'E3 native parity: five cases, deterministic repeat, normal and sanitizer passed\n'
    ;;
  prepare)
    printf 'E3 native parity: deterministic reference preparation passed\n'
    ;;
  native)
    printf 'E3 native parity: five cases, deterministic repeat, normal and sanitizer execution passed\n'
    ;;
  verify)
    printf 'E3 native parity: strict reference and transcript comparison passed\n'
    ;;
esac
