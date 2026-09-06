#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
require_command ar
require_command cmp
require_command grep
require_command nm
require_command objdump
require_command python3
require_command rg
require_command sed
require_command sha256sum
require_command timeout

n2_provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
n2_cc="$(tsv_value "${n2_provenance}" cc_path)"
n2_cxx="$(tsv_value "${n2_provenance}" cxx_path)"
n2_tmp="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-transformer-n2.XXXXXX")"
trap 'rm -rf -- "${n2_tmp}"' EXIT
n2_artifact_dir="$(project_build_dir)/n2"
n2_library="${n2_artifact_dir}/libeshkol_transformer_n2.a"
k1_library="$(project_build_dir)/k1/libeshkol_transformer_k1.a"
i1_library="$(project_build_dir)/i1/libeshkol_transformer_i64.a"
i2_library="$(project_build_dir)/i2/libeshkol_transformer_f32.a"
n2_runner="$(eshkol_build_dir)/eshkol-run"
n2_cflags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic
  -ffp-contract=off -fexcess-precision=standard -fno-fast-math
  -fstack-protector-all -I "${PROJECT_ROOT}/include"
)

[[ -r "${n2_library}" ]] || die "canonical N2 archive is missing"
[[ -r "${k1_library}" ]] || die "canonical K1 archive is missing"
[[ -r "${i1_library}" ]] || die "canonical I1 archive is missing"
[[ -r "${i2_library}" ]] || die "canonical I2 archive is missing"
[[ "$(ar t "${n2_library}")" == 'n2_primitives_provider.o' ]] || \
  die "canonical N2 archive must contain exactly n2_primitives_provider.o"

nm -g --defined-only "${n2_library}" | \
  awk '$2 ~ /^[TDBR]$/ { print $3 }' | LC_ALL=C sort -u \
  >"${n2_tmp}/defined.txt"
cmp "${PROJECT_ROOT}/tests/n2/expected/n2_defined_symbols.txt" \
  "${n2_tmp}/defined.txt"
nm -u "${n2_artifact_dir}/n2_primitives_provider.o" | \
  awk 'NF != 0 && $NF !~ /:$/ { print $NF }' | LC_ALL=C sort -u \
  >"${n2_tmp}/undefined.txt"
cmp "${PROJECT_ROOT}/tests/n2/expected/n2_allowed_undefined_symbols.txt" \
  "${n2_tmp}/undefined.txt"
rg -l 'ET_N2_PRIMITIVES_ABI|et_n2_kernel_provider_v1' \
  "${PROJECT_ROOT}/include" "${PROJECT_ROOT}/native" | \
  sed "s#^${PROJECT_ROOT}/##" | LC_ALL=C sort \
  >"${n2_tmp}/source-closure.txt"
cmp "${PROJECT_ROOT}/tests/n2/expected/n2_source_closure.txt" \
  "${n2_tmp}/source-closure.txt"

if nm -g --defined-only "${n2_library}" | \
    grep -E 'et_n2_test_|eshkol_transformer_kernel_provider_v1' >/dev/null; then
  die "private transport or canonical K1 provider symbol leaked into N2"
fi
if nm -u "${n2_artifact_dir}/n2_primitives_provider.o" | \
    grep -E '(^|[[:space:]_])(malloc|calloc|realloc|free|dlopen|dlsym)(@|$)' \
      >/dev/null; then
  die "N2 provider has a hidden allocation or dynamic-loader dependency"
fi
if grep -Ein \
    'python|pytorch|torch|finite[-_ ]difference|scalar[-_ ]fallback|cpu[-_ ]fallback' \
    "${PROJECT_ROOT}/native/n2_primitives_provider.c" \
    "${PROJECT_ROOT}/include/eshkol_transformer/n2_primitives_abi.h"; then
  die "N2 production path contains a test oracle or fallback reference"
fi
if objdump -d "${n2_artifact_dir}/n2_primitives_provider.o" | \
    grep -E '\b(v?fmadd|fma)' >/dev/null; then
  die "N2 provider contains a fused multiply-add instruction"
fi
"${n2_cc}" "${n2_cflags[@]}" -O2 -S -emit-llvm \
  "${PROJECT_ROOT}/native/n2_primitives_provider.c" \
  -o "${n2_tmp}/n2-provider.ll"
if grep -E 'llvm\.(fma|fmuladd)|(^|[,( ])double([, )]|$)' \
    "${n2_tmp}/n2-provider.ll" >/dev/null; then
  die "N2 provider IR contains FMA or binary64 arithmetic"
fi

# Linking N2 alone must leave K1's canonical provider-free report unchanged.
"${n2_cc}" "${n2_cflags[@]}" -O2 \
  "${PROJECT_ROOT}/tests/k1/report_baseline.c" \
  "${n2_library}" "${k1_library}" -lm \
  -o "${n2_tmp}/report-baseline-n2-linked"
LC_ALL=C "${n2_tmp}/report-baseline-n2-linked" \
  >"${n2_tmp}/baseline_report_v1.json"
(cd "${n2_tmp}" && sha256sum -c \
  "${PROJECT_ROOT}/tests/k1/expected/baseline_report_v1.sha256")

/usr/bin/bash "${PROJECT_ROOT}/scripts/build-n2.sh" \
  "${n2_tmp}/fresh-a" normal
/usr/bin/bash "${PROJECT_ROOT}/scripts/build-n2.sh" \
  "${n2_tmp}/fresh-b" normal
cmp "${n2_tmp}/fresh-a/n2_primitives_provider.o" \
  "${n2_tmp}/fresh-b/n2_primitives_provider.o"
cmp "${n2_tmp}/fresh-a/libeshkol_transformer_n2.a" \
  "${n2_tmp}/fresh-b/libeshkol_transformer_n2.a"
cmp "${n2_artifact_dir}/n2_primitives_provider.o" \
  "${n2_tmp}/fresh-a/n2_primitives_provider.o"
cmp "${n2_library}" \
  "${n2_tmp}/fresh-a/libeshkol_transformer_n2.a"

"${n2_cc}" "${n2_cflags[@]}" -O2 \
  "${PROJECT_ROOT}/tests/n2/test_primitives_provider.c" \
  "${n2_library}" "${k1_library}" -lm \
  -o "${n2_tmp}/test-provider"
for run in 1 2; do
  timeout --foreground --signal=TERM --kill-after=5s 60s \
    "${n2_tmp}/test-provider" >"${n2_tmp}/provider-${run}.stdout"
done
cmp "${n2_tmp}/provider-1.stdout" "${n2_tmp}/provider-2.stdout"
grep -E '^N2 provider PASS \([0-9]+ checks\)$' \
  "${n2_tmp}/provider-1.stdout" >/dev/null

# Bind all 32 output bits of the official Random123 vectors to the production
# implementation through a test-only hook that is absent from the normal object.
"${n2_cc}" "${n2_cflags[@]}" -O2 -DET_N2_TESTING -c \
  "${PROJECT_ROOT}/native/n2_primitives_provider.c" \
  -o "${n2_tmp}/n2-provider-testing.o"
n2_testing_exports="$(nm -g --defined-only \
  "${n2_tmp}/n2-provider-testing.o" | \
  awk '$2 == "T" { print $3 }' | LC_ALL=C sort)"
[[ "${n2_testing_exports}" == \
  $'et_n2_kernel_provider_v1\net_n2_test_philox4x32_10_v1' ]] || \
  die "N2 testing object has an unexpected export boundary"
"${n2_cc}" "${n2_cflags[@]}" -O2 -DET_N2_TESTING \
  "${PROJECT_ROOT}/tests/n2/test_philox_vectors.c" \
  "${n2_tmp}/n2-provider-testing.o" "${k1_library}" -lm \
  -o "${n2_tmp}/test-native-philox"
timeout --foreground --signal=TERM --kill-after=5s 60s \
  "${n2_tmp}/test-native-philox" >"${n2_tmp}/native-philox.stdout"
grep -E '^N2 native Philox PASS \([0-9]+ checks\)$' \
  "${n2_tmp}/native-philox.stdout" >/dev/null

"${n2_cc}" "${n2_cflags[@]}" -O2 \
  "${PROJECT_ROOT}/tests/n2/test_bit_goldens.c" \
  "${n2_library}" "${k1_library}" -lm \
  -o "${n2_tmp}/test-bit-goldens"
for run in 1 2; do
  timeout --foreground --signal=TERM --kill-after=5s 60s \
    "${n2_tmp}/test-bit-goldens" >"${n2_tmp}/bit-goldens-${run}.stdout"
done
cmp "${n2_tmp}/bit-goldens-1.stdout" \
  "${n2_tmp}/bit-goldens-2.stdout"
grep -F 'N2 bit goldens PASS' "${n2_tmp}/bit-goldens-1.stdout" >/dev/null

"${n2_cc}" "${n2_cflags[@]}" -O2 \
  "${PROJECT_ROOT}/tests/n2/test_negative_atomicity.c" \
  "${n2_library}" "${k1_library}" -lm \
  -o "${n2_tmp}/test-negative-atomicity"
for run in 1 2; do
  timeout --foreground --signal=TERM --kill-after=5s 60s \
    "${n2_tmp}/test-negative-atomicity" \
    >"${n2_tmp}/negative-atomicity-${run}.stdout"
done
cmp "${n2_tmp}/negative-atomicity-1.stdout" \
  "${n2_tmp}/negative-atomicity-2.stdout"
grep -E '^N2 negative atomicity PASS \([0-9]+ checks\)$' \
  "${n2_tmp}/negative-atomicity-1.stdout" >/dev/null

# Exercise every N2 operation through real accepted I2 f32 and I1 exact-i64
# leases.  The test also binds N2 backward outputs to I2's atomic parameter-
# gradient plans and proves fixed residual-edge combination for one identity.
"${n2_cc}" "${n2_cflags[@]}" -O2 -I "${PROJECT_ROOT}/native" \
  "${PROJECT_ROOT}/tests/n2/test_i2_integration.c" \
  "${n2_library}" "${i2_library}" "${i1_library}" "${k1_library}" -lm \
  -o "${n2_tmp}/test-i2-integration"
for run in 1 2; do
  timeout --foreground --signal=TERM --kill-after=5s 60s \
    "${n2_tmp}/test-i2-integration" \
    >"${n2_tmp}/i2-integration-${run}.stdout"
done
cmp "${n2_tmp}/i2-integration-1.stdout" \
  "${n2_tmp}/i2-integration-2.stdout"
grep -E '^N2 I1/I2 integration PASS \([0-9]+ checks\)$' \
  "${n2_tmp}/i2-integration-1.stdout" >/dev/null

for n2_mutation in \
  batch-index-collapse \
  reduction-order \
  philox-mask-inversion \
  embedding-reduction-order \
  layer-norm-variance-formula \
  layer-norm-mean-order \
  gelu-inv-sqrt-two \
  gelu-slope-order \
  philox-lane-rotation \
  philox-key-halves \
  philox-counter-offset \
  dropout-sentinel-bypass \
  relu-zero-kink \
  residual-second-edge-zero; do
  python3 "${PROJECT_ROOT}/tests/n2/apply_source_mutation.py" \
    "${n2_mutation}" \
    "${PROJECT_ROOT}/native/n2_primitives_provider.c" \
    "${n2_tmp}/n2-${n2_mutation}.c"
  "${n2_cc}" "${n2_cflags[@]}" -O2 -c \
    "${n2_tmp}/n2-${n2_mutation}.c" \
    -o "${n2_tmp}/n2-${n2_mutation}.o"
  "${n2_cc}" "${n2_cflags[@]}" -O2 \
    "${PROJECT_ROOT}/tests/n2/test_bit_goldens.c" \
    "${n2_tmp}/n2-${n2_mutation}.o" "${k1_library}" -lm \
    -o "${n2_tmp}/bit-${n2_mutation}"
  "${n2_cc}" "${n2_cflags[@]}" -O2 \
    "${PROJECT_ROOT}/tests/n2/test_primitives_provider.c" \
    "${n2_tmp}/n2-${n2_mutation}.o" "${k1_library}" -lm \
    -o "${n2_tmp}/oracle-${n2_mutation}"
  n2_bit_rejected=0
  if timeout --foreground --signal=TERM --kill-after=5s 60s \
      "${n2_tmp}/bit-${n2_mutation}" \
      >"${n2_tmp}/mutation-bit-${n2_mutation}.stdout" \
      2>"${n2_tmp}/mutation-bit-${n2_mutation}.stderr"; then
    :
  else
    n2_bit_rejected=1
    grep -E 'BIT-GOLDEN|dispatch failure' \
      "${n2_tmp}/mutation-bit-${n2_mutation}.stderr" >/dev/null
  fi
  n2_oracle_rejected=0
  if timeout --foreground --signal=TERM --kill-after=5s 60s \
      "${n2_tmp}/oracle-${n2_mutation}" \
      >"${n2_tmp}/mutation-oracle-${n2_mutation}.stdout" \
      2>"${n2_tmp}/mutation-oracle-${n2_mutation}.stderr"; then
    :
  else
    n2_oracle_rejected=1
    grep -E 'FAIL|mismatch|unexpected' \
      "${n2_tmp}/mutation-oracle-${n2_mutation}.stderr" >/dev/null
  fi
  if [[ "${n2_bit_rejected}" == 0 && "${n2_oracle_rejected}" == 0 ]]; then
    die "N2 source mutation survived bit and frozen-oracle evidence: ${n2_mutation}"
  fi
  printf 'N2 mutation rejected: %s (bit=%s frozen=%s)\n' \
    "${n2_mutation}" "${n2_bit_rejected}" "${n2_oracle_rejected}"
done

"${n2_cxx}" -std=c++17 -Wall -Wextra -Werror -Wpedantic \
  -I "${PROJECT_ROOT}/include" "${PROJECT_ROOT}/tests/n2/header_cpp.cpp" \
  "${n2_library}" "${k1_library}" -lm -o "${n2_tmp}/header-cpp"
"${n2_tmp}/header-cpp"

# This fixed-buffer bridge is private test evidence. It exercises real K1/N2
# dispatch from compiled Eshkol without adding a tensor carrier or resolver to
# the production archive.
"${n2_cc}" "${n2_cflags[@]}" -O2 -c \
  "${PROJECT_ROOT}/tests/n2/aot_private_bridge.c" \
  -o "${n2_tmp}/aot_private_bridge.o"
n2_test_exports="$(nm -g --defined-only "${n2_tmp}/aot_private_bridge.o" | \
  awk '$2 ~ /^[TDBR]$/ { print $3 }' | LC_ALL=C sort)"
[[ "${n2_test_exports}" == 'et_n2_test_provider_transport_v1' ]] || \
  die "N2 private AOT bridge has an unexpected export boundary"
nm -u "${n2_tmp}/aot_private_bridge.o" | \
  grep -F 'et_n2_kernel_provider_v1' >/dev/null
ar rcsD "${n2_tmp}/libeshkol_transformer_n2_private_aot.a" \
  "${n2_tmp}/aot_private_bridge.o" \
  "${n2_artifact_dir}/n2_primitives_provider.o" \
  "$(project_build_dir)/k1/kernel_abi.o"
timeout --foreground --signal=TERM --kill-after=5s 60s \
  "${n2_runner}" --strict-types --emit-object \
  --emit-depfile "${n2_tmp}/aot_private.d" --no-stdlib \
  "${PROJECT_ROOT}/tests/n2/aot_private.esk" \
  -o "${n2_tmp}/aot_private.o" \
  >"${n2_tmp}/aot-private-object.stdout" \
  2>"${n2_tmp}/aot-private-object.stderr"
grep -F 'tests/n2/aot_private.esk' "${n2_tmp}/aot_private.d" >/dev/null
test ! -s "${n2_tmp}/aot-private-object.stderr"
for n2_aot_run in 1 2; do
  timeout --foreground --signal=TERM --kill-after=5s 60s \
    "${n2_runner}" --strict-types --no-stdlib \
    -L "${n2_tmp}" --lib eshkol_transformer_n2_private_aot \
    "${PROJECT_ROOT}/tests/n2/aot_private.esk" \
    -o "${n2_tmp}/aot-private-${n2_aot_run}" \
    >"${n2_tmp}/aot-private-compile-${n2_aot_run}.stdout" \
    2>"${n2_tmp}/aot-private-compile-${n2_aot_run}.stderr"
  test ! -s "${n2_tmp}/aot-private-compile-${n2_aot_run}.stderr"
  timeout --foreground --signal=TERM --kill-after=5s 30s \
    "${n2_tmp}/aot-private-${n2_aot_run}" \
    >"${n2_tmp}/aot-private-${n2_aot_run}.stdout" \
    2>"${n2_tmp}/aot-private-${n2_aot_run}.stderr"
  cmp "${PROJECT_ROOT}/tests/n2/expected/aot_private.stdout" \
    "${n2_tmp}/aot-private-${n2_aot_run}.stdout"
  test ! -s "${n2_tmp}/aot-private-${n2_aot_run}.stderr"
done
cmp "${n2_tmp}/aot-private-1" "${n2_tmp}/aot-private-2"
cmp "${n2_tmp}/aot-private-1.stdout" "${n2_tmp}/aot-private-2.stdout"

/usr/bin/bash "${PROJECT_ROOT}/scripts/build-n2.sh" \
  "${n2_tmp}/sanitized" sanitize
"${n2_cc}" "${n2_cflags[@]}" -O1 -g \
  -DET_N2_AOT_BRIDGE_TESTING \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  "${PROJECT_ROOT}/tests/n2/aot_private_bridge.c" \
  "${n2_tmp}/sanitized/libeshkol_transformer_n2.a" \
  "${PROJECT_ROOT}/native/kernel_abi.c" -lm \
  -o "${n2_tmp}/aot-private-bridge-sanitized"
ASAN_OPTIONS=detect_leaks="${N2_ASAN_DETECT_LEAKS:-0}":halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1 \
  timeout --foreground --signal=TERM --kill-after=5s 60s \
  "${n2_tmp}/aot-private-bridge-sanitized"
"${n2_cc}" "${n2_cflags[@]}" -O1 -g \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  "${PROJECT_ROOT}/tests/n2/test_primitives_provider.c" \
  "${n2_tmp}/sanitized/libeshkol_transformer_n2.a" \
  "${PROJECT_ROOT}/native/kernel_abi.c" -lm \
  -o "${n2_tmp}/test-provider-sanitized"
ASAN_OPTIONS=detect_leaks="${N2_ASAN_DETECT_LEAKS:-0}":halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1 \
  timeout --foreground --signal=TERM --kill-after=5s 60s \
  "${n2_tmp}/test-provider-sanitized" \
  >"${n2_tmp}/provider-sanitized.stdout"
cmp "${n2_tmp}/provider-1.stdout" "${n2_tmp}/provider-sanitized.stdout"
"${n2_cc}" "${n2_cflags[@]}" -O1 -g -DET_N2_TESTING \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  "${PROJECT_ROOT}/tests/n2/test_philox_vectors.c" \
  "${PROJECT_ROOT}/native/n2_primitives_provider.c" \
  "${PROJECT_ROOT}/native/kernel_abi.c" -lm \
  -o "${n2_tmp}/test-native-philox-sanitized"
ASAN_OPTIONS=detect_leaks="${N2_ASAN_DETECT_LEAKS:-0}":halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1 \
  timeout --foreground --signal=TERM --kill-after=5s 60s \
  "${n2_tmp}/test-native-philox-sanitized" \
  >"${n2_tmp}/native-philox-sanitized.stdout"
cmp "${n2_tmp}/native-philox.stdout" \
  "${n2_tmp}/native-philox-sanitized.stdout"
"${n2_cc}" "${n2_cflags[@]}" -O1 -g \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  "${PROJECT_ROOT}/tests/n2/test_negative_atomicity.c" \
  "${n2_tmp}/sanitized/libeshkol_transformer_n2.a" \
  "${PROJECT_ROOT}/native/kernel_abi.c" -lm \
  -o "${n2_tmp}/test-negative-atomicity-sanitized"
ASAN_OPTIONS=detect_leaks="${N2_ASAN_DETECT_LEAKS:-0}":halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1 \
  timeout --foreground --signal=TERM --kill-after=5s 60s \
  "${n2_tmp}/test-negative-atomicity-sanitized" \
  >"${n2_tmp}/negative-atomicity-sanitized.stdout"
cmp "${n2_tmp}/negative-atomicity-1.stdout" \
  "${n2_tmp}/negative-atomicity-sanitized.stdout"
"${n2_cc}" "${n2_cflags[@]}" -O1 -g \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  "${PROJECT_ROOT}/tests/n2/test_bit_goldens.c" \
  "${n2_tmp}/sanitized/libeshkol_transformer_n2.a" \
  "${PROJECT_ROOT}/native/kernel_abi.c" -lm \
  -o "${n2_tmp}/test-bit-goldens-sanitized"
ASAN_OPTIONS=detect_leaks="${N2_ASAN_DETECT_LEAKS:-0}":halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1 \
  timeout --foreground --signal=TERM --kill-after=5s 60s \
  "${n2_tmp}/test-bit-goldens-sanitized" \
  >"${n2_tmp}/bit-goldens-sanitized.stdout"
cmp "${n2_tmp}/bit-goldens-1.stdout" \
  "${n2_tmp}/bit-goldens-sanitized.stdout"

/usr/bin/bash "${PROJECT_ROOT}/scripts/build-i1.sh" \
  "${n2_tmp}/sanitized-i1" sanitize-test
/usr/bin/bash "${PROJECT_ROOT}/scripts/build-i2.sh" \
  "${n2_tmp}/sanitized-i2" sanitize-test
"${n2_cc}" "${n2_cflags[@]}" -O1 -g \
  -DET_F32_TENSOR_TESTING -DET_I64_TENSOR_TESTING \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  -I "${PROJECT_ROOT}/native" \
  "${PROJECT_ROOT}/tests/n2/test_i2_integration.c" \
  "${n2_tmp}/sanitized/libeshkol_transformer_n2.a" \
  "${n2_tmp}/sanitized-i2/libeshkol_transformer_f32.a" \
  "${n2_tmp}/sanitized-i1/libeshkol_transformer_i64.a" \
  "${PROJECT_ROOT}/native/kernel_abi.c" -lm \
  -o "${n2_tmp}/test-i2-integration-sanitized"
ASAN_OPTIONS=detect_leaks="${N2_ASAN_DETECT_LEAKS:-0}":halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1 \
  timeout --foreground --signal=TERM --kill-after=5s 60s \
  "${n2_tmp}/test-i2-integration-sanitized" \
  >"${n2_tmp}/i2-integration-sanitized.stdout"
cmp "${n2_tmp}/i2-integration-1.stdout" \
  "${n2_tmp}/i2-integration-sanitized.stdout"

python3 -m unittest -v tests.n2.test_reference tests.n2.test_numerical_gradients
if [[ -n "${N2_ORACLE_PYTHON:-}" ]]; then
  [[ -x "${N2_ORACLE_PYTHON}" ]] || die "N2_ORACLE_PYTHON is not executable"
  "${N2_ORACLE_PYTHON}" -c '
import sys
import torch

assert sys.version_info[:3] == (3, 14, 6), sys.version
assert torch.__version__ == "2.13.0+cpu", torch.__version__
assert torch.version.cuda is None, torch.version.cuda
assert not torch.cuda.is_available()
assert torch.empty(1).device.type == "cpu"
'
  "${N2_ORACLE_PYTHON}" -m unittest -v \
    tests.n2.test_reference tests.n2.test_numerical_gradients
fi

printf 'N2 PASS: carrier-neutral ABI, parity, gradients, RNG, private Eshkol AOT, and sanitizers\n'
