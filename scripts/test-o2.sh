#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
for command in ar cmp grep nm python3 rg strings timeout tr; do
  require_command "${command}"
done

provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
cc="$(tsv_value "${provenance}" cc_path)"
cxx="$(tsv_value "${provenance}" cxx_path)"
runner="$(eshkol_build_dir)/eshkol-run"
compiler_timeout="${O2_COMPILER_TIMEOUT_SECONDS:-360}"
[[ "${compiler_timeout}" =~ ^[1-9][0-9]*$ ]] || \
  die "O2_COMPILER_TIMEOUT_SECONDS must be a positive integer"
temporary_dir="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-transformer-o2.XXXXXX")"
trap 'rm -rf -- "${temporary_dir}"' EXIT

artifact_dir="$(project_build_dir)/o2"
object="${artifact_dir}/o2_wave2.o"
library="${artifact_dir}/libeshkol_transformer_wave2.a"
evidence="${object}.evidence"
[[ -r "${object}" && -r "${library}" ]] || \
  die "canonical O2 aggregate is missing"
i2_aggregate="$(project_build_dir)/i2/libeshkol_transformer_wave2.a"
t2_aggregate="$(project_build_dir)/t2/libeshkol_transformer_wave2.a"
d2_aggregate="$(project_build_dir)/d2/libeshkol_transformer_wave2.a"

cflags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic
  -ffp-contract=off -fexcess-precision=standard -frounding-math
  -fPIC -fvisibility=hidden -fno-common -fstack-protector-all
  -I "${PROJECT_ROOT}/include" -I "${PROJECT_ROOT}/native"
  -DET_O2_TESTING -DET_F32_TENSOR_TESTING
)
k1_library="$(project_build_dir)/k1/libeshkol_transformer_k1.a"
[[ -r "${k1_library}" ]] || die "canonical K1 archive is missing"

for source in test_native_optimizer test_optimizer_native; do
  "${cc}" "${cflags[@]}" -O2 \
    "${PROJECT_ROOT}/tests/o2/${source}.c" \
    "${PROJECT_ROOT}/native/f32_tensor.c" \
    "${PROJECT_ROOT}/native/o2_optimizer.c" \
    "${PROJECT_ROOT}/native/kernel_abi.c" -lm \
    -o "${temporary_dir}/${source}"
  for run in 1 2; do
    timeout --foreground --signal=TERM --kill-after=5s 300s \
      "${temporary_dir}/${source}" \
      >"${temporary_dir}/${source}-${run}.stdout"
  done
  cmp "${temporary_dir}/${source}-1.stdout" \
    "${temporary_dir}/${source}-2.stdout"

  "${cc}" "${cflags[@]}" -O1 \
    -fsanitize=address,undefined -fno-omit-frame-pointer \
    "${PROJECT_ROOT}/tests/o2/${source}.c" \
    "${PROJECT_ROOT}/native/f32_tensor.c" \
    "${PROJECT_ROOT}/native/o2_optimizer.c" \
    "${PROJECT_ROOT}/native/kernel_abi.c" -lm \
    -o "${temporary_dir}/${source}-sanitized"
  ASAN_OPTIONS=detect_leaks="${O2_ASAN_DETECT_LEAKS:-1}":halt_on_error=1 \
  UBSAN_OPTIONS=halt_on_error=1 \
    timeout --foreground --signal=TERM --kill-after=5s 300s \
      "${temporary_dir}/${source}-sanitized" \
      >"${temporary_dir}/${source}-sanitized.stdout"
done
grep -Fx 'O2 native optimizer PASS' \
  "${temporary_dir}/test_native_optimizer-1.stdout" >/dev/null
grep -Fx 'O2 native adversarial PASS: 6201 checks' \
  "${temporary_dir}/test_optimizer_native-1.stdout" >/dev/null
[[ "$(ar t "${library}")" == "o2_wave2.o" ]] || \
  die "canonical O2 aggregate must contain exactly o2_wave2.o"

cmp "${PROJECT_ROOT}/native/o2_wave2_defined_symbols.txt" \
  "${evidence}/global-defined.txt"
cmp "${PROJECT_ROOT}/native/o2_wave2_undefined_symbols.txt" \
  "${evidence}/undefined.txt"
cmp "${PROJECT_ROOT}/native/o2_wave2_public_exports.txt" \
  "${evidence}/package-exports.txt"
cmp "${PROJECT_ROOT}/native/o2_wave2_public_strings.txt" \
  "${evidence}/public-strings.txt"
sed -e 's/^[^:]*://' -e 's/\\//g' "${evidence}/private.d" | \
  tr -s '[:space:]' '\n' | grep -F "${PROJECT_ROOT}/" | \
  sed "s#^${PROJECT_ROOT}/##" >"${temporary_dir}/source-closure.txt"
cmp "${PROJECT_ROOT}/native/o2_wave2_source_closure.txt" \
  "${temporary_dir}/source-closure.txt"

for evidence_name in global-defined.txt package-exports.txt undefined.txt \
    expected-undefined.txt public-strings.txt readelf-symbols.txt nm.txt \
    strings.txt private.d native-source-closure.txt link.map \
    allowlist-provenance.tsv; do
  [[ -s "${evidence}/${evidence_name}" ]] || \
    die "O2 aggregate evidence omits ${evidence_name}"
done
[[ "$(wc -l <"${evidence}/global-defined.txt")" == 53 ]] || \
  die "O2 aggregate must expose exactly 53 globals"
[[ "$(wc -l <"${evidence}/package-exports.txt")" == 47 ]] || \
  die "O2 aggregate must expose exactly 47 package operations"
grep -Fx $'package_policy\to2-wave2-aggregate' \
  "${evidence}/allowlist-provenance.tsv" >/dev/null

for public_symbol in \
    et_e1b_public_o2_optimizer_create_v1 \
    et_e1b_public_o2_optimizer_load_state_v1 \
    et_e1b_public_o2_optimizer_state_release_v1 \
    et_e1b_public_o2_optimizer_state_v1 \
    et_e1b_public_o2_optimizer_step_v1 \
    et_e1b_public_o2_optimizer_zero_grad_v1; do
  grep -Fx "${public_symbol}" "${evidence}/global-defined.txt" >/dev/null
done

nm -s "${library}" | \
  awk '/^Archive index:$/ { in_index = 1; next }
       in_index && /^$/ { in_index = 0; next }
       in_index { print }' >"${temporary_dir}/archive-index.txt"
if grep -E 'et_(o2|i2|f32|p1_private|e1b_private)_|o2-(native|provider|state-tag)' \
    "${temporary_dir}/archive-index.txt" >/dev/null; then
  die "O2 aggregate archive index exposes localized authority"
fi
for localized_symbol in et_o2_optimizer_state_lifecycle_v1 \
    et_o2_optimizer_state_release_v1 \
    et_o2_private_optimizer_state_release_v1; do
  grep -E "[[:space:]]LOCAL[[:space:]].*[[:space:]]${localized_symbol}$" \
    "${evidence}/readelf-symbols.txt" >/dev/null || \
    die "O2 lifecycle authority is not localized: ${localized_symbol}"
done
if nm -g --defined-only --format=posix "${object}" | \
    awk '{ print $1 }' | grep -Fx et_kernel_provider_v1 >/dev/null; then
  die "O2 aggregate defines K1 canonical provider authority"
fi

run_compiler() {
  local cache_name=$1
  shift
  mkdir -p "${temporary_dir}/cache/${cache_name}"
  env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR \
    ESHKOL_JIT_CACHE=0 \
    XDG_CACHE_HOME="${temporary_dir}/cache/${cache_name}" \
    ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" \
    ESHKOL_CXX_COMPILER="${cxx}" \
    timeout --foreground --signal=TERM --kill-after=5s \
      "${compiler_timeout}s" "${runner}" "$@"
}

for run in 1 2; do
  mkdir -p "${temporary_dir}/run-${run}"
  run_compiler "public-${run}" --strict-types --no-stdlib \
    -I "${PROJECT_ROOT}/lib" -L "${artifact_dir}" \
    --lib eshkol_transformer_wave2 \
    "${PROJECT_ROOT}/tests/o2/package_smoke.esk" \
    -o "${temporary_dir}/run-${run}/package-smoke" \
    >"${temporary_dir}/compile-${run}.log" 2>&1
  timeout --foreground --signal=TERM --kill-after=5s 90s \
    "${temporary_dir}/run-${run}/package-smoke" \
    >"${temporary_dir}/run-${run}.stdout"
done
cmp "${temporary_dir}/run-1/package-smoke" \
  "${temporary_dir}/run-2/package-smoke"
cmp "${temporary_dir}/run-1.stdout" "${temporary_dir}/run-2.stdout"
grep -Fx 'O2 package smoke PASS: 2 checks' \
  "${temporary_dir}/run-1.stdout" >/dev/null

run_compiler reverse-public --strict-types --no-stdlib \
  -I "${PROJECT_ROOT}/lib" -L "${artifact_dir}" \
  --lib eshkol_transformer_wave2 \
  "${PROJECT_ROOT}/tests/o2/package_smoke_reverse.esk" \
  -o "${temporary_dir}/package-smoke-reverse" \
  >"${temporary_dir}/compile-reverse.log" 2>&1
timeout --foreground --signal=TERM --kill-after=5s 90s \
  "${temporary_dir}/package-smoke-reverse" \
  >"${temporary_dir}/package-smoke-reverse.stdout"
grep -Fx 'O2 reverse import PASS' \
  "${temporary_dir}/package-smoke-reverse.stdout" >/dev/null

runtime_dir="${temporary_dir}/runtime"
mkdir -p "${runtime_dir}"
runtime_cflags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic
  -ffp-contract=off -fexcess-precision=standard -frounding-math
  -fPIC -fvisibility=hidden -fno-common -fstack-protector-all
  -I "${PROJECT_ROOT}/include" -I "${PROJECT_ROOT}/native"
)
"${cc}" "${runtime_cflags[@]}" -DET_O2_NATIVE_HELPERS_ONLY \
  -c "${PROJECT_ROOT}/native/o2_wave2_package_bridge.c" \
  -o "${runtime_dir}/o2_bridge.o"
"${cc}" "${runtime_cflags[@]}" -DET_I2_NATIVE_HELPERS_ONLY \
  -c "${PROJECT_ROOT}/native/i2_wave2_package_bridge.c" \
  -o "${runtime_dir}/i2_bridge.o"
"${cc}" "${runtime_cflags[@]}" -DET_P1_TRUSTED_BUILD=1 \
  -c "${PROJECT_ROOT}/native/p1_identity.c" \
  -o "${runtime_dir}/p1_identity.o"
for runtime_source in data_io checkpoint_io kernel_abi i64_tensor \
    t1_i64_shell f32_tensor o2_optimizer; do
  runtime_testing=()
  if [[ "${runtime_source}" == o2_optimizer ]]; then
    runtime_testing=(-DET_O2_TESTING)
  fi
  "${cc}" "${runtime_cflags[@]}" "${runtime_testing[@]}" \
    -c "${PROJECT_ROOT}/native/${runtime_source}.c" \
    -o "${runtime_dir}/${runtime_source}.o"
done
ar rcsD "${runtime_dir}/libeshkol_transformer_o2_runtime.a" \
  "${runtime_dir}"/*.o

mkdir -p "${temporary_dir}/config-cache"
env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR \
  ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="${temporary_dir}/config-cache" \
  ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" ESHKOL_CXX_COMPILER="${cxx}" \
  timeout --foreground --signal=TERM --kill-after=5s \
    "${compiler_timeout}s" "${runner}" --strict-types --optimize 0 --no-stdlib \
    -I "${PROJECT_ROOT}/internal/p1/lib" \
    -I "${PROJECT_ROOT}/internal/c1/lib" \
    -I "${PROJECT_ROOT}/internal/t1/lib" \
    -I "${PROJECT_ROOT}/src" -I "${PROJECT_ROOT}/lib" \
    -I "${PROJECT_ROOT}/native" -L "${runtime_dir}" \
    --lib eshkol_transformer_o2_runtime \
    "${PROJECT_ROOT}/tests/o2/config_negatives_runtime.esk" \
    -o "${temporary_dir}/config-negatives" \
    >"${temporary_dir}/config-compile.log" 2>&1
timeout --foreground --signal=TERM --kill-after=5s 900s \
  "${temporary_dir}/config-negatives" \
  >"${temporary_dir}/config-negatives.stdout"
grep -Fx 'O2 config/state adversarial PASS: 57 checks' \
  "${temporary_dir}/config-negatives.stdout" >/dev/null

for private_binding in o2-provider-name o2-native-builder-create \
    o2-native-optimizer-step o2-optimizer-tag o2-state-tag \
    o2-optimizer-completed-updates-internal \
    o2-optimizer-schedule-factor-bits-internal \
    i2-native-owned-release tensor-provider-release-owned-internal!; do
  if run_compiler "private-${private_binding}" --strict-types --no-stdlib \
      -I "${PROJECT_ROOT}/lib" -L "${artifact_dir}" \
      --lib eshkol_transformer_wave2 \
      -e "(begin (require transformer.optim) ${private_binding})" \
      >"${temporary_dir}/private-${private_binding}.stdout" \
      2>"${temporary_dir}/private-${private_binding}.stderr"; then
    die "O2 aggregate exposed private binding ${private_binding}"
  fi
  grep -F "${private_binding}" \
    "${temporary_dir}/private-${private_binding}.stderr" >/dev/null || \
    die "O2 private-binding negative failed for the wrong reason"
done

mkdir -p "${temporary_dir}/shadow/transformer"
printf '(error "hostile O2 root loaded")\n' \
  >"${temporary_dir}/shadow/o2_wave2_root.esk"
printf '(error "hostile optimizer facade loaded")\n' \
  >"${temporary_dir}/shadow/transformer/optim.esk"
ESHKOL_PATH="${temporary_dir}/shadow" ESHKOL_LIB_DIR="${temporary_dir}/shadow" \
E1B_COMPILER_TIMEOUT_SECONDS="${compiler_timeout}" \
  /usr/bin/bash "${PROJECT_ROOT}/scripts/build-o2.sh" \
    "${temporary_dir}/hostile-build"
cmp "${object}" "${temporary_dir}/hostile-build/o2_wave2.o"
for deterministic_evidence in global-defined.txt package-exports.txt \
    undefined.txt expected-undefined.txt public-strings.txt private.d link.map; do
  cmp "${evidence}/${deterministic_evidence}" \
    "${temporary_dir}/hostile-build/o2_wave2.o.evidence/${deterministic_evidence}"
done
if grep -F "${temporary_dir}/shadow" \
    "${temporary_dir}/hostile-build/o2_wave2.o.evidence/private.d" >/dev/null; then
  die "O2 build admitted a hostile module path"
fi

reject_builder_input() {
  local label=$1
  local expected=$2
  shift 2
  if E1B_COMPILER_TIMEOUT_SECONDS="${compiler_timeout}" \
      "${PROJECT_ROOT}/scripts/build-e1b-consumer.sh" "$@" \
      >"${temporary_dir}/${label}.stdout" \
      2>"${temporary_dir}/${label}.stderr"; then
    die "O2 builder admitted ${label}"
  fi
  grep -F "${expected}" "${temporary_dir}/${label}.stderr" >/dev/null || \
    die "O2 ${label} rejection reported the wrong reason"
}
cp "${PROJECT_ROOT}/native/o2_wave2_root.esk" \
  "${temporary_dir}/copied-o2-root.esk"
for rejected_root in "${temporary_dir}/copied-o2-root.esk" "${object}"; do
  label="rejected-root-$(basename -- "${rejected_root}")"
  reject_builder_input "${label}" \
    'repository package components require their exact repository-owned private root' \
    "${rejected_root}" \
    "${PROJECT_ROOT}/native/o2_wave2_package_bridge.c" \
    "${PROJECT_ROOT}/native/o2_wave2_private_renames.txt" \
    "${PROJECT_ROOT}/native/o2_wave2_public_exports.txt" \
    "${temporary_dir}/${label}.o" \
    "${PROJECT_ROOT}/internal/p1/lib" "${PROJECT_ROOT}/internal/c1/lib" \
    "${PROJECT_ROOT}/internal/t1/lib" "${PROJECT_ROOT}/src"
  [[ ! -e "${temporary_dir}/${label}.o" && \
     ! -e "${temporary_dir}/${label}.o.evidence" ]] || \
    die "rejected O2 builder input published an artifact"
done
label=rejected-root-i2_wave2_root.esk
reject_builder_input "${label}" \
  'I2 aggregate policy requires the exact repository bridge' \
  "${PROJECT_ROOT}/native/i2_wave2_root.esk" \
  "${PROJECT_ROOT}/native/o2_wave2_package_bridge.c" \
  "${PROJECT_ROOT}/native/o2_wave2_private_renames.txt" \
  "${PROJECT_ROOT}/native/o2_wave2_public_exports.txt" \
  "${temporary_dir}/${label}.o" \
  "${PROJECT_ROOT}/internal/p1/lib" "${PROJECT_ROOT}/internal/c1/lib" \
  "${PROJECT_ROOT}/internal/t1/lib" "${PROJECT_ROOT}/src"
[[ ! -e "${temporary_dir}/${label}.o" && \
   ! -e "${temporary_dir}/${label}.o.evidence" ]] || \
  die "rejected I2/O2 package tuple published an artifact"

if "${cc}" -r -Wl,--whole-archive "${library}" "${library}" \
    -Wl,--no-whole-archive -o "${temporary_dir}/duplicate-authority.o" \
    >"${temporary_dir}/duplicate.stdout" \
    2>"${temporary_dir}/duplicate.stderr"; then
  die "linker admitted duplicate O2 aggregate authority"
fi
grep -E 'multiple definition.*et_e1b_error_(predicate|category)_v1' \
  "${temporary_dir}/duplicate.stderr" >/dev/null || \
  die "duplicate O2 authority rejection did not identify E1 ownership"

for aggregate_pair in i2 t2 d2; do
  if [[ "${aggregate_pair}" == i2 ]]; then
    other_aggregate="${i2_aggregate}"
  elif [[ "${aggregate_pair}" == t2 ]]; then
    other_aggregate="${t2_aggregate}"
  else
    other_aggregate="${d2_aggregate}"
  fi
  [[ -r "${other_aggregate}" ]] || \
    die "${aggregate_pair^^} aggregate is missing for O2 collision evidence"
  if "${cc}" -r -Wl,--whole-archive "${library}" "${other_aggregate}" \
      -Wl,--no-whole-archive \
      -o "${temporary_dir}/o2-${aggregate_pair}-collision.o" \
      >"${temporary_dir}/o2-${aggregate_pair}-collision.stdout" \
      2>"${temporary_dir}/o2-${aggregate_pair}-collision.stderr"; then
    die "linker admitted O2 with ${aggregate_pair^^} registry authority"
  fi
  grep -E 'multiple definition.*et_e1b_error_(predicate|category)_v1' \
    "${temporary_dir}/o2-${aggregate_pair}-collision.stderr" >/dev/null || \
    die "O2/${aggregate_pair^^} collision did not identify E1 ownership"
done

if grep -Ein 'python|pytorch|torch' \
    "${PROJECT_ROOT}/native/o2_wave2_root.esk" \
    "${PROJECT_ROOT}/native/o2_wave2_package_bridge.c" \
    "${PROJECT_ROOT}/native/o2_optimizer.c" \
    "${PROJECT_ROOT}/native/o2_optimizer_internal.h" \
    "${PROJECT_ROOT}/lib/transformer/optim.esk"; then
  die "O2 production path contains a forbidden Python/PyTorch reference"
fi
if strings "${object}" | grep -Ein 'python|pytorch|torch' >/dev/null; then
  die "O2 aggregate contains a Python/PyTorch runtime reference"
fi

python3 -m unittest -v tests.o2.test_reference \
  >"${temporary_dir}/reference.stdout" 2>"${temporary_dir}/reference.stderr"
grep -F 'Ran 5 tests' "${temporary_dir}/reference.stderr" >/dev/null
grep -F 'OK' "${temporary_dir}/reference.stderr" >/dev/null
if [[ -z "${O2_ORACLE_PYTHON:-}" ]]; then
  grep -F 'OK (skipped=1)' "${temporary_dir}/reference.stderr" >/dev/null
else
  if grep -F 'skipped=' "${temporary_dir}/reference.stderr" >/dev/null; then
    die "O2 pinned-oracle regeneration was unexpectedly skipped"
  fi
fi

if [[ "${O2_ASAN_DETECT_LEAKS:-1}" == 1 ]]; then
  sanitizer_summary='sanitizers/LSan'
else
  sanitizer_summary='sanitizers (LSan disabled by caller)'
fi
printf 'O2 PASS: native optimizer, 6201 adversarial checks, config/state AOT, reference parity, exact 53-global successor, authority isolation, %s, and production isolation\n' \
  "${sanitizer_summary}"
