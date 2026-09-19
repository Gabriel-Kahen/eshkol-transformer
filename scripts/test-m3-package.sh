#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
for command in ar cmp env nm timeout; do require_command "${command}"; done
m3_time="${M3_TIME_EXECUTABLE:-/usr/bin/time}"
[[ "${m3_time}" == /* && -x "${m3_time}" ]] || die "M3 RSS witness requires an absolute GNU time executable"
"${m3_time}" --version | grep -F "GNU Time" >/dev/null || die "M3 RSS witness requires GNU time"
python3 -m unittest -v tests.m3.test_package_contract tests.m3.test_retention_report
m3_test_tmp="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-transformer-m3.XXXXXX")"
m3_cleanup() {
  local status=$?
  if [[ "${status}" == 0 ]]; then
    rm -rf -- "${m3_test_tmp}"
  else
    printf 'M3 failure evidence retained at %s\n' "${m3_test_tmp}" >&2
  fi
}
trap m3_cleanup EXIT
m3_canonical="$(project_build_dir)/m3"
m3_i2_object="$(project_build_dir)/i2/i2_wave2.o"
m3_caller_install="${m3_test_tmp}/caller-install"
[[ -s "${m3_canonical}/m3_package.o" && -s "${m3_canonical}/libeshkol_transformer_m3.a" && \
   -s "${m3_i2_object}" ]] || die "M3 and I2 canonical CI prerequisites are missing"
cmp "${PROJECT_ROOT}/native/m3_package_defined_symbols.txt" \
  "${m3_canonical}/m3_package.o.evidence/global-defined.txt"
m3_provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
m3_cxx="$(tsv_value "${m3_provenance}" cxx_path)"
m3_runner="$(eshkol_build_dir)/eshkol-run"
m3_timeout="${M3_COMPILER_TIMEOUT_SECONDS:-600}"
[[ "${m3_timeout}" =~ ^[1-9][0-9]*$ ]] || die "M3 compiler timeout must be positive"
[[ -s "${PROJECT_ROOT}/tests/m3/public_runtime.esk" ]] || die "M3 public transport witness missing"
m3_compile() {
  local installed=$1
  shift
  env -u CPATH -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH -u OBJC_INCLUDE_PATH \
    -u GCC_EXEC_PREFIX -u COMPILER_PATH -u LIBRARY_PATH \
    -u CLANG_CONFIG_FILE -u CCC_OVERRIDE_OPTIONS -u CCC_CC -u CCC_CXX \
    -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
    XDG_CACHE_HOME="${m3_test_tmp}/cache" ESHKOL_LIB_DIR="${installed}/facades" \
    ESHKOL_CXX_COMPILER="${m3_cxx}" \
    timeout --foreground --signal=TERM --kill-after=5s "${m3_timeout}s" \
    "${m3_runner}" --strict-types --no-stdlib -I "${installed}/facades" "$@"
}
# Imported diagnostic filenames are embedded in caller objects. Use the same
# clean installation path for each independently built package, without stripping.
m3_install_for_caller() {
  rm -rf -- "${m3_caller_install:?}"
  mkdir -p "${m3_caller_install}"
  cp -a "$1/facades" "${m3_caller_install}/facades"
  cp "$1/libeshkol_transformer_m3.a" "${m3_caller_install}/"
}
m3_compile_public() {
  local installed=$1 source=$2 output=$3
  shift 3
  rm -f -- "${output}.d"
  m3_compile "${installed}" "$@" --compile-only --emit-depfile "${output}.d" \
    "${source}" -o "${output}.o"
  python3 "${PROJECT_ROOT}/tests/m3/check_public_closure.py" \
    "${installed}" "${source}" "${output}.d"
  m3_compile "${installed}" "$@" -L "${installed}" --lib eshkol_transformer_m3 \
    "${source}" -o "${output}"
}
for repetition in 1 2; do
  installed="${m3_test_tmp}/fresh-${repetition}"
  if [[ "${repetition}" == 2 ]]; then
    mkdir -p "${m3_test_tmp}/hostile/eshkol_transformer"
    printf '#error hostile ambient include entered trusted compilation\n' \
      >"${m3_test_tmp}/hostile/stdint.h"
    cp "${m3_test_tmp}/hostile/stdint.h" \
      "${m3_test_tmp}/hostile/eshkol_transformer/n3k_primitives_abi.h"
    env CPATH="${m3_test_tmp}/hostile" C_INCLUDE_PATH="${m3_test_tmp}/hostile" \
      CPLUS_INCLUDE_PATH="${m3_test_tmp}/hostile" ESHKOL_PATH="${m3_test_tmp}/hostile" \
      CLANG_CONFIG_FILE="${m3_test_tmp}/hostile/forbidden.cfg" \
      /usr/bin/bash "${PROJECT_ROOT}/scripts/build-m3.sh" "${installed}"
  else
    /usr/bin/bash "${PROJECT_ROOT}/scripts/build-m3.sh" "${installed}"
  fi
  for pair in 'defined_symbols global-defined' 'public_exports package-exports' \
      'public_strings public-strings' 'undefined_symbols undefined' \
      'source_closure source-closure' 'native_source_closure native-source-closure' \
      'native_objects native-objects'; do
    read -r expected observed <<<"${pair}"
    cmp "${PROJECT_ROOT}/native/m3_package_${expected}.txt" \
      "${installed}/m3_package.o.evidence/${observed}.txt"
  done
  (cd "${installed}/facades" && find . -type f -printf '%P\n' | LC_ALL=C sort) \
    >"${m3_test_tmp}/facades-${repetition}.txt"
  cmp "${PROJECT_ROOT}/native/m3_package_facades.txt" "${m3_test_tmp}/facades-${repetition}.txt"
  m3_install_for_caller "${installed}"
  m3_compile "${m3_caller_install}" --compile-only \
    "${PROJECT_ROOT}/tests/m3/compile_api.esk" -o "${installed}/api.o"
  m3_compile_public "${m3_caller_install}" \
    "${PROJECT_ROOT}/tests/m3/public_runtime.esk" "${installed}/public"
  if ! timeout --foreground --signal=TERM --kill-after=5s 120s "${installed}/public" \
      >"${installed}/public.stdout" 2>"${installed}/public.stderr"; then
    cat "${installed}/public.stdout" "${installed}/public.stderr" >&2
    die "M3 installed public witness failed"
  fi
  python3 -m tests.m3.check_public_parity \
    --reference "$(project_build_dir)/m3-reference/development.json" \
    --public-output "${installed}/public.stdout"
  if nm -u --format=posix "${installed}/api.o" | \
      grep -E 'et_(m3|m3t|i2|f32|p1|kernel)_private|et_e1b_private_'; then
    die "M3 facade retained private authority"
  fi
  nm -u --format=posix "${installed}/api.o" | awk '$1 ~ /^et_e1b_public_m3_/ {print $1}' | LC_ALL=C sort -u \
    >"${installed}/api-undefined.txt"
  grep '^et_e1b_public_m3_' "${PROJECT_ROOT}/native/m3_package_public_exports.txt" \
    >"${installed}/expected-api-undefined.txt"
  cmp "${installed}/expected-api-undefined.txt" "${installed}/api-undefined.txt"
  printf 'M3 fresh package %s: exact artifact and installed public witness passed\n' "${repetition}"
done
for output in m3_package.o libeshkol_transformer_m3.a api.o public public.stdout; do
  cmp "${m3_test_tmp}/fresh-1/${output}" "${m3_test_tmp}/fresh-2/${output}"
done
cmp "${m3_canonical}/m3_package.o" "${m3_test_tmp}/fresh-1/m3_package.o"
cmp "${m3_canonical}/libeshkol_transformer_m3.a" "${m3_test_tmp}/fresh-1/libeshkol_transformer_m3.a"
# Reversed import order still obtains the aggregate's one E1 registry.
{ printf '(require transformer.error_public)\n'; cat "${PROJECT_ROOT}/tests/m3/public_runtime.esk"; } \
  >"${m3_test_tmp}/reverse.esk"
m3_install_for_caller "${m3_test_tmp}/fresh-1"
m3_compile_public "${m3_caller_install}" "${m3_test_tmp}/reverse.esk" "${m3_test_tmp}/reverse"
"${m3_test_tmp}/reverse" >"${m3_test_tmp}/reverse.stdout"
cmp "${m3_test_tmp}/fresh-1/public.stdout" "${m3_test_tmp}/reverse.stdout"
# Local symbols cannot become callable merely by spelling a private extern.
for private in et_m3_private_graph_capture_v1 et_m3_private_workspace_reset_v1 \
    et_m3_private_i64_unborrowed_v1 et_e1b_private_m3_model_forward_cabi_v1 \
    et_m3t_private_input_create_v1 et_p1_private_construction_begin_v1 \
    et_f32_tensor_scoped_begin_internal; do
  printf '(extern ptr hidden :real %s)\n(display (hidden))\n' "${private}" \
    >"${m3_test_tmp}/negative.esk"
  m3_compile "${m3_caller_install}" --compile-only \
    "${m3_test_tmp}/negative.esk" -o "${m3_test_tmp}/negative.o"
  nm -u --format=posix "${m3_test_tmp}/negative.o" | \
    awk '{print $1}' | grep -Fx "${private}" >/dev/null || \
    die "M3 private-link negative omitted its intended symbol"
  if m3_compile "${m3_caller_install}" -L "${m3_caller_install}" \
      --lib eshkol_transformer_m3 "${m3_test_tmp}/negative.esk" \
      -o "${m3_test_tmp}/negative" >"${m3_test_tmp}/negative.log" 2>&1; then
    die "M3 private symbol escaped localization: ${private}"
  fi
  grep -F "${private}" "${m3_test_tmp}/negative.log" >/dev/null || \
    die "M3 private symbol fixture failed for another reason"
done
for private in m3-public-model-forward m3-forward-schedule-internal m3-reverse-schedule-internal \
    m3t-workspace-contributions-internal i2-construction-begin-internal; do
  printf '(require transformer.model)\n%s\n' "${private}" \
    >"${m3_test_tmp}/private-binding.esk"
  if m3_compile "${m3_caller_install}" --compile-only \
      "${m3_test_tmp}/private-binding.esk" -o "${m3_test_tmp}/private-binding.o" \
      >"${m3_test_tmp}/private-binding.log" 2>&1; then
    die "M3 facade imported private binding ${private}"
  fi
  grep -F "${private}" "${m3_test_tmp}/private-binding.log" >/dev/null || \
    die "M3 private-binding fixture rejected for another reason"
done
for entry in 'model-forward 2' 'model-output-logits 1' 'model-output-loss 1' \
    'model-output-rng 1' 'model-output-release! 1' 'diagnostic-output-vjp! 4' \
    'diagnostic-model-logits-bits 1' 'diagnostic-model-logits-release! 1'; do
  read -r public arity <<<"${entry}"
  # Too few arguments covers the variadic minimum as well as fixed arities.
  printf '(require transformer.model)\n(%s)\n' "${public}" >"${m3_test_tmp}/arity.esk"
  if m3_compile "${m3_caller_install}" --compile-only \
      "${m3_test_tmp}/arity.esk" -o "${m3_test_tmp}/arity.o" \
      >"${m3_test_tmp}/arity.log" 2>&1; then
    die "M3 wrong arity was accepted for ${public}"
  fi
  grep -F "${public}" "${m3_test_tmp}/arity.log" >/dev/null || \
    die "M3 arity fixture rejected for another binding"
  if [[ "${public}" == model-forward ]]; then
    # The compiler diagnoses the lowered variadic call, including its rest list.
    grep -F 'Arity mismatch in no-capture call to model-forward:' \
      "${m3_test_tmp}/arity.log" >/dev/null || \
      die "M3 variadic minimum rejected for another reason"
  else
    grep -F "Arity mismatch: ${public} expects ${arity} arguments but got 0" \
      "${m3_test_tmp}/arity.log" >/dev/null || \
      die "M3 arity fixture rejected for another reason"
  fi
  if [[ "${public}" != model-forward ]]; then
    {
      printf '(require transformer.model)\n(%s' "${public}"
      for ((argument = 0; argument <= arity; ++argument)); do printf ' #f'; done
      printf ')\n'
    } >"${m3_test_tmp}/arity.esk"
    if m3_compile "${m3_caller_install}" --compile-only \
        "${m3_test_tmp}/arity.esk" -o "${m3_test_tmp}/arity.o" \
        >"${m3_test_tmp}/arity.log" 2>&1; then
      die "M3 excess arguments were accepted for ${public}"
    fi
    grep -F "${public}" "${m3_test_tmp}/arity.log" >/dev/null || \
      die "M3 excess-arity fixture rejected for another binding"
    grep -F "function '${public}' expects " "${m3_test_tmp}/arity.log" | \
      grep -F "expects ${arity} arguments, got $((arity + 1))" >/dev/null || \
      die "M3 excess-arity fixture rejected for another reason"
  fi
done
# Two registry objects are always rejected, regardless of archive lazy extraction.
if "${m3_cxx}" -r "${m3_test_tmp}/fresh-1/m3_package.o" \
    "${m3_test_tmp}/fresh-2/m3_package.o" -o "${m3_test_tmp}/duplicate.o" \
    >"${m3_test_tmp}/duplicate.log" 2>&1; then
  die "M3 duplicate registry objects linked"
fi
grep -E 'multiple definition|duplicate symbol' "${m3_test_tmp}/duplicate.log" >/dev/null || \
  die "M3 duplicate registry fixture failed for another reason"
if "${m3_cxx}" -r "${m3_test_tmp}/fresh-1/m3_package.o" \
    "${m3_i2_object}" -o "${m3_test_tmp}/cross-aggregate.o" \
    >"${m3_test_tmp}/cross-aggregate.log" 2>&1; then
  die "M3 and I2 registry objects linked into one process"
fi
grep -E 'multiple definition|duplicate symbol' "${m3_test_tmp}/cross-aggregate.log" >/dev/null || \
  die "M3 cross-aggregate fixture failed for another reason"
# Root-owned output/graph lifetime witness uses only the installed public API.
m3_compile_public "${m3_caller_install}" "${PROJECT_ROOT}/tests/m3/lifetime.esk" \
  "${m3_test_tmp}/lifetime" -O 2
timeout --foreground --signal=TERM --kill-after=5s 240s "${m3_test_tmp}/lifetime" \
  >"${m3_test_tmp}/lifetime.stdout" 2>"${m3_test_tmp}/lifetime.stderr"
printf 'M3-LIFETIME-PASS\n' >"${m3_test_tmp}/lifetime.expected"
cmp "${m3_test_tmp}/lifetime.expected" "${m3_test_tmp}/lifetime.stdout"

# Ordinary control retirement is explicitly measured; no flat-memory assertion.
m3_compile_public "${m3_caller_install}" "${PROJECT_ROOT}/tests/m3/arena_retention.esk" \
  "${m3_test_tmp}/arena-retention" -O 2
printf 'mode\thorizon\tarena_bytes\tpeak_rss_kib\n' >"${m3_test_tmp}/arena-measurements.tsv"
for mode in forward logits vjp reset failure; do
  for horizon in 1024 8192; do
    prefix="${m3_test_tmp}/arena-${mode}-${horizon}"
    ESHKOL_ARENA_REPORT=1 ESHKOL_ARENA_POISON=1 \
      "${m3_time}" -f 'max_rss_kib=%M\nelapsed_seconds=%e' \
      timeout --foreground --signal=TERM --kill-after=5s 600s \
      "${m3_test_tmp}/arena-retention" "${horizon}" "${mode}" \
      >"${prefix}.stdout" 2>"${prefix}.stderr"
    printf 'M3-ARENA-RETENTION-PASS %s %s\n' "${mode}" "${horizon}" >"${prefix}.expected"
    cmp "${prefix}.expected" "${prefix}.stdout"
    retained="$(awk -F= '/global_total_allocated_bytes=/{print $2}' "${prefix}.stderr" | tail -1)"
    peak_rss="$(awk -F= '/^max_rss_kib=/{print $2}' "${prefix}.stderr")"
    [[ "${retained}" =~ ^[0-9]+$ && "${peak_rss}" =~ ^[0-9]+$ ]] || die "M3 arena/RSS counters missing"
    printf '%s\t%s\t%s\t%s\n' "${mode}" "${horizon}" "${retained}" "${peak_rss}" \
      >>"${m3_test_tmp}/arena-measurements.tsv"
  done
done
python3 "${PROJECT_ROOT}/tests/m3/check_m3_retention.py" "${m3_test_tmp}/arena-measurements.tsv"
m3_evidence_dir="$(project_build_dir)/m3-public-evidence"
mkdir -p "${m3_evidence_dir}"
cp "${m3_test_tmp}/arena-measurements.tsv" "${m3_test_tmp}"/arena-*.stdout \
  "${m3_test_tmp}"/arena-*.stderr "${m3_test_tmp}/lifetime.stdout" \
  "${m3_test_tmp}/lifetime.stderr" "${m3_evidence_dir}/"
printf 'M3 packaging and installed-facade AOT gate passed\n'
