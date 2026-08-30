#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_command timeout
require_command python3
require_command ar
require_command diff
require_command env
require_command find
require_command nm
require_command readelf
require_command strings
verify_toolchain

D1_TIMEOUT=$(command -v timeout)
D1_PYTHON=$(command -v python3)
D1_RUNNER="$(eshkol_build_dir)/eshkol-run"
D1_CC="$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cc_path)"
D1_CXX="$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cxx_path)"
D1_TIMEOUT_SECONDS=${D1_TIMEOUT_SECONDS:-60}
D1_ARTIFACT_DIR="$(project_build_dir)/d1"
D1_LIBRARY="${D1_ARTIFACT_DIR}/libeshkol_transformer_d1.a"
D1_EVIDENCE="${D1_LIBRARY}.evidence"
D1_PUBLIC_EXPORTS="${PROJECT_ROOT}/native/d1_e1b_public_exports.txt"
D1_UNDEFINED_SYMBOLS="${PROJECT_ROOT}/native/d1_e1b_undefined_symbols.txt"
[[ "${D1_TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*$ ]] || \
  die "D1_TIMEOUT_SECONDS must be a positive integer"
[[ -r "${D1_LIBRARY}" ]] || die "canonical D1 combined archive is missing"
[[ "$(ar t "${D1_LIBRARY}")" == "stdlib.o" ]] || \
  die "canonical D1 combined archive must contain exactly stdlib.o"
for evidence_file in global-defined.txt package-exports.txt undefined.txt \
    expected-undefined.txt readelf-symbols.txt nm.txt strings.txt private.d \
    link.map allowlist-provenance.tsv; do
  [[ -r "${D1_EVIDENCE}/${evidence_file}" ]] || \
    die "canonical D1 evidence is missing ${evidence_file}"
done

D1_TMP=$(mktemp -d "${TMPDIR:-/tmp}/eshkol-transformer-d1.XXXXXX")
d1_cleanup() {
  rm -rf -- "${D1_TMP}"
  if [[ -n "${D1_DIRECT_FAULT_OUTPUT:-}" ]]; then
    rm -f -- "${D1_DIRECT_FAULT_OUTPUT}"
    rm -rf -- "${D1_DIRECT_FAULT_OUTPUT}.evidence"
  fi
}
trap d1_cleanup EXIT

d1_assert_builder_rejected() {
  local name=$1 expected=$2 policy=$3 root=$4 bridge=$5 renames=$6 exports=$7
  shift 7
  local case_dir="${D1_TMP}/builder-negative-${name}"
  local output="${case_dir}/candidate.o"
  local evidence="${output}.evidence"
  local log="${case_dir}/rejected.log"
  local empty_tmp="${case_dir}/tmp"
  local missing_toolchain="${case_dir}/missing-toolchain"
  local -a builder_command
  mkdir -p "${evidence}" "${empty_tmp}"
  printf 'preexisting-output:%s\n' "${name}" >"${output}"
  printf 'preexisting-evidence:%s\n' "${name}" >"${evidence}/sentinel"
  cp "${output}" "${case_dir}/expected-output"
  cp "${evidence}/sentinel" "${case_dir}/expected-evidence"

  builder_command=(
    "${PROJECT_ROOT}/scripts/build-e1b-consumer.sh"
    "${root}" "${bridge}" "${renames}" "${exports}" "${output}" "$@"
  )
  if [[ "${policy}" == unset ]]; then
    if env -u E1B_PACKAGE_POLICY TMPDIR="${empty_tmp}" \
        ESHKOL_BUILD_DIR="${missing_toolchain}" \
        "${builder_command[@]}" >"${log}" 2>&1; then
      die "D1 builder negative unexpectedly passed: ${name}"
    fi
  else
    if env E1B_PACKAGE_POLICY="${policy}" TMPDIR="${empty_tmp}" \
        ESHKOL_BUILD_DIR="${missing_toolchain}" \
        "${builder_command[@]}" >"${log}" 2>&1; then
      die "D1 builder policy override unexpectedly passed: ${name}"
    fi
  fi
  grep -F "${expected}" "${log}" >/dev/null || \
    die "D1 builder negative emitted the wrong diagnostic: ${name}"
  cmp --silent "${case_dir}/expected-output" "${output}" || \
    die "D1 builder negative mutated its preexisting output: ${name}"
  cmp --silent "${case_dir}/expected-evidence" "${evidence}/sentinel" || \
    die "D1 builder negative mutated its evidence sentinel: ${name}"
  [[ "$(find "${evidence}" -mindepth 1 -maxdepth 1 -type f -printf '%f\n')" == sentinel ]] || \
    die "D1 builder negative published evidence: ${name}"
  [[ -z "$(find "${empty_tmp}" -mindepth 1 -print -quit)" ]] || \
    die "D1 builder negative created a build temporary: ${name}"
  if find "${case_dir}" -mindepth 1 \
      \( -name 'candidate.o.tmp.*' -o -name 'candidate.o.evidence.tmp.*' \) \
      -print -quit | grep . >/dev/null; then
    die "D1 builder negative left a publication temporary: ${name}"
  fi
}

D1_ADMISSION_FIXTURES="${D1_TMP}/builder-admission-fixtures"
mkdir -p "${D1_ADMISSION_FIXTURES}/copied-src" \
  "${D1_ADMISSION_FIXTURES}/shadow/eshkol_transformer"
cp "${PROJECT_ROOT}/native/d1_e1b_private.esk" \
  "${D1_ADMISSION_FIXTURES}/copied-root.esk"
cp "${PROJECT_ROOT}/tests/d1/d1_e1b_fault_root.esk" \
  "${D1_ADMISSION_FIXTURES}/copied-fault-root.esk"
cp "${PROJECT_ROOT}/native/d1_e1b_package_bridge.c" \
  "${D1_ADMISSION_FIXTURES}/copied-bridge.c"
cp "${PROJECT_ROOT}/native/d1_e1b_private_renames.txt" \
  "${D1_ADMISSION_FIXTURES}/copied-renames.txt"
cp "${PROJECT_ROOT}/native/d1_e1b_public_exports.txt" \
  "${D1_ADMISSION_FIXTURES}/copied-exports.txt"
printf '(error "shadow token shard must never load")\n' \
  >"${D1_ADMISSION_FIXTURES}/shadow/eshkol_transformer/token_shard.esk"
printf '(error "shadow D1 private root must never load")\n' \
  >"${D1_ADMISSION_FIXTURES}/shadow/d1_e1b_private.esk"

d1_assert_builder_rejected copied-root \
  'repository package components require their exact repository-owned private root' \
  unset "${D1_ADMISSION_FIXTURES}/copied-root.esk" \
  "${PROJECT_ROOT}/native/d1_e1b_package_bridge.c" \
  "${PROJECT_ROOT}/native/d1_e1b_private_renames.txt" \
  "${PROJECT_ROOT}/native/d1_e1b_public_exports.txt" "${PROJECT_ROOT}/src"
d1_assert_builder_rejected arbitrary-root \
  'repository package components require their exact repository-owned private root' \
  unset "${PROJECT_ROOT}/tests/fixtures/e1b/trusted_fixture_package.esk" \
  "${PROJECT_ROOT}/native/d1_e1b_package_bridge.c" \
  "${PROJECT_ROOT}/native/d1_e1b_private_renames.txt" \
  "${PROJECT_ROOT}/native/d1_e1b_public_exports.txt" "${PROJECT_ROOT}/src"
d1_assert_builder_rejected copied-fault-root \
  'repository package components require their exact repository-owned private root' \
  unset "${D1_ADMISSION_FIXTURES}/copied-fault-root.esk" \
  "${PROJECT_ROOT}/native/d1_e1b_package_bridge.c" \
  "${PROJECT_ROOT}/native/d1_e1b_private_renames.txt" \
  "${PROJECT_ROOT}/native/d1_e1b_public_exports.txt" "${PROJECT_ROOT}/src"
d1_assert_builder_rejected copied-bridge \
  'D1 package policy requires the exact repository D1 bridge' unset \
  "${PROJECT_ROOT}/native/d1_e1b_private.esk" \
  "${D1_ADMISSION_FIXTURES}/copied-bridge.c" \
  "${PROJECT_ROOT}/native/d1_e1b_private_renames.txt" \
  "${PROJECT_ROOT}/native/d1_e1b_public_exports.txt" "${PROJECT_ROOT}/src"
d1_assert_builder_rejected copied-renames \
  'D1 package policy requires the exact repository D1 rename map' unset \
  "${PROJECT_ROOT}/native/d1_e1b_private.esk" \
  "${PROJECT_ROOT}/native/d1_e1b_package_bridge.c" \
  "${D1_ADMISSION_FIXTURES}/copied-renames.txt" \
  "${PROJECT_ROOT}/native/d1_e1b_public_exports.txt" "${PROJECT_ROOT}/src"
d1_assert_builder_rejected copied-exports \
  'D1 package policy requires the exact repository D1 export list' unset \
  "${PROJECT_ROOT}/native/d1_e1b_private.esk" \
  "${PROJECT_ROOT}/native/d1_e1b_package_bridge.c" \
  "${PROJECT_ROOT}/native/d1_e1b_private_renames.txt" \
  "${D1_ADMISSION_FIXTURES}/copied-exports.txt" "${PROJECT_ROOT}/src"
d1_assert_builder_rejected copied-include \
  'D1 package policy requires exactly the repository src include directory' unset \
  "${PROJECT_ROOT}/native/d1_e1b_private.esk" \
  "${PROJECT_ROOT}/native/d1_e1b_package_bridge.c" \
  "${PROJECT_ROOT}/native/d1_e1b_private_renames.txt" \
  "${PROJECT_ROOT}/native/d1_e1b_public_exports.txt" \
  "${D1_ADMISSION_FIXTURES}/copied-src"
d1_assert_builder_rejected missing-include \
  'D1 package policy requires exactly the repository src include directory' unset \
  "${PROJECT_ROOT}/native/d1_e1b_private.esk" \
  "${PROJECT_ROOT}/native/d1_e1b_package_bridge.c" \
  "${PROJECT_ROOT}/native/d1_e1b_private_renames.txt" \
  "${PROJECT_ROOT}/native/d1_e1b_public_exports.txt"
d1_assert_builder_rejected extra-include \
  'D1 package policy requires exactly the repository src include directory' unset \
  "${PROJECT_ROOT}/native/d1_e1b_private.esk" \
  "${PROJECT_ROOT}/native/d1_e1b_package_bridge.c" \
  "${PROJECT_ROOT}/native/d1_e1b_private_renames.txt" \
  "${PROJECT_ROOT}/native/d1_e1b_public_exports.txt" \
  "${PROJECT_ROOT}/src" "${PROJECT_ROOT}/native"
for override in base-e1b x1 d1 d1-test-faults arbitrary; do
  d1_assert_builder_rejected "policy-override-${override}" \
    'E1B_PACKAGE_POLICY overrides are forbidden' "${override}" \
    "${PROJECT_ROOT}/tests/fixtures/e1b/trusted_fixture_package.esk" \
    "${PROJECT_ROOT}/tests/fixtures/e1b/fixture_package_bridge.c" \
    "${PROJECT_ROOT}/tests/fixtures/e1b/fixture_package_renames.txt" \
    "${PROJECT_ROOT}/tests/fixtures/e1b/fixture_public_exports.txt" \
    "${PROJECT_ROOT}/tests/fixtures/e1b/public"
done

D1_DIRECT_FAULT_OUTPUT="${D1_ARTIFACT_DIR}/.rejected-direct-fault-route.o"
D1_DIRECT_FAULT_EVIDENCE="${D1_DIRECT_FAULT_OUTPUT}.evidence"
mkdir -p "${D1_DIRECT_FAULT_EVIDENCE}"
printf 'preexisting-direct-fault-output\n' >"${D1_DIRECT_FAULT_OUTPUT}"
printf 'preexisting-direct-fault-evidence\n' \
  >"${D1_DIRECT_FAULT_EVIDENCE}/sentinel"
cp "${D1_DIRECT_FAULT_OUTPUT}" "${D1_TMP}/expected-direct-fault-output"
cp "${D1_DIRECT_FAULT_EVIDENCE}/sentinel" \
  "${D1_TMP}/expected-direct-fault-evidence"
if env -u E1B_PACKAGE_POLICY \
    ESHKOL_BUILD_DIR="${D1_TMP}/missing-direct-fault-toolchain" \
    "${PROJECT_ROOT}/scripts/build-e1b-consumer.sh" \
      "${PROJECT_ROOT}/tests/d1/d1_e1b_fault_root.esk" \
      "${PROJECT_ROOT}/native/d1_e1b_package_bridge.c" \
      "${PROJECT_ROOT}/native/d1_e1b_private_renames.txt" \
      "${PROJECT_ROOT}/native/d1_e1b_public_exports.txt" \
      "${D1_DIRECT_FAULT_OUTPUT}" "${PROJECT_ROOT}/src" \
      >"${D1_TMP}/direct-fault-route.stdout" \
      2>"${D1_TMP}/direct-fault-route.stderr"; then
  die "direct D1 test-fault route unexpectedly targeted the production directory"
fi
grep -F 'D1 test-fault object cannot target the canonical production directory' \
  "${D1_TMP}/direct-fault-route.stderr" >/dev/null
cmp --silent "${D1_TMP}/expected-direct-fault-output" \
  "${D1_DIRECT_FAULT_OUTPUT}" || \
  die "rejected direct fault route mutated its production-directory output"
cmp --silent "${D1_TMP}/expected-direct-fault-evidence" \
  "${D1_DIRECT_FAULT_EVIDENCE}/sentinel" || \
  die "rejected direct fault route mutated its production-directory evidence"
[[ "$(find "${D1_DIRECT_FAULT_EVIDENCE}" -mindepth 1 -maxdepth 1 \
    -type f -printf '%f\n')" == sentinel ]] || \
  die "rejected direct fault route published production-directory evidence"
rm -f -- "${D1_DIRECT_FAULT_OUTPUT}"
rm -rf -- "${D1_DIRECT_FAULT_EVIDENCE}"

D1_PUBLIC_ROOT="${D1_TMP}/public-root"
mkdir -p "${D1_PUBLIC_ROOT}/transformer"
for public_module in data error_consumer error_public; do
  cp "${PROJECT_ROOT}/lib/transformer/${public_module}.esk" \
    "${D1_PUBLIC_ROOT}/transformer/${public_module}.esk"
done

D1_PUBLIC_NAMES=(
  token-corpus-write!
  token-corpus-validate
  token-corpus-summary-shard-count
  token-corpus-summary-total-tokens
  token-corpus-summary-vocab-size
  token-corpus-summary-shard-token-limit
  token-corpus-summary-tokenizer-fingerprint
  token-corpus-summary-total-shard-bytes
)
mapfile -t D1_DECLARED_NAMES < <(
  sed -n '/^(provide /,/)/p' "${PROJECT_ROOT}/lib/transformer/data.esk" |
    tr '()' '  ' | tr -s '[:space:]' '\n' |
    grep -v -e '^provide$' -e '^$'
)
[[ "${D1_DECLARED_NAMES[*]}" == "${D1_PUBLIC_NAMES[*]}" ]] || \
  die "transformer.data must provide exactly the eight accepted D1 names"

LC_ALL=C sort -cu "${D1_PUBLIC_EXPORTS}" || \
  die "D1 public native export manifest is not C-sorted unique text"
[[ "$(wc -l <"${D1_PUBLIC_EXPORTS}")" == 8 ]] || \
  die "D1 public native export manifest must contain exactly eight names"
cmp --silent "${D1_PUBLIC_EXPORTS}" \
  "${D1_EVIDENCE}/package-exports.txt" || \
  die "D1 artifact package exports differ from their repository manifest"

{
  printf '%s\n' \
    et_e1b_error_category_v1 \
    et_e1b_error_cause_v1 \
    et_e1b_error_details_v1 \
    et_e1b_error_message_v1 \
    et_e1b_error_operation_v1 \
    et_e1b_error_predicate_v1
  cat "${D1_PUBLIC_EXPORTS}"
} | LC_ALL=C sort -u >"${D1_TMP}/expected-global-defined.txt"
[[ "$(wc -l <"${D1_TMP}/expected-global-defined.txt")" == 14 ]] || \
  die "D1 expected global allowlist must contain six E1B plus eight D1 names"
cmp --silent "${D1_TMP}/expected-global-defined.txt" \
  "${D1_EVIDENCE}/global-defined.txt" || \
  die "D1 combined artifact differs from the exact 14-symbol global allowlist"
nm -g --defined-only --format=posix "${D1_LIBRARY}" |
  awk '$2 ~ /^[A-Za-z]$/ { print $1 }' | LC_ALL=C sort -u \
    >"${D1_TMP}/archive-global-defined.txt"
cmp --silent "${D1_TMP}/expected-global-defined.txt" \
  "${D1_TMP}/archive-global-defined.txt" || \
  die "nm found a D1 archive global-definition mismatch"
readelf --wide --symbols "${D1_LIBRARY}" |
  awk '$5 == "GLOBAL" && $7 != "UND" { print $8 }' | LC_ALL=C sort -u \
    >"${D1_TMP}/archive-readelf-global-defined.txt"
cmp --silent "${D1_TMP}/expected-global-defined.txt" \
  "${D1_TMP}/archive-readelf-global-defined.txt" || \
  die "readelf found a D1 archive global-definition mismatch"

LC_ALL=C sort -cu "${D1_UNDEFINED_SYMBOLS}" || \
  die "D1 undefined-symbol manifest is not C-sorted unique text"
cmp --silent "${D1_UNDEFINED_SYMBOLS}" \
  "${D1_EVIDENCE}/expected-undefined.txt" || \
  die "D1 artifact selected the wrong repository undefined-symbol manifest"
cmp --silent "${D1_UNDEFINED_SYMBOLS}" "${D1_EVIDENCE}/undefined.txt" || \
  die "D1 artifact differs from its exact undefined-symbol manifest"
grep -Fx $'allowlist\t'"$(basename -- "${D1_UNDEFINED_SYMBOLS}")" \
  "${D1_EVIDENCE}/allowlist-provenance.tsv" >/dev/null || \
  die "D1 artifact evidence does not identify its repository allowlist"
grep -Fx $'package_policy\td1' \
  "${D1_EVIDENCE}/allowlist-provenance.tsv" >/dev/null || \
  die "D1 artifact evidence does not identify the exact normal package route"

grep -F 'src/eshkol_transformer/token_shard.esk' \
  "${D1_EVIDENCE}/private.d" >/dev/null
grep -F 'lib/transformer/error_core.esk' \
  "${D1_EVIDENCE}/private.d" >/dev/null
grep -F 'lib/transformer/error_internal.esk' \
  "${D1_EVIDENCE}/private.d" >/dev/null
grep -F 'native/e1b_error_consumer_private.esk' \
  "${D1_EVIDENCE}/private.d" >/dev/null

d1_require_local_symbol() {
  local symbol=$1
  grep -E "[[:space:]]LOCAL[[:space:]].*[[:space:]]${symbol}$" \
    "${D1_EVIDENCE}/readelf-symbols.txt" >/dev/null || \
    die "D1 required private symbol is not local: ${symbol}"
}

D1_REQUIRED_LOCAL_SYMBOLS=(
  e1-internal-dispatch
  et-e1b-private-raise__eshkol_internal_abi
  transformer-error-make
  transformer-error-raise
  transformer-error-wrap-foreign
  et_e1b_private_raise_cabi_v1
  et_e1b_consumer_raise_v1
  et_e1b_box_value_v1
  et_e1b_ensure_private_initialized_v1
  et_d1_checked_write_new_v1
  et_e1b_private_d1_token_corpus_write_cabi_v1
  et_e1b_private_d1_token_corpus_validate_cabi_v1
)
while IFS= read -r private_cabi; do
  D1_REQUIRED_LOCAL_SYMBOLS+=("${private_cabi}")
done < <(awk '{ print $2 }' "${PROJECT_ROOT}/native/d1_e1b_private_renames.txt")
for private_symbol in "${D1_REQUIRED_LOCAL_SYMBOLS[@]}"; do
  d1_require_local_symbol "${private_symbol}"
done
for private_text in et-e1b-private-raise e1-internal-dispatch \
    transformer-error-make transformer-error-raise \
    transformer-error-wrap-foreign et_e1b_private_raise_cabi_v1 \
    et_d1_checked_write_new_v1; do
  grep -F "${private_text}" "${D1_EVIDENCE}/strings.txt" >/dev/null || \
    die "D1 artifact strings evidence omitted private name ${private_text}"
done

d1_compile() {
  local source=$1 output=$2
  local cache_home=${D1_CACHE_HOME:-${D1_TMP}/cache}
  local library_dir=${D1_LIBRARY_DIR:-${D1_ARTIFACT_DIR}}
  local library_name=${D1_LIBRARY_NAME:-eshkol_transformer_d1}
  local include_root=${D1_INCLUDE_ROOT:-${D1_PUBLIC_ROOT}}
  local source_root=${D1_SOURCE_ROOT:-}
  local include_args=(-I "${include_root}")
  if [[ -n "${source_root}" ]]; then
    include_args+=(-I "${source_root}")
  fi
  shift 2
  mkdir -p "${cache_home}"
  (
    cd "${include_root}"
    "${D1_TIMEOUT}" --foreground --signal=TERM --kill-after=5s "${D1_TIMEOUT_SECONDS}s" \
      env XDG_CACHE_HOME="${cache_home}" ESHKOL_CXX_COMPILER="${D1_CXX}" \
      ESHKOL_LIB_DIR="${include_root}" \
      "${D1_RUNNER}" \
      --strict-types --no-stdlib \
      "${include_args[@]}" \
      -L "${library_dir}" --lib "${library_name}" \
      "$@" "${source}" -o "${output}"
  )
}

d1_run_source() {
  local source=$1
  local cache_home=${D1_CACHE_HOME:-${D1_TMP}/cache}
  local library_dir=${D1_LIBRARY_DIR:-${D1_ARTIFACT_DIR}}
  local library_name=${D1_LIBRARY_NAME:-eshkol_transformer_d1}
  local include_root=${D1_INCLUDE_ROOT:-${D1_PUBLIC_ROOT}}
  local source_root=${D1_SOURCE_ROOT:-}
  local include_args=(-I "${include_root}")
  if [[ -n "${source_root}" ]]; then
    include_args+=(-I "${source_root}")
  fi
  mkdir -p "${cache_home}"
  (
    cd "${include_root}"
    "${D1_TIMEOUT}" --foreground --signal=TERM --kill-after=5s "${D1_TIMEOUT_SECONDS}s" \
      env XDG_CACHE_HOME="${cache_home}" ESHKOL_CXX_COMPILER="${D1_CXX}" \
      ESHKOL_LIB_DIR="${include_root}" \
      "${D1_RUNNER}" \
      --strict-types --no-stdlib \
      "${include_args[@]}" \
      -L "${library_dir}" --lib "${library_name}" \
      -r "${source}"
  )
}

d1_assert_private_module_unavailable() {
  local form=$1 source_tag=$2 mode run output log source
  source="${D1_TMP}/negative-module-${source_tag}.esk"
  printf '%s\n' "${form}" >"${source}"
  for mode in source object aot; do
    for run in 1; do
      output="${D1_TMP}/negative-module-${source_tag}-${mode}-${run}"
      log="${output}.log"
      if [[ "${mode}" == source ]]; then
        if D1_CACHE_HOME="${output}-cache" \
            d1_run_source "${source}" >"${log}" 2>&1; then
          die "D1 private module ${source_tag} unexpectedly loaded from public source"
        fi
      elif [[ "${mode}" == object ]]; then
        if D1_CACHE_HOME="${output}-cache" \
            d1_compile "${source}" "${output}.o" --compile-only \
              >"${log}" 2>&1; then
          die "D1 private module ${source_tag} unexpectedly compiled publicly"
        fi
        test ! -e "${output}.o"
      else
        if D1_CACHE_HOME="${output}-cache" \
            d1_compile "${source}" "${output}.bin" >"${log}" 2>&1; then
          die "D1 private module ${source_tag} unexpectedly linked publicly"
        fi
        test ! -e "${output}.bin"
      fi
      test -s "${log}"
    done
  done
}

d1_assert_private_module_unavailable \
  '(require transformer.error_core)' error-core
d1_assert_private_module_unavailable \
  '(require transformer.error_internal)' error-internal
d1_assert_private_module_unavailable \
  '(load "eshkol_transformer/token_shard.esk")' token-shard
d1_assert_private_module_unavailable \
  '(load "d1_e1b_private.esk")' trusted-root

d1_assert_private_binding() {
  local hidden_name=$1 hidden_source=$2 mode run output log depfile
  local source="${D1_TMP}/negative-private-${hidden_source}.esk"
  printf '(require transformer.data)\n\n(define leaked-private-binding %s)\n' \
    "${hidden_name}" >"${source}"
  for mode in source object aot; do
    for run in 1; do
      output="${D1_TMP}/negative-private-${hidden_source}-${mode}-${run}"
      log="${output}.log"
      depfile="${output}.d"
      if [[ "${mode}" == source ]]; then
        if D1_CACHE_HOME="${D1_TMP}/negative-private-${hidden_source}-${mode}-cache-${run}" \
            d1_run_source "${source}" >"${log}" 2>&1; then
          die "D1 private ${hidden_name} unexpectedly ran from public source"
        fi
      elif [[ "${mode}" == object ]]; then
        output="${output}.o"
        if D1_CACHE_HOME="${D1_TMP}/negative-private-${hidden_source}-${mode}-cache-${run}" \
            d1_compile "${source}" "${output}" --emit-depfile "${depfile}" \
              --compile-only >"${log}" 2>&1; then
          die "D1 private ${hidden_name} unexpectedly compiled to an object"
        fi
      else
        output="${output}.bin"
        if D1_CACHE_HOME="${D1_TMP}/negative-private-${hidden_source}-${mode}-cache-${run}" \
            d1_compile "${source}" "${output}" >"${log}" 2>&1; then
          die "D1 private ${hidden_name} unexpectedly AOT-linked"
        fi
      fi
      grep -F "Unbound variable: ${hidden_name}" "${log}" >/dev/null
      test ! -e "${output}"
    done
  done
}

d1_assert_unavailable_call() {
  local hidden_name=$1 hidden_source=$2 mode output log source
  source="${D1_TMP}/negative-direct-${hidden_source}.esk"
  printf '(require transformer.data)\n\n(%s)\n' "${hidden_name}" >"${source}"
  for mode in source object aot; do
    output="${D1_TMP}/negative-direct-${hidden_source}-${mode}"
    log="${output}.log"
    if [[ "${mode}" == source ]]; then
      if D1_CACHE_HOME="${output}-cache" \
          d1_run_source "${source}" >"${log}" 2>&1; then
        die "D1 private ${hidden_name} unexpectedly ran as a direct source call"
      fi
    elif [[ "${mode}" == object ]]; then
      if D1_CACHE_HOME="${output}-cache" \
          d1_compile "${source}" "${output}.o" --compile-only >"${log}" 2>&1; then
        die "D1 private ${hidden_name} unexpectedly compiled as a direct call"
      fi
      test ! -e "${output}.o"
    else
      if D1_CACHE_HOME="${output}-cache" \
          d1_compile "${source}" "${output}.bin" >"${log}" 2>&1; then
        die "D1 private ${hidden_name} unexpectedly linked as a direct call"
      fi
      test ! -e "${output}.bin"
    fi
    grep -F "Unknown function: ${hidden_name}" "${log}" >/dev/null
  done
}

d1_assert_public_wrong_arity() {
  local public_name=$1 source_tag=$2 mode output log source
  source="${D1_TMP}/negative-arity-${source_tag}.esk"
  printf '(require transformer.data)\n\n(%s)\n' "${public_name}" >"${source}"
  for mode in source object aot; do
    output="${D1_TMP}/negative-arity-${source_tag}-${mode}"
    log="${output}.log"
    if [[ "${mode}" == source ]]; then
      if D1_CACHE_HOME="${output}-cache" \
          d1_run_source "${source}" >"${log}" 2>&1; then
        die "D1 public ${public_name} accepted a wrong-arity source call"
      fi
    elif [[ "${mode}" == object ]]; then
      if D1_CACHE_HOME="${output}-cache" \
          d1_compile "${source}" "${output}.o" --compile-only >"${log}" 2>&1; then
        die "D1 public ${public_name} accepted a wrong-arity object call"
      fi
      test ! -e "${output}.o"
    else
      if D1_CACHE_HOME="${output}-cache" \
          d1_compile "${source}" "${output}.bin" >"${log}" 2>&1; then
        die "D1 public ${public_name} accepted a wrong-arity AOT call"
      fi
      test ! -e "${output}.bin"
    fi
    grep -F "Arity mismatch: ${public_name}" "${log}" >/dev/null
  done
}

D1_FAULT_ARTIFACT_DIR="${D1_TMP}/fault-artifacts"
E1B_PACKAGE_POLICY=d1 \
  ESHKOL_PATH="${D1_ADMISSION_FIXTURES}/shadow" \
  ESHKOL_LIB_DIR="${D1_ADMISSION_FIXTURES}/shadow" \
  /usr/bin/bash "${PROJECT_ROOT}/scripts/build-d1.sh" \
  "${D1_FAULT_ARTIFACT_DIR}" test-faults \
  >"${D1_TMP}/fault-build.stdout" 2>"${D1_TMP}/fault-build.stderr"
D1_FAULT_LIBRARY="${D1_FAULT_ARTIFACT_DIR}/libeshkol_transformer_d1_faults.a"
D1_FAULT_EVIDENCE="${D1_FAULT_LIBRARY}.evidence"
D1_FAULT_UNDEFINED_SYMBOLS="${PROJECT_ROOT}/native/d1_e1b_fault_undefined_symbols.txt"
[[ -r "${D1_FAULT_LIBRARY}" ]] || die "D1 fault-test archive is missing"
[[ "$(ar t "${D1_FAULT_LIBRARY}")" == "stdlib.o" ]] || \
  die "D1 fault-test archive must contain exactly stdlib.o"
LC_ALL=C sort -cu "${D1_FAULT_UNDEFINED_SYMBOLS}" || \
  die "D1 fault undefined-symbol manifest is not C-sorted unique text"
cmp --silent "${D1_TMP}/expected-global-defined.txt" \
  "${D1_FAULT_EVIDENCE}/global-defined.txt" || \
  die "D1 fault artifact differs from the exact 14-symbol global allowlist"
cmp --silent "${D1_PUBLIC_EXPORTS}" \
  "${D1_FAULT_EVIDENCE}/package-exports.txt" || \
  die "D1 fault artifact package exports differ from their repository manifest"
cmp --silent "${D1_FAULT_UNDEFINED_SYMBOLS}" \
  "${D1_FAULT_EVIDENCE}/expected-undefined.txt" || \
  die "D1 fault artifact selected the wrong undefined-symbol manifest"
cmp --silent "${D1_FAULT_UNDEFINED_SYMBOLS}" \
  "${D1_FAULT_EVIDENCE}/undefined.txt" || \
  die "D1 fault artifact differs from its exact undefined-symbol manifest"
grep -Fx $'allowlist\t'"$(basename -- "${D1_FAULT_UNDEFINED_SYMBOLS}")" \
  "${D1_FAULT_EVIDENCE}/allowlist-provenance.tsv" >/dev/null || \
  die "D1 fault artifact evidence does not identify its repository allowlist"
grep -Fx $'package_policy\td1-test-faults' \
  "${D1_FAULT_EVIDENCE}/allowlist-provenance.tsv" >/dev/null || \
  die "D1 fault artifact evidence does not identify the exact test route"
grep -F 'native/d1_e1b_private.esk' \
  "${D1_FAULT_EVIDENCE}/private.d" >/dev/null || \
  die "D1 fault route omitted the canonical private root"
grep -F 'src/eshkol_transformer/token_shard.esk' \
  "${D1_FAULT_EVIDENCE}/private.d" >/dev/null || \
  die "D1 fault route omitted the canonical token-shard source"
if grep -F "${D1_ADMISSION_FIXTURES}/shadow" \
    "${D1_FAULT_EVIDENCE}/private.d" >/dev/null; then
  die "D1 fault route admitted a caller-controlled ESHKOL_PATH module"
fi

D1_OVERRIDE_ARTIFACT_DIR="${D1_TMP}/inherited-policy-normal"
E1B_PACKAGE_POLICY=d1-test-faults \
  ESHKOL_PATH="${D1_ADMISSION_FIXTURES}/shadow" \
  ESHKOL_LIB_DIR="${D1_ADMISSION_FIXTURES}/shadow" \
  /usr/bin/bash "${PROJECT_ROOT}/scripts/build-d1.sh" \
    "${D1_OVERRIDE_ARTIFACT_DIR}" normal \
    >"${D1_TMP}/inherited-policy.stdout" \
    2>"${D1_TMP}/inherited-policy.stderr"
D1_OVERRIDE_LIBRARY="${D1_OVERRIDE_ARTIFACT_DIR}/libeshkol_transformer_d1.a"
cmp --silent "${D1_UNDEFINED_SYMBOLS}" \
  "${D1_OVERRIDE_LIBRARY}.evidence/expected-undefined.txt" || \
  die "inherited policy changed the canonical D1 wrapper route"
grep -Fx $'package_policy\td1' \
  "${D1_OVERRIDE_LIBRARY}.evidence/allowlist-provenance.tsv" >/dev/null || \
  die "normal D1 wrapper did not suppress an inherited package-policy override"
grep -F 'src/eshkol_transformer/token_shard.esk' \
  "${D1_OVERRIDE_LIBRARY}.evidence/private.d" >/dev/null || \
  die "normal D1 route omitted the canonical token-shard source"
if grep -F "${D1_ADMISSION_FIXTURES}/shadow" \
    "${D1_OVERRIDE_LIBRARY}.evidence/private.d" >/dev/null; then
  die "normal D1 route admitted a caller-controlled ESHKOL_PATH module"
fi

D1_PRODUCTION_SNAPSHOT="${D1_TMP}/production-before-fault-route"
mkdir -p "${D1_PRODUCTION_SNAPSHOT}"
cp "${D1_LIBRARY}" "${D1_PRODUCTION_SNAPSHOT}/library"
cp -a "${D1_EVIDENCE}" "${D1_PRODUCTION_SNAPSHOT}/evidence"
if /usr/bin/bash "${PROJECT_ROOT}/scripts/build-d1.sh" \
    "$(project_build_dir)/d1" test-faults \
    >"${D1_TMP}/production-fault-route.stdout" \
    2>"${D1_TMP}/production-fault-route.stderr"; then
  die "D1 test-fault route unexpectedly targeted the production directory"
fi
grep -F 'D1 test-fault artifact cannot target the canonical production directory' \
  "${D1_TMP}/production-fault-route.stderr" >/dev/null
cmp --silent "${D1_PRODUCTION_SNAPSHOT}/library" "${D1_LIBRARY}" || \
  die "rejected test-fault route mutated the production D1 archive"
diff -r "${D1_PRODUCTION_SNAPSHOT}/evidence" "${D1_EVIDENCE}" >/dev/null || \
  die "rejected test-fault route mutated the production D1 evidence"
for forbidden_fault_text in ET_D1_TEST_FAULT ET_D1_TEST_FAIL_CALL \
    short-write write-enospc write-eio close-eio; do
  if grep -F "${forbidden_fault_text}" \
      "${D1_EVIDENCE}/strings.txt" >/dev/null; then
    die "canonical D1 combined artifact contains test-fault control: ${forbidden_fault_text}"
  fi
done
if grep -E '^(getenv|strtoull)$' "${D1_EVIDENCE}/undefined.txt" >/dev/null; then
  die "canonical D1 combined artifact references test-fault runtime helpers"
fi

D1_GUESSED_NATIVE_SYMBOLS=(
  d1_token_write_temporary_new
  et_d1_token_write_temporary_new_v1
  d1_token_publish_temporary
  et_d1_token_publish_temporary_v1
  d1_token_atomic_write_new
  et_d1_token_atomic_write_new_v1
  d1_token_corpus_write_impl
  et_d1_token_corpus_write_impl_v1
  d1_token_corpus_validate_impl
  et_d1_token_corpus_validate_impl_v1
  d1_token_list_ref
  et_d1_token_list_ref_v1
  et_d1_checked_write_new_v1
  et_e1b_private_raise_cabi_v1
  et_e1b_consumer_raise_v1
  et_e1b_box_value_v1
  et_e1b_ensure_private_initialized_v1
)
while IFS= read -r private_cabi; do
  D1_GUESSED_NATIVE_SYMBOLS+=("${private_cabi}")
done < <(awk '{ print $2 }' "${PROJECT_ROOT}/native/d1_e1b_private_renames.txt")
for guessed_symbol in "${D1_GUESSED_NATIVE_SYMBOLS[@]}"; do
  guessed_object="${D1_TMP}/negative-native-${guessed_symbol}.o"
  guessed_combined="${D1_TMP}/negative-native-${guessed_symbol}-combined.o"
  "${D1_CC}" -std=c11 -Wall -Wextra -Werror -Wpedantic \
    -DD1_PRIVATE_SYMBOL="${guessed_symbol}" -c \
    "${PROJECT_ROOT}/tests/d1/negative_native_symbol_link.c" \
    -o "${guessed_object}"
  "${D1_CC}" -r "${guessed_object}" -Wl,--whole-archive "${D1_LIBRARY}" \
    -Wl,--no-whole-archive -o "${guessed_combined}"
  nm -u "${guessed_combined}" | awk '{ print $NF }' |
    grep -Fx "${guessed_symbol}" >/dev/null || \
    die "guessed private native symbol resolved from D1 artifact: ${guessed_symbol}"
done

for run in 1 2; do
  D1_CACHE_HOME="${D1_TMP}/data-api-${run}-cache" \
    d1_compile "${PROJECT_ROOT}/tests/d1/compile_data_api.esk" \
    "${D1_TMP}/data-api-${run}.o" \
    --emit-depfile "${D1_TMP}/data-api-${run}.d" --compile-only \
    >"${D1_TMP}/data-api-${run}.stdout" 2>"${D1_TMP}/data-api-${run}.stderr"
  test -s "${D1_TMP}/data-api-${run}.o"
  grep -F "${D1_PUBLIC_ROOT}/transformer/data.esk" \
    "${D1_TMP}/data-api-${run}.d" >/dev/null
  grep -F "${D1_PUBLIC_ROOT}/transformer/error_consumer.esk" \
    "${D1_TMP}/data-api-${run}.d" >/dev/null
  for private_dependency in \
      src/eshkol_transformer/token_shard.esk \
      src/eshkol_transformer/sha256.esk \
      lib/transformer/error_core.esk \
      lib/transformer/error_internal.esk \
      native/e1b_error_consumer_private.esk \
      native/d1_e1b_private.esk; do
    if grep -F "${private_dependency}" \
        "${D1_TMP}/data-api-${run}.d" >/dev/null; then
      die "D1 public data graph unexpectedly includes ${private_dependency}"
    fi
  done
done
cmp --silent "${D1_TMP}/data-api-1.o" "${D1_TMP}/data-api-2.o"
cmp --silent "${D1_TMP}/data-api-1.stdout" "${D1_TMP}/data-api-2.stdout"
for compile_log in "${D1_TMP}"/data-api-*.stdout "${D1_TMP}"/data-api-*.stderr; do
  if grep -F 'ERROR:' "${compile_log}" >/dev/null; then
    die "D1 compile-only gate emitted ERROR in ${compile_log}"
  fi
done
if nm -g --defined-only --format=posix "${D1_TMP}/data-api-1.o" |
    awk '{ print $1 }' | grep -E '^(d1-(token|sha)|eshkol_g_d1_2D)' \
      >/dev/null; then
  die "nm found a globally defined private D1 binding in the public consumer object"
fi
if readelf --wide --symbols "${D1_TMP}/data-api-1.o" |
    awk '$5 == "GLOBAL" && $7 != "UND" { print $8 }' |
    grep -E '^(d1-(token|sha)|eshkol_g_d1_2D)' >/dev/null; then
  die "readelf found a globally defined private D1 binding in the public consumer object"
fi
if strings -a "${D1_TMP}/data-api-1.o" |
    grep -E '(d1-token-(write-temporary-new!|publish-temporary!|atomic-write-new!|corpus-(write|validate)-impl|list-(length|ref))(_sexpr|__eshkol)|eshkol_g_d1_2D|et-e1b-private-raise|e1-internal-dispatch|transformer-error-(make|raise|wrap-foreign))' \
      >/dev/null; then
  die "public consumer object contains a compiled private D1 symbol name"
fi

D1_GUESSED_ESHKOL_SYMBOLS=(
  d1-token-write-temporary-new!
  d1-token-publish-temporary!
  d1-token-atomic-write-new!
  d1-token-corpus-write-impl
  d1-token-corpus-validate-impl
  d1-token-list-ref
  d1-sha256
  d1-token-checked-write-new
  et-e1b-private-raise
  e1-internal-dispatch
  transformer-error-make
  transformer-error-raise
  transformer-error-wrap-foreign
  et-d1-private-token-corpus-write
  et-d1-private-token-corpus-validate
  et-d1-private-token-corpus-summary-shard-count
  d1-token-write-temporary-new!_sexpr
  eshkol_g_d1_2Dtoken_2Dsummary_2Dpublic_2Doperations
  d1-compiled-public-operations
  eshkol_g_d1_2Dcompiled_2Dpublic_2Doperations
)
guessed_index=0
for guessed_symbol in "${D1_GUESSED_ESHKOL_SYMBOLS[@]}"; do
  guessed_index=$((guessed_index + 1))
  guessed_object="${D1_TMP}/negative-eshkol-link-${guessed_index}.o"
  guessed_combined="${D1_TMP}/negative-eshkol-link-${guessed_index}-combined.o"
  "${D1_CC}" -std=c11 -Wall -Wextra -Werror -Wpedantic \
    -DD1_RELOCATABLE_PROBE \
    -DD1_PRIVATE_LINK_NAME="\"${guessed_symbol}\"" -c \
    "${PROJECT_ROOT}/tests/d1/negative_native_symbol_link.c" \
    -o "${guessed_object}"
  "${D1_CC}" -r "${guessed_object}" "${D1_TMP}/data-api-1.o" \
    -Wl,--whole-archive "${D1_LIBRARY}" -Wl,--no-whole-archive \
    -o "${guessed_combined}"
  nm -u "${guessed_combined}" | awk '{ print $NF }' |
    grep -Fx "${guessed_symbol}" >/dev/null || \
    die "guessed private Eshkol symbol resolved from public object: ${guessed_symbol}"
done

D1_PRIVATE_BINDING_NAMES=(
  d1-token-write-temporary-new!
  d1-token-publish-temporary!
  d1-token-atomic-write-new!
  d1-token-corpus-write-impl
  d1-token-corpus-validate-impl
  d1-token-list-ref
  d1-token-derived-sizes
  d1-token-reraise-or-wrap
  d1-token-summary-public-operations
  d1-compiled-public-operations
  d1-sha256
  d1-token-checked-write-new
  et-e1b-private-raise
  e1-internal-dispatch
  transformer-error-make
  transformer-error-raise
  transformer-error-wrap-foreign
  et-d1-private-token-corpus-write
  et-d1-private-token-corpus-validate
  et-d1-private-token-corpus-summary-shard-count
)
D1_PRIVATE_BINDING_SOURCES=(
  write-temporary
  publish-temporary
  atomic-write
  corpus-write-impl
  corpus-validate-impl
  list-ref
  derived-sizes
  error-mapping
  summary-operations
  compiled-operations
  sha256
  checked-write-ffi
  private-raise
  internal-dispatch
  error-make
  error-raise
  error-wrap
  private-write-cabi
  private-validate-cabi
  private-summary-accessor-cabi
)
for hidden_index in "${!D1_PRIVATE_BINDING_NAMES[@]}"; do
  d1_assert_private_binding \
    "${D1_PRIVATE_BINDING_NAMES[${hidden_index}]}" \
    "${D1_PRIVATE_BINDING_SOURCES[${hidden_index}]}"
done

D1_DIRECT_PRIVATE_NAMES=(
  d1-token-checked-write-new
  d1-token-write-temporary-new!
  d1-token-publish-temporary!
  d1-token-atomic-write-new!
  d1-token-corpus-write-impl
  d1-token-corpus-validate-impl
  d1-token-list-ref
  d1-sha256
  et-e1b-private-raise
  e1-internal-dispatch
  transformer-error-make
  transformer-error-raise
  transformer-error-wrap-foreign
  et-d1-private-token-corpus-write
  et-d1-private-token-corpus-validate
  et-d1-private-token-corpus-summary-shard-count
)
D1_DIRECT_PRIVATE_SOURCES=(
  checked-write-ffi
  write-temporary
  publish-temporary
  atomic-write
  corpus-write-impl
  corpus-validate-impl
  list-ref
  sha256
  private-raise
  internal-dispatch
  error-make
  error-raise
  error-wrap
  private-write-cabi
  private-validate-cabi
  private-summary-accessor-cabi
)
for hidden_index in "${!D1_DIRECT_PRIVATE_NAMES[@]}"; do
  d1_assert_unavailable_call \
    "${D1_DIRECT_PRIVATE_NAMES[${hidden_index}]}" \
    "${D1_DIRECT_PRIVATE_SOURCES[${hidden_index}]}"
done

d1_assert_public_wrong_arity token-corpus-write! writer
d1_assert_public_wrong_arity token-corpus-validate validator
d1_assert_public_wrong_arity token-corpus-summary-shard-count accessor

hidden_error_names=(
  transformer-error-make
  transformer-error-raise
  transformer-error-wrap-foreign
)
hidden_error_sources=(make raise wrap)
for hidden_index in "${!hidden_error_names[@]}"; do
  hidden_name="${hidden_error_names[${hidden_index}]}"
  hidden_source="${hidden_error_sources[${hidden_index}]}"
  for run in 1 2; do
    if D1_CACHE_HOME="${D1_TMP}/negative-error-${hidden_source}-cache-${run}" \
        d1_compile \
        "${PROJECT_ROOT}/tests/d1/negative_error_${hidden_source}_public.esk" \
        "${D1_TMP}/negative-error-${hidden_source}-${run}.o" \
        --compile-only \
        >"${D1_TMP}/negative-error-${hidden_source}-${run}.log" 2>&1; then
      die "D1 hidden E1 ${hidden_source} helper unexpectedly compiled publicly"
    fi
    grep -F "Unbound variable: ${hidden_name}" \
      "${D1_TMP}/negative-error-${hidden_source}-${run}.log" >/dev/null
    test ! -e "${D1_TMP}/negative-error-${hidden_source}-${run}.o"
  done
  cmp --silent "${D1_TMP}/negative-error-${hidden_source}-1.log" \
    "${D1_TMP}/negative-error-${hidden_source}-2.log"
done

hidden_summary_names=(d1-token-summary-create d1-token-make-summary)
hidden_summary_sources=(constructor maker)
for hidden_index in "${!hidden_summary_names[@]}"; do
  hidden_name="${hidden_summary_names[${hidden_index}]}"
  hidden_source="${hidden_summary_sources[${hidden_index}]}"
  for run in 1 2; do
    if D1_CACHE_HOME="${D1_TMP}/negative-summary-${hidden_source}-cache-${run}" \
        d1_compile \
        "${PROJECT_ROOT}/tests/d1/negative_summary_${hidden_source}_public.esk" \
        "${D1_TMP}/negative-summary-${hidden_source}-${run}.o" \
        --compile-only \
        >"${D1_TMP}/negative-summary-${hidden_source}-${run}.log" 2>&1; then
      die "D1 hidden summary ${hidden_source} unexpectedly compiled publicly"
    fi
    grep -F "Unbound variable: ${hidden_name}" \
      "${D1_TMP}/negative-summary-${hidden_source}-${run}.log" >/dev/null
    test ! -e "${D1_TMP}/negative-summary-${hidden_source}-${run}.o"
  done
  cmp --silent "${D1_TMP}/negative-summary-${hidden_source}-1.log" \
    "${D1_TMP}/negative-summary-${hidden_source}-2.log"
done

D1_INCLUDE_ROOT="${PROJECT_ROOT}/lib" D1_SOURCE_ROOT="${PROJECT_ROOT}/src" \
  d1_compile "${PROJECT_ROOT}/tests/d1/sha256_probe.esk" \
  "${D1_TMP}/sha256-probe" \
  >"${D1_TMP}/sha256.compile.stdout" 2>"${D1_TMP}/sha256.compile.stderr"
D1_INCLUDE_ROOT="${PROJECT_ROOT}/lib" D1_SOURCE_ROOT="${PROJECT_ROOT}/src" \
  d1_compile "${PROJECT_ROOT}/tests/d1/primitive_probe.esk" \
  "${D1_TMP}/primitive-probe" \
  >"${D1_TMP}/primitive.compile.stdout" 2>"${D1_TMP}/primitive.compile.stderr"
d1_compile "${PROJECT_ROOT}/tests/d1/corpus_tool.esk" \
  "${D1_TMP}/corpus-tool" \
  --emit-depfile "${D1_TMP}/corpus-tool.d" \
  >"${D1_TMP}/corpus.compile.stdout" 2>"${D1_TMP}/corpus.compile.stderr"
D1_LIBRARY_DIR="${D1_FAULT_ARTIFACT_DIR}" \
D1_LIBRARY_NAME=eshkol_transformer_d1_faults \
D1_CACHE_HOME="${D1_TMP}/fault-tool-cache" \
  d1_compile "${PROJECT_ROOT}/tests/d1/corpus_tool.esk" \
  "${D1_TMP}/corpus-tool-faults" \
  >"${D1_TMP}/corpus-faults.compile.stdout" \
  2>"${D1_TMP}/corpus-faults.compile.stderr"
d1_compile "${PROJECT_ROOT}/tests/d1/arithmetic_probe.esk" \
  "${D1_TMP}/arithmetic-probe" \
  >"${D1_TMP}/arithmetic.compile.stdout" 2>"${D1_TMP}/arithmetic.compile.stderr"
d1_compile "${PROJECT_ROOT}/tests/d1/e1_mapping_probe.esk" \
  "${D1_TMP}/e1-mapping-probe" \
  >"${D1_TMP}/e1-mapping.compile.stdout" 2>"${D1_TMP}/e1-mapping.compile.stderr"
d1_compile "${PROJECT_ROOT}/tests/d1/summary_opacity_probe.esk" \
  "${D1_TMP}/summary-opacity-probe" \
  >"${D1_TMP}/summary-opacity.compile.stdout" \
  2>"${D1_TMP}/summary-opacity.compile.stderr"
for import_order in data_first error_first; do
  D1_CACHE_HOME="${D1_TMP}/e1b-${import_order}-cache" \
    d1_compile "${PROJECT_ROOT}/tests/d1/e1b_import_${import_order}.esk" \
      "${D1_TMP}/e1b-${import_order}-probe" \
      >"${D1_TMP}/e1b-${import_order}.compile.stdout" \
      2>"${D1_TMP}/e1b-${import_order}.compile.stderr"
done

for compile_log in "${D1_TMP}"/*.compile.stdout "${D1_TMP}"/*.compile.stderr; do
  if grep -F 'ERROR:' "${compile_log}" >/dev/null; then
    die "D1 compiler emitted ERROR in ${compile_log}"
  fi
done

mkdir "${D1_TMP}/primitive-root"
PATH=/definitely-not-a-real-path \
  "${D1_TIMEOUT}" --foreground --signal=TERM --kill-after=5s "${D1_TIMEOUT_SECONDS}s" \
  "${D1_TMP}/sha256-probe" >"${D1_TMP}/sha256.stdout" 2>"${D1_TMP}/sha256.stderr"
grep -Fx 'D1_SHA256_PASS' "${D1_TMP}/sha256.stdout" >/dev/null
test ! -s "${D1_TMP}/sha256.stderr"

PATH=/definitely-not-a-real-path \
  "${D1_TIMEOUT}" --foreground --signal=TERM --kill-after=5s "${D1_TIMEOUT_SECONDS}s" \
  "${D1_TMP}/primitive-probe" "${D1_TMP}/primitive-root" \
  >"${D1_TMP}/primitive.stdout" 2>"${D1_TMP}/primitive.stderr"
grep -Fx 'D1_PRIMITIVES_PASS' "${D1_TMP}/primitive.stdout" >/dev/null
test ! -s "${D1_TMP}/primitive.stderr"

PATH=/definitely-not-a-real-path \
  "${D1_TIMEOUT}" --foreground --signal=TERM --kill-after=5s "${D1_TIMEOUT_SECONDS}s" \
  "${D1_TMP}/arithmetic-probe" >"${D1_TMP}/arithmetic.stdout" \
  2>"${D1_TMP}/arithmetic.stderr"
grep -Fx 'D1_ARITHMETIC_PASS' "${D1_TMP}/arithmetic.stdout" >/dev/null
test ! -s "${D1_TMP}/arithmetic.stderr"

PATH=/definitely-not-a-real-path \
  "${D1_TIMEOUT}" --foreground --signal=TERM --kill-after=5s "${D1_TIMEOUT_SECONDS}s" \
  "${D1_TMP}/e1-mapping-probe" "${D1_TMP}/missing-corpus" \
  >"${D1_TMP}/e1-mapping.stdout" \
  2>"${D1_TMP}/e1-mapping.stderr"
grep -Fx 'D1_E1_MAPPING_PASS' "${D1_TMP}/e1-mapping.stdout" >/dev/null
test ! -s "${D1_TMP}/e1-mapping.stderr"

for import_order in data_first error_first; do
  PATH=/definitely-not-a-real-path \
    "${D1_TIMEOUT}" --foreground --signal=TERM --kill-after=5s "${D1_TIMEOUT_SECONDS}s" \
    "${D1_TMP}/e1b-${import_order}-probe" "${D1_TMP}/missing-${import_order}" \
    >"${D1_TMP}/e1b-${import_order}.stdout" \
    2>"${D1_TMP}/e1b-${import_order}.stderr"
  grep -Fx "D1_E1B_$(printf '%s' "${import_order}" | tr '[:lower:]' '[:upper:]')_PASS" \
    "${D1_TMP}/e1b-${import_order}.stdout" >/dev/null
  test ! -s "${D1_TMP}/e1b-${import_order}.stderr"
done

mkdir "${D1_TMP}/summary-writer" "${D1_TMP}/summary-validator"
PATH=/definitely-not-a-real-path \
  "${D1_TIMEOUT}" --foreground --signal=TERM --kill-after=5s "${D1_TIMEOUT_SECONDS}s" \
  "${D1_TMP}/summary-opacity-probe" \
  "${D1_TMP}/summary-writer" "${D1_TMP}/summary-validator" \
  >"${D1_TMP}/summary-opacity.stdout" \
  2>"${D1_TMP}/summary-opacity.stderr"
grep -Fx 'D1_SUMMARY_OPACITY_PASS' "${D1_TMP}/summary-opacity.stdout" >/dev/null
test ! -s "${D1_TMP}/summary-opacity.stderr"

PYTHONDONTWRITEBYTECODE=1 D1_CORPUS_TOOL="${D1_TMP}/corpus-tool" \
  D1_FAULT_CORPUS_TOOL="${D1_TMP}/corpus-tool-faults" \
  "${D1_PYTHON}" -m unittest discover -s "${PROJECT_ROOT}/tests/d1" \
  -p 'test_*.py' -v

if find "${PROJECT_ROOT}/src" -type f \( -name '*.py' -o -name '*.pyc' \) -print -quit | \
    grep -q .; then
  die "D1 production source contains Python"
fi

printf 'D1 PASS: deterministic native writer, strict validator, primitive probes, and format tests\n'
