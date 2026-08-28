#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK_FILE="${PROJECT_ROOT}/toolchain/eshkol.lock"

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

lock_value() {
  local key="$1" value count
  count="$(awk -F '\t' -v key="${key}" '$1 == key { count++ } END { print count + 0 }' "${LOCK_FILE}")"
  [[ "${count}" == 1 ]] || die "expected exactly one '${key}' entry in ${LOCK_FILE}; found ${count}"
  value="$(awk -F '\t' -v key="${key}" '$1 == key { print $2 }' "${LOCK_FILE}")"
  [[ -n "${value}" ]] || die "empty '${key}' entry in ${LOCK_FILE}"
  printf '%s\n' "${value}"
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

check_supported_version() {
  local name="$1" actual="$2" expected="$3"
  if [[ "${actual}" != "${expected}" ]]; then
    [[ "${ESHKOL_ALLOW_UNSUPPORTED_HOST:-0}" == 1 ]] || \
      die "unsupported ${name} version: expected ${expected}, got ${actual}"
    printf 'warning: unsupported %s compatibility probe with %s (supported: %s)\n' \
      "${name}" "${actual}" "${expected}" >&2
  fi
}

tsv_value() {
  local file="$1" key="$2" value count
  count="$(awk -F '\t' -v key="${key}" '$1 == key { count++ } END { print count + 0 }' "${file}")"
  [[ "${count}" == 1 ]] || die "expected exactly one '${key}' entry in ${file}; found ${count}"
  value="$(awk -F '\t' -v key="${key}" '$1 == key { print $2 }' "${file}")"
  [[ -n "${value}" ]] || die "empty '${key}' entry in ${file}"
  printf '%s\n' "${value}"
}

eshkol_source_dir() {
  printf '%s\n' "${ESHKOL_SOURCE_DIR:-${PROJECT_ROOT}/.deps/eshkol-src}"
}

eshkol_build_dir() {
  printf '%s\n' "${ESHKOL_BUILD_DIR:-${PROJECT_ROOT}/.deps/eshkol-build}"
}

project_build_dir() {
  printf '%s\n' "${BUILD_DIR:-${PROJECT_ROOT}/build}"
}

llvm_config() {
  printf '%s\n' "${LLVM_CONFIG_EXECUTABLE:-llvm-config-21}"
}

verify_supported_host() {
  local expected_os actual_os actual_version expected_arch actual_arch
  expected_os="$(lock_value supported_os)"
  expected_arch="$(lock_value supported_arch)"
  actual_arch="$(uname -m)"
  [[ "$(uname -s)" == Linux ]] || die "unsupported operating system: expected Linux"
  [[ "${actual_arch}" == "${expected_arch}" ]] || die "unsupported architecture: expected ${expected_arch}, got ${actual_arch}"
  [[ -r /etc/os-release ]] || die "cannot identify Linux release: /etc/os-release is unavailable"
  actual_os="$(. /etc/os-release; printf '%s' "${ID:-unknown}")"
  actual_version="$(. /etc/os-release; printf '%s' "${VERSION_ID:-unknown}")"
  if [[ "${actual_os}-${actual_version}" != "${expected_os}" ]]; then
    [[ "${ESHKOL_ALLOW_UNSUPPORTED_HOST:-0}" == 1 ]] || \
      die "unsupported Linux release: expected ${expected_os}, got ${actual_os}-${actual_version}; set ESHKOL_ALLOW_UNSUPPORTED_HOST=1 only for an explicitly unsupported compatibility probe"
    printf 'warning: unsupported compatibility probe on %s-%s\n' "${actual_os}" "${actual_version}" >&2
  fi
}

verify_llvm() {
  local command expected actual
  command="$(llvm_config)"
  expected="$(lock_value llvm_version)"
  require_command "${command}"
  actual="$("${command}" --version)"
  check_supported_version LLVM "${actual}" "${expected}"
}

verify_eshkol_checkout() {
  local source_dir expected actual
  source_dir="$(eshkol_source_dir)"
  expected="$(lock_value eshkol_commit)"
  [[ -d "${source_dir}/.git" ]] || die "Eshkol source checkout not found at ${source_dir}; run 'make toolchain'"
  actual="$(git -C "${source_dir}" rev-parse HEAD 2>/dev/null || true)"
  [[ "${actual}" == "${expected}" ]] || die "unsupported Eshkol revision at ${source_dir}: expected ${expected}, got ${actual:-unreadable}"
}

verify_eshkol_binary() {
  local binary expected actual
  binary="$(eshkol_build_dir)/eshkol-run"
  expected="$(lock_value eshkol_version)"
  [[ -x "${binary}" ]] || die "pinned Eshkol compiler not built at ${binary}; run 'make toolchain'"
  actual="$("${binary}" --version 2>&1)"
  [[ "${actual}" == "Eshkol Compiler v${expected}" ]] || \
    die "unsupported Eshkol compiler identity: expected 'Eshkol Compiler v${expected}', got '${actual}'"
}

verify_eshkol_provenance() {
  local file binary actual_hash expected_hash source_dir cache_home
  file="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
  binary="$(eshkol_build_dir)/eshkol-run"
  [[ -r "${file}" ]] || die "Eshkol build provenance not found at ${file}; run 'make toolchain'"
  [[ "$(tsv_value "${file}" eshkol_commit)" == "$(lock_value eshkol_commit)" ]] || \
    die "Eshkol provenance commit does not match toolchain lock"
  source_dir="$(readlink -f "$(eshkol_source_dir)")"
  [[ "$(tsv_value "${file}" eshkol_source_dir)" == "${source_dir}" ]] || \
    die "Eshkol provenance source directory does not match ${source_dir}"
  cache_home="$(awk -F= '$1 == "CMAKE_HOME_DIRECTORY:INTERNAL" { print $2 }' "$(eshkol_build_dir)/CMakeCache.txt")"
  [[ "$(readlink -f "${cache_home}")" == "${source_dir}" ]] || \
    die "Eshkol CMake cache is bound to ${cache_home}, not ${source_dir}"
  expected_hash="$(tsv_value "${file}" eshkol_binary_sha256)"
  actual_hash="$(sha256sum "${binary}" | awk '{ print $1 }')"
  [[ "${actual_hash}" == "${expected_hash}" ]] || \
    die "Eshkol compiler hash differs from its build provenance; run 'make toolchain'"
  check_supported_version LLVM "$(tsv_value "${file}" llvm_version)" "$(lock_value llvm_version)"
  check_supported_version Clang "$(tsv_value "${file}" cc_version)" "$(lock_value clang_version)"
  check_supported_version Clang++ "$(tsv_value "${file}" cxx_version)" "$(lock_value clang_version)"
  check_supported_version CMake "$(tsv_value "${file}" cmake_version)" "$(lock_value cmake_version)"
  check_supported_version Ninja "$(tsv_value "${file}" ninja_version)" "$(lock_value ninja_version)"
  check_supported_version Make "$(tsv_value "${file}" make_version)" "$(lock_value make_version)"
  check_supported_version Bash "$(tsv_value "${file}" bash_version)" "$(lock_value bash_version)"
  check_supported_version Git "$(tsv_value "${file}" git_version)" "$(lock_value git_version)"
}

verify_toolchain() {
  verify_supported_host
  require_command git
  verify_llvm
  verify_eshkol_checkout
  verify_eshkol_binary
  verify_eshkol_provenance
}
