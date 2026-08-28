#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_supported_host
require_command git
require_command cmake
require_command ninja
verify_llvm

repository="$(lock_value eshkol_repository)"
commit="$(lock_value eshkol_commit)"
source_dir="$(eshkol_source_dir)"
build_dir="$(eshkol_build_dir)"
llvm="$(llvm_config)"
llvm_major="$("${llvm}" --version | cut -d. -f1)"
cc="${CC:-$(lock_value supported_cc)}"
cxx="${CXX:-$(lock_value supported_cxx)}"
require_command "${cc}"
require_command "${cxx}"

if [[ ! -e "${source_dir}" ]]; then
  mkdir -p "$(dirname -- "${source_dir}")"
  git clone --filter=blob:none --no-checkout "${repository}" "${source_dir}"
  git -C "${source_dir}" fetch --depth 1 origin "${commit}"
  git -C "${source_dir}" checkout --detach "${commit}"
fi
verify_eshkol_checkout

mkdir -p "${build_dir}"
configure_log="${build_dir}/eshkol-transformer-configure.log"
CC="${cc}" CXX="${cxx}" cmake \
  -S "${source_dir}" \
  -B "${build_dir}" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_CONFIG_EXECUTABLE="$(command -v "${llvm}")" \
  -DESHKOL_REQUIRED_LLVM_MAJOR="${llvm_major}" \
  -DESHKOL_BUILD_TESTS=OFF \
  -DESHKOL_BUILD_EXAMPLES=OFF \
  -DESHKOL_BUILD_INTEGRATION_TESTS=OFF \
  -DESHKOL_BUILD_AGENT_FFI=OFF \
  -DESHKOL_XLA_ENABLED=OFF \
  -DESHKOL_QUANTUM_ENABLED=OFF \
  -DESHKOL_TENSORCORE_ENABLED=OFF 2>&1 | tee "${configure_log}"

if grep -F 'Manually-specified variables were not used' "${configure_log}" >/dev/null; then
  die "pinned Eshkol ignored one or more required CMake options; see ${configure_log}"
fi
cache_file="${build_dir}/CMakeCache.txt"
cache_cc="$(awk -F= '$1 == "CMAKE_C_COMPILER:FILEPATH" { print $2 }' "${cache_file}")"
cache_cxx="$(awk -F= '$1 == "CMAKE_CXX_COMPILER:FILEPATH" { print $2 }' "${cache_file}")"
[[ -x "${cache_cc}" && -x "${cache_cxx}" ]] || \
  die "Eshkol CMake cache does not identify executable C and C++ compilers"
[[ "$(readlink -f "${cache_cc}")" == "$(readlink -f "$(command -v "${cc}")")" ]] || \
  die "Eshkol build cache retained C compiler ${cache_cc}, not requested ${cc}; use a fresh ESHKOL_BUILD_DIR"
[[ "$(readlink -f "${cache_cxx}")" == "$(readlink -f "$(command -v "${cxx}")")" ]] || \
  die "Eshkol build cache retained C++ compiler ${cache_cxx}, not requested ${cxx}; use a fresh ESHKOL_BUILD_DIR"
cmake --build "${build_dir}" --target eshkol-run --parallel "${ESHKOL_JOBS:-2}"
verify_eshkol_binary
cc_version="$("${cache_cc}" --version | sed -n '1s/.*version \([0-9][0-9.]*\).*/\1/p')"
cxx_version="$("${cache_cxx}" --version | sed -n '1s/.*version \([0-9][0-9.]*\).*/\1/p')"
check_supported_version Clang "${cc_version}" "$(lock_value clang_version)"
check_supported_version Clang++ "${cxx_version}" "$(lock_value clang_version)"
check_minimum_version CMake "$(cmake --version | awk 'NR == 1 { print $3 }')" "$(lock_value minimum_cmake_version)"
check_minimum_version Ninja "$(ninja --version)" "$(lock_value minimum_ninja_version)"
check_minimum_version Make "$(make --version | awk 'NR == 1 { print $3 }')" "$(lock_value minimum_make_version)"
check_minimum_version Bash "${BASH_VERSION%%(*}" "$(lock_value minimum_bash_version)"
check_minimum_version Git "$(git --version | awk '{ print $3 }')" "$(lock_value minimum_git_version)"

{
  printf 'format_version\t1\n'
  printf 'eshkol_repository\t%s\n' "${repository}"
  printf 'eshkol_commit\t%s\n' "${commit}"
  printf 'eshkol_source_dir\t%s\n' "$(readlink -f "${source_dir}")"
  printf 'eshkol_binary_sha256\t%s\n' "$(sha256sum "${build_dir}/eshkol-run" | awk '{ print $1 }')"
  printf 'llvm_version\t%s\n' "$("${llvm}" --version)"
  printf 'cc_path\t%s\n' "${cache_cc}"
  printf 'cc_version\t%s\n' "${cc_version}"
  printf 'cxx_path\t%s\n' "${cache_cxx}"
  printf 'cxx_version\t%s\n' "${cxx_version}"
  printf 'cmake_version\t%s\n' "$(cmake --version | awk 'NR == 1 { print $3 }')"
  printf 'ninja_version\t%s\n' "$(ninja --version)"
  printf 'make_version\t%s\n' "$(make --version | awk 'NR == 1 { print $3 }')"
  printf 'bash_version\t%s\n' "${BASH_VERSION%%(*}"
  printf 'git_version\t%s\n' "$(git --version | awk '{ print $3 }')"
  printf 'feature_profile\tminimal-no-stdlib-no-agent-ffi-no-xla-no-quantum-no-tensorcore\n'
} > "${build_dir}/eshkol-transformer-provenance.tsv"
printf 'toolchain ready: Eshkol %s (%s)\n' "$(lock_value eshkol_version)" "${commit}"
