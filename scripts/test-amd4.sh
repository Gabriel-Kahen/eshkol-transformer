#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cc="${CC:-cc}"
cxx="${CXX:-c++}"
work="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-transformer-amd4.XXXXXX")"
trap 'rm -rf -- "${work}"' EXIT
lib=libeshkol_transformer_amd4.a
warnings=(-Wall -Wextra -Werror -Wpedantic -I "${root}/include")
sanitize=(-fsanitize=address,undefined -fno-omit-frame-pointer)
test_sources=("${root}/tests/amd4/test_amd4_observability.c")
fixture_flags=()
k1_source="${root}/native/kernel_abi.c"

if [[ "${AMD4_TEST_LOADER_FIXTURES:-0}" == 1 ]]; then
  test_sources+=("${root}/tests/amd4/test_amd4_loader.c")
  fixture_flags=(-DAMD4_TEST_LOADER_FIXTURES -Wl,--export-dynamic-symbol=amd4_loader_fixture)
fi

run_loader_case() {
  local runner="$1" fixtures="$2" scenario="$3" hip="$4" blas="$5"
  local libraries="${fixtures}/${scenario}"
  mkdir -- "${libraries}"
  ln -s -- "../${hip}.so" "${libraries}/libamdhip64.so"
  ln -s -- "../${blas}.so" "${libraries}/librocblas.so"
  env -u LD_PRELOAD -u LD_AUDIT LD_LIBRARY_PATH="${libraries}" \
    "${runner}" "${scenario}"
}

run_loader_fixtures() {
  local runner="$1" fixtures="$2"
  shift 2
  local shared=(-std=c11 "${warnings[@]}" "$@" -fPIC -shared)
  local source="${root}/tests/amd4/loader_fixture_library.c"
  mkdir -- "${fixtures}"
  "${cc}" "${shared[@]}" -DAMD4_FIXTURE_HIP "${source}" -o "${fixtures}/hip.so"
  "${cc}" "${shared[@]}" "${source}" -o "${fixtures}/blas.so"
  "${cc}" "${shared[@]}" -DAMD4_FIXTURE_HIP -DAMD4_FIXTURE_MISSING_EXPORT \
    "${source}" -o "${fixtures}/hip-missing.so"
  "${cc}" "${shared[@]}" -DAMD4_FIXTURE_MISSING_EXPORT \
    "${source}" -o "${fixtures}/blas-missing.so"
  "${cc}" "${shared[@]}" -DAMD4_FIXTURE_UNAVAILABLE -Wl,-z,lazy \
    "${source}" -o "${fixtures}/unavailable.so"

  run_loader_case "${runner}" "${fixtures}" ready hip blas
  run_loader_case "${runner}" "${fixtures}" hip-missing-export hip-missing blas
  run_loader_case "${runner}" "${fixtures}" blas-missing-export hip blas-missing
  run_loader_case "${runner}" "${fixtures}" hip-unavailable unavailable blas
  run_loader_case "${runner}" "${fixtures}" blas-unavailable hip unavailable
}

export LC_ALL=C
export ASAN_OPTIONS="detect_leaks=${AMD4_ASAN_DETECT_LEAKS:-0}:halt_on_error=1"
export UBSAN_OPTIONS=halt_on_error=1

bash "${root}/scripts/build-amd4.sh" "${work}/first"
bash "${root}/scripts/build-amd4.sh" "${work}/second"
cmp -- "${work}/first/${lib}" "${work}/second/${lib}"
(cd -- "${work}/first" && sha256sum -c "${lib}.sha256")

"${cc}" -std=c11 "${warnings[@]}" "${fixture_flags[@]}" \
  "${test_sources[@]}" "${k1_source}" "${work}/first/${lib}" \
  -ldl -o "${work}/test-amd4"
"${work}/test-amd4"
"${cxx}" -std=c++17 "${warnings[@]}" "${root}/tests/amd4/test_amd4_header_cpp.cpp" \
  "${work}/first/${lib}" -ldl -o "${work}/header-cpp"
"${work}/header-cpp"
printf 'AMD4 C++17 linkage passed\n'

"${cc}" -std=c11 "${warnings[@]}" "${sanitize[@]}" -g \
  "${fixture_flags[@]}" "${test_sources[@]}" "${k1_source}" \
  "${root}/native/amd4_hip_observability.c" \
  -ldl -o "${work}/test-amd4-sanitized"
"${work}/test-amd4-sanitized"
printf 'AMD4 archive reproducibility, ASan and UBSan passed (detect_leaks=%s)\n' \
  "${AMD4_ASAN_DETECT_LEAKS:-0}"

if [[ "${AMD4_TEST_LOADER_FIXTURES:-0}" == 1 ]]; then
  run_loader_fixtures "${work}/test-amd4" "${work}/loader-fixtures"
  run_loader_fixtures "${work}/test-amd4-sanitized" \
    "${work}/loader-fixtures-sanitized" "${sanitize[@]}" -g
fi
