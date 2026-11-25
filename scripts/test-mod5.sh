#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root}"
cc="${CC:-clang}"
cxx="${CXX:-clang++}"
bash scripts/build-mod5.sh

scratch="$(mktemp -d "${root}/build/mod5/.test.XXXXXX")"
trap 'rm -rf -- "${scratch}"' EXIT
common=(-Wall -Wextra -Werror -Wpedantic -ffp-contract=off -fno-fast-math -I include)
cflags=(-std=c11 "${common[@]}" -fexcess-precision=standard)
cppflags=(-std=c++17 "${common[@]}")
sanitizers=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
tests=(test_mod5_provider test_mod5_training_smoke)

"${cc}" "${cflags[@]}" -O2 -c native/kernel_abi.c -o "${scratch}/kernel.o"
objects=(build/mod5/libeshkol_transformer_mod5.a "${scratch}/kernel.o")
for test in "${tests[@]}"; do
  "${cc}" "${cflags[@]}" -O2 "tests/mod5/${test}.c" "${objects[@]}" -lm -o "${scratch}/${test}"
  "${scratch}/${test}"
done
"${cxx}" "${cppflags[@]}" "tests/mod5/test_mod5_header_cpp.cpp" "${objects[@]}" -lm \
  -o "${scratch}/header"
"${scratch}/header"
printf 'MOD5 C++17 header and linkage: PASS\n'
python3 tests/mod5/test_mod5_reference.py "${scratch}/test_mod5_provider"

export ASAN_OPTIONS="detect_leaks=${MOD5_ASAN_DETECT_LEAKS:-0}:halt_on_error=1"
export UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1"
for source in kernel_abi mod5_primitives_provider; do
  "${cc}" "${cflags[@]}" "${sanitizers[@]}" -c "native/${source}.c" -o "${scratch}/${source}.san.o"
done
san_objects=("${scratch}/mod5_primitives_provider.san.o" "${scratch}/kernel_abi.san.o")
for test in "${tests[@]}"; do
  "${cc}" "${cflags[@]}" "${sanitizers[@]}" "tests/mod5/${test}.c" "${san_objects[@]}" \
    -lm -o "${scratch}/${test}.san"
  "${scratch}/${test}.san"
done
"${cxx}" "${cppflags[@]}" "${sanitizers[@]}" tests/mod5/test_mod5_header_cpp.cpp \
  "${san_objects[@]}" -lm -o "${scratch}/header.san"
"${scratch}/header.san"
printf 'MOD5 ASan/UBSan: PASS (LSan detect_leaks=%s)\n' "${MOD5_ASAN_DETECT_LEAKS:-0}"

python3 -m unittest -v tests.q0.test_python_isolation
bash scripts/check-ci-topology.sh
