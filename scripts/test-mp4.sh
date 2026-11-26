#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root}"
cc="${CC:-clang}"
cxx="${CXX:-clang++}"
bash scripts/build-mp4.sh

scratch="$(mktemp -d "${root}/build/mp4/.test.XXXXXX")"
trap 'rm -rf -- "${scratch}"' EXIT
cp build/mp4/libeshkol_transformer_mp4.a "${scratch}/first.a"
bash scripts/build-mp4.sh
cmp "${scratch}/first.a" build/mp4/libeshkol_transformer_mp4.a
printf 'MP4 archive reproducibility: PASS\n'

common=(-Wall -Wextra -Werror -Wpedantic -ffp-contract=off -fno-fast-math -I include)
cflags=(-std=c11 "${common[@]}" -fexcess-precision=standard)
cppflags=(-std=c++17 "${common[@]}")
sanitizers=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
test_source=tests/mp4/test_mp4_provider.c
header_test=tests/mp4/test_mp4_header_cpp.cpp

"${cc}" "${cflags[@]}" -O2 -c native/kernel_abi.c -o "${scratch}/kernel.o"
objects=(build/mp4/libeshkol_transformer_mp4.a "${scratch}/kernel.o")
"${cc}" "${cflags[@]}" -O2 "${test_source}" "${objects[@]}" -lm -o "${scratch}/provider"
"${scratch}/provider"
"${cxx}" "${cppflags[@]}" "${header_test}" "${objects[@]}" -lm -o "${scratch}/header"
"${scratch}/header"
printf 'MP4 C++17 header and linkage: PASS\n'
python3 tests/mp4/test_mp4_reference.py "${scratch}/provider"

export ASAN_OPTIONS="detect_leaks=${MP4_ASAN_DETECT_LEAKS:-0}:halt_on_error=1"
export UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1"
"${cc}" "${cflags[@]}" "${sanitizers[@]}" "${test_source}" \
  native/mp4_mixed_precision_provider.c native/kernel_abi.c -lm -o "${scratch}/provider.san"
"${scratch}/provider.san"
printf 'MP4 ASan/UBSan: PASS (LSan detect_leaks=%s)\n' "${MP4_ASAN_DETECT_LEAKS:-0}"

python3 -m unittest -v tests.q0.test_python_isolation
bash scripts/check-ci-topology.sh
