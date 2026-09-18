#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
for command in cmp nm python3 timeout; do require_command "${command}"; done
cc=
cxx=
resolve_provenance_compilers cc cxx \
  "${CC:-$(lock_value supported_cc)}" "${CXX:-$(lock_value supported_cxx)}"
runner="$(eshkol_build_dir)/eshkol-run"
tmp="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-c2-x1.XXXXXX")"
trap 'rm -rf -- "${tmp}"' EXIT
cflags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -fstack-protector-all -fno-common -I "${PROJECT_ROOT}/native")
src=("${PROJECT_ROOT}/native/c2_x1_canonical.c")
"${cc}" "${cflags[@]}" "${src[@]}" "${PROJECT_ROOT}/tests/c2/c2_x1_canonical_unit.c" -o "${tmp}/unit"
"${cc}" "${cflags[@]}" "${src[@]}" "${PROJECT_ROOT}/tests/c2/c2_x1_canonical_driver.c" -o "${tmp}/driver"
"${cxx}" -std=c++17 -Wall -Wextra -Werror -Wpedantic -I "${PROJECT_ROOT}/native" "${PROJECT_ROOT}/tests/c2/c2_x1_canonical_header_cpp.cpp" -o "${tmp}/cpp"
"${tmp}/unit"; "${tmp}/cpp"
timeout --foreground --signal=TERM --kill-after=5s 60s python3 "${PROJECT_ROOT}/tests/c2/test_c2_x1_canonical.py" "${tmp}/driver"
for compiler in "${cc}" /usr/bin/gcc; do
  [[ -x "${compiler}" ]] || continue
  "${compiler}" "${cflags[@]}" -O2 "${src[@]}" "${PROJECT_ROOT}/tests/c2/c2_x1_canonical_unit.c" -o "${tmp}/unit-$(basename "${compiler}")"
  "${tmp}/unit-$(basename "${compiler}")"
done
"${cc}" "${cflags[@]}" -O1 -fsanitize=address,undefined -fno-omit-frame-pointer "${src[@]}" "${PROJECT_ROOT}/tests/c2/c2_x1_canonical_unit.c" -o "${tmp}/san"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 "${tmp}/san"
"${cc}" "${cflags[@]}" -fPIC -shared "${src[@]}" -o "${tmp}/libc2_x1.so"
nm -D --defined-only "${tmp}/libc2_x1.so" | awk '{print $3}' | \
  grep '^et_c2_private_x1_' | sort >"${tmp}/symbols"
sort "${PROJECT_ROOT}/tests/c2/expected/c2_x1_canonical_symbols.txt" >"${tmp}/expected-symbols"
cmp "${tmp}/expected-symbols" "${tmp}/symbols"
(cd "${tmp}" && \
  ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="${tmp}/cache" ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" \
    ESHKOL_CXX_COMPILER="${cxx}" \
    timeout --foreground --signal=TERM --kill-after=5s 180s "${runner}" \
      --strict-types --optimize 0 --no-stdlib -I "${PROJECT_ROOT}/lib" \
      -I "${PROJECT_ROOT}/native" -L "${tmp}" --lib c2_x1 \
      "${PROJECT_ROOT}/tests/c2/c2_x1_canonical_runtime.esk" -o "${tmp}/runtime")
LD_LIBRARY_PATH="${tmp}" "${tmp}/runtime" | grep -Fx 'C2 X1 runtime PASS: 4 checks'
cmp "${PROJECT_ROOT}/native/c2_x1_canonical_source_closure.txt" \
  <(printf '%s\n' native/c2_x1_canonical.c native/c2_x1_canonical.h \
    native/c2_x1_canonical_extension.esk)
printf 'C2 X1 CANONICAL PASS: C/C++, 85 hostile checks, exact private symbols, Eshkol runtime, GCC/Clang, ASan/UBSan\n'
