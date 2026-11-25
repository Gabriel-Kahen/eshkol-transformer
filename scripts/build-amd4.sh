#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
output="${1:-${root}/build/amd4}"
cc="${CC:-cc}"
cxx="${CXX:-c++}"
export LC_ALL=C
mkdir -p -- "${output}"
work="$(mktemp -d "${output}/.amd4-build.XXXXXX")"
trap 'rm -rf -- "${work}"' EXIT
lib=libeshkol_transformer_amd4.a
warnings=(-Wall -Wextra -Werror -Wpedantic -I "${root}/include")

"${cc}" -std=c11 "${warnings[@]}" -O2 -fPIC \
  -c "${root}/native/amd4_hip_observability.c" -o "${work}/amd4_hip_observability.o"
"${cxx}" -std=c++17 "${warnings[@]}" \
  -c "${root}/tests/amd4/test_amd4_header_cpp.cpp" -o "${work}/header-cpp.o"
ar rcsD "${work}/${lib}" "${work}/amd4_hip_observability.o"
[[ "$(ar t "${work}/${lib}")" == amd4_hip_observability.o ]]

nm -P -g --defined-only "${work}/${lib}" |
  awk 'NF > 1 { print $2, $1 }' | sort > "${work}/exports.actual"
diff -u - "${work}/exports.actual" <<'SYMBOLS'
T et_amd4_format_report_v1
T et_amd4_kernel_provider_v1
T et_amd4_probe_v1
T et_amd4_probe_with_ops_v1
SYMBOLS

nm -P -u "${work}/${lib}" |
  awk '
    NF > 1 && $1 !~ /^(_GLOBAL_OFFSET_TABLE_|__errno_location|__stack_chk_fail|access|bcmp|dlclose|dlopen|dlsym|memcmp|memchr|memcpy|memset|strncmp|vsnprintf)$/ {
      print "unexpected AMD4 import: " $1 > "/dev/stderr"
      failed = 1
    }
    END { exit failed }
  '

(cd -- "${work}" && sha256sum "${lib}") > "${work}/${lib}.sha256"
mv -f -- "${work}/${lib}" "${work}/${lib}.sha256" "${output}/"
printf 'AMD4 archive export/import audit and C++17 compilation passed\n'
cat -- "${output}/${lib}.sha256"
