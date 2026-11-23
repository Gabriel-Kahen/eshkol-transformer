#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
output="${1:-${root}/build/amd4_aot}"
cc="${CC:-cc}"
cxx="${CXX:-c++}"
export LC_ALL=C
mkdir -p -- "${output}"
work="$(mktemp -d "${output}/.amd4-aot-build.XXXXXX")"
trap 'rm -rf -- "${work}"' EXIT

lib=libeshkol_transformer_amd4_aot.a
warnings=(-Wall -Wextra -Werror -Wpedantic -I "${root}/include" -I "${root}/native")

"${cc}" -std=c11 "${warnings[@]}" -O2 -fPIC \
  -c "${root}/native/amd4_eshkol_bridge.c" -o "${work}/amd4_eshkol_bridge.o"
"${cxx}" -std=c++17 "${warnings[@]}" \
  -c "${root}/tests/amd4/header_bridge_cpp.cpp" -o "${work}/header_bridge_cpp.o"

ar rcsD "${work}/${lib}" "${work}/amd4_eshkol_bridge.o"
[[ "$(ar t "${work}/${lib}")" == amd4_eshkol_bridge.o ]]

nm -P -g --defined-only "${work}/${lib}" |
  awk 'NF > 1 { print $2, $1 }' | sort > "${work}/exports.actual"
diff -u - "${work}/exports.actual" <<'SYMBOLS'
T et_amd4_aot_format_report
T et_amd4_aot_k1_capability_count
T et_amd4_aot_k1_capability_status
T et_amd4_aot_k1_close
T et_amd4_aot_k1_dispatch
T et_amd4_aot_k1_open
T et_amd4_aot_k1_report_json
T et_amd4_aot_load_u32
T et_amd4_aot_probe_live
T et_amd4_aot_probe_scenario
T et_amd4_aot_read_byte
T et_amd4_aot_report_device_compute_units
T et_amd4_aot_report_device_count
T et_amd4_aot_report_device_free_vram
T et_amd4_aot_report_device_total_vram
T et_amd4_aot_report_device_wavefront_size
T et_amd4_aot_report_hip_library
T et_amd4_aot_report_kfd
T et_amd4_aot_report_reported_device_count
T et_amd4_aot_report_rocblas_library
T et_amd4_aot_report_size
T et_amd4_aot_report_source
T et_amd4_aot_report_status
T et_amd4_aot_store_u32
SYMBOLS

nm -P -u "${work}/${lib}" |
  awk '
    NF > 1 && $1 !~ /^(_GLOBAL_OFFSET_TABLE_|__stack_chk_fail|et_amd4_format_report_v1|et_amd4_kernel_provider_v1|et_amd4_probe_v1|et_amd4_probe_with_ops_v1|et_kernel_runtime_capability_count|et_kernel_runtime_capability_find|et_kernel_runtime_destroy|et_kernel_runtime_discover|et_kernel_runtime_dispatch|et_kernel_runtime_report_json|memcpy|memset|strcmp)$/ {
      print "unexpected AMD4 AOT bridge import: " $1 > "/dev/stderr"
      failed = 1
    }
    END { exit failed }
  '

(cd -- "${work}" && sha256sum "${lib}") > "${work}/${lib}.sha256"
mv -f -- "${work}/${lib}" "${work}/${lib}.sha256" "${output}/"
printf 'AMD4 AOT archive export/import audit and C++17 header compilation passed\n'
cat -- "${output}/${lib}.sha256"
