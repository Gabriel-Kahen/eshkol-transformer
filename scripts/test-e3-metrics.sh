#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
for e3_command in ar cmp nm objdump python3 rg sha256sum timeout; do require_command "${e3_command}"; done
e3_provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
e3_cc="$(tsv_value "${e3_provenance}" cc_path)"
e3_cxx="$(tsv_value "${e3_provenance}" cxx_path)"
e3_runner="$(eshkol_build_dir)/eshkol-run"
e3_tmp="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-transformer-e3-metrics.XXXXXX")"
e3_error_log=''
cleanup_e3() {
  local status=$?
  if [[ "${status}" != 0 && -n "${e3_error_log}" && -s "${e3_error_log}" ]]; then
    sed -n '1,160p' "${e3_error_log}" >&2
  fi
  rm -rf -- "${e3_tmp}"
  exit "${status}"
}
trap cleanup_e3 EXIT
e3_build="$(project_build_dir)"
e3_dir="${e3_build}/e3-metrics"
e3_tests="${PROJECT_ROOT}/tests/e3_metrics"
e3_source="${PROJECT_ROOT}/native/e3_evaluation_metrics_provider.c"
e3_library="${e3_dir}/libeshkol_transformer_e3_metrics.a"
e3_k1="${e3_build}/k1/libeshkol_transformer_k1.a"
e3_l2="${e3_build}/l2/libeshkol_transformer_l2.a"
e3_l3s="${e3_build}/l3s/libeshkol_transformer_l3s.a"
e3_i1="${e3_build}/i1/libeshkol_transformer_i64.a"
e3_i2="${e3_build}/i2/libeshkol_transformer_f32.a"
e3_libraries=("${e3_library}" "${e3_l3s}" "${e3_l2}" "${e3_i2}" "${e3_i1}" "${e3_k1}" -lm)
e3_cflags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -ffp-contract=off
  -fexcess-precision=standard -fno-fast-math -frounding-math -fstack-protector-all
  -I "${PROJECT_ROOT}/include" -I "${e3_tests}")
for e3_artifact in "${e3_library}" "${e3_k1}" "${e3_l2}" "${e3_l3s}" "${e3_i1}" "${e3_i2}"; do
  [[ -r "${e3_artifact}" ]] || die "required canonical archive missing: ${e3_artifact}"
done
"${e3_cc}" "${e3_cflags[@]}" -O2 -S -emit-llvm "${e3_source}" -o "${e3_tmp}/provider.ll"
python3 "${e3_tests}/test_package_contract.py" "${e3_dir}" "${e3_tmp}/provider.ll"
for e3_fresh in a b; do
  /usr/bin/bash "${PROJECT_ROOT}/scripts/build-e3-metrics.sh" "${e3_tmp}/fresh-${e3_fresh}"
  for e3_file in e3_evaluation_metrics_provider.o e3_evaluation_metrics_provider.d libeshkol_transformer_e3_metrics.a; do
    cmp "${e3_dir}/${e3_file}" "${e3_tmp}/fresh-${e3_fresh}/${e3_file}"
  done
done
"${e3_cxx}" -std=c++17 -Wall -Wextra -Werror -Wpedantic -I "${PROJECT_ROOT}/include" \
  "${e3_tests}/header_cpp.cpp" "${e3_library}" "${e3_k1}" -lm -o "${e3_tmp}/header-cpp"
"${e3_tmp}/header-cpp"
"${e3_cc}" "${e3_cflags[@]}" -O2 "${PROJECT_ROOT}/tests/k1/report_baseline.c" \
  -Wl,--whole-archive "${e3_library}" -Wl,--no-whole-archive "${e3_k1}" -lm -o "${e3_tmp}/baseline"
"${e3_cc}" "${e3_cflags[@]}" -O2 "${e3_tests}/report_provider.c" "${e3_library}" "${e3_k1}" -lm -o "${e3_tmp}/report"
for e3_locale in C C.UTF-8; do
  LC_ALL="${e3_locale}" "${e3_tmp}/baseline" >"${e3_tmp}/baseline-${e3_locale}.json"
  LC_ALL="${e3_locale}" "${e3_tmp}/report" >"${e3_tmp}/report-${e3_locale}.json"
done
cmp "${e3_tmp}/baseline-C.json" "${e3_tmp}/baseline-C.UTF-8.json"
cp "${e3_tmp}/baseline-C.json" "${e3_tmp}/baseline_report_v1.json"
(cd "${e3_tmp}" && sha256sum -c "${PROJECT_ROOT}/tests/k1/expected/baseline_report_v1.sha256")
cmp "${e3_tmp}/report-C.json" "${e3_tmp}/report-C.UTF-8.json"
cmp "${e3_tests}/expected/provider_report.json" "${e3_tmp}/report-C.json"

"${e3_cc}" "${e3_cflags[@]}" -O2 "${e3_tests}/native_test.c" "${e3_library}" "${e3_k1}" -lm -o "${e3_tmp}/native_test"
"${e3_cc}" "${e3_cflags[@]}" -O2 -DET_E3_COMPOSITION_STANDALONE "${e3_tests}/composition.c" \
  "${e3_libraries[@]}" -o "${e3_tmp}/composition"
"${e3_cc}" "${e3_cflags[@]}" -O2 "${e3_tests}/fenv_fault_test.c" "${e3_library}" "${e3_k1}" -lm \
  -Wl,--wrap=fegetenv -Wl,--wrap=fesetenv -o "${e3_tmp}/fenv_fault_test"
for e3_test in native_test composition fenv_fault_test; do
  for e3_run in 1 2; do
    e3_error_log="${e3_tmp}/${e3_test}-${e3_run}.stderr"
    timeout --foreground --signal=TERM --kill-after=5s 60s "${e3_tmp}/${e3_test}" \
      >"${e3_tmp}/${e3_test}-${e3_run}.stdout" 2>"${e3_error_log}"
    test ! -s "${e3_error_log}"
  done
  cmp "${e3_tmp}/${e3_test}-1.stdout" "${e3_tmp}/${e3_test}-2.stdout"
  grep -E '^E3-METRICS .*PASS' "${e3_tmp}/${e3_test}-1.stdout" >/dev/null
  cat "${e3_tmp}/${e3_test}-1.stdout"
done
[[ "${Q0_PYTHON:-}" == /* && -x "${Q0_PYTHON:-}" ]] ||
  die 'E3-METRICS requires Q0_PYTHON pointing to the pinned Python 3.14.6 / PyTorch 2.13.0+cpu oracle'
for e3_generation in 1 2; do
  ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE "${Q0_PYTHON}" "${e3_tests}/generate_reference.py" "${e3_tmp}/reference-${e3_generation}.json"
done
cmp "${e3_tmp}/reference-1.json" "${e3_tmp}/reference-2.json"
cmp "${e3_tmp}/reference-1.provenance.json" "${e3_tmp}/reference-2.provenance.json"
cmp "${e3_tests}/bool_metrics_v1.json" "${e3_tmp}/reference-1.json"
cmp "${e3_tests}/bool_metrics_v1.provenance.json" "${e3_tmp}/reference-1.provenance.json"
python3 "${e3_tests}/reference_probe.py"
python3 "${e3_tests}/test_reference.py" "${e3_tmp}/reference-1.json" "${e3_tmp}/composition"
python3 "${e3_tests}/test_mutations.py" "${e3_cc}" "${e3_tmp}/mutations" "${e3_tmp}/composition"
python3 -m unittest -v tests.q0.test_python_isolation

# The C bridge and test translation units have no production linkage authority.
"${e3_cc}" "${e3_cflags[@]}" -O2 -c "${e3_tests}/aot_private_bridge.c" -o "${e3_tmp}/aot_private_bridge.o"
"${e3_cc}" "${e3_cflags[@]}" -O2 -c "${e3_tests}/composition.c" -o "${e3_tmp}/composition.o"
nm -g --defined-only --format=posix "${e3_tmp}/aot_private_bridge.o" | awk 'NF >= 2 {print $1}' | LC_ALL=C sort >"${e3_tmp}/bridge-exports.txt"
cmp "${e3_tests}/expected/bridge_exports.txt" "${e3_tmp}/bridge-exports.txt"
nm -g --defined-only --format=posix "${e3_tmp}/composition.o" | awk 'NF >= 2 {print $1}' | LC_ALL=C sort >"${e3_tmp}/composition-exports.txt"
cmp "${e3_tests}/expected/composition_exports.txt" "${e3_tmp}/composition-exports.txt"
cat >"${e3_tmp}/private-link.c" <<'C'
#include <stdint.h>
extern int64_t et_e3_metrics_test_aot_bridge_v1(void);
int main(void) { return (int)et_e3_metrics_test_aot_bridge_v1(); }
C
if "${e3_cc}" "${e3_cflags[@]}" "${e3_tmp}/private-link.c" "${e3_libraries[@]}" \
  -o "${e3_tmp}/private-link" >"${e3_tmp}/private-link.stdout" 2>"${e3_tmp}/private-link.stderr"; then
  die 'private E3-METRICS AOT symbol unexpectedly links from production'
fi
grep -F et_e3_metrics_test_aot_bridge_v1 "${e3_tmp}/private-link.stderr" >/dev/null
grep -Ei 'undefined reference|undefined symbol' "${e3_tmp}/private-link.stderr" >/dev/null
ar rcsD "${e3_tmp}/libeshkol_transformer_e3_metrics_private_aot.a" \
  "${e3_tmp}/aot_private_bridge.o" "${e3_tmp}/composition.o" "${e3_dir}/e3_evaluation_metrics_provider.o" \
  "${e3_build}/l3s/l3s_masked_objective_provider.o" "${e3_build}/l2/indexed_cross_entropy.o" \
  "${e3_build}/i2/f32_tensor.o" "${e3_build}/i1/i64_tensor.o" "${e3_build}/k1/kernel_abi.o"
ar t "${e3_tmp}/libeshkol_transformer_e3_metrics_private_aot.a" >"${e3_tmp}/private-members.txt"
cmp "${e3_tests}/expected/private_archive_members.txt" "${e3_tmp}/private-members.txt"
mkdir -p "${e3_tmp}/cache-object"
e3_error_log="${e3_tmp}/aot-object.stderr"
timeout --foreground --signal=TERM --kill-after=5s 60s \
  env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="${e3_tmp}/cache-object" \
  "${e3_runner}" --strict-types --emit-object --emit-depfile "${e3_tmp}/aot_private.d" --no-stdlib \
  "${e3_tests}/aot_private.esk" -o "${e3_tmp}/aot_private.o" \
  >"${e3_tmp}/aot-object.stdout" 2>"${e3_error_log}"
test ! -s "${e3_error_log}"
grep -F 'tests/e3_metrics/aot_private.esk' "${e3_tmp}/aot_private.d" >/dev/null
cat "${e3_tmp}/native_test-1.stdout" >"${e3_tmp}/aot-expected.stdout"
printf 'E3-METRICS PRIVATE AOT PASS: errors, atomicity and real L2/L3S/I1/I2\n' >>"${e3_tmp}/aot-expected.stdout"
for e3_run in 1 2; do
  mkdir -p "${e3_tmp}/cache-${e3_run}"
  e3_error_log="${e3_tmp}/aot-compile-${e3_run}.stderr"
  timeout --foreground --signal=TERM --kill-after=5s 60s \
    env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="${e3_tmp}/cache-${e3_run}" \
    "${e3_runner}" --strict-types --no-stdlib -L "${e3_tmp}" --lib eshkol_transformer_e3_metrics_private_aot \
    "${e3_tests}/aot_private.esk" -o "${e3_tmp}/aot-private-${e3_run}" \
    >"${e3_tmp}/aot-compile-${e3_run}.stdout" 2>"${e3_error_log}"
  test ! -s "${e3_error_log}"
  e3_error_log="${e3_tmp}/aot-${e3_run}.stderr"
  timeout --foreground --signal=TERM --kill-after=5s 30s "${e3_tmp}/aot-private-${e3_run}" \
    >"${e3_tmp}/aot-${e3_run}.stdout" 2>"${e3_error_log}"
  test ! -s "${e3_error_log}"
  cmp "${e3_tmp}/aot-expected.stdout" "${e3_tmp}/aot-${e3_run}.stdout"
done
cmp "${e3_tmp}/aot-private-1" "${e3_tmp}/aot-private-2"
python3 "${e3_tests}/test_aot_contract.py" "${e3_tmp}"

/usr/bin/bash "${PROJECT_ROOT}/scripts/build-e3-metrics.sh" "${e3_tmp}/sanitized" sanitize
/usr/bin/bash "${PROJECT_ROOT}/scripts/build-l3s.sh" "${e3_tmp}/sanitized-l3s" sanitize
/usr/bin/bash "${PROJECT_ROOT}/scripts/build-l2.sh" "${e3_tmp}/sanitized-l2" sanitize
/usr/bin/bash "${PROJECT_ROOT}/scripts/build-i1.sh" "${e3_tmp}/sanitized-i1" sanitize-test
/usr/bin/bash "${PROJECT_ROOT}/scripts/build-i2.sh" "${e3_tmp}/sanitized-i2" sanitize-test
e3_sanitize=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
"${e3_cc}" "${e3_cflags[@]}" "${e3_sanitize[@]}" -c "${PROJECT_ROOT}/native/kernel_abi.c" -o "${e3_tmp}/kernel-sanitized.o"
e3_sanitized_libraries=("${e3_tmp}/sanitized/libeshkol_transformer_e3_metrics.a"
  "${e3_tmp}/sanitized-l3s/libeshkol_transformer_l3s.a" "${e3_tmp}/sanitized-l2/libeshkol_transformer_l2.a"
  "${e3_tmp}/sanitized-i2/libeshkol_transformer_f32.a" "${e3_tmp}/sanitized-i1/libeshkol_transformer_i64.a"
  "${e3_tmp}/kernel-sanitized.o" -lm)
for e3_test in native_test composition fenv_fault_test aot_private_bridge; do
  e3_test_flags=()
  e3_test_sources=("${e3_tests}/${e3_test}.c")
  if [[ "${e3_test}" == fenv_fault_test ]]; then e3_test_flags+=(-Wl,--wrap=fegetenv -Wl,--wrap=fesetenv); fi
  if [[ "${e3_test}" == composition ]]; then e3_test_flags+=(-DET_E3_COMPOSITION_STANDALONE); fi
  if [[ "${e3_test}" == aot_private_bridge ]]; then
    e3_test_flags+=(-DET_E3_METRICS_AOT_BRIDGE_STANDALONE)
    e3_test_sources+=("${e3_tests}/composition.c")
  fi
  "${e3_cc}" "${e3_cflags[@]}" "${e3_sanitize[@]}" -DET_F32_TENSOR_TESTING -DET_I64_TENSOR_TESTING \
    "${e3_test_flags[@]}" "${e3_test_sources[@]}" "${e3_sanitized_libraries[@]}" -o "${e3_tmp}/${e3_test}-sanitized"
  e3_error_log="${e3_tmp}/${e3_test}-sanitized.stderr"
  ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
    timeout --foreground --signal=TERM --kill-after=5s 60s "${e3_tmp}/${e3_test}-sanitized" \
    >"${e3_tmp}/${e3_test}-sanitized.stdout" 2>"${e3_error_log}"
  test ! -s "${e3_error_log}"
  if [[ "${e3_test}" != aot_private_bridge ]]; then
    cmp "${e3_tmp}/${e3_test}-1.stdout" "${e3_tmp}/${e3_test}-sanitized.stdout"
  fi
done
printf 'E3-METRICS PASS: exact package, rational/Decimal references, mutations, native/carrier checks, two fresh private AOT, ASan/UBSan/LSan (detect_leaks=1)\n'
