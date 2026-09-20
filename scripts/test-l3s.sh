#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
for l3s_command in ar cmp nm objdump python3 rg sha256sum timeout; do
  require_command "${l3s_command}"
done
l3s_provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
l3s_cc="$(tsv_value "${l3s_provenance}" cc_path)"
l3s_cxx="$(tsv_value "${l3s_provenance}" cxx_path)"
l3s_runner="$(eshkol_build_dir)/eshkol-run"
l3s_tmp="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-transformer-l3s.XXXXXX")"
l3s_error_log=''
cleanup_l3s() {
  local status=$?
  if [[ "${status}" != 0 && -n "${l3s_error_log}" && -s "${l3s_error_log}" ]]; then
    sed -n '1,160p' "${l3s_error_log}" >&2
  fi
  rm -rf -- "${l3s_tmp}"
  exit "${status}"
}
trap cleanup_l3s EXIT
l3s_build="$(project_build_dir)"
l3s_dir="${l3s_build}/l3s"
l3s_library="${l3s_dir}/libeshkol_transformer_l3s.a"
l3s_l2="${l3s_build}/l2/libeshkol_transformer_l2.a"
l3s_k1="${l3s_build}/k1/libeshkol_transformer_k1.a"
l3s_i1="${l3s_build}/i1/libeshkol_transformer_i64.a"
l3s_i2="${l3s_build}/i2/libeshkol_transformer_f32.a"
l3s_source="${PROJECT_ROOT}/native/l3s_masked_objective_provider.c"
l3s_tests_dir="${PROJECT_ROOT}/tests/l3s"
l3s_cflags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -ffp-contract=off
  -fexcess-precision=standard -fno-fast-math -frounding-math -fstack-protector-all
  -I "${PROJECT_ROOT}/include" -I "${l3s_tests_dir}")
for l3s_artifact in "${l3s_library}" "${l3s_l2}" "${l3s_k1}" "${l3s_i1}" "${l3s_i2}"; do
  [[ -r "${l3s_artifact}" ]] || die "required canonical archive missing: ${l3s_artifact}"
done
python3 "${l3s_tests_dir}/test_package_contract.py" "${l3s_dir}"
if objdump -d "${l3s_dir}/l3s_masked_objective_provider.o" | grep -E '\b(v?fmadd|fma)' >/dev/null; then
  die 'L3S contains a fused multiply-add instruction'
fi
"${l3s_cc}" "${l3s_cflags[@]}" -O2 -S -emit-llvm "${l3s_source}" -o "${l3s_tmp}/provider.ll"
if grep -E 'llvm\.(fma|fmuladd)|(^|[,( ])double([, )]|$)' "${l3s_tmp}/provider.ll" >/dev/null; then
  die 'L3S contains FMA or binary64 arithmetic'
fi
for l3s_fresh in a b; do
  /usr/bin/bash "${PROJECT_ROOT}/scripts/build-l3s.sh" "${l3s_tmp}/fresh-${l3s_fresh}"
  for l3s_file in l3s_masked_objective_provider.o l3s_masked_objective_provider.d libeshkol_transformer_l3s.a; do
    cmp "${l3s_dir}/${l3s_file}" "${l3s_tmp}/fresh-${l3s_fresh}/${l3s_file}"
  done
done
"${l3s_cxx}" -std=c++17 -Wall -Wextra -Werror -Wpedantic -I "${PROJECT_ROOT}/include" \
  "${l3s_tests_dir}/header_cpp.cpp" "${l3s_library}" "${l3s_k1}" -lm -o "${l3s_tmp}/header-cpp"
"${l3s_tmp}/header-cpp"
"${l3s_cc}" "${l3s_cflags[@]}" -O2 "${PROJECT_ROOT}/tests/k1/report_baseline.c" \
  -Wl,--whole-archive "${l3s_library}" -Wl,--no-whole-archive "${l3s_k1}" -lm -o "${l3s_tmp}/baseline"
for l3s_locale in C C.UTF-8; do
  LC_ALL="${l3s_locale}" "${l3s_tmp}/baseline" >"${l3s_tmp}/baseline-${l3s_locale}.json"
done
cmp "${l3s_tmp}/baseline-C.json" "${l3s_tmp}/baseline-C.UTF-8.json"
cp "${l3s_tmp}/baseline-C.json" "${l3s_tmp}/baseline_report_v1.json"
(cd "${l3s_tmp}" && sha256sum -c "${PROJECT_ROOT}/tests/k1/expected/baseline_report_v1.sha256")

l3s_tests=(test_l3s oracle_runner test_i2_integration)
for l3s_test in "${l3s_tests[@]}"; do
  "${l3s_cc}" "${l3s_cflags[@]}" -O2 "${l3s_tests_dir}/${l3s_test}.c" \
    "${l3s_library}" "${l3s_l2}" "${l3s_i2}" "${l3s_i1}" "${l3s_k1}" -lm -o "${l3s_tmp}/${l3s_test}"
  for l3s_run in 1 2; do
    l3s_error_log="${l3s_tmp}/${l3s_test}-${l3s_run}.stderr"
    timeout --foreground --signal=TERM --kill-after=5s 60s "${l3s_tmp}/${l3s_test}" \
      >"${l3s_tmp}/${l3s_test}-${l3s_run}.stdout" 2>"${l3s_error_log}"
    test ! -s "${l3s_error_log}"
  done
  cmp "${l3s_tmp}/${l3s_test}-1.stdout" "${l3s_tmp}/${l3s_test}-2.stdout"
  grep -E '^L3S .*PASS' "${l3s_tmp}/${l3s_test}-1.stdout" >/dev/null
  cat "${l3s_tmp}/${l3s_test}-1.stdout"
done
[[ "${Q0_PYTHON:-}" == /* && -x "${Q0_PYTHON:-}" ]] ||
  die 'L3S requires Q0_PYTHON pointing to the pinned Python 3.14.6 / PyTorch 2.13.0+cpu oracle'
for l3s_generation in 1 2; do
  ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE "${Q0_PYTHON}" "${l3s_tests_dir}/generate_reference.py" "${l3s_tmp}/reference-${l3s_generation}.json"
done
cmp "${l3s_tmp}/reference-1.json" "${l3s_tmp}/reference-2.json"
cmp "${l3s_tests_dir}/masked_objective_v1.json" "${l3s_tmp}/reference-1.json"
python3 "${l3s_tests_dir}/test_reference.py" "${l3s_tmp}/oracle_runner"
python3 "${l3s_tests_dir}/reference_probe.py"
python3 "${l3s_tests_dir}/test_mutations.py" --cc "${l3s_cc}" --build-dir "${l3s_tmp}/mutations"
python3 -m unittest -v tests.q0.test_python_isolation

# Private test-only archive: these bridge/carrier symbols never enter production.
"${l3s_cc}" "${l3s_cflags[@]}" -O2 -c "${l3s_tests_dir}/aot_private_bridge.c" -o "${l3s_tmp}/aot_private_bridge.o"
[[ "$(nm -g --defined-only --format=posix "${l3s_tmp}/aot_private_bridge.o" |
  awk 'NF >= 2 {print $1}' | LC_ALL=C sort)" == et_l3s_test_aot_bridge_v1 ]] ||
  die 'private L3S AOT bridge export boundary changed'
nm -u "${l3s_tmp}/aot_private_bridge.o" | grep -F 'et_l3s_kernel_provider_v1' >/dev/null
cat >"${l3s_tmp}/private-link.c" <<'C'
#include <stdint.h>
extern int64_t et_l3s_test_aot_bridge_v1(void);
int main(void) { return (int)et_l3s_test_aot_bridge_v1(); }
C
if "${l3s_cc}" "${l3s_cflags[@]}" "${l3s_tmp}/private-link.c" \
  "${l3s_library}" "${l3s_l2}" "${l3s_i2}" "${l3s_i1}" "${l3s_k1}" -lm \
  -o "${l3s_tmp}/private-link" >"${l3s_tmp}/private-link.stdout" 2>"${l3s_tmp}/private-link.stderr"; then
  die 'private L3S AOT symbol unexpectedly links from production'
fi
grep -F et_l3s_test_aot_bridge_v1 "${l3s_tmp}/private-link.stderr" >/dev/null
grep -Ei 'undefined reference|undefined symbol' "${l3s_tmp}/private-link.stderr" >/dev/null
ar rcsD "${l3s_tmp}/libeshkol_transformer_l3s_private_aot.a" \
  "${l3s_tmp}/aot_private_bridge.o" "${l3s_dir}/l3s_masked_objective_provider.o" \
  "${l3s_build}/l2/indexed_cross_entropy.o" "${l3s_build}/i2/f32_tensor.o" \
  "${l3s_build}/i1/i64_tensor.o" "${l3s_build}/k1/kernel_abi.o"
ar t "${l3s_tmp}/libeshkol_transformer_l3s_private_aot.a" >"${l3s_tmp}/private-members.txt"
cmp "${PROJECT_ROOT}/native/l3s_private_aot_archive_members.txt" "${l3s_tmp}/private-members.txt"
mkdir -p "${l3s_tmp}/cache-object"
l3s_error_log="${l3s_tmp}/aot-object.stderr"
timeout --foreground --signal=TERM --kill-after=5s 60s \
  env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="${l3s_tmp}/cache-object" \
  "${l3s_runner}" --strict-types --emit-object --emit-depfile "${l3s_tmp}/aot_private.d" --no-stdlib \
  "${l3s_tests_dir}/aot_private.esk" -o "${l3s_tmp}/aot_private.o" \
  >"${l3s_tmp}/aot-object.stdout" 2>"${l3s_error_log}"
test ! -s "${l3s_error_log}"
grep -F 'tests/l3s/aot_private.esk' "${l3s_tmp}/aot_private.d" >/dev/null
for l3s_run in 1 2; do
  mkdir -p "${l3s_tmp}/cache-${l3s_run}"
  l3s_error_log="${l3s_tmp}/aot-compile-${l3s_run}.stderr"
  timeout --foreground --signal=TERM --kill-after=5s 60s \
    env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="${l3s_tmp}/cache-${l3s_run}" \
    "${l3s_runner}" --strict-types --no-stdlib -L "${l3s_tmp}" --lib eshkol_transformer_l3s_private_aot \
    "${l3s_tests_dir}/aot_private.esk" -o "${l3s_tmp}/aot-private-${l3s_run}" \
    >"${l3s_tmp}/aot-compile-${l3s_run}.stdout" 2>"${l3s_error_log}"
  test ! -s "${l3s_error_log}"
  l3s_error_log="${l3s_tmp}/aot-${l3s_run}.stderr"
  timeout --foreground --signal=TERM --kill-after=5s 30s "${l3s_tmp}/aot-private-${l3s_run}" \
    >"${l3s_tmp}/aot-${l3s_run}.stdout" 2>"${l3s_error_log}"
  test ! -s "${l3s_error_log}"
  cmp "${l3s_tests_dir}/expected/aot_private.stdout" "${l3s_tmp}/aot-${l3s_run}.stdout"
done
cmp "${l3s_tmp}/aot-private-1" "${l3s_tmp}/aot-private-2"

/usr/bin/bash "${PROJECT_ROOT}/scripts/build-l3s.sh" "${l3s_tmp}/sanitized" sanitize
/usr/bin/bash "${PROJECT_ROOT}/scripts/build-l2.sh" "${l3s_tmp}/sanitized-l2" sanitize
/usr/bin/bash "${PROJECT_ROOT}/scripts/build-i1.sh" "${l3s_tmp}/sanitized-i1" sanitize-test
/usr/bin/bash "${PROJECT_ROOT}/scripts/build-i2.sh" "${l3s_tmp}/sanitized-i2" sanitize-test
l3s_sanitize=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
"${l3s_cc}" "${l3s_cflags[@]}" "${l3s_sanitize[@]}" -c "${PROJECT_ROOT}/native/kernel_abi.c" -o "${l3s_tmp}/kernel-sanitized.o"
for l3s_test in "${l3s_tests[@]}" aot_private_bridge; do
  "${l3s_cc}" "${l3s_cflags[@]}" "${l3s_sanitize[@]}" \
    -DET_F32_TENSOR_TESTING -DET_I64_TENSOR_TESTING -DET_L3S_AOT_BRIDGE_TESTING \
    "${l3s_tests_dir}/${l3s_test}.c" "${l3s_tmp}/sanitized/libeshkol_transformer_l3s.a" \
    "${l3s_tmp}/sanitized-l2/libeshkol_transformer_l2.a" \
    "${l3s_tmp}/sanitized-i2/libeshkol_transformer_f32.a" \
    "${l3s_tmp}/sanitized-i1/libeshkol_transformer_i64.a" "${l3s_tmp}/kernel-sanitized.o" -lm \
    -o "${l3s_tmp}/${l3s_test}-sanitized"
  l3s_error_log="${l3s_tmp}/${l3s_test}-sanitized.stderr"
  ASAN_OPTIONS=detect_leaks="${L3S_ASAN_DETECT_LEAKS:-1}":halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
    timeout --foreground --signal=TERM --kill-after=5s 60s "${l3s_tmp}/${l3s_test}-sanitized" \
    >"${l3s_tmp}/${l3s_test}-sanitized.stdout" 2>"${l3s_error_log}"
  test ! -s "${l3s_error_log}"
  if [[ "${l3s_test}" != aot_private_bridge ]]; then
    cmp "${l3s_tmp}/${l3s_test}-1.stdout" "${l3s_tmp}/${l3s_test}-sanitized.stdout"
  fi
done
printf 'L3S PASS: exact native masks/reduction/seeds, references, atomicity, I1/I2 borrows, fresh private AOT, ABI/isolation, ASan/UBSan (detect_leaks=%s)\n' "${L3S_ASAN_DETECT_LEAKS:-1}"
