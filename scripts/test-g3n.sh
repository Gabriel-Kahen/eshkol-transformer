#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
for g3n_command in ar cmp diff nm objdump python3 rg sha256sum timeout; do
  require_command "${g3n_command}"
done
g3n_provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
g3n_cc="$(tsv_value "${g3n_provenance}" cc_path)"
g3n_cxx="$(tsv_value "${g3n_provenance}" cxx_path)"
g3n_tmp="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-transformer-g3n.XXXXXX")"
g3n_error_log=''
cleanup_g3n() {
  local status=$?
  if [[ "${status}" != 0 && -n "${g3n_error_log}" && -s "${g3n_error_log}" ]]; then
    sed -n '1,160p' "${g3n_error_log}" >&2
  fi
  rm -rf -- "${g3n_tmp}"
  exit "${status}"
}
trap cleanup_g3n EXIT
g3n_dir="$(project_build_dir)/g3n"
g3n_library="${g3n_dir}/libeshkol_transformer_g3n.a"
g3n_k1="$(project_build_dir)/k1/libeshkol_transformer_k1.a"
g3n_i1="$(project_build_dir)/i1/libeshkol_transformer_i64.a"
g3n_i2="$(project_build_dir)/i2/libeshkol_transformer_f32.a"
g3n_runner="$(eshkol_build_dir)/eshkol-run"
g3n_expect="${PROJECT_ROOT}/tests/g3n/expected"
g3n_source="${PROJECT_ROOT}/native/g3n_primitives_provider.c"
g3n_header="${PROJECT_ROOT}/include/eshkol_transformer/g3n_primitives_abi.h"
g3n_cflags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic
  -ffp-contract=off -fexcess-precision=standard -fno-fast-math
  -fstack-protector-all -I "${PROJECT_ROOT}/include"
  -I "${PROJECT_ROOT}/native" -I "${PROJECT_ROOT}/tests/g3n" -I "${g3n_tmp}"
)
for g3n_artifact in "${g3n_library}" "${g3n_k1}" "${g3n_i1}" "${g3n_i2}"; do
  [[ -r "${g3n_artifact}" ]] || die "required canonical archive missing: ${g3n_artifact}"
done
[[ "$(ar t "${g3n_library}")" == 'g3n_primitives_provider.o' ]] || \
  die "G3N must contain exactly one g3n_primitives_provider.o archive member"

(cd "${PROJECT_ROOT}" && sha256sum --quiet -c "${g3n_expect}/predecessor_sources.sha256")
nm -g --defined-only --format=posix "${g3n_library}" | \
  awk 'NF >= 2 { print $1 }' | LC_ALL=C sort -u >"${g3n_tmp}/defined.txt"
diff -u "${g3n_expect}/g3n_defined_symbols.txt" "${g3n_tmp}/defined.txt"
nm -u "${g3n_dir}/g3n_primitives_provider.o" | \
  awk 'NF != 0 && $NF !~ /:$/ { print $NF }' | LC_ALL=C sort -u >"${g3n_tmp}/undefined.txt"
if [[ -n "$(comm -23 "${g3n_tmp}/undefined.txt" "${g3n_expect}/g3n_undefined_ceiling.txt")" ]]; then
  die "G3-N undefined dependencies exceed accepted ABI ceiling"
fi
diff -u "${g3n_expect}/g3n_allowed_undefined_symbols.txt" "${g3n_tmp}/undefined.txt"
rg -l 'ET_G3N_PRIMITIVES_ABI|et_g3n_kernel_provider_v1' \
  "${PROJECT_ROOT}/include" "${PROJECT_ROOT}/native" "${PROJECT_ROOT}/lib" "${PROJECT_ROOT}/src" | \
  sed "s#^${PROJECT_ROOT}/##" | LC_ALL=C sort >"${g3n_tmp}/source-closure.txt"
cmp "${g3n_expect}/g3n_source_closure.txt" "${g3n_tmp}/source-closure.txt"
# Inspect the actual normal-build depfile, including every non-system input,
# rather than inferring include closure from symbol-bearing source files.
python3 - "${PROJECT_ROOT}" "${g3n_dir}/g3n_primitives_provider.d" \
  >"${g3n_tmp}/compile-closure.txt" <<'PY'
from pathlib import Path
import shlex
import sys

root = Path(sys.argv[1]).resolve()
depfile = Path(sys.argv[2]).read_text(encoding="utf-8").replace("\\\n", "")
target, dependencies = depfile.split(":", 1)
assert target == "g3n_primitives_provider.o", target
paths = set()
for entry in shlex.split(dependencies):
    path = Path(entry)
    path = (path if path.is_absolute() else root / path).resolve()
    paths.add(path.relative_to(root).as_posix())
for path in sorted(paths):
    print(path)
PY
cmp "${g3n_expect}/g3n_compile_closure.txt" "${g3n_tmp}/compile-closure.txt"
if nm -g --defined-only "${g3n_library}" | \
    grep -E 'et_g3n_test_|eshkol_transformer_kernel_provider_v1' >/dev/null; then
  die "private transport or canonical K1 resolver leaked into G3N"
fi
if nm -u "${g3n_dir}/g3n_primitives_provider.o" | \
    grep -E '(^|[[:space:]_])(malloc|calloc|realloc|free|dlopen|dlsym)(@|$)' >/dev/null; then
  die "G3N has an allocation or dynamic-loader dependency"
fi
if rg -in 'python|pytorch|torch|finite[-_ ]difference|scalar[-_ ]fallback|cpu[-_ ]fallback' \
    "${g3n_source}" "${g3n_header}"; then
  die "G3N production source contains an oracle or fallback reference"
fi
if objdump -d "${g3n_dir}/g3n_primitives_provider.o" | \
    grep -E '\b(v?fmadd|fma)' >/dev/null; then
  die "G3N contains a fused multiply-add instruction"
fi
"${g3n_cc}" "${g3n_cflags[@]}" -O2 -S -emit-llvm "${g3n_source}" -o "${g3n_tmp}/provider.ll"
if grep -E 'llvm\.(fma|fmuladd)|(^|[,( ])double([, )]|$)' "${g3n_tmp}/provider.ll" >/dev/null; then
  die "G3N provider IR contains FMA or binary64 arithmetic"
fi

# An explicitly linked provider does not replace K1's provider-free baseline.
"${g3n_cc}" "${g3n_cflags[@]}" -O2 "${PROJECT_ROOT}/tests/k1/report_baseline.c" \
  -Wl,--whole-archive "${g3n_library}" -Wl,--no-whole-archive "${g3n_k1}" -lm \
  -o "${g3n_tmp}/report-baseline"
LC_ALL=C "${g3n_tmp}/report-baseline" >"${g3n_tmp}/baseline_report_v1.json"
(cd "${g3n_tmp}" && sha256sum -c "${PROJECT_ROOT}/tests/k1/expected/baseline_report_v1.sha256")
"${g3n_cc}" "${g3n_cflags[@]}" -O2 "${PROJECT_ROOT}/tests/g3n/report_provider.c" \
  "${g3n_library}" "${g3n_k1}" -lm -o "${g3n_tmp}/report-provider"
for g3n_report_run in 1 2; do
  LC_ALL=C "${g3n_tmp}/report-provider" >"${g3n_tmp}/provider-${g3n_report_run}.json"
done
cmp "${g3n_tmp}/provider-1.json" "${g3n_tmp}/provider-2.json"
(cd "${g3n_tmp}" && sha256sum -c "${g3n_expect}/provider_report_v1.sha256")
python3 "${PROJECT_ROOT}/tests/g3n/check_report.py" \
  "${g3n_tmp}/baseline_report_v1.json" "${g3n_tmp}/provider-1.json"
for g3n_predecessor in n2 n3k a2; do
  g3n_old="$(project_build_dir)/${g3n_predecessor}/libeshkol_transformer_${g3n_predecessor}.a"
  [[ -r "${g3n_old}" ]] || die "required predecessor archive missing: ${g3n_old}"
  for g3n_with in without with; do
    g3n_extra=()
    if [[ "${g3n_with}" == with ]]; then
      g3n_extra=(-Wl,--whole-archive "${g3n_library}" -Wl,--no-whole-archive)
    fi
    "${g3n_cc}" "${g3n_cflags[@]}" -O2 \
      -DG3N_REPORT_ACCESSOR="et_${g3n_predecessor}_kernel_provider_v1" \
      "${PROJECT_ROOT}/tests/g3n/report_provider.c" "${g3n_extra[@]}" \
      "${g3n_old}" "${g3n_k1}" -lm -o "${g3n_tmp}/report-${g3n_predecessor}-${g3n_with}"
    LC_ALL=C "${g3n_tmp}/report-${g3n_predecessor}-${g3n_with}" \
      >"${g3n_tmp}/report-${g3n_predecessor}-${g3n_with}.json"
  done
  cmp "${g3n_tmp}/report-${g3n_predecessor}-without.json" "${g3n_tmp}/report-${g3n_predecessor}-with.json"
  if grep -F 'g3n.' "${g3n_tmp}/report-${g3n_predecessor}-with.json" >/dev/null; then
    die "G3-N changed an explicitly selected predecessor report"
  fi
done
for g3n_fresh in a b; do
  /usr/bin/bash "${PROJECT_ROOT}/scripts/build-g3n.sh" "${g3n_tmp}/fresh-${g3n_fresh}" normal
  cmp "${g3n_dir}/g3n_primitives_provider.o" "${g3n_tmp}/fresh-${g3n_fresh}/g3n_primitives_provider.o"
  cmp "${g3n_dir}/g3n_primitives_provider.d" "${g3n_tmp}/fresh-${g3n_fresh}/g3n_primitives_provider.d"
  cmp "${g3n_library}" "${g3n_tmp}/fresh-${g3n_fresh}/libeshkol_transformer_g3n.a"
done
"${g3n_cxx}" -std=c++17 -Wall -Wextra -Werror -Wpedantic -I "${PROJECT_ROOT}/include" \
  "${PROJECT_ROOT}/tests/g3n/header_cpp.cpp" "${g3n_library}" "${g3n_k1}" -lm -o "${g3n_tmp}/header-cpp"
"${g3n_tmp}/header-cpp"
"${g3n_cc}" "${g3n_cflags[@]}" -O2 "${PROJECT_ROOT}/tests/g3n/header_c.c" \
  "${g3n_library}" "${g3n_k1}" -lm -o "${g3n_tmp}/header-c"
"${g3n_tmp}/header-c"
if "${g3n_cc}" "${g3n_cflags[@]}" -O2 \
    "${PROJECT_ROOT}/tests/g3n/negative_private_symbol_link.c" \
    "${g3n_library}" "${g3n_k1}" -lm -o "${g3n_tmp}/negative-private" \
    >"${g3n_tmp}/negative-private.stdout" 2>"${g3n_tmp}/negative-private.stderr"; then
  die "private AOT transport unexpectedly links from production archives"
fi
grep -F 'et_g3n_test_provider_transport_v1' "${g3n_tmp}/negative-private.stderr" >/dev/null
grep -Ei 'undefined reference|undefined symbol' "${g3n_tmp}/negative-private.stderr" >/dev/null

if "${g3n_cc}" "${g3n_cflags[@]}" -O2 \
    "${PROJECT_ROOT}/tests/g3n/negative_canonical_symbol_link.c" \
    "${g3n_library}" "${g3n_k1}" -lm -o "${g3n_tmp}/negative-canonical" \
    >"${g3n_tmp}/negative-canonical.stdout" 2>"${g3n_tmp}/negative-canonical.stderr"; then
  die "canonical provider symbol unexpectedly links from production archives"
fi
grep -F 'eshkol_transformer_kernel_provider_v1' "${g3n_tmp}/negative-canonical.stderr" >/dev/null
grep -Ei 'undefined reference|undefined symbol' "${g3n_tmp}/negative-canonical.stderr" >/dev/null

python3 "${PROJECT_ROOT}/tests/g3n/check_helper_provenance.py"
python3 "${PROJECT_ROOT}/tests/g3n/test_mutations.py" --cc "${g3n_cc}" --k1 "${g3n_k1}"
g3n_tests=(test_numerical test_negative_atomicity test_i2_integration)
for g3n_test in "${g3n_tests[@]}"; do
  g3n_extra=()
  if [[ "${g3n_test}" == test_negative_atomicity ]]; then
    g3n_extra=(-Wl,--wrap=fegetenv,--wrap=fesetenv)
  elif [[ "${g3n_test}" == test_i2_integration ]]; then
    g3n_extra=(-Wl,--wrap=malloc,--wrap=calloc,--wrap=realloc,--wrap=aligned_alloc,--wrap=posix_memalign,--wrap=free)
  fi
  "${g3n_cc}" "${g3n_cflags[@]}" "${g3n_extra[@]}" -O2 "${PROJECT_ROOT}/tests/g3n/${g3n_test}.c" \
    "${g3n_library}" "${g3n_i2}" "${g3n_i1}" "${g3n_k1}" -lm -o "${g3n_tmp}/${g3n_test}"
  for g3n_run in 1 2; do
    g3n_error_log="${g3n_tmp}/${g3n_test}-${g3n_run}.stderr"
    timeout --foreground --signal=TERM --kill-after=5s 60s "${g3n_tmp}/${g3n_test}" \
      >"${g3n_tmp}/${g3n_test}-${g3n_run}.stdout" 2>"${g3n_tmp}/${g3n_test}-${g3n_run}.stderr"
    test ! -s "${g3n_tmp}/${g3n_test}-${g3n_run}.stderr"
  done
  cmp "${g3n_tmp}/${g3n_test}-1.stdout" "${g3n_tmp}/${g3n_test}-2.stdout"
  grep -E '^G3-?N .*: [0-9]+ checks' "${g3n_tmp}/${g3n_test}-1.stdout" >/dev/null
  cat "${g3n_tmp}/${g3n_test}-1.stdout"
done

"${g3n_cc}" "${g3n_cflags[@]}" -O2 -c "${PROJECT_ROOT}/tests/g3n/aot_private_bridge.c" \
  -o "${g3n_tmp}/aot_private_bridge.o"
[[ "$(nm -g --defined-only --format=posix "${g3n_tmp}/aot_private_bridge.o" | \
    awk 'NF >= 2 { print $1 }' | LC_ALL=C sort)" == 'et_g3n_test_provider_transport_v1' ]] || \
  die "private G3N AOT bridge export boundary changed"
nm -u "${g3n_tmp}/aot_private_bridge.o" | grep -F 'et_g3n_kernel_provider_v1' >/dev/null
ar rcsD "${g3n_tmp}/libeshkol_transformer_g3n_private_aot.a" \
  "${g3n_tmp}/aot_private_bridge.o" "${g3n_dir}/g3n_primitives_provider.o" "$(project_build_dir)/k1/kernel_abi.o"
mkdir -p "${g3n_tmp}/cache-object"
g3n_error_log="${g3n_tmp}/aot-object.stderr"
timeout --foreground --signal=TERM --kill-after=5s 60s \
  env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="${g3n_tmp}/cache-object" \
  "${g3n_runner}" --strict-types --emit-object --emit-depfile "${g3n_tmp}/aot_private.d" --no-stdlib \
  "${PROJECT_ROOT}/tests/g3n/aot_private.esk" -o "${g3n_tmp}/aot_private.o" \
  >"${g3n_tmp}/aot-object.stdout" 2>"${g3n_tmp}/aot-object.stderr"
grep -F 'tests/g3n/aot_private.esk' "${g3n_tmp}/aot_private.d" >/dev/null
test ! -s "${g3n_tmp}/aot-object.stderr"
for g3n_run in 1 2; do
  mkdir -p "${g3n_tmp}/cache-${g3n_run}"
  g3n_error_log="${g3n_tmp}/aot-compile-${g3n_run}.stderr"
  timeout --foreground --signal=TERM --kill-after=5s 60s \
    env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="${g3n_tmp}/cache-${g3n_run}" \
    "${g3n_runner}" --strict-types --no-stdlib -L "${g3n_tmp}" --lib eshkol_transformer_g3n_private_aot \
    "${PROJECT_ROOT}/tests/g3n/aot_private.esk" -o "${g3n_tmp}/aot-private-${g3n_run}" \
    >"${g3n_tmp}/aot-compile-${g3n_run}.stdout" 2>"${g3n_tmp}/aot-compile-${g3n_run}.stderr"
  test ! -s "${g3n_tmp}/aot-compile-${g3n_run}.stderr"
  g3n_error_log="${g3n_tmp}/aot-private-${g3n_run}.stderr"
  timeout --foreground --signal=TERM --kill-after=5s 30s "${g3n_tmp}/aot-private-${g3n_run}" \
    >"${g3n_tmp}/aot-private-${g3n_run}.stdout" 2>"${g3n_tmp}/aot-private-${g3n_run}.stderr"
  cmp "${g3n_expect}/aot_private.stdout" "${g3n_tmp}/aot-private-${g3n_run}.stdout"
  test ! -s "${g3n_tmp}/aot-private-${g3n_run}.stderr"
done
cmp "${g3n_tmp}/aot-private-1" "${g3n_tmp}/aot-private-2"

/usr/bin/bash "${PROJECT_ROOT}/scripts/build-g3n.sh" "${g3n_tmp}/sanitized" sanitize
nm -u "${g3n_tmp}/sanitized/g3n_primitives_provider.o" | \
  awk 'NF != 0 && $NF !~ /:$/ { print $NF }' | LC_ALL=C sort -u >"${g3n_tmp}/sanitized-undefined.txt"
nm -g --defined-only --format=posix "${g3n_tmp}/sanitized/g3n_primitives_provider.o" | \
  awk 'NF >= 2 { print $1 }' | LC_ALL=C sort -u >"${g3n_tmp}/sanitized-defined.txt"
diff -u "${g3n_expect}/g3n_sanitized_undefined_symbols.txt" "${g3n_tmp}/sanitized-undefined.txt"
diff -u "${g3n_expect}/g3n_sanitized_defined_symbols.txt" "${g3n_tmp}/sanitized-defined.txt"
/usr/bin/bash "${PROJECT_ROOT}/scripts/build-i1.sh" "${g3n_tmp}/sanitized-i1" sanitize-test
/usr/bin/bash "${PROJECT_ROOT}/scripts/build-i2.sh" "${g3n_tmp}/sanitized-i2" sanitize-test
g3n_sanitize=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
"${g3n_cc}" "${g3n_cflags[@]}" "${g3n_sanitize[@]}" -c "${PROJECT_ROOT}/native/kernel_abi.c" -o "${g3n_tmp}/kernel-sanitized.o"
for g3n_test in "${g3n_tests[@]}" aot_private_bridge; do
  g3n_extra=()
  if [[ "${g3n_test}" == test_negative_atomicity ]]; then
    g3n_extra=(-Wl,--wrap=fegetenv,--wrap=fesetenv)
  elif [[ "${g3n_test}" == test_i2_integration ]]; then
    g3n_extra=(-Wl,--wrap=malloc,--wrap=calloc,--wrap=realloc,--wrap=aligned_alloc,--wrap=posix_memalign,--wrap=free)
  fi
  "${g3n_cc}" "${g3n_cflags[@]}" "${g3n_sanitize[@]}" "${g3n_extra[@]}" \
    -DET_F32_TENSOR_TESTING -DET_I64_TENSOR_TESTING -DET_G3N_AOT_BRIDGE_TESTING \
    "${PROJECT_ROOT}/tests/g3n/${g3n_test}.c" \
    "${g3n_tmp}/sanitized/libeshkol_transformer_g3n.a" \
    "${g3n_tmp}/sanitized-i2/libeshkol_transformer_f32.a" \
    "${g3n_tmp}/sanitized-i1/libeshkol_transformer_i64.a" "${g3n_tmp}/kernel-sanitized.o" -lm \
    -o "${g3n_tmp}/${g3n_test}-sanitized"
  g3n_error_log="${g3n_tmp}/${g3n_test}-sanitized.stderr"
  ASAN_OPTIONS=detect_leaks="${G3N_ASAN_DETECT_LEAKS:-1}":halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
    timeout --foreground --signal=TERM --kill-after=5s 60s "${g3n_tmp}/${g3n_test}-sanitized" \
    >"${g3n_tmp}/${g3n_test}-sanitized.stdout" 2>"${g3n_tmp}/${g3n_test}-sanitized.stderr"
  test ! -s "${g3n_tmp}/${g3n_test}-sanitized.stderr"
  if [[ "${g3n_test}" == test_i2_integration ]]; then
    # Instrumented I2 additionally checks the native live-allocation snapshot.
    g3n_plain_checks="$(sed -n 's/^G3-N I1\/I2 integration: \([0-9][0-9]*\) checks$/\1/p' "${g3n_tmp}/${g3n_test}-1.stdout")"
    [[ -n "${g3n_plain_checks}" ]] || die "missing I1/I2 normal check count"
    printf 'G3-N I1/I2 integration: %s checks\n' "$((g3n_plain_checks + 1))" >"${g3n_tmp}/i2-sanitized-expected.stdout"
    cmp "${g3n_tmp}/i2-sanitized-expected.stdout" "${g3n_tmp}/${g3n_test}-sanitized.stdout"
    printf 'sanitized '
    cat "${g3n_tmp}/${g3n_test}-sanitized.stdout"
  elif [[ "${g3n_test}" != aot_private_bridge ]]; then
    cmp "${g3n_tmp}/${g3n_test}-1.stdout" "${g3n_tmp}/${g3n_test}-sanitized.stdout"
  fi
done
printf 'G3-N PASS: eleven forward rows, independent references, atomicity, I1/I2 borrows, fresh private AOT, ABI/isolation, ASan/UBSan (detect_leaks=%s)\n' \
  "${G3N_ASAN_DETECT_LEAKS:-1}"
