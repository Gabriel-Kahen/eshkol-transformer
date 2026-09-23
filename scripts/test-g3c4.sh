#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
for g3c4_command in ar cmp comm diff nm objdump python3 rg sha256sum timeout; do
  require_command "${g3c4_command}"
done
g3c4_provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
g3c4_cc="$(tsv_value "${g3c4_provenance}" cc_path)"
g3c4_cxx="$(tsv_value "${g3c4_provenance}" cxx_path)"
g3c4_tmp="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-transformer-g3c4.XXXXXX")"
g3c4_error_log=''
cleanup_g3c4() {
  local status=$?
  if [[ "${status}" != 0 && -n "${g3c4_error_log}" && -s "${g3c4_error_log}" ]]; then
    sed -n '1,200p' "${g3c4_error_log}" >&2
  fi
  rm -rf -- "${g3c4_tmp}"
  exit "${status}"
}
trap cleanup_g3c4 EXIT

g3c4_dir="$(project_build_dir)/g3c4"
g3c4_library="${g3c4_dir}/libeshkol_transformer_g3c4.a"
g3c4_k1="$(project_build_dir)/k1/libeshkol_transformer_k1.a"
g3c4_i1="$(project_build_dir)/i1/libeshkol_transformer_i64.a"
g3c4_i2="$(project_build_dir)/i2/libeshkol_transformer_f32.a"
g3c4_runner="$(eshkol_build_dir)/eshkol-run"
g3c4_expect="${PROJECT_ROOT}/tests/g3c4/expected"
g3c4_source="${PROJECT_ROOT}/native/g3c4_primitives_provider.c"
g3c4_header="${PROJECT_ROOT}/include/eshkol_transformer/g3c4_primitives_abi.h"
g3c4_cflags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic
  -ffp-contract=off -fexcess-precision=standard -fno-fast-math
  -fstack-protector-all -I "${PROJECT_ROOT}/include"
  -I "${PROJECT_ROOT}/native" -I "${PROJECT_ROOT}/tests/g3c4" -I "${g3c4_tmp}"
)
for g3c4_artifact in "${g3c4_library}" "${g3c4_k1}" "${g3c4_i1}" "${g3c4_i2}"; do
  [[ -r "${g3c4_artifact}" ]] || die "required canonical archive missing: ${g3c4_artifact}"
done

# Freeze the only supported evidence subset while labelling allowed local probes.
{
  printf 'supported_os\t%s\n' "$(lock_value supported_os)"
  printf 'supported_arch\t%s\n' "$(lock_value supported_arch)"
  printf 'llvm_version\t%s\n' "$(lock_value llvm_version)"
  printf 'cc_version\t%s\n' "$(lock_value clang_version)"
  printf 'cxx_version\t%s\n' "$(lock_value clang_version)"
  printf 'eshkol_version\t%s\n' "$(lock_value eshkol_version)"
  printf 'eshkol_commit\t%s\n' "$(lock_value eshkol_commit)"
} >"${g3c4_tmp}/supported-subset.tsv"
cmp "${g3c4_expect}/supported_subset.tsv" "${g3c4_tmp}/supported-subset.tsv"
g3c4_actual_os="$(. /etc/os-release; printf '%s-%s' "${ID:-unknown}" "${VERSION_ID:-unknown}")"
g3c4_actual_llvm="$("$(llvm_config)" --version)"
g3c4_actual_cc="$("${g3c4_cc}" --version | sed -n '1s/.*version \([0-9][0-9.]*\).*/\1/p')"
g3c4_actual_cxx="$("${g3c4_cxx}" --version | sed -n '1s/.*version \([0-9][0-9.]*\).*/\1/p')"
g3c4_lane=compatibility
if [[ "${g3c4_actual_os}" == "$(lock_value supported_os)" &&
      "$(uname -m)" == "$(lock_value supported_arch)" &&
      "${g3c4_actual_llvm}" == "$(lock_value llvm_version)" &&
      "${g3c4_actual_cc}" == "$(lock_value clang_version)" &&
      "${g3c4_actual_cxx}" == "$(lock_value clang_version)" &&
      "$(tsv_value "${g3c4_provenance}" llvm_version)" == "$(lock_value llvm_version)" &&
      "$(tsv_value "${g3c4_provenance}" cc_version)" == "$(lock_value clang_version)" &&
      "$(tsv_value "${g3c4_provenance}" cxx_version)" == "$(lock_value clang_version)" ]]; then
  g3c4_lane=supported
fi

[[ "$(ar t "${g3c4_library}")" == 'g3c4_primitives_provider.o' ]] ||
  die "G3-C4-N must contain exactly one g3c4_primitives_provider.o archive member"
(cd "${PROJECT_ROOT}" && sha256sum --quiet -c "${g3c4_expect}/predecessor_sources.sha256")
nm -g --defined-only --format=posix "${g3c4_library}" |
  awk 'NF >= 2 { print $1 }' | LC_ALL=C sort -u >"${g3c4_tmp}/defined.txt"
diff -u "${g3c4_expect}/g3c4_defined_symbols.txt" "${g3c4_tmp}/defined.txt"
nm -u "${g3c4_dir}/g3c4_primitives_provider.o" |
  awk 'NF != 0 && $NF !~ /:$/ { print $NF }' | LC_ALL=C sort -u >"${g3c4_tmp}/undefined.txt"
if [[ -n "$(comm -23 "${g3c4_tmp}/undefined.txt" "${g3c4_expect}/g3c4_undefined_ceiling.txt")" ]]; then
  die "G3-C4-N undefined dependencies exceed the accepted ABI ceiling"
fi
diff -u "${g3c4_expect}/g3c4_allowed_undefined_symbols.txt" "${g3c4_tmp}/undefined.txt"
rg -l 'ET_G3C4_PRIMITIVES_ABI|et_g3c4_kernel_provider_v1' \
  "${PROJECT_ROOT}/include" "${PROJECT_ROOT}/native" "${PROJECT_ROOT}/lib" "${PROJECT_ROOT}/src" |
  sed "s#^${PROJECT_ROOT}/##" | LC_ALL=C sort >"${g3c4_tmp}/source-closure.txt"
cmp "${g3c4_expect}/g3c4_source_closure.txt" "${g3c4_tmp}/source-closure.txt"
python3 - "${PROJECT_ROOT}" "${g3c4_dir}/g3c4_primitives_provider.d" \
  >"${g3c4_tmp}/compile-closure.txt" <<'PY'
from pathlib import Path
import shlex
import sys

root = Path(sys.argv[1]).resolve()
text = Path(sys.argv[2]).read_text(encoding="utf-8").replace("\\\n", "")
target, dependencies = text.split(":", 1)
assert target == "g3c4_primitives_provider.o", target
paths = set()
for entry in shlex.split(dependencies):
    path = Path(entry)
    path = (path if path.is_absolute() else root / path).resolve()
    paths.add(path.relative_to(root).as_posix())
for path in sorted(paths):
    print(path)
PY
cmp "${g3c4_expect}/g3c4_compile_closure.txt" "${g3c4_tmp}/compile-closure.txt"
if nm -g --defined-only "${g3c4_library}" |
    grep -E 'et_g3c4_test_|eshkol_transformer_kernel_provider_v1' >/dev/null; then
  die "private transport or canonical K1 resolver leaked into G3-C4-N"
fi
if nm -u "${g3c4_dir}/g3c4_primitives_provider.o" |
    grep -E '(^|[[:space:]_])(malloc|calloc|realloc|free|dlopen|dlsym)(@|$)' >/dev/null; then
  die "G3-C4-N has an allocation or dynamic-loader dependency"
fi
if rg -in 'python|pytorch|torch|finite[-_ ]difference|scalar[-_ ]fallback|cpu[-_ ]fallback' \
    "${g3c4_source}" "${g3c4_header}"; then
  die "G3-C4-N production source contains an oracle or fallback reference"
fi
if objdump -d "${g3c4_dir}/g3c4_primitives_provider.o" | grep -E '\b(v?fmadd|fma)' >/dev/null; then
  die "G3-C4-N contains a fused multiply-add instruction"
fi
"${g3c4_cc}" "${g3c4_cflags[@]}" -O2 -S -emit-llvm "${g3c4_source}" -o "${g3c4_tmp}/provider.ll"
if grep -E 'llvm\.(fma|fmuladd)|(^|[,( ])double([, )]|$)' "${g3c4_tmp}/provider.ll" >/dev/null; then
  die "G3-C4-N provider IR contains FMA or binary64 arithmetic"
fi

# A linked provider leaves K1's provider-free report unchanged.
"${g3c4_cc}" "${g3c4_cflags[@]}" -O2 "${PROJECT_ROOT}/tests/k1/report_baseline.c" \
  -Wl,--whole-archive "${g3c4_library}" -Wl,--no-whole-archive "${g3c4_k1}" -lm \
  -o "${g3c4_tmp}/report-baseline"
LC_ALL=C "${g3c4_tmp}/report-baseline" >"${g3c4_tmp}/baseline_report_v1.json"
(cd "${g3c4_tmp}" && sha256sum -c "${PROJECT_ROOT}/tests/k1/expected/baseline_report_v1.sha256")
"${g3c4_cc}" "${g3c4_cflags[@]}" -O2 "${PROJECT_ROOT}/tests/g3c4/report_provider.c" \
  "${g3c4_library}" "${g3c4_k1}" -lm -o "${g3c4_tmp}/report-provider"
for g3c4_report_run in 1 2; do
  LC_ALL=C "${g3c4_tmp}/report-provider" >"${g3c4_tmp}/provider-${g3c4_report_run}.json"
done
cmp "${g3c4_tmp}/provider-1.json" "${g3c4_tmp}/provider-2.json"
(cd "${g3c4_tmp}" && sha256sum -c "${g3c4_expect}/provider_report_v1.sha256")
python3 "${PROJECT_ROOT}/tests/g3c4/check_report.py" \
  "${g3c4_tmp}/baseline_report_v1.json" "${g3c4_tmp}/provider-1.json"

# Explicit selection of each predecessor, including G3-N, is report-stable.
for g3c4_predecessor in n2 n3k a2 g3n; do
  g3c4_old="$(project_build_dir)/${g3c4_predecessor}/libeshkol_transformer_${g3c4_predecessor}.a"
  [[ -r "${g3c4_old}" ]] || die "required predecessor archive missing: ${g3c4_old}"
  for g3c4_with in without with; do
    g3c4_extra=()
    if [[ "${g3c4_with}" == with ]]; then
      g3c4_extra=(-Wl,--whole-archive "${g3c4_library}" -Wl,--no-whole-archive)
    fi
    "${g3c4_cc}" "${g3c4_cflags[@]}" -O2 \
      -DG3C4_REPORT_ACCESSOR="et_${g3c4_predecessor}_kernel_provider_v1" \
      "${PROJECT_ROOT}/tests/g3c4/report_provider.c" "${g3c4_extra[@]}" \
      "${g3c4_old}" "${g3c4_k1}" -lm -o "${g3c4_tmp}/report-${g3c4_predecessor}-${g3c4_with}"
    LC_ALL=C "${g3c4_tmp}/report-${g3c4_predecessor}-${g3c4_with}" \
      >"${g3c4_tmp}/report-${g3c4_predecessor}-${g3c4_with}.json"
  done
  cmp "${g3c4_tmp}/report-${g3c4_predecessor}-without.json" \
      "${g3c4_tmp}/report-${g3c4_predecessor}-with.json"
  if grep -F 'g3c4.' "${g3c4_tmp}/report-${g3c4_predecessor}-with.json" >/dev/null; then
    die "G3-C4-N changed an explicitly selected predecessor report"
  fi
done

for g3c4_fresh in a b; do
  /usr/bin/bash "${PROJECT_ROOT}/scripts/build-g3c4.sh" "${g3c4_tmp}/fresh-${g3c4_fresh}" normal
  cmp "${g3c4_dir}/g3c4_primitives_provider.o" "${g3c4_tmp}/fresh-${g3c4_fresh}/g3c4_primitives_provider.o"
  cmp "${g3c4_dir}/g3c4_primitives_provider.d" "${g3c4_tmp}/fresh-${g3c4_fresh}/g3c4_primitives_provider.d"
  cmp "${g3c4_library}" "${g3c4_tmp}/fresh-${g3c4_fresh}/libeshkol_transformer_g3c4.a"
done
"${g3c4_cxx}" -std=c++17 -Wall -Wextra -Werror -Wpedantic -I "${PROJECT_ROOT}/include" \
  "${PROJECT_ROOT}/tests/g3c4/header_cpp.cpp" "${g3c4_library}" "${g3c4_k1}" -lm -o "${g3c4_tmp}/header-cpp"
"${g3c4_tmp}/header-cpp"
"${g3c4_cc}" "${g3c4_cflags[@]}" -O2 "${PROJECT_ROOT}/tests/g3c4/header_c.c" \
  "${g3c4_library}" "${g3c4_k1}" -lm -o "${g3c4_tmp}/header-c"
"${g3c4_tmp}/header-c"
if "${g3c4_cc}" "${g3c4_cflags[@]}" -O2 \
    "${PROJECT_ROOT}/tests/g3c4/negative_private_symbol_link.c" \
    "${g3c4_library}" "${g3c4_k1}" -lm -o "${g3c4_tmp}/negative-private" \
    >"${g3c4_tmp}/negative-private.stdout" 2>"${g3c4_tmp}/negative-private.stderr"; then
  die "private AOT transport unexpectedly links from production archives"
fi
grep -F 'et_g3c4_test_provider_transport_v1' "${g3c4_tmp}/negative-private.stderr" >/dev/null
grep -Ei 'undefined reference|undefined symbol' "${g3c4_tmp}/negative-private.stderr" >/dev/null
if "${g3c4_cc}" "${g3c4_cflags[@]}" -O2 \
    "${PROJECT_ROOT}/tests/g3c4/negative_canonical_symbol_link.c" \
    "${g3c4_library}" "${g3c4_k1}" -lm -o "${g3c4_tmp}/negative-canonical" \
    >"${g3c4_tmp}/negative-canonical.stdout" 2>"${g3c4_tmp}/negative-canonical.stderr"; then
  die "canonical provider symbol unexpectedly links from production archives"
fi
grep -F 'eshkol_transformer_kernel_provider_v1' "${g3c4_tmp}/negative-canonical.stderr" >/dev/null
grep -Ei 'undefined reference|undefined symbol' "${g3c4_tmp}/negative-canonical.stderr" >/dev/null

python3 "${PROJECT_ROOT}/tests/g3c4/check_helper_provenance.py"
python3 "${PROJECT_ROOT}/tests/g3c4/test_mutations.py" --cc "${g3c4_cc}" --k1 "${g3c4_k1}"
g3c4_tests=(test_numerical test_negative_atomicity test_i2_integration)
for g3c4_test in "${g3c4_tests[@]}"; do
  g3c4_extra=()
  if [[ "${g3c4_test}" == test_negative_atomicity ]]; then
    g3c4_extra=(-Wl,--wrap=fegetenv,--wrap=fesetenv)
  elif [[ "${g3c4_test}" == test_i2_integration ]]; then
    g3c4_extra=(-Wl,--wrap=malloc,--wrap=calloc,--wrap=realloc,--wrap=aligned_alloc,--wrap=posix_memalign,--wrap=free)
  fi
  "${g3c4_cc}" "${g3c4_cflags[@]}" "${g3c4_extra[@]}" -O2 \
    "${PROJECT_ROOT}/tests/g3c4/${g3c4_test}.c" \
    "${g3c4_library}" "${g3c4_i2}" "${g3c4_i1}" "${g3c4_k1}" -lm \
    -o "${g3c4_tmp}/${g3c4_test}"
  for g3c4_run in 1 2; do
    g3c4_error_log="${g3c4_tmp}/${g3c4_test}-${g3c4_run}.stderr"
    timeout --foreground --signal=TERM --kill-after=5s 120s "${g3c4_tmp}/${g3c4_test}" \
      >"${g3c4_tmp}/${g3c4_test}-${g3c4_run}.stdout" \
      2>"${g3c4_tmp}/${g3c4_test}-${g3c4_run}.stderr"
    test ! -s "${g3c4_tmp}/${g3c4_test}-${g3c4_run}.stderr"
  done
  cmp "${g3c4_tmp}/${g3c4_test}-1.stdout" "${g3c4_tmp}/${g3c4_test}-2.stdout"
  grep -E '^G3-C4-N .*: [0-9]+ checks$' "${g3c4_tmp}/${g3c4_test}-1.stdout" >/dev/null
  cat "${g3c4_tmp}/${g3c4_test}-1.stdout"
done

"${g3c4_cc}" "${g3c4_cflags[@]}" -O2 -c "${PROJECT_ROOT}/tests/g3c4/aot_private_bridge.c" \
  -o "${g3c4_tmp}/aot_private_bridge.o"
[[ "$(nm -g --defined-only --format=posix "${g3c4_tmp}/aot_private_bridge.o" |
    awk 'NF >= 2 { print $1 }' | LC_ALL=C sort)" == 'et_g3c4_test_provider_transport_v1' ]] ||
  die "private G3-C4-N AOT bridge export boundary changed"
nm -u "${g3c4_tmp}/aot_private_bridge.o" | grep -F 'et_g3c4_kernel_provider_v1' >/dev/null
ar rcsD "${g3c4_tmp}/libeshkol_transformer_g3c4_private_aot.a" \
  "${g3c4_tmp}/aot_private_bridge.o" "${g3c4_dir}/g3c4_primitives_provider.o" \
  "$(project_build_dir)/k1/kernel_abi.o"
mkdir -p "${g3c4_tmp}/cache-object"
g3c4_error_log="${g3c4_tmp}/aot-object.stderr"
timeout --foreground --signal=TERM --kill-after=5s 60s \
  env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="${g3c4_tmp}/cache-object" \
  "${g3c4_runner}" --strict-types --emit-object --emit-depfile "${g3c4_tmp}/aot_private.d" --no-stdlib \
  "${PROJECT_ROOT}/tests/g3c4/aot_private.esk" -o "${g3c4_tmp}/aot_private.o" \
  >"${g3c4_tmp}/aot-object.stdout" 2>"${g3c4_tmp}/aot-object.stderr"
grep -F 'tests/g3c4/aot_private.esk' "${g3c4_tmp}/aot_private.d" >/dev/null
test ! -s "${g3c4_tmp}/aot-object.stderr"
for g3c4_run in 1 2; do
  mkdir -p "${g3c4_tmp}/cache-${g3c4_run}"
  g3c4_error_log="${g3c4_tmp}/aot-compile-${g3c4_run}.stderr"
  timeout --foreground --signal=TERM --kill-after=5s 60s \
    env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="${g3c4_tmp}/cache-${g3c4_run}" \
    "${g3c4_runner}" --strict-types --no-stdlib -L "${g3c4_tmp}" --lib eshkol_transformer_g3c4_private_aot \
    "${PROJECT_ROOT}/tests/g3c4/aot_private.esk" -o "${g3c4_tmp}/aot-private-${g3c4_run}" \
    >"${g3c4_tmp}/aot-compile-${g3c4_run}.stdout" 2>"${g3c4_tmp}/aot-compile-${g3c4_run}.stderr"
  test ! -s "${g3c4_tmp}/aot-compile-${g3c4_run}.stderr"
  g3c4_error_log="${g3c4_tmp}/aot-private-${g3c4_run}.stderr"
  timeout --foreground --signal=TERM --kill-after=5s 60s "${g3c4_tmp}/aot-private-${g3c4_run}" \
    >"${g3c4_tmp}/aot-private-${g3c4_run}.stdout" 2>"${g3c4_tmp}/aot-private-${g3c4_run}.stderr"
  cmp "${g3c4_expect}/aot_private.stdout" "${g3c4_tmp}/aot-private-${g3c4_run}.stdout"
  test ! -s "${g3c4_tmp}/aot-private-${g3c4_run}.stderr"
done
cmp "${g3c4_tmp}/aot-private-1" "${g3c4_tmp}/aot-private-2"

/usr/bin/bash "${PROJECT_ROOT}/scripts/build-g3c4.sh" "${g3c4_tmp}/sanitized" sanitize
nm -u "${g3c4_tmp}/sanitized/g3c4_primitives_provider.o" |
  awk 'NF != 0 && $NF !~ /:$/ { print $NF }' | LC_ALL=C sort -u >"${g3c4_tmp}/sanitized-undefined.txt"
nm -g --defined-only --format=posix "${g3c4_tmp}/sanitized/g3c4_primitives_provider.o" |
  awk 'NF >= 2 { print $1 }' | LC_ALL=C sort -u >"${g3c4_tmp}/sanitized-defined.txt"
diff -u "${g3c4_expect}/g3c4_sanitized_undefined_symbols.txt" "${g3c4_tmp}/sanitized-undefined.txt"
diff -u "${g3c4_expect}/g3c4_sanitized_defined_symbols.txt" "${g3c4_tmp}/sanitized-defined.txt"
/usr/bin/bash "${PROJECT_ROOT}/scripts/build-i1.sh" "${g3c4_tmp}/sanitized-i1" sanitize-test
/usr/bin/bash "${PROJECT_ROOT}/scripts/build-i2.sh" "${g3c4_tmp}/sanitized-i2" sanitize-test
g3c4_sanitize=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
"${g3c4_cc}" "${g3c4_cflags[@]}" "${g3c4_sanitize[@]}" -c \
  "${PROJECT_ROOT}/native/kernel_abi.c" -o "${g3c4_tmp}/kernel-sanitized.o"
for g3c4_test in "${g3c4_tests[@]}" aot_private_bridge; do
  g3c4_extra=()
  if [[ "${g3c4_test}" == test_negative_atomicity ]]; then
    g3c4_extra=(-Wl,--wrap=fegetenv,--wrap=fesetenv)
  elif [[ "${g3c4_test}" == test_i2_integration ]]; then
    g3c4_extra=(-Wl,--wrap=malloc,--wrap=calloc,--wrap=realloc,--wrap=aligned_alloc,--wrap=posix_memalign,--wrap=free)
  fi
  "${g3c4_cc}" "${g3c4_cflags[@]}" "${g3c4_sanitize[@]}" "${g3c4_extra[@]}" \
    -DET_F32_TENSOR_TESTING -DET_I64_TENSOR_TESTING -DET_G3C4_AOT_BRIDGE_TESTING \
    "${PROJECT_ROOT}/tests/g3c4/${g3c4_test}.c" \
    "${g3c4_tmp}/sanitized/libeshkol_transformer_g3c4.a" \
    "${g3c4_tmp}/sanitized-i2/libeshkol_transformer_f32.a" \
    "${g3c4_tmp}/sanitized-i1/libeshkol_transformer_i64.a" \
    "${g3c4_tmp}/kernel-sanitized.o" -lm -o "${g3c4_tmp}/${g3c4_test}-sanitized"
  g3c4_error_log="${g3c4_tmp}/${g3c4_test}-sanitized.stderr"
  ASAN_OPTIONS=detect_leaks="${G3C4_ASAN_DETECT_LEAKS:-1}":halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
    timeout --foreground --signal=TERM --kill-after=5s 120s "${g3c4_tmp}/${g3c4_test}-sanitized" \
    >"${g3c4_tmp}/${g3c4_test}-sanitized.stdout" 2>"${g3c4_tmp}/${g3c4_test}-sanitized.stderr"
  test ! -s "${g3c4_tmp}/${g3c4_test}-sanitized.stderr"
  if [[ "${g3c4_test}" == test_i2_integration ]]; then
    g3c4_plain_checks="$(sed -n 's/^G3-C4-N I1\/I2 integration: \([0-9][0-9]*\) checks$/\1/p' "${g3c4_tmp}/${g3c4_test}-1.stdout")"
    [[ -n "${g3c4_plain_checks}" ]] || die "missing I1/I2 normal check count"
    printf 'G3-C4-N I1/I2 integration: %s checks\n' "$((g3c4_plain_checks + 1))" >"${g3c4_tmp}/i2-sanitized-expected.stdout"
    cmp "${g3c4_tmp}/i2-sanitized-expected.stdout" "${g3c4_tmp}/${g3c4_test}-sanitized.stdout"
  elif [[ "${g3c4_test}" != aot_private_bridge ]]; then
    cmp "${g3c4_tmp}/${g3c4_test}-1.stdout" "${g3c4_tmp}/${g3c4_test}-sanitized.stdout"
  fi
done
printf 'G3-C4-N PASS (%s lane on %s): 28 forward rows, independent references, atomicity, I1/I2 borrows, two fresh AOT builds, ABI/isolation, ASan/UBSan/LSan (detect_leaks=%s)\n' \
  "${g3c4_lane}" "${g3c4_actual_os}" "${G3C4_ASAN_DETECT_LEAKS:-1}"
