#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
for g3s_command in ar cmp nm objdump python3 rg sha256sum timeout; do
  require_command "${g3s_command}"
done
g3s_provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
g3s_cc="$(tsv_value "${g3s_provenance}" cc_path)"
g3s_cxx="$(tsv_value "${g3s_provenance}" cxx_path)"
g3s_tmp="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-transformer-g3s.XXXXXX")"
g3s_error_log=''
cleanup_g3s() {
  local status=$?
  if [[ "${status}" != 0 && -n "${g3s_error_log}" && -s "${g3s_error_log}" ]]; then
    sed -n '1,160p' "${g3s_error_log}" >&2
  fi
  rm -rf -- "${g3s_tmp}"
  exit "${status}"
}
trap cleanup_g3s EXIT
g3s_dir="$(project_build_dir)/g3s"
g3s_library="${g3s_dir}/libeshkol_transformer_g3s.a"
g3s_k1="$(project_build_dir)/k1/libeshkol_transformer_k1.a"
g3s_i1="$(project_build_dir)/i1/libeshkol_transformer_i64.a"
g3s_i2="$(project_build_dir)/i2/libeshkol_transformer_f32.a"
g3s_runner="$(eshkol_build_dir)/eshkol-run"
g3s_expect="${PROJECT_ROOT}/tests/g3s/expected"
g3s_source="${PROJECT_ROOT}/native/g3s_sampling_provider.c"
g3s_header="${PROJECT_ROOT}/include/eshkol_transformer/g3s_sampling_abi.h"
g3s_cflags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic
  -ffp-contract=off -fexcess-precision=standard -fno-fast-math
  -fstack-protector-all -I "${PROJECT_ROOT}/include"
  -I "${PROJECT_ROOT}/native" -I "${PROJECT_ROOT}/tests/g3s" -I "${g3s_tmp}"
)
for g3s_artifact in "${g3s_library}" "${g3s_k1}" "${g3s_i1}" "${g3s_i2}"; do
  [[ -r "${g3s_artifact}" ]] || die "required canonical archive missing: ${g3s_artifact}"
done
[[ "$(ar t "${g3s_library}")" == 'g3s_sampling_provider.o' ]] || \
  die "G3S must contain exactly one g3s_sampling_provider.o archive member"

(cd "${PROJECT_ROOT}" && sha256sum --quiet -c "${g3s_expect}/predecessor_sources.sha256")
nm -g --defined-only --format=posix "${g3s_library}" | \
  awk 'NF >= 2 { print $1 }' | LC_ALL=C sort -u >"${g3s_tmp}/defined.txt"
cmp "${g3s_expect}/g3s_defined_symbols.txt" "${g3s_tmp}/defined.txt"
nm -u "${g3s_dir}/g3s_sampling_provider.o" | \
  awk 'NF != 0 && $NF !~ /:$/ { print $NF }' | LC_ALL=C sort -u >"${g3s_tmp}/undefined.txt"
case "$(tsv_value "${g3s_provenance}" cc_version)" in
  21.1.8) g3s_undefined_manifest=g3s_allowed_undefined_symbols.txt ;;
  22.1.6) g3s_undefined_manifest=g3s_llvm22_compat_undefined_symbols.txt ;;
  *) die "no reviewed exact G3S object inventory for this compiler version" ;;
esac
cmp "${g3s_expect}/${g3s_undefined_manifest}" "${g3s_tmp}/undefined.txt"
if [[ -n "$(comm -23 "${g3s_tmp}/undefined.txt" "${g3s_expect}/g3s_undefined_ceiling.txt")" ]]; then
  die "normal undefined subset exceeds the accepted ceiling"
fi
rg -l 'ET_G3S_SAMPLING_ABI|et_g3s_kernel_provider_v1' \
  "${PROJECT_ROOT}/include" "${PROJECT_ROOT}/native" | \
  sed "s#^${PROJECT_ROOT}/##" | LC_ALL=C sort >"${g3s_tmp}/source-closure.txt"
cmp "${g3s_expect}/g3s_source_closure.txt" "${g3s_tmp}/source-closure.txt"
# Inspect the actual normal-build depfile, including every non-system input,
# rather than inferring include closure from symbol-bearing source files.
python3 - "${PROJECT_ROOT}" "${g3s_dir}/g3s_sampling_provider.d" \
  >"${g3s_tmp}/compile-closure.txt" <<'PY'
from pathlib import Path
import shlex
import sys

root = Path(sys.argv[1]).resolve()
depfile = Path(sys.argv[2]).read_text(encoding="utf-8").replace("\\\n", "")
target, dependencies = depfile.split(":", 1)
assert target == "g3s_sampling_provider.o", target
paths = set()
for entry in shlex.split(dependencies):
    path = Path(entry)
    path = (path if path.is_absolute() else root / path).resolve()
    paths.add(path.relative_to(root).as_posix())
for path in sorted(paths):
    print(path)
PY
cmp "${g3s_expect}/g3s_compile_closure.txt" "${g3s_tmp}/compile-closure.txt"
if nm -g --defined-only "${g3s_library}" | \
    grep -E 'et_g3s_test_|eshkol_transformer_kernel_provider_v1' >/dev/null; then
  die "private transport or canonical K1 resolver leaked into G3S"
fi
if nm -u "${g3s_dir}/g3s_sampling_provider.o" | \
    grep -E '(^|[[:space:]_])(malloc|calloc|realloc|free|qsort|alloca|dlopen|dlsym)(@|$)' >/dev/null; then
  die "G3S has an allocation or dynamic-loader dependency"
fi
if rg -in 'python|pytorch|torch|finite[-_ ]difference|scalar[-_ ]fallback|cpu[-_ ]fallback' \
    "${g3s_source}" "${g3s_header}"; then
  die "G3S production source contains an oracle or fallback reference"
fi
if objdump -d "${g3s_dir}/g3s_sampling_provider.o" | \
    grep -E '\b(v?fmadd|fma)' >/dev/null; then
  die "G3S contains a fused multiply-add instruction"
fi
"${g3s_cc}" "${g3s_cflags[@]}" -O2 -S -emit-llvm "${g3s_source}" -o "${g3s_tmp}/provider.ll"
if grep -E 'llvm\.(fma|fmuladd)|(^|[,( ])double([, )]|$)' "${g3s_tmp}/provider.ll" >/dev/null; then
  die "G3S provider IR contains FMA or binary64 arithmetic"
fi

# Installed Eshkol source must stay independent of G3S. Native G3-C4's
# source-private sampler consumer lives under src but is not an installed
# Eshkol source file; its own package and symbol gates audit that boundary.
if rg -n --glob '*.esk' 'g3s[._-]|et_g3s|ET_G3S' \
    "${PROJECT_ROOT}/lib" "${PROJECT_ROOT}/src"; then
  die "G3S must add zero installed Eshkol exports or source dependencies"
fi

# An explicitly linked provider does not replace K1's provider-free baseline.
"${g3s_cc}" "${g3s_cflags[@]}" -O2 "${PROJECT_ROOT}/tests/k1/report_baseline.c" \
  -Wl,--whole-archive "${g3s_library}" -Wl,--no-whole-archive "${g3s_k1}" -lm \
  -o "${g3s_tmp}/report-baseline"
LC_ALL=C "${g3s_tmp}/report-baseline" >"${g3s_tmp}/baseline_report_v1.json"
(cd "${g3s_tmp}" && sha256sum -c "${PROJECT_ROOT}/tests/k1/expected/baseline_report_v1.sha256")
"${g3s_cc}" "${g3s_cflags[@]}" -O2 "${PROJECT_ROOT}/tests/g3s/report.c" \
  "${g3s_library}" "${g3s_k1}" -lm -o "${g3s_tmp}/provider-report"
for g3s_run in 1 2; do
  LC_ALL=C "${g3s_tmp}/provider-report" >"${g3s_tmp}/provider-report-${g3s_run}.json"
  cmp "${g3s_expect}/provider_report.json" "${g3s_tmp}/provider-report-${g3s_run}.json"
done
(cd "${g3s_expect}" && sha256sum --quiet -c provider_report.sha256)
for g3s_fresh in a b; do
  /usr/bin/bash "${PROJECT_ROOT}/scripts/build-g3s.sh" "${g3s_tmp}/fresh-${g3s_fresh}" normal
  cmp "${g3s_dir}/g3s_sampling_provider.o" "${g3s_tmp}/fresh-${g3s_fresh}/g3s_sampling_provider.o"
  cmp "${g3s_dir}/g3s_sampling_provider.d" "${g3s_tmp}/fresh-${g3s_fresh}/g3s_sampling_provider.d"
  cmp "${g3s_dir}/g3s_sampling_provider.su" "${g3s_tmp}/fresh-${g3s_fresh}/g3s_sampling_provider.su"
  cmp "${g3s_library}" "${g3s_tmp}/fresh-${g3s_fresh}/libeshkol_transformer_g3s.a"
done
"${g3s_cxx}" -std=c++17 -Wall -Wextra -Werror -Wpedantic -I "${PROJECT_ROOT}/include" \
  "${PROJECT_ROOT}/tests/g3s/header_cpp.cpp" "${g3s_library}" "${g3s_k1}" -lm -o "${g3s_tmp}/header-cpp"
"${g3s_tmp}/header-cpp"
if "${g3s_cc}" "${g3s_cflags[@]}" -O2 \
    "${PROJECT_ROOT}/tests/g3s/negative_private_symbol_link.c" \
    "${g3s_library}" "${g3s_k1}" -lm -o "${g3s_tmp}/negative-private" \
    >"${g3s_tmp}/negative-private.stdout" 2>"${g3s_tmp}/negative-private.stderr"; then
  die "private AOT transport unexpectedly links from production archives"
fi
grep -F 'et_g3s_test_provider_transport_v1' "${g3s_tmp}/negative-private.stderr" >/dev/null
grep -Ei 'undefined reference|undefined symbol' "${g3s_tmp}/negative-private.stderr" >/dev/null

# Independent literal f32 and Decimal80 mathematical oracles; no production Python.
python3 "${PROJECT_ROOT}/tests/g3s/reference.py" --header "${g3s_tmp}/g3s_reference.h"
python3 "${PROJECT_ROOT}/tests/g3s/reference.py" --header "${g3s_tmp}/g3s_reference_again.h"
cmp "${g3s_tmp}/g3s_reference.h" "${g3s_tmp}/g3s_reference_again.h"
g3s_wrap=(-Wl,--wrap=malloc,--wrap=calloc,--wrap=realloc,--wrap=free,--wrap=fegetenv,--wrap=fesetenv,--wrap=expf)
g3s_tests=(numerical adversarial)
for g3s_test in "${g3s_tests[@]}"; do
  if [[ "${g3s_test}" == numerical ]]; then
    g3s_link=("${g3s_k1}") # test-only source inclusion observes internal rounded intermediates
  else
    g3s_link=("${g3s_library}" "${g3s_i2}" "${g3s_i1}" "${g3s_k1}" "${g3s_wrap[@]}")
  fi
  "${g3s_cc}" "${g3s_cflags[@]}" -O2 "${PROJECT_ROOT}/tests/g3s/${g3s_test}.c" \
    "${g3s_link[@]}" -lm -o "${g3s_tmp}/${g3s_test}"
  for g3s_run in 1 2; do
    g3s_error_log="${g3s_tmp}/${g3s_test}-${g3s_run}.stderr"
    timeout --foreground --signal=TERM --kill-after=5s 60s "${g3s_tmp}/${g3s_test}" \
      >"${g3s_tmp}/${g3s_test}-${g3s_run}.stdout" 2>"${g3s_error_log}"
    test ! -s "${g3s_error_log}"
  done
  cmp "${g3s_tmp}/${g3s_test}-1.stdout" "${g3s_tmp}/${g3s_test}-2.stdout"
  grep -F 'PASS' "${g3s_tmp}/${g3s_test}-1.stdout" >/dev/null
  cat "${g3s_tmp}/${g3s_test}-1.stdout"
done
python3 "${PROJECT_ROOT}/tests/g3s/mutations.py" --cc "${g3s_cc}" \
  --header-dir "${g3s_tmp}" --k1 "${g3s_k1}"
python3 "${PROJECT_ROOT}/tests/g3s/check_stack.py" \
  "${g3s_dir}/g3s_sampling_provider.su" "${g3s_tmp}/provider.ll"
"${g3s_cc}" "${g3s_cflags[@]}" -O2 "${PROJECT_ROOT}/tests/g3s/runtime_stack.c" \
  "${PROJECT_ROOT}/tests/g3s/runtime_stack.S" "${g3s_library}" "${g3s_k1}" -lm \
  -o "${g3s_tmp}/runtime-stack"
"${g3s_tmp}/runtime-stack"

"${g3s_cc}" "${g3s_cflags[@]}" -O2 -c "${PROJECT_ROOT}/tests/g3s/aot_private_bridge.c" \
  -o "${g3s_tmp}/aot_private_bridge.o"
[[ "$(nm -g --defined-only --format=posix "${g3s_tmp}/aot_private_bridge.o" | \
    awk 'NF >= 2 { print $1 }' | LC_ALL=C sort)" == 'et_g3s_test_provider_transport_v1' ]] || \
  die "private G3S AOT bridge export boundary changed"
nm -u "${g3s_tmp}/aot_private_bridge.o" | grep -F 'et_g3s_kernel_provider_v1' >/dev/null
ar rcsD "${g3s_tmp}/libeshkol_transformer_g3s_private_aot.a" \
  "${g3s_tmp}/aot_private_bridge.o" "${g3s_dir}/g3s_sampling_provider.o" "$(project_build_dir)/k1/kernel_abi.o"
mkdir -p "${g3s_tmp}/cache-object"
g3s_error_log="${g3s_tmp}/aot-object.stderr"
timeout --foreground --signal=TERM --kill-after=5s 60s \
  env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="${g3s_tmp}/cache-object" \
  "${g3s_runner}" --strict-types --emit-object --emit-depfile "${g3s_tmp}/aot_private.d" --no-stdlib \
  "${PROJECT_ROOT}/tests/g3s/aot_private.esk" -o "${g3s_tmp}/aot_private.o" \
  >"${g3s_tmp}/aot-object.stdout" 2>"${g3s_tmp}/aot-object.stderr"
grep -F 'tests/g3s/aot_private.esk' "${g3s_tmp}/aot_private.d" >/dev/null
test ! -s "${g3s_tmp}/aot-object.stderr"
for g3s_run in 1 2; do
  mkdir -p "${g3s_tmp}/cache-${g3s_run}"
  g3s_error_log="${g3s_tmp}/aot-compile-${g3s_run}.stderr"
  timeout --foreground --signal=TERM --kill-after=5s 60s \
    env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="${g3s_tmp}/cache-${g3s_run}" \
    "${g3s_runner}" --strict-types --no-stdlib -L "${g3s_tmp}" --lib eshkol_transformer_g3s_private_aot \
    "${PROJECT_ROOT}/tests/g3s/aot_private.esk" -o "${g3s_tmp}/aot-private-${g3s_run}" \
    >"${g3s_tmp}/aot-compile-${g3s_run}.stdout" 2>"${g3s_tmp}/aot-compile-${g3s_run}.stderr"
  test ! -s "${g3s_tmp}/aot-compile-${g3s_run}.stderr"
  g3s_error_log="${g3s_tmp}/aot-private-${g3s_run}.stderr"
  timeout --foreground --signal=TERM --kill-after=5s 30s "${g3s_tmp}/aot-private-${g3s_run}" \
    >"${g3s_tmp}/aot-private-${g3s_run}.stdout" 2>"${g3s_tmp}/aot-private-${g3s_run}.stderr"
  cmp "${g3s_expect}/aot_private.stdout" "${g3s_tmp}/aot-private-${g3s_run}.stdout"
  test ! -s "${g3s_tmp}/aot-private-${g3s_run}.stderr"
done
cmp "${g3s_tmp}/aot-private-1" "${g3s_tmp}/aot-private-2"

/usr/bin/bash "${PROJECT_ROOT}/scripts/build-g3s.sh" "${g3s_tmp}/sanitized" sanitize
# Instrumented dependency evidence stays separate from the canonical normal object.
nm -u "${g3s_tmp}/sanitized/g3s_sampling_provider.o" | \
  awk 'NF != 0 && $NF !~ /:$/ { print $NF }' | LC_ALL=C sort -u \
  >"${g3s_tmp}/sanitized/undefined.txt"
python3 - "${g3s_expect}/g3s_undefined_ceiling.txt" "${g3s_tmp}/sanitized/undefined.txt" <<'PY_SANITIZER'
from pathlib import Path
import sys
ceiling = set(Path(sys.argv[1]).read_text().splitlines())
actual = set(Path(sys.argv[2]).read_text().splitlines())
instrumentation = {s for s in actual if s.startswith(("__asan_", "__ubsan_", "__sanitizer_")) or s in ("__start_asan_globals", "__stop_asan_globals")}
assert instrumentation, actual
assert actual - instrumentation <= ceiling, actual - instrumentation - ceiling
print("G3S sanitized-only undefined inventory:", ", ".join(sorted(actual)))
PY_SANITIZER
/usr/bin/bash "${PROJECT_ROOT}/scripts/build-i1.sh" "${g3s_tmp}/sanitized-i1" sanitize-test
/usr/bin/bash "${PROJECT_ROOT}/scripts/build-i2.sh" "${g3s_tmp}/sanitized-i2" sanitize-test
g3s_sanitize=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
"${g3s_cc}" "${g3s_cflags[@]}" "${g3s_sanitize[@]}" -c "${PROJECT_ROOT}/native/kernel_abi.c" -o "${g3s_tmp}/kernel-sanitized.o"
for g3s_test in "${g3s_tests[@]}" aot_private_bridge; do
  if [[ "${g3s_test}" == numerical ]]; then
    g3s_link=("${g3s_tmp}/kernel-sanitized.o")
  else
    g3s_link=("${g3s_tmp}/sanitized/libeshkol_transformer_g3s.a"
      "${g3s_tmp}/sanitized-i2/libeshkol_transformer_f32.a"
      "${g3s_tmp}/sanitized-i1/libeshkol_transformer_i64.a"
      "${g3s_tmp}/kernel-sanitized.o")
    if [[ "${g3s_test}" == adversarial ]]; then g3s_link+=("${g3s_wrap[@]}"); fi
  fi
  "${g3s_cc}" "${g3s_cflags[@]}" "${g3s_sanitize[@]}" \
    -DET_F32_TENSOR_TESTING -DET_I64_TENSOR_TESTING -DET_G3S_AOT_BRIDGE_TESTING \
    "${PROJECT_ROOT}/tests/g3s/${g3s_test}.c" "${g3s_link[@]}" -lm \
    -o "${g3s_tmp}/${g3s_test}-sanitized"
  g3s_error_log="${g3s_tmp}/${g3s_test}-sanitized.stderr"
  ASAN_OPTIONS=detect_leaks="${G3S_ASAN_DETECT_LEAKS:-1}":halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
    timeout --foreground --signal=TERM --kill-after=5s 60s "${g3s_tmp}/${g3s_test}-sanitized" \
    >"${g3s_tmp}/${g3s_test}-sanitized.stdout" 2>"${g3s_error_log}"
  test ! -s "${g3s_error_log}"
  if [[ "${g3s_test}" != aot_private_bridge ]]; then
    cmp "${g3s_tmp}/${g3s_test}-1.stdout" "${g3s_tmp}/${g3s_test}-sanitized.stdout"
  fi
done
printf 'G3S PASS: exact sampler rows, independent references, atomicity, I1/I2 borrows, stack bounds, fresh private AOT, ABI/isolation, ASan/UBSan (detect_leaks=%s)\n' \
  "${G3S_ASAN_DETECT_LEAKS:-1}"
