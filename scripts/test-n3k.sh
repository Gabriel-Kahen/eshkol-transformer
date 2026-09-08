#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
for n3k_command in ar cmp nm objdump python3 rg sha256sum timeout; do
  require_command "${n3k_command}"
done
n3k_provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
n3k_cc="$(tsv_value "${n3k_provenance}" cc_path)"
n3k_cxx="$(tsv_value "${n3k_provenance}" cxx_path)"
n3k_tmp="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-transformer-n3k.XXXXXX")"
n3k_error_log=''
cleanup_n3k() {
  local status=$?
  if [[ "${status}" != 0 && -n "${n3k_error_log}" && -s "${n3k_error_log}" ]]; then
    sed -n '1,160p' "${n3k_error_log}" >&2
  fi
  rm -rf -- "${n3k_tmp}"
  exit "${status}"
}
trap cleanup_n3k EXIT
n3k_dir="$(project_build_dir)/n3k"
n3k_library="${n3k_dir}/libeshkol_transformer_n3k.a"
n3k_k1="$(project_build_dir)/k1/libeshkol_transformer_k1.a"
n3k_i1="$(project_build_dir)/i1/libeshkol_transformer_i64.a"
n3k_i2="$(project_build_dir)/i2/libeshkol_transformer_f32.a"
n3k_runner="$(eshkol_build_dir)/eshkol-run"
n3k_expect="${PROJECT_ROOT}/tests/n3k/expected"
n3k_source="${PROJECT_ROOT}/native/n3k_primitives_provider.c"
n3k_header="${PROJECT_ROOT}/include/eshkol_transformer/n3k_primitives_abi.h"
n3k_cflags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic
  -ffp-contract=off -fexcess-precision=standard -fno-fast-math
  -fstack-protector-all -I "${PROJECT_ROOT}/include"
  -I "${PROJECT_ROOT}/native" -I "${PROJECT_ROOT}/tests/n3k" -I "${n3k_tmp}"
)
for n3k_artifact in "${n3k_library}" "${n3k_k1}" "${n3k_i1}" "${n3k_i2}"; do
  [[ -r "${n3k_artifact}" ]] || die "required canonical archive missing: ${n3k_artifact}"
done
[[ "$(ar t "${n3k_library}")" == 'n3k_primitives_provider.o' ]] || \
  die "N3K must contain exactly one n3k_primitives_provider.o archive member"

(cd "${PROJECT_ROOT}" && sha256sum --quiet -c "${n3k_expect}/predecessor_sources.sha256")
nm -g --defined-only --format=posix "${n3k_library}" | \
  awk 'NF >= 2 { print $1 }' | LC_ALL=C sort -u >"${n3k_tmp}/defined.txt"
cmp "${n3k_expect}/n3k_defined_symbols.txt" "${n3k_tmp}/defined.txt"
nm -u "${n3k_dir}/n3k_primitives_provider.o" | \
  awk 'NF != 0 && $NF !~ /:$/ { print $NF }' | LC_ALL=C sort -u >"${n3k_tmp}/undefined.txt"
cmp "${n3k_expect}/n3k_allowed_undefined_symbols.txt" "${n3k_tmp}/undefined.txt"
rg -l 'ET_N3K_PRIMITIVES_ABI|et_n3k_kernel_provider_v1' \
  "${PROJECT_ROOT}/include" "${PROJECT_ROOT}/native" | \
  sed "s#^${PROJECT_ROOT}/##" | LC_ALL=C sort >"${n3k_tmp}/source-closure.txt"
cmp "${n3k_expect}/n3k_source_closure.txt" "${n3k_tmp}/source-closure.txt"
# Inspect the actual normal-build depfile, including every non-system input,
# rather than inferring include closure from symbol-bearing source files.
python3 - "${PROJECT_ROOT}" "${n3k_dir}/n3k_primitives_provider.d" \
  >"${n3k_tmp}/compile-closure.txt" <<'PY'
from pathlib import Path
import shlex
import sys

root = Path(sys.argv[1]).resolve()
depfile = Path(sys.argv[2]).read_text(encoding="utf-8").replace("\\\n", "")
target, dependencies = depfile.split(":", 1)
assert target == "n3k_primitives_provider.o", target
paths = set()
for entry in shlex.split(dependencies):
    path = Path(entry)
    path = (path if path.is_absolute() else root / path).resolve()
    paths.add(path.relative_to(root).as_posix())
for path in sorted(paths):
    print(path)
PY
cmp "${n3k_expect}/n3k_compile_closure.txt" "${n3k_tmp}/compile-closure.txt"
if nm -g --defined-only "${n3k_library}" | \
    grep -E 'et_n3k_test_|eshkol_transformer_kernel_provider_v1' >/dev/null; then
  die "private transport or canonical K1 resolver leaked into N3K"
fi
if nm -u "${n3k_dir}/n3k_primitives_provider.o" | \
    grep -E '(^|[[:space:]_])(malloc|calloc|realloc|free|dlopen|dlsym)(@|$)' >/dev/null; then
  die "N3K has an allocation or dynamic-loader dependency"
fi
if rg -in 'python|pytorch|torch|finite[-_ ]difference|scalar[-_ ]fallback|cpu[-_ ]fallback' \
    "${n3k_source}" "${n3k_header}"; then
  die "N3K production source contains an oracle or fallback reference"
fi
if objdump -d "${n3k_dir}/n3k_primitives_provider.o" | \
    grep -E '\b(v?fmadd|fma)' >/dev/null; then
  die "N3K contains a fused multiply-add instruction"
fi
"${n3k_cc}" "${n3k_cflags[@]}" -O2 -S -emit-llvm "${n3k_source}" -o "${n3k_tmp}/provider.ll"
if grep -E 'llvm\.(fma|fmuladd)|(^|[,( ])double([, )]|$)' "${n3k_tmp}/provider.ll" >/dev/null; then
  die "N3K provider IR contains FMA or binary64 arithmetic"
fi

# An explicitly linked provider does not replace K1's provider-free baseline.
"${n3k_cc}" "${n3k_cflags[@]}" -O2 "${PROJECT_ROOT}/tests/k1/report_baseline.c" \
  -Wl,--whole-archive "${n3k_library}" -Wl,--no-whole-archive "${n3k_k1}" -lm \
  -o "${n3k_tmp}/report-baseline"
LC_ALL=C "${n3k_tmp}/report-baseline" >"${n3k_tmp}/baseline_report_v1.json"
(cd "${n3k_tmp}" && sha256sum -c "${PROJECT_ROOT}/tests/k1/expected/baseline_report_v1.sha256")
for n3k_fresh in a b; do
  /usr/bin/bash "${PROJECT_ROOT}/scripts/build-n3k.sh" "${n3k_tmp}/fresh-${n3k_fresh}" normal
  cmp "${n3k_dir}/n3k_primitives_provider.o" "${n3k_tmp}/fresh-${n3k_fresh}/n3k_primitives_provider.o"
  cmp "${n3k_dir}/n3k_primitives_provider.d" "${n3k_tmp}/fresh-${n3k_fresh}/n3k_primitives_provider.d"
  cmp "${n3k_library}" "${n3k_tmp}/fresh-${n3k_fresh}/libeshkol_transformer_n3k.a"
done
"${n3k_cxx}" -std=c++17 -Wall -Wextra -Werror -Wpedantic -I "${PROJECT_ROOT}/include" \
  "${PROJECT_ROOT}/tests/n3k/header_cpp.cpp" "${n3k_library}" "${n3k_k1}" -lm -o "${n3k_tmp}/header-cpp"
"${n3k_tmp}/header-cpp"
if "${n3k_cc}" "${n3k_cflags[@]}" -O2 \
    "${PROJECT_ROOT}/tests/n3k/negative_private_symbol_link.c" \
    "${n3k_library}" "${n3k_k1}" -lm -o "${n3k_tmp}/negative-private" \
    >"${n3k_tmp}/negative-private.stdout" 2>"${n3k_tmp}/negative-private.stderr"; then
  die "private AOT transport unexpectedly links from production archives"
fi
grep -F 'et_n3k_test_provider_transport_v1' "${n3k_tmp}/negative-private.stderr" >/dev/null
grep -Ei 'undefined reference|undefined symbol' "${n3k_tmp}/negative-private.stderr" >/dev/null

# Deterministic development-only generation; no large fixtures enter Git.
python3 "${PROJECT_ROOT}/tests/n3k/reference.py" --header "${n3k_tmp}/reference_vectors.h"
python3 "${PROJECT_ROOT}/tests/n3k/generate_composition_reference.py" --output "${n3k_tmp}/composition_reference.h"
python3 -m unittest -v tests.n3k.test_reference tests.n3k.test_initializer_reference tests.n3k.test_composition_reference
n3k_tests=(test_primitives_provider test_composition_provider test_negative_atomicity test_i2_integration)
for n3k_test in "${n3k_tests[@]}"; do
  "${n3k_cc}" "${n3k_cflags[@]}" -O2 "${PROJECT_ROOT}/tests/n3k/${n3k_test}.c" \
    "${n3k_library}" "${n3k_i2}" "${n3k_i1}" "${n3k_k1}" -lm -o "${n3k_tmp}/${n3k_test}"
  for n3k_run in 1 2; do
    n3k_error_log="${n3k_tmp}/${n3k_test}-${n3k_run}.stderr"
    timeout --foreground --signal=TERM --kill-after=5s 60s "${n3k_tmp}/${n3k_test}" \
      >"${n3k_tmp}/${n3k_test}-${n3k_run}.stdout" 2>"${n3k_tmp}/${n3k_test}-${n3k_run}.stderr"
    test ! -s "${n3k_tmp}/${n3k_test}-${n3k_run}.stderr"
  done
  cmp "${n3k_tmp}/${n3k_test}-1.stdout" "${n3k_tmp}/${n3k_test}-2.stdout"
  case "${n3k_test}" in
    test_primitives_provider)
      n3k_success='^N3K primitives: [0-9]+ checks, 3248 actual-native central derivatives; max reference absolute error [0-9.e+-]+$' ;;
    test_composition_provider)
      n3k_success='^N3K composition/initializer PASS \([0-9]+ checks\)$' ;;
    test_negative_atomicity)
      n3k_success='^N3K negative atomicity: [0-9]+ checks$' ;;
    test_i2_integration)
      n3k_success='^N3K I1/I2 integration: [0-9]+ checks$' ;;
  esac
  grep -E "${n3k_success}" "${n3k_tmp}/${n3k_test}-1.stdout" >/dev/null
  cat "${n3k_tmp}/${n3k_test}-1.stdout"
done

"${n3k_cc}" "${n3k_cflags[@]}" -O2 -c "${PROJECT_ROOT}/tests/n3k/aot_private_bridge.c" \
  -o "${n3k_tmp}/aot_private_bridge.o"
[[ "$(nm -g --defined-only --format=posix "${n3k_tmp}/aot_private_bridge.o" | \
    awk 'NF >= 2 { print $1 }' | LC_ALL=C sort)" == 'et_n3k_test_provider_transport_v1' ]] || \
  die "private N3K AOT bridge export boundary changed"
nm -u "${n3k_tmp}/aot_private_bridge.o" | grep -F 'et_n3k_kernel_provider_v1' >/dev/null
ar rcsD "${n3k_tmp}/libeshkol_transformer_n3k_private_aot.a" \
  "${n3k_tmp}/aot_private_bridge.o" "${n3k_dir}/n3k_primitives_provider.o" "$(project_build_dir)/k1/kernel_abi.o"
mkdir -p "${n3k_tmp}/cache-object"
n3k_error_log="${n3k_tmp}/aot-object.stderr"
timeout --foreground --signal=TERM --kill-after=5s 60s \
  env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="${n3k_tmp}/cache-object" \
  "${n3k_runner}" --strict-types --emit-object --emit-depfile "${n3k_tmp}/aot_private.d" --no-stdlib \
  "${PROJECT_ROOT}/tests/n3k/aot_private.esk" -o "${n3k_tmp}/aot_private.o" \
  >"${n3k_tmp}/aot-object.stdout" 2>"${n3k_tmp}/aot-object.stderr"
grep -F 'tests/n3k/aot_private.esk' "${n3k_tmp}/aot_private.d" >/dev/null
test ! -s "${n3k_tmp}/aot-object.stderr"
for n3k_run in 1 2; do
  mkdir -p "${n3k_tmp}/cache-${n3k_run}"
  n3k_error_log="${n3k_tmp}/aot-compile-${n3k_run}.stderr"
  timeout --foreground --signal=TERM --kill-after=5s 60s \
    env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="${n3k_tmp}/cache-${n3k_run}" \
    "${n3k_runner}" --strict-types --no-stdlib -L "${n3k_tmp}" --lib eshkol_transformer_n3k_private_aot \
    "${PROJECT_ROOT}/tests/n3k/aot_private.esk" -o "${n3k_tmp}/aot-private-${n3k_run}" \
    >"${n3k_tmp}/aot-compile-${n3k_run}.stdout" 2>"${n3k_tmp}/aot-compile-${n3k_run}.stderr"
  test ! -s "${n3k_tmp}/aot-compile-${n3k_run}.stderr"
  n3k_error_log="${n3k_tmp}/aot-private-${n3k_run}.stderr"
  timeout --foreground --signal=TERM --kill-after=5s 30s "${n3k_tmp}/aot-private-${n3k_run}" \
    >"${n3k_tmp}/aot-private-${n3k_run}.stdout" 2>"${n3k_tmp}/aot-private-${n3k_run}.stderr"
  cmp "${n3k_expect}/aot_private.stdout" "${n3k_tmp}/aot-private-${n3k_run}.stdout"
  test ! -s "${n3k_tmp}/aot-private-${n3k_run}.stderr"
done
cmp "${n3k_tmp}/aot-private-1" "${n3k_tmp}/aot-private-2"

/usr/bin/bash "${PROJECT_ROOT}/scripts/build-n3k.sh" "${n3k_tmp}/sanitized" sanitize
/usr/bin/bash "${PROJECT_ROOT}/scripts/build-i1.sh" "${n3k_tmp}/sanitized-i1" sanitize-test
/usr/bin/bash "${PROJECT_ROOT}/scripts/build-i2.sh" "${n3k_tmp}/sanitized-i2" sanitize-test
n3k_sanitize=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
"${n3k_cc}" "${n3k_cflags[@]}" "${n3k_sanitize[@]}" -c "${PROJECT_ROOT}/native/kernel_abi.c" -o "${n3k_tmp}/kernel-sanitized.o"
for n3k_test in "${n3k_tests[@]}" aot_private_bridge; do
  "${n3k_cc}" "${n3k_cflags[@]}" "${n3k_sanitize[@]}" \
    -DET_F32_TENSOR_TESTING -DET_I64_TENSOR_TESTING -DET_N3K_AOT_BRIDGE_TESTING \
    "${PROJECT_ROOT}/tests/n3k/${n3k_test}.c" \
    "${n3k_tmp}/sanitized/libeshkol_transformer_n3k.a" \
    "${n3k_tmp}/sanitized-i2/libeshkol_transformer_f32.a" \
    "${n3k_tmp}/sanitized-i1/libeshkol_transformer_i64.a" "${n3k_tmp}/kernel-sanitized.o" -lm \
    -o "${n3k_tmp}/${n3k_test}-sanitized"
  n3k_error_log="${n3k_tmp}/${n3k_test}-sanitized.stderr"
  ASAN_OPTIONS=detect_leaks="${N3K_ASAN_DETECT_LEAKS:-1}":halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
    timeout --foreground --signal=TERM --kill-after=5s 60s "${n3k_tmp}/${n3k_test}-sanitized" \
    >"${n3k_tmp}/${n3k_test}-sanitized.stdout" 2>"${n3k_tmp}/${n3k_test}-sanitized.stderr"
  test ! -s "${n3k_tmp}/${n3k_test}-sanitized.stderr"
  if [[ "${n3k_test}" != aot_private_bridge ]]; then
    cmp "${n3k_tmp}/${n3k_test}-1.stdout" "${n3k_tmp}/${n3k_test}-sanitized.stdout"
  fi
done
printf 'N3K PASS: exact native rows/VJPs, independent references, atomicity, I1/I2 borrows, fresh private AOT, ABI/isolation, ASan/UBSan (detect_leaks=%s)\n' \
  "${N3K_ASAN_DETECT_LEAKS:-1}"
