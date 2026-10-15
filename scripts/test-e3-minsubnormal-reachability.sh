#!/usr/bin/env bash
# Noninstalled recipe. Run only after the dedicated private tuple is compiled.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
for command in env grep realpath timeout; do require_command "${command}"; done
[[ $# -eq 1 ]] || die "usage: $0 FRESH_OUTPUT_DIRECTORY"
out="$(realpath -m -- "$1")"
[[ ! -e "${out}" && ! -L "${out}" ]] || die "E3 output must be fresh"
mkdir -p "${out}"
q0_python="${Q0_PYTHON:-}"
[[ "${q0_python}" == /* && -x "${q0_python}" ]] ||
  die "E3 min-subnormal reachability requires absolute pinned Q0_PYTHON"
versions="$(ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE \
  "${q0_python}" -B - <<'PY'
import platform, torch
print(platform.python_version())
print(torch.__version__)
PY
)"
[[ "${versions}" == $'3.14.6\n2.13.0+cpu' ]] ||
  die "E3 min-subnormal oracle versions differ"
ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE "${q0_python}" -B \
  -c 'from pathlib import Path; import sys; from tests.e3_reference.minsubnormal import materialize; materialize(Path(sys.argv[1]))' \
  "${out}/fixture" >"${out}/fixture.stdout" 2>"${out}/fixture.stderr"

"${PROJECT_ROOT}/scripts/build-e3-minsubnormal-test.sh" "${out}/package" \
  >"${out}/package.stdout" 2>"${out}/package.stderr"
prefix="${PROJECT_ROOT}/tests/e3_private/minsubnormal_package"
evidence="${out}/package/e3_minsubnormal_package.o.evidence"
for pair in defined_symbols:global-defined public_exports:package-exports \
    public_strings:public-strings source_closure:source-closure \
    native_source_closure:native-source-closure native_objects:native-objects \
    undefined_symbols:undefined; do
  expected=${pair%%:*}; actual=${pair#*:}
  cmp "${prefix}_${expected}.txt" "${evidence}/${actual}.txt"
done

provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
cxx="$(tsv_value "${provenance}" cxx_path)"
runner="$(eshkol_build_dir)/eshkol-run"
compiler_timeout="${E3_COMPILER_TIMEOUT_SECONDS:-900}"
[[ "${compiler_timeout}" =~ ^[1-9][0-9]*$ ]] || die "invalid compiler timeout"
env -u CPATH -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH -u OBJC_INCLUDE_PATH \
  -u DEPENDENCIES_OUTPUT -u SUNPRO_DEPENDENCIES -u GCC_EXEC_PREFIX \
  -u COMPILER_PATH -u LIBRARY_PATH -u CLANG_CONFIG_FILE \
  -u CCC_OVERRIDE_OPTIONS -u CCC_CC -u CCC_CXX \
  -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
  XDG_CACHE_HOME="${out}/cache" ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" \
  ESHKOL_CXX_COMPILER="${cxx}" \
  timeout --foreground --signal=TERM --kill-after=5s "${compiler_timeout}s" \
  "${runner}" --strict-types --no-stdlib -O 2 \
    -L "${out}/package" --lib eshkol_transformer_e3_minsubnormal \
    "${PROJECT_ROOT}/tests/e3_private/minsubnormal_runtime.esk" \
    -o "${out}/runtime" >"${out}/compile.stdout" 2>"${out}/compile.stderr"

for variant in finite-minsubnormal bias-clear; do
  checkpoint=finite-state.etcp
  [[ "${variant}" != bias-clear ]] || checkpoint=bias-clear.etcp
  timeout --foreground --signal=TERM --kill-after=5s 900s \
    "${out}/runtime" "${variant}" "${out}/fixture/finite-minsubnormal" \
    "${out}/fixture/${checkpoint}" \
    >"${out}/run-${variant}.stdout" 2>"${out}/run-${variant}.stderr"
  [[ ! -s "${out}/run-${variant}.stderr" ]] ||
    die "E3 ${variant} emitted runtime diagnostics"
done

python3 -B -m tests.e3_private.parse_minsubnormal_observation \
  --stdout "${out}/run-finite-minsubnormal.stdout" --batch-count 3 \
  --reference "${out}/fixture/reference.json" \
  --observed "${out}/observed.json" --roles "${out}/roles.tsv"
ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE "${q0_python}" -B \
  -m tests.e3_native_parity.verify_observation \
  --reference "${out}/fixture/reference.json" --case finite-minsubnormal \
  --observed "${out}/observed.json" --roles "${out}/roles.tsv" \
  --transcript "${out}/transcript.json" \
  >"${out}/verify.stdout" 2>"${out}/verify.stderr"
if python3 -B -m tests.e3_private.parse_minsubnormal_observation \
  --stdout "${out}/run-bias-clear.stdout" --batch-count 3 \
  --reference "${out}/fixture/reference.json" \
  --observed "${out}/mutant-observed.json" --roles "${out}/mutant-roles.tsv" \
  >"${out}/mutant-parse.stdout" 2>"${out}/mutant-parse.stderr"; then
  die "bias-clear mutant survived the first physical logit"
fi
[[ ! -e "${out}/mutant-observed.json" ]] || die "mutant published observation"
grep -F 'first physical logit differs' "${out}/mutant-parse.stderr" >/dev/null ||
  die "bias-clear mutant failed outside the first-logit gate"
printf 'E3 finite min-subnormal reachability and bias-clear rejection passed\n'
