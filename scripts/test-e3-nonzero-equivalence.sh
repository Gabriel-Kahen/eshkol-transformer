#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
for command in ar cmp diff env grep python3 realpath strings timeout; do
  require_command "${command}"
done
[[ $# -le 1 ]] || die "usage: $0 [FRESH_OUTPUT_DIRECTORY]"
out="$(realpath -m -- "${1:-$(project_build_dir)/e3-nonzero-equivalence}")"
[[ ! -e "${out}" && ! -L "${out}" ]] || die "nonzero output must be fresh"
mkdir -p "${out}"
q0_python="${Q0_PYTHON:-}"
[[ "${q0_python}" == /* && -x "${q0_python}" ]] ||
  die "nonzero equivalence requires absolute pinned Q0_PYTHON"
versions="$(ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE "${q0_python}" -B - <<'PY'
import platform, torch
print(platform.python_version())
print(torch.__version__)
PY
)"
[[ "${versions}" == $'3.14.6\n2.13.0+cpu' ]] || die "Q0 versions differ"

for copy in a b; do
  ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE "${q0_python}" -B \
    -m tests.e3_reference.nonzero_equivalence \
    --output "${out}/fixture-${copy}" \
    --reference "${out}/reference-${copy}.json" \
    >"${out}/reference-${copy}.stdout" \
    2>"${out}/reference-${copy}.stderr"
done
diff -r "${out}/fixture-a" "${out}/fixture-b"
cmp "${out}/reference-a.json" "${out}/reference-b.json"

prefix="${PROJECT_ROOT}/tests/e3_private/uniform_package"
variants=(a b two-term-reversal bool-fma weight-from-tokens f64-add-cast)
for variant in "${variants[@]}"; do
  mode="${variant}"
  if [[ "${variant}" == a || "${variant}" == b ]]; then mode=normal; fi
  "${PROJECT_ROOT}/scripts/build-e3-uniform-test.sh" \
    "${out}/package-${variant}" "${mode}" \
    >"${out}/package-${variant}.stdout" \
    2>"${out}/package-${variant}.stderr"
  evidence="${out}/package-${variant}/e3_uniform_package.o.evidence"
  for pair in defined_symbols:global-defined public_exports:package-exports \
      public_strings:public-strings source_closure:source-closure \
      native_source_closure:native-source-closure native_objects:native-objects; do
    expected=${pair%%:*}
    actual=${pair#*:}
    cmp "${prefix}_${expected}.txt" "${evidence}/${actual}.txt"
  done
  undefined="${prefix}_undefined_symbols.txt"
  if [[ "${variant}" == bool-fma ]]; then
    undefined="${PROJECT_ROOT}/tests/e3_private/uniform_bool_fma_undefined_symbols.txt"
  fi
  cmp "${undefined}" "${evidence}/undefined.txt"
  grep -F 'E3ROW' "${evidence}/strings.txt" >/dev/null ||
    die "${variant} omitted the E3 frame observer"
done
cmp "${out}/package-a/e3_uniform_package.o" \
    "${out}/package-b/e3_uniform_package.o"
cmp "${out}/package-a/libeshkol_transformer_e3_uniform.a" \
    "${out}/package-b/libeshkol_transformer_e3_uniform.a"

provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
cxx="$(tsv_value "${provenance}" cxx_path)"
runner="$(eshkol_build_dir)/eshkol-run"
compiler_timeout="${E3_COMPILER_TIMEOUT_SECONDS:-900}"
[[ "${compiler_timeout}" =~ ^[1-9][0-9]*$ ]] || die "compiler timeout differs"
for variant in "${variants[@]}"; do
  env -u CPATH -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH -u OBJC_INCLUDE_PATH \
    -u DEPENDENCIES_OUTPUT -u SUNPRO_DEPENDENCIES -u GCC_EXEC_PREFIX \
    -u COMPILER_PATH -u LIBRARY_PATH -u CLANG_CONFIG_FILE \
    -u CCC_OVERRIDE_OPTIONS -u CCC_CC -u CCC_CXX \
    -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
    XDG_CACHE_HOME="${out}/cache-${variant}" ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" \
    ESHKOL_CXX_COMPILER="${cxx}" \
    timeout --foreground --signal=TERM --kill-after=5s "${compiler_timeout}s" \
    "${runner}" --strict-types --no-stdlib -O 2 \
      -L "${out}/package-${variant}" --lib eshkol_transformer_e3_uniform \
      "${PROJECT_ROOT}/tests/e3_private/uniform_runtime.esk" \
      -o "${out}/runtime-${variant}" \
      >"${out}/compile-${variant}.stdout" \
      2>"${out}/compile-${variant}.stderr"
done
cmp "${out}/runtime-a" "${out}/runtime-b"

for variant in "${variants[@]}"; do
  ESHKOL_ARENA_POISON=0 \
    timeout --foreground --signal=TERM --kill-after=5s 900s \
    "${out}/runtime-${variant}" nonzero-equivalence \
    "${out}/fixture-a/nonzero-equivalence" \
    "${out}/fixture-a/zero-state.etcp" \
    "${out}/fixture-a/bad-checksum.etcp" \
    "${out}/fixture-a/bad-provider.etcp" \
    "${out}/fixture-a/bad-shape.etcp" \
    "${out}/fixture-a/bad-tie.etcp" \
    >"${out}/run-${variant}.stdout" \
    2>"${out}/run-${variant}.stderr"
  [[ ! -s "${out}/run-${variant}.stderr" ]] ||
    die "${variant} emitted runtime diagnostics"
  python3 -B -m tests.e3_private.parse_uniform_observation \
    --stdout "${out}/run-${variant}.stdout" --batch-count 2 \
    --case nonzero-equivalence --reference "${out}/reference-a.json" \
    --observed "${out}/observed-${variant}.json" \
    --roles "${out}/roles-${variant}.tsv"
  ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE "${q0_python}" -B \
    -m tests.e3_native_parity.verify_observation \
    --reference "${out}/reference-a.json" --case nonzero-equivalence \
    --observed "${out}/observed-${variant}.json" \
    --roles "${out}/roles-${variant}.tsv" \
    --transcript "${out}/transcript-${variant}.json" \
    >"${out}/verify-${variant}.stdout" \
    2>"${out}/verify-${variant}.stderr"
done
for variant in "${variants[@]:1}"; do
  cmp "${out}/run-a.stdout" "${out}/run-${variant}.stdout"
  cmp "${out}/observed-a.json" "${out}/observed-${variant}.json"
  cmp "${out}/roles-a.tsv" "${out}/roles-${variant}.tsv"
  cmp "${out}/transcript-a.json" "${out}/transcript-${variant}.json"
done
printf 'E3 nonzero private equivalent controls passed: %s\n' "${variants[*]}"
