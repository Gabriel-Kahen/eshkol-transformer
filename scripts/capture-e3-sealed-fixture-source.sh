#!/usr/bin/env bash
# One compiler invocation for the noninstalled sealed-M3 source inventory.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
[[ $# -eq 1 ]] || die "usage: $0 FRESH_EVIDENCE_DIRECTORY"
out="$(realpath -m -- "$1")"
[[ ! -e "${out}" && ! -L "${out}" ]] || die "evidence directory must be fresh"
mkdir -p -- "${out}"
trap 'status=$?; printf "%s\n" "${status}" >"${out}/capture.exit"; exit "${status}"' EXIT
verify_toolchain
require_command timeout
require_command sha256sum
git -C "${PROJECT_ROOT}" rev-parse HEAD HEAD^{tree} >"${out}/source-pin.txt"
git -C "$(eshkol_source_dir)" rev-parse HEAD >"${out}/runtime-pin.txt"
sha256sum "$(eshkol_build_dir)/eshkol-run" \
  "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" >"${out}/toolchain.sha256"
generated="${out}/.e3-sealed-fixture-generated"
python3 -B "${PROJECT_ROOT}/scripts/generate-e3-d2-source.py" \
  --output-dir "${generated}" >"${out}/d2-generation.json"
overlay="${generated}/sealed-overlay"
python3 -B "${PROJECT_ROOT}/scripts/generate-e3-sealed-fixture-source.py" \
  --output-dir "${overlay}" >"${out}/overlay-generation.txt"
python3 -B "${PROJECT_ROOT}/scripts/generate-e3-sealed-fixture-source.py" \
  --output-dir "${overlay}" --check >"${out}/overlay-check.txt"
cmp "${out}/overlay-generation.txt" "${out}/overlay-check.txt"
source_root="${generated}/source"
mkdir -p -- "${source_root}/transformer" "${out}/cache" "${out}/compiler"
for member in transformer/module.esk i2_wave2_root.esk m3t_transport_extension.esk; do
  cp -- "${overlay}/${member}" "${source_root}/${member}"
  cmp "${overlay}/${member}" "${source_root}/${member}"
done
provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
cxx="$(tsv_value "${provenance}" cxx_path)"
timeout_seconds="${E3_COMPILER_TIMEOUT_SECONDS:-900}"
[[ "${timeout_seconds}" =~ ^[1-9][0-9]*$ ]] || die "invalid compiler timeout"
(
  cd "${out}/compiler"
  env -u CPATH -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH -u OBJC_INCLUDE_PATH \
    -u DEPENDENCIES_OUTPUT -u SUNPRO_DEPENDENCIES -u GCC_EXEC_PREFIX \
    -u COMPILER_PATH -u LIBRARY_PATH -u CLANG_CONFIG_FILE \
    -u CCC_OVERRIDE_OPTIONS -u CCC_CC -u CCC_CXX -u ESHKOL_PATH \
    -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
    XDG_CACHE_HOME="${out}/cache" ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" \
    ESHKOL_CXX_COMPILER="${cxx}" \
    timeout --foreground --signal=TERM --kill-after=5s \
      "${timeout_seconds}s" "$(eshkol_build_dir)/eshkol-run" \
      --strict-types --no-stdlib \
      -I "${source_root}" \
      -I "${PROJECT_ROOT}/internal/p1/lib" \
      -I "${PROJECT_ROOT}/internal/c1/lib" \
      -I "${PROJECT_ROOT}/internal/t1/lib" \
      -I "${PROJECT_ROOT}/src" \
      -I "${PROJECT_ROOT}/internal/d2/lib" \
      -I "${PROJECT_ROOT}/internal/e3/lib" \
      -I "${source_root}" -I "${PROJECT_ROOT}/lib" -I "${PROJECT_ROOT}/native" \
      --shared-lib --dump-ir --emit-depfile "${out}/private.d" \
      "${PROJECT_ROOT}/tests/e3_private/sealed_fixture_driver_root.esk" \
      -o private >"${out}/compiler.stdout" 2>"${out}/compiler.stderr"
)
[[ -s "${out}/private.d" && -s "${out}/compiler/private.ll" ]] || \
  die "compiler did not emit depfile and IR"
python3 -B "${PROJECT_ROOT}/scripts/extract-e3-sealed-fixture-inventory.py" \
  source "${PROJECT_ROOT}" "${source_root}" "${out}/private.d" \
  "${out}/observed-source-closure.txt"
printf 'sealed-fixture raw depfile captured once: %s\n' "${out}/private.d"
