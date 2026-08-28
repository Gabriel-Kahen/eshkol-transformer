#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

[[ $# == 1 ]] || die "usage: compile-smoke.sh OUTPUT_DIRECTORY"
verify_toolchain
output_dir="$1"
mkdir -p "${output_dir}"
binary="${output_dir}/eshkol-transformer-smoke"
object="${output_dir}/eshkol-transformer-smoke.o"
depfile="${output_dir}/eshkol-transformer-smoke.d"
"$(eshkol_build_dir)/eshkol-run" \
  --no-stdlib \
  -I "${PROJECT_ROOT}/src" \
  --emit-depfile "${depfile}" \
  --compile-only \
  "${PROJECT_ROOT}/tests/smoke.esk" \
  -o "${object}"
"$(eshkol_build_dir)/eshkol-run" \
  --no-stdlib \
  -I "${PROJECT_ROOT}/src" \
  "${PROJECT_ROOT}/tests/smoke.esk" \
  -o "${binary}"
[[ -s "${object}" ]] || die "Eshkol did not emit compile-only object ${object}"
[[ -x "${binary}" ]] || die "Eshkol reported success but did not emit ${binary}"
[[ -s "${depfile}" ]] || die "Eshkol did not emit dependency evidence at ${depfile}"
grep -F 'src/eshkol_transformer/smoke.esk' "${depfile}" >/dev/null || \
  die "dependency file does not prove compilation of src/eshkol_transformer/smoke.esk"
