#!/usr/bin/env bash
# Diagnostic only: exit 1 means the required failure-atomicity contract is broken.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
repo_root="${PROJECT_ROOT}"
eshkol_source="$(eshkol_source_dir)"
eshkol_build="$(eshkol_build_dir)"
provenance="${eshkol_build}/eshkol-transformer-provenance.tsv"
probe_cc="$(tsv_value "${provenance}" cc_path)"
probe_cxx="$(tsv_value "${provenance}" cxx_path)"
resolve_provenance_compilers probe_cc probe_cxx "${probe_cc}" "${probe_cxx}"
evidence="${E3_P1_BARRIER_EVIDENCE:-$(project_build_dir)/e3-p1-runtime-barrier-diagnostic}"
mkdir -p "${evidence}"
git -C "${eshkol_source}" rev-parse HEAD > "${evidence}/runtime-pin.txt"
cp "${provenance}" "${evidence}/toolchain-provenance.tsv"
sha256sum "${eshkol_build}/eshkol-run" "${eshkol_build}/libeshkol-runtime.a" \
  "${provenance}" "${probe_cc}" "${probe_cxx}" > "${evidence}/artifact-hashes.sha256"
: > "${evidence}/commands.sh"
run_recorded() {
  printf '%q ' "$@" >> "${evidence}/commands.sh"
  printf '\n' >> "${evidence}/commands.sh"
  "$@"
}
sha256sum "${eshkol_build}/libeshkol-runtime.a" > "${evidence}/runtime-library.sha256"
run_recorded "${probe_cxx}" -std=c++17 -O2 -I"${eshkol_source}/lib/core" -I"${eshkol_source}/inc" \
  "${repo_root}/tests/e3_p1/runtime_barrier_failure.cpp" \
  "${eshkol_build}/libeshkol-runtime.a" -ldl -lpthread -lm -o "${evidence}/probe"
run_recorded "${eshkol_build}/eshkol-run" --strict-types --no-stdlib --emit-object --dump-ir \
  "${repo_root}/tests/e3_p1/runtime_barrier_publication.esk" -o "${evidence}/publication.o" \
  > "${evidence}/aot-compile.log" 2>&1
run_recorded "${probe_cxx}" -std=c++17 -O2 -I"${eshkol_source}/lib/core" -I"${eshkol_source}/inc" \
  "${repo_root}/tests/e3_p1/runtime_barrier_control.cpp" "${evidence}/publication.o" \
  "${eshkol_build}/libeshkol-runtime.a" -ldl -lpthread -lm -o "${evidence}/publication"
: > "${evidence}/results.txt"
observed=0
for stage in -1 0 1 2 3 4 5; do
  result=0
  run_recorded env ESHKOL_ARENA_POISON=1 "${evidence}/probe" "${stage}" \
    > "${evidence}/stage-${stage}.stdout" 2> "${evidence}/stage-${stage}.stderr" || result=$?
  cat "${evidence}/stage-${stage}.stdout" | tee -a "${evidence}/results.txt"
  printf 'stage=%s exit=%s\n' "${stage}" "${result}" >> "${evidence}/results.txt"
  if [[ "${stage}" == -1 ]]; then
    [[ "${result}" == 0 ]] || exit 2
  elif [[ "${result}" == 1 ]]; then
    observed=$((observed + 1))
  elif [[ "${result}" != 0 ]]; then
    exit 2
  fi
done
for stage in -1 0 1 2 3 4 5; do
  result=0
  run_recorded env ESHKOL_ARENA_POISON=1 E3_P1_BARRIER_STAGE="${stage}" "${evidence}/publication" \
    > "${evidence}/aot-stage-${stage}.stdout" 2> "${evidence}/aot-stage-${stage}.stderr" || result=$?
  cat "${evidence}/aot-stage-${stage}.stdout" | tee -a "${evidence}/results.txt"
  printf 'aot_stage=%s exit=%s\n' "${stage}" "${result}" >> "${evidence}/results.txt"
  if [[ "${stage}" == -1 ]]; then
    [[ "${result}" == 0 ]] || exit 2
  elif [[ "${result}" == 1 ]]; then
    observed=$((observed + 1))
  elif [[ "${result}" != 0 ]]; then
    exit 2
  fi
done
if [[ "${observed}" != 0 ]]; then
  printf 'BLOCKED: %s promotion allocation stages published regional edges; evidence %s\n' "${observed}" "${evidence}" >&2
  exit 1
fi
printf 'No regional edges observed; this diagnostic alone does not prove all allocation-failure semantics.\n'
