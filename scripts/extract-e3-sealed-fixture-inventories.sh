#!/usr/bin/env bash
# Inactive observation route. It cannot publish the normal fixture object.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
source "${PROJECT_ROOT}/scripts/e3-sealed-fixture-extract-owned-tmp.sh"
[[ $# -eq 2 ]] || die "usage: $0 FRESH_EXTRACTION_DIRECTORY normal|sanitize"
mode=$2
[[ "${mode}" == normal || "${mode}" == sanitize ]] ||
  die "unknown sealed-fixture extraction mode"
destination="$(realpath -m -- "$1")"
[[ ! -e "${destination}" && ! -L "${destination}" ]] ||
  die "sealed-fixture extraction destination must be fresh"
mkdir -p -- "$(dirname -- "${destination}")"
stage="$(mktemp -d "$(dirname -- "${destination}")/.e3-sealed-fixture.XXXXXX")"
cleanup() {
  local result=$?
  if [[ "${result}" != 0 ]]; then
    printf 'sealed-fixture extraction failure retained at %s\n' "${stage}" >&2
  fi
  return "${result}"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
prefix="${PROJECT_ROOT}/tests/e3_private/sealed_fixture_package"
compiler_provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
[[ -f "${compiler_provenance}" && ! -L "${compiler_provenance}" ]] ||
  die "selected compiler provenance is absent or symlinked"
inputs=(
  "${PROJECT_ROOT}/toolchain/eshkol.lock" "${compiler_provenance}"
  "${PROJECT_ROOT}/scripts/build-e1b-consumer.sh"
  "${PROJECT_ROOT}/scripts/e3-private-package-policy.sh"
  "${PROJECT_ROOT}/scripts/e3-private-native-inputs.sh"
  "${PROJECT_ROOT}/scripts/e3-sealed-fixture-generated-source-stage.sh"
  "${PROJECT_ROOT}/scripts/e3-sealed-fixture-extract-owned-tmp.sh"
  "${PROJECT_ROOT}/scripts/generate-e3-sealed-fixture-source.py"
  "${PROJECT_ROOT}/native/p1_identity.c"
  "${PROJECT_ROOT}/native/f32_tensor.c"
  "${PROJECT_ROOT}/src/eshkol_transformer/m3t_transport.c"
  "${PROJECT_ROOT}/src/eshkol_transformer/m3_call_f32_integration.c"
  "${PROJECT_ROOT}/src/eshkol_transformer/e3_frame.c"
  "${PROJECT_ROOT}/tests/e3_private/sealed_fixture_driver_root.esk"
  "${PROJECT_ROOT}/tests/e3_private/sealed_fixture_driver.esk"
  "${PROJECT_ROOT}/tests/e3_private/sealed_fixture_bridge.c"
  "${prefix}_private_renames.txt" "${prefix}_public_exports.txt"
)
for input in "${inputs[@]}"; do
  [[ -f "${input}" && ! -L "${input}" ]] ||
    die "sealed-fixture extraction input is absent or symlinked: ${input}"
done
sha256sum -- "${inputs[@]}" >"${stage}/inputs.before.sha256"
invocation=(
  /usr/bin/bash "${PROJECT_ROOT}/scripts/build-e1b-consumer.sh"
  "${PROJECT_ROOT}/tests/e3_private/sealed_fixture_driver_root.esk"
  "${PROJECT_ROOT}/tests/e3_private/sealed_fixture_bridge.c"
  "${prefix}_private_renames.txt" "${prefix}_public_exports.txt"
  "${stage}/e3_sealed_fixture_package.o"
  "${PROJECT_ROOT}/internal/p1/lib" "${PROJECT_ROOT}/internal/c1/lib"
  "${PROJECT_ROOT}/internal/t1/lib" "${PROJECT_ROOT}/src"
  "${PROJECT_ROOT}/internal/d2/lib" "${PROJECT_ROOT}/internal/e3/lib"
)
if ! e3_extract_owned_command "${stage}" /usr/bin/env \
    E3_SEALED_FIXTURE_EXTRACT=1 \
    E3_SEALED_FIXTURE_SANITIZE="$([[ "${mode}" == sanitize ]] && printf 1 || printf 0)" \
    E1B_COMPILER_TIMEOUT_SECONDS="${E3_COMPILER_TIMEOUT_SECONDS:-900}" \
    "${invocation[@]}"; then
  die "sealed-fixture extraction failed; raw stage preserved"
fi
[[ -d "${stage}/tmp" && ! -L "${stage}/tmp" ]] ||
  die "sealed-fixture builder scratch escaped its owned stage"
rmdir -- "${stage}/tmp" ||
  die "sealed-fixture successful builder left unreviewed scratch"
sha256sum -- "${inputs[@]}" >"${stage}/inputs.after.sha256"
cmp -s "${stage}/inputs.before.sha256" "${stage}/inputs.after.sha256" ||
  die "sealed-fixture extraction inputs changed during emission"
[[ -d "${stage}/e3_sealed_fixture_package.o.extract" &&
   ! -L "${stage}/e3_sealed_fixture_package.o.extract" &&
   ! -e "${stage}/e3_sealed_fixture_package.o" &&
   ! -e "${stage}/e3_sealed_fixture_package.o.evidence" ]] ||
  die "sealed-fixture extraction produced an ordinary package artifact"
cmp -s <(printf '%s\n' \
  'EXTRACTION_ONLY: emitted bytes require separate review; no package admitted') \
  "${stage}/e3_sealed_fixture_package.o.extract/STATUS" ||
  die "sealed-fixture extraction status is not source-only"
python3 -B "${PROJECT_ROOT}/scripts/e3-atomic-publish.py" \
  "${stage}" "${destination}"
printf 'retained unreviewed sealed-fixture emissions: %s\n' "${destination}"
