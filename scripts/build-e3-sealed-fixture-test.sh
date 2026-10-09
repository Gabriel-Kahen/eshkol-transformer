#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
[[ $# -eq 2 ]] || die "usage: $0 FRESH_ARTIFACT_DIRECTORY normal|sanitize"
mode=$2
[[ "${mode}" == normal || "${mode}" == sanitize ]] || \
  die "unknown E3 sealed-fixture build mode"
# These inventories must come from a separately reviewed exact emitted-source
# extraction. Their absence is a source-only gate, never a guessed allowlist.
prefix="${PROJECT_ROOT}/tests/e3_private/sealed_fixture_package"
for manifest in defined_symbols undefined_symbols public_strings \
    native_objects archive_members source_closure native_source_closure; do
  [[ -f "${prefix}_${manifest}.txt" && ! -L "${prefix}_${manifest}.txt" ]] || \
    die "sealed-fixture emitted manifest is not admitted: ${manifest}"
done
if [[ "${mode}" == sanitize ]]; then
  for manifest in sanitized_undefined_symbols sanitizer_undefined_symbols; do
    [[ -f "${prefix}_${manifest}.txt" && ! -L "${prefix}_${manifest}.txt" ]] || \
      die "sealed-fixture sanitizer manifest is not admitted: ${manifest}"
  done
fi
verify_toolchain
for command in ar cmp; do require_command "${command}"; done
artifact="$(realpath -m -- "$1")"
[[ ! -e "${artifact}" && ! -L "${artifact}" ]] || \
  die "E3 sealed-fixture artifact directory must be fresh"
mkdir -p "$(dirname -- "${artifact}")"
staging="$(mktemp -d "$(dirname -- "${artifact}")/.e3-sealed-fixture.XXXXXX")"
cleanup() {
  local status=$?
  if [[ "${status}" == 0 ]]; then
    rm -rf -- "${staging}"
  else
    printf 'E3 sealed-fixture rejected stage retained at %s\n' "${staging}" >&2
  fi
  return "${status}"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
E3_SEALED_FIXTURE_SANITIZE="$([[ "${mode}" == sanitize ]] && printf 1 || printf 0)" \
E1B_COMPILER_TIMEOUT_SECONDS="${E3_COMPILER_TIMEOUT_SECONDS:-900}" \
  /usr/bin/bash "${PROJECT_ROOT}/scripts/build-e1b-consumer.sh" \
  "${PROJECT_ROOT}/tests/e3_private/sealed_fixture_driver_root.esk" \
  "${PROJECT_ROOT}/tests/e3_private/sealed_fixture_bridge.c" \
  "${prefix}_private_renames.txt" "${prefix}_public_exports.txt" \
  "${staging}/e3_sealed_fixture_package.o" \
  "${PROJECT_ROOT}/internal/p1/lib" "${PROJECT_ROOT}/internal/c1/lib" \
  "${PROJECT_ROOT}/internal/t1/lib" "${PROJECT_ROOT}/src" \
  "${PROJECT_ROOT}/internal/d2/lib" "${PROJECT_ROOT}/internal/e3/lib"
cmp "${prefix}_defined_symbols.txt" \
  "${staging}/e3_sealed_fixture_package.o.evidence/global-defined.txt"
ar rcsD "${staging}/libeshkol_transformer_e3_sealed_fixture.a" \
  "${staging}/e3_sealed_fixture_package.o"
ar t "${staging}/libeshkol_transformer_e3_sealed_fixture.a" | \
  cmp "${prefix}_archive_members.txt" -
python3 "${PROJECT_ROOT}/scripts/e3-atomic-publish.py" "${staging}" "${artifact}"
printf 'built closed E3 sealed-fixture test artifact: %s\n' \
  "${artifact}/libeshkol_transformer_e3_sealed_fixture.a"
