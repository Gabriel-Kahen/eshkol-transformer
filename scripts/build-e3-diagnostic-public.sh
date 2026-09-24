#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
[[ $# -le 1 ]] || die "usage: $0 [ARTIFACT_DIR]"
verify_toolchain
for command in ar cmp python3; do require_command "${command}"; done
artifact="${1:-$(project_build_dir)/e3-diagnostic}"
mkdir -p "$(dirname -- "${artifact}")"
temporary="$(mktemp -d "$(dirname -- "${artifact}")/.e3-diagnostic-build.XXXXXX")"
trap 'rm -rf -- "${temporary}"' EXIT
prefix="${PROJECT_ROOT}/native/e3_diagnostic_public_package"
E1B_COMPILER_TIMEOUT_SECONDS="${E3_COMPILER_TIMEOUT_SECONDS:-900}" \
  /usr/bin/bash "${PROJECT_ROOT}/scripts/build-e1b-consumer.sh" \
  "${PROJECT_ROOT}/native/e3_diagnostic_public_root.esk" \
  "${PROJECT_ROOT}/native/e3_diagnostic_public_bridge.c" \
  "${prefix}_private_renames.txt" "${prefix}_public_exports.txt" \
  "${temporary}/e3_diagnostic_public_package.o" \
  "${PROJECT_ROOT}/internal/p1/lib" "${PROJECT_ROOT}/internal/c1/lib" \
  "${PROJECT_ROOT}/internal/t1/lib" "${PROJECT_ROOT}/src" \
  "${PROJECT_ROOT}/internal/d2/lib" "${PROJECT_ROOT}/internal/e3/lib"
cmp "${prefix}_defined_symbols.txt" \
  "${temporary}/e3_diagnostic_public_package.o.evidence/global-defined.txt"
ar rcsD "${temporary}/libeshkol_transformer_e3_diagnostic.a" \
  "${temporary}/e3_diagnostic_public_package.o"
ar t "${temporary}/libeshkol_transformer_e3_diagnostic.a" \
  >"${temporary}/archive-members.txt"
cmp "${prefix}_archive_members.txt" "${temporary}/archive-members.txt"
cp "${temporary}/e3_diagnostic_public_package.o.evidence/d2-generation.json" \
  "${temporary}/d2-generation.json"
mkdir -p "${temporary}/facades/transformer"
cp "${prefix}_facades.txt" "${temporary}/facades.txt"
while IFS= read -r facade; do
  source="${PROJECT_ROOT}/lib/${facade}"
  [[ "$(realpath -- "${source}")" == "${source}" ]] || \
    die "E3 installation rejects symlinked public facade inputs"
  cp "${source}" "${temporary}/facades/${facade}"
done <"${temporary}/facades.txt"
python3 "${PROJECT_ROOT}/scripts/e3-atomic-publish.py" \
  "${temporary}" "${artifact}"
printf 'built installed E3 diagnostic candidate: %s\n' \
  "${artifact}/libeshkol_transformer_e3_diagnostic.a"
