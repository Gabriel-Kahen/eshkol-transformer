#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
[[ $# -le 1 ]] || die "usage: $0 [EVIDENCE_DIR]"
for command in ar cmp find realpath sha256sum sort xargs; do require_command "${command}"; done
isolation_evidence="${1:-$(project_build_dir)/g3g-g0-isolation}"
mkdir -p "${isolation_evidence}"
v1="$(project_build_dir)/g3g"
v2="$(project_build_dir)/g3g-g0"
[[ "$(realpath -m -- "${v1}")" != "$(realpath -m -- "${v2}")" ]] || \
  die "G3-G revisions share a default artifact directory"
/usr/bin/bash "${PROJECT_ROOT}/scripts/build-g3g.sh" \
  >"${isolation_evidence}/v1-build.stdout" 2>"${isolation_evidence}/v1-build.stderr"
(
  cd "${v1}"
  find facades -type f -print0 | sort -z | xargs -0 sha256sum
  sha256sum g3g_package.o libeshkol_transformer_g3g.a
) >"${isolation_evidence}/v1-before.sha256"
/usr/bin/bash "${PROJECT_ROOT}/scripts/build-g3g-g0.sh" \
  >"${isolation_evidence}/v2-build.stdout" 2>"${isolation_evidence}/v2-build.stderr"
(
  cd "${v1}"
  find facades -type f -print0 | sort -z | xargs -0 sha256sum
  sha256sum g3g_package.o libeshkol_transformer_g3g.a
) >"${isolation_evidence}/v1-after.sha256"
cmp "${isolation_evidence}/v1-before.sha256" \
    "${isolation_evidence}/v1-after.sha256"
[[ ! -e "${v1}/libeshkol_transformer_g3g_g0.a" && \
   ! -e "${v2}/libeshkol_transformer_g3g.a" ]] || \
  die "G3-G revision archives mixed in one installation"
ar t "${v1}/libeshkol_transformer_g3g.a" >"${isolation_evidence}/v1-members.txt"
ar t "${v2}/libeshkol_transformer_g3g_g0.a" \
  >"${isolation_evidence}/v2-members.txt"
cmp "${PROJECT_ROOT}/native/g3g_package_archive_members.txt" \
    "${isolation_evidence}/v1-members.txt"
cmp "${PROJECT_ROOT}/native/g3g_g0_package_archive_members.txt" \
    "${isolation_evidence}/v2-members.txt"
printf 'G3-G default revision isolation: PASS\n'
