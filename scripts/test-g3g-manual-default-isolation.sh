#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
[[ $# -le 1 ]] || die "usage: $0 [EVIDENCE_DIR]"
for command in ar cmp find realpath sha256sum sort xargs; do require_command "${command}"; done
isolation_evidence="${1:-$(project_build_dir)/g3g-manual-isolation}"
mkdir -p "${isolation_evidence}"
v1="$(project_build_dir)/g3g"
v2="$(project_build_dir)/g3g-g0"
v3="$(project_build_dir)/g3g-manual"
[[ "$(realpath -m -- "${v1}")" != "$(realpath -m -- "${v2}")" &&
   "$(realpath -m -- "${v1}")" != "$(realpath -m -- "${v3}")" &&
   "$(realpath -m -- "${v2}")" != "$(realpath -m -- "${v3}")" ]] ||
  die "G3-G revisions share a default artifact directory"
/usr/bin/bash "${PROJECT_ROOT}/scripts/test-g3g-g0-default-isolation.sh" \
  "${isolation_evidence}/predecessors" \
  >"${isolation_evidence}/predecessors.stdout" \
  2>"${isolation_evidence}/predecessors.stderr"
for revision in v1 v2; do
  directory="${!revision}"
  (cd "${directory}";
    find facades -type f -print0 | sort -z | xargs -0 sha256sum;
    if [[ "${revision}" == v1 ]]; then
      sha256sum g3g_package.o libeshkol_transformer_g3g.a
    else
      sha256sum g3g_g0_package.o libeshkol_transformer_g3g_g0.a
    fi) >"${isolation_evidence}/${revision}-before.sha256"
done
/usr/bin/bash "${PROJECT_ROOT}/scripts/build-g3g-manual.sh" \
  >"${isolation_evidence}/v3-build.stdout" \
  2>"${isolation_evidence}/v3-build.stderr"
for revision in v1 v2; do
  directory="${!revision}"
  (cd "${directory}";
    find facades -type f -print0 | sort -z | xargs -0 sha256sum;
    if [[ "${revision}" == v1 ]]; then
      sha256sum g3g_package.o libeshkol_transformer_g3g.a
    else
      sha256sum g3g_g0_package.o libeshkol_transformer_g3g_g0.a
    fi) >"${isolation_evidence}/${revision}-after.sha256"
  cmp "${isolation_evidence}/${revision}-before.sha256" \
      "${isolation_evidence}/${revision}-after.sha256"
done
for revision in v1 v2 v3; do
  directory="${!revision}"
  case "${revision}" in
    v1) stem=g3g ;;
    v2) stem=g3g_g0 ;;
    v3) stem=g3g_manual ;;
  esac
  ar t "${directory}/libeshkol_transformer_${stem}.a" \
    >"${isolation_evidence}/${revision}-members.txt"
  cmp "${PROJECT_ROOT}/native/${stem}_package_archive_members.txt" \
      "${isolation_evidence}/${revision}-members.txt"
done
[[ ! -e "${v1}/libeshkol_transformer_g3g_manual.a" &&
   ! -e "${v2}/libeshkol_transformer_g3g_manual.a" &&
   ! -e "${v3}/libeshkol_transformer_g3g.a" &&
   ! -e "${v3}/libeshkol_transformer_g3g_g0.a" ]] ||
  die "G3-G revision archives mixed in one installation"
printf 'G3-G manual default revision isolation: PASS\n'
