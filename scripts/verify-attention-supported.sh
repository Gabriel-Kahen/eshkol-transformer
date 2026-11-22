#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

[[ ! ${ESHKOL_ALLOW_UNSUPPORTED_HOST+x} ]] || \
  die "supported attention measurement forbids ESHKOL_ALLOW_UNSUPPORTED_HOST"
verify_toolchain
resolve_provenance_compilers supported_cc supported_cxx \
  "${1:-$(lock_value supported_cc)}" "${CXX:-$(lock_value supported_cxx)}"

provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
printf 'os_id\t%s\n' "$(. /etc/os-release; printf '%s' "${ID:-unknown}")"
printf 'os_version_id\t%s\n' "$(. /etc/os-release; printf '%s' "${VERSION_ID:-unknown}")"
printf 'architecture\t%s\n' "$(uname -m)"
printf 'cc_path\t%s\n' "${supported_cc}"
printf 'cc_version\t%s\n' "$(tsv_value "${provenance}" cc_version)"
printf 'llvm_version\t%s\n' "$("$(llvm_config)" --version)"
printf 'eshkol_repository\t%s\n' "$(lock_value eshkol_repository)"
printf 'eshkol_commit\t%s\n' "$(lock_value eshkol_commit)"
printf 'eshkol_version\t%s\n' "$(lock_value eshkol_version)"
printf 'eshkol_binary_sha256\t%s\n' "$(tsv_value "${provenance}" eshkol_binary_sha256)"
printf 'lock_sha256\t%s\n' "$(sha256sum "${LOCK_FILE}" | awk '{print $1}')"
printf 'provenance_sha256\t%s\n' "$(sha256sum "${provenance}" | awk '{print $1}')"
