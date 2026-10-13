#!/usr/bin/env bash
# Emits evidence only. Never builds an admitted object/archive or runs the tuple.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
[[ $# == 0 && -n "${RUNNER_TEMP:-}" && -d "${RUNNER_TEMP}" &&
   ! -L "${RUNNER_TEMP}" && -O "${RUNNER_TEMP}" ]] ||
  die "E3 zero discovery requires an owned RUNNER_TEMP and no arguments"
root="${RUNNER_TEMP}/e3-n3-zero-discovery"
[[ -d "${root}" && ! -L "${root}" && -O "${root}" &&
   -f "${root}/status.tsv" && ! -L "${root}/status.tsv" ]] ||
  die "E3 zero discovery requires a reserved evidence directory"
[[ ! -e "${root}/normal" && ! -L "${root}/normal" &&
   ! -e "${root}/sanitize" && ! -L "${root}/sanitize" ]] ||
  die "E3 zero discovery stage must be fresh"
finish() {
  local status=$?
  trap - EXIT TERM INT
  printf 'harness_exit\t%s\n' "${status}" >>"${root}/status.tsv"
  (cd -- "${root}" && find . -type f ! -path './SHA256SUMS' -print0 | \
    LC_ALL=C sort -z | xargs -0 -r sha256sum) >"${root}/SHA256SUMS" || true
  exit "${status}"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
preflight() {
  local mode=$1 memory_kib disk_bytes
  memory_kib="$(awk '$1 == "MemAvailable:" { print $2 }' /proc/meminfo)"
  disk_bytes="$(df -B1 --output=avail "${RUNNER_TEMP}" | tail -n 1 | tr -d '[:space:]')"
  [[ "${memory_kib}" =~ ^[0-9]+$ && "${disk_bytes}" =~ ^[0-9]+$ ]] ||
    die "E3 zero discovery cannot read host resources"
  printf '%s\tmemory_available_kib\t%s\ndisk_available_bytes\t%s\n' \
    "${mode}" "${memory_kib}" "${disk_bytes}" >>"${root}/resources.tsv"
  (( memory_kib >= 8 * 1024 * 1024 && disk_bytes >= 6 * 1024 * 1024 * 1024 )) ||
    die "E3 zero discovery host resources below 8 GiB memory / 6 GiB scratch"
}
preflight toolchain
verify_toolchain
printf 'head\t%s\ntree\t%s\n' "$(git -C "${PROJECT_ROOT}" rev-parse HEAD)" \
  "$(git -C "${PROJECT_ROOT}" rev-parse HEAD^{tree})" >>"${root}/status.tsv"
git -C "${PROJECT_ROOT}" status --porcelain --untracked-files=all \
  >"${root}/source-status.txt"
[[ ! -s "${root}/source-status.txt" ]] || die "E3 zero discovery source checkout is dirty"
sha256sum "${PROJECT_ROOT}/toolchain/eshkol.lock" \
  "${PROJECT_ROOT}/tests/e3_private/zero_control_p1_overlay_manifest.json" \
  "${PROJECT_ROOT}/tests/e3_private/asan_ir_sources.sha256" \
  "$(eshkol_build_dir)/eshkol-run" \
  "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" \
  >"${root}/inputs.sha256"
prefix="${PROJECT_ROOT}/tests/e3_private/zero_control_package"
for mode in normal sanitize; do
  preflight "${mode}"
  gate="${root}/${mode}"
  mkdir -m 0700 -- "${gate}"
  mkdir -m 0700 -- "${gate}/evidence"
  stage="$(mktemp -d "${gate}/.e3-zero-control.XXXXXX")"
  object="${stage}/e3_zero_control_package.o"
  args=("${PROJECT_ROOT}/tests/e3_private/zero_control_driver_root.esk"
        "${PROJECT_ROOT}/tests/e3_private/zero_control_bridge.c"
        "${prefix}_private_renames.txt" "${prefix}_public_exports.txt"
        "${object}" "${PROJECT_ROOT}/internal/p1/lib"
        "${PROJECT_ROOT}/internal/c1/lib" "${PROJECT_ROOT}/internal/t1/lib"
        "${PROJECT_ROOT}/src" "${PROJECT_ROOT}/internal/d2/lib"
        "${PROJECT_ROOT}/internal/e3/lib")
  printf '%q ' /usr/bin/bash "${PROJECT_ROOT}/scripts/build-e1b-consumer.sh" \
    "${args[@]}" >"${gate}/command.txt"
  printf '\n' >>"${gate}/command.txt"
  printf 'mode\t%s\nE3_ZERO_CONTROL_DISCOVERY\t1\nE3_ZERO_CONTROL_SANITIZE\t%s\n' \
    "${mode}" "$([[ "${mode}" == sanitize ]] && printf 1 || printf 0)" \
    >"${gate}/environment.tsv"
  E3_ZERO_CONTROL_DISCOVERY=1 \
  E3_ZERO_CONTROL_SANITIZE="$([[ "${mode}" == sanitize ]] && printf 1 || printf 0)" \
  E1B_COMPILER_TIMEOUT_SECONDS="${E3_COMPILER_TIMEOUT_SECONDS:-900}" \
    /usr/bin/bash "${PROJECT_ROOT}/scripts/build-e1b-consumer.sh" "${args[@]}" \
    >"${gate}/builder.stdout" 2>"${gate}/builder.stderr"
  [[ ! -e "${object}" && ! -L "${object}" &&
     "$(awk -F '\t' '$1 == "result" { print $2 }' "${gate}/evidence/builder-status.tsv")" == success ]] ||
    die "E3 zero discovery unexpectedly published an object or omitted evidence"
done
printf 'discovery\tcomplete-no-admission\n' >>"${root}/status.tsv"
