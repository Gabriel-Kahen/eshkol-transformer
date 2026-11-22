#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

[[ $# == 1 && "$1" == /* ]] || die "usage: $0 /absolute/new/evidence-directory"
evidence=$1
[[ ! -e "${evidence}" && ! -L "${evidence}" ]] || \
  die "DATA4 installed evidence path already exists"
evidence="$(realpath -m -- "${evidence}")"
project_real="$(realpath -- "${PROJECT_ROOT}")"
[[ "${evidence}" != "${project_real}" && "${evidence}" != "${project_real}/"* ]] || \
  die "DATA4 installed evidence must be outside checkout"
mkdir -m 0700 -- "${evidence}"
mkdir -- "${evidence}/steps"
export TMPDIR="${evidence}"

finish() {
  local status=$?
  trap - EXIT
  git -C "${PROJECT_ROOT}" rev-parse HEAD >"${evidence}/final-head.txt" || status=1
  git -C "${PROJECT_ROOT}" rev-parse 'HEAD^{tree}' >"${evidence}/final-tree.txt" || status=1
  git -C "${PROJECT_ROOT}" status --porcelain --untracked-files=all \
    >"${evidence}/final-status.txt" || status=1
  [[ ! -s "${evidence}/final-status.txt" ]] || status=1
  cmp -s "${evidence}/head.txt" "${evidence}/final-head.txt" || status=1
  cmp -s "${evidence}/tree.txt" "${evidence}/final-tree.txt" || status=1
  printf '%s\n' "${status}" >"${evidence}/overall.status"
  (cd "${evidence}" && find . -type f ! -name SHA256SUMS -print0 | \
    LC_ALL=C sort -z | xargs -0 -r sha256sum) >"${evidence}/SHA256SUMS"
  exit "${status}"
}
trap finish EXIT

step() {
  local name=$1 status=0
  shift
  "$@" >"${evidence}/steps/${name}.stdout" \
    2>"${evidence}/steps/${name}.stderr" || status=$?
  printf '%s\n' "${status}" >"${evidence}/steps/${name}.status"
  return "${status}"
}

for command in cc cmp find git python3 realpath sha256sum sort xargs; do
  require_command "${command}"
done
git -C "${PROJECT_ROOT}" rev-parse HEAD >"${evidence}/head.txt"
git -C "${PROJECT_ROOT}" rev-parse 'HEAD^{tree}' >"${evidence}/tree.txt"
git -C "${PROJECT_ROOT}" status --porcelain --untracked-files=all \
  >"${evidence}/initial-status.txt"
[[ ! -s "${evidence}/initial-status.txt" ]] || die "checkout is dirty"
sha256sum "${LOCK_FILE}" >"${evidence}/lock.sha256"
step verify-toolchain verify_toolchain
runner="$(eshkol_build_dir)/eshkol-run"
provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
sha256sum "${runner}" "${provenance}" >"${evidence}/toolchain.sha256"
step prerequisite-plan /usr/bin/bash "${PROJECT_ROOT}/scripts/ci-build-prerequisites.sh" \
  --plan cli3-commands
step configure /usr/bin/bash "${PROJECT_ROOT}/scripts/configure.sh"
step data4-source /usr/bin/bash "${PROJECT_ROOT}/scripts/test-data4-source.sh"

scratch="${RUNNER_TEMP:-${TMPDIR:-/tmp}}/data4-installed-build-${GITHUB_RUN_ID:-local}"
[[ ! -e "${scratch}" && ! -L "${scratch}" ]] || die "scratch already exists"
mkdir -m 0700 -- "${scratch}"
printf '%s\n' "${scratch}" >"${evidence}/scratch-path.txt"
step build-cli3 env CLI3_COMPILER_TIMEOUT_SECONDS=1200 \
  /usr/bin/bash "${PROJECT_ROOT}/scripts/build-cli3.sh" "${scratch}"

check_package() {
  local name
  for name in source-closure native-source-closure public-strings \
              package-exports undefined; do
    local manifest
    case "${name}" in
      source-closure) manifest=source_closure ;;
      native-source-closure) manifest=native_source_closure ;;
      public-strings) manifest=public_strings ;;
      package-exports) manifest=public_exports ;;
      undefined) manifest=undefined_symbols ;;
    esac
    cmp "${PROJECT_ROOT}/native/cli3_generate_${manifest}.txt" \
      "${scratch}/cli3.o.evidence/${name}.txt"
    cp "${scratch}/cli3.o.evidence/${name}.txt" \
      "${evidence}/${name}.txt"
  done
  sha256sum "${scratch}/cli3.o" \
    "${scratch}/libeshkol_transformer_cli3.a" \
    "${scratch}/eshkol-transformer" >"${evidence}/package.sha256"
}
step package-manifests check_package

cat >"${scratch}/config.json" <<'JSON'
{"config-schema-major":1,"config-schema-minor":0,"model.context-length":2,"model.hidden-size":4,"model.layer-count":1,"model.query-head-count":2,"model.vocabulary-size":256,"run.seed":1729,"training.accumulation-steps":2}
JSON
step tokenizer "${scratch}/eshkol-transformer" tokenizer byte \
  --config "${scratch}/config.json" --output "${scratch}/byte.t1"
step fault-shim cc -std=c11 -Wall -Wextra -Werror -fPIC -shared \
  "${PROJECT_ROOT}/tests/cli3/write_fault.c" -ldl \
  -o "${scratch}/data4-write-fault.so"
step installed-data4 env \
  CLI3_DATA4_BINARY="${scratch}/eshkol-transformer" \
  CLI3_DATA4_TOKENIZER="${scratch}/byte.t1" \
  CLI3_DATA4_FAULT_SHIM="${scratch}/data4-write-fault.so" \
  PYTHONDONTWRITEBYTECODE=1 \
  python3 -m unittest -v tests.cli3.test_installed_data4
