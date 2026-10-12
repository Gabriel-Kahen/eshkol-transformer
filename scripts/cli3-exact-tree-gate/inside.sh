#!/usr/bin/env bash
set -euo pipefail
source /workspace/scripts/common.sh

pin() {
  awk -F '\t' -v key="$1" '$1 == key { value=$2; count++ }
       END { if (count == 1 && value != "") print value; else exit 1 }' /gate/pins.tsv
}
step() {
  local name=$1 limit=$2 status
  shift 2
  if /usr/bin/time -v -o "/out/steps/${name}.time" \
      timeout --foreground --signal=TERM --kill-after=30s "${limit}s" "$@" \
      >"/out/steps/${name}.stdout" 2>"/out/steps/${name}.stderr"; then
    status=0
  else
    status=$?
  fi
  printf '%s\n' "${status}" >"/out/steps/${name}.status"
  (( status == 0 )) || exit "${status}"
}
check_pins() {
  [[ "$(git -C /workspace rev-parse HEAD)" == "$(pin target_commit)" ]]
  [[ "$(git -C /workspace rev-parse 'HEAD^{tree}')" == "$(pin target_tree)" ]]
  [[ -z "$(git -C /workspace status --porcelain --untracked-files=all)" ]]
  [[ "$(git -C "${ESHKOL_SOURCE_DIR}" rev-parse HEAD)" == "$(pin eshkol_commit)" ]]
  [[ "$(git -C "${ESHKOL_SOURCE_DIR}" rev-parse 'HEAD^{tree}')" == \
     "$(pin eshkol_tree)" ]]
  [[ -z "$(git -C "${ESHKOL_SOURCE_DIR}" status --porcelain --untracked-files=all)" ]]
  [[ "$(sha256sum /workspace/toolchain/eshkol.lock | awk '{print $1}')" == \
     "$(pin toolchain_lock_sha256)" ]]
  [[ "$(sha256sum "${ESHKOL_BUILD_DIR}/eshkol-run" | awk '{print $1}')" == \
     "$(pin eshkol_runner_sha256)" ]]
  [[ "$(sha256sum "${ESHKOL_BUILD_DIR}/libeshkol-runtime.a" | awk '{print $1}')" == \
     "$(pin eshkol_runtime_sha256)" ]]
  [[ "$(sha256sum "${ESHKOL_BUILD_DIR}/eshkol-transformer-provenance.tsv" | \
      awk '{print $1}')" == "$(pin eshkol_provenance_sha256)" ]]
  verify_toolchain
  printf 'target_commit\t%s\ntarget_tree\t%s\neshkol_commit\t%s\n' \
    "$(pin target_commit)" "$(pin target_tree)" "$(pin eshkol_commit)"
}
check_symbols() {
  local scratch d1 cli3 object depfile
  scratch="$(cat /out/production/scratch-path.txt)"
  [[ "${scratch}" == /out/production/.build.* && -d "${scratch}" ]]
  d1="${BUILD_DIR}/d1/libeshkol_transformer_d1.a"
  cli3="${scratch}/libeshkol_transformer_cli3.a"
  object="${scratch}/cli3.o"
  depfile="${scratch}/eshkol-transformer.d"
  [[ "$(ar t "${d1}")" == stdlib.o ]]
  [[ "$(ar t "${cli3}")" == cli3.o ]]
  cmp /workspace/native/d1_e1b_public_exports.txt \
    "${d1}.evidence/package-exports.txt"
  cmp /workspace/native/cli3_generate_public_exports.txt \
    "${scratch}/cli3.o.evidence/package-exports.txt"
  cmp /workspace/native/cli3_generate_defined_symbols.txt \
    "${scratch}/cli3.o.evidence/global-defined.txt"
  cmp /workspace/native/cli3_generate_undefined_symbols.txt \
    "${scratch}/cli3.o.evidence/undefined.txt"
  nm -g --defined-only --format=posix "${object}" | \
    awk '$2 ~ /^[A-Za-z]$/ { print $1 }' | LC_ALL=C sort -u \
    > /out/checks/cli3-nm-defined.txt
  cmp /workspace/native/cli3_generate_defined_symbols.txt \
    /out/checks/cli3-nm-defined.txt
  readelf --wide --symbols "${object}" | \
    awk '$5 == "GLOBAL" && $7 != "UND" { print $8 }' | LC_ALL=C sort -u \
    > /out/checks/cli3-readelf-defined.txt
  cmp /workspace/native/cli3_generate_defined_symbols.txt \
    /out/checks/cli3-readelf-defined.txt
  nm -u --format=posix "${object}" | awk '{ print $1 }' | \
    LC_ALL=C sort -u > /out/checks/cli3-nm-undefined.txt
  cmp /workspace/native/cli3_generate_undefined_symbols.txt \
    /out/checks/cli3-nm-undefined.txt
  for module in data error_consumer cli3_corpus_validate; do
    grep -F "/public/transformer/${module}.esk" "${depfile}" >/dev/null
  done
  if grep -F '/workspace/lib/transformer/data.esk' "${depfile}" >/dev/null; then
    return 1
  fi
  cmp /workspace/lib/transformer/cli3_corpus_validate.esk \
    "${scratch}/cli3-corpus-validate.esk"
  nm -u "${scratch}/eshkol-transformer" \
    > /out/checks/installed-nm-undefined.txt
  if grep -E 'et_e1b_public_(d1_token_corpus_write|d2_)' \
      /out/checks/installed-nm-undefined.txt >/dev/null; then
    return 1
  fi
  sha256sum "${d1}" "${cli3}" "${object}" \
    "${scratch}/eshkol-transformer" "${depfile}" \
    > /out/checks/artifact-sha256.txt
}

case "${1:-run}" in
  check-pins) check_pins; exit 0 ;;
  check-symbols) check_symbols; exit 0 ;;
  run) ;;
  *) printf 'unknown inside gate mode: %s\n' "$1" >&2; exit 64 ;;
esac

mkdir -p /out/steps /out/checks "${BUILD_DIR}" "${TMPDIR}" "${XDG_CACHE_HOME}" "${HOME}"
step initial-pins 120 /usr/bin/bash /gate/inside.sh check-pins
step standalone-d1 3600 /usr/bin/bash /workspace/scripts/test-cli3-corpus-validate.sh \
  /out/standalone
step installed-cli3 7200 /usr/bin/bash /workspace/scripts/test-cli3-production-generate.sh \
  /out/production
step archive-and-app-symbols 300 /usr/bin/bash /gate/inside.sh check-symbols
step cross-command 7200 /usr/bin/bash /workspace/scripts/test-cli3-cross-command-chain.sh \
  /out/cross-command
step final-pins 120 /usr/bin/bash /gate/inside.sh check-pins
