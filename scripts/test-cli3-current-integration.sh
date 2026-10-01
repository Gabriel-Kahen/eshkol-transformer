#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
for command in cmp find realpath sort; do require_command "${command}"; done
[[ $# == 1 ]] || die "usage: $0 FRESH_DISK_BACKED_EVIDENCE_DIR"
evidence="$(realpath -m -- "$1")"
[[ ! -e "${evidence}" ]] || die "integration evidence must be fresh: ${evidence}"
mkdir -p -- "${evidence}"

# Separate compiler and application cache roots; neither build consumes the other.
for name in first second; do
  mkdir "${evidence}/${name}-cache"
  TMPDIR="${evidence}/${name}-cache" CLI3_COMPILER_TIMEOUT_SECONDS=1200 \
    /usr/bin/bash "${PROJECT_ROOT}/scripts/build-cli3.sh" "${evidence}/${name}" \
    >"${evidence}/${name}-build.stdout" 2>"${evidence}/${name}-build.stderr"
done
for file in cli3.o libeshkol_transformer_cli3.a eshkol-transformer; do
  cmp "${evidence}/first/${file}" "${evidence}/second/${file}"
done
for name in first second; do
  find "${evidence}/${name}/cli3.o.evidence" -type f -printf '%P\n' | \
    LC_ALL=C sort >"${evidence}/${name}-evidence-files.txt"
done
cmp "${evidence}/first-evidence-files.txt" "${evidence}/second-evidence-files.txt"
while IFS= read -r file; do
  cmp "${evidence}/first/cli3.o.evidence/${file}" \
      "${evidence}/second/cli3.o.evidence/${file}"
done <"${evidence}/first-evidence-files.txt"
printf 'CLI3-FRESH-BUILD-BYTES-PASS\n' >"${evidence}/build-comparison.stdout"

for name in first second; do
  # The production-generate suite runs test-cli3.sh, which invokes CLI3-B.
  # Retain that real train/evaluate/resume witness without running it twice.
  CLI3_B_EVIDENCE_DIR="${evidence}/${name}-train-evaluate" \
    /usr/bin/bash "${PROJECT_ROOT}/scripts/test-cli3-production-generate.sh" \
      "${evidence}/${name}-generate" "${evidence}/${name}" \
      >"${evidence}/${name}-generate.stdout" 2>"${evidence}/${name}-generate.stderr"
done
for mode in greedy categorical; do
  cmp "${evidence}/first-generate/${mode}-first.json" \
      "${evidence}/second-generate/${mode}-first.json"
done
printf 'CLI3-CURRENT-INTEGRATION-PASS\n'
