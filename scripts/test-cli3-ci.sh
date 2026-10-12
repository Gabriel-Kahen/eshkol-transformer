#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

evidence="$(mktemp -d "${RUNNER_TEMP:-${TMPDIR:-/tmp}}/eshkol-cli3-ci.XXXXXX")"
printf 'CLI3 CI evidence: %s\n' "${evidence}"
/usr/bin/bash "${PROJECT_ROOT}/scripts/test-cli3-production-generate.sh" \
  "${evidence}/production-generate"
/usr/bin/bash "${PROJECT_ROOT}/scripts/test-cli3-cross-command-chain.sh" \
  "${evidence}/cross-command-chain"
