#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

binary="$(project_build_dir)/eshkol-transformer-smoke"
expected="${PROJECT_ROOT}/tests/expected/smoke.stdout"
[[ -x "${binary}" ]] || die "smoke artifact not found at ${binary}; run 'make build'"
actual="$(mktemp "${TMPDIR:-/tmp}/eshkol-transformer-smoke.XXXXXX")"
trap 'rm -f -- "${actual}"' EXIT
(cd -- "${PROJECT_ROOT}" && "${binary}") > "${actual}"
cmp --silent "${expected}" "${actual}" || {
  printf 'expected smoke bytes:\n' >&2
  od -An -tx1c "${expected}" >&2
  printf 'actual smoke bytes:\n' >&2
  od -An -tx1c "${actual}" >&2
  die "smoke output mismatch"
}
cat "${actual}"
