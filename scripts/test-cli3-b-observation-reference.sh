#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
require_command clang-21

scratch="$(mktemp -d "${TMPDIR:-/tmp}/cli3-observation-reference.XXXXXX")"
trap 'rm -rf -- "${scratch}"' EXIT
for optimization in 0 2; do
  clang-21 -std=c11 "-O${optimization}" -ffp-model=strict \
    -ffp-contract=off \
    "${PROJECT_ROOT}/tests/cli3/observation_reference.c" -lm \
    -o "${scratch}/reference-O${optimization}"
  "${scratch}/reference-O${optimization}" \
    >"${scratch}/reference-O${optimization}.stdout"
done
cmp "${scratch}/reference-O0.stdout" "${scratch}/reference-O2.stdout"
grep -Fx \
  'first=40b21550 second=40aacc63 total-weight=40e00000 whole=40aef60f mutant=40aeacea' \
  "${scratch}/reference-O0.stdout" >/dev/null
printf 'CLI3-B-OBSERVATION-REFERENCE-PASS\n'
