#!/usr/bin/env bash
set -euo pipefail

[[ $# == 2 && $1 == /* && $2 == /* && $1 != "$2" ]] || {
  echo 'usage: test-e3-n5-edge-source.sh FRESH_ABSOLUTE_DIR_A FRESH_ABSOLUTE_DIR_B' >&2
  exit 2
}
cd "$(dirname "$0")/.."
python3 -m unittest tests.e3_reference.test_edge_corpus \
  tests.e3_private.test_edge_source tests.e3_private.test_edge_observation -v
python3 -m tests.e3_reference.edge_corpus --output "$1"
python3 -m tests.e3_reference.edge_corpus --output "$2"
diff -qr "$1" "$2"
if [[ -n ${Q0_PYTHON:-} ]]; then
  ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE "$Q0_PYTHON" \
    -m tests.e3_reference.edge_oracle --corpus "$1" --output "$1/edge-oracle.json"
  ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE "$Q0_PYTHON" \
    -m tests.e3_reference.edge_oracle --corpus "$2" --output "$2/edge-oracle.json"
  cmp "$1/edge-oracle.json" "$2/edge-oracle.json"
fi
