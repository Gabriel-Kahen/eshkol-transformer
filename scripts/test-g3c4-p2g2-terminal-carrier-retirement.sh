#!/usr/bin/env bash
set -euo pipefail
script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if (( $# != 1 )); then
  printf '%s\n' "Usage: $0 EVIDENCE_DIRECTORY" >&2
  exit 2
fi
G3C4_TERMINAL_RETIREMENT_WITNESS=1 \
  exec bash "$script_directory/test-g3c4-p2g2-terminal-pending-anchor.sh" "$1"
