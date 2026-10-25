#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec env G3C4_TERMINAL_EXTENT_WITNESS=1 \
  "$root/test-g3c4-p2g2-terminal-pending-anchor.sh" "$@"
