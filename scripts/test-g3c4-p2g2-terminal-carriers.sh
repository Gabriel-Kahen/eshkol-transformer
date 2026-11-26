#!/usr/bin/env bash
set -euo pipefail
export G3C4_TERMINAL_EXTENT_WITNESS=1
exec bash "$(dirname -- "$0")/test-g3c4-p2g2-terminal-pending-anchor.sh" "$@"
