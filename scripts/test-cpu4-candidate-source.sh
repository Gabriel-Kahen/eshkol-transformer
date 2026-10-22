#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
GUILE_AUTO_COMPILE=0 guile --no-auto-compile -s scripts/test-cpu4-candidate-source.scm
git diff --check -- internal/p1/lib/transformer/module.esk native/cpu4_candidate_extension.esk tests/cpu4_candidate scripts/test-cpu4-candidate-source.scm scripts/test-cpu4-candidate-source.sh
