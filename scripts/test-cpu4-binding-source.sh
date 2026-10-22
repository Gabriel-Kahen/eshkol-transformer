#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
GUILE_AUTO_COMPILE=0 guile --no-auto-compile -s tests/cpu4_binding/source_test.scm
git diff --check -- internal/p1/lib/transformer/module.esk native/cpu4_model_binding_extension.esk native/cpu4_p1_fixed_set_extension.esk tests/cpu4_binding
