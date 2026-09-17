#!/usr/bin/bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

/usr/bin/bash "${PROJECT_ROOT}/scripts/test-c2-core.sh"
/usr/bin/bash "${PROJECT_ROOT}/scripts/test-c2-checkpoint-inspect.sh"
/usr/bin/bash "${PROJECT_ROOT}/scripts/test-c2-codec.sh"
/usr/bin/bash "${PROJECT_ROOT}/scripts/test-c2-x1-canonical.sh"
/usr/bin/bash "${PROJECT_ROOT}/scripts/test-c2-d2-cursor-pair.sh"
/usr/bin/bash "${PROJECT_ROOT}/scripts/test-c2-persistence-policy.sh"
/usr/bin/bash "${PROJECT_ROOT}/scripts/test-c2-training-state-owner.sh"
/usr/bin/bash "${PROJECT_ROOT}/scripts/test-c2-model-encode.sh"
/usr/bin/bash "${PROJECT_ROOT}/scripts/test-c2-o2-encode.sh"
/usr/bin/bash "${PROJECT_ROOT}/scripts/test-c2-checkpoint-save.sh"
/usr/bin/bash "${PROJECT_ROOT}/scripts/test-c2-checkpoint-load.sh"
/usr/bin/bash "${PROJECT_ROOT}/scripts/test-c2-public.sh"
/usr/bin/bash "${PROJECT_ROOT}/scripts/test-c2-checkpoint-operational.sh"

printf 'C2 GATE PASS: private parser/codec/lifecycle, exact public aggregate, K2-authenticated load/save/release, and fixed-ceiling operational checkpoint\n'
