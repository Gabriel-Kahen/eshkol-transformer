#!/usr/bin/env bash

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

printf 'C2 PRIVATE GATE PASS: parser, codec, same-fd load, inspect, X1, D2, policy, ownership, model encode, O2 encode, atomic full save\n'
