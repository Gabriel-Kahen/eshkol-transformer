#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

/usr/bin/bash "${PROJECT_ROOT}/scripts/compile-smoke.sh" "$(project_build_dir)"
printf 'built: %s\n' "$(project_build_dir)/eshkol-transformer-smoke"
