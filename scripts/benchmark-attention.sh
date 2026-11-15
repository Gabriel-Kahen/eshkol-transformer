#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
require_command python3
require_command flock
# This direct-native development benchmark does not configure/build Eshkol.
df -h /home
flock /tmp/eshkol-transformer-heavy-build.lock \
  python3 -B "${PROJECT_ROOT}/tests/b0/attention_benchmark.py" run \
  "${PROJECT_ROOT}/benchmarks/attention_v1.json" \
  "${BENCHMARK_REPORT:-${PROJECT_ROOT}/build/benchmarks/attention-v1-report.json}" \
  --cc "${CC:-clang}"
