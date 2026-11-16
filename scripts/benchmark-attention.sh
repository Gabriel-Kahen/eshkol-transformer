#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
require_command python3
require_command flock
# The default remains a standalone compatibility probe; --supported verifies F0.
df -h /home
if [[ "${1:-}" == --supported ]]; then
  compiler="${CC:-$(lock_value supported_cc)}"
  default_report="${PROJECT_ROOT}/build/benchmarks/attention-v2-supported-report.json"
else
  compiler="${CC:-clang}"
  default_report="${PROJECT_ROOT}/build/benchmarks/attention-v1-report.json"
fi
flock /tmp/eshkol-transformer-heavy-build.lock \
  python3 -B "${PROJECT_ROOT}/tests/b0/attention_benchmark.py" run \
  "${PROJECT_ROOT}/benchmarks/attention_v1.json" \
  "${BENCHMARK_REPORT:-${default_report}}" \
  --cc "${compiler}" "$@"
