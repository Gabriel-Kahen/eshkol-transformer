#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
require_command python3
require_command flock
mkdir -p "${PROJECT_ROOT}/build/benchmarks/compare-tmp"
export TMPDIR="${PROJECT_ROOT}/build/benchmarks/compare-tmp"
df -h /home
args=()
if [[ -n "${BENCHMARK_CONTAINER_IMAGE:-}" ]]; then
  args+=(--container-image "${BENCHMARK_CONTAINER_IMAGE}")
fi
flock /tmp/eshkol-transformer-heavy-build.lock \
  python3 -B "${PROJECT_ROOT}/tests/b0/attention_compare.py" run \
  "${PROJECT_ROOT}/benchmarks/attention_compare_v1.json" \
  "${BENCHMARK_REPORT:-${PROJECT_ROOT}/build/benchmarks/attention-compare-v1-report.json}" \
  --cc "${CC:-clang}" "${args[@]}"
