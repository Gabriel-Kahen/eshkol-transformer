#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
require_command python3
build_dir="$(project_build_dir)"
target="${build_dir}/eshkol-transformer-smoke"
[[ -x "${target}" ]] || die "smoke artifact not found at ${target}; run 'make build'"

provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
manifest="${build_dir}/toolchain-manifest.tsv"
cc="$(tsv_value "${provenance}" cc_path)"
[[ -x "${cc}" ]] || die "provenance C compiler is unavailable: ${cc}"

benchmark_dir="${build_dir}/benchmarks"
mkdir -p "${benchmark_dir}"
helper="${benchmark_dir}/measure-linux-v1"
"${cc}" -std=c11 -O2 -Wall -Wextra -Werror \
  "${PROJECT_ROOT}/benchmarks/measure_linux.c" -o "${helper}"

report="${BENCHMARK_REPORT:-${benchmark_dir}/smoke-v1-report.json}"
support_status=supported
if [[ "${ESHKOL_ALLOW_UNSUPPORTED_HOST:-0}" == 1 ]]; then
  support_status=compatibility-only
fi
python3 "${PROJECT_ROOT}/tests/b0/run_benchmark.py" run \
  "${PROJECT_ROOT}/benchmarks/smoke_v1.json" "${report}" \
  --helper "${helper}" \
  --provenance "${provenance}" \
  --toolchain-manifest "${manifest}" \
  --support-status "${support_status}"
