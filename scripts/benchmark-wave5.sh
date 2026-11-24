#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root}"

cc="${CC:-clang}"
cxx="${CXX:-clang++}"
out_dir="${root}/build/b0_wave5"
report="${WAVE5_BENCHMARK_REPORT:-${out_dir}/wave5-bench-v1-report.json}"
mkdir -p "${out_dir}" "$(dirname -- "${report}")"

# Build prerequisite archives + b0_wave5_bench if not already built (or forced).
if [[ "${WAVE5_SKIP_BUILD:-0}" != "1" || ! -x "${out_dir}/b0_wave5_bench" ]]; then
  CC="${cc}" CXX="${cxx}" bash "${root}/scripts/build-b0-wave5.sh"
fi

compiler_id="$("${cc}" --version | head -n 1)"
git_commit="$(git rev-parse HEAD 2>/dev/null || printf 'unknown')"
git_dirty="0"
if [[ -n "$(git status --porcelain --untracked-files=all 2>/dev/null || true)" ]]; then
  git_dirty="1"
fi

# Check whether caller already passed --output.
has_output=0
for arg in "$@"; do
  if [[ "${arg}" == "--output" ]]; then
    has_output=1
    break
  fi
done

extra_args=()
if [[ "${has_output}" == "0" ]]; then
  extra_args+=(--output "${report}")
fi

"${out_dir}/b0_wave5_bench" \
  "${extra_args[@]}" \
  --archive-dir "${root}/build" \
  --compiler-id "${compiler_id}" \
  --cflags "-std=c11 -O2 -ffp-contract=off -fexcess-precision=standard -fno-fast-math" \
  --build-mode normal \
  --git-commit "${git_commit}" \
  --git-dirty "${git_dirty}" \
  "$@"

if [[ "${has_output}" == "0" ]]; then
  python3 -B "${root}/tests/b0/test_b0_wave5.py" --verify "${report}"
fi
