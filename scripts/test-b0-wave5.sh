#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root}"

cc="${CC:-clang}"
cxx="${CXX:-clang++}"
out_dir="${root}/build/b0_wave5"
mkdir -p "${out_dir}"

# 1. Deterministic build of all 11 native archives + normal & sanitized runners.
CC="${cc}" CXX="${cxx}" bash "${root}/scripts/build-b0-wave5.sh"

compiler_id="$("${cc}" --version | head -n 1)"
git_commit="$(git rev-parse HEAD 2>/dev/null || printf 'unknown')"
git_dirty="0"
if [[ -n "$(git status --porcelain --untracked-files=all 2>/dev/null || true)" ]]; then
  git_dirty="1"
fi

report_run1="${WAVE5_BENCHMARK_REPORT:-${out_dir}/wave5-bench-v1-smoke-run1.json}"
report_run2="${out_dir}/wave5-bench-v1-smoke-run2.json"
report_san="${out_dir}/wave5-bench-v1-sanitize-smoke.json"
report_group="${out_dir}/wave5-bench-v1-group-lora.json"

# 2. Normal --smoke run #1 and run #2 (for two-run checksum identity).
WAVE5_SKIP_BUILD=1 CC="${cc}" CXX="${cxx}" WAVE5_BENCHMARK_REPORT="${report_run1}" \
  bash "${root}/scripts/benchmark-wave5.sh" --smoke

WAVE5_SKIP_BUILD=1 CC="${cc}" CXX="${cxx}" WAVE5_BENCHMARK_REPORT="${report_run2}" \
  bash "${root}/scripts/benchmark-wave5.sh" --smoke

# 3. Single-group filter smoke test (--group lora).
WAVE5_SKIP_BUILD=1 CC="${cc}" CXX="${cxx}" WAVE5_BENCHMARK_REPORT="${report_group}" \
  bash "${root}/scripts/benchmark-wave5.sh" --smoke --group lora

# 4. ASan/UBSan/LSan --smoke run with strict leak detection and halt-on-error.
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  "${out_dir}/b0_wave5_bench_san" \
  --smoke \
  --output "${report_san}" \
  --archive-dir "${out_dir}/san_archives" \
  --compiler-id "${compiler_id}" \
  --cflags "-std=c11 -O1 -g -fsanitize=address,undefined -ffp-contract=off -fexcess-precision=standard -fno-fast-math" \
  --build-mode sanitize \
  --git-commit "${git_commit}" \
  --git-dirty "${git_dirty}"

python3 -B "${root}/tests/b0/test_b0_wave5.py" --verify "${report_san}"
printf 'B0 Wave 5 ASan/UBSan/LSan smoke: PASS\n'

# 5. Full Python schema, archive SHA-256, two-run checksum invariance, and negative tests.
WAVE5_BENCHMARK_REPORT="${report_run1}" \
WAVE5_BENCHMARK_REPORT_2="${report_run2}" \
WAVE5_BENCHMARK_REPORT_SAN="${report_san}" \
WAVE5_BENCHMARK_REPORT_GROUP="${report_group}" \
WAVE5_BENCHMARK_BIN="${out_dir}/b0_wave5_bench" \
WAVE5_ARCHIVE_DIR="${root}/build" \
  python3 -B -m unittest -v tests.b0.test_b0_wave5

printf 'B0 Wave 5 extended benchmark suite: PASS\n'
