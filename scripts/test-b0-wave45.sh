#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root}"

cc="${CC:-clang}"
cxx="${CXX:-clang++}"
out_dir="${root}/build/benchmarks"
report="${WAVE45_BENCHMARK_REPORT:-${out_dir}/wave45-bench-v1-report.json}"
mkdir -p "${out_dir}"

# 1. Normal -O2 measured build, execution, and schema/checksum verification
CC="${cc}" CXX="${cxx}" WAVE45_BENCHMARK_REPORT="${report}" \
  bash scripts/benchmark-wave45.sh

# 2. ASan/UBSan/LSan build + smoke execution + report verification
scratch="$(mktemp -d "${out_dir}/.san.XXXXXX")"
trap 'rm -rf -- "${scratch}"' EXIT

san_flags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic
  -O1 -g -ffp-contract=off -fexcess-precision=standard -fno-fast-math
  -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all
  -I "${root}/include"
)
native_sources=(
  "${root}/native/kernel_abi.c"
  "${root}/native/mod5_primitives_provider.c"
  "${root}/native/pa4_wider_attention_provider.c"
  "${root}/native/mp4_mixed_precision_provider.c"
  "${root}/native/ac4_checkpointing.c"
  "${root}/native/dist5_collectives.c"
)

san_bin="${scratch}/bench_wave45.san"
"${cc}" "${san_flags[@]}" "${root}/tests/b0/bench_wave45.c" "${native_sources[@]}" \
  -lm -o "${san_bin}"

compiler_id="$("${cc}" --version | head -n 1)"
git_commit="$(git rev-parse HEAD 2>/dev/null || printf 'unknown')"
git_dirty="0"
if [[ -n "$(git status --porcelain --untracked-files=all 2>/dev/null || true)" ]]; then
  git_dirty="1"
fi

san_report="${scratch}/wave45-bench-v1-sanitize-smoke.json"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  "${san_bin}" \
  --smoke \
  --output "${san_report}" \
  --compiler-id "${compiler_id}" \
  --cflags "-std=c11 -O1 -g -fsanitize=address,undefined -ffp-contract=off -fexcess-precision=standard -fno-fast-math" \
  --build-mode sanitize \
  --git-commit "${git_commit}" \
  --git-dirty "${git_dirty}"

python3 -B "${root}/tests/b0/test_wave45_benchmark.py" --verify "${san_report}"
printf 'B0 Wave 4/5 ASan/UBSan/LSan smoke: PASS\n'

# 3. Full Python validation and negative test suite against the -O2 report
WAVE45_BENCHMARK_REPORT="${report}" \
  python3 -B -m unittest -v tests.b0.test_wave45_benchmark
printf 'B0 Wave 4/5 benchmark suite: PASS\n'
