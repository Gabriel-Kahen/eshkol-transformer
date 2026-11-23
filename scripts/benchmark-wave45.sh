#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root}"

cc="${CC:-clang}"
cxx="${CXX:-clang++}"
out_dir="${root}/build/benchmarks"
report="${WAVE45_BENCHMARK_REPORT:-${out_dir}/wave45-bench-v1-report.json}"
mkdir -p "${out_dir}" "$(dirname -- "${report}")"

# Build all six Wave 4/5 native static libraries with their symbol/ABI audits.
if [[ -r "${ESHKOL_BUILD_DIR:-${root}/.deps/eshkol-build}/eshkol-transformer-provenance.tsv" ]]; then
  bash scripts/build-k1.sh "${root}/build/k1" normal
else
  mkdir -p "${root}/build/k1"
  "${cc}" -std=c11 -Wall -Wextra -Werror -Wpedantic -O2 -ffp-contract=off \
    -fPIC -I "${root}/include" -c "${root}/native/kernel_abi.c" \
    -o "${root}/build/k1/kernel_abi.o"
  ar rcsD "${root}/build/k1/libeshkol_transformer_k1.a" \
    "${root}/build/k1/kernel_abi.o"
fi

CC="${cc}" bash scripts/build-mod5.sh
CC="${cc}" bash scripts/build-pa4-wider.sh
CC="${cc}" bash scripts/build-mp4.sh
CC="${cc}" CXX="${cxx}" bash scripts/build-ac4.sh
CC="${cc}" CXX="${cxx}" bash scripts/build-dist5.sh

cflags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic
  -O2 -ffp-contract=off -fexcess-precision=standard -fno-fast-math
  -I "${root}/include"
)
archives=(
  "${root}/build/ac4/libeshkol_ac4.a"
  "${root}/build/mod5/libeshkol_transformer_mod5.a"
  "${root}/build/pa4_wider/libeshkol_transformer_pa4_wider.a"
  "${root}/build/mp4/libeshkol_transformer_mp4.a"
  "${root}/build/dist5/libeshkol_dist5.a"
  "${root}/build/k1/libeshkol_transformer_k1.a"
)

bench_bin="${out_dir}/bench_wave45"
"${cc}" "${cflags[@]}" "${root}/tests/b0/bench_wave45.c" "${archives[@]}" \
  -lm -o "${bench_bin}"

compiler_id="$("${cc}" --version | head -n 1)"
git_commit="$(git rev-parse HEAD 2>/dev/null || printf 'unknown')"
git_dirty="0"
if [[ -n "$(git status --porcelain --untracked-files=all 2>/dev/null || true)" ]]; then
  git_dirty="1"
fi

"${bench_bin}" \
  --output "${report}" \
  --compiler-id "${compiler_id}" \
  --cflags "-std=c11 -O2 -ffp-contract=off -fexcess-precision=standard -fno-fast-math" \
  --build-mode normal \
  --git-commit "${git_commit}" \
  --git-dirty "${git_dirty}" \
  "$@"

python3 -B "${root}/tests/b0/test_wave45_benchmark.py" --verify "${report}"
