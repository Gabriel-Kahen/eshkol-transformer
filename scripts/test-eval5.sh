#!/usr/bin/env bash
set -euo pipefail

eval5_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${eval5_root}"
eval5_output="${1:-build/eval5}"
eval5_cc="${CC:-cc}"
eval5_leaks="${EVAL5_ASAN_DETECT_LEAKS:-0}"
case "${eval5_leaks}" in
  0|1) ;;
  *) printf 'EVAL5_ASAN_DETECT_LEAKS must be 0 or 1\n' >&2; exit 2 ;;
esac

mkdir -p -- "${eval5_output}"
eval5_tmp="$(mktemp -d "${eval5_output}/.eval5-test.XXXXXX")"
trap 'rm -rf -- "${eval5_tmp}"' EXIT
eval5_flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -I include
  -fno-fast-math -ffp-contract=off -frounding-math -fexcess-precision=standard)
eval5_sanitize=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
bash scripts/build-eval5.sh "${eval5_output}/normal" normal
bash scripts/build-eval5.sh "${eval5_output}/sanitize" sanitize

"${eval5_cc}" "${eval5_flags[@]}" -O2 tests/eval5/test_eval5_benchmark.c \
  "${eval5_output}/normal/libeshkol_eval5.a" -lm -o "${eval5_tmp}/normal"
"${eval5_cc}" "${eval5_flags[@]}" "${eval5_sanitize[@]}" \
  tests/eval5/test_eval5_benchmark.c "${eval5_output}/sanitize/libeshkol_eval5.a" \
  -lm -o "${eval5_tmp}/sanitize"
mkdir -- "${eval5_tmp}/normal-data" "${eval5_tmp}/sanitize-data"
"${eval5_tmp}/normal" "${eval5_tmp}/normal-data" > "${eval5_tmp}/normal-data/results"
ASAN_OPTIONS="detect_leaks=${eval5_leaks}:halt_on_error=1" UBSAN_OPTIONS="halt_on_error=1" \
  "${eval5_tmp}/sanitize" "${eval5_tmp}/sanitize-data" > "${eval5_tmp}/sanitize-data/results"
cmp "${eval5_tmp}/normal-data/results" "${eval5_tmp}/sanitize-data/results"
cmp "${eval5_tmp}/normal-data/benchmark.manifest" "${eval5_tmp}/sanitize-data/benchmark.manifest"
printf 'EVAL5 C driver passed normally and under ASan/UBSan (detect_leaks=%s)\n' "${eval5_leaks}"
python3 tests/eval5/test_eval5_oracle.py "${eval5_tmp}/normal" "${eval5_tmp}/normal-data"
