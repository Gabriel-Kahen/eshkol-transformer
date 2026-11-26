#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
scratch="$(mktemp -d "${TMPDIR:-/tmp}/c2-sha-spans.XXXXXX")"
trap 'rm -rf -- "$scratch"' EXIT
cc="${CC:-clang}"
flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -Wconversion
       -Wsign-conversion -Wshadow -I include -I native)
"$cc" "${flags[@]}" -O2 -c native/c2_checkpoint_format.c \
  -o "$scratch/ordinary.o"
if nm -g --defined-only "$scratch/ordinary.o" | \
    grep -F et_c2_private_model_sha256_spans_experiment_v1; then
  echo 'ordinary C2 format object exposed the experimental helper' >&2
  exit 1
fi
for mode in normal sanitized; do
  options=(-O2)
  if [[ "$mode" == sanitized ]]; then
    options=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
  fi
  "$cc" "${flags[@]}" "${options[@]}" \
    -DET_C2_SHA_SPANS_EXPERIMENT \
    tests/c2/c2_sha_spans_experiment.c \
    native/c2_checkpoint_format.c native/c2_x1_canonical.c \
    -o "$scratch/test-$mode"
  ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
    UBSAN_OPTIONS=halt_on_error=1 "$scratch/test-$mode"
done
