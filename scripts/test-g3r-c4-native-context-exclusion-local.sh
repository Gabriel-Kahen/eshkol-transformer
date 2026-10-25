#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
cc="${CC:-clang}"
tmp="$(mktemp -d "${TMPDIR:-/home/gabe/.cache}/g3r-c4-context-exclusion.XXXXXX")"
trap 'rm -rf -- "$tmp"' EXIT
flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -g -O1
  -ffp-contract=off -fexcess-precision=standard -fno-fast-math
  -I include -I native -I src/eshkol_transformer)
sources=(tests/g3c4/test_no_live_owner_context.c
  tests/m3t/kernel_fail_allocator.c native/n3k_primitives_provider.c)
for mode in normal san; do
  extra=()
  if [[ $mode == san ]]; then
    extra=(-fsanitize=address,undefined -fno-omit-frame-pointer)
  fi
  "$cc" "${flags[@]}" "${extra[@]}" -DET_A2_KV_CACHE_TESTING \
    -DET_G3R_C4_CONTEXT_EXCLUSION_A2_TU \
    -c tests/g3c4/test_no_live_owner_context.c -o "$tmp/a2-$mode.o"
  "$cc" "${flags[@]}" "${extra[@]}" -DET_A2_KV_CACHE_TESTING \
    "${sources[@]}" "$tmp/a2-$mode.o" -lm -o "$tmp/$mode"
  if [[ $mode == san ]]; then
    ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
      UBSAN_OPTIONS=halt_on_error=1 "$tmp/$mode" >"$tmp/$mode.stdout" \
      2>"$tmp/$mode.stderr"
  else
    "$tmp/$mode" >"$tmp/$mode.stdout" 2>"$tmp/$mode.stderr"
    "$tmp/$mode" >"$tmp/repeat.stdout" 2>"$tmp/repeat.stderr"
    cmp "$tmp/$mode.stdout" "$tmp/repeat.stdout"
  fi
  test ! -s "$tmp/$mode.stderr"
done
test ! -s "$tmp/repeat.stderr"
cmp "$tmp/normal.stdout" "$tmp/san.stdout"
"$cc" "${flags[@]}" -DET_G3C4_CONTEXT_PRIVATE -DET_G3C4_NATIVE_PINS_PRIVATE \
  -DET_G3C4_ACTIVE_CALL_PRIVATE -DET_G3C4_GENERATOR_PRIVATE \
  -c src/eshkol_transformer/g3c4_model_owner.c -o "$tmp/off.o"
"$cc" "${flags[@]}" -DET_G3C4_CONTEXT_PRIVATE -DET_G3C4_NATIVE_PINS_PRIVATE \
  -DET_G3C4_ACTIVE_CALL_PRIVATE -DET_G3C4_GENERATOR_PRIVATE \
  -DET_G3R_C4_CONTEXT_EXCLUSION_PRIVATE \
  -c src/eshkol_transformer/g3c4_model_owner.c -o "$tmp/on.o"
symbol=et_g3c4_private_no_live_owner_context_status_v1
if nm -g --defined-only "$tmp/off.o" | awk '{print $3}' | rg -Fx "$symbol"; then
  echo "feature-off C4 context exclusion symbol exported" >&2
  exit 1
fi
nm -g --defined-only "$tmp/on.o" | awk '{print $3}' | rg -Fx "$symbol" >/dev/null
if "$cc" "${flags[@]}" -DET_G3R_C4_CONTEXT_EXCLUSION_PRIVATE \
    -c src/eshkol_transformer/g3c4_model_owner.c -o "$tmp/invalid.o" \
    2>"$tmp/invalid.stderr"; then
  echo "context exclusion compiled without generation registry" >&2
  exit 1
fi
rg -F 'C4 context exclusion requires the generation transport registry' \
  "$tmp/invalid.stderr" >/dev/null
cat "$tmp/normal.stdout"
