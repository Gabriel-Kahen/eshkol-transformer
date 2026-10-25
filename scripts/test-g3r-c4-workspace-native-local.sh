#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
cc="${CC:-clang}"
tmp="$(mktemp -d "${TMPDIR:-/home/gabe/.cache}/g3r-c4-workspace.XXXXXX")"
trap 'rm -rf -- "$tmp"' EXIT
flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -g -O1
  -ffp-contract=off -fexcess-precision=standard -fno-fast-math
  -I include -I native -I src/eshkol_transformer)
sources=(tests/g3c4/test_workspace_owner_native.c
  tests/m3t/kernel_fail_allocator.c native/n3k_primitives_provider.c)
for mode in normal san; do
  extra=()
  if [[ $mode == san ]]; then
    extra=(-fsanitize=address,undefined -fno-omit-frame-pointer)
  fi
  "$cc" "${flags[@]}" "${extra[@]}" -DET_A2_KV_CACHE_TESTING \
    -DET_G3R_C4_WORKSPACE_A2_TU \
    -c tests/g3c4/test_workspace_owner_native.c -o "$tmp/a2-$mode.o"
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
base=(-DET_G3C4_CONTEXT_PRIVATE -DET_G3C4_NATIVE_PINS_PRIVATE
  -DET_G3C4_ACTIVE_CALL_PRIVATE -DET_G3C4_GENERATOR_PRIVATE
  -DET_G3R_C4_OWNER_TOPOLOGY_PRIVATE -DET_G3R_C4_OWNER_ASSOCIATION_PRIVATE
  -DET_G3R_C4_OWNER_PARAMETER_LEASE_PRIVATE
  -DET_G3R_C4_CONTEXT_EXCLUSION_PRIVATE)
"$cc" "${flags[@]}" "${base[@]}" -c \
  src/eshkol_transformer/g3c4_model_owner.c -o "$tmp/off.o"
"$cc" "${flags[@]}" "${base[@]}" -DET_G3R_C4_WORKSPACE_PRIVATE -c \
  src/eshkol_transformer/g3c4_model_owner.c -o "$tmp/on.o"
for symbol in et_g3c4_private_workspace_stage_v1 \
              et_g3c4_private_workspace_staged_nonce_v1 \
              et_g3c4_private_workspace_abort_v1 \
              et_g3c4_private_workspace_prepare_commit_v1 \
              et_g3c4_private_workspace_commit_v1 \
              et_g3c4_private_workspace_live_status_v1 \
              et_g3c4_private_workspace_binding_status_v1 \
              et_g3c4_private_workspace_release_v1; do
  if nm -g --defined-only "$tmp/off.o" | awk '{print $3}' | rg -Fx "$symbol"; then
    echo "feature-off workspace symbol exported: $symbol" >&2
    exit 1
  fi
  nm -g --defined-only "$tmp/on.o" | awk '{print $3}' | rg -Fx "$symbol" >/dev/null
done
cat "$tmp/normal.stdout"
