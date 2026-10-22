#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
out=${1:?usage: test-cpu4-training-lease.sh EVIDENCE_DIR}
mkdir -p "$out"
cc=$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cc_path)
flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -ffp-contract=off
  -fexcess-precision=standard -fno-fast-math -fstack-protector-all
  -I "$PROJECT_ROOT/include" -I "$PROJECT_ROOT/native" -I "$PROJECT_ROOT/src")
mkdir -p "$out/base"
git -C "$PROJECT_ROOT" show 574b98e42ee0d81491458fc5439c7efb64082014:src/eshkol_transformer/g3c4_model_owner.c >"$out/base/g3c4_model_owner.c"
for profile in owner context generator; do
  macros=()
  if [[ "$profile" != owner ]]; then macros+=(-DET_G3C4_CONTEXT_PRIVATE); fi
  if [[ "$profile" == generator ]]; then macros+=(-DET_G3C4_NATIVE_PINS_PRIVATE -DET_G3C4_ACTIVE_CALL_PRIVATE -DET_G3C4_GENERATOR_PRIVATE); fi
  for tree in base current; do
    source="$out/base/g3c4_model_owner.c"
    if [[ "$tree" == current ]]; then source="$PROJECT_ROOT/src/eshkol_transformer/g3c4_model_owner.c"; fi
    "$cc" "${flags[@]}" -I "$PROJECT_ROOT/src/eshkol_transformer" -O2 "${macros[@]}" \
      -c "$source" -o "$out/$tree-$profile.o"
  done
  cmp "$out/base-$profile.o" "$out/current-$profile.o"
done
for mode in normal sanitize; do
  extra=(-O2)
  if [[ "$mode" == sanitize ]]; then extra=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer); fi
  "$cc" "${flags[@]}" "${extra[@]}" "$PROJECT_ROOT/tests/cpu4_lease/test_training_lease.c" \
    "$PROJECT_ROOT/tests/m3t/kernel_fail_allocator.c" "$PROJECT_ROOT/native/n3k_primitives_provider.c" \
    "$PROJECT_ROOT/native/a2_kv_cache.c" -lm -o "$out/lease-$mode"
  for repeat in 1 2; do
    env ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
      timeout --foreground --signal=TERM --kill-after=5s 120s "$out/lease-$mode" \
      >"$out/$mode-$repeat.stdout" 2>"$out/$mode-$repeat.stderr"
    test ! -s "$out/$mode-$repeat.stderr"
  done
  cmp "$out/$mode-1.stdout" "$out/$mode-2.stdout"
done
cmp "$out/normal-1.stdout" "$out/sanitize-1.stdout"
cat "$out/normal-1.stdout"
