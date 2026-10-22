#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
out=${1:?usage: test-cpu4-training-transport.sh EVIDENCE_DIR}
mkdir -p "$out"
cc=$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cc_path)
flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -ffp-contract=off
  -fexcess-precision=standard -fno-fast-math -fstack-protector-all
  -I "$PROJECT_ROOT/include" -I "$PROJECT_ROOT/native" -I "$PROJECT_ROOT/src" -I "$out")
python3 "$PROJECT_ROOT/tests/cpu4_transport/reference.py" --self-check --header "$out/reference.h" >"$out/reference.stdout"
python3 "$PROJECT_ROOT/tests/cpu4_transport/reference.py" --header "$out/reference-repeat.h" >"$out/reference-repeat.stdout"
cmp "$out/reference.h" "$out/reference-repeat.h"
mkdir -p "$out/base"
git -C "$PROJECT_ROOT" show 063bc4c93ee2c1c62d108d0d4818182613c93a9f:src/eshkol_transformer/g3c4_model_owner.c >"$out/base/g3c4_model_owner.c"
for tree in base current; do
  source="$out/base/g3c4_model_owner.c"
  if [[ "$tree" == current ]]; then source="$PROJECT_ROOT/src/eshkol_transformer/g3c4_model_owner.c"; fi
  "$cc" "${flags[@]}" -I "$PROJECT_ROOT/src/eshkol_transformer" -O2 -c "$source" -o "$out/$tree-owner.o"
done
cmp "$out/base-owner.o" "$out/current-owner.o"
"$cc" "${flags[@]}" -O2 -c "$PROJECT_ROOT/src/eshkol_transformer/cpu4_training_transport.c" -o "$out/transport-featureoff.o"
test -z "$(nm --defined-only --extern-only "$out/transport-featureoff.o")"
for mode in normal sanitize; do
  extra=(-O2)
  if [[ "$mode" == sanitize ]]; then extra=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer); fi
  "$cc" "${flags[@]}" "${extra[@]}" "$PROJECT_ROOT/tests/cpu4_transport/test_transport.c" \
    "$PROJECT_ROOT/tests/m3t/kernel_fail_allocator.c" \
    "$PROJECT_ROOT/native/n3k_primitives_provider.c" "$PROJECT_ROOT/native/n2_primitives_provider.c" \
    "$PROJECT_ROOT/native/a2_attention_provider.c" -lm -ldl -o "$out/transport-$mode"
  for repeat in 1 2; do
    env ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
      timeout --foreground --signal=TERM --kill-after=5s 120s "$out/transport-$mode" \
      >"$out/$mode-$repeat.stdout" 2>"$out/$mode-$repeat.stderr"
    test ! -s "$out/$mode-$repeat.stderr"
  done
  cmp "$out/$mode-1.stdout" "$out/$mode-2.stdout"
done
cmp "$out/normal-1.stdout" "$out/sanitize-1.stdout"
cat "$out/normal-1.stdout"
