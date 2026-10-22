#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
out=${1:?usage: test-cpu4-training-objective.sh FRESH_EVIDENCE}
mkdir -m 700 "$out"
cc=$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cc_path)
flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -ffp-contract=off
 -fexcess-precision=standard -fno-fast-math -fstack-protector-all
 -I "$PROJECT_ROOT/include" -I "$PROJECT_ROOT/native" -I "$PROJECT_ROOT/src" -I "$out")
PYTHONDONTWRITEBYTECODE=1 python3 "$PROJECT_ROOT/tests/cpu4_objective/reference.py" "$out/reference.h" >"$out/reference.stdout"
PYTHONDONTWRITEBYTECODE=1 python3 "$PROJECT_ROOT/tests/cpu4_objective/reference.py" "$out/reference-repeat.h" >"$out/reference-repeat.stdout"
cmp "$out/reference.h" "$out/reference-repeat.h"
mkdir "$out/base"
git -C "$PROJECT_ROOT" show 78f39c93adafb698cbf883bd6b172c1159a03e24:src/eshkol_transformer/cpu4_training_transport.c >"$out/base/cpu4_training_transport.c"
for tree in base current; do
 source="$out/base/cpu4_training_transport.c"
 if [[ "$tree" == current ]]; then source="$PROJECT_ROOT/src/eshkol_transformer/cpu4_training_transport.c"; fi
 "$cc" "${flags[@]}" -I "$PROJECT_ROOT/src/eshkol_transformer" -O2 \
  -DET_CPU4_TRAINING_PRIVATE -DET_G3R_CANDIDATE_RETIRE_PRIVATE -c "$source" -o "$out/$tree-default-transport.o"
done
cmp "$out/base-default-transport.o" "$out/current-default-transport.o"
for mode in normal sanitize; do
 extra=(-O2)
 if [[ "$mode" == sanitize ]]; then extra=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer); fi
 "$cc" "${flags[@]}" "${extra[@]}" "$PROJECT_ROOT/tests/cpu4_objective/test_objective.c" \
  "$PROJECT_ROOT/tests/m3t/kernel_fail_allocator.c" "$PROJECT_ROOT/native/n3k_primitives_provider.c" \
  "$PROJECT_ROOT/native/n2_primitives_provider.c" "$PROJECT_ROOT/native/a2_attention_provider.c" \
  "$PROJECT_ROOT/native/indexed_cross_entropy.c" "$PROJECT_ROOT/native/l3s_masked_objective_provider.c" \
  -lm -ldl -o "$out/objective-$mode"
 for repeat in 1 2; do
  env ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
   timeout --foreground --signal=TERM --kill-after=5s 120s "$out/objective-$mode" \
   >"$out/$mode-$repeat.stdout" 2>"$out/$mode-$repeat.stderr"
  test ! -s "$out/$mode-$repeat.stderr"
 done
 cmp "$out/$mode-1.stdout" "$out/$mode-2.stdout"
done
cmp "$out/normal-1.stdout" "$out/sanitize-1.stdout"
cat "$out/normal-1.stdout"
