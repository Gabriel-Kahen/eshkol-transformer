#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

(( $# == 2 )) || die "usage: $0 /absolute/clang /fresh/absolute/output"
cc=$1 out=$2
[[ $cc == /* && -x $cc && $out == /* && ! -e $out && ! -L $out ]] || \
  die "compiler or output path is invalid"
mkdir -- "$out"
python3 "$PROJECT_ROOT/scripts/check-g3r-o2-lease-preflight.py" \
  >"$out/source.stdout" 2>"$out/source.stderr"
flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -O1 -fno-common
       -DET_G3R_CANDIDATE_RETIRE_PRIVATE=1
       -DET_G3R_O2_GRAPH_PREFLIGHT_PRIVATE=1
       -DET_F32_TENSOR_TESTING=1 -DET_TR3_O2_STEP_CLEAR_NATIVE=1
       -I "$PROJECT_ROOT/include" -I "$PROJECT_ROOT/native"
       -I "$PROJECT_ROOT/src")
"$cc" "${flags[@]}" \
  "$PROJECT_ROOT/tests/g3r/o2_candidate_graph_preflight_native_test.c" \
  "$PROJECT_ROOT/native/o2_optimizer.c" \
  "$PROJECT_ROOT/src/eshkol_transformer/m3t_f32_integration.c" \
  "$PROJECT_ROOT/native/kernel_abi.c" \
  "$PROJECT_ROOT/native/i64_tensor.c" \
  "$PROJECT_ROOT/native/t1_i64_shell.c" \
  "$PROJECT_ROOT/native/n3k_primitives_provider.c" \
  "$PROJECT_ROOT/native/n2_primitives_provider.c" \
  "$PROJECT_ROOT/native/a2_attention_provider.c" \
  -lm -o "$out/native-test" >"$out/compile.stdout" 2>"$out/compile.stderr"
test ! -s "$out/compile.stderr"
for run in a b; do
  timeout --foreground --signal=TERM --kill-after=5s 120s \
    "$out/native-test" >"$out/$run.stdout" 2>"$out/$run.stderr"
  test ! -s "$out/$run.stderr"
done
cmp "$out/a.stdout" "$out/b.stdout"
grep -Fx 'G3-R native O2 graph identity PASS' "$out/a.stdout" >/dev/null
grep -E '^checks=[0-9]+$' "$out/a.stdout" >/dev/null
nm -g --defined-only --format=posix "$out/native-test" \
  >"$out/defined.nm"
grep -F 'et_m3t_private_candidate_o2_graph_preflight_v1 ' \
  "$out/defined.nm" >/dev/null
grep -F 'et_o2_private_candidate_graph_entry_match_v1 ' \
  "$out/defined.nm" >/dev/null
printf 'G3-R O2 graph native normal/repeat PASS\n'
