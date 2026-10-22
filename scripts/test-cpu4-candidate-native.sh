#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
out=${1:?usage: test-cpu4-candidate-native.sh FRESH_EVIDENCE_DIR}
[[ "$out" == /* && ! -e "$out" && ! -L "$out" ]]
mkdir -p "$out"
cc=$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cc_path)
common_flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -ffp-contract=off
  -fexcess-precision=standard -fno-fast-math -fstack-protector-all)
flags=("${common_flags[@]}" -I "$PROJECT_ROOT/include" -I "$PROJECT_ROOT/native" -I "$PROJECT_ROOT/src/eshkol_transformer")
feature=(-DET_CPU4_CANDIDATE_PRIVATE=1 -DET_G3R_CANDIDATE_RETIRE_PRIVATE=1
  -DET_P1_TRUSTED_BUILD=1)
# Compare each family with an authenticated, complete candidate-off closure.
provider_base=063bc4c93ee2c1c62d108d0d4818182613c93a9f
owner_base=11d3d5b386441a52175a7a3def7c39975ec023df
for family in provider owner; do
  revision=$provider_base
  [[ "$family" == provider ]] || revision=$owner_base
  mkdir -p "$out/base-$family"
  git -C "$PROJECT_ROOT" archive "$revision" include native src | tar -x -C "$out/base-$family"
  printf '%s\n' "$revision" >"$out/base-$family.commit"
done
check_baseline_deps() {
  python3 - "$1" "$2" <<'PYDEPS'
from pathlib import Path
import shlex
import sys
root, depfile = map(Path, sys.argv[1:])
root = root.resolve(strict=True)
paths = shlex.split(depfile.read_text().partition(':')[2].replace('\\\n', ' '))
assert paths, 'empty baseline dependency closure'
for name in paths:
    path = Path(name).resolve(strict=True)
    assert path.is_relative_to(root), f'baseline escaped frozen closure: {path}'
PYDEPS
}
for file in native/f32_tensor.c native/o2_optimizer.c native/a2_kv_cache.c native/p1_identity.c \
    src/eshkol_transformer/m3t_f32_integration.c; do
  name=${file##*/}
  snapshot="$out/base-provider"
  "$cc" "${common_flags[@]}" -I "$snapshot/include" -I "$snapshot/native" \
    -I "$snapshot/src/eshkol_transformer" -MMD -MF "$out/base-$name.d" -O2 \
    -c "$snapshot/$file" -o "$out/base-$name.o"
  check_baseline_deps "$snapshot" "$out/base-$name.d"
  "$cc" "${flags[@]}" -MMD -MF "$out/current-$name.d" -O2 \
    -c "$PROJECT_ROOT/$file" -o "$out/current-$name.o"
  cmp "$out/base-$name.o" "$out/current-$name.o"
  nm -g --defined-only "$out/current-$name.o" >"$out/current-$name.symbols"
  ! grep -F 'et_cpu4_' "$out/current-$name.symbols"
done
# Candidate off, training on: include accepted typed/transport/control-alias code.
for profile in owner context generator; do
  macros=(-DET_CPU4_TRAINING_PRIVATE=1)
  [[ "$profile" == owner ]] || macros+=(-DET_G3C4_CONTEXT_PRIVATE)
  if [[ "$profile" == generator ]]; then
    macros+=(-DET_G3C4_NATIVE_PINS_PRIVATE -DET_G3C4_ACTIVE_CALL_PRIVATE -DET_G3C4_GENERATOR_PRIVATE)
  fi
  snapshot="$out/base-owner"
  "$cc" "${common_flags[@]}" -I "$snapshot/include" -I "$snapshot/native" \
    -I "$snapshot/src/eshkol_transformer" "${macros[@]}" -MMD \
    -MF "$out/base-owner-$profile.d" -O2 \
    -c "$snapshot/src/eshkol_transformer/g3c4_model_owner.c" -o "$out/base-owner-$profile.o"
  check_baseline_deps "$snapshot" "$out/base-owner-$profile.d"
  "$cc" "${flags[@]}" "${macros[@]}" -MMD -MF "$out/current-owner-$profile.d" -O2 \
    -c "$PROJECT_ROOT/src/eshkol_transformer/g3c4_model_owner.c" -o "$out/current-owner-$profile.o"
  cmp "$out/base-owner-$profile.o" "$out/current-owner-$profile.o"
done
for mode in normal sanitize; do
  extra=(-O2)
  [[ "$mode" == normal ]] || extra=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
  "$cc" "${flags[@]}" "${feature[@]}" "${extra[@]}" \
    "$PROJECT_ROOT/tests/cpu4_candidate/test_candidate.c" \
    "$PROJECT_ROOT/tests/m3t/kernel_fail_allocator.c" \
    "$PROJECT_ROOT/native/n3k_primitives_provider.c" \
    "$PROJECT_ROOT/native/a2_kv_cache.c" "$PROJECT_ROOT/native/p1_identity.c" \
    "$PROJECT_ROOT/native/o2_optimizer.c" -lm -o "$out/candidate-$mode" \
    >"$out/$mode.compile.stdout" 2>"$out/$mode.compile.stderr"
  for repeat in 1 2; do
    env ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
      timeout --foreground --signal=TERM --kill-after=5s 120s "$out/candidate-$mode" \
      >"$out/$mode-$repeat.stdout" 2>"$out/$mode-$repeat.stderr"
    test ! -s "$out/$mode-$repeat.stderr"
  done
  cmp "$out/$mode-1.stdout" "$out/$mode-2.stdout"
  env ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
    timeout --foreground --signal=TERM --kill-after=5s 120s "$out/candidate-$mode" second-consumer \
    >"$out/$mode-second-consumer.stdout" 2>"$out/$mode-second-consumer.stderr"
  test ! -s "$out/$mode-second-consumer.stderr"
  for cut in {1..17}; do
    set +e
    env ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
      timeout --foreground --signal=TERM --kill-after=5s 120s "$out/candidate-$mode" "$cut" \
      >"$out/$mode-fault-$cut.stdout" 2>"$out/$mode-fault-$cut.stderr"
    status=$?
    set -e
    printf '%s\n' "$status" >"$out/$mode-fault-$cut.exit"
    [[ "$status" == 134 && ! -s "$out/$mode-fault-$cut.stderr" ]]
  done
  "$cc" "${flags[@]}" "${extra[@]}" \
    "$PROJECT_ROOT/tests/i2/test_cpu4_parameter_destroy_preflight.c" \
    "$PROJECT_ROOT/native/kernel_abi.c" -o "$out/f32-$mode" \
    >"$out/f32-$mode.compile.stdout" 2>"$out/f32-$mode.compile.stderr"
  for repeat in 1 2; do
    env ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
      timeout --foreground --signal=TERM --kill-after=5s 120s "$out/f32-$mode" \
      >"$out/f32-$mode-$repeat.stdout" 2>"$out/f32-$mode-$repeat.stderr"
    test ! -s "$out/f32-$mode-$repeat.stderr"
  done
  cmp "$out/f32-$mode-1.stdout" "$out/f32-$mode-2.stdout"
  "$cc" "${flags[@]}" "${feature[@]}" "${extra[@]}" \
    "$PROJECT_ROOT/tests/p1/test_g3r_candidate_roster.c" \
    -Wl,--wrap=calloc -o "$out/p1-$mode" \
    >"$out/p1-$mode.compile.stdout" 2>"$out/p1-$mode.compile.stderr"
  "$cc" "${flags[@]}" "${feature[@]}" "${extra[@]}" \
    -DET_F32_TENSOR_TESTING=1 -DET_TR3_O2_STEP_CLEAR_NATIVE=1 \
    "$PROJECT_ROOT/tests/o2/test_g3r_o2_retire.c" \
    "$PROJECT_ROOT/native/f32_tensor.c" "$PROJECT_ROOT/native/kernel_abi.c" \
    -o "$out/o2-$mode" >"$out/o2-$mode.compile.stdout" 2>"$out/o2-$mode.compile.stderr"
  for leaf in p1 o2; do
    for repeat in 1 2; do
      env ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
        timeout --foreground --signal=TERM --kill-after=5s 120s "$out/$leaf-$mode" \
        >"$out/$leaf-$mode-$repeat.stdout" 2>"$out/$leaf-$mode-$repeat.stderr"
      test ! -s "$out/$leaf-$mode-$repeat.stderr"
    done
    cmp "$out/$leaf-$mode-1.stdout" "$out/$leaf-$mode-2.stdout"
  done
done
cmp "$out/normal-1.stdout" "$out/sanitize-1.stdout"
for leaf in p1 o2; do cmp "$out/$leaf-normal-1.stdout" "$out/$leaf-sanitize-1.stdout"; done
cmp "$out/f32-normal-1.stdout" "$out/f32-sanitize-1.stdout"
cat "$out/normal-1.stdout"
printf 'CPU4 candidate native gate PASS: normal/repeat/sanitizers,17 fail-stop stages,feature-off object parity\n'
