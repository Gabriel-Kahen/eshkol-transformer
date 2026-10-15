#!/usr/bin/env bash
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
out=${1:-$(mktemp -d)}
if [[ $# -eq 0 ]]; then
  trap 'rm -rf -- "$out"' EXIT
fi
mkdir -p "$out"
cc=${CC:-cc}
flags=(-std=c11 -O1 -Wall -Wextra -Werror -Wpedantic -Wconversion
  -Wsign-conversion -Wshadow -I "$root/include" -I "$root/native"
  -I "$root/src")

"$cc" "${flags[@]}" -c "$root/native/o2_optimizer.c" -o "$out/o2-off.o"
"$cc" "${flags[@]}" -DET_O2_NATIVE_HELPERS_ONLY=1 \
  -c "$root/native/o2_wave2_package_bridge.c" -o "$out/bridge-off.o"
if nm -g --defined-only "$out/o2-off.o" | grep -E \
  'et_g3r_private_o2_save_(begin|check|completed_updates|state|issued_match|issued_identity|end)_v1'; then
  echo 'feature-off O2 exposes held-save contract' >&2
  exit 1
fi
if nm -g --defined-only "$out/bridge-off.o" | grep -E \
  'et_g3r_private_(trainer_o2_save|o2_adopter)'; then
  echo 'feature-off bridge exposes held-save contract' >&2
  exit 1
fi
if "$cc" "${flags[@]}" -DET_G3R_SAME_OWNER_SAVE_PRIVATE=1 \
  -c "$root/native/o2_optimizer.c" -o "$out/missing-prerequisite.o" \
  >"$out/missing-prerequisite.stdout" 2>"$out/missing-prerequisite.stderr"; then
  echo 'held-save compiled without candidate-retire prerequisite' >&2
  exit 1
fi
grep -F 'same-owner O2 SAVE requires I2 owned-span provenance' \
  "$out/missing-prerequisite.stderr" >/dev/null

for mode in normal repeat san; do
  mode_flags=("${flags[@]}")
  if [[ "$mode" == san ]]; then
    mode_flags+=(-g -fno-omit-frame-pointer -fsanitize=address,undefined)
  fi
  mkdir -p "$out/$mode"
  # An existing enum conversion in f32_parameter_save_witness_begin_v1 warns
  # under -Wsign-conversion; keep the other sources strict.
  "$cc" "${mode_flags[@]}" -Wno-sign-conversion \
    -DET_F32_TENSOR_TESTING=1 -DET_G3R_CANDIDATE_RETIRE_PRIVATE=1 \
    -c "$root/native/f32_tensor.c" -o "$out/$mode/f32.o"
  "$cc" "${mode_flags[@]}" -c "$root/native/kernel_abi.c" \
    -o "$out/$mode/kernel.o"
  "$cc" "${mode_flags[@]}" "$root/tests/g3r/same_owner_o2_hold_test.c" \
    "$out/$mode/f32.o" "$out/$mode/kernel.o" -o "$out/$mode/hold-test"
  "$cc" "${mode_flags[@]}" -ffunction-sections -fdata-sections \
    -DET_O2_NATIVE_HELPERS_ONLY=1 -DET_G3R_SAME_OWNER_SAVE_PRIVATE=1 \
    -DET_G3R_CANDIDATE_RETIRE_PRIVATE=1 \
    "$root/native/o2_wave2_package_bridge.c" \
    "$root/tests/g3r/original_owner_o2_bridge_test.c" \
    -Wl,--gc-sections -o "$out/$mode/bridge-test"
  ASAN_OPTIONS=${ASAN_OPTIONS:-detect_leaks=0} "$out/$mode/hold-test" \
    >"$out/$mode/hold.stdout" 2>"$out/$mode/hold.stderr"
  ASAN_OPTIONS=${ASAN_OPTIONS:-detect_leaks=0} "$out/$mode/bridge-test" \
    >"$out/$mode/bridge.stdout" 2>"$out/$mode/bridge.stderr"
  test ! -s "$out/$mode/hold.stderr"
  test ! -s "$out/$mode/bridge.stderr"
  grep -Fx 'same-owner O2 hold PASS: 440 checks' \
    "$out/$mode/hold.stdout" >/dev/null
  grep -Fx 'G3R-ORIGINAL-O2-BRIDGE-PASS' \
    "$out/$mode/bridge.stdout" >/dev/null
done
cmp "$out/normal/hold.stdout" "$out/repeat/hold.stdout"
cmp "$out/normal/hold.stdout" "$out/san/hold.stdout"
cmp "$out/normal/bridge.stdout" "$out/repeat/bridge.stdout"
cmp "$out/normal/bridge.stdout" "$out/san/bridge.stdout"
printf 'G3R O2 held-save native PASS (normal, repeat, sanitizer, feature off)\n'
