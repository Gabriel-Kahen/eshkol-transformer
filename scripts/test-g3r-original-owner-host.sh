#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
tmp=$(mktemp -d)
trap 'rm -rf -- "$tmp"' EXIT
cc=${CC:-cc}
export PYTHONDONTWRITEBYTECODE=1
python3 "$root/tests/g3r/test_original_owner_coordinator_source.py"
python3 "$root/tests/g3r/test_tr3_held_o2_snapshot_recheck_source.py"
python3 "$root/tests/g3r/test_held_o2_count_source.py"
bash "$root/scripts/test-g3r-trainer-g3-save-host.sh"
flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -Wconversion
  -Wsign-conversion -Wshadow -ffunction-sections -fdata-sections
  -I "$root/include" -I "$root/native" -I "$root/src")
"$cc" "${flags[@]}" -c "$root/native/o2_optimizer.c" -o "$tmp/o2-off.o"
if nm -g --defined-only "$tmp/o2-off.o" | grep -F \
  'et_g3r_private_o2_save_begin_v1'; then
  echo 'feature-off O2 optimizer exports held-save entry' >&2
  exit 1
fi
"$cc" "${flags[@]}" -DET_O2_NATIVE_HELPERS_ONLY=1 \
  -c "$root/native/o2_wave2_package_bridge.c" -o "$tmp/o2-bridge-off.o"
if nm -g --defined-only "$tmp/o2-bridge-off.o" | grep -F \
  'et_g3r_private_trainer_o2_save_begin_v1'; then
  echo 'feature-off O2 bridge exports coordinator hold entry' >&2
  exit 1
fi
"$cc" "${flags[@]}" -DET_O2_NATIVE_HELPERS_ONLY=1 \
  -DET_G3R_SAME_OWNER_SAVE_PRIVATE=1 \
  -DET_G3R_CANDIDATE_RETIRE_PRIVATE=1 \
  "$root/native/o2_wave2_package_bridge.c" \
  "$root/tests/g3r/original_owner_o2_bridge_test.c" \
  -Wl,--gc-sections -o "$tmp/o2-bridge-test"
"$tmp/o2-bridge-test"
"$cc" "${flags[@]}" -Wno-sign-conversion \
  -DET_F32_TENSOR_TESTING=1 -DET_G3R_CANDIDATE_RETIRE_PRIVATE=1 \
  -c "$root/native/f32_tensor.c" -o "$tmp/f32.o"
"$cc" "${flags[@]}" -c "$root/native/kernel_abi.c" -o "$tmp/kernel.o"
"$cc" "${flags[@]}" "$root/tests/g3r/same_owner_o2_hold_test.c" \
  "$tmp/f32.o" "$tmp/kernel.o" -o "$tmp/o2-hold-test"
"$tmp/o2-hold-test"
