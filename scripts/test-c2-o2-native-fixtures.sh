#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cc="${CC:-cc}"
out="${1:-$(mktemp -d)}"
mkdir -p "$out"
flags=(-std=c11 -O1 -Wall -Wextra -Werror -Wpedantic -Wconversion
  -Wsign-conversion -Wshadow -I "$root/include" -I "$root/native")

# Both private surfaces must disappear when their feature macros are absent.
"$cc" "${flags[@]}" -c "$root/native/o2_optimizer.c" -o "$out/o2-off.o"
"$cc" "${flags[@]}" -c "$root/native/c2_save_image_receipt.c" \
  -o "$out/image-off.o"
if nm -g --defined-only "$out/o2-off.o" | grep -E 'et_g3r_private_o2_save_(begin|owner_match)_v1'; then
  echo 'O2 feature-off object exports held-save API' >&2
  exit 1
fi
if nm -g --defined-only "$out/image-off.o" | grep -F 'et_c2_private_save_image_adopt_v1'; then
  echo 'C2 feature-off object exports image receipt API' >&2
  exit 1
fi

for mode in normal repeat san; do
  mode_flags=("${flags[@]}")
  if [[ "$mode" == san ]]; then
    mode_flags+=(-g -fno-omit-frame-pointer -fsanitize=address,undefined,leak)
  fi
  mkdir -p "$out/$mode"
  # Pre-existing f32_parameter_save_witness_begin_v1 enum-to-int conversion
  # warns under -Wsign-conversion. Keep the new fixture and other sources strict.
  "$cc" "${mode_flags[@]}" -Wno-sign-conversion \
    -DET_F32_TENSOR_TESTING=1 -DET_G3R_CANDIDATE_RETIRE_PRIVATE=1 \
    -c "$root/native/f32_tensor.c" -o "$out/$mode/f32.o"
  "$cc" "${mode_flags[@]}" -c "$root/native/kernel_abi.c" \
    -o "$out/$mode/kernel.o"
  "$cc" "${mode_flags[@]}" -DET_CHECKPOINT_IO_TESTING=1 \
    -c "$root/native/checkpoint_io.c" -o "$out/$mode/io.o"
  "$cc" "${mode_flags[@]}" -DET_C2_SAVE_IMAGE_RECEIPT_PRIVATE=1 \
    -DET_C2_SAVE_IMAGE_RECEIPT_TESTING=1 \
    -c "$root/native/c2_save_image_receipt.c" -o "$out/$mode/image.o"
  "$cc" "${mode_flags[@]}" "$root/tests/g3r/same_owner_o2_hold_test.c" \
    "$out/$mode/f32.o" "$out/$mode/kernel.o" -o "$out/$mode/o2"
  "$cc" "${mode_flags[@]}" -DET_C2_SAVE_IMAGE_RECEIPT_PRIVATE=1 \
    -DET_C2_SAVE_IMAGE_RECEIPT_TESTING=1 -DET_CHECKPOINT_IO_TESTING=1 \
    "$root/tests/c2/save_image_receipt_native.c" \
    "$out/$mode/image.o" "$out/$mode/io.o" -o "$out/$mode/image"
  for fixture in o2 image; do
    ASAN_OPTIONS=detect_leaks=1 timeout --foreground --signal=TERM \
      --kill-after=5s 120s "$out/$mode/$fixture" \
      >"$out/$mode/$fixture.stdout" 2>"$out/$mode/$fixture.stderr"
    test ! -s "$out/$mode/$fixture.stderr"
  done
  grep -Fx 'same-owner O2 hold PASS: 375 checks' \
    "$out/$mode/o2.stdout" >/dev/null
  grep -Fx 'C2 private image receipt native checks passed' \
    "$out/$mode/image.stdout" >/dev/null
done
template='eshkol-c2-image-receipt-XXXXXX'
test "$(env -u TMPDIR "$out/normal/image" --print-temp-template)" = "/tmp/$template"
test "$(TMPDIR= "$out/normal/image" --print-temp-template)" = "/tmp/$template"
long_tmp="$out/normal/$(printf '%0130d' 0)"
mkdir -p "$long_tmp"
test "$(TMPDIR="$long_tmp" "$out/normal/image" --print-temp-template)" = \
  "$long_tmp/$template"
TMPDIR="$long_tmp" "$out/normal/image" >"$out/normal/image-long.stdout" \
  2>"$out/normal/image-long.stderr"
test ! -s "$out/normal/image-long.stderr"
cmp "$out/normal/image.stdout" "$out/normal/image-long.stdout"
rmdir "$long_tmp"
for fixture in o2 image; do
  cmp "$out/normal/$fixture.stdout" "$out/repeat/$fixture.stdout"
  cmp "$out/normal/$fixture.stdout" "$out/san/$fixture.stdout"
done
printf 'C2/O2 native fixtures PASS (normal, repeat, sanitizer, feature off)\n'
