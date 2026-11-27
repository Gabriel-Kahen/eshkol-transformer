#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
[[ $# == 1 && ! -e "$1" && ! -L "$1" ]] || {
  printf '%s\n' "usage: $0 FRESH_EVIDENCE_DIRECTORY" >&2
  exit 2
}
mkdir -p -- "$1"
out="$(realpath -- "$1")"
cc="${CC:-clang}"
flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -Wconversion -Wsign-conversion
       -ffp-contract=off -fexcess-precision=standard -frounding-math
       -fno-fast-math -Iinclude -Inative -Itests/mod5)
sources=(tests/mod5/test_mod5_v256_c2_transfer.c tests/mod5/mod5_v256_c2_transfer.c
         native/f32_tensor.c native/kernel_abi.c)
"$cc" --version >"$out/compiler.txt"
git show -s --format='commit %H%ntree %T' HEAD >"$out/revision.txt"
sha256sum "${sources[@]}" tests/mod5/mod5_v256_c2_transfer.h >"$out/source.sha256"
for variant in normal sanitize; do
  optimization=(-O2)
  if [[ "$variant" == sanitize ]]; then
    optimization=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
  fi
  "$cc" "${flags[@]}" "${optimization[@]}" "${sources[@]}" -lm -o "$out/$variant"
  ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
    "$out/$variant" >"$out/$variant.stdout" 2>"$out/$variant.stderr"
  [[ ! -s "$out/$variant.stderr" ]]
done
cmp "$out/normal.stdout" "$out/sanitize.stdout"
grep -Fx 'MOD5 V256 I2 gradient transfer PASS' "$out/normal.stdout"
(cd "$out" && sha256sum compiler.txt revision.txt source.sha256 \
  normal normal.stdout normal.stderr sanitize sanitize.stdout sanitize.stderr >SHA256SUMS)
