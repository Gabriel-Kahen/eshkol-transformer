#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
scratch="$(mktemp -d)"
trap 'rm -rf -- "$scratch"' EXIT
cc="${CC:-clang}"

python3 -B - "$scratch/t1-mapping.bin" <<'PY'
from pathlib import Path
import sys
from tests.t1.reference import Identity, decode_artifact, encode_artifact, encode_bytes

pinned = "sha256:eshkol-byte-tokenizer-v1:aabb31f49216963582383a00fd2e85d7c20b658d5bef0e7945344ed6bfe50704"
parsed = decode_artifact(encode_artifact(Identity()))
alphabet = bytes(range(256))
mapping = bytes(encode_bytes(parsed.identity, alphabet))
if parsed.fingerprint != pinned or mapping != alphabet:
    raise SystemExit("G3-C4 T1 raw V256 mapping differs")
Path(sys.argv[1]).write_bytes(mapping)
PY

flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -ffp-contract=off
       -fexcess-precision=standard -fno-fast-math -frounding-math
       -Iinclude -Inative -Isrc -Isrc/eshkol_transformer)
sources=(tests/eval5/test_g3c4_live_mc.c native/kernel_abi.c native/eval5_benchmark.c
         native/n3k_primitives_provider.c native/g3c4_primitives_provider.c
         native/g3s_sampling_provider.c native/a2_kv_cache.c)

"$cc" "${flags[@]}" -O2 "${sources[@]}" -lm -o "$scratch/test"
"$scratch/test" "$scratch/t1-mapping.bin" "$scratch/normal.bits" >"$scratch/normal.out"
"$scratch/test" "$scratch/t1-mapping.bin" "$scratch/repeat.bits" >"$scratch/repeat.out"
"$cc" "${flags[@]}" -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer \
  "${sources[@]}" -lm -o "$scratch/test.san"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  "$scratch/test.san" "$scratch/t1-mapping.bin" "$scratch/sanitize.bits" >"$scratch/sanitize.out"
for ext in bits out; do
  cmp "$scratch/normal.$ext" "$scratch/repeat.$ext"
  cmp "$scratch/normal.$ext" "$scratch/sanitize.$ext"
done
cat "$scratch/normal.out"
printf 'EVAL5 live MC native normal/repeat/ASan/UBSan/LSan trace parity: PASS\n'
