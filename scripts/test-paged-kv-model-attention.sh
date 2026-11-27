#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root}"
scratch="$(mktemp -d)"
trap 'rm -rf -- "${scratch}"' EXIT
cc="${CC:-clang}"
flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -ffp-contract=off
       -fexcess-precision=standard -fno-fast-math -I include -I native
       -I src -I src/eshkol_transformer)
sources=(tests/paged_kv/test_model_attention.c tests/m3t/kernel_fail_allocator.c
         native/n3k_primitives_provider.c native/g3c4_primitives_provider.c
         native/g3s_sampling_provider.c native/a2_kv_cache.c native/paged_kv.c
         native/data4_pipeline.c)

PYTHONDONTWRITEBYTECODE=1 python3 - "${scratch}/mapping.bin" <<'PY'
from pathlib import Path
import sys

from tests.t1.reference import Identity, decode_artifact, encode_artifact, encode_bytes

# Match the baseline V256 fingerprint required by g3c4_call_entry_extension.esk.
baseline = "sha256:eshkol-byte-tokenizer-v1:aabb31f49216963582383a00fd2e85d7c20b658d5bef0e7945344ed6bfe50704"
parsed = decode_artifact(encode_artifact(Identity()))
octets = bytes(range(256))
mapping = bytes(encode_bytes(parsed.identity, octets))
if parsed.fingerprint != baseline or mapping != octets:
    raise SystemExit("G3-C4 raw V256 tokenizer reference mismatch")
Path(sys.argv[1]).write_bytes(mapping)
PY

"${cc}" "${flags[@]}" -O2 "${sources[@]}" \
  -Wl,--wrap=et_kernel_runtime_dispatch -lm -o "${scratch}/witness"
"${scratch}/witness" "${scratch}/mapping.bin" >"${scratch}/normal.out"
"${scratch}/witness" "${scratch}/mapping.bin" >"${scratch}/repeat.out"
"${cc}" "${flags[@]}" -O1 -g -fsanitize=address,undefined \
  -fno-omit-frame-pointer "${sources[@]}" \
  -Wl,--wrap=et_kernel_runtime_dispatch -lm -o "${scratch}/witness.san"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  "${scratch}/witness.san" "${scratch}/mapping.bin" >"${scratch}/san.out"
cmp "${scratch}/normal.out" "${scratch}/repeat.out"
cmp "${scratch}/normal.out" "${scratch}/san.out"
cat "${scratch}/normal.out"
printf 'PKV5 model attention normal/repeat/ASan/UBSan/LSan parity: PASS\n'
