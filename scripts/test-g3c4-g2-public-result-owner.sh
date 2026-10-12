#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cc="${CC:-clang}"
tmp="$(mktemp -d)"
trap 'rm -rf -- "$tmp"' EXIT
common=(-std=c11 -Wall -Wextra -Werror -Wpedantic
        -ffunction-sections -fdata-sections
        -I "$root/include" -I "$root/native" -I "$root/src"
        -I "$root/src/eshkol_transformer"
        -DET_F32_TENSOR_STORAGE_QUERY_PRIVATE -DET_A2_KV_CACHE_TESTING
        -DET_I64_TENSOR_TESTING -DET_I64_TENSOR_STORAGE_QUERY_PRIVATE)
sources=("$root/tests/g3c4/test_g2_public_result_owner.c"
         "$root/src/eshkol_transformer/m3_i64_integration.c"
         "$root/tests/m3t/kernel_fail_allocator.c"
         "$root/native/t1_i64_shell.c"
         "$root/native/n3k_primitives_provider.c"
         "$root/native/g3c4_primitives_provider.c"
         "$root/native/g3s_sampling_provider.c"
         "$root/native/g3n_primitives_provider.c"
         "$root/native/n2_primitives_provider.c"
         "$root/native/a2_kv_cache.c")
for mode in normal sanitize; do
  flags=(-O1)
  if [[ "$mode" == sanitize ]]; then
    flags+=(-g -fsanitize=address,undefined -fno-omit-frame-pointer)
  fi
  "$cc" "${common[@]}" "${flags[@]}" "${sources[@]}" \
    -Wl,--gc-sections -Wl,--wrap=et_i64_tensor_copy_from_v1 -lm \
    -o "$tmp/$mode"
  ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
    UBSAN_OPTIONS=halt_on_error=1 "$tmp/$mode"
done
"$cc" "${common[@]}" -O1 -DET_G3C4_T1_I1_EXACT_PAIR_PRIVATE \
  -DET_G3C4_P2_G1_PUBLICATION_PRIVATE -DET_G3C4_P2_G1_COPYOUT_PRIVATE \
  -DET_G3C4_P2_G1_COORDINATOR_PRIVATE \
  -DET_G3C4_P2_G1_PUBLIC_RESULT_PRIVATE \
  -c "$root/tests/g3c4/p2_g1_shell_native.c" -o "$tmp/off.o"
if nm -g --defined-only "$tmp/off.o" | \
    grep -q 'et_g3c4_private_result_tensor_create_rank1_v1'; then
  echo 'G2 result constructor leaked into feature-off object' >&2
  exit 1
fi
echo 'G3-C4 G2 result owner feature-off PASS'
python3 - "$root" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
source = (root / "native/g3c4_public_g2_result_owner_extension.esk").read_text()
depth = 0
quoted = False
escaped = False
for char in source:
    if char == '"' and not escaped:
        quoted = not quoted
    if not quoted:
        depth += (char == "(") - (char == ")")
    escaped = char == "\\" and not escaped
assert depth == 0
assert source.index("(answer (if") < source.index("(g3c4-native-rank1-result-create")
assert source.index("(g3c4-native-rank1-result-create") < source.index("(vector-set! entry 2 'live)")
assert source.index("(vector-set! entry 2 'live)") < source.rindex("answer))))")
print("G3-C4 G2 result owner Eshkol publication order PASS (source only)")
PY
