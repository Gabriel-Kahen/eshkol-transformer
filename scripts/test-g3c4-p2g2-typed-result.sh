#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cc="${CC:-clang}"
out="${1:-$(mktemp -d)}"
mkdir -p "$out"
python3 - "$root" <<'PY_SOURCE_CLOSURE'
from pathlib import Path
import sys
root = Path(sys.argv[1])
base = (root / "native/g3c4_p2g2_terminal_ids_owner_source_closure.txt").read_text().splitlines()
added = [
    "tests/g3c4/test_p2_g2_terminal_ids_clone.c",
    "tests/g3c4/test_p2_g2_typed_result.c",
    "scripts/test-g3c4-p2g2-typed-result.sh",
    "native/g3c4_p2g2_typed_result_source_closure.txt",
    "docs/g3/G3_C4_P2_G2_TYPED_COPY_LEAF.md",
]
closure = (root / "native/g3c4_p2g2_typed_result_source_closure.txt").read_text().splitlines()
assert closure == base + added and len(closure) == len(set(closure))
assert all((root / name).is_file() for name in closure)
PY_SOURCE_CLOSURE
flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic
  -I "$root/include" -I "$root/native" -I "$root/src"
  -I "$root/src/eshkol_transformer"
  -DET_G3C4_T1_I1_EXACT_PAIR_PRIVATE
  -DET_I64_TENSOR_TESTING -DET_I64_TENSOR_STORAGE_QUERY_PRIVATE
  -DET_F32_TENSOR_STORAGE_QUERY_PRIVATE -DET_A2_KV_CACHE_TESTING
  -DET_A2_KV_CACHE_STORAGE_QUERY_PRIVATE
  -DET_A2_KV_CACHE_TERMINAL_WITNESS_PRIVATE)
sources=("$root/tests/g3c4/test_p2_g2_typed_result.c"
  "$root/src/eshkol_transformer/m3_i64_integration.c"
  "$root/tests/m3t/kernel_fail_allocator.c"
  "$root/native/t1_i64_shell.c"
  "$root/native/n3k_primitives_provider.c"
  "$root/native/g3c4_primitives_provider.c"
  "$root/native/g3s_sampling_provider.c"
  "$root/native/g3n_primitives_provider.c"
  "$root/native/n2_primitives_provider.c"
  "$root/native/a2_kv_cache.c")
owner_macros=(-DET_G3C4_CONTEXT_PRIVATE -DET_G3C4_NATIVE_PINS_PRIVATE
  -DET_G3C4_ACTIVE_CALL_PRIVATE -DET_G3C4_GENERATOR_PRIVATE
  -DET_G3C4_PROMPT_T1_BORROW_PRIVATE -DET_G3C4_PROVIDER_ROUTES_PRIVATE
  -DET_G3C4_FULL_PREFIX_FORWARD_PRIVATE -DET_G3C4_SAMPLER_TRANSPORT_PRIVATE
  -DET_G3C4_TOKEN_FRAME_PRIVATE -DET_G3C4_TOKEN_FORWARD_PRIVATE
  -DET_G3C4_PREFILL3_PRIVATE -DET_G3C4_PREFILL1_PRIVATE
  -DET_G3C4_PREFILL2_PRIVATE -DET_G3C4_PROMPT_PREFILL_PRIVATE
  -DET_G3C4_OUTPUT_RESERVATION_PRIVATE -DET_G3C4_LAST_LOGIT_FRAME_PRIVATE
  -DET_G3C4_OUTPUT_PREPARE_PRIVATE -DET_G3C4_OUTPUT_DECODE_IDS_PRIVATE
  -DET_G3C4_OUTPUT_TEXT_PRIVATE -DET_G3C4_P2_G1_PENDING_PRIVATE
  -DET_G3C4_P2_G1_PUBLICATION_PRIVATE -DET_G3C4_P2_G1_COPYOUT_PRIVATE
  -DET_G3C4_P2_G1_PUBLIC_RESULT_PRIVATE
  -DET_G3C4_P2_G2_FIRST_FRAME_PRIVATE -DET_G3C4_P2_G2_PREFIX_COMMIT_PRIVATE
  -DET_G3C4_P2_G2_CARRIER_BRIDGE_PRIVATE
  -DET_G3C4_P2_G2_EOS_FIRST_FRAME_PRIVATE
  -DET_G3C4_P2_G2_SECOND_FRAME_PRIVATE
  -DET_G3C4_P2_G2_TERMINAL_PRIVATE
  -DET_G3C4_P2_G2_TERMINAL_IDS_CLONE_PRIVATE)
for feature in off on; do
  enabled=()
  if [[ "$feature" == on ]]; then
    enabled=(-DET_G3C4_P2_G2_TYPED_RESULT_PRIVATE)
  fi
  "$cc" "${flags[@]}" "${owner_macros[@]}" "${enabled[@]}" \
    -O2 -c "$root/src/eshkol_transformer/g3c4_model_owner.c" \
    -o "$out/owner-$feature.o" \
    >"$out/owner-$feature.compile.stdout" \
    2>"$out/owner-$feature.compile.stderr"
  test ! -s "$out/owner-$feature.compile.stderr"
  nm -g --defined-only --format=posix "$out/owner-$feature.o" |
    awk 'NF >= 2 {print $1}' | LC_ALL=C sort -u \
    >"$out/owner-$feature.symbols"
done
comm -13 "$out/owner-off.symbols" "$out/owner-on.symbols" \
  >"$out/typed-only.symbols"
printf '%s\n' \
  et_g3c4_private_p2g2_typed_copy_v1 \
  et_g3c4_private_p2g2_typed_element_v1 \
  et_g3c4_private_p2g2_typed_ids_create_v1 \
  et_g3c4_private_p2g2_typed_length_v1 \
  et_g3c4_private_p2g2_typed_release_v1 \
  et_g3c4_private_p2g2_typed_scalar_create_v1 \
  >"$out/typed-expected.symbols"
cmp "$out/typed-expected.symbols" "$out/typed-only.symbols"
if "$cc" "${flags[@]}" -DET_G3C4_P2_G2_TYPED_RESULT_PRIVATE \
    -c "$root/src/eshkol_transformer/g3c4_model_owner.c" \
    -o "$out/isolated.o" >"$out/isolated.stdout" \
    2>"$out/isolated.stderr"; then
  echo 'isolated typed feature unexpectedly compiled' >&2
  exit 1
fi
grep -F 'typed P2/G2 results require the logical I1 clone prerequisite' \
  "$out/isolated.stderr" >/dev/null
wraps=(-Wl,--wrap=et_kernel_runtime_dispatch
  -Wl,--wrap=et_i64_tensor_borrow_view_v1
  -Wl,--wrap=et_a2_kv_cache_transaction_view_end_v1
  -Wl,--wrap=et_a2_kv_cache_transaction_abort_v1
  -Wl,--wrap=et_i64_tensor_copy_from_v1
  -Wl,--wrap=et_a2_kv_cache_transaction_commit_v1)
for mode in normal sanitize; do
  extra=(-O2)
  if [[ "$mode" == sanitize ]]; then
    extra=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
  fi
  "$cc" "${flags[@]}" "${extra[@]}" "${sources[@]}" \
    "${wraps[@]}" -lm -o "$out/$mode" \
    >"$out/$mode.compile.stdout" 2>"$out/$mode.compile.stderr"
  test ! -s "$out/$mode.compile.stderr"
  ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
    UBSAN_OPTIONS=halt_on_error=1 "$out/$mode" \
    >"$out/$mode.stdout" 2>"$out/$mode.stderr"
  test ! -s "$out/$mode.stderr"
done
"$out/normal" >"$out/repeat.stdout" 2>"$out/repeat.stderr"
test ! -s "$out/repeat.stderr"
cmp "$out/normal.stdout" "$out/repeat.stdout"
cmp "$out/normal.stdout" "$out/sanitize.stdout"
cat "$out/normal.stdout"
