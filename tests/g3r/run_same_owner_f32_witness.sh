#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$root"
out=$(mktemp -d /tmp/g3r-f32-witness.XXXXXX)
printf 'retained test output: %s\n' "$out"

defs=(-DET_F32_TENSOR_TESTING=1 -DET_G3R_SAME_OWNER_SAVE_PRIVATE
  -DET_G3R_G3T_SAVE_LEDGER_PRIVATE -DET_G3T_PREFILL_SAMPLE_PRIVATE
  -DET_G3T_FINAL_PUBLICATION_PRIVATE -DET_G3T_ZERO_BUDGET_PRIVATE
  -DET_G3T_P2_ZERO_BUDGET_PRIVATE -DET_G3T_MANUAL_P1_PREFILL_PRIVATE
  -DET_G3T_MANUAL_P2_PREFILL_PRIVATE -DET_G3T_MANUAL_DECODE_PRIVATE
  -DET_G3T_MANUAL_LOGITS_PRIVATE -DET_G3T_OWNED_TOKEN_INPUT_PRIVATE
  -DET_G3T_INPUT_FROM_T1_PRIVATE -DET_G3T_OUTPUT_TEXT_PRIVATE
  -DET_I64_TENSOR_STORAGE_QUERY_PRIVATE -DET_A2_KV_CACHE_STORAGE_QUERY_PRIVATE
  -DET_G3T_OUTPUT_IDS_CLONE_PRIVATE -DET_G3T_OUTPUT_LENGTHS_CLONE_PRIVATE
  -DET_G3T_OUTPUT_CACHE_LENGTHS_CLONE_PRIVATE
  -DET_G3T_OUTPUT_RNG_CLONE_PRIVATE -DET_G3T_GENERATOR_RNG_PRIVATE
  -DET_G3T_RECORD_RNG_IMPORT_PRIVATE)
sources=(tests/g3r/same_owner_combined_test.c native/g3r_same_owner_save.c
  native/kernel_abi.c native/data_io.c native/checkpoint_io.c
  src/eshkol_transformer/m3_i64_integration.c
  src/eshkol_transformer/m3_call_f32_integration.c native/t1_i64_shell.c
  native/a2_kv_cache.c native/n2_primitives_provider.c
  native/n3k_primitives_provider.c native/a2_attention_provider.c
  native/g3n_primitives_provider.c native/g3s_sampling_provider.c)

cc -std=c11 -O0 -I. -Iinclude -Inative -Isrc "${defs[@]}" \
  "${sources[@]}" -lm -o "$out/normal"
"$out/normal"
cc -std=c11 -O0 -g -fsanitize=address,undefined -fno-omit-frame-pointer \
  -I. -Iinclude -Inative -Isrc "${defs[@]}" \
  "${sources[@]}" -lm -o "$out/sanitized"
ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=halt_on_error=1 "$out/sanitized"

mkdir "$out/basis"
git archive 3f596c783745084950ffed905e987278a5f91d97 | \
  tar -x -C "$out/basis"
cc -std=c11 -O0 -Iinclude -Inative -c native/f32_tensor.c \
  -o "$out/current-f32-off.o"
(cd "$out/basis" && cc -std=c11 -O0 -Iinclude -Inative \
  -c native/f32_tensor.c -o "$out/basis-f32-off.o")
cmp "$out/current-f32-off.o" "$out/basis-f32-off.o"
cc -std=c11 -O0 -Iinclude -Inative -Isrc \
  -c native/g3r_same_owner_save.c -o "$out/coordinator-off.o"
test -z "$(nm -g --defined-only "$out/coordinator-off.o")"
printf 'normal, ASan/UBSan, F32 OFF byte parity, coordinator OFF symbols: PASS\n'
