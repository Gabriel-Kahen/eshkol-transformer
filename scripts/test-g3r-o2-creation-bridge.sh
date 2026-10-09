#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
tmp=$(mktemp -d /tmp/g3r-o2-creation.XXXXXX)
printf 'test output: %s\n' "$tmp"
defs=(-DET_G3R_SAME_OWNER_SAVE_PRIVATE -DET_G3R_G3T_SAVE_LEDGER_PRIVATE
  -DET_G3T_PREFILL_SAMPLE_PRIVATE -DET_G3T_FINAL_PUBLICATION_PRIVATE
  -DET_G3T_ZERO_BUDGET_PRIVATE -DET_G3T_P2_ZERO_BUDGET_PRIVATE
  -DET_G3T_MANUAL_P1_PREFILL_PRIVATE -DET_G3T_MANUAL_P2_PREFILL_PRIVATE
  -DET_G3T_MANUAL_DECODE_PRIVATE -DET_G3T_MANUAL_LOGITS_PRIVATE
  -DET_G3T_OWNED_TOKEN_INPUT_PRIVATE -DET_G3T_INPUT_FROM_T1_PRIVATE
  -DET_G3T_OUTPUT_TEXT_PRIVATE -DET_I64_TENSOR_STORAGE_QUERY_PRIVATE
  -DET_A2_KV_CACHE_STORAGE_QUERY_PRIVATE
  -DET_G3T_OUTPUT_IDS_CLONE_PRIVATE
  -DET_G3T_OUTPUT_LENGTHS_CLONE_PRIVATE
  -DET_G3T_OUTPUT_CACHE_LENGTHS_CLONE_PRIVATE
  -DET_G3T_OUTPUT_RNG_CLONE_PRIVATE
  -DET_G3T_GENERATOR_RNG_PRIVATE
  -DET_G3T_RECORD_RNG_IMPORT_PRIVATE
  -DET_G3R_O2_CREATION_TESTING -DET_F32_TENSOR_TESTING -DET_O2_TESTING)
sources=(tests/g3r/o2_creation_bridge_test.c native/g3r_o2_creation_bridge.c
  native/kernel_abi.c native/data_io.c native/checkpoint_io.c
  src/eshkol_transformer/m3_i64_integration.c
  src/eshkol_transformer/m3_call_f32_integration.c
  native/t1_i64_shell.c native/a2_kv_cache.c
  native/n2_primitives_provider.c native/n3k_primitives_provider.c
  native/a2_attention_provider.c native/g3n_primitives_provider.c
  native/g3s_sampling_provider.c)
cc -std=c11 -O0 -I. -Iinclude -Inative -Isrc "${defs[@]}" \
  "${sources[@]}" -lm -o "$tmp/normal"
"$tmp/normal"
cc -std=c11 -O0 -g -fsanitize=address,undefined -fno-omit-frame-pointer \
  -I. -Iinclude -Inative -Isrc "${defs[@]}" "${sources[@]}" -lm \
  -o "$tmp/san"
ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=halt_on_error=1 "$tmp/san"
cc -std=c11 -Wall -Wextra -Werror -Wpedantic -I. -Iinclude -Inative \
  -Isrc -c native/g3r_o2_creation_bridge.c -o "$tmp/off.o"
test -z "$(nm -g --defined-only "$tmp/off.o")"
printf '#include "native/g3r_o2_creation_bridge.h"\n' | \
  cc -std=c11 -DET_G3R_SAME_OWNER_SAVE_PRIVATE -I. -Iinclude -Inative -Isrc \
  -x c -fsyntax-only - >"$tmp/partial.stdout" 2>"$tmp/partial.stderr" \
  && exit 17 || true
rg -q 'same-owner SAVE requires the G3R committed-history ledger' \
  "$tmp/partial.stderr"
printf 'OFF symbols empty; partial flag rejected\n'
