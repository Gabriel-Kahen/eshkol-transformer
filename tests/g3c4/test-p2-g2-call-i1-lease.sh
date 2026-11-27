#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
evidence="${1:?Supply an evidence directory}"
mkdir -p "$evidence"
evidence="$(cd -- "$evidence" && pwd)"
build="$(mktemp -d "$evidence/build.XXXXXX")"
trap 'status=$?; if (( status == 0 )); then rm -rf -- "$build"; else
  printf "P2/G2 call I1 failure retained in %s\n" "$build" >&2; fi' EXIT
a2_witness="${G3C4_CALL_A2_READ_WITNESS:-0}"
[[ "$a2_witness" == 0 || "$a2_witness" == 1 ]] || exit 2
cc="${CC:-cc}"
"$cc" --version >"$evidence/compiler.txt"
cd "$ROOT"
flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -fvisibility=hidden
  -fstack-protector-all -ffp-contract=off -fexcess-precision=standard -fno-fast-math
  -I include -I native -I src -I src/eshkol_transformer
  -DET_I64_TENSOR_TESTING -DET_I64_TENSOR_STORAGE_QUERY_PRIVATE
  -DET_A2_KV_CACHE_TESTING -DET_A2_KV_CACHE_STORAGE_QUERY_PRIVATE)
owner_flags=(-DET_G3C4_NATIVE_OWNER_PRIVATE -DET_G3C4_CONTEXT_PRIVATE
  -DET_G3C4_NATIVE_PINS_PRIVATE -DET_G3C4_ACTIVE_CALL_PRIVATE -DET_G3C4_GENERATOR_PRIVATE
  -DET_G3C4_PROMPT_T1_BORROW_PRIVATE -DET_G3C4_PROVIDER_ROUTES_PRIVATE
  -DET_G3C4_FULL_PREFIX_FORWARD_PRIVATE -DET_G3C4_SAMPLER_TRANSPORT_PRIVATE
  -DET_G3C4_TOKEN_FRAME_PRIVATE -DET_G3C4_TOKEN_FORWARD_PRIVATE
  -DET_G3C4_PREFILL3_PRIVATE -DET_G3C4_PREFILL1_PRIVATE -DET_G3C4_PREFILL2_PRIVATE
  -DET_G3C4_PROMPT_PREFILL_PRIVATE -DET_G3C4_OUTPUT_RESERVATION_PRIVATE
  -DET_G3C4_LAST_LOGIT_FRAME_PRIVATE -DET_G3C4_OUTPUT_PREPARE_PRIVATE
  -DET_G3C4_OUTPUT_DECODE_IDS_PRIVATE -DET_G3C4_OUTPUT_TEXT_PRIVATE
  -DET_G3C4_P2_G1_PENDING_PRIVATE -DET_G3C4_P2_G2_FIRST_FRAME_PRIVATE)
feature=-DET_G3C4_P2_G2_CALL_I1_LEASE_PRIVATE
prerequisite='C4 P2/G2 call I1 lease requires P2/G2 first-frame ownership'
if [[ "$a2_witness" == 1 ]]; then
  feature=-DET_G3C4_P2_G2_CALL_A2_READ_PRIVATE
  prerequisite='C4 P2/G2 committed A2 read requires P2/G2 first-frame ownership'
fi
owner=src/eshkol_transformer/g3c4_model_owner.c
for mode in off on; do
  profile=("${owner_flags[@]}")
  [[ "$mode" == on ]] && profile+=("$feature")
  "$cc" "${flags[@]}" "${profile[@]}" -O2 -c "$owner" -o "$build/$mode.o"
  nm -g --defined-only --format=posix "$build/$mode.o" |
    awk '{print $1}' | LC_ALL=C sort >"$evidence/$mode.symbols"
done
cmp "$evidence/off.symbols" "$evidence/on.symbols"
if "$cc" "${flags[@]}" "$feature" -fsyntax-only "$owner" \
    >"$evidence/isolated.stdout" 2>"$evidence/isolated.stderr"; then
  printf '%s\n' "The isolated lease feature unexpectedly compiled" >&2
  exit 1
fi
grep -F "$prerequisite" \
  "$evidence/isolated.stderr" >/dev/null

sources=(src/eshkol_transformer/m3_i64_integration.c tests/m3t/kernel_fail_allocator.c
  native/t1_i64_shell.c native/n3k_primitives_provider.c
  native/g3c4_primitives_provider.c native/g3s_sampling_provider.c
  native/g3n_primitives_provider.c native/n2_primitives_provider.c native/a2_kv_cache.c)

run_fixture() {
  local label=$1 source=$2 expected=$3 mode output
  shift 3
  local -a optimization
  for mode in normal sanitize; do
    optimization=(-O2)
    if [[ "$mode" == sanitize ]]; then
      optimization=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
    fi
    output="$label-$mode"
    "$cc" "${flags[@]}" "${optimization[@]}" "$source" "${sources[@]}" "$@" \
      -Wl,--wrap=et_kernel_runtime_dispatch -Wl,--wrap=et_i64_tensor_borrow_view_v1 \
      -lm -o "$build/$output" >"$evidence/$output.compile.stdout" \
      2>"$evidence/$output.compile.stderr"
    ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
      "$build/$output" >"$evidence/$output.stdout" 2>"$evidence/$output.stderr"
    test ! -s "$evidence/$output.stderr"
    grep -E "$expected" "$evidence/$output.stdout" >/dev/null
  done
  "$build/$label-normal" >"$evidence/$label-repeat.stdout" 2>"$evidence/$label-repeat.stderr"
  test ! -s "$evidence/$label-repeat.stderr"
  cmp "$evidence/$label-normal.stdout" "$evidence/$label-repeat.stdout"
  cmp "$evidence/$label-normal.stdout" "$evidence/$label-sanitize.stdout"
  cat "$evidence/$label-normal.stdout"
}

terminal_flags=(-DET_A2_KV_CACHE_TERMINAL_WITNESS_PRIVATE
  -DET_F32_TENSOR_STORAGE_QUERY_PRIVATE
  -Wl,--wrap=et_a2_kv_cache_transaction_view_end_v1
  -Wl,--wrap=et_a2_kv_cache_transaction_abort_v1
  -Wl,--wrap=et_i64_tensor_copy_from_v1 -Wl,--wrap=et_a2_kv_cache_transaction_commit_v1)
if [[ "$a2_witness" == 1 ]]; then
  run_fixture a2-read tests/g3c4/test_p2_g2_call_a2_read.c \
    '^G3-C4 P2/G2 call A2 committed read PASS: checks=[1-9][0-9]*$' \
    "${terminal_flags[@]}"
  run_fixture a2-i1 tests/g3c4/test_p2_g2_call_a2_read.c \
    '^G3-C4 P2/G2 call A2 committed read PASS: checks=[1-9][0-9]*$' \
    "${terminal_flags[@]}" -DET_G3C4_P2_G2_CALL_I1_LEASE_PRIVATE \
    -DET_G3C4_P2_G1_PUBLICATION_PRIVATE
else
  run_fixture lease tests/g3c4/test_p2_g2_call_i1_lease.c \
    '^G3-C4 P2/G2 call I1 lease PASS: checks=[1-9][0-9]*$'
  run_fixture publication tests/g3c4/test_p2_g2_call_i1_publication.c \
    '^G3-C4 P2/G2 call I1 publication guard PASS: checks=[1-9][0-9]*$' \
    "${terminal_flags[@]}"
fi
