#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
for command in ar cmp diff python3 timeout; do require_command "${command}"; done
cc="${CC:-/usr/bin/clang}"; cxx="${CXX:-/usr/bin/clang++}"
runner="$(eshkol_build_dir)/eshkol-run"
tmp="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-c2-owner.XXXXXX")"
cleanup() {
  local status=$?
  if (( status != 0 )); then
    find "${tmp}" -type f \( -name '*.stdout' -o -name '*.stderr' \) \
      -print -exec sed -n '1,240p' {} \; >&2 || true
  fi
  rm -rf -- "${tmp}"
  return "${status}"
}
trap cleanup EXIT

(cd "${PROJECT_ROOT}" && PYTHONDONTWRITEBYTECODE=1 \
  python3 -m tests.d2.prepare_public_resources --output "${tmp}/resources-a" && \
  PYTHONDONTWRITEBYTECODE=1 \
  python3 -m tests.d2.prepare_public_resources --output "${tmp}/resources-b")
diff -ru "${tmp}/resources-a" "${tmp}/resources-b"

runtime="${tmp}/runtime"
mkdir -p "${runtime}"
cflags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -fPIC -fvisibility=hidden
        -fno-common -fstack-protector-all -I "${PROJECT_ROOT}/include"
        -I "${PROJECT_ROOT}/native" -DET_F32_TENSOR_TESTING -DET_O2_TESTING)
"${cc}" "${cflags[@]}" -DET_I2_NATIVE_HELPERS_ONLY \
  -DET_C2_I2_MODEL_COPY \
  -c "${PROJECT_ROOT}/native/i2_wave2_package_bridge.c" \
  -o "${runtime}/i2_wave2_native_bridge.o"
"${cc}" "${cflags[@]}" -DET_O2_NATIVE_HELPERS_ONLY \
  -DET_C2_O2_RECONSTRUCT_BRIDGE \
  -c "${PROJECT_ROOT}/native/o2_wave2_package_bridge.c" \
  -o "${runtime}/o2_wave2_package_bridge.o"
"${cc}" "${cflags[@]}" -DET_P1_TRUSTED_BUILD=1 \
  -c "${PROJECT_ROOT}/native/p1_identity.c" -o "${runtime}/p1_identity.o"
for source in data_io checkpoint_io kernel_abi t1_i64_shell f32_tensor \
              i64_tensor d2_native o2_optimizer c2_x1_canonical; do
  "${cc}" "${cflags[@]}" -c "${PROJECT_ROOT}/native/${source}.c" \
    -o "${runtime}/${source}.o"
done
"${cc}" "${cflags[@]}" -c \
  "${PROJECT_ROOT}/tests/c2/c2_training_state_owner_test_bridge.c" \
  -o "${runtime}/c2_training_state_owner_test_bridge.o"
ar rcsD "${runtime}/libeshkol_transformer_c2_owner.a" "${runtime}"/*.o

compile() {
  local label=$1
  mkdir -p "${tmp}/${label}"
  if ! (cd "${tmp}/${label}" && env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR \
    ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="${tmp}/cache-${label}" \
    ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" ESHKOL_CXX_COMPILER="${cxx}" \
    timeout --foreground --signal=TERM --kill-after=5s 420s "${runner}" \
      --strict-types --optimize 0 --no-stdlib \
      -I "${PROJECT_ROOT}/internal/p1/lib" \
      -I "${PROJECT_ROOT}/internal/c1/lib" \
      -I "${PROJECT_ROOT}/internal/t2/lib" \
      -I "${PROJECT_ROOT}/internal/t1/lib" \
      -I "${PROJECT_ROOT}/internal/d2/lib" \
      -I "${PROJECT_ROOT}/src" -I "${PROJECT_ROOT}/lib" \
      -I "${PROJECT_ROOT}/native" -L "${runtime}" \
      --lib eshkol_transformer_c2_owner \
      "${PROJECT_ROOT}/tests/c2/c2_training_state_owner_runtime.esk" \
      -o "${tmp}/${label}/owner" \
      >"${tmp}/${label}/compile.stdout" 2>"${tmp}/${label}/compile.stderr"); then
    cat "${tmp}/${label}/compile.stdout" >&2
    cat "${tmp}/${label}/compile.stderr" >&2
    return 1
  fi
  if test -s "${tmp}/${label}/compile.stderr"; then
    cat "${tmp}/${label}/compile.stderr" >&2
    return 1
  fi
}
compile a
compile b
cmp "${tmp}/a/owner" "${tmp}/b/owner"
for label in a b; do
  ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM --kill-after=5s 180s \
    "${tmp}/${label}/owner" \
    "${PROJECT_ROOT}/tests/t2/fixtures/bpe_tokenizer_v1.tsv" \
    "${tmp}/resources-a" >"${tmp}/${label}/run.stdout" \
    2>"${tmp}/${label}/run.stderr"
  test ! -s "${tmp}/${label}/run.stderr"
  grep -E '^C2 TRAINING STATE OWNER PASS: [0-9]+ checks$' \
    "${tmp}/${label}/run.stdout" >/dev/null
done
cmp "${tmp}/a/run.stdout" "${tmp}/b/run.stdout"

for source in o2_optimizer f32_tensor kernel_abi; do
  "${cc}" "${cflags[@]}" -O1 -fsanitize=address,undefined \
    -fno-omit-frame-pointer -c "${PROJECT_ROOT}/native/${source}.c" \
    -o "${tmp}/san-${source}.o"
done
"${cc}" -fsanitize=address,undefined "${tmp}/san-o2_optimizer.o" \
  "${tmp}/san-f32_tensor.o" "${tmp}/san-kernel_abi.o" -lm \
  "${PROJECT_ROOT}/tests/o2/test_optimizer_native.c" \
  "${cflags[@]}" -o "${tmp}/o2-san"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
  timeout --foreground --signal=TERM --kill-after=5s 300s "${tmp}/o2-san" \
  >"${tmp}/o2-san.stdout"
grep -Fx 'O2 native adversarial PASS: 6201 checks' "${tmp}/o2-san.stdout" >/dev/null

cmp "${PROJECT_ROOT}/native/c2_training_state_source_closure.txt" \
  <(printf '%s\n' \
    native/d2_wave2_root.esk native/t2_wave2_root.esk native/t1_wave1_root.esk \
    src/eshkol_transformer/token_shard.esk \
    internal/t1/lib/transformer/tokenizer_internal.esk \
    internal/t1/lib/t1_tokenizer_core.esk native/e1b_error_consumer_private.esk \
    lib/transformer/error_internal.esk lib/transformer/error_core.esk \
    native/x1_config_private.esk internal/p1/lib/transformer/module.esk \
    internal/c1/lib/transformer/persistence_policy_internal.esk \
    internal/c1/lib/transformer/checkpoint_internal.esk internal/c1/lib/c1_sha256.esk \
    native/i2_wave2_extension.esk internal/t2/lib/t2_bpe_core.esk \
    native/d2_semantic_core.esk internal/d2/lib/d2_dataset.esk \
    native/c2_d2_cursor_pair_extension.esk native/o2_wave2_extension.esk \
    native/c2_o2_reconstruct_extension.esk native/c2_o2_encode_extension.esk \
    native/c2_model_encode_extension.esk native/c2_x1_canonical_extension.esk \
    native/c2_training_state_extension.esk)
printf 'C2 TRAINING STATE OWNER PASS: strict repeated AOT/runtime, ownership/failpoints/topology, deterministic fixtures, ASan/UBSan, source closure\n'
