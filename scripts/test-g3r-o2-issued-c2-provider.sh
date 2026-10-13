#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

[[ $# == 1 && "$1" == /* && ! -e "$1" && ! -L "$1" ]] || \
  die "usage: test-g3r-o2-issued-c2-provider.sh /fresh/absolute/output"
for command in ar cmp diff python3 timeout; do require_command "$command"; done
verify_toolchain
cc=
cxx=
resolve_provenance_compilers cc cxx \
  "${CC:-$(lock_value supported_cc)}" "${CXX:-$(lock_value supported_cxx)}"
runner="$(eshkol_build_dir)/eshkol-run"
output=$1
mkdir -p "$output/runtime" "$output/tmp" "$output/cache"
export TMPDIR="$output/tmp"

(cd "$PROJECT_ROOT" && PYTHONDONTWRITEBYTECODE=1 \
  python3 -m tests.d2.prepare_public_resources --output "$output/resources")

cflags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -fPIC -fvisibility=hidden
        -fno-common -fstack-protector-all -I "$PROJECT_ROOT/include"
        -I "$PROJECT_ROOT/native" -DET_F32_TENSOR_TESTING -DET_O2_TESTING)
"$cc" "${cflags[@]}" -DET_I2_NATIVE_HELPERS_ONLY \
  -DET_C2_I2_MODEL_COPY \
  -c "$PROJECT_ROOT/native/i2_wave2_package_bridge.c" \
  -o "$output/runtime/i2_wave2_native_bridge.o"
"$cc" "${cflags[@]}" -DET_O2_NATIVE_HELPERS_ONLY \
  -DET_C2_O2_RECONSTRUCT_BRIDGE -DET_G3R_SAME_OWNER_SAVE_PRIVATE \
  -c "$PROJECT_ROOT/native/o2_wave2_package_bridge.c" \
  -o "$output/runtime/o2_wave2_package_bridge.o"
"$cc" "${cflags[@]}" -DET_P1_TRUSTED_BUILD=1 \
  -c "$PROJECT_ROOT/native/p1_identity.c" \
  -o "$output/runtime/p1_identity.o"
for source in data_io checkpoint_io kernel_abi t1_i64_shell f32_tensor \
              i64_tensor d2_native c2_x1_canonical; do
  "$cc" "${cflags[@]}" -c "$PROJECT_ROOT/native/${source}.c" \
    -o "$output/runtime/${source}.o"
done
"$cc" "${cflags[@]}" \
  -c "$PROJECT_ROOT/tests/g3r/o2_issued_c2_provider_bridge.c" \
  -o "$output/runtime/o2_issued_c2_provider_bridge.o"
ar rcsD "$output/runtime/libeshkol_g3r_o2_issued_c2_provider.a" \
  "$output/runtime"/*.o

for label in a b; do
  mkdir -p "$output/$label"
  (cd "$output/$label" && env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR \
    ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="$output/cache/$label" \
    ESHKOL_LIB_DIR="$PROJECT_ROOT/lib" ESHKOL_CXX_COMPILER="$cxx" \
    timeout --foreground --signal=TERM --kill-after=5s 420s "$runner" \
      --strict-types --optimize 0 --no-stdlib \
      -I "$PROJECT_ROOT/internal/p1/lib" \
      -I "$PROJECT_ROOT/internal/c1/lib" \
      -I "$PROJECT_ROOT/internal/t2/lib" \
      -I "$PROJECT_ROOT/internal/t1/lib" \
      -I "$PROJECT_ROOT/internal/d2/lib" \
      -I "$PROJECT_ROOT/src" -I "$PROJECT_ROOT/lib" \
      -I "$PROJECT_ROOT/native" -L "$output/runtime" \
      --lib eshkol_g3r_o2_issued_c2_provider \
      "$PROJECT_ROOT/tests/g3r/o2_issued_c2_provider_runtime.esk" \
      -o "$output/$label/provider" \
      >"$output/$label/compile.stdout" \
      2>"$output/$label/compile.stderr")
  test ! -s "$output/$label/compile.stderr"
  ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM --kill-after=5s \
    180s "$output/$label/provider" \
    "$PROJECT_ROOT/tests/t2/fixtures/bpe_tokenizer_v1.tsv" \
    "$output/resources" >"$output/$label/run.stdout" \
    2>"$output/$label/run.stderr"
  test ! -s "$output/$label/run.stderr"
  grep -E '^O2 ISSUED C2 PROVIDER PASS: [0-9]+ checks$' \
    "$output/$label/run.stdout" >/dev/null
done
cmp "$output/a/provider" "$output/b/provider"
cmp "$output/a/run.stdout" "$output/b/run.stdout"
printf 'O2 issued C2 provider gate PASS: %s\n' "$output"
