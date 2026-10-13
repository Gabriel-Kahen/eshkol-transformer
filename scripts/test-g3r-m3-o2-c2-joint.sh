#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

[[ $# == 1 && "$1" == /* && ! -e "$1" && ! -L "$1" ]] || \
  die "usage: test-g3r-m3-o2-c2-joint.sh /fresh/absolute/output"
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
  python3 -m tests.tr3b.prepare_fixture --output "$output/resources")

cflags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -fPIC -fvisibility=hidden
        -fno-common -fstack-protector-all -I "$PROJECT_ROOT/include"
        -I "$PROJECT_ROOT/native" -I "$PROJECT_ROOT/src"
        -DET_F32_TENSOR_TESTING -DET_O2_TESTING
        -DET_G3R_SAME_OWNER_SAVE_PRIVATE)
"$cc" "${cflags[@]}" -DET_I2_NATIVE_HELPERS_ONLY \
  -DET_C2_I2_MODEL_COPY -DET_M3T_PACKAGE_BUILD \
  -c "$PROJECT_ROOT/native/i2_wave2_package_bridge.c" \
  -o "$output/runtime/i2_wave2_native_bridge.o"
"$cc" "${cflags[@]}" -DET_O2_NATIVE_HELPERS_ONLY \
  -DET_C2_O2_RECONSTRUCT_BRIDGE -DET_G3R_SAME_OWNER_SAVE_PRIVATE \
  -c "$PROJECT_ROOT/native/o2_wave2_package_bridge.c" \
  -o "$output/runtime/o2_wave2_package_bridge.o"
"$cc" "${cflags[@]}" -DET_P1_TRUSTED_BUILD=1 \
  -c "$PROJECT_ROOT/native/p1_identity.c" \
  -o "$output/runtime/p1_identity.o"
for source in data_io checkpoint_io kernel_abi t1_i64_shell \
              i64_tensor d2_native c2_x1_canonical \
              n2_primitives_provider n3k_primitives_provider \
              a2_attention_provider c2_checkpoint_codec \
              c2_checkpoint_format c2_checkpoint_save_bridge; do
  "$cc" "${cflags[@]}" -c "$PROJECT_ROOT/native/${source}.c" \
    -o "$output/runtime/${source}.o"
done
"$cc" "${cflags[@]}" -DET_C2_CARRIER_FACTORIES -c \
  "$PROJECT_ROOT/native/k2_capabilities.c" \
  -o "$output/runtime/k2_capabilities.o"
"$cc" "${cflags[@]}" -c \
  "$PROJECT_ROOT/src/eshkol_transformer/m3t_f32_integration.c" \
  -o "$output/runtime/m3t_f32_integration.o"
"$cc" "${cflags[@]}" -c \
  "$PROJECT_ROOT/src/eshkol_transformer/m3t_transport.c" \
  -o "$output/runtime/m3t_transport.o"
"$cc" "${cflags[@]}" \
  -c "$PROJECT_ROOT/tests/g3r/m3_o2_c2_joint_bridge.c" \
  -o "$output/runtime/m3_o2_c2_joint_bridge.o"
ar rcsD "$output/runtime/libeshkol_g3r_m3_o2_c2_joint.a" \
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
      --lib eshkol_g3r_m3_o2_c2_joint \
      "$PROJECT_ROOT/tests/g3r/m3_o2_c2_joint_runtime.esk" \
      -o "$output/$label/provider" \
      >"$output/$label/compile.stdout" \
      2>"$output/$label/compile.stderr")
  test ! -s "$output/$label/compile.stderr"
  ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM --kill-after=5s \
    180s "$output/$label/provider" \
    "$output/resources" "$output/$label" >"$output/$label/run.stdout" \
    2>"$output/$label/run.stderr"
  test ! -s "$output/$label/run.stderr"
  grep -E '^M3 O2 C2 JOINT PROVIDER PASS: [0-9]+ checks$' \
    "$output/$label/run.stdout" >/dev/null
  test -s "$output/$label/joint.c2"
done
cmp "$output/a/provider" "$output/b/provider"
cmp "$output/a/run.stdout" "$output/b/run.stdout"
cmp "$output/a/joint.c2" "$output/b/joint.c2"
printf 'M3 O2 C2 joint gate PASS: %s\n' "$output"
