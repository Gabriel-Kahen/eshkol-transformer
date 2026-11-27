#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
for command in ar cmp grep nm readelf sha256sum timeout; do
  require_command "$command"
done
cc= cxx=
resolve_provenance_compilers cc cxx \
  "${CC:-$(lock_value supported_cc)}" "${CXX:-$(lock_value supported_cxx)}"
runner="$(eshkol_build_dir)/eshkol-run"
out="${1:-$(project_build_dir)/k2-mod5-v256}"
mkdir -p "$out"
out="$(realpath "$out")"
cd "$PROJECT_ROOT"
export LC_ALL=C

flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic
  -Wconversion -Wsign-conversion -D_POSIX_C_SOURCE=200809L
  -fPIC -fvisibility=hidden -fno-common -fstack-protector-all
  -I include -I native -I "$(eshkol_source_dir)/inc")
profile=(-DET_K2_PRIVATE_MOD5_V256 -DET_I2_PRIVATE_MOD5_WIDTH16_STORAGE_COPY
  -DET_I2_PRIVATE_MOD5_WIDTH16_V256_STORAGE_COPY)
native_sources=(tests/k2/test_capabilities_native.c native/k2_capabilities.c
  native/kernel_abi.c native/f32_tensor.c)
"$cc" "${flags[@]}" -DET_K2_TESTING -O2 "${native_sources[@]}" -lm -o "$out/native-default"
for mode in normal sanitize; do
  instrumentation=(-O2)
  if [[ "$mode" == sanitize ]]; then
    instrumentation=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
  fi
  "$cc" "${flags[@]}" "${profile[@]}" -DET_K2_TESTING "${instrumentation[@]}" \
    "${native_sources[@]}" -lm -o "$out/native-v256-$mode"
  ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
    timeout --foreground --kill-after=5s 90s "$out/native-v256-$mode" \
    >"$out/native-v256-$mode.stdout"
  grep -Fxq 'K2 native capability checks passed' "$out/native-v256-$mode.stdout"
done
"$out/native-default" >"$out/native-default.stdout"
cmp "$out/native-default.stdout" "$out/native-v256-normal.stdout"
cmp "$out/native-v256-normal.stdout" "$out/native-v256-sanitize.stdout"

"$cc" "${flags[@]}" -c native/k2_capabilities.c -o "$out/native-default.o"
if nm -g "$out/native-default.o" | grep -E 'et_k2_.*mod5_v256|et_i2_.*v256'; then
  die "ordinary K2 object exposes the opt-in profile"
fi

mkdir -p "$out/objects" "$out/cache"
"$cc" "${flags[@]}" -DET_I2_NATIVE_HELPERS_ONLY \
  -c native/i2_wave2_package_bridge.c -o "$out/objects/i2_wave2_package_bridge.o"
"$cc" "${flags[@]}" -DET_P1_TRUSTED_BUILD=1 \
  -c native/p1_identity.c -o "$out/objects/p1_identity.o"
for source in data_io checkpoint_io kernel_abi i64_tensor t1_i64_shell; do
  "$cc" "${flags[@]}" -c "native/$source.c" -o "$out/objects/$source.o"
done
"$cc" "${flags[@]}" "${profile[@]}" -c native/f32_tensor.c -o "$out/objects/f32_tensor.o"
"$cc" "${flags[@]}" "${profile[@]}" -DET_K2_TESTING \
  -c native/k2_capabilities.c -o "$out/objects/k2_capabilities.o"
readelf -Ws "$out/objects/k2_capabilities.o" >"$out/k2-v256-symbols.txt"
awk '$5 == "GLOBAL" && $6 == "HIDDEN" &&
     $8 ~ /^et_k2_private_mod5_v256_/ {n++} END {exit n != 5}' \
  "$out/k2-v256-symbols.txt"
ar rcsD "$out/libeshkol_k2_mod5_v256_witness.a" "$out"/objects/*.o

env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR -u ESHKOL_AOT_MODULE_CACHE_DIR \
  ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="$out/cache" ESHKOL_CXX_COMPILER="$cxx" \
  ESHKOL_LIB_DIR="$PROJECT_ROOT/lib" \
  timeout --foreground --signal=TERM --kill-after=5s 600s \
  "$runner" --strict-types --optimize 0 --no-stdlib \
  -I "$PROJECT_ROOT/native" -I "$PROJECT_ROOT/src" -I "$PROJECT_ROOT/lib" \
  -I "$PROJECT_ROOT/internal/p1/lib" -I "$PROJECT_ROOT/internal/c1/lib" \
  -I "$PROJECT_ROOT/internal/t1/lib" -L "$out" --lib eshkol_k2_mod5_v256_witness \
  "$PROJECT_ROOT/tests/k2/mod5_v256_runtime.esk" -o "$out/witness" \
  >"$out/compile.stdout" 2>"$out/compile.stderr"
test ! -s "$out/compile.stderr"
for repeat in a b; do
  timeout --foreground --signal=TERM --kill-after=5s 90s \
    "$out/witness" >"$out/witness-$repeat.stdout" 2>"$out/witness-$repeat.stderr"
  test ! -s "$out/witness-$repeat.stderr"
done
cmp "$out/witness-a.stdout" "$out/witness-b.stdout"
grep -Fx 'K2 MOD5 V256 PASS 14' "$out/witness-a.stdout"
