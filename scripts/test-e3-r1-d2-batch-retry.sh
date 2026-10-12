#!/usr/bin/env bash
# Supported-container-only R1 D2 batch retry gate; launch through a sealed bundle.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
[[ $# == 1 && "$1" == /* && -d "$1" && ! -L "$1" ]] ||
  die "usage: $0 /ABSOLUTE_FRESH_OUTPUT_DIRECTORY"
out=$1
[[ -z "$(find "$out" -mindepth 1 -maxdepth 1 -print -quit)" ]] ||
  die "R1 gate output must be empty"
verify_toolchain
for tool in ar cmp diff env grep nm python3 timeout; do require_command "$tool"; done
provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
cc="$(tsv_value "$provenance" cc_path)"
cxx="$(tsv_value "$provenance" cxx_path)"
runner="$(eshkol_build_dir)/eshkol-run"
prefix="${PROJECT_ROOT}/tests/e3_private/r1_d2_batch_retry_package"

capture() {
  local name=$1 limit=$2 status
  shift 2
  set +e
  timeout --foreground --signal=TERM --kill-after=30s "${limit}s" "$@" \
    >"$out/${name}.stdout" 2>"$out/${name}.stderr"
  status=$?
  set -e
  printf '%s\n' "$status" >"$out/${name}.exit"
  [[ "$status" == 0 ]] || die "${name} failed with ${status}"
}

capture corpus 120 python3 -m tests.e3_reference.corpus --output "$out/corpus"
corpus="$out/corpus/packed-single"
[[ -d "$corpus" && -n "$(find "$corpus" -type f -print -quit)" ]] ||
  die "genuine D1/D2 packed corpus missing"
for mode in a b sanitized; do
  if [[ "$mode" == sanitized ]]; then kind=sanitize; else kind=normal; fi
  capture "package-$mode" 5400 /usr/bin/bash \
    "${PROJECT_ROOT}/scripts/build-e3-r1-d2-batch-retry-test.sh" \
    "$out/private-$mode" "$kind"
  evidence="$out/private-$mode/e3_retention_package.o.evidence"
  for pair in defined_symbols:global-defined public_exports:package-exports \
      public_strings:public-strings source_closure:source-closure \
      native_source_closure:native-source-closure native_objects:native-objects; do
    expected=${pair%%:*}; observed=${pair#*:}
    cmp "${prefix}_${expected}.txt" "$evidence/${observed}.txt"
  done
  if [[ "$mode" == sanitized ]]; then
    cmp "${prefix}_sanitized_instrumentation_undefined_symbols.txt" \
      "$evidence/sanitizer-undefined.txt"
    cmp "${prefix}_sanitized_noninstrumentation_undefined_symbols.txt" \
      "$evidence/noninstrumentation-undefined.txt"
  else
    cmp "${prefix}_undefined_symbols.txt" "$evidence/undefined.txt"
  fi
  grep -E '[[:space:]]LOCAL[[:space:]].*[[:space:]]et_e3_test_preinvoke_arm_v1$' \
    "$evidence/readelf-symbols.txt" >/dev/null ||
    die "R1 preinvoke arm is absent or externally exported"
  grep -E '[[:space:]]LOCAL[[:space:]].*[[:space:]]et_d2_batch_test_fail_stage_v1$' \
    "$evidence/readelf-symbols.txt" >/dev/null ||
    die "D2 batch fault selector is absent or externally exported"
  mkdir "$out/link-$mode"
  ar p "$out/private-$mode/libeshkol_transformer_e3_r1_d2_batch_retry.a" \
    e3_retention_package.o >"$out/link-$mode/package.o"
  cmp "$out/private-$mode/e3_retention_package.o" \
      "$out/link-$mode/package.o"
done
cmp "$out/private-a/e3_retention_package.o" \
    "$out/private-b/e3_retention_package.o"
cmp "$out/private-a/e3_retention_package.o.evidence/private.d" \
    "$out/private-b/e3_retention_package.o.evidence/private.d"
diff -r "$out/private-a/e3_retention_package.o.evidence/native-depfiles" \
        "$out/private-b/e3_retention_package.o.evidence/native-depfiles"

for mode in a b sanitized; do
  if [[ "$mode" == sanitized ]]; then
    flags=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
  else
    flags=(-O2)
  fi
  capture "arena-$mode" 120 "$cxx" -std=c++17 -Wall -Wextra -Werror \
    -fstack-protector-all -fPIC -fvisibility=hidden -fno-common \
    "${flags[@]}" -I "$(eshkol_source_dir)/lib/core" \
    -I "$(eshkol_source_dir)/inc" \
    -c "${PROJECT_ROOT}/tests/e3_private/retention_arena.cpp" \
    -o "$out/link-$mode/retention_arena.o"
  nm -g --defined-only "$out/link-$mode/retention_arena.o" | \
    grep -E '[[:space:]]T et_e3_retention_region_used_v1$' >/dev/null ||
    die "retention arena supplement is missing"
  ar rcsD "$out/link-$mode/libeshkol_transformer_e3_r1_d2_batch_retry_test.a" \
    "$out/link-$mode/package.o" "$out/link-$mode/retention_arena.o"
done
cmp "$out/link-a/libeshkol_transformer_e3_r1_d2_batch_retry_test.a" \
    "$out/link-b/libeshkol_transformer_e3_r1_d2_batch_retry_test.a"
cat >"$out/cxx-sanitized" <<WRAPPER
#!/usr/bin/env bash
exec $(printf '%q' "$cxx") -fsanitize=address,undefined -fno-omit-frame-pointer "\$@"
WRAPPER
chmod 0500 "$out/cxx-sanitized"

for mode in a b sanitized; do
  if [[ "$mode" == sanitized ]]; then selected_cxx="$out/cxx-sanitized";
  else selected_cxx="$cxx"; fi
  mkdir "$out/cache-$mode"
  capture "compile-$mode" 5400 env -u CPATH -u C_INCLUDE_PATH \
    -u CPLUS_INCLUDE_PATH -u OBJC_INCLUDE_PATH -u DEPENDENCIES_OUTPUT \
    -u SUNPRO_DEPENDENCIES -u GCC_EXEC_PREFIX -u COMPILER_PATH \
    -u LIBRARY_PATH -u CLANG_CONFIG_FILE -u CCC_OVERRIDE_OPTIONS \
    -u CCC_CC -u CCC_CXX -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR \
    ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="$out/cache-$mode" \
    ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" \
    ESHKOL_CXX_COMPILER="$selected_cxx" "$runner" \
    --strict-types --no-stdlib -O 2 -L "$out/link-$mode" \
    --lib eshkol_transformer_e3_r1_d2_batch_retry_test \
    "${PROJECT_ROOT}/tests/e3_private/r1_d2_batch_retry_runtime.esk" \
    -o "$out/runtime-$mode"
  [[ -x "$out/runtime-$mode" ]] || die "AOT executable missing"
  if nm -u "$out/runtime-$mode" | \
       grep -E '[[:space:]]U (et_|eshkol_)[A-Za-z0-9_]*(@|$)' >/dev/null; then
    die "R1 AOT executable retains project-private imports"
  fi
done
cmp "$out/runtime-a" "$out/runtime-b"
for mode in normal-a normal-b normal-repeat poison sanitized; do
  case "$mode" in
    normal-a|normal-repeat) binary=a; poison=0 ;;
    normal-b) binary=b; poison=0 ;;
    poison) binary=a; poison=1 ;;
    sanitized) binary=sanitized; poison=1 ;;
    *) die "unknown R1 runtime mode: $mode" ;;
  esac
  capture "runtime-$mode" 900 env ESHKOL_ARENA_POISON="$poison" \
    ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
    UBSAN_OPTIONS=halt_on_error=1 \
    "$out/runtime-$binary" "$corpus"
  [[ "$(cat "$out/runtime-$mode.stdout")" == E3-R1-D2-BATCH-RETRY-PASS ]] ||
    die "R1 runtime pass marker differs"
  [[ ! -s "$out/runtime-$mode.stderr" ]] ||
    die "R1 runtime emitted stderr"
done
for mode in normal-b normal-repeat poison sanitized; do
  cmp "$out/runtime-normal-a.stdout" "$out/runtime-$mode.stdout"
  cmp "$out/runtime-normal-a.stderr" "$out/runtime-$mode.stderr"
  cmp "$out/runtime-normal-a.exit" "$out/runtime-$mode.exit"
done
printf 'E3-R1-COMPILED-GATE-PASS\n'
