#!/usr/bin/env bash
# Supported f31/LLVM21 container only. The host launcher owns heavy.lock.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
[[ $# == 3 && "$1" == /* && -d "$1" && ! -L "$1" &&
   "$2" =~ ^[0-9a-f]{40}$ && "$3" =~ ^[0-9a-f]{40}$ ]] ||
  die "usage: $0 /ABSOLUTE_EMPTY_OUTPUT_DIRECTORY EXPECTED_HEAD EXPECTED_TREE"
out=$1
expected_head=$2
expected_tree=$3
[[ -z "$(find "$out" -mindepth 1 -maxdepth 1 -print -quit)" ]] ||
  die "E3 R1 late-D2/stage-6 output must be empty"
for tool in ar cmp env find git grep nm python3 sha256sum timeout; do
  require_command "$tool"
done
[[ -z "$(git -C "$PROJECT_ROOT" status --porcelain)" ]] ||
  die "E3 late-D2/stage-6 source tree must be clean"
[[ "$(git -C "$PROJECT_ROOT" rev-parse HEAD)" == "$expected_head" &&
   "$(git -C "$PROJECT_ROOT" rev-parse HEAD^{tree})" == "$expected_tree" ]] ||
  die "E3 late-D2/stage-6 checkout differs from the host-pinned HEAD/tree"
verify_toolchain
git -C "$PROJECT_ROOT" rev-parse HEAD >"$out/source.commit"
git -C "$PROJECT_ROOT" rev-parse HEAD^{tree} >"$out/source.tree"
prefix="${PROJECT_ROOT}/tests/e3_private/fault_package"
provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
cxx="$(tsv_value "$provenance" cxx_path)"
runner="$(eshkol_build_dir)/eshkol-run"

capture() {
  local name=$1 limit=$2 status
  shift 2
  set +e
  timeout --foreground --signal=TERM --kill-after=30s "${limit}s" "$@" \
    >"$out/${name}.stdout" 2>"$out/${name}.stderr"
  status=$?
  set -e
  printf '%s\n' "$status" >"$out/${name}.exit"
  [[ "$status" == 0 ]] || die "${name} failed with status ${status}"
}

capture corpus 120 python3 -m tests.e3_reference.corpus --output "$out/corpus"
corpus="$out/corpus/packed-single"
[[ -d "$corpus" && -n "$(find "$corpus" -type f -print -quit)" ]] ||
  die "genuine packed D1/D2 corpus is missing"
python3 - "$out/corpus/corpus-reference.json" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
case = next(item for item in payload["cases"] if item["name"] == "packed-single")
assert case["packed"] is True and case["start_ordinal"] == 0
assert case["total_rows"] == len(case["batches"]) == 3
assert case["corpus"]["shard_token_counts"] == [6]
PY

for mode in normal sanitized; do
  kind=normal
  [[ "$mode" != sanitized ]] || kind=sanitize
  capture "package-$mode" 5400 /usr/bin/bash \
    "${PROJECT_ROOT}/scripts/build-e3-r1-late-d2-stage6-test.sh" \
    "$out/private-$mode" "$kind"
  evidence="$out/private-$mode/e3_fault_package.o.evidence"
  for pair in defined_symbols:global-defined public_exports:package-exports \
      public_strings:public-strings source_closure:source-closure \
      native_source_closure:native-source-closure native_objects:native-objects; do
    expected=${pair%%:*}
    observed=${pair#*:}
    cmp "${prefix}_${expected}.txt" "$evidence/${observed}.txt"
  done
  if [[ "$mode" == sanitized ]]; then
    cmp "${prefix}_sanitized_instrumentation_undefined_symbols.txt" \
      "$evidence/sanitizer-undefined.txt"
    cmp "${prefix}_sanitized_noninstrumentation_undefined_symbols.txt" \
      "$evidence/noninstrumentation-undefined.txt"
    nm -u "$out/private-$mode/e3_fault_package.o" | \
      grep -E '[[:space:]]U __asan_report' >/dev/null ||
      die "sanitized E3 object lacks ASan report imports"
  else
    cmp "${prefix}_undefined_symbols.txt" "$evidence/undefined.txt"
  fi
  for symbol in et_e3_test_fault_arm_v1 et_e3_test_fault_ref_v1 \
      et_e3_test_fault_reset_v1 et_e3_test_fault_seed_publication_v1 \
      et_d2_exact_read_test_late_arm_v1 \
      et_d2_exact_read_test_late_ref_v1 \
      et_d2_exact_read_test_late_reset_v1; do
    grep -E "[[:space:]]LOCAL[[:space:]].*[[:space:]]${symbol}$" \
      "$evidence/readelf-symbols.txt" >/dev/null ||
      die "fault selector ${symbol} escaped or is missing"
  done
  mkdir "$out/link-$mode" "$out/cache-$mode"
  ar p "$out/private-$mode/libeshkol_transformer_e3_r1_late_d2_stage6.a" \
    e3_fault_package.o >"$out/link-$mode/package.o"
  cmp "$out/private-$mode/e3_fault_package.o" "$out/link-$mode/package.o"
  flags=(-O2)
  if [[ "$mode" == sanitized ]]; then
    flags=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
  fi
  capture "arena-$mode" 120 "$cxx" -std=c++17 -Wall -Wextra -Werror \
    -fstack-protector-all -fPIC -fvisibility=hidden -fno-common \
    "${flags[@]}" -I "$(eshkol_source_dir)/lib/core" \
    -I "$(eshkol_source_dir)/inc" \
    -c "${PROJECT_ROOT}/tests/e3_private/retention_arena.cpp" \
    -o "$out/link-$mode/retention_arena.o"
  ar rcsD "$out/link-$mode/libeshkol_transformer_e3_r1_late_d2_stage6_test.a" \
    "$out/link-$mode/package.o" "$out/link-$mode/retention_arena.o"
done

cat >"$out/cxx-sanitized" <<WRAPPER
#!/usr/bin/env bash
exec $(printf '%q' "$cxx") -fsanitize=address,undefined -fno-omit-frame-pointer "\$@"
WRAPPER
chmod 0500 "$out/cxx-sanitized"
for mode in normal sanitized; do
  selected_cxx=$cxx
  [[ "$mode" != sanitized ]] || selected_cxx="$out/cxx-sanitized"
  capture "compile-$mode" 5400 env -u CPATH -u C_INCLUDE_PATH \
    -u CPLUS_INCLUDE_PATH -u OBJC_INCLUDE_PATH -u DEPENDENCIES_OUTPUT \
    -u SUNPRO_DEPENDENCIES -u GCC_EXEC_PREFIX -u COMPILER_PATH \
    -u LIBRARY_PATH -u CLANG_CONFIG_FILE -u CCC_OVERRIDE_OPTIONS \
    -u CCC_CC -u CCC_CXX -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR \
    ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="$out/cache-$mode" \
    ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" \
    ESHKOL_CXX_COMPILER="$selected_cxx" "$runner" \
    --strict-types --no-stdlib -O 2 -L "$out/link-$mode" \
    --lib eshkol_transformer_e3_r1_late_d2_stage6_test \
    "${PROJECT_ROOT}/tests/e3_private/r1_late_d2_stage6_runtime.esk" \
    -o "$out/runtime-$mode"
  [[ -x "$out/runtime-$mode" ]] || die "AOT executable is missing"
  if nm -u "$out/runtime-$mode" | \
      grep -E '[[:space:]]U (et_|eshkol_)[A-Za-z0-9_]*(@|$)' >/dev/null; then
    die "AOT executable retains private imports"
  fi
done

for mode in normal normal-repeat poisoned sanitized; do
  binary=${mode%-repeat}
  [[ "$mode" != poisoned ]] || binary=normal
  poison=0
  [[ "$mode" != poisoned && "$mode" != sanitized ]] || poison=1
  for scenario in 6-0 6-1 6-2 7-0; do
    site=${scenario%-*}
    ordinal=${scenario#*-}
    for gradient in absent present; do
      capture "runtime-$mode-$scenario-$gradient" 1200 env \
        ESHKOL_ARENA_POISON="$poison" \
        ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
        UBSAN_OPTIONS=halt_on_error=1 \
        "$out/runtime-$binary" "$corpus" "$gradient" "$site" "$ordinal"
      [[ "$(cat "$out/runtime-$mode-$scenario-$gradient.stdout")" == \
         "E3-R1-LATE-D2-PASS site=$site ordinal=$ordinal gradient=$gradient" ]] ||
        die "late-D2/stage-6 runtime marker differs"
      [[ ! -s "$out/runtime-$mode-$scenario-$gradient.stderr" ]] ||
        die "late-D2/stage-6 runtime wrote stderr"
    done
  done
done
for scenario in 6-0 6-1 6-2 7-0; do
  for gradient in absent present; do
    cmp "$out/runtime-normal-$scenario-$gradient.stdout" \
        "$out/runtime-normal-repeat-$scenario-$gradient.stdout"
    cmp "$out/runtime-normal-$scenario-$gradient.stdout" \
        "$out/runtime-poisoned-$scenario-$gradient.stdout"
    cmp "$out/runtime-normal-$scenario-$gradient.stdout" \
        "$out/runtime-sanitized-$scenario-$gradient.stdout"
  done
done
printf 'E3-R1-LATE-D2-COMPILED-PASS\n' >"$out/RESULT"
(cd "$out" && find . -type f ! -name SHA256SUMS -print0 | \
  LC_ALL=C sort -z | xargs -0 sha256sum >SHA256SUMS)
cat "$out/RESULT"
