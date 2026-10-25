#!/usr/bin/env bash
# Diagnostic only: re-AOT the changed runtime against sealed attempt-1 archives.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
[[ $# == 4 && "$1" == /* && "$2" == /* && "$3" =~ ^[0-9a-f]{40}$ &&
   "$4" =~ ^[0-9a-f]{40}$ ]] ||
  die "usage: $0 /EMPTY_OUT /SEALED_PRIOR EXPECTED_HEAD EXPECTED_TREE"
out=$1 prior=$2 expected_head=$3 expected_tree=$4
prior_seal=12d127e794607f120edc1d936e4063e1e64c50ea3dad0eded8ed42c633bd4b0e
[[ -d "$out" && ! -L "$out" && -z "$(find "$out" -mindepth 1 -maxdepth 1 -print -quit)" ]] ||
  die "phase output must be a fresh empty directory"
[[ -d "$prior" && ! -L "$prior" ]] || die "prior receipt mount is invalid"
[[ "$(git -C "$PROJECT_ROOT" rev-parse HEAD)" == "$expected_head" &&
   "$(git -C "$PROJECT_ROOT" rev-parse HEAD^{tree})" == "$expected_tree" &&
   -z "$(git -C "$PROJECT_ROOT" status --porcelain)" ]] ||
  die "phase source differs from pinned clean HEAD/tree"
verify_toolchain
for tool in cmp cut find grep nm python3 sha256sum timeout xargs; do
  require_command "$tool"
done
[[ -x /usr/bin/time ]] || die "GNU time is required"
[[ "$(sha256sum "$prior/POSTEXIT-SHA256SUMS" | cut -d' ' -f1)" == "$prior_seal" ]] ||
  die "prior seal identity differs"
(cd "$prior" && sha256sum -c POSTEXIT-SHA256SUMS) >"$out/prior-verify.log" ||
  die "prior receipt bytes differ from the authenticated manifest"
python3 - "$prior/source.json" <<'PY' || die "prior source identity differs"
import json, sys
source = json.load(open(sys.argv[1]))
assert source['head'] == 'dd2c7ee68c99099e5b633a27737f14639d8290ad'
assert source['tree'] == 'fb4e9ae1e5d69b6ec4833dc7d821a2c851f12a35'
PY
for path in "$prior/gate/corpus/packed-single/manifest.etm" \
    "$prior/gate/corpus/packed-single/shard-0000000000000000.ets" \
    "$prior/gate/link-a/libeshkol_transformer_e3_retention_test.a" \
    "$prior/gate/link-b/libeshkol_transformer_e3_retention_test.a" \
    "$prior/gate/link-sanitized/libeshkol_transformer_e3_retention_test.a"; do
  [[ -f "$path" && ! -L "$path" ]] || die "required sealed prior input is missing: $path"
done
cmp "$prior/gate/link-a/libeshkol_transformer_e3_retention_test.a" \
    "$prior/gate/link-b/libeshkol_transformer_e3_retention_test.a"
provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
cxx="$(tsv_value "$provenance" cxx_path)"
runner="$(eshkol_build_dir)/eshkol-run"
compiler_timeout=900
runtime_timeout=1200
for profile in a b sanitized; do
  mkdir "$out/cache-$profile"
  link="$prior/gate/link-$profile"
  selected_cxx=$cxx
  [[ "$profile" != sanitized ]] || selected_cxx="$prior/gate/cxx-sanitized"
  status=0
  env -u CPATH -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH -u OBJC_INCLUDE_PATH \
    -u DEPENDENCIES_OUTPUT -u SUNPRO_DEPENDENCIES -u GCC_EXEC_PREFIX \
    -u COMPILER_PATH -u LIBRARY_PATH -u CLANG_CONFIG_FILE \
    -u CCC_OVERRIDE_OPTIONS -u CCC_CC -u CCC_CXX -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR \
    ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="$out/cache-$profile" \
    ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" ESHKOL_CXX_COMPILER="$selected_cxx" \
    timeout --foreground --signal=TERM --kill-after=5s "${compiler_timeout}s" \
    "$runner" --strict-types --no-stdlib -O 2 -L "$link" \
      --lib eshkol_transformer_e3_retention_test \
      "${PROJECT_ROOT}/tests/e3_private/retention_arena_diagnostic_runtime.esk" \
      -o "$out/runtime-$profile" \
      >"$out/compile-$profile.stdout" 2>"$out/compile-$profile.stderr" || status=$?
  printf '%s\n' "$status" >"$out/compile-$profile.exit"
  [[ "$status" == 0 && -x "$out/runtime-$profile" ]] || die "AOT $profile failed"
  if nm -u "$out/runtime-$profile" | grep -E '[[:space:]]U et_e3_retention_region_used_v1(@|$)' >/dev/null; then
    die "retention observer remained unresolved in $profile AOT"
  fi
done
cmp "$out/runtime-a" "$out/runtime-b"
printf 'profile\tmode\texit\tsample\tpostcheck\n' >"$out/STATUS.tsv"
overall=0
for profile in normal-a normal-b poisoned-a sanitized; do
  binary=a poison=0
  [[ "$profile" != normal-b ]] || binary=b
  [[ "$profile" != sanitized ]] || binary=sanitized
  [[ "$profile" != poisoned-a && "$profile" != sanitized ]] || poison=1
  pair_status=0
  /usr/bin/bash "${PROJECT_ROOT}/scripts/e3-retention-arena-phase-pair.sh" \
    "$out" "$out/runtime-$binary" "$prior/gate/corpus/packed-single" \
    "$profile" "$poison" "$runtime_timeout" /usr/bin/time >>"$out/STATUS.tsv" || pair_status=$?
  [[ "$pair_status" == 0 ]] || overall=1
done
printf '%s\n' "$overall" >"$out/OVERALL.exit"
(cd "$out" && find . -type f ! -name SHA256SUMS -print0 | LC_ALL=C sort -z | xargs -0 sha256sum >SHA256SUMS)
exit "$overall"
