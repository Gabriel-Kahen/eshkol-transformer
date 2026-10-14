#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

case "${1:-}" in
  --check) exec python3 "$PROJECT_ROOT/scripts/check-g3r-replayed-prefix-linked.py" ;;
  --run) ;;
  *) die "usage: $0 --check | --run" ;;
esac
[[ "${G3R_LINKED_RUNTIME_RELEASE:-}" == 1 ]] || \
  die "linked runtime gate requires explicit G3R_LINKED_RUNTIME_RELEASE=1"
[[ "${G3R_LINKED_EXPECTED_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]] || \
  die "linked runtime gate requires exact G3R_LINKED_EXPECTED_COMMIT"
[[ "$(git -C "$PROJECT_ROOT" rev-parse HEAD)" == \
   "$G3R_LINKED_EXPECTED_COMMIT" ]] || die "checkout SHA changed"
[[ -z "$(git -C "$PROJECT_ROOT" status --porcelain)" ]] || \
  die "linked runtime gate requires a clean checkout"
python3 "$PROJECT_ROOT/scripts/check-g3r-replayed-prefix-linked.py"
verify_toolchain
for command in ar cmp nm python3 rg sha256sum timeout; do require_command "$command"; done
cc= cxx=
resolve_provenance_compilers cc cxx \
  "${CC:-$(lock_value supported_cc)}" "${CXX:-$(lock_value supported_cxx)}"

evidence="${G3R_LINKED_EVIDENCE:-$(project_build_dir)/g3r-replayed-prefix-linked}"
mkdir -p "$evidence"
cp "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" \
  "$evidence/eshkol-transformer-provenance.tsv"
sha256sum "$(eshkol_build_dir)/eshkol-run" "$cc" "$cxx" \
  >"$evidence/compiler-binaries.sha256"
git -C "$(eshkol_source_dir)" rev-parse HEAD >"$evidence/eshkol-source.sha"
[[ "$(cat "$evidence/eshkol-source.sha")" == \
   "fe9dfd5241a1f4c4f58dee8442f44e4ff95e55b9" ]] || \
  die "Eshkol source SHA differs from linked gate manifest"

commands="$evidence/raw-commands-and-exits.log"
: >"$commands"
run_logged() {
  printf 'command:' >>"$commands"
  printf ' %q' "$@" >>"$commands"
  printf '\n' >>"$commands"
  local status=0
  "$@" || status=$?
  printf 'exit: %d\n' "$status" >>"$commands"
  return "$status"
}
temporary="$(mktemp -d "$evidence/tmp.XXXXXX")"
trap 'status=$?; if (( status != 0 )); then echo "G3-R linked gate retained: $temporary" >&2; else rm -rf -- "$temporary"; fi' EXIT
sources=(native/data_io.c native/checkpoint_io.c native/kernel_abi.c
  src/eshkol_transformer/m3_i64_integration.c native/t1_i64_shell.c
  src/eshkol_transformer/m3_call_f32_integration.c
  src/eshkol_transformer/g3t_transport.c native/a2_kv_cache.c
  native/n2_primitives_provider.c native/n3k_primitives_provider.c
  native/a2_attention_provider.c native/g3n_primitives_provider.c
  native/g3s_sampling_provider.c)
build_mode() {
  local mode="$1" directory="$temporary/$1" source stem
  local -a flags=(-O2) runtime=(env)
  if [[ "$mode" == sanitize ]]; then
    flags=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
    runtime=(env ASAN_OPTIONS=detect_leaks=1:halt_on_error=1
      UBSAN_OPTIONS=halt_on_error=1)
  fi
  mkdir -p "$directory/objects" "$directory/cache"
  local -a common=(-std=c11 -Wall -Wextra -Werror -Wpedantic -fPIC
    -Wno-misleading-indentation -Wno-unused-parameter
    -fvisibility=hidden -fno-common -fstack-protector-all -ffp-contract=off
    -fexcess-precision=standard -fno-fast-math -I "$PROJECT_ROOT/include"
    -I "$PROJECT_ROOT/native" -I "$PROJECT_ROOT/src"
    -I "$PROJECT_ROOT/src/eshkol_transformer" -I "$(eshkol_source_dir)/inc")
  run_logged "$cc" "${common[@]}" "${flags[@]}" -DET_I2_NATIVE_HELPERS_ONLY \
    -DET_M3T_PACKAGE_BUILD -c "$PROJECT_ROOT/native/i2_wave2_package_bridge.c" \
    -o "$directory/objects/i2_helpers.o"
  run_logged "$cc" "${common[@]}" "${flags[@]}" -DET_P1_TRUSTED_BUILD=1 \
    -c "$PROJECT_ROOT/native/p1_identity.c" -o "$directory/objects/p1_identity.o"
  for source in "${sources[@]}"; do
    stem="$(basename "$source" .c)"
    local -a extra=()
    [[ "$stem" == g3t_transport ]] && extra=(-DET_G3T_TESTING
      -DET_F32_TENSOR_TESTING -DET_A2_KV_CACHE_TESTING -DET_I64_TENSOR_TESTING
      -DET_G3T_PREFILL_SAMPLE_PRIVATE -DET_G3T_OUTPUT_TEXT_PRIVATE
      -DET_G3T_FINAL_PUBLICATION_PRIVATE -DET_G3T_ZERO_BUDGET_PRIVATE
      -DET_G3T_P2_ZERO_BUDGET_PRIVATE -DET_G3T_FULL_REQUEST_PREFLIGHT_PRIVATE
      -DET_G3T_MANUAL_LOGITS_PRIVATE -DET_G3T_MANUAL_P1_PREFILL_PRIVATE
      -DET_G3T_MANUAL_P2_PREFILL_PRIVATE -DET_G3T_MANUAL_DECODE_PRIVATE
      -DET_G3T_MANUAL_LOGITS_MATERIALIZE_PRIVATE
      -DET_G3T_OUTPUT_IDS_CLONE_PRIVATE -DET_G3T_OUTPUT_LENGTHS_CLONE_PRIVATE
      -DET_G3T_OUTPUT_CACHE_LENGTHS_CLONE_PRIVATE
      -DET_G3T_OUTPUT_RNG_CLONE_PRIVATE -DET_G3T_GENERATOR_RNG_PRIVATE
      -DET_G3T_RECORD_RNG_IMPORT_PRIVATE -DET_G3R_COMMITTED_HISTORY_PRIVATE
      -DET_G3R_REPLAYED_PREFIX_PRIVATE -DET_G3T_INPUT_FROM_T1_PRIVATE
      -DET_G3T_OWNED_TOKEN_INPUT_PRIVATE -DET_I64_TENSOR_STORAGE_QUERY_PRIVATE
      -DET_A2_KV_CACHE_STORAGE_QUERY_PRIVATE)
    [[ "$stem" == m3_i64_integration ]] && extra=(-DET_M3_TESTING
      -DET_I64_TENSOR_TESTING -DET_I64_TENSOR_STORAGE_QUERY_PRIVATE)
    [[ "$stem" == m3_call_f32_integration ]] && extra=(-DET_F32_TENSOR_TESTING)
    [[ "$stem" == a2_kv_cache ]] && extra=(-DET_A2_KV_CACHE_TESTING
      -DET_A2_KV_CACHE_STORAGE_QUERY_PRIVATE)
    run_logged "$cc" "${common[@]}" "${flags[@]}" "${extra[@]}" \
      -MMD -MF "$directory/objects/$stem.d" \
      -c "$PROJECT_ROOT/$source" -o "$directory/objects/$stem.o"
  done
  run_logged "$cc" "${common[@]}" "${flags[@]}" \
    -c "$PROJECT_ROOT/src/eshkol_transformer/g3t_transport.c" \
    -o "$directory/g3t_feature_off.o"
  run_logged nm -g --defined-only "$directory/g3t_feature_off.o" \
    >"$directory/feature-off.symbols"
  if rg -q 'et_g3t_private_replayed_prefix_call_acquire_v1' \
      "$directory/feature-off.symbols"; then
    die "feature-off replayed-prefix symbol escaped"
  fi
  run_logged nm -g --defined-only "$directory/objects/g3t_transport.o" \
    >"$directory/feature-on.symbols"
  rg -q 'et_g3t_private_replayed_prefix_call_acquire_v1' \
    "$directory/feature-on.symbols" || \
    die "feature-on replayed-prefix symbol missing"
  run_logged "$cc" "${common[@]}" "${flags[@]}" -DET_G3T_TESTING \
    -DET_G3T_PREFILL_SAMPLE_PRIVATE \
    -DET_G3T_MANUAL_P1_PREFILL_PRIVATE -DET_G3R_COMMITTED_HISTORY_PRIVATE \
    -c "$PROJECT_ROOT/tests/g3r/replayed_prefix_linked_probe.c" \
    -o "$directory/objects/replayed_prefix_linked_probe.o"
  run_logged ar rcsD "$directory/libg3r_replayed_prefix_linked.a" \
    "$directory"/objects/*.o
  local sanitizer_link=
  [[ "$mode" == sanitize ]] && sanitizer_link='-fsanitize=address,undefined -fno-omit-frame-pointer'
  cat >"$directory/cxx-wrap" <<WRAPPER
#!/usr/bin/env bash
exec $(printf '%q' "$cxx") -Wl,--wrap=et_kernel_runtime_dispatch $sanitizer_link "\$@"
WRAPPER
  chmod 0500 "$directory/cxx-wrap"
  run_logged env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
    XDG_CACHE_HOME="$directory/cache" ESHKOL_CXX_COMPILER="$directory/cxx-wrap" \
    ESHKOL_LIB_DIR="$PROJECT_ROOT/lib" \
    timeout --foreground --signal=TERM --kill-after=5s 600s \
    "$(eshkol_build_dir)/eshkol-run" --strict-types --optimize 0 --no-stdlib \
    -I "$PROJECT_ROOT/internal/p1/lib" -I "$PROJECT_ROOT/internal/c1/lib" \
    -I "$PROJECT_ROOT/internal/t1/lib" -I "$PROJECT_ROOT/src" \
    -I "$PROJECT_ROOT/lib" -I "$PROJECT_ROOT/native" \
    -L "$directory" --lib g3r_replayed_prefix_linked \
    "$PROJECT_ROOT/tests/g3r/replayed_prefix_linked_test.esk" \
    -o "$directory/caller"
  run_logged "${runtime[@]}" ESHKOL_ARENA_POISON=1 \
    timeout --foreground --signal=TERM --kill-after=5s 120s \
    "$directory/caller" >"$directory/stdout" 2>"$directory/stderr"
  [[ ! -s "$directory/stderr" ]] || die "$mode runtime stderr is nonempty"
  grep -E '^G3-R linked replayed P1 PASS: checks=[1-9][0-9]*$' \
    "$directory/stdout" >/dev/null || die "$mode PASS line missing"
}
build_mode normal
run_logged env ESHKOL_ARENA_POISON=1 "$temporary/normal/caller" \
  >"$temporary/normal/repeat.stdout" 2>"$temporary/normal/repeat.stderr"
[[ ! -s "$temporary/normal/repeat.stderr" ]] || die "repeat stderr is nonempty"
build_mode sanitize
run_logged cmp "$temporary/normal/stdout" "$temporary/normal/repeat.stdout"
run_logged cmp "$temporary/normal/stdout" "$temporary/sanitize/stdout"
for mode in normal sanitize; do
  cp "$temporary/$mode/stdout" "$evidence/$mode.stdout"
  cp "$temporary/$mode/stderr" "$evidence/$mode.stderr"
done
echo "G3-R linked private replayed-prefix gate PASS: $evidence"
