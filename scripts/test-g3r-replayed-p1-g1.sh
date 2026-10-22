#!/usr/bin/env bash
# Genuine same-process private P1 replay; native-provider sanitizer coverage only.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
for command in ar cmp git grep nm python3 realpath rg sha256sum timeout; do
  require_command "$command"
done
[[ $# == 1 ]] || die "usage: $0 FRESH_OUTPUT_DIRECTORY"
[[ -z "$(git -C "$PROJECT_ROOT" status --porcelain --untracked-files=all)" ]] || \
  die "G3-R replay admission requires a clean source tree"
out="$(realpath -m -- "$1")"
[[ ! -e "$out" && ! -L "$out" ]] || die "G3-R replay output must be fresh"
mkdir -p "$out"
cc= cxx=
resolve_provenance_compilers cc cxx \
  "${CC:-$(lock_value supported_cc)}" "${CXX:-$(lock_value supported_cxx)}"
cp "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" "$out/compiler-provenance.tsv"
git -C "$PROJECT_ROOT" rev-parse HEAD HEAD^{tree} >"$out/source.identity"
python3 "$PROJECT_ROOT/scripts/check-g3r-replayed-p1-g1.py" >"$out/source-check.stdout"
"$cc" -std=c11 -O1 -Wall -Wextra -Werror -Wpedantic \
  -I "$PROJECT_ROOT/include" -I "$PROJECT_ROOT/native" -I "$PROJECT_ROOT/src" \
  -I "$PROJECT_ROOT/src/eshkol_transformer" \
  -c "$PROJECT_ROOT/src/eshkol_transformer/g3t_transport.c" -o "$out/feature-off.o"
nm -g --defined-only "$out/feature-off.o" >"$out/feature-off-symbols.txt"
if grep -Eq 'et_g3t_private_replayed_p1_g1_(preflight|acquire)_v1' \
    "$out/feature-off-symbols.txt"; then
  die "private replay ABI leaked in feature-off object"
fi
sources=(native/data_io.c native/checkpoint_io.c native/kernel_abi.c
  src/eshkol_transformer/m3_i64_integration.c native/t1_i64_shell.c
  src/eshkol_transformer/m3_call_f32_integration.c
  src/eshkol_transformer/g3t_transport.c native/a2_kv_cache.c
  native/n2_primitives_provider.c native/n3k_primitives_provider.c
  native/a2_attention_provider.c native/g3n_primitives_provider.c
  native/g3s_sampling_provider.c)
build_mode() {
  local mode="$1" directory="$out/$1" source stem
  local -a flags=(-O2) runtime=(env) extra=()
  if [[ "$mode" == sanitize ]]; then
    flags=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
    runtime=(env ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1
      LSAN_OPTIONS=exitcode=23:report_objects=1)
  fi
  mkdir -p "$directory/objects" "$directory/cache"
  local -a common=(-std=c11 -Wall -Wextra -Werror -Wpedantic -fPIC
    -fvisibility=hidden -fno-common -fstack-protector-all -ffp-contract=off
    -fexcess-precision=standard -fno-fast-math -I "$PROJECT_ROOT/include"
    -I "$PROJECT_ROOT/native" -I "$PROJECT_ROOT/src"
    -I "$PROJECT_ROOT/src/eshkol_transformer" -I "$(eshkol_source_dir)/inc")
  "$cc" "${common[@]}" "${flags[@]}" -DET_I2_NATIVE_HELPERS_ONLY \
    -DET_M3T_PACKAGE_BUILD -c "$PROJECT_ROOT/native/i2_wave2_package_bridge.c" \
    -o "$directory/objects/i2_helpers.o"
  "$cc" "${common[@]}" "${flags[@]}" -DET_P1_TRUSTED_BUILD=1 \
    -c "$PROJECT_ROOT/native/p1_identity.c" -o "$directory/objects/p1_identity.o"
  for source in "${sources[@]}"; do
    stem="$(basename "$source" .c)"
    extra=()
    [[ "$stem" == g3t_transport ]] && extra=(-DET_G3T_TESTING
      -DET_G3R_COMMITTED_HISTORY_PRIVATE -DET_G3R_REPLAYED_P1_G1_PRIVATE
      -DET_F32_TENSOR_TESTING
      -DET_A2_KV_CACHE_TESTING -DET_I64_TENSOR_TESTING
      -DET_G3T_PREFILL_SAMPLE_PRIVATE -DET_G3T_OUTPUT_TEXT_PRIVATE
      -DET_G3T_FINAL_PUBLICATION_PRIVATE -DET_G3T_ZERO_BUDGET_PRIVATE
      -DET_G3T_P2_ZERO_BUDGET_PRIVATE
      -DET_G3T_FULL_REQUEST_PREFLIGHT_PRIVATE
      -DET_G3T_MANUAL_LOGITS_PRIVATE
      -DET_G3T_MANUAL_LOGITS_MATERIALIZE_PRIVATE
      -DET_G3T_MANUAL_P1_PREFILL_PRIVATE
      -DET_G3T_MANUAL_P2_PREFILL_PRIVATE
      -DET_G3T_MANUAL_DECODE_PRIVATE
      -DET_G3T_OUTPUT_IDS_CLONE_PRIVATE
      -DET_G3T_OUTPUT_LENGTHS_CLONE_PRIVATE
      -DET_G3T_OUTPUT_CACHE_LENGTHS_CLONE_PRIVATE
      -DET_G3T_OUTPUT_RNG_CLONE_PRIVATE
      -DET_G3T_GENERATOR_RNG_PRIVATE
      -DET_G3T_INPUT_FROM_T1_PRIVATE
      -DET_G3T_OWNED_TOKEN_INPUT_PRIVATE
      -DET_I64_TENSOR_STORAGE_QUERY_PRIVATE -DET_A2_KV_CACHE_STORAGE_QUERY_PRIVATE)
    [[ "$stem" == m3_i64_integration ]] && extra=(-DET_M3_TESTING -DET_I64_TENSOR_TESTING -DET_I64_TENSOR_STORAGE_QUERY_PRIVATE)
    [[ "$stem" == m3_call_f32_integration ]] && extra=(-DET_F32_TENSOR_TESTING)
    [[ "$stem" == a2_kv_cache ]] && extra=(-DET_A2_KV_CACHE_TESTING -DET_A2_KV_CACHE_STORAGE_QUERY_PRIVATE)
    "$cc" "${common[@]}" "${flags[@]}" "${extra[@]}" \
      -MMD -MF "$directory/objects/$stem.d" \
      -c "$PROJECT_ROOT/$source" -o "$directory/objects/$stem.o"
  done
  ar rcsD "$directory/libg3r_replayed_p1_g1.a" "$directory"/objects/*.o
  nm -g --defined-only "$directory/objects/g3t_transport.o" \
    >"$directory/feature-on-symbols.txt"
  for symbol in et_g3t_private_replayed_p1_g1_preflight_v1 \
                et_g3t_private_replayed_p1_g1_acquire_v1; do
    grep -Eq "[[:space:]]$symbol$" "$directory/feature-on-symbols.txt" || \
      die "private replay symbol missing: $symbol"
  done
  nm -u --format=posix "$directory/libg3r_replayed_p1_g1.a" \
    >"$directory/native-undefined-symbols.txt"
  local sanitizer_link=
  [[ "$mode" == sanitize ]] && sanitizer_link='-fsanitize=address,undefined -fno-omit-frame-pointer'
  cat >"$directory/cxx-wrap" <<WRAPPER
#!/usr/bin/env bash
exec $(printf '%q' "$cxx") $sanitizer_link "\$@"
WRAPPER
  chmod 0500 "$directory/cxx-wrap"
  local -a inputs=(
    -I "$PROJECT_ROOT/internal/p1/lib" -I "$PROJECT_ROOT/internal/c1/lib"
    -I "$PROJECT_ROOT/internal/t1/lib" -I "$PROJECT_ROOT/src"
    -I "$PROJECT_ROOT/lib" -I "$PROJECT_ROOT/native"
    -L "$directory" --lib g3r_replayed_p1_g1
    "$PROJECT_ROOT/tests/g3r/replayed_p1_g1_same_process_test.esk")
  env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
    XDG_CACHE_HOME="$directory/cache" ESHKOL_CXX_COMPILER="$directory/cxx-wrap" \
    ESHKOL_LIB_DIR="$PROJECT_ROOT/lib" \
    timeout --foreground --signal=TERM --kill-after=5s 1200s \
    "$(eshkol_build_dir)/eshkol-run" --strict-types --optimize 0 --no-stdlib --dump-ir \
    "${inputs[@]}" -o "$directory/caller" \
    >"$directory/compile.stdout" 2>"$directory/compile.stderr"
  test ! -s "$directory/compile.stderr"
  test -s "$directory/caller.ll"
  nm -u --format=posix "$directory/caller" >"$directory/linked-imports.txt"
  rg 'et_g3t_private_replayed_p1_g1_(preflight|acquire)_v1' \
    "$directory/caller.ll" >"$directory/replay-ir-calls.txt"
  if [[ "$mode" == normal ]]; then
    env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
      XDG_CACHE_HOME="$directory/cache" ESHKOL_CXX_COMPILER="$directory/cxx-wrap" \
      ESHKOL_LIB_DIR="$PROJECT_ROOT/lib" \
      timeout --foreground --signal=TERM --kill-after=5s 1200s \
      "$(eshkol_build_dir)/eshkol-run" --strict-types --optimize 0 --no-stdlib \
      --emit-object --emit-depfile "$directory/caller.d" "${inputs[@]}" \
      -o "$directory/caller.o" >"$directory/object.stdout" 2>"$directory/object.stderr"
    test ! -s "$directory/object.stderr"
    nm -u --format=posix "$directory/caller.o" >"$directory/eshkol-imports.txt"
    for symbol in et_g3t_private_replayed_p1_g1_preflight_v1 \
                  et_g3t_private_replayed_p1_g1_acquire_v1; do
      rg "^$symbol U" "$directory/eshkol-imports.txt" >/dev/null
    done
    sed -e 's/^[^:]*://' -e 's/\\//g' "$directory/caller.d" | \
      tr -s '[:space:]' '\n' | rg "^$PROJECT_ROOT/" | \
      sed "s#^$PROJECT_ROOT/##" >"$directory/source-closure.txt"
    while IFS= read -r source; do
      [[ -f "$PROJECT_ROOT/$source" ]] || die "missing replay source: $source"
    done <"$directory/source-closure.txt"
    rg -x 'native/g3r_replayed_p1_g1_extension.esk' "$directory/source-closure.txt" >/dev/null
    rg -x 'tests/g3r/replayed_p1_g1_same_process_test.esk' "$directory/source-closure.txt" >/dev/null
  fi
  "${runtime[@]}" ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM \
    --kill-after=5s 120s "$directory/caller" \
    >"$directory/stdout" 2>"$directory/stderr"
  test ! -s "$directory/stderr"
  grep -E '^G3-R replayed P1/G1 private PASS: checks=[1-9][0-9]*$' \
    "$directory/stdout" >/dev/null
}
build_mode normal
ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM --kill-after=5s 120s \
  "$out/normal/caller" >"$out/normal/repeat.stdout" 2>"$out/normal/repeat.stderr"
test ! -s "$out/normal/repeat.stderr"
cmp "$out/normal/stdout" "$out/normal/repeat.stdout"
build_mode sanitize
cmp "$out/normal/stdout" "$out/sanitize/stdout"
printf 'G3-R replayed P1/G1 compiled PASS: normal/repeat/native-sanitize\n'
