#!/usr/bin/env bash
# Compiled union recipe. Runtime requires an explicit release after the E3 load.
set -euo pipefail
mode="${1:-}"
if [[ $# != 1 || ( "$mode" != --check && "$mode" != --run ) ]]; then
  echo 'usage: check-g3r-current-private-union.sh --check|--run' >&2
  exit 74
fi
if [[ "$mode" == --run && "${G3R_PRIVATE_UNION_RUNTIME_RELEASE:-0}" != 1 ]]; then
  echo 'HOLD: Eshkol runtime release is required for --run' >&2
  exit 74
fi
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
base=86f4eb36f2fcf079953049d3cdb11377cea5981c
reviewed=57a8e80ac6f862696d7c1d0a451c8ce9fc95f320
[[ "$(git -C "$root" merge-base "$base" HEAD)" == "$base" ]] || {
  echo 'current-main base differs' >&2; exit 74;
}
[[ "$(git -C "$root" merge-base "$reviewed" HEAD)" == "$reviewed" ]] || {
  echo 'reviewed G3-R source head is not an ancestor' >&2; exit 74;
}
if ! git -C "$root" diff --quiet "$reviewed" -- \
  include internal lib native src toolchain scripts/common.sh; then
  echo 'reviewed G3-R provider/toolchain source changed' >&2; exit 74
fi
compiler=clang-22
if [[ "$mode" == --run ]]; then
  source "$root/scripts/common.sh"
  verify_toolchain
  cc= cxx=
  resolve_provenance_compilers cc cxx \
    "${CC:-$(lock_value supported_cc)}" "${CXX:-$(lock_value supported_cxx)}"
  compiler="$cc"
fi
command -v "$compiler" >/dev/null
command -v ar >/dev/null
command -v nm >/dev/null
out="$(mktemp -d "${TMPDIR:-/tmp}/g3r-private-union.XXXXXX")"
trap 'rm -rf -- "$out"' EXIT
common=(-std=c11 -Wall -Wextra -Werror -Wpedantic -fPIC
  -fvisibility=hidden -fno-common -fstack-protector-all -ffp-contract=off
  -fexcess-precision=standard -fno-fast-math -O1
  -I "$root/include" -I "$root/native" -I "$root/src"
  -I "$root/src/eshkol_transformer")
sources=(native/data_io.c native/checkpoint_io.c native/kernel_abi.c
  src/eshkol_transformer/m3_i64_integration.c native/t1_i64_shell.c
  src/eshkol_transformer/m3_call_f32_integration.c
  src/eshkol_transformer/g3t_transport.c native/a2_kv_cache.c
  native/n2_primitives_provider.c native/n3k_primitives_provider.c
  native/a2_attention_provider.c native/g3n_primitives_provider.c
  native/g3s_sampling_provider.c
  tests/g3r/current_history_private_union_probe.c)
for source in "${sources[@]}"; do
  stem="$(basename "$source" .c)"
  extra=()
  case "$stem" in
    m3_i64_integration)
      extra=(-DET_M3_TESTING -DET_I64_TENSOR_TESTING
        -DET_I64_TENSOR_STORAGE_QUERY_PRIVATE) ;;
    m3_call_f32_integration)
      extra=(-DET_F32_TENSOR_TESTING -DET_F32_TENSOR_STORAGE_QUERY_PRIVATE
        -DET_G3R_SAME_OWNER_SAVE_PRIVATE) ;;
    a2_kv_cache)
      extra=(-DET_A2_KV_CACHE_TESTING
        -DET_A2_KV_CACHE_STORAGE_QUERY_PRIVATE) ;;
    g3t_transport)
      extra=(-DET_G3T_TESTING -DET_F32_TENSOR_TESTING
        -DET_A2_KV_CACHE_TESTING -DET_I64_TENSOR_TESTING
        -DET_G3T_PREFILL_SAMPLE_PRIVATE -DET_G3T_OUTPUT_TEXT_PRIVATE
        -DET_G3T_FINAL_PUBLICATION_PRIVATE -DET_G3T_ZERO_BUDGET_PRIVATE
        -DET_G3T_P2_ZERO_BUDGET_PRIVATE -DET_G3T_FULL_REQUEST_PREFLIGHT_PRIVATE
        -DET_G3T_MANUAL_LOGITS_PRIVATE -DET_G3T_MANUAL_P1_PREFILL_PRIVATE
        -DET_G3T_MANUAL_P2_PREFILL_PRIVATE -DET_G3T_MANUAL_DECODE_PRIVATE
        -DET_G3T_MANUAL_LOGITS_MATERIALIZE_PRIVATE
        -DET_G3T_INPUT_FROM_T1_PRIVATE -DET_G3T_OWNED_TOKEN_INPUT_PRIVATE
        -DET_G3T_OUTPUT_RNG_CLONE_PRIVATE -DET_G3T_GENERATOR_RNG_PRIVATE
        -DET_G3R_COMMITTED_HISTORY_PRIVATE
        -DET_G3R_COMMITTED_SAVE_SNAPSHOT_PRIVATE
        -DET_G3R_SAME_OWNER_SAVE_PRIVATE
        -DET_I64_TENSOR_STORAGE_QUERY_PRIVATE
        -DET_A2_KV_CACHE_STORAGE_QUERY_PRIVATE
        -DET_F32_TENSOR_STORAGE_QUERY_PRIVATE) ;;
  esac
  "$compiler" "${common[@]}" "${extra[@]}" -c "$root/$source" \
    -o "$out/$stem.o"
done
"$compiler" "${common[@]}" -DET_I2_NATIVE_HELPERS_ONLY \
  -DET_M3T_PACKAGE_BUILD -c "$root/native/i2_wave2_package_bridge.c" \
  -o "$out/i2_helpers.o"
"$compiler" "${common[@]}" -DET_P1_TRUSTED_BUILD=1 \
  -c "$root/native/p1_identity.c" -o "$out/p1_identity.o"
ar rcsD "$out/libg3r_current_private_union.a" "$out"/*.o
defined="$(nm -g --defined-only "$out/libg3r_current_private_union.a")"
for symbol in et_m3t_private_owner_save_roster_v1 \
  et_f32_parameter_save_witness_begin_v1 \
  et_f32_parameter_save_witness_check_v1 \
  et_f32_parameter_save_witness_end_v1 \
  et_g3r_private_g3t_save_begin_v1 \
  et_g3r_private_g3t_save_check_v1 \
  et_g3r_private_g3t_save_copy_v1 \
  et_g3r_private_g3t_save_end_v1 \
  et_g3r_current_private_union_probe_v1; do
  grep -Eq "[[:space:]]${symbol}$" <<<"$defined" || {
    echo "private union symbol absent: $symbol" >&2; exit 1;
  }
done
for source in src/eshkol_transformer/m3t_transport.c \
  src/eshkol_transformer/g3t_transport.c \
  src/eshkol_transformer/m3_call_f32_integration.c; do
  stem="$(basename "$source" .c)"
  "$compiler" "${common[@]}" -c "$root/$source" -o "$out/off-$stem.o"
done
if nm -g --defined-only "$out"/off-*.o | grep -Eq \
  'et_m3t_private_owner_save_roster_v1|et_f32_parameter_save_witness_(begin|check|end)_v1|et_g3r_private_g3t_save_(begin|check|copy|end)_v1'; then
  echo 'private union symbol escaped feature-off objects' >&2; exit 1
fi
if [[ "$mode" == --check ]]; then
  echo 'G3-R current private union source archive and feature-off checks PASS; runtime HOLD'
  exit 0
fi
mkdir -p "$out/cache"
cat >"$out/cxx-wrap" <<WRAPPER
#!/usr/bin/env bash
exec $(printf '%q' "$cxx") "\$@"
WRAPPER
chmod 0500 "$out/cxx-wrap"
env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
  XDG_CACHE_HOME="$out/cache" ESHKOL_CXX_COMPILER="$out/cxx-wrap" \
  ESHKOL_LIB_DIR="$root/lib" \
  timeout --foreground --signal=TERM --kill-after=5s 600s \
  "$(eshkol_build_dir)/eshkol-run" --strict-types --optimize 0 --no-stdlib \
  -I "$root/internal/p1/lib" -I "$root/internal/c1/lib" \
  -I "$root/internal/t1/lib" -I "$root/src" -I "$root/lib" \
  -I "$root/native" -L "$out" --lib g3r_current_private_union \
  "$root/tests/g3r/current_history_private_union_test.esk" \
  -o "$out/caller" >"$out/compile.stdout" 2>"$out/compile.stderr"
ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM --kill-after=5s \
  120s "$out/caller" >"$out/stdout" 2>"$out/stderr"
test ! -s "$out/stderr"
grep -Eq '^G3-R current private union PASS: checks=[1-9][0-9]*$' \
  "$out/stdout"
cat "$out/stdout"
