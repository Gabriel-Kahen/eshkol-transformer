#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root}"

output_dir="${1:-build/mod5_gen}"
mode="${2:-normal}"

export CC="${CC:-clang}"
export CXX="${CXX:-clang++}"
ar_bin="${AR:-ar}"
nm_bin="${NM:-nm}"

cflags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic
  -Wconversion -Wsign-conversion -Wshadow
  -ffp-contract=off -fexcess-precision=standard -frounding-math -fno-fast-math
  -fPIC -fvisibility=hidden -fno-common -fstack-protector-all
  -I "${root}/include"
)

case "${mode}" in
  normal)
    cflags+=(-O2)
    ;;
  sanitize)
    cflags+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer
      "-ffile-prefix-map=${root}=.")
    ;;
  *)
    printf 'unknown MOD5 generate build mode: %s\n' "${mode}" >&2
    exit 2
    ;;
esac

mkdir -p -- "${output_dir}"
scratch="$(mktemp -d "${output_dir}/.build.XXXXXX")"
trap 'rm -rf -- "${scratch}"' EXIT

# C++17 header syntax and standard-layout check.
"${CXX}" -std=c++17 -Wall -Wextra -Werror -Wpedantic -I "${root}/include" \
  -x c++ -fsyntax-only - <<'CPP'
#include "eshkol_transformer/mod5_generate_abi.h"
#include <type_traits>
static_assert(std::is_standard_layout_v<et_mod5_gen_config>);
static_assert(std::is_standard_layout_v<et_mod5_gen_layer_weights>);
static_assert(std::is_standard_layout_v<et_mod5_gen_model_weights>);
static_assert(std::is_standard_layout_v<et_mod5_gen_rng>);
static_assert(std::is_standard_layout_v<et_mod5_gen_sampling_config>);
static_assert(std::is_standard_layout_v<et_mod5_gen_result>);
static_assert(std::is_same_v<et_mod5_gen_status, int32_t>);
CPP

# Deterministic archive check: compile twice into fresh dirs and compare bytes.
for trial in first second; do
  tdir="${scratch}/${trial}"
  mkdir -- "${tdir}"
  "${CC}" "${cflags[@]}" -c "${root}/native/mod5_generate.c" \
    -o "${tdir}/mod5_generate.o"
  "${ar_bin}" rcsD "${tdir}/libeshkol_transformer_mod5_generate.a" \
    "${tdir}/mod5_generate.o"
done

archive="${scratch}/first/libeshkol_transformer_mod5_generate.a"
cmp "${archive}" "${scratch}/second/libeshkol_transformer_mod5_generate.a"
digest="$(sha256sum "${archive}" | awk '{print $1}')"
printf 'MOD5 generate %s deterministic archive SHA-256: %s\n' "${mode}" "${digest}"

# Exact exported symbol audit.
cat > "${scratch}/expected-exports" <<'SYMBOLS'
et_mod5_gen_aot_decode_step
et_mod5_gen_aot_engine_create
et_mod5_gen_aot_engine_destroy
et_mod5_gen_aot_generate
et_mod5_gen_aot_kv_cache_create
et_mod5_gen_aot_kv_cache_destroy
et_mod5_gen_aot_kv_cache_len
et_mod5_gen_aot_kv_cache_reset
et_mod5_gen_aot_load_f64
et_mod5_gen_aot_max_abs_diff_f32
et_mod5_gen_aot_prefill
et_mod5_gen_aot_recompute_forward
et_mod5_gen_aot_set_global_weights
et_mod5_gen_aot_set_layer_weights
et_mod5_gen_aot_verify_provider
et_mod5_gen_decode_step
et_mod5_gen_engine_create
et_mod5_gen_engine_destroy
et_mod5_gen_engine_load_weights
et_mod5_gen_engine_set_global_weights
et_mod5_gen_engine_set_layer_weights
et_mod5_gen_generate
et_mod5_gen_kv_cache_capacity
et_mod5_gen_kv_cache_create
et_mod5_gen_kv_cache_destroy
et_mod5_gen_kv_cache_len
et_mod5_gen_kv_cache_read_slot
et_mod5_gen_kv_cache_reset
et_mod5_gen_prefill
et_mod5_gen_recompute_forward
et_mod5_gen_rng_init
et_mod5_gen_rng_next_f64
et_mod5_gen_rng_next_u64
et_mod5_gen_sample
et_mod5_gen_validate_config
et_mod5_gen_verify_provider
SYMBOLS

"${nm_bin}" -g --defined-only -P "${archive}" |
  awk '$2 ~ /^[A-Za-z]$/ && ($1 !~ /^__/ || mode == "normal") {print $1}' mode="${mode}" |
  LC_ALL=C sort > "${scratch}/actual-exports"
diff -u "${scratch}/expected-exports" "${scratch}/actual-exports"

# Undefined symbol audit in normal mode: verify linkage to MOD5 K1 provider
# and reject unexpected threading, random, or foreign runtime dependencies.
if [[ "${mode}" == "normal" ]]; then
  imports="$("${nm_bin}" -u "${scratch}/first/mod5_generate.o" | awk 'NF {print $NF}')"
  saw_provider=0
  while IFS= read -r symbol; do
    case "${symbol}" in
      '') ;;
      et_mod5_kernel_provider_v1) saw_provider=1 ;;
      __stack_chk_fail|_GLOBAL_OFFSET_TABLE_|calloc|exp|expf|fegetround|free|\
      memcpy|memset|sqrtf|strcmp|\
      et_kernel_runtime_destroy|et_kernel_runtime_discover|et_kernel_runtime_dispatch) ;;
      *)
        printf 'unexpected MOD5 generate dependency: %s\n' "${symbol}" >&2
        exit 1
        ;;
    esac
  done <<< "${imports}"
  if [[ "${saw_provider}" -ne 1 ]]; then
    printf 'MOD5 generate archive did not link et_mod5_kernel_provider_v1\n' >&2
    exit 1
  fi
fi

mv -f -- "${archive}" "${output_dir}/libeshkol_transformer_mod5_generate.a"
printf 'MOD5 generate %s build, C++17 header, and symbol audit: PASS (%s)\n' \
  "${mode}" "${output_dir}/libeshkol_transformer_mod5_generate.a"
