#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cc="${CC:-clang}"
artifact="${root}/build/mod5_adamw"
mkdir -p "${artifact}"
scratch="$(mktemp -d "${artifact}/.build.XXXXXX")"
trap 'rm -rf -- "${scratch}"' EXIT

"${cc}" -std=c11 -Wall -Wextra -Werror -Wpedantic \
  -Wconversion -Wsign-conversion -Wshadow \
  -O2 -ffp-contract=off -fexcess-precision=standard -fno-fast-math \
  -fstack-protector-all -fPIC -fvisibility=hidden -fno-common \
  -I "${root}/include" \
  -c "${root}/native/mod5_adamw_state.c" \
  -o "${scratch}/mod5_adamw_state.o"
ar rcsD "${scratch}/libeshkol_transformer_mod5_adamw.a" \
  "${scratch}/mod5_adamw_state.o"

exports="$(nm -g --defined-only "${scratch}/libeshkol_transformer_mod5_adamw.a" |
  awk 'NF >= 3 {print $(NF-1), $NF}' | LC_ALL=C sort)"
expected="$(cat <<'EOF'
T et_mod5_adamw_aot_compute_lr
T et_mod5_adamw_aot_create
T et_mod5_adamw_aot_destroy
T et_mod5_adamw_aot_file_sha256_hex
T et_mod5_adamw_aot_get_cursor_epoch
T et_mod5_adamw_aot_get_cursor_offset
T et_mod5_adamw_aot_get_cursor_shard
T et_mod5_adamw_aot_get_m_bits
T et_mod5_adamw_aot_get_param_bits
T et_mod5_adamw_aot_get_param_f64
T et_mod5_adamw_aot_get_rng_word
T et_mod5_adamw_aot_get_step
T et_mod5_adamw_aot_get_tokens
T et_mod5_adamw_aot_get_v_bits
T et_mod5_adamw_aot_load
T et_mod5_adamw_aot_load_f64
T et_mod5_adamw_aot_read_d1_shard
T et_mod5_adamw_aot_rng_next
T et_mod5_adamw_aot_save
T et_mod5_adamw_aot_set_cursor
T et_mod5_adamw_aot_set_param_f64
T et_mod5_adamw_aot_set_rng
T et_mod5_adamw_aot_state_sha256_hex
T et_mod5_adamw_aot_step
T et_mod5_adamw_aot_write_d1_shard
T et_mod5_adamw_compute_config_hash
T et_mod5_adamw_compute_lr
T et_mod5_adamw_create
T et_mod5_adamw_destroy
T et_mod5_adamw_get_cumulative_tokens
T et_mod5_adamw_get_cursor
T et_mod5_adamw_get_moments
T et_mod5_adamw_get_params
T et_mod5_adamw_get_rng
T et_mod5_adamw_get_step
T et_mod5_adamw_load_snapshot
T et_mod5_adamw_rng_next_u64
T et_mod5_adamw_save_snapshot
T et_mod5_adamw_set_cursor
T et_mod5_adamw_set_params
T et_mod5_adamw_set_rng
T et_mod5_adamw_sha256_bytes
T et_mod5_adamw_sha256_file
T et_mod5_adamw_step
EOF
)"
if [[ "${exports}" != "${expected}" ]]; then
  printf 'unexpected MOD5 AdamW exports:\n%s\n' "${exports}" >&2
  exit 1
fi

# Explicit allowlist: no threads, foreign runtimes, dlopen, or exec imports.
imports="$(nm -u "${scratch}/mod5_adamw_state.o" | awk 'NF {print $NF}')"
while IFS= read -r symbol; do
  case "${symbol}" in
    ''|__errno_location|__stack_chk_fail|_GLOBAL_OFFSET_TABLE_|\
    bcmp|calloc|close|cos|fegetround|free|fstat|fstatat|fsync|getrandom|\
    malloc|memcmp|memcpy|memset|open|openat|read|renameat|\
    sqrt|sqrtf|strchr|strlen|strrchr|unlinkat|write) ;;
    *)
      printf 'unexpected MOD5 AdamW dependency: %s\n' "${symbol}" >&2
      exit 1
      ;;
  esac
done <<< "${imports}"

mv -f "${scratch}/libeshkol_transformer_mod5_adamw.a" "${artifact}/"
printf 'MOD5 AdamW deterministic archive and symbol audit: PASS (%s)\n' \
  "${artifact}/libeshkol_transformer_mod5_adamw.a"
