#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CC="${CC:-clang}"
export CXX="${CXX:-clang++}"
export LC_ALL=C

output="${1:-${root}/build/constrained_decode}"
mkdir -p -- "${output}"
scratch="$(mktemp -d "${output}/.build.XXXXXX")"
trap 'rm -rf -- "${scratch}"' EXIT

cflags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic
  -O2 -ffp-contract=off -fexcess-precision=standard -fno-fast-math
  -fstack-protector-all -fPIC
  "-ffile-prefix-map=${root}=."
  -I "${root}/include" -I "${root}/native"
)

# C++17 header check on public ABI and bridge headers.
"${CXX}" -std=c++17 -Wall -Wextra -Werror -Wpedantic \
  -I "${root}/include" -I "${root}/native" -x c++ -fsyntax-only - <<'CPP'
#include "eshkol_transformer/constrained_decode_abi.h"
#include "constrained_decode_eshkol_bridge.h"
#include <type_traits>
static_assert(std::is_standard_layout_v<et_cd_json_field_spec>);
static_assert(std::is_standard_layout_v<et_cd_dfa_info>);
static_assert(std::is_standard_layout_v<et_cd_vocab_index_info>);
static_assert(std::is_same_v<et_cd_status, int32_t>);
CPP

# Build both archives twice from fresh trial directories to verify determinism.
for trial in first second; do
  tdir="${scratch}/${trial}"
  mkdir -p -- "${tdir}"
  "${CC}" "${cflags[@]}" -c "${root}/native/constrained_decode.c" \
    -o "${tdir}/constrained_decode.o"
  ar rcsD "${tdir}/libeshkol_transformer_constrained_decode.a" \
    "${tdir}/constrained_decode.o"

  "${CC}" "${cflags[@]}" -c "${root}/native/constrained_decode_eshkol_bridge.c" \
    -o "${tdir}/constrained_decode_eshkol_bridge.o"
  ar rcsD "${tdir}/libeshkol_transformer_constrained_decode_aot.a" \
    "${tdir}/constrained_decode_eshkol_bridge.o"
done

cmp "${scratch}/first/libeshkol_transformer_constrained_decode.a" \
    "${scratch}/second/libeshkol_transformer_constrained_decode.a"
cmp "${scratch}/first/libeshkol_transformer_constrained_decode_aot.a" \
    "${scratch}/second/libeshkol_transformer_constrained_decode_aot.a"

# Audit defined global exports on libeshkol_transformer_constrained_decode.a.
cat > "${scratch}/expected-core-exports" <<'EOF'
T et_cd_advance_token_v1
T et_cd_apply_logit_mask_v1
T et_cd_dfa_accepts_bytes
T et_cd_dfa_artifact_size
T et_cd_dfa_build_raw
T et_cd_dfa_compile_flat_json_schema
T et_cd_dfa_compile_literal_choices
T et_cd_dfa_destroy
T et_cd_dfa_info_get
T et_cd_dfa_is_accepting
T et_cd_dfa_is_live
T et_cd_dfa_load
T et_cd_dfa_next_state
T et_cd_dfa_parse
T et_cd_dfa_save
T et_cd_dfa_serialize
T et_cd_vocab_index_allowed_count
T et_cd_vocab_index_build
T et_cd_vocab_index_copy_mask_words
T et_cd_vocab_index_destroy
T et_cd_vocab_index_info_get
T et_cd_vocab_index_is_token_allowed
EOF
nm -g --defined-only "${scratch}/first/libeshkol_transformer_constrained_decode.a" |
  awk 'NF >= 3 {print $(NF-1), $NF}' | LC_ALL=C sort \
  > "${scratch}/actual-core-exports"
diff -u "${scratch}/expected-core-exports" "${scratch}/actual-core-exports"

# Audit undefined imports on constrained_decode.o (no threads/dlopen/exec/stdio).
core_imports="$(nm -u "${scratch}/first/constrained_decode.o" | awk 'NF {print $NF}')"
while IFS= read -r symbol; do
  case "${symbol}" in
    ''|__errno_location|__stack_chk_fail|_GLOBAL_OFFSET_TABLE_|\
    bcmp|calloc|close|fegetround|free|fstat|fsync|malloc|memcmp|memcpy|memset|\
    mkstemp|open|read|rename|strcmp|strlen|strrchr|unlink|write) ;;
    *)
      printf 'unexpected constrained_decode core dependency: %s\n' "${symbol}" >&2
      exit 1
      ;;
  esac
done <<< "${core_imports}"

# Audit defined global exports on libeshkol_transformer_constrained_decode_aot.a.
cat > "${scratch}/expected-bridge-exports" <<'EOF'
T et_cd_aot_advance_token
T et_cd_aot_apply_mask
T et_cd_aot_artifact_sha256_word
T et_cd_aot_check_dead_end_dfa
T et_cd_aot_check_dead_state_mask
T et_cd_aot_check_illegal_advance
T et_cd_aot_check_inf_logit_preserves_output
T et_cd_aot_check_nan_logit_preserves_output
T et_cd_aot_check_tampered_artifact
T et_cd_aot_create_enum_grammar
T et_cd_aot_create_json_schema
T et_cd_aot_current_state
T et_cd_aot_destroy
T et_cd_aot_eos_token_id
T et_cd_aot_find_token_by_tag
T et_cd_aot_get_masked_logit
T et_cd_aot_is_finished
T et_cd_aot_live_states
T et_cd_aot_masked_is_allowed
T et_cd_aot_num_states
T et_cd_aot_output_accepted_by_dfa
T et_cd_aot_output_byte
T et_cd_aot_output_len
T et_cd_aot_reset
T et_cd_aot_set_logit
T et_cd_aot_start_state
T et_cd_aot_verify_artifact_roundtrip
T et_cd_aot_vocab_size
EOF
nm -g --defined-only "${scratch}/first/libeshkol_transformer_constrained_decode_aot.a" |
  awk 'NF >= 3 {print $(NF-1), $NF}' | LC_ALL=C sort \
  > "${scratch}/actual-bridge-exports"
diff -u "${scratch}/expected-bridge-exports" "${scratch}/actual-bridge-exports"

# Audit undefined imports on constrained_decode_eshkol_bridge.o.
bridge_imports="$(nm -u "${scratch}/first/constrained_decode_eshkol_bridge.o" | awk 'NF {print $NF}')"
while IFS= read -r symbol; do
  case "${symbol}" in
    ''|__stack_chk_fail|_GLOBAL_OFFSET_TABLE_|\
    bcmp|calloc|et_cd_*|free|malloc|memcmp|memcpy|memset|strlen) ;;
    *)
      printf 'unexpected constrained_decode bridge dependency: %s\n' "${symbol}" >&2
      exit 1
      ;;
  esac
done <<< "${bridge_imports}"

mv -f -- "${scratch}/first/libeshkol_transformer_constrained_decode.a" "${output}/"
mv -f -- "${scratch}/first/libeshkol_transformer_constrained_decode_aot.a" "${output}/"

core_sha="$(sha256sum "${output}/libeshkol_transformer_constrained_decode.a")"
bridge_sha="$(sha256sum "${output}/libeshkol_transformer_constrained_decode_aot.a")"
printf 'CONSTRAINED_DECODE deterministic archives and symbol audit: PASS (core=%s bridge=%s)\n' \
  "${core_sha%% *}" "${bridge_sha%% *}"
