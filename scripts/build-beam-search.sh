#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CC="${CC:-clang}"
export CXX="${CXX:-clang++}"
export LC_ALL=C

output="${1:-${root}/build/beam_search}"
mkdir -p -- "${output}"
scratch="$(mktemp -d "${output}/.build.XXXXXX")"
trap 'rm -rf -- "${scratch}"' EXIT

warnings=(-Wall -Wextra -Werror -Wpedantic)
cflags=(
  -std=c11 "${warnings[@]}"
  -O2 -ffp-contract=off -fexcess-precision=standard -fno-fast-math
  -fstack-protector-all -fPIC
  "-ffile-prefix-map=${root}=."
  -I "${root}/include" -I "${root}/native"
)

# C++17 header syntax and standard-layout verification.
"${CXX}" -std=c++17 "${warnings[@]}" \
  -I "${root}/include" -I "${root}/native" -x c++ -fsyntax-only - <<'CPP'
#include "eshkol_transformer/beam_search_abi.h"
#include "beam_search_eshkol_bridge.h"
#include <type_traits>
static_assert(ET_BEAM_ABI_MAJOR == 1u);
static_assert(ET_BEAM_OK == 0);
static_assert(std::is_same_v<et_beam_status, int32_t>);
static_assert(std::is_standard_layout_v<et_beam_config_v1>);
static_assert(std::is_standard_layout_v<et_beam_hypothesis_v1>);
static_assert(std::is_standard_layout_v<et_beam_state_v1>);
CPP

# Build both archives twice from fresh trial directories to prove determinism.
for trial in first second; do
  tdir="${scratch}/${trial}"
  mkdir -p -- "${tdir}"
  "${CC}" "${cflags[@]}" -c "${root}/native/beam_search.c" \
    -o "${tdir}/beam_search.o"
  ar rcsD "${tdir}/libeshkol_transformer_beam_search.a" \
    "${tdir}/beam_search.o"

  "${CC}" "${cflags[@]}" -c "${root}/native/beam_search_eshkol_bridge.c" \
    -o "${tdir}/beam_search_eshkol_bridge.o"
  ar rcsD "${tdir}/libeshkol_transformer_beam_search_aot.a" \
    "${tdir}/beam_search_eshkol_bridge.o"
done

cmp -- "${scratch}/first/libeshkol_transformer_beam_search.a" \
  "${scratch}/second/libeshkol_transformer_beam_search.a"
cmp -- "${scratch}/first/libeshkol_transformer_beam_search_aot.a" \
  "${scratch}/second/libeshkol_transformer_beam_search_aot.a"

# Audit defined global exports on libeshkol_transformer_beam_search.a.
cat > "${scratch}/expected-core-exports" <<'EOF'
T et_beam_active_count_v1
T et_beam_apply_min_length_v1
T et_beam_apply_no_repeat_ngram_v1
T et_beam_apply_repetition_penalty_v1
T et_beam_finalize_v1
T et_beam_finished_count_v1
T et_beam_get_active_beam_v1
T et_beam_init_v1
T et_beam_is_done_v1
T et_beam_log_softmax_v1
T et_beam_score_v1
T et_beam_step_v1
T et_beam_steps_done_v1
T et_beam_stopped_early_v1
T et_beam_validate_config_v1
T et_beam_wu_length_penalty_v1
EOF
nm -g --defined-only "${scratch}/first/libeshkol_transformer_beam_search.a" |
  awk 'NF >= 3 {print $(NF-1), $NF}' | LC_ALL=C sort \
  > "${scratch}/actual-core-exports"
diff -u "${scratch}/expected-core-exports" "${scratch}/actual-core-exports"

# Audit undefined imports on beam_search.o: ZERO heap allocations (no malloc/calloc/realloc/free).
core_imports="$(nm -u "${scratch}/first/beam_search.o" | awk 'NF {print $NF}')"
while IFS= read -r symbol; do
  case "${symbol}" in
    ''|__stack_chk_fail|_GLOBAL_OFFSET_TABLE_|\
    exp|fegetround|log|memcpy|memset|pow) ;;
    *)
      printf 'unexpected beam_search core dependency: %s\n' "${symbol}" >&2
      exit 1
      ;;
  esac
done <<< "${core_imports}"

# Audit defined global exports on libeshkol_transformer_beam_search_aot.a.
cat > "${scratch}/expected-bridge-exports" <<'EOF'
T et_beam_aot_active_count
T et_beam_aot_active_len
T et_beam_aot_active_log_prob
T et_beam_aot_active_token
T et_beam_aot_apply_min_length
T et_beam_aot_apply_no_repeat_ngram
T et_beam_aot_apply_rep_penalty
T et_beam_aot_config_bytes
T et_beam_aot_finalize
T et_beam_aot_finished_count
T et_beam_aot_hyp_bytes
T et_beam_aot_hyp_finished
T et_beam_aot_hyp_gen_len
T et_beam_aot_hyp_score
T et_beam_aot_hyp_seq_len
T et_beam_aot_hyp_sum_log_prob
T et_beam_aot_hyp_token
T et_beam_aot_init
T et_beam_aot_is_done
T et_beam_aot_load_f64
T et_beam_aot_load_i64
T et_beam_aot_load_u32
T et_beam_aot_log_softmax
T et_beam_aot_nan_f64
T et_beam_aot_neg_inf_f64
T et_beam_aot_pos_inf_f64
T et_beam_aot_set_config
T et_beam_aot_state_bytes
T et_beam_aot_step
T et_beam_aot_steps_done
T et_beam_aot_stopped_early
T et_beam_aot_store_f64
T et_beam_aot_store_u32
EOF
nm -g --defined-only "${scratch}/first/libeshkol_transformer_beam_search_aot.a" |
  awk 'NF >= 3 {print $(NF-1), $NF}' | LC_ALL=C sort \
  > "${scratch}/actual-bridge-exports"
diff -u "${scratch}/expected-bridge-exports" "${scratch}/actual-bridge-exports"

# Audit undefined imports on beam_search_eshkol_bridge.o: ZERO heap allocations.
bridge_imports="$(nm -u "${scratch}/first/beam_search_eshkol_bridge.o" | awk 'NF {print $NF}')"
while IFS= read -r symbol; do
  case "${symbol}" in
    ''|__stack_chk_fail|_GLOBAL_OFFSET_TABLE_|\
    et_beam_*|memcpy|memset) ;;
    *)
      printf 'unexpected beam_search bridge dependency: %s\n' "${symbol}" >&2
      exit 1
      ;;
  esac
done <<< "${bridge_imports}"

mv -f -- "${scratch}/first/libeshkol_transformer_beam_search.a" "${output}/"
mv -f -- "${scratch}/first/libeshkol_transformer_beam_search_aot.a" "${output}/"

core_sha="$(sha256sum "${output}/libeshkol_transformer_beam_search.a")"
bridge_sha="$(sha256sum "${output}/libeshkol_transformer_beam_search_aot.a")"
printf 'BEAM_SEARCH deterministic archives and zero-alloc symbol audit: PASS (core=%s bridge=%s)\n' \
  "${core_sha%% *}" "${bridge_sha%% *}"
