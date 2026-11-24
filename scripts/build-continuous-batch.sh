#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CC="${CC:-clang}"
export CXX="${CXX:-clang++}"
export LC_ALL=C

output="${1:-${root}/build/continuous_batch}"
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

# C++17 header syntax and layout check on public ABI and bridge headers.
"${CXX}" -std=c++17 -Wall -Wextra -Werror -Wpedantic \
  -I "${root}/include" -I "${root}/native" -x c++ -fsyntax-only - <<'CPP'
#include "eshkol_transformer/continuous_batch_abi.h"
#include "continuous_batch_eshkol_bridge.h"
#include <type_traits>
static_assert(std::is_standard_layout_v<et_cb_config>);
static_assert(std::is_standard_layout_v<et_cb_request_spec>);
static_assert(std::is_standard_layout_v<et_cb_plan_entry>);
static_assert(std::is_standard_layout_v<et_cb_step_batch>);
static_assert(std::is_standard_layout_v<et_cb_commit_item>);
static_assert(std::is_standard_layout_v<et_cb_request_info>);
static_assert(std::is_standard_layout_v<et_cb_stats>);
static_assert(std::is_same_v<et_cb_status, int32_t>);
CPP

# Build both archives twice from fresh trial directories to verify determinism.
for trial in first second; do
  tdir="${scratch}/${trial}"
  mkdir -p -- "${tdir}"
  "${CC}" "${cflags[@]}" -c "${root}/native/continuous_batch.c" \
    -o "${tdir}/continuous_batch.o"
  ar rcsD "${tdir}/libeshkol_transformer_continuous_batch.a" \
    "${tdir}/continuous_batch.o"

  "${CC}" "${cflags[@]}" -c "${root}/native/continuous_batch_eshkol_bridge.c" \
    -o "${tdir}/continuous_batch_eshkol_bridge.o"
  ar rcsD "${tdir}/libeshkol_transformer_continuous_batch_aot.a" \
    "${tdir}/continuous_batch_eshkol_bridge.o"
done

cmp "${scratch}/first/libeshkol_transformer_continuous_batch.a" \
    "${scratch}/second/libeshkol_transformer_continuous_batch.a"
cmp "${scratch}/first/libeshkol_transformer_continuous_batch_aot.a" \
    "${scratch}/second/libeshkol_transformer_continuous_batch_aot.a"

# Audit defined global exports on libeshkol_transformer_continuous_batch.a.
cat > "${scratch}/expected-core-exports" <<'EOF'
T et_cb_cancel
T et_cb_enqueue
T et_cb_get_request_info
T et_cb_get_stats
T et_cb_init
T et_cb_step_commit
T et_cb_step_plan
T et_cb_validate_config
T et_cb_verify_invariants
EOF
nm -g --defined-only "${scratch}/first/libeshkol_transformer_continuous_batch.a" |
  awk 'NF >= 3 {print $(NF-1), $NF}' | LC_ALL=C sort \
  > "${scratch}/actual-core-exports"
diff -u "${scratch}/expected-core-exports" "${scratch}/actual-core-exports"

# Audit undefined imports on continuous_batch.o: strictly ZERO malloc/calloc/realloc/free!
core_imports="$(nm -u "${scratch}/first/continuous_batch.o" | awk 'NF {print $NF}')"
while IFS= read -r symbol; do
  case "${symbol}" in
    ''|__stack_chk_fail|_GLOBAL_OFFSET_TABLE_|memcpy|memmove|memset) ;;
    *)
      printf 'unexpected continuous_batch core dependency: %s\n' "${symbol}" >&2
      exit 1
      ;;
  esac
done <<< "${core_imports}"

# Audit defined global exports on libeshkol_transformer_continuous_batch_aot.a.
cat > "${scratch}/expected-bridge-exports" <<'EOF'
T et_cb_aot_cancel
T et_cb_aot_cancelled_count
T et_cb_aot_commit_begin
T et_cb_aot_commit_set_eos
T et_cb_aot_commit_set_raw
T et_cb_aot_create
T et_cb_aot_current_step
T et_cb_aot_destroy
T et_cb_aot_enqueue
T et_cb_aot_finished_count
T et_cb_aot_free_blocks
T et_cb_aot_plan_decode_tokens
T et_cb_aot_plan_entry_chunk_tokens
T et_cb_aot_plan_entry_count
T et_cb_aot_plan_entry_is_prefill
T et_cb_aot_plan_entry_new_blocks
T et_cb_aot_plan_entry_produces_token
T et_cb_aot_plan_entry_req_id
T et_cb_aot_plan_entry_token_offset
T et_cb_aot_plan_entry_total_blocks
T et_cb_aot_plan_free_blocks
T et_cb_aot_plan_preempted_count
T et_cb_aot_plan_preempted_id
T et_cb_aot_plan_prefill_tokens
T et_cb_aot_plan_step_index
T et_cb_aot_plan_total_tokens
T et_cb_aot_plan_used_blocks
T et_cb_aot_req_allocated_blocks
T et_cb_aot_req_completion_step
T et_cb_aot_req_computed_tokens
T et_cb_aot_req_eos_stopped
T et_cb_aot_req_generated_tokens
T et_cb_aot_req_preempt_count
T et_cb_aot_req_state
T et_cb_aot_req_ttft_steps
T et_cb_aot_running_count
T et_cb_aot_step_commit
T et_cb_aot_step_plan
T et_cb_aot_total_generated_tokens
T et_cb_aot_total_preemptions
T et_cb_aot_total_scheduled_tokens
T et_cb_aot_used_blocks
T et_cb_aot_verify_invariants
T et_cb_aot_waiting_count
EOF
nm -g --defined-only "${scratch}/first/libeshkol_transformer_continuous_batch_aot.a" |
  awk 'NF >= 3 {print $(NF-1), $NF}' | LC_ALL=C sort \
  > "${scratch}/actual-bridge-exports"
diff -u "${scratch}/expected-bridge-exports" "${scratch}/actual-bridge-exports"

# Audit undefined imports on continuous_batch_eshkol_bridge.o.
bridge_imports="$(nm -u "${scratch}/first/continuous_batch_eshkol_bridge.o" | awk 'NF {print $NF}')"
while IFS= read -r symbol; do
  case "${symbol}" in
    ''|__stack_chk_fail|_GLOBAL_OFFSET_TABLE_|calloc|et_cb_*|free|memset) ;;
    *)
      printf 'unexpected continuous_batch bridge dependency: %s\n' "${symbol}" >&2
      exit 1
      ;;
  esac
done <<< "${bridge_imports}"

mv -f -- "${scratch}/first/libeshkol_transformer_continuous_batch.a" "${output}/"
mv -f -- "${scratch}/first/libeshkol_transformer_continuous_batch_aot.a" "${output}/"

core_sha="$(sha256sum "${output}/libeshkol_transformer_continuous_batch.a")"
bridge_sha="$(sha256sum "${output}/libeshkol_transformer_continuous_batch_aot.a")"
printf 'CONTINUOUS_BATCH deterministic archives and symbol audit: PASS (core=%s bridge=%s)\n' \
  "${core_sha%% *}" "${bridge_sha%% *}"
