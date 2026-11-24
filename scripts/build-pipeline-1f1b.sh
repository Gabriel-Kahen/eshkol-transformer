#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CC="${CC:-clang}"
export CXX="${CXX:-clang++}"
export LC_ALL=C

output="${1:-${root}/build/pipeline_1f1b}"
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

# C++17 header syntax and standard-layout check on public ABI and bridge headers.
"${CXX}" -std=c++17 -Wall -Wextra -Werror -Wpedantic \
  -I "${root}/include" -I "${root}/native" -x c++ -fsyntax-only - <<'CPP'
#include "eshkol_transformer/pipeline_1f1b_abi.h"
#include "pipeline_1f1b_eshkol_bridge.h"
#include <type_traits>
static_assert(std::is_standard_layout_v<et_pp_frame_header_t>);
static_assert(std::is_standard_layout_v<et_pp_schedule_step_t>);
static_assert(std::is_standard_layout_v<et_pp_schedule_t>);
static_assert(std::is_standard_layout_v<et_pp_config_t>);
static_assert(std::is_standard_layout_v<et_pp_run_telemetry_t>);
static_assert(std::is_same_v<et_pp_status_t, int32_t>);
CPP

# Build both static archives twice from fresh directories to prove determinism.
for trial in first second; do
  tdir="${scratch}/${trial}"
  mkdir -p -- "${tdir}"

  "${CC}" "${cflags[@]}" -c "${root}/native/pipeline_1f1b.c" \
    -o "${tdir}/pipeline_1f1b.o"
  ar rcsD "${tdir}/libeshkol_transformer_pipeline_1f1b.a" \
    "${tdir}/pipeline_1f1b.o"

  "${CC}" "${cflags[@]}" -c "${root}/native/pipeline_1f1b_eshkol_bridge.c" \
    -o "${tdir}/pipeline_1f1b_eshkol_bridge.o"
  ar rcsD "${tdir}/libeshkol_transformer_pipeline_1f1b_aot.a" \
    "${tdir}/pipeline_1f1b_eshkol_bridge.o"
done

cmp "${scratch}/first/libeshkol_transformer_pipeline_1f1b.a" \
    "${scratch}/second/libeshkol_transformer_pipeline_1f1b.a"
cmp "${scratch}/first/libeshkol_transformer_pipeline_1f1b_aot.a" \
    "${scratch}/second/libeshkol_transformer_pipeline_1f1b_aot.a"

# Audit exact defined global exports on libeshkol_transformer_pipeline_1f1b.a.
cat > "${scratch}/expected-core-exports" <<'EOF'
T et_pp_build_schedule_v1
T et_pp_layer_backward_v1
T et_pp_layer_forward_v1
T et_pp_layer_param_count_v1
T et_pp_p2p_recv_frame_v1
T et_pp_p2p_send_frame_v1
T et_pp_param_count_v1
T et_pp_run_multiprocess_step_v1
T et_pp_run_reference_step_v1
T et_pp_train_multiprocess_sgd_v1
T et_pp_train_reference_sgd_v1
T et_pp_validate_config_v1
EOF
nm -g --defined-only "${scratch}/first/libeshkol_transformer_pipeline_1f1b.a" |
  awk 'NF >= 3 {print $(NF-1), $NF}' | LC_ALL=C sort \
  > "${scratch}/actual-core-exports"
diff -u "${scratch}/expected-core-exports" "${scratch}/actual-core-exports"

# Audit undefined imports on pipeline_1f1b.o (no shared-memory / stdio / exec / thread imports).
core_imports="$(nm -u "${scratch}/first/pipeline_1f1b.o" | awk 'NF {print $NF}')"
while IFS= read -r symbol; do
  case "${symbol}" in
    ''|__errno_location|__stack_chk_fail|_GLOBAL_OFFSET_TABLE_|_exit|\
    calloc|clock_gettime|close|exp|fork|free|getpeername|getpid|getsockname|\
    getsockopt|kill|log|malloc|memcpy|memset|poll|recv|send|socketpair|sqrt|\
    waitpid) ;;
    *)
      printf 'unexpected pipeline_1f1b core dependency: %s\n' "${symbol}" >&2
      exit 1
      ;;
  esac
done <<< "${core_imports}"

# Audit exact defined global exports on libeshkol_transformer_pipeline_1f1b_aot.a.
cat > "${scratch}/expected-bridge-exports" <<'EOF'
T et_pp_aot_build_schedule
T et_pp_aot_layer_backward
T et_pp_aot_layer_forward
T et_pp_aot_load_f64
T et_pp_aot_load_i64
T et_pp_aot_make_nonfinite
T et_pp_aot_param_count
T et_pp_aot_probe_transport_error
T et_pp_aot_run_multiprocess_step
T et_pp_aot_run_reference_step
T et_pp_aot_store_f64
T et_pp_aot_store_i32
T et_pp_aot_train_multiprocess_sgd
T et_pp_aot_train_reference_sgd
EOF
nm -g --defined-only "${scratch}/first/libeshkol_transformer_pipeline_1f1b_aot.a" |
  awk 'NF >= 3 {print $(NF-1), $NF}' | LC_ALL=C sort \
  > "${scratch}/actual-bridge-exports"
diff -u "${scratch}/expected-bridge-exports" "${scratch}/actual-bridge-exports"

# Audit undefined imports on pipeline_1f1b_eshkol_bridge.o.
bridge_imports="$(nm -u "${scratch}/first/pipeline_1f1b_eshkol_bridge.o" | awk 'NF {print $NF}')"
while IFS= read -r symbol; do
  case "${symbol}" in
    ''|__stack_chk_fail|_GLOBAL_OFFSET_TABLE_|close|et_pp_*|memcpy|socketpair) ;;
    *)
      printf 'unexpected pipeline_1f1b bridge dependency: %s\n' "${symbol}" >&2
      exit 1
      ;;
  esac
done <<< "${bridge_imports}"

mv -f -- "${scratch}/first/libeshkol_transformer_pipeline_1f1b.a" "${output}/"
mv -f -- "${scratch}/first/libeshkol_transformer_pipeline_1f1b_aot.a" "${output}/"

core_sha="$(sha256sum "${output}/libeshkol_transformer_pipeline_1f1b.a")"
bridge_sha="$(sha256sum "${output}/libeshkol_transformer_pipeline_1f1b_aot.a")"
printf 'PIPELINE_1F1B deterministic archives and symbol audit: PASS (core=%s bridge=%s)\n' \
  "${core_sha%% *}" "${bridge_sha%% *}"
