#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

cd "${PROJECT_ROOT}"
verify_toolchain
export LC_ALL=C
cc="${CC:-$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cc_path)}"
cxx="${CXX:-clang++}"
runner="$(eshkol_build_dir)/eshkol-run"
provider_dir="${PROJECT_ROOT}/build/pa4_wider"
bash scripts/build-pa4-wider.sh
scratch="$(mktemp -d "${provider_dir}/.aot.XXXXXX")"
diagnostic=''
cleanup() {
  local status=$?
  if [[ "${status}" != 0 && -n "${diagnostic}" && -s "${diagnostic}" ]]; then
    cat "${diagnostic}" >&2
  fi
  rm -rf -- "${scratch}"
}
trap cleanup EXIT
bash scripts/build-k1.sh "${scratch}/k1"

common=(-Wall -Wextra -Werror -Wpedantic -I include -I native)
cflags=(-std=c11 "${common[@]}" -ffp-contract=off -fno-fast-math)
bridge="${scratch}/pa4_wider_eshkol_bridge.o"
"${cc}" "${cflags[@]}" -O2 -c native/pa4_wider_eshkol_bridge.c -o "${bridge}"
exports="$(nm -g --defined-only "${bridge}" | awk 'NF {print $(NF-1), $NF}' | sort)"
expected_exports="$(printf 'T et_pa4_wider_aot_%s\n' \
  backward close f32_bits forward load_f32 open store_f32 store_i64 store_u8)"
[[ "${exports}" == "${expected_exports}" ]] || die "unexpected bridge exports: ${exports}"

imports="$(nm -u "${bridge}" | awk 'NF {print $NF}')"
while IFS= read -r symbol; do
  case "${symbol}" in
    __stack_chk_fail|_GLOBAL_OFFSET_TABLE_|memcpy|memset|strcmp|\
    et_pa4_wider_online_provider_v1|et_pa4_wider_ref_provider_v1|\
    et_kernel_runtime_discover|et_kernel_runtime_destroy|et_kernel_runtime_dispatch) ;;
    *) die "unexpected bridge import: ${symbol}" ;;
  esac
done <<< "${imports}"
printf 'PA4-Wider AOT bridge symbol allowlist: PASS\n'

for run in 1 2; do
  mkdir "${scratch}/cache-${run}" "${scratch}/link-${run}"
  # The pinned compiler removes positional object inputs after linking.
  cp "${bridge}" "${scratch}/link-${run}/"
  diagnostic="${scratch}/compile-${run}.log"
  timeout --foreground --kill-after=5s 60s \
    env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR -u ESHKOL_AOT_MODULE_CACHE_DIR \
    ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="${scratch}/cache-${run}" \
    "${runner}" --strict-types --no-stdlib \
      -L "${provider_dir}" -L "${scratch}/k1" \
      --lib eshkol_transformer_pa4_wider --lib eshkol_transformer_k1 \
      tests/pa4_wider/pa4_wider_aot_driver.esk "${scratch}/link-${run}/pa4_wider_eshkol_bridge.o" \
      -o "${scratch}/driver-${run}" >"${diagnostic}" 2>&1
  diagnostic="${scratch}/run-${run}.stderr"
  timeout --foreground --kill-after=5s 30s "${scratch}/driver-${run}" \
    >"${scratch}/run-${run}.stdout" 2>"${diagnostic}" || {
    cat "${scratch}/run-${run}.stdout" >&2
    die "compiled driver failed on run ${run}"
  }
  [[ ! -s "${diagnostic}" ]] || die "compiled driver emitted stderr"
done
diagnostic=''
readelf -h "${scratch}/driver-1" >/dev/null
cmp "${scratch}/driver-1" "${scratch}/driver-2"
cmp "${scratch}/run-1.stdout" "${scratch}/run-2.stdout"
awk '
  BEGIN {
    number = "^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$"
    heads[16] = 2; heads[32] = 1; heads[64] = 2
  }
  $0 == "PA4-Wider AOT malformed statuses PASS" { ++negative; next }
  $0 == "PA4-Wider AOT TRAIN MQA steps=12 gradients=6 PASS" { ++train; next }
  $1 == "PA4-Wider" && $2 == "AOT" && $3 == "PARITY" {
    tokens = $4
    if (!(tokens in heads) || ++parity[tokens] != 1 || NF != 10 ||
        $5 != heads[tokens] || $6 !~ number || $6 + 0 > 2e-7 ||
        $7 != "dQ=" (tokens * 64) || $8 != "dK=" (tokens * $5 * 16) ||
        $9 != "dV=" (tokens * $5 * 16) || $10 != "bitwise=PASS")
      exit 1
    next
  }
  $1 == "PA4-Wider" && $2 == "AOT" && $3 == "LOSS" {
    if (NF != 5 || $4 !~ /^[0-9]+$/ || $4 != steps || $5 !~ number ||
        (steps && $5 + 0 >= previous))
      exit 1
    previous = $5 + 0
    ++steps
    next
  }
  { exit 1 }
  END {
    if (negative != 1 || train != 1 || steps != 13 ||
        parity[16] != 1 || parity[32] != 1 || parity[64] != 1)
      exit 1
  }
' "${scratch}/run-1.stdout" || {
  cat "${scratch}/run-1.stdout" >&2
  die "invalid compiled parity or loss trace"
}
cat "${scratch}/run-1.stdout"
digest="$(sha256sum "${scratch}/driver-1")"
printf 'PA4-Wider AOT ELF SHA-256 %s\n' "${digest%% *}"
printf 'PA4-Wider AOT two fresh compilations, byte-identical ELF and stdout: PASS\n'

cat >"${scratch}/bridge_cpp_check.cpp" <<'CPP'
#include "pa4_wider_eshkol_bridge.h"

int main() {
  et_kernel_runtime *runtime = nullptr;
  if (et_pa4_wider_aot_open(1, &runtime) != 0 || runtime == nullptr)
    return 1;
  float q = 0, k = 0, v = 0, out = 0, dy = 0.5f, dq = 1, dk = 1, dv = 0;
  int64_t qpos, kpos;
  uint8_t mask;
  et_pa4_wider_aot_store_i64(&qpos, 0, 2);
  et_pa4_wider_aot_store_i64(&kpos, 0, 1);
  et_pa4_wider_aot_store_u8(&mask, 0, 1);
  et_pa4_wider_aot_store_f32(&v, 0, 1.25);
  const int forward = et_pa4_wider_aot_forward(
      runtime, 1, 1, 1, 1, 1, 1, &q, &k, &v, &qpos, &kpos, &mask, &out);
  const int backward = et_pa4_wider_aot_backward(
      runtime, 1, 1, 1, 1, 1, 1, &q, &k, &v, &qpos, &kpos, &mask,
      &dy, &dq, &dk, &dv);
  et_pa4_wider_aot_close(runtime);
  return forward == 0 && backward == 0 &&
         et_pa4_wider_aot_load_f32(&out, 0) == 1.25 &&
         et_pa4_wider_aot_f32_bits(&out, 0) == 0x3fa00000 &&
         dq == 0 && dk == 0 && dv == dy ? 0 : 2;
}
CPP
"${cxx}" -std=c++17 "${common[@]}" "${scratch}/bridge_cpp_check.cpp" "${bridge}" \
  "${provider_dir}/libeshkol_transformer_pa4_wider.a" \
  "${scratch}/k1/libeshkol_transformer_k1.a" -lm -o "${scratch}/bridge-cpp"
"${scratch}/bridge-cpp"
printf 'PA4-Wider AOT bridge C++17 header and linkage: PASS\n'

test=test_pa4_wider_eshkol_bridge
wrap=(-Wl,--wrap=malloc,--wrap=calloc,--wrap=realloc,--wrap=free)
"${cc}" "${cflags[@]}" -O2 "tests/pa4_wider/${test}.c" "${bridge}" \
  "${provider_dir}/libeshkol_transformer_pa4_wider.a" \
  "${scratch}/k1/libeshkol_transformer_k1.a" "${wrap[@]}" -lm -o "${scratch}/${test}"
"${scratch}/${test}"

"${cc}" "${cflags[@]}" -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined \
  -fno-sanitize-recover=all "tests/pa4_wider/${test}.c" native/pa4_wider_eshkol_bridge.c \
  native/pa4_wider_attention_provider.c native/kernel_abi.c "${wrap[@]}" -lm \
  -o "${scratch}/bridge-sanitized"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
  "${scratch}/bridge-sanitized"
printf 'PA4-Wider AOT bridge ASan/UBSan/LSan: PASS\n'
