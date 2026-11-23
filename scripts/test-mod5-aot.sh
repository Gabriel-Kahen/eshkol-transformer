#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

cd "${PROJECT_ROOT}"
verify_toolchain
export LC_ALL=C
cc="${CC:-$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cc_path)}"
runner="$(eshkol_build_dir)/eshkol-run"
provider_dir="${PROJECT_ROOT}/build/mod5"
bash scripts/build-mod5.sh
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

cflags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -ffp-contract=off
        -fexcess-precision=standard -fno-fast-math -I include -I native)
"${cc}" "${cflags[@]}" -O2 -c native/mod5_eshkol_bridge.c -o "${scratch}/mod5_eshkol_bridge.o"
exports="$(nm -g --defined-only "${scratch}/mod5_eshkol_bridge.o" |
  awk 'NF && ($2 != "T" || $3 !~ /^et_mod5_aot_/) { print }')"
[[ -z "${exports}" ]] || die "unexpected bridge exports: ${exports}"

for run in 1 2; do
  mkdir "${scratch}/cache-${run}" "${scratch}/link-${run}"
  # The pinned driver removes positional object inputs after linking.
  cp "${scratch}/mod5_eshkol_bridge.o" "${scratch}/link-${run}/"
  diagnostic="${scratch}/compile-${run}.log"
  timeout --foreground --kill-after=5s 60s \
    env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR -u ESHKOL_AOT_MODULE_CACHE_DIR \
    ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="${scratch}/cache-${run}" \
    "${runner}" --strict-types --no-stdlib \
      -L "${provider_dir}" -L "${scratch}/k1" \
      --lib eshkol_transformer_mod5 --lib eshkol_transformer_k1 \
      tests/mod5/mod5_aot_training.esk "${scratch}/link-${run}/mod5_eshkol_bridge.o" \
      -o "${scratch}/training-${run}" >"${diagnostic}" 2>&1
  diagnostic="${scratch}/run-${run}.stderr"
  timeout --foreground --kill-after=5s 30s "${scratch}/training-${run}" \
    >"${scratch}/run-${run}.stdout" 2>"${diagnostic}"
  [[ ! -s "${diagnostic}" ]] || die "compiled training emitted stderr"
done
diagnostic=''
cmp "${scratch}/run-1.stdout" "${scratch}/run-2.stdout"
awk '
  BEGIN { number = "^-?[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$" }
  $0 == "MOD5 AOT malformed statuses PASS" { ++negative; next }
  $1 == "MOD5" && $2 == "AOT" && $3 == "LOSS" {
    variant = $4
    if ((variant != "MQA" && variant != "GQA") || NF != 6 ||
        $5 !~ /^[0-9]+$/ || $5 != steps[variant] || $6 !~ number ||
        $6 + 0 < 0 || (steps[variant] && $6 + 0 >= previous[variant]))
      exit 1
    if (!steps[variant]) initial[variant] = $6
    final[variant] = $6
    previous[variant] = $6 + 0
    ++steps[variant]
    next
  }
  $1 == "MOD5" && $2 == "AOT" && $3 == "RESULT" {
    variant = $4
    if (steps[variant] != 17 || NF != 8 || $5 != "initial=" initial[variant] ||
        $6 != "final=" final[variant] || $7 !~ /^checksum=/ ||
        substr($7, 10) !~ number || $8 != "gradients=30")
      exit 1
    ++result[variant]
    next
  }
  { exit 1 }
  END {
    if (negative != 1 || result["MQA"] != 1 || result["GQA"] != 1 ||
        steps["MQA"] != 17 || steps["GQA"] != 17)
      exit 1
  }
' "${scratch}/run-1.stdout" || die "invalid AOT loss trace"
cat "${scratch}/run-1.stdout"
printf 'MOD5 AOT two fresh compilations, byte-identical stdout: PASS\n'

san=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
wrap=(-Wl,--wrap=malloc,--wrap=calloc,--wrap=realloc,--wrap=free)
"${cc}" "${cflags[@]}" "${san[@]}" tests/mod5/test_mod5_eshkol_bridge.c \
  native/mod5_eshkol_bridge.c native/mod5_primitives_provider.c native/kernel_abi.c \
  "${wrap[@]}" -lm -o "${scratch}/bridge-san"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  "${scratch}/bridge-san"
