#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

cd "${PROJECT_ROOT}"
verify_toolchain
export LC_ALL=C
export CC="${CC:-$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cc_path)}"
export CXX="${CXX:-$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cxx_path)}"

runner="$(eshkol_build_dir)/eshkol-run"
moe_dir="${PROJECT_ROOT}/build/mod5_moe"

bash scripts/build-mod5-moe.sh

scratch="$(mktemp -d "${moe_dir}/.test.XXXXXX")"
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

common=(-Wall -Wextra -Werror -Wpedantic -ffp-contract=off -fno-fast-math
        -I include -I native)
cflags=(-std=c11 "${common[@]}" -fexcess-precision=standard)
cppflags=(-std=c++17 "${common[@]}")
san=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer
     -fno-sanitize-recover=all)
wrap=(-Wl,--wrap=malloc,--wrap=calloc,--wrap=realloc,--wrap=free)

# 1. C++17 header and linkage check.
"${CXX}" "${cppflags[@]}" -O2 tests/mod5_moe/header_moe_cpp.cpp \
  "${moe_dir}/mod5_moe_eshkol_bridge.o" \
  "${moe_dir}/libeshkol_transformer_mod5_moe.a" \
  "${scratch}/k1/libeshkol_transformer_k1.a" -lm -o "${scratch}/header-cpp"
"${scratch}/header-cpp"
printf 'MOD5 MoE C++17 header and linkage: PASS\n'

# 2. Native C11 suite (normal -O2, allocation-trapped dispatch).
"${CC}" "${cflags[@]}" -O2 tests/mod5_moe/test_mod5_moe_native.c \
  "${moe_dir}/mod5_moe_eshkol_bridge.o" \
  "${moe_dir}/libeshkol_transformer_mod5_moe.a" \
  "${scratch}/k1/libeshkol_transformer_k1.a" "${wrap[@]}" -lm \
  -o "${scratch}/test_mod5_moe_native"
"${scratch}/test_mod5_moe_native"

# 3. Independent binary64 Python oracle.
python3 tests/mod5_moe/test_mod5_moe_oracle.py "${scratch}/test_mod5_moe_native"

# 4. Native C11 + C++17 under ASan/UBSan/LSan.
for src in kernel_abi mod5_moe_provider mod5_moe_eshkol_bridge; do
  "${CC}" "${cflags[@]}" "${san[@]}" -c "native/${src}.c" \
    -o "${scratch}/${src}.san.o"
done
san_objs=("${scratch}/mod5_moe_eshkol_bridge.san.o"
          "${scratch}/mod5_moe_provider.san.o"
          "${scratch}/kernel_abi.san.o")
"${CC}" "${cflags[@]}" "${san[@]}" tests/mod5_moe/test_mod5_moe_native.c \
  "${san_objs[@]}" "${wrap[@]}" -lm -o "${scratch}/test_mod5_moe_native.san"
"${CXX}" "${cppflags[@]}" "${san[@]}" tests/mod5_moe/header_moe_cpp.cpp \
  "${san_objs[@]}" -lm -o "${scratch}/header-cpp.san"

ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
  UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  "${scratch}/test_mod5_moe_native.san"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
  UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  "${scratch}/header-cpp.san"
printf 'MOD5 MoE ASan/UBSan/LSan: PASS\n'

# 5. Compiled Eshkol AOT 4-expert Top-2 MoE driver (compiled twice from fresh dirs).
for run in 1 2; do
  mkdir "${scratch}/cache-${run}" "${scratch}/link-${run}"
  # The pinned driver removes positional object inputs after linking.
  cp "${moe_dir}/mod5_moe_eshkol_bridge.o" "${scratch}/link-${run}/"
  diagnostic="${scratch}/compile-${run}.log"
  timeout --foreground --kill-after=5s 90s \
    env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR -u ESHKOL_AOT_MODULE_CACHE_DIR \
    ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="${scratch}/cache-${run}" \
    "${runner}" --strict-types --optimize 0 --no-stdlib \
      -L "${moe_dir}" -L "${scratch}/k1" \
      --lib eshkol_transformer_mod5_moe \
      --lib eshkol_transformer_k1 \
      tests/mod5_moe/test_mod5_moe_aot.esk \
      "${scratch}/link-${run}/mod5_moe_eshkol_bridge.o" \
      -o "${scratch}/moe-aot-${run}" >"${diagnostic}" 2>&1
  diagnostic="${scratch}/run-${run}.stderr"
  timeout --foreground --kill-after=5s 60s "${scratch}/moe-aot-${run}" \
    >"${scratch}/run-${run}.stdout" 2>"${diagnostic}" || {
    cat "${scratch}/run-${run}.stdout" >&2
    die "compiled MoE AOT executable failed on run ${run}"
  }
  [[ ! -s "${diagnostic}" ]] || die "compiled MoE AOT emitted stderr"
done
diagnostic=''

readelf -h "${scratch}/moe-aot-1" >/dev/null
cmp "${scratch}/moe-aot-1" "${scratch}/moe-aot-2"
cmp "${scratch}/run-1.stdout" "${scratch}/run-2.stdout"

awk '
  BEGIN { number = "^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$" }
  $0 == "MOD5 MOE AOT malformed statuses PASS" { ++negative; next }
  $1 == "MOD5" && $2 == "MOE" && $3 == "AOT" && $4 == "FD_GRADIENTS" {
    if (NF != 7 || $5 != "checked=20" || $6 !~ /^max_abs_err=/ || $7 != "PASS")
      exit 1
    ++fd_ok
    next
  }
  $1 == "MOD5" && $2 == "MOE" && $3 == "AOT" && $4 == "STEP" {
    if (NF != 10 || $5 !~ /^[0-9]+$/ || $5 != steps ||
        $6 !~ /^total=/ || $7 !~ /^task=/ || $8 !~ /^aux=/ ||
        $9 != "evals=16" || $10 !~ /^counts=[0-9]+,[0-9]+,[0-9]+,[0-9]+$/)
      exit 1
    tot_str = substr($6, 7)
    if (tot_str !~ number || tot_str + 0 < 0 || (steps && tot_str + 0 >= previous))
      exit 1
    if (!steps) initial = tot_str
    final = tot_str
    previous = tot_str + 0
    ++steps
    next
  }
  $1 == "MOD5" && $2 == "MOE" && $3 == "AOT" && $4 == "RESULT" {
    if (steps != 13 || NF != 10 || $5 != "experts=4" || $6 != "top_k=2" ||
        $7 != "steps=12" || $8 != "initial=" initial ||
        $9 != "final=" final || $10 !~ /^checksum=/)
      exit 1
    ++result_ok
    next
  }
  { exit 1 }
  END {
    if (negative != 1 || fd_ok != 1 || result_ok != 1 || steps != 13)
      exit 1
  }
' "${scratch}/run-1.stdout" || {
  cat "${scratch}/run-1.stdout" >&2
  die "invalid MoE AOT output trace"
}

cat "${scratch}/run-1.stdout"
digest="$(sha256sum "${scratch}/moe-aot-1")"
printf 'MOD5 MoE AOT ELF SHA-256 %s\n' "${digest%% *}"
printf 'MOD5 MoE AOT two fresh compilations, byte-identical ELF and stdout: PASS\n'
