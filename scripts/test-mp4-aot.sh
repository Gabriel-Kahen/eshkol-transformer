#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

cd "${PROJECT_ROOT}"
verify_toolchain
export LC_ALL=C

cc="${CC:-$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cc_path)}"
cxx="${CXX:-$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cxx_path)}"
runner="$(eshkol_build_dir)/eshkol-run"

provider_dir="${PROJECT_ROOT}/build/mp4"
aot_dir="${PROJECT_ROOT}/build/mp4_aot"
bash scripts/build-mp4.sh
bash scripts/build-mp4-aot.sh

scratch="$(mktemp -d "${aot_dir}/.test.XXXXXX")"
diagnostic=''
cleanup() {
  local status=$?
  if [[ "${status}" != 0 && -n "${diagnostic}" && -s "${diagnostic}" ]]; then
    cat "${diagnostic}" >&2
  fi
  rm -rf -- "${scratch}"
}
trap cleanup EXIT

cp "${aot_dir}/libeshkol_transformer_mp4_aot.a" "${scratch}/first.a"
bash scripts/build-mp4-aot.sh
cmp "${scratch}/first.a" "${aot_dir}/libeshkol_transformer_mp4_aot.a"
printf 'MP4 AOT archive reproducibility: PASS\n'

bash scripts/build-k1.sh "${scratch}/k1"

common=(-Wall -Wextra -Werror -Wpedantic -ffp-contract=off -fno-fast-math -I include -I native)
cflags=(-std=c11 "${common[@]}" -fexcess-precision=standard)
cppflags=(-std=c++17 "${common[@]}")
san=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
wrap=(-Wl,--wrap=malloc,--wrap=calloc,--wrap=realloc,--wrap=free)

# 1. C++17 header and linkage check.
"${cxx}" "${cppflags[@]}" tests/mp4/header_bridge_cpp.cpp \
  "${aot_dir}/libeshkol_transformer_mp4_aot.a" \
  "${provider_dir}/libeshkol_transformer_mp4.a" \
  "${scratch}/k1/libeshkol_transformer_k1.a" \
  -lm -o "${scratch}/header-bridge-cpp"
"${scratch}/header-bridge-cpp"
printf 'MP4 AOT bridge C++17 header and linkage: PASS\n'

# 2. Native C11 bridge test (normal mode + allocation-free dispatch guard).
"${cc}" "${cflags[@]}" -O2 -DET_MP4_BRIDGE_WRAP_ALLOC \
  tests/mp4/test_mp4_aot_bridge.c \
  "${aot_dir}/libeshkol_transformer_mp4_aot.a" \
  "${provider_dir}/libeshkol_transformer_mp4.a" \
  "${scratch}/k1/libeshkol_transformer_k1.a" \
  "${wrap[@]}" -lm -o "${scratch}/bridge-normal"
"${scratch}/bridge-normal"

# 3. Native C11 bridge test under ASan/UBSan/LSan.
"${cc}" "${cflags[@]}" "${san[@]}" tests/mp4/test_mp4_aot_bridge.c \
  native/mp4_eshkol_bridge.c native/mp4_mixed_precision_provider.c \
  native/kernel_abi.c -lm -o "${scratch}/bridge-san"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  "${scratch}/bridge-san"
printf 'MP4 AOT bridge ASan/UBSan/LSan: PASS\n'

# 4. Compiled Eshkol AOT training verification: two fresh compilations ->
#    byte-identical ELFs and identical stdout.
for run in 1 2; do
  mkdir -p "${scratch}/build-${run}" "${scratch}/cache-${run}"
  diagnostic="${scratch}/compile-${run}.log"
  timeout --foreground --kill-after=5s 60s \
    env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR -u ESHKOL_AOT_MODULE_CACHE_DIR \
    ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="${scratch}/cache-${run}" \
    "${runner}" --strict-types --optimize 0 --no-stdlib \
      -L "${aot_dir}" -L "${provider_dir}" -L "${scratch}/k1" \
      --lib eshkol_transformer_mp4_aot \
      --lib eshkol_transformer_mp4 \
      --lib eshkol_transformer_k1 \
      tests/mp4/test_mp4_aot.esk \
      -o "${scratch}/build-${run}/mp4-aot-test" >"${diagnostic}" 2>&1
  diagnostic="${scratch}/run-${run}.stderr"
  timeout --foreground --kill-after=5s 30s "${scratch}/build-${run}/mp4-aot-test" \
    >"${scratch}/run-${run}.stdout" 2>"${diagnostic}"
  [[ ! -s "${diagnostic}" ]] || die "compiled MP4 AOT binary emitted stderr"
done
diagnostic=''

cmp "${scratch}/build-1/mp4-aot-test" "${scratch}/build-2/mp4-aot-test"
cmp "${scratch}/run-1.stdout" "${scratch}/run-2.stdout"

awk '
  BEGIN { number = "^-?[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$" }
  $0 == "MP4 AOT STORAGE f16=512 bf16=512 f32=1024 ratio=2:1 PASS" { ++storage; next }
  $0 == "MP4 AOT RNE f16=8/8 bf16=8/8 PASS" { ++rne; next }
  $0 == "MP4 AOT NEGATIVE PASS" { ++negative; next }
  $1 == "MP4" && $2 == "AOT" && $3 == "LOSS" {
    mode = $4
    if ((mode != "F16" && mode != "BF16") || NF != 6 ||
        $5 !~ /^[0-9]+$/ || $5 != steps[mode] || $6 !~ number ||
        $6 + 0 < 0 || (steps[mode] && $6 + 0 >= prev[mode]))
      exit 1
    if (!steps[mode]) initial[mode] = $6
    final[mode] = $6
    prev[mode] = $6 + 0
    ++steps[mode]
    next
  }
  $1 == "MP4" && $2 == "AOT" && $3 == "OVERFLOW" {
    mode = $4
    if ((mode != "F16" && mode != "BF16") || NF != 9 ||
        $5 !~ /^scale_before=64(\.0+)?$/ || $6 !~ /^scale_after=32(\.0+)?$/ ||
        $7 != "overflows=1" || $8 != "grads_zeroed=8" || $9 != "PASS")
      exit 1
    ++overflow[mode]
    next
  }
  $1 == "MP4" && $2 == "AOT" && $3 == "RESULT" {
    mode = $4
    if ((mode != "F16" && mode != "BF16") || steps[mode] != 13 || NF != 9 ||
        $5 != "steps=12" || $6 != "overflows=1" ||
        $7 !~ /^final_scale=256(\.0+)?$/ ||
        $8 != "initial=" initial[mode] || $9 != "final=" final[mode])
      exit 1
    ++result[mode]
    next
  }
  { exit 1 }
  END {
    if (storage != 1 || rne != 1 || negative != 1 ||
        overflow["F16"] != 1 || overflow["BF16"] != 1 ||
        result["F16"] != 1 || result["BF16"] != 1 ||
        steps["F16"] != 13 || steps["BF16"] != 13)
      exit 1
  }
' "${scratch}/run-1.stdout" || die "invalid MP4 AOT trace"

elf_sha256="$(sha256sum "${scratch}/build-1/mp4-aot-test" | awk '{print $1}')"
cat "${scratch}/run-1.stdout"
printf 'MP4 AOT compiled ELF SHA-256: %s\n' "${elf_sha256}"
printf 'MP4 AOT two fresh compilations, byte-identical ELF and stdout: PASS\n'
