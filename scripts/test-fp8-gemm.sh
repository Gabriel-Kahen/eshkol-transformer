#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

cd "${PROJECT_ROOT}"
verify_toolchain
export LC_ALL=C

export CC="${CC:-$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cc_path)}"
export CXX="${CXX:-$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cxx_path)}"
runner="$(eshkol_build_dir)/eshkol-run"

fp8_dir="${PROJECT_ROOT}/build/fp8_gemm"
bash scripts/build-fp8-gemm.sh "${fp8_dir}"

scratch="$(mktemp -d "${fp8_dir}/.test.XXXXXX")"
diagnostic=''
cleanup() {
  local status=$?
  if [[ "${status}" != 0 && -n "${diagnostic}" && -s "${diagnostic}" ]]; then
    cat "${diagnostic}" >&2
  fi
  rm -rf -- "${scratch}"
}
trap cleanup EXIT

common=(-Wall -Wextra -Werror -Wpedantic -ffp-contract=off -fno-fast-math
        -I include -I native)
cflags=(-std=c11 "${common[@]}" -fexcess-precision=standard)
cppflags=(-std=c++17 "${common[@]}")
san=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer
     -fno-sanitize-recover=all)
wrap=(-Wl,--wrap=malloc,--wrap=calloc,--wrap=realloc,--wrap=free)

archives=(
  "${fp8_dir}/libeshkol_transformer_fp8_gemm_aot.a"
  "${fp8_dir}/libeshkol_transformer_fp8_gemm.a"
)

# 1. C++17 header compile and linkage check.
"${CXX}" "${cppflags[@]}" -O2 tests/fp8_gemm/header_fp8_cpp.cpp \
  "${archives[@]}" -lm -o "${scratch}/header_fp8_cpp"
"${scratch}/header_fp8_cpp"
printf 'FP8 C++17 header and linkage: PASS\n'

# 2. Native C11 normal (-O2, zero-alloc wrapped) tests + Python exact oracle.
"${CC}" "${cflags[@]}" -O2 tests/fp8_gemm/test_fp8_native.c \
  "${archives[@]}" "${wrap[@]}" -lm -o "${scratch}/test_fp8_native"
"${scratch}/test_fp8_native"
python3 tests/fp8_gemm/test_fp8_oracle.py "${scratch}/test_fp8_native"

# 3. Native C11 (zero-alloc wrapped) + C++17 under ASan/UBSan/LSan.
export ASAN_OPTIONS=detect_leaks=1:halt_on_error=1
export UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1

"${CC}" "${cflags[@]}" "${san[@]}" -c native/fp8_gemm.c \
  -o "${scratch}/fp8_gemm.san.o"
"${CC}" "${cflags[@]}" "${san[@]}" -c native/fp8_gemm_eshkol_bridge.c \
  -o "${scratch}/fp8_gemm_eshkol_bridge.san.o"

"${CC}" "${cflags[@]}" "${san[@]}" tests/fp8_gemm/test_fp8_native.c \
  "${scratch}/fp8_gemm_eshkol_bridge.san.o" "${scratch}/fp8_gemm.san.o" \
  "${wrap[@]}" -lm -o "${scratch}/test_fp8_native.san"
"${scratch}/test_fp8_native.san"

"${CXX}" "${cppflags[@]}" "${san[@]}" tests/fp8_gemm/header_fp8_cpp.cpp \
  "${scratch}/fp8_gemm_eshkol_bridge.san.o" "${scratch}/fp8_gemm.san.o" \
  -lm -o "${scratch}/header_fp8_cpp.san"
"${scratch}/header_fp8_cpp.san"
printf 'FP8 ASan/UBSan/LSan tests: PASS\n'

# 4. Compile tests/fp8_gemm/test_fp8_gemm_aot.esk TWICE from fresh dirs and verify
#    byte-identical ELFs and stdout.
for run in 1 2; do
  mkdir -p "${scratch}/cache-${run}" "${scratch}/out-${run}" \
           "${scratch}/libs-${run}"
  bash scripts/build-fp8-gemm.sh "${scratch}/libs-${run}" >/dev/null
  diagnostic="${scratch}/compile-${run}.log"
  timeout --foreground --kill-after=5s 90s \
    env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR -u ESHKOL_AOT_MODULE_CACHE_DIR \
    ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="${scratch}/cache-${run}" \
    "${runner}" --strict-types --optimize 0 --no-stdlib \
      -L "${scratch}/libs-${run}" \
      --lib eshkol_transformer_fp8_gemm_aot \
      --lib eshkol_transformer_fp8_gemm \
      tests/fp8_gemm/test_fp8_gemm_aot.esk \
      -o "${scratch}/out-${run}/test_fp8_gemm_aot" >"${diagnostic}" 2>&1

  diagnostic="${scratch}/run-${run}.stderr"
  timeout --foreground --kill-after=5s 60s \
    "${scratch}/out-${run}/test_fp8_gemm_aot" \
    >"${scratch}/run-${run}.stdout" 2>"${diagnostic}" || {
      cat "${scratch}/run-${run}.stdout" >&2
      die "compiled FP8 AOT executable failed on run ${run}"
    }
  [[ ! -s "${diagnostic}" ]] || die "compiled FP8 AOT emitted stderr"
done
diagnostic=''

readelf -h "${scratch}/out-1/test_fp8_gemm_aot" >/dev/null
cmp "${scratch}/out-1/test_fp8_gemm_aot" "${scratch}/out-2/test_fp8_gemm_aot"
cmp "${scratch}/run-1.stdout" "${scratch}/run-2.stdout"

awk '
  $0 == "FP8 AOT CODEC e4m3_finite=254 e4m3_reject=2 e5m2_finite=248 e5m2_reject=8 PASS" {
    ++codec_ok; next
  }
  $1 == "FP8" && $2 == "AOT" && $3 == "DELAYED_SCALE" {
    if (NF != 9 || $4 != "steps=10" || $5 != "history=4" || $6 != "margin=1" ||
        $7 !~ /^final_amax=/ || $8 !~ /^final_scale=/ || $9 != "PASS")
      exit 1
    ++ds_ok; next
  }
  $1 == "FP8" && $2 == "AOT" && $3 == "GEMM" {
    if (NF != 9 || $4 !~ /^parity_max_err=/ || $5 !~ /^block_vs_f32=/ ||
        $6 !~ /^pertensor_vs_f32=/ || $7 !~ /^ste_fd_checked=/ ||
        $8 !~ /^ste_fd_max_err=/ || $9 != "PASS")
      exit 1
    ++gemm_ok; next
  }
  $0 == "FP8 AOT NEGATIVE PASS" { ++neg_ok; next }
  { exit 1 }
  END {
    if (codec_ok != 1 || ds_ok != 1 || gemm_ok != 1 || neg_ok != 1)
      exit 1
  }
' "${scratch}/run-1.stdout" || {
  cat "${scratch}/run-1.stdout" >&2
  die "invalid compiled FP8 AOT output"
}

cat "${scratch}/run-1.stdout"
elf_sha="$(sha256sum "${scratch}/out-1/test_fp8_gemm_aot")"
printf 'FP8 AOT ELF SHA-256 %s\n' "${elf_sha%% *}"
printf 'FP8 AOT two fresh compilations, byte-identical ELF and stdout: PASS\n'
