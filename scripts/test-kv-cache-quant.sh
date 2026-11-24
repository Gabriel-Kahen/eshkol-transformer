#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

cd "${PROJECT_ROOT}"
verify_toolchain
export LC_ALL=C

export CC="${CC:-$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cc_path)}"
export CXX="${CXX:-$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cxx_path)}"
runner="$(eshkol_build_dir)/eshkol-run"

kvq_dir="${PROJECT_ROOT}/build/kv_cache_quant"
bash scripts/build-kv-cache-quant.sh "${kvq_dir}"

scratch="$(mktemp -d "${kvq_dir}/.test.XXXXXX")"
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
  "${kvq_dir}/libeshkol_transformer_kv_cache_quant_aot.a"
  "${kvq_dir}/libeshkol_transformer_kv_cache_quant.a"
)

# 1. C++17 header compile and linkage check.
"${CXX}" "${cppflags[@]}" -O2 tests/kv_cache_quant/header_kvq_cpp.cpp \
  "${archives[@]}" -lm -o "${scratch}/header_kvq_cpp"
"${scratch}/header_kvq_cpp"
printf 'KVQ C++17 header and linkage: PASS\n'

# 2. Native C11 (-O2, hot-path zero-alloc wrapped) tests + Python oracle.
"${CC}" "${cflags[@]}" -O2 tests/kv_cache_quant/test_kv_cache_quant_native.c \
  "${archives[@]}" "${wrap[@]}" -lm -o "${scratch}/test_kv_cache_quant_native"
"${scratch}/test_kv_cache_quant_native"
python3 tests/kv_cache_quant/test_kv_cache_quant_oracle.py \
  "${scratch}/test_kv_cache_quant_native"

# 3. Native C11 (zero-alloc wrapped) + C++17 under ASan/UBSan/LSan.
export ASAN_OPTIONS=detect_leaks=1:halt_on_error=1
export UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1

"${CC}" "${cflags[@]}" "${san[@]}" -c native/kv_cache_quant.c \
  -o "${scratch}/kv_cache_quant.san.o"
"${CC}" "${cflags[@]}" "${san[@]}" -c native/kv_cache_quant_eshkol_bridge.c \
  -o "${scratch}/kv_cache_quant_eshkol_bridge.san.o"

"${CC}" "${cflags[@]}" "${san[@]}" \
  tests/kv_cache_quant/test_kv_cache_quant_native.c \
  "${scratch}/kv_cache_quant_eshkol_bridge.san.o" \
  "${scratch}/kv_cache_quant.san.o" \
  "${wrap[@]}" -lm -o "${scratch}/test_kv_cache_quant_native.san"
"${scratch}/test_kv_cache_quant_native.san"

"${CXX}" "${cppflags[@]}" "${san[@]}" tests/kv_cache_quant/header_kvq_cpp.cpp \
  "${scratch}/kv_cache_quant_eshkol_bridge.san.o" \
  "${scratch}/kv_cache_quant.san.o" \
  -lm -o "${scratch}/header_kvq_cpp.san"
"${scratch}/header_kvq_cpp.san"
printf 'KVQ ASan/UBSan/LSan tests: PASS\n'

# 4. Compile tests/kv_cache_quant/test_kv_cache_quant_aot.esk TWICE from fresh
#    dirs and verify byte-identical ELFs and byte-identical stdout.
for run in 1 2; do
  mkdir -p "${scratch}/cache-${run}" "${scratch}/out-${run}" \
           "${scratch}/libs-${run}" "${scratch}/snap-${run}"
  bash scripts/build-kv-cache-quant.sh "${scratch}/libs-${run}" >/dev/null
  diagnostic="${scratch}/compile-${run}.log"
  timeout --foreground --kill-after=5s 90s \
    env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR -u ESHKOL_AOT_MODULE_CACHE_DIR \
    ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="${scratch}/cache-${run}" \
    "${runner}" --strict-types --optimize 0 --no-stdlib \
      -L "${scratch}/libs-${run}" \
      --lib eshkol_transformer_kv_cache_quant_aot \
      --lib eshkol_transformer_kv_cache_quant \
      tests/kv_cache_quant/test_kv_cache_quant_aot.esk \
      -o "${scratch}/out-${run}/test_kv_cache_quant_aot" >"${diagnostic}" 2>&1

  diagnostic="${scratch}/run-${run}.stderr"
  timeout --foreground --kill-after=5s 60s \
    env ET_KVQ_SNAPSHOT_DIR="${scratch}/snap-${run}" \
    "${scratch}/out-${run}/test_kv_cache_quant_aot" \
    >"${scratch}/run-${run}.stdout" 2>"${diagnostic}" || {
      cat "${scratch}/run-${run}.stdout" >&2
      die "compiled KVQ AOT executable failed on run ${run}"
    }
  [[ ! -s "${diagnostic}" ]] || die "compiled KVQ AOT emitted stderr"
done
diagnostic=''

readelf -h "${scratch}/out-1/test_kv_cache_quant_aot" >/dev/null
cmp "${scratch}/out-1/test_kv_cache_quant_aot" \
    "${scratch}/out-2/test_kv_cache_quant_aot"
cmp "${scratch}/run-1.stdout" "${scratch}/run-2.stdout"

awk '
  $1 == "KVQ" && $2 == "AOT" && $3 == "COMPRESSION" {
    if (NF != 8 || $4 != "fp32_bytes=65536" || $5 != "int8_bytes=17408" ||
        $6 != "fp8_bytes=17408" || $7 !~ /^ratio=3\.7647/ || $8 != "status=PASS")
      exit 1
    ++comp_ok; next
  }
  $1 == "KVQ" && $2 == "AOT" && $3 == "DECODE_PARITY" {
    if (NF != 11 || $4 != "int8_deq_bits=0" || $5 != "fp8_deq_bits=0" ||
        $6 !~ /^int8_rmse=/ || $7 !~ /^fp8_rmse=/ ||
        $8 !~ /^int8_outlier_rmse=/ || $9 !~ /^fp8_outlier_rmse=/ ||
        $10 != "top1_steps=4" || $11 != "status=PASS")
      exit 1
    ++dec_ok; next
  }
  $0 == "KVQ AOT TRUNCATE_WINDOW trunc_from=32 trunc_to=20 reappend_bits=0 window=8 window_bits=0 status=PASS" {
    ++tw_ok; next
  }
  $0 == "KVQ AOT SNAPSHOT int8_bits=0 fp8_bits=0 mismatch=REJECT tamper=REJECT status=PASS" {
    ++snap_ok; next
  }
  $0 == "KVQ AOT NEGATIVE status=PASS" {
    ++neg_ok; next
  }
  { exit 1 }
  END {
    if (comp_ok != 1 || dec_ok != 1 || tw_ok != 1 || snap_ok != 1 || neg_ok != 1)
      exit 1
  }
' "${scratch}/run-1.stdout" || {
  cat "${scratch}/run-1.stdout" >&2
  die "invalid compiled KVQ AOT output"
}

cat "${scratch}/run-1.stdout"
elf_sha="$(sha256sum "${scratch}/out-1/test_kv_cache_quant_aot")"
printf 'KVQ AOT ELF SHA-256 %s\n' "${elf_sha%% *}"
printf 'KVQ AOT two fresh compilations, byte-identical ELF and stdout: PASS\n'
