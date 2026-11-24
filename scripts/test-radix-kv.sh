#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

cd "${PROJECT_ROOT}"
verify_toolchain
export LC_ALL=C

export CC="${CC:-$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cc_path)}"
export CXX="${CXX:-$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cxx_path)}"
runner="$(eshkol_build_dir)/eshkol-run"

rx_dir="${PROJECT_ROOT}/build/radix_kv"
bash scripts/build-radix-kv.sh "${rx_dir}"

scratch="$(mktemp -d "${rx_dir}/.test.XXXXXX")"
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
san=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)

archives=(
  "${rx_dir}/libeshkol_transformer_radix_kv_aot.a"
  "${rx_dir}/libeshkol_transformer_radix_kv.a"
)

# 1. C++17 header compile and linkage check.
"${CXX}" "${cppflags[@]}" tests/radix_kv/header_radix_cpp.cpp "${archives[@]}" -lm \
  -o "${scratch}/header_radix_cpp"
"${scratch}/header_radix_cpp"
printf 'RADIX_KV C++17 header and linkage: PASS\n'

# 2. Native C11 normal (-O2) tests + Python reference oracle.
"${CC}" "${cflags[@]}" -O2 tests/radix_kv/test_radix_kv_native.c "${archives[@]}" -lm \
  -o "${scratch}/test_radix_kv_native"
"${scratch}/test_radix_kv_native"
python3 tests/radix_kv/test_radix_kv_oracle.py "${scratch}/test_radix_kv_native"

# 3. Native C11 and C++17 tests under ASan/UBSan/LSan.
export ASAN_OPTIONS=detect_leaks=1:halt_on_error=1
export UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1
"${CC}" "${cflags[@]}" "${san[@]}" tests/radix_kv/test_radix_kv_native.c \
  native/radix_kv_eshkol_bridge.c native/radix_kv.c -lm \
  -o "${scratch}/test_radix_kv_native.san"
"${scratch}/test_radix_kv_native.san"

"${CC}" "${cflags[@]}" "${san[@]}" -c native/radix_kv.c \
  -o "${scratch}/radix_kv.san.o"
"${CC}" "${cflags[@]}" "${san[@]}" -c native/radix_kv_eshkol_bridge.c \
  -o "${scratch}/radix_kv_eshkol_bridge.san.o"
"${CXX}" "${cppflags[@]}" "${san[@]}" tests/radix_kv/header_radix_cpp.cpp \
  "${scratch}/radix_kv_eshkol_bridge.san.o" "${scratch}/radix_kv.san.o" -lm \
  -o "${scratch}/header_radix_cpp.san"
"${scratch}/header_radix_cpp.san"
printf 'RADIX_KV ASan/UBSan/LSan tests: PASS\n'

# 4. Compile tests/radix_kv/test_radix_kv_aot.esk TWICE from fresh dirs and
#    verify byte-identical ELFs and byte-identical stdout.
for run in 1 2; do
  mkdir -p "${scratch}/cache-${run}" "${scratch}/out-${run}" \
           "${scratch}/libs-${run}"
  bash scripts/build-radix-kv.sh "${scratch}/libs-${run}" >/dev/null
  diagnostic="${scratch}/compile-${run}.log"
  timeout --foreground --kill-after=5s 90s \
    env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR -u ESHKOL_AOT_MODULE_CACHE_DIR \
    ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="${scratch}/cache-${run}" \
    "${runner}" --strict-types --optimize 0 --no-stdlib \
      -L "${scratch}/libs-${run}" \
      --lib eshkol_transformer_radix_kv_aot --lib eshkol_transformer_radix_kv \
      tests/radix_kv/test_radix_kv_aot.esk \
      -o "${scratch}/out-${run}/test_radix_kv_aot" >"${diagnostic}" 2>&1

  diagnostic="${scratch}/run-${run}.stderr"
  timeout --foreground --kill-after=5s 60s \
    env ET_RADIX_KV_AOT_SNAPSHOT_PATH="${scratch}/snap-${run}.etrx" \
    "${scratch}/out-${run}/test_radix_kv_aot" \
    >"${scratch}/run-${run}.stdout" 2>"${diagnostic}" || {
      cat "${scratch}/run-${run}.stdout" >&2
      die "compiled RADIX_KV AOT binary failed on run ${run}"
    }
  [[ ! -s "${diagnostic}" ]] || die "compiled RADIX_KV AOT emitted stderr"
done
diagnostic=''

readelf -h "${scratch}/out-1/test_radix_kv_aot" >/dev/null
cmp "${scratch}/out-1/test_radix_kv_aot" "${scratch}/out-2/test_radix_kv_aot"
cmp "${scratch}/run-1.stdout" "${scratch}/run-2.stdout"

awk '
  $0 == "RADIX_KV AOT PREFIX_REUSE requests=4 logical=96 physical=40 reused=56 splits=3 status=PASS" {
    ++reuse_seen; next
  }
  $0 == "RADIX_KV AOT GQA_PARITY prefix=20 suffix=4 max_diff=0 bitwise=PASS" ||
  $0 == "RADIX_KV AOT GQA_PARITY prefix=20 suffix=4 max_diff=0.0 bitwise=PASS" {
    ++gqa_seen; next
  }
  $0 == "RADIX_KV AOT PIN_LRU_EVICT pinned_survived=1 evicted_tokens=12 used=42 status=PASS" {
    ++evict_seen; next
  }
  $0 == "RADIX_KV AOT OOM_ROLLBACK oom_rejections=1 invariants=PASS" {
    ++oom_seen; next
  }
  $0 == "RADIX_KV AOT SNAPSHOT format=ETRX1000 reload_hit=24 tamper_reject=PASS" {
    ++snap_seen; next
  }
  $0 == "RADIX_KV AOT NEGATIVE status=PASS" {
    ++neg_seen; next
  }
  { exit 1 }
  END {
    if (reuse_seen != 1 || gqa_seen != 1 || evict_seen != 1 ||
        oom_seen != 1 || snap_seen != 1 || neg_seen != 1)
      exit 1
  }
' "${scratch}/run-1.stdout" || {
  cat "${scratch}/run-1.stdout" >&2
  die "invalid compiled RADIX_KV AOT output"
}

cat "${scratch}/run-1.stdout"
elf_sha="$(sha256sum "${scratch}/out-1/test_radix_kv_aot")"
printf 'RADIX_KV AOT ELF SHA-256 %s\n' "${elf_sha%% *}"
printf 'RADIX_KV AOT two fresh compilations, byte-identical ELF and stdout: PASS\n'
