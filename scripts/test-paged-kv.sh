#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

cd "${PROJECT_ROOT}"
verify_toolchain
export LC_ALL=C

export CC="${CC:-$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cc_path)}"
export CXX="${CXX:-$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cxx_path)}"
runner="$(eshkol_build_dir)/eshkol-run"

kv_dir="${PROJECT_ROOT}/build/paged_kv"
bash scripts/build-paged-kv.sh "${kv_dir}"

scratch="$(mktemp -d "${kv_dir}/.test.XXXXXX")"
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
  "${kv_dir}/libeshkol_transformer_paged_kv_aot.a"
  "${kv_dir}/libeshkol_transformer_paged_kv.a"
)

# 1. C++17 header compile and linkage check.
"${CXX}" "${cppflags[@]}" tests/paged_kv/header_paged_kv_cpp.cpp "${archives[@]}" -lm \
  -o "${scratch}/header_paged_kv_cpp"
"${scratch}/header_paged_kv_cpp"
printf 'PAGED_KV C++17 header and linkage: PASS\n'

# 2. Native C11 normal (-O2) tests + Python double-precision oracle.
"${CC}" "${cflags[@]}" -O2 tests/paged_kv/test_paged_kv_native.c "${archives[@]}" -lm \
  -o "${scratch}/test_paged_kv_native"
"${scratch}/test_paged_kv_native"
python3 tests/paged_kv/test_paged_kv_oracle.py "${scratch}/test_paged_kv_native"

# 3. Native C11 and C++17 tests under ASan/UBSan/LSan.
export ASAN_OPTIONS=detect_leaks=1:halt_on_error=1
export UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1
"${CC}" "${cflags[@]}" "${san[@]}" tests/paged_kv/test_paged_kv_native.c \
  native/paged_kv_eshkol_bridge.c native/paged_kv.c -lm \
  -o "${scratch}/test_paged_kv_native.san"
"${scratch}/test_paged_kv_native.san"

"${CC}" "${cflags[@]}" "${san[@]}" -c native/paged_kv.c \
  -o "${scratch}/paged_kv.san.o"
"${CC}" "${cflags[@]}" "${san[@]}" -c native/paged_kv_eshkol_bridge.c \
  -o "${scratch}/paged_kv_eshkol_bridge.san.o"
"${CXX}" "${cppflags[@]}" "${san[@]}" tests/paged_kv/header_paged_kv_cpp.cpp \
  "${scratch}/paged_kv_eshkol_bridge.san.o" "${scratch}/paged_kv.san.o" -lm \
  -o "${scratch}/header_paged_kv_cpp.san"
"${scratch}/header_paged_kv_cpp.san"
printf 'PAGED_KV ASan/UBSan/LSan tests: PASS\n'

# 4. Compile tests/paged_kv/test_paged_kv_aot.esk TWICE from fresh dirs and
#    verify byte-identical ELFs and byte-identical stdout.
for run in 1 2; do
  mkdir -p "${scratch}/cache-${run}" "${scratch}/out-${run}" \
           "${scratch}/libs-${run}"
  bash scripts/build-paged-kv.sh "${scratch}/libs-${run}" >/dev/null
  diagnostic="${scratch}/compile-${run}.log"
  timeout --foreground --kill-after=5s 90s \
    env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR -u ESHKOL_AOT_MODULE_CACHE_DIR \
    ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="${scratch}/cache-${run}" \
    "${runner}" --strict-types --optimize 0 --no-stdlib \
      -L "${scratch}/libs-${run}" \
      --lib eshkol_transformer_paged_kv_aot --lib eshkol_transformer_paged_kv \
      tests/paged_kv/test_paged_kv_aot.esk \
      -o "${scratch}/out-${run}/test_paged_kv_aot" >"${diagnostic}" 2>&1

  diagnostic="${scratch}/run-${run}.stderr"
  timeout --foreground --kill-after=5s 60s \
    "${scratch}/out-${run}/test_paged_kv_aot" \
    >"${scratch}/run-${run}.stdout" 2>"${diagnostic}" || {
      cat "${scratch}/run-${run}.stdout" >&2
      die "compiled PAGED_KV AOT binary failed on run ${run}"
    }
  [[ ! -s "${diagnostic}" ]] || die "compiled PAGED_KV AOT emitted stderr"
done
diagnostic=''

readelf -h "${scratch}/out-1/test_paged_kv_aot" >/dev/null
cmp "${scratch}/out-1/test_paged_kv_aot" "${scratch}/out-2/test_paged_kv_aot"
cmp "${scratch}/run-1.stdout" "${scratch}/run-2.stdout"

awk '
  $0 == "PAGED_KV AOT COW forks=3 cow_copies=3 shared_blocks=1 prefix_hits=1" {
    ++cow_seen; next
  }
  $1 == "PAGED_KV" && $2 == "AOT" && $3 == "GQA_PARITY" {
    if (NF != 6 || $4 != "checks=8" || $6 != "status=PASS") exit 1
    if ($5 !~ /^max_abs_err=[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$/) exit 1
    ++gqa_seen; next
  }
  $0 == "PAGED_KV AOT TRUNCATE from=13 to=9 freed_blocks=1 bitwise_restore=PASS" {
    ++trunc_seen; next
  }
  $0 == "PAGED_KV AOT OOM_AND_NEGATIVE status=PASS" {
    ++oom_seen; next
  }
  $0 == "PAGED_KV AOT CLEANUP free_blocks=10 total_blocks=10 status=PASS" {
    ++clean_seen; next
  }
  { exit 1 }
  END {
    if (cow_seen != 1 || gqa_seen != 1 || trunc_seen != 1 ||
        oom_seen != 1 || clean_seen != 1)
      exit 1
  }
' "${scratch}/run-1.stdout" || {
  cat "${scratch}/run-1.stdout" >&2
  die "invalid compiled PAGED_KV AOT output"
}

cat "${scratch}/run-1.stdout"
elf_sha="$(sha256sum "${scratch}/out-1/test_paged_kv_aot")"
printf 'PAGED_KV AOT ELF SHA-256 %s\n' "${elf_sha%% *}"
printf 'PAGED_KV AOT two fresh compilations, byte-identical ELF and stdout: PASS\n'
