#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

cd "${PROJECT_ROOT}"
verify_toolchain
export LC_ALL=C

export CC="${CC:-$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cc_path)}"
export CXX="${CXX:-$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cxx_path)}"
runner="$(eshkol_build_dir)/eshkol-run"

cb_dir="${PROJECT_ROOT}/build/continuous_batch"
bash scripts/build-continuous-batch.sh "${cb_dir}"

scratch="$(mktemp -d "${cb_dir}/.test.XXXXXX")"
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
  "${cb_dir}/libeshkol_transformer_continuous_batch_aot.a"
  "${cb_dir}/libeshkol_transformer_continuous_batch.a"
)

# 1. C++17 header compile and linkage check.
"${CXX}" "${cppflags[@]}" -O2 tests/continuous_batch/header_cb_cpp.cpp \
  "${archives[@]}" -o "${scratch}/header_cb_cpp"
"${scratch}/header_cb_cpp"
printf 'CONTINUOUS_BATCH C++17 header and linkage: PASS\n'

# 2. Native C11 normal (-O2, allocation-trapped hot path) + Python oracle.
"${CC}" "${cflags[@]}" -O2 tests/continuous_batch/test_continuous_batch_native.c \
  "${archives[@]}" "${wrap[@]}" -o "${scratch}/test_continuous_batch_native"
"${scratch}/test_continuous_batch_native"
python3 tests/continuous_batch/test_continuous_batch_oracle.py \
  "${scratch}/test_continuous_batch_native"

# 3. Native C11 and C++17 tests under ASan/UBSan/LSan (detect_leaks=1:halt_on_error=1).
export ASAN_OPTIONS=detect_leaks=1:halt_on_error=1
export UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1
"${CC}" "${cflags[@]}" "${san[@]}" \
  tests/continuous_batch/test_continuous_batch_native.c \
  native/continuous_batch_eshkol_bridge.c native/continuous_batch.c \
  "${wrap[@]}" -o "${scratch}/test_continuous_batch_native.san"
"${scratch}/test_continuous_batch_native.san"

"${CC}" "${cflags[@]}" "${san[@]}" -c native/continuous_batch.c \
  -o "${scratch}/continuous_batch.san.o"
"${CC}" "${cflags[@]}" "${san[@]}" -c native/continuous_batch_eshkol_bridge.c \
  -o "${scratch}/continuous_batch_eshkol_bridge.san.o"
"${CXX}" "${cppflags[@]}" "${san[@]}" tests/continuous_batch/header_cb_cpp.cpp \
  "${scratch}/continuous_batch_eshkol_bridge.san.o" \
  "${scratch}/continuous_batch.san.o" \
  -o "${scratch}/header_cb_cpp.san"
"${scratch}/header_cb_cpp.san"
printf 'CONTINUOUS_BATCH ASan/UBSan/LSan tests: PASS\n'

# 4. Compile tests/continuous_batch/test_continuous_batch_aot.esk TWICE from
#    fresh build & cache dirs and verify byte-identical ELFs and stdout.
for run in 1 2; do
  mkdir -p "${scratch}/cache-${run}" "${scratch}/out-${run}" "${scratch}/libs-${run}"
  bash scripts/build-continuous-batch.sh "${scratch}/libs-${run}" >/dev/null
  diagnostic="${scratch}/compile-${run}.log"
  timeout --foreground --kill-after=5s 90s \
    env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR -u ESHKOL_AOT_MODULE_CACHE_DIR \
    ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="${scratch}/cache-${run}" \
    "${runner}" --strict-types --optimize 0 --no-stdlib \
      -L "${scratch}/libs-${run}" \
      --lib eshkol_transformer_continuous_batch_aot \
      --lib eshkol_transformer_continuous_batch \
      tests/continuous_batch/test_continuous_batch_aot.esk \
      -o "${scratch}/out-${run}/test_continuous_batch_aot" >"${diagnostic}" 2>&1

  diagnostic="${scratch}/run-${run}.stderr"
  timeout --foreground --kill-after=5s 60s \
    "${scratch}/out-${run}/test_continuous_batch_aot" \
    >"${scratch}/run-${run}.stdout" 2>"${diagnostic}" || {
      cat "${scratch}/run-${run}.stdout" >&2
      die "compiled continuous_batch AOT binary failed on run ${run}"
    }
  [[ ! -s "${diagnostic}" ]] || die "compiled continuous_batch AOT emitted stderr"
done
diagnostic=''

readelf -h "${scratch}/out-1/test_continuous_batch_aot" >/dev/null
cmp "${scratch}/out-1/test_continuous_batch_aot" \
    "${scratch}/out-2/test_continuous_batch_aot"
cmp "${scratch}/run-1.stdout" "${scratch}/run-2.stdout"

awk '
  $1 == "CB" && $2 == "AOT" && $3 == "STEP" {
    if (NF != 11 || $4 !~ /^[0-9]+$/ || $4 + 0 != steps) exit 1
    u = $10; f = $11
    sub(/^used_kv=/, "", u); sub(/^free_kv=/, "", f)
    if (u + f != 6) exit 1
    ++steps
    next
  }
  $1 == "CB" && $2 == "AOT" && $3 == "REQ" {
    if (NF != 10) exit 1
    ++reqs
    next
  }
  $1 == "CB" && $2 == "AOT" && $3 == "SUMMARY" {
    if (NF != 11 || $6 != "finished=4" || $7 != "cancelled=1" ||
        $8 != "preemptions=1" || $9 != "generated=11" || $11 != "free_kv=6")
      exit 1
    ++summaries
    next
  }
  $0 == "CB AOT NEGATIVE PASS" { ++neg; next }
  { exit 1 }
  END {
    if (steps != 8 || reqs != 5 || summaries != 2 || neg != 1) exit 1
  }
' "${scratch}/run-1.stdout" || {
  cat "${scratch}/run-1.stdout" >&2
  die "invalid compiled continuous_batch AOT output"
}

cat "${scratch}/run-1.stdout"
elf_sha="$(sha256sum "${scratch}/out-1/test_continuous_batch_aot")"
printf 'CONTINUOUS_BATCH AOT ELF SHA-256 %s\n' "${elf_sha%% *}"
printf 'CONTINUOUS_BATCH AOT two fresh compilations, byte-identical ELF and stdout: PASS\n'
