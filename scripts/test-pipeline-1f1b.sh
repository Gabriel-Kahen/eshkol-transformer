#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

cd "${PROJECT_ROOT}"
verify_toolchain
export LC_ALL=C

export CC="${CC:-$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cc_path)}"
export CXX="${CXX:-$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cxx_path)}"
runner="$(eshkol_build_dir)/eshkol-run"

pp_dir="${PROJECT_ROOT}/build/pipeline_1f1b"
bash scripts/build-pipeline-1f1b.sh "${pp_dir}"

scratch="$(mktemp -d "${pp_dir}/.test.XXXXXX")"
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

archives=(
  "${pp_dir}/libeshkol_transformer_pipeline_1f1b_aot.a"
  "${pp_dir}/libeshkol_transformer_pipeline_1f1b.a"
)

# 1. C++17 header compile and linkage check.
"${CXX}" "${cppflags[@]}" -O2 tests/pipeline_1f1b/header_pp_cpp.cpp \
  "${archives[@]}" -lm -o "${scratch}/header_pp_cpp"
"${scratch}/header_pp_cpp"
printf 'PIPELINE_1F1B C++17 header and linkage: PASS\n'

# 2. Native C11 normal (-O2) tests + independent Python binary64 oracle.
"${CC}" "${cflags[@]}" -O2 tests/pipeline_1f1b/test_pipeline_1f1b_native.c \
  "${archives[@]}" -lm -o "${scratch}/test_pipeline_1f1b_native"
"${scratch}/test_pipeline_1f1b_native"
python3 tests/pipeline_1f1b/test_pipeline_1f1b_oracle.py \
  "${scratch}/test_pipeline_1f1b_native"

# 3. Native C11 and C++17 tests under ASan/UBSan/LSan (detect_leaks=1:halt_on_error=1).
export ASAN_OPTIONS=detect_leaks=1:halt_on_error=1
export UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1
"${CC}" "${cflags[@]}" "${san[@]}" -c native/pipeline_1f1b.c \
  -o "${scratch}/pipeline_1f1b.san.o"
"${CC}" "${cflags[@]}" "${san[@]}" -c native/pipeline_1f1b_eshkol_bridge.c \
  -o "${scratch}/pipeline_1f1b_eshkol_bridge.san.o"
san_objs=(
  "${scratch}/pipeline_1f1b_eshkol_bridge.san.o"
  "${scratch}/pipeline_1f1b.san.o"
)

"${CC}" "${cflags[@]}" "${san[@]}" tests/pipeline_1f1b/test_pipeline_1f1b_native.c \
  "${san_objs[@]}" -lm -o "${scratch}/test_pipeline_1f1b_native.san"
"${scratch}/test_pipeline_1f1b_native.san"

"${CXX}" "${cppflags[@]}" "${san[@]}" tests/pipeline_1f1b/header_pp_cpp.cpp \
  "${san_objs[@]}" -lm -o "${scratch}/header_pp_cpp.san"
"${scratch}/header_pp_cpp.san"
printf 'PIPELINE_1F1B ASan/UBSan/LSan tests: PASS\n'

# 4. Compile tests/pipeline_1f1b/test_pipeline_1f1b_aot.esk TWICE from fresh
#    dirs and verify byte-identical ELFs and byte-identical stdout.
for run in 1 2; do
  mkdir -p "${scratch}/cache-${run}" "${scratch}/out-${run}" \
           "${scratch}/libs-${run}"
  bash scripts/build-pipeline-1f1b.sh "${scratch}/libs-${run}" >/dev/null
  diagnostic="${scratch}/compile-${run}.log"
  timeout --foreground --kill-after=5s 90s \
    env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR -u ESHKOL_AOT_MODULE_CACHE_DIR \
    ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="${scratch}/cache-${run}" \
    "${runner}" --strict-types --optimize 0 --no-stdlib \
      -L "${scratch}/libs-${run}" \
      --lib eshkol_transformer_pipeline_1f1b_aot \
      --lib eshkol_transformer_pipeline_1f1b \
      tests/pipeline_1f1b/test_pipeline_1f1b_aot.esk \
      -o "${scratch}/out-${run}/test_pipeline_1f1b_aot" >"${diagnostic}" 2>&1

  diagnostic="${scratch}/run-${run}.stderr"
  timeout --foreground --kill-after=5s 60s \
    "${scratch}/out-${run}/test_pipeline_1f1b_aot" \
    >"${scratch}/run-${run}.stdout" 2>"${diagnostic}" || {
      cat "${scratch}/run-${run}.stdout" >&2
      die "compiled pipeline_1f1b AOT executable failed on run ${run}"
    }
  [[ ! -s "${diagnostic}" ]] || die "compiled pipeline_1f1b AOT emitted stderr"
done
diagnostic=''

readelf -h "${scratch}/out-1/test_pipeline_1f1b_aot" >/dev/null
cmp "${scratch}/out-1/test_pipeline_1f1b_aot" \
    "${scratch}/out-2/test_pipeline_1f1b_aot"
cmp "${scratch}/run-1.stdout" "${scratch}/run-2.stdout"

awk '
  $0 == "PP1F1B AOT SCHEDULE P=4 M=8 1f1b_peaks=4,3,2,1 gpipe_peaks=8,8,8,8 bubble=3/11 PASS" {
    ++sched_ok; next
  }
  $1 == "PP1F1B" && $2 == "AOT" && $3 == "FD_GRADIENTS" {
    if (NF != 7 || $4 != "checked=84" || $5 !~ /^max_dx_err=/ ||
        $6 !~ /^max_dp_err=/ || $7 != "PASS")
      exit 1
    ++fd_ok; next
  }
  $1 == "PP1F1B" && $2 == "AOT" && $3 == "PARITY" {
    if (NF != 13 || $4 != "M=8" || $5 !~ /^p2_gerr=/ || $6 !~ /^p4_gerr=/ ||
        $7 !~ /^p4_mse_gerr=/ || $8 != "pids_p2=2" || $9 != "pids_p4=4" ||
        $10 != "peak_1f1b_s0=4" || $11 != "peak_1f1b_s3=1" ||
        $12 != "peak_gpipe_s0=8" || $13 != "PASS")
      exit 1
    ++parity_ok; next
  }
  $1 == "PP1F1B" && $2 == "AOT" && $3 == "SGD10" {
    if (NF != 9 || $4 !~ /^loss0=/ || $5 !~ /^loss9=/ ||
        $6 !~ /^max_lerr=/ || $7 !~ /^w2_err=/ || $8 !~ /^w4_err=/ ||
        $9 != "PASS")
      exit 1
    ++sgd_ok; next
  }
  $0 == "PP1F1B AOT NEGATIVE checks=9 PASS" { ++neg_ok; next }
  $0 == "PP1F1B AOT SUMMARY stages=2,4 microbatches=8 sgd_steps=10 PASS" {
    ++sum_ok; next
  }
  { exit 1 }
  END {
    if (sched_ok != 1 || fd_ok != 1 || parity_ok != 1 || sgd_ok != 1 ||
        neg_ok != 1 || sum_ok != 1)
      exit 1
  }
' "${scratch}/run-1.stdout" || {
  cat "${scratch}/run-1.stdout" >&2
  die "invalid compiled pipeline_1f1b AOT output"
}

cat "${scratch}/run-1.stdout"
elf_sha="$(sha256sum "${scratch}/out-1/test_pipeline_1f1b_aot")"
printf 'PIPELINE_1F1B AOT ELF SHA-256 %s\n' "${elf_sha%% *}"
printf 'PIPELINE_1F1B AOT two fresh compilations, byte-identical ELF and stdout: PASS\n'
