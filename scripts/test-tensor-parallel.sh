#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

cd "${PROJECT_ROOT}"
for command in ar cmp nm readelf sha256sum timeout; do
  require_command "${command}"
done
verify_toolchain
export LC_ALL=C

export CC="${CC:-$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cc_path)}"
export CXX="${CXX:-$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cxx_path)}"
runner="$(eshkol_build_dir)/eshkol-run"

tp_dir="${PROJECT_ROOT}/build/tensor_parallel"
bash scripts/build-tensor-parallel.sh "${tp_dir}"

scratch="$(mktemp -d "${tp_dir}/.test.XXXXXX")"
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
  "${tp_dir}/libeshkol_transformer_tensor_parallel_aot.a"
  "${tp_dir}/libeshkol_transformer_tensor_parallel.a"
)

# 1. C++17 header compile and linkage check.
"${CXX}" "${cppflags[@]}" tests/tensor_parallel/header_tp_cpp.cpp \
  "${archives[@]}" -lm -o "${scratch}/header_tp_cpp"
"${scratch}/header_tp_cpp"
printf 'TP C++17 header and linkage: PASS\n'

# 2. Native C11 normal (-O2) tests + independent Python binary64 oracle + f64 FD.
"${CC}" "${cflags[@]}" -O2 tests/tensor_parallel/test_tensor_parallel_native.c \
  "${archives[@]}" -lm -o "${scratch}/test_tensor_parallel_native"
"${scratch}/test_tensor_parallel_native"
python3 tests/tensor_parallel/test_tensor_parallel_oracle.py \
  "${scratch}/test_tensor_parallel_native"

# 3. Native C11 and C++17 tests under ASan/UBSan/LSan.
export ASAN_OPTIONS=detect_leaks=1:halt_on_error=1
export UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1
"${CC}" "${cflags[@]}" "${san[@]}" \
  tests/tensor_parallel/test_tensor_parallel_native.c \
  native/tensor_parallel_eshkol_bridge.c native/tensor_parallel.c -lm \
  -o "${scratch}/test_tensor_parallel_native.san"
"${scratch}/test_tensor_parallel_native.san"

"${CC}" "${cflags[@]}" "${san[@]}" -c native/tensor_parallel.c \
  -o "${scratch}/tensor_parallel.san.o"
"${CC}" "${cflags[@]}" "${san[@]}" -c native/tensor_parallel_eshkol_bridge.c \
  -o "${scratch}/tensor_parallel_eshkol_bridge.san.o"
"${CXX}" "${cppflags[@]}" "${san[@]}" tests/tensor_parallel/header_tp_cpp.cpp \
  "${scratch}/tensor_parallel_eshkol_bridge.san.o" \
  "${scratch}/tensor_parallel.san.o" -lm \
  -o "${scratch}/header_tp_cpp.san"
"${scratch}/header_tp_cpp.san"
printf 'TP ASan/UBSan/LSan tests: PASS\n'

# 4. Compile tests/tensor_parallel/test_tensor_parallel_aot.esk TWICE from fresh
#    dirs and verify byte-identical ELFs and byte-identical stdout.
for run in 1 2; do
  mkdir -p "${scratch}/cache-${run}" "${scratch}/out-${run}" \
           "${scratch}/libs-${run}"
  bash scripts/build-tensor-parallel.sh "${scratch}/libs-${run}" >/dev/null
  diagnostic="${scratch}/compile-${run}.log"
  timeout --foreground --kill-after=5s 90s \
    env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR -u ESHKOL_AOT_MODULE_CACHE_DIR \
    ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="${scratch}/cache-${run}" \
    "${runner}" --strict-types --optimize 0 --no-stdlib \
      -L "${scratch}/libs-${run}" \
      --lib eshkol_transformer_tensor_parallel_aot \
      --lib eshkol_transformer_tensor_parallel \
      tests/tensor_parallel/test_tensor_parallel_aot.esk \
      -o "${scratch}/out-${run}/test_tensor_parallel_aot" >"${diagnostic}" 2>&1

  diagnostic="${scratch}/run-${run}.stderr"
  timeout --foreground --kill-after=5s 60s \
    "${scratch}/out-${run}/test_tensor_parallel_aot" \
    >"${scratch}/run-${run}.stdout" 2>"${diagnostic}" || {
      cat "${scratch}/run-${run}.stdout" >&2
      die "compiled TP AOT binary failed on run ${run}"
    }
  [[ ! -s "${diagnostic}" ]] || die "compiled TP AOT binary emitted stderr"
done
diagnostic=''

readelf -h "${scratch}/out-1/test_tensor_parallel_aot" >/dev/null
cmp "${scratch}/out-1/test_tensor_parallel_aot" \
    "${scratch}/out-2/test_tensor_parallel_aot"
cmp "${scratch}/run-1.stdout" "${scratch}/run-2.stdout"

awk '
  BEGIN { num = "^-?[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$" }
  $0 == "TP AOT MEMORY full_sharded=24576 w1=24576 w2=12288 w4=6144 replicated=256 PASS" {
    ++mem_seen; next
  }
  $1 == "TP" && $2 == "AOT" && $3 == "STEP" {
    if (NF != 7 || $4 !~ /^[0-9]+$/ || $4 + 0 != step) exit 1
    split($5, ls, "="); split($6, w2, "="); split($7, w4, "=")
    if (ls[1] != "loss" || ls[2] !~ num || ls[2] + 0 < 0) exit 1
    if (w2[1] != "w2_out_err" || w2[2] !~ num || w2[2] + 0 > 1e-6) exit 1
    if (w4[1] != "w4_out_err" || w4[2] !~ num || w4[2] + 0 > 1e-6) exit 1
    if (step > 0 && ls[2] + 0 >= prev_loss) exit 1
    if (step == 0) init_l = ls[2]
    final_l = ls[2]
    prev_loss = ls[2] + 0
    ++step
    next
  }
  $1 == "TP" && $2 == "AOT" && $3 == "TRAIN" {
    if (NF != 13 || $4 != "steps=10" || $5 != "init_loss=" init_l ||
        $6 != "final_loss=" final_l || $11 != "ipc_ar=40" ||
        $12 != "ipc_ag=21" || $13 != "PASS")
      exit 1
    for (k = 7; k <= 10; ++k) {
      split($k, kv, "=")
      if (kv[2] !~ num || kv[2] + 0 > 1e-6) exit 1
    }
    ++train_seen
    next
  }
  $1 == "TP" && $2 == "AOT" && $3 == "FD" {
    if (NF != 6 || $4 != "checks=10" || $6 != "PASS") exit 1
    split($5, me, "=")
    if (me[1] != "max_err" || me[2] !~ num || me[2] + 0 > 5e-4) exit 1
    ++fd_seen
    next
  }
  $0 == "TP AOT NEGATIVE PASS checks=12" { ++neg_seen; next }
  { exit 1 }
  END {
    if (mem_seen != 1 || step != 10 || train_seen != 1 || fd_seen != 1 || neg_seen != 1)
      exit 1
  }
' "${scratch}/run-1.stdout" || {
  cat "${scratch}/run-1.stdout" >&2
  die "invalid compiled TP AOT output"
}

cat "${scratch}/run-1.stdout"
elf_sha="$(sha256sum "${scratch}/out-1/test_tensor_parallel_aot")"
printf 'TP AOT ELF SHA-256 %s\n' "${elf_sha%% *}"
printf 'TP AOT two fresh compilations, byte-identical ELF and stdout: PASS\n'
