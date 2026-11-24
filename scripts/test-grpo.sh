#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

cd "${PROJECT_ROOT}"
verify_toolchain
export LC_ALL=C

export CC="${CC:-$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cc_path)}"
export CXX="${CXX:-$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cxx_path)}"
runner="$(eshkol_build_dir)/eshkol-run"

grpo_dir="${PROJECT_ROOT}/build/grpo"
bash scripts/build-grpo.sh "${grpo_dir}"

scratch="$(mktemp -d "${grpo_dir}/.test.XXXXXX")"
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
  "${grpo_dir}/libeshkol_transformer_grpo_aot.a"
  "${grpo_dir}/libeshkol_transformer_grpo.a"
)

# 1. C++17 header compile and linkage check.
"${CXX}" "${cppflags[@]}" tests/grpo/header_grpo_cpp.cpp "${archives[@]}" -lm \
  -o "${scratch}/header_grpo_cpp"
"${scratch}/header_grpo_cpp"
printf 'GRPO C++17 header and linkage: PASS\n'

# 2. Native C11 normal (-O2) tests + Python double-precision oracle + f64 FD.
"${CC}" "${cflags[@]}" -O2 tests/grpo/test_grpo_native.c "${archives[@]}" -lm \
  -o "${scratch}/test_grpo_native"
"${scratch}/test_grpo_native"
python3 tests/grpo/test_grpo_oracle.py "${scratch}/test_grpo_native"

# 3. Native C11 and C++17 tests under ASan/UBSan/LSan.
export ASAN_OPTIONS=detect_leaks=1:halt_on_error=1
export UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1
"${CC}" "${cflags[@]}" "${san[@]}" tests/grpo/test_grpo_native.c \
  native/grpo_eshkol_bridge.c native/grpo.c -lm \
  -o "${scratch}/test_grpo_native.san"
"${scratch}/test_grpo_native.san"

"${CC}" "${cflags[@]}" "${san[@]}" -c native/grpo.c -o "${scratch}/grpo.san.o"
"${CC}" "${cflags[@]}" "${san[@]}" -c native/grpo_eshkol_bridge.c \
  -o "${scratch}/grpo_eshkol_bridge.san.o"
"${CXX}" "${cppflags[@]}" "${san[@]}" tests/grpo/header_grpo_cpp.cpp \
  "${scratch}/grpo_eshkol_bridge.san.o" "${scratch}/grpo.san.o" -lm \
  -o "${scratch}/header_grpo_cpp.san"
"${scratch}/header_grpo_cpp.san"
printf 'GRPO ASan/UBSan/LSan tests: PASS\n'

# 4. Compile tests/grpo/test_grpo_aot.esk TWICE from fresh dirs and verify
#    byte-identical ELFs and byte-identical stdout.
for run in 1 2; do
  mkdir -p "${scratch}/cache-${run}" "${scratch}/out-${run}" \
           "${scratch}/libs-${run}"
  bash scripts/build-grpo.sh "${scratch}/libs-${run}" >/dev/null
  diagnostic="${scratch}/compile-${run}.log"
  timeout --foreground --kill-after=5s 90s \
    env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR -u ESHKOL_AOT_MODULE_CACHE_DIR \
    ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="${scratch}/cache-${run}" \
    "${runner}" --strict-types --optimize 0 --no-stdlib \
      -L "${scratch}/libs-${run}" \
      --lib eshkol_transformer_grpo_aot --lib eshkol_transformer_grpo \
      tests/grpo/test_grpo_aot.esk \
      -o "${scratch}/out-${run}/test_grpo_aot" >"${diagnostic}" 2>&1

  diagnostic="${scratch}/run-${run}.stderr"
  timeout --foreground --kill-after=5s 60s \
    "${scratch}/out-${run}/test_grpo_aot" \
    >"${scratch}/run-${run}.stdout" 2>"${diagnostic}" || {
      cat "${scratch}/run-${run}.stdout" >&2
      die "compiled GRPO AOT binary failed on run ${run}"
    }
  [[ ! -s "${diagnostic}" ]] || die "compiled GRPO AOT binary emitted stderr"
done
diagnostic=''

readelf -h "${scratch}/out-1/test_grpo_aot" >/dev/null
cmp "${scratch}/out-1/test_grpo_aot" "${scratch}/out-2/test_grpo_aot"
cmp "${scratch}/run-1.stdout" "${scratch}/run-2.stdout"

awk '
  BEGIN { num = "^-?[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$" }
  $0 == "GRPO AOT VERIFIER exact=PASS format=PASS" { ++verifier_seen; next }
  $1 == "GRPO" && $2 == "AOT" && $3 == "STEP" {
    if (NF != 7 || $4 !~ /^[0-9]+$/ || $4 + 0 != step) exit 1
    split($5, er, "="); split($6, kl, "="); split($7, cf, "=")
    if (er[1] != "exp_reward" || er[2] !~ num) exit 1
    if (kl[1] != "kl" || kl[2] !~ num || kl[2] + 0 < 0) exit 1
    if (cf[1] != "clip_frac" || cf[2] !~ num || cf[2] + 0 < 0 || cf[2] + 0 > 1) exit 1
    if (step > 0 && er[2] + 0 <= prev_er) exit 1
    if (step == 0) init_er = er[2]
    final_er = er[2]
    prev_er = er[2] + 0
    ++step
    next
  }
  $1 == "GRPO" && $2 == "AOT" && $3 == "TRAIN" {
    if (NF != 10 || $4 != "steps=10" || $5 != "init_exp_r=" init_er ||
        $6 != "final_exp_r=" final_er || $7 != "final_greedy_r=1" ||
        $10 != "grad_checks=8")
      exit 1
    split($8, mkl, "="); split($9, tc, "=")
    if (mkl[1] != "max_kl" || mkl[2] !~ num || mkl[2] + 0 <= 0) exit 1
    if (tc[1] != "total_clipped" || tc[2] !~ /^[0-9]+$/ || tc[2] + 0 <= 0) exit 1
    ++train_seen
    next
  }
  $0 == "GRPO AOT NEGATIVE PASS checks=9" { ++neg_seen; next }
  { exit 1 }
  END {
    if (verifier_seen != 1 || step != 11 || train_seen != 1 || neg_seen != 1)
      exit 1
  }
' "${scratch}/run-1.stdout" || {
  cat "${scratch}/run-1.stdout" >&2
  die "invalid compiled GRPO AOT output"
}

cat "${scratch}/run-1.stdout"
elf_sha="$(sha256sum "${scratch}/out-1/test_grpo_aot")"
printf 'GRPO AOT ELF SHA-256 %s\n' "${elf_sha%% *}"
printf 'GRPO AOT two fresh compilations, byte-identical ELF and stdout: PASS\n'
