#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

cd "${PROJECT_ROOT}"
verify_toolchain
export LC_ALL=C

export CC="${CC:-$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cc_path)}"
export CXX="${CXX:-$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cxx_path)}"
runner="$(eshkol_build_dir)/eshkol-run"

muon_dir="${PROJECT_ROOT}/build/o2_muon"
bash scripts/build-o2-muon.sh "${muon_dir}"

scratch="$(mktemp -d "${muon_dir}/.test.XXXXXX")"
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
  "${muon_dir}/libeshkol_transformer_o2_muon_aot.a"
  "${muon_dir}/libeshkol_transformer_o2_muon.a"
)

# 1. C++17 header compile and linkage check.
"${CXX}" "${cppflags[@]}" tests/o2_muon/header_muon_cpp.cpp "${archives[@]}" -lm \
  -o "${scratch}/header_muon_cpp"
"${scratch}/header_muon_cpp"
printf 'O2 MUON C++17 header and linkage: PASS\n'

# 2. Native C11 normal (-O2) tests + Python binary64 oracle.
"${CC}" "${cflags[@]}" -O2 tests/o2_muon/test_o2_muon_native.c "${archives[@]}" -lm \
  -o "${scratch}/test_o2_muon_native"
"${scratch}/test_o2_muon_native"
python3 tests/o2_muon/test_o2_muon_oracle.py "${scratch}/test_o2_muon_native"

# 3. Native C11 and C++17 tests under ASan/UBSan/LSan.
export ASAN_OPTIONS=detect_leaks=1:halt_on_error=1
export UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1
"${CC}" "${cflags[@]}" "${san[@]}" tests/o2_muon/test_o2_muon_native.c \
  native/o2_muon_eshkol_bridge.c native/o2_muon_optimizer.c -lm \
  -o "${scratch}/test_o2_muon_native.san"
"${scratch}/test_o2_muon_native.san"

"${CC}" "${cflags[@]}" "${san[@]}" -c native/o2_muon_optimizer.c \
  -o "${scratch}/o2_muon_optimizer.san.o"
"${CC}" "${cflags[@]}" "${san[@]}" -c native/o2_muon_eshkol_bridge.c \
  -o "${scratch}/o2_muon_eshkol_bridge.san.o"
"${CXX}" "${cppflags[@]}" "${san[@]}" tests/o2_muon/header_muon_cpp.cpp \
  "${scratch}/o2_muon_eshkol_bridge.san.o" "${scratch}/o2_muon_optimizer.san.o" -lm \
  -o "${scratch}/header_muon_cpp.san"
"${scratch}/header_muon_cpp.san"
printf 'O2 MUON ASan/UBSan/LSan tests: PASS\n'

# 4. Compile tests/o2_muon/test_o2_muon_aot.esk TWICE from fresh dirs and verify
#    byte-identical ELFs and identical stdout.
for run in 1 2; do
  mkdir -p "${scratch}/cache-${run}" "${scratch}/out-${run}" \
           "${scratch}/libs-${run}"
  bash scripts/build-o2-muon.sh "${scratch}/libs-${run}" >/dev/null
  diagnostic="${scratch}/compile-${run}.log"
  timeout --foreground --kill-after=5s 90s \
    env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR -u ESHKOL_AOT_MODULE_CACHE_DIR \
    ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="${scratch}/cache-${run}" \
    "${runner}" --strict-types --optimize 0 --no-stdlib \
      -L "${scratch}/libs-${run}" \
      --lib eshkol_transformer_o2_muon_aot --lib eshkol_transformer_o2_muon \
      tests/o2_muon/test_o2_muon_aot.esk \
      -o "${scratch}/out-${run}/test_o2_muon_aot" >"${diagnostic}" 2>&1

  diagnostic="${scratch}/run-${run}.stderr"
  timeout --foreground --kill-after=5s 60s \
    "${scratch}/out-${run}/test_o2_muon_aot" \
    >"${scratch}/run-${run}.stdout" 2>"${diagnostic}" || {
      cat "${scratch}/run-${run}.stdout" >&2
      die "compiled O2 Muon AOT executable failed on run ${run}"
    }
  [[ ! -s "${diagnostic}" ]] || die "compiled O2 Muon AOT emitted stderr"
done
diagnostic=''

readelf -h "${scratch}/out-1/test_o2_muon_aot" >/dev/null
cmp "${scratch}/out-1/test_o2_muon_aot" "${scratch}/out-2/test_o2_muon_aot"
cmp "${scratch}/run-1.stdout" "${scratch}/run-2.stdout"

awk '
  BEGIN { number = "^-?[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$" }
  $0 == "O2 MUON AOT NEGATIVE PASS" { ++neg_seen; next }
  $1 == "O2" && $2 == "MUON" && $3 == "AOT" && $4 == "LOSS" {
    if (NF != 6 || $5 !~ /^[0-9]+$/ || $5 + 0 != loss_step ||
        $6 !~ number || $6 + 0 < 0)
      exit 1
    if (loss_step > 0 && $6 + 0 >= prev_loss) exit 1
    if (loss_step == 0) init_loss = $6
    final_loss = $6
    prev_loss = $6 + 0
    ++loss_step
    next
  }
  $1 == "O2" && $2 == "MUON" && $3 == "AOT" && $4 == "STEP" {
    if (NF != 10 || $5 !~ /^[0-9]+$/ || $5 + 0 != step_idx)
      exit 1
    ++step_idx
    next
  }
  $1 == "O2" && $2 == "MUON" && $3 == "AOT" && $4 == "SUMMARY" {
    if (NF != 10 || $5 != "steps=15" || $6 != "initial=" init_loss ||
        $7 != "final=" final_loss || $9 != "phases=4/5/6" ||
        $10 != "grad_checks=6")
      exit 1
    ++summary_seen
    next
  }
  { exit 1 }
  END {
    if (neg_seen != 1 || loss_step != 16 || step_idx != 15 || summary_seen != 1)
      exit 1
  }
' "${scratch}/run-1.stdout" || {
  cat "${scratch}/run-1.stdout" >&2
  die "invalid compiled O2 Muon AOT output"
}

cat "${scratch}/run-1.stdout"
elf_sha="$(sha256sum "${scratch}/out-1/test_o2_muon_aot")"
printf 'O2 MUON AOT ELF SHA-256 %s\n' "${elf_sha%% *}"
printf 'O2 MUON AOT two fresh compilations, byte-identical ELF and stdout: PASS\n'
