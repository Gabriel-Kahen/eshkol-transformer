#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

cd "${PROJECT_ROOT}"
verify_toolchain
export LC_ALL=C

export CC="${CC:-$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cc_path)}"
export CXX="${CXX:-$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cxx_path)}"
runner="$(eshkol_build_dir)/eshkol-run"

dpo_dir="${PROJECT_ROOT}/build/dpo"
bash scripts/build-dpo.sh "${dpo_dir}"

scratch="$(mktemp -d "${dpo_dir}/.test.XXXXXX")"
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
  "${dpo_dir}/libeshkol_transformer_dpo_aot.a"
  "${dpo_dir}/libeshkol_transformer_dpo.a"
)

# 1. C++17 header compile and linkage check.
"${CXX}" "${cppflags[@]}" tests/dpo/header_dpo_cpp.cpp "${archives[@]}" -lm \
  -o "${scratch}/header_dpo_cpp"
"${scratch}/header_dpo_cpp"
printf 'DPO C++17 header and linkage: PASS\n'

# 2. Native C11 normal (-O2) tests + independent binary64 Python oracle.
"${CC}" "${cflags[@]}" -O2 tests/dpo/test_dpo_native.c "${archives[@]}" -lm \
  -o "${scratch}/test_dpo_native"
"${scratch}/test_dpo_native"
python3 tests/dpo/test_dpo_oracle.py "${scratch}/test_dpo_native"

# 3. Native C11 and C++17 tests under ASan/UBSan/LSan.
export ASAN_OPTIONS=detect_leaks=1:halt_on_error=1
export UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1
"${CC}" "${cflags[@]}" "${san[@]}" tests/dpo/test_dpo_native.c \
  native/dpo_eshkol_bridge.c native/dpo_alignment.c -lm \
  -o "${scratch}/test_dpo_native.san"
"${scratch}/test_dpo_native.san"

"${CC}" "${cflags[@]}" "${san[@]}" -c native/dpo_alignment.c \
  -o "${scratch}/dpo_alignment.san.o"
"${CC}" "${cflags[@]}" "${san[@]}" -c native/dpo_eshkol_bridge.c \
  -o "${scratch}/dpo_eshkol_bridge.san.o"
"${CXX}" "${cppflags[@]}" "${san[@]}" tests/dpo/header_dpo_cpp.cpp \
  "${scratch}/dpo_eshkol_bridge.san.o" "${scratch}/dpo_alignment.san.o" -lm \
  -o "${scratch}/header_dpo_cpp.san"
"${scratch}/header_dpo_cpp.san"
printf 'DPO ASan/UBSan/LSan tests: PASS\n'

# 4. Compile tests/dpo/test_dpo_aot.esk TWICE from fresh dirs and verify
#    byte-identical ELFs and identical stdout.
for run in 1 2; do
  mkdir -p "${scratch}/cache-${run}" "${scratch}/out-${run}" \
           "${scratch}/libs-${run}"
  bash scripts/build-dpo.sh "${scratch}/libs-${run}" >/dev/null
  diagnostic="${scratch}/compile-${run}.log"
  timeout --foreground --kill-after=5s 90s \
    env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR -u ESHKOL_AOT_MODULE_CACHE_DIR \
    ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="${scratch}/cache-${run}" \
    "${runner}" --strict-types --optimize 0 --no-stdlib \
      -L "${scratch}/libs-${run}" \
      --lib eshkol_transformer_dpo_aot --lib eshkol_transformer_dpo \
      tests/dpo/test_dpo_aot.esk \
      -o "${scratch}/out-${run}/test_dpo_aot" >"${diagnostic}" 2>&1

  diagnostic="${scratch}/run-${run}.stderr"
  timeout --foreground --kill-after=5s 60s \
    "${scratch}/out-${run}/test_dpo_aot" \
    >"${scratch}/run-${run}.stdout" 2>"${diagnostic}" || {
      cat "${scratch}/run-${run}.stdout" >&2
      die "compiled DPO AOT executable failed on run ${run}"
    }
  [[ ! -s "${diagnostic}" ]] || die "compiled DPO AOT emitted stderr"
done
diagnostic=''

readelf -h "${scratch}/out-1/test_dpo_aot" >/dev/null
cmp "${scratch}/out-1/test_dpo_aot" "${scratch}/out-2/test_dpo_aot"
cmp "${scratch}/run-1.stdout" "${scratch}/run-2.stdout"

awk '
  BEGIN { number = "^-?[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$" }
  $1 == "DPO" && $2 == "AOT" && $3 == "STEP" {
    if (NF != 9 || $4 !~ /^[0-9]+$/ || $4 + 0 != step) exit 1
    if ($5 !~ /^loss=/ || $6 !~ /^margin=/ || $7 !~ /^r_w=/ ||
        $8 !~ /^r_l=/ || $9 !~ /^acc=/) exit 1
    loss = substr($5, 6); margin = substr($6, 8); acc = substr($9, 5)
    if (loss !~ number || margin !~ number || acc !~ number) exit 1
    if (step == 0) {
      init_loss = loss; init_margin = margin
    } else {
      if (loss + 0 >= prev_loss) exit 1
      if (margin + 0 <= prev_margin) exit 1
    }
    final_loss = loss; final_margin = margin; final_acc = acc
    prev_loss = loss + 0; prev_margin = margin + 0
    ++step
    next
  }
  $1 == "DPO" && $2 == "AOT" && $3 == "MODES" {
    if (NF != 5 || $4 !~ /^cdpo_loss=/ || $5 !~ /^ref_free_loss=/) exit 1
    ++modes_seen
    next
  }
  $1 == "DPO" && $2 == "AOT" && $3 == "RESULT" {
    if (NF != 11 || $4 != "steps=12" ||
        $5 != "initial_loss=" init_loss ||
        $6 != "final_loss=" final_loss ||
        $7 != "initial_margin=" init_margin ||
        $8 != "final_margin=" final_margin ||
        $9 != "final_acc=" final_acc ||
        $10 != "ref_frozen=PASS" || $11 != "grad_checks=10") exit 1
    if (final_acc + 0 != 1.0) exit 1
    ++result_seen
    next
  }
  $1 == "DPO" && $2 == "AOT" && $3 == "REF_SHA256" {
    if (NF != 7) exit 1
    ++sha_seen
    next
  }
  $0 == "DPO AOT NEGATIVE PASS" { ++neg_seen; next }
  { exit 1 }
  END {
    if (step != 13 || modes_seen != 1 || result_seen != 1 ||
        sha_seen != 1 || neg_seen != 1) exit 1
  }
' "${scratch}/run-1.stdout" || {
  cat "${scratch}/run-1.stdout" >&2
  die "invalid compiled DPO AOT output"
}

cat "${scratch}/run-1.stdout"
elf_sha="$(sha256sum "${scratch}/out-1/test_dpo_aot")"
printf 'DPO AOT ELF SHA-256 %s\n' "${elf_sha%% *}"
printf 'DPO AOT two fresh compilations, byte-identical ELF and stdout: PASS\n'
