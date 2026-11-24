#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

cd "${PROJECT_ROOT}"
verify_toolchain
export LC_ALL=C

export CC="${CC:-$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cc_path)}"
export CXX="${CXX:-$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cxx_path)}"
runner="$(eshkol_build_dir)/eshkol-run"

lora_dir="${PROJECT_ROOT}/build/lora"
bash scripts/build-lora.sh "${lora_dir}"

scratch="$(mktemp -d "${lora_dir}/.test.XXXXXX")"
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
  "${lora_dir}/libeshkol_transformer_lora_aot.a"
  "${lora_dir}/libeshkol_transformer_lora.a"
)

# 1. C++17 header compile and linkage check.
"${CXX}" "${cppflags[@]}" tests/lora/header_lora_cpp.cpp "${archives[@]}" -lm \
  -o "${scratch}/header_lora_cpp"
"${scratch}/header_lora_cpp"
printf 'LORA C++17 header and linkage: PASS\n'

# 2. Native C11 normal (-O2) tests + Python double-precision oracle.
"${CC}" "${cflags[@]}" -O2 tests/lora/test_lora_native.c "${archives[@]}" -lm \
  -o "${scratch}/test_lora_native"
"${scratch}/test_lora_native"
python3 tests/lora/test_lora_oracle.py "${scratch}/test_lora_native"

# 3. Native C11 and C++17 tests under ASan/UBSan/LSan.
export ASAN_OPTIONS=detect_leaks=1:halt_on_error=1
export UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1
"${CC}" "${cflags[@]}" "${san[@]}" tests/lora/test_lora_native.c \
  native/lora_eshkol_bridge.c native/lora_adapter.c -lm \
  -o "${scratch}/test_lora_native.san"
"${scratch}/test_lora_native.san"

"${CC}" "${cflags[@]}" "${san[@]}" -c native/lora_adapter.c \
  -o "${scratch}/lora_adapter.san.o"
"${CC}" "${cflags[@]}" "${san[@]}" -c native/lora_eshkol_bridge.c \
  -o "${scratch}/lora_eshkol_bridge.san.o"
"${CXX}" "${cppflags[@]}" "${san[@]}" tests/lora/header_lora_cpp.cpp \
  "${scratch}/lora_eshkol_bridge.san.o" "${scratch}/lora_adapter.san.o" -lm \
  -o "${scratch}/header_lora_cpp.san"
"${scratch}/header_lora_cpp.san"
printf 'LORA ASan/UBSan/LSan tests: PASS\n'

# 4. Compile tests/lora/test_lora_aot.esk TWICE from fresh dirs and run
#    two separate OS processes per build (train -> save, fresh reload -> verify).
for run in 1 2; do
  mkdir -p "${scratch}/cache-${run}" "${scratch}/out-${run}" \
           "${scratch}/libs-${run}" "${scratch}/run-${run}"
  bash scripts/build-lora.sh "${scratch}/libs-${run}" >/dev/null
  diagnostic="${scratch}/compile-${run}.log"
  timeout --foreground --kill-after=5s 90s \
    env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR -u ESHKOL_AOT_MODULE_CACHE_DIR \
    ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="${scratch}/cache-${run}" \
    "${runner}" --strict-types --optimize 0 --no-stdlib \
      -L "${scratch}/libs-${run}" \
      --lib eshkol_transformer_lora_aot --lib eshkol_transformer_lora \
      tests/lora/test_lora_aot.esk \
      -o "${scratch}/out-${run}/test_lora_aot" >"${diagnostic}" 2>&1

  ckpt="${scratch}/run-${run}/adapter.etlr"
  tamper="${scratch}/run-${run}/tampered.etlr"

  # Process 1: 12-step LoRA fine-tuning + frozen-base check + checkpoint save.
  diagnostic="${scratch}/train-${run}.stderr"
  timeout --foreground --kill-after=5s 60s \
    env LORA_AOT_PHASE=train LORA_AOT_CKPT="${ckpt}" LORA_AOT_TAMPER="${tamper}" \
    "${scratch}/out-${run}/test_lora_aot" \
    >"${scratch}/train-${run}.stdout" 2>"${diagnostic}" || {
      cat "${scratch}/train-${run}.stdout" >&2
      die "compiled LoRA AOT train process failed on run ${run}"
    }
  [[ ! -s "${diagnostic}" ]] || die "compiled LoRA AOT train emitted stderr"

  # Process 2: fresh OS process reloads checkpoint and verifies parity + tampers.
  diagnostic="${scratch}/reload-${run}.stderr"
  timeout --foreground --kill-after=5s 60s \
    env LORA_AOT_PHASE=reload LORA_AOT_CKPT="${ckpt}" LORA_AOT_TAMPER="${tamper}" \
    "${scratch}/out-${run}/test_lora_aot" \
    >"${scratch}/reload-${run}.stdout" 2>"${diagnostic}" || {
      cat "${scratch}/reload-${run}.stdout" >&2
      die "compiled LoRA AOT reload process failed on run ${run}"
    }
  [[ ! -s "${diagnostic}" ]] || die "compiled LoRA AOT reload emitted stderr"

  cat "${scratch}/train-${run}.stdout" "${scratch}/reload-${run}.stdout" \
    >"${scratch}/combined-${run}.stdout"
done
diagnostic=''

readelf -h "${scratch}/out-1/test_lora_aot" >/dev/null
cmp "${scratch}/out-1/test_lora_aot" "${scratch}/out-2/test_lora_aot"
cmp "${scratch}/combined-1.stdout" "${scratch}/combined-2.stdout"

awk '
  BEGIN { number = "^-?[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$" }
  $1 == "LORA" && $2 == "AOT" && $3 == "LOSS" {
    if (NF != 5 || $4 !~ /^[0-9]+$/ || $4 + 0 != step || $5 !~ number || $5 + 0 < 0)
      exit 1
    if (step > 0 && $5 + 0 >= prev_loss) exit 1
    if (step == 0) init_loss = $5
    final_loss = $5
    prev_loss = $5 + 0
    ++step
    next
  }
  $1 == "LORA" && $2 == "AOT" && $3 == "TRAIN" {
    if (NF != 8 || $4 != "steps=12" || $5 != "initial=" init_loss ||
        $6 != "final=" final_loss || $7 != "base_frozen=PASS" ||
        $8 != "grad_checks=9")
      exit 1
    ++train_seen
    next
  }
  $1 == "LORA" && $2 == "AOT" && $3 == "Y_SHA256" {
    if (NF != 7) exit 1
    train_y_sha = $4 " " $5 " " $6 " " $7
    ++train_y_seen
    next
  }
  $1 == "LORA" && $2 == "AOT" && $3 == "BASE_SHA256" {
    if (NF != 7) exit 1
    train_b_sha = $4 " " $5 " " $6 " " $7
    ++train_b_seen
    next
  }
  $1 == "LORA" && $2 == "AOT" && $3 == "RELOAD_Y_SHA256" {
    if (NF != 7) exit 1
    reload_y_sha = $4 " " $5 " " $6 " " $7
    ++reload_y_seen
    next
  }
  $1 == "LORA" && $2 == "AOT" && $3 == "RELOAD_BASE_SHA256" {
    if (NF != 7) exit 1
    reload_b_sha = $4 " " $5 " " $6 " " $7
    ++reload_b_seen
    next
  }
  $1 == "LORA" && $2 == "AOT" && $3 == "RELOAD" {
    if (NF != 6 || $6 != "parity=PASS") exit 1
    ++reload_seen
    next
  }
  $0 == "LORA AOT NEGATIVE PASS" { ++neg_seen; next }
  { exit 1 }
  END {
    if (step != 13 || train_seen != 1 || train_y_seen != 1 || train_b_seen != 1 ||
        reload_y_seen != 1 || reload_b_seen != 1 || reload_seen != 1 || neg_seen != 1)
      exit 1
    if (train_y_sha != reload_y_sha || train_b_sha != reload_b_sha)
      exit 1
  }
' "${scratch}/combined-1.stdout" || {
  cat "${scratch}/combined-1.stdout" >&2
  die "invalid compiled LoRA AOT output"
}

cat "${scratch}/combined-1.stdout"
elf_sha="$(sha256sum "${scratch}/out-1/test_lora_aot")"
printf 'LORA AOT ELF SHA-256 %s\n' "${elf_sha%% *}"
printf 'LORA AOT two fresh compilations, byte-identical ELF and stdout: PASS\n'
