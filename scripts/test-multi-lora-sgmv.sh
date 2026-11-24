#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

cd "${PROJECT_ROOT}"
verify_toolchain
export LC_ALL=C

export CC="${CC:-$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cc_path)}"
export CXX="${CXX:-$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cxx_path)}"
runner="$(eshkol_build_dir)/eshkol-run"

mlora_dir="${PROJECT_ROOT}/build/multi_lora_sgmv"
bash scripts/build-multi-lora-sgmv.sh "${mlora_dir}"

scratch="$(mktemp -d "${mlora_dir}/.test.XXXXXX")"
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
  "${mlora_dir}/libeshkol_transformer_multi_lora_sgmv_aot.a"
  "${mlora_dir}/libeshkol_transformer_multi_lora_sgmv.a"
)

# 1. C++17 header compile and linkage check.
"${CXX}" "${cppflags[@]}" -O2 tests/multi_lora_sgmv/header_mlora_cpp.cpp \
  "${archives[@]}" -lm -o "${scratch}/header_mlora_cpp"
"${scratch}/header_mlora_cpp"
printf 'MULTI-LORA SGMV C++17 header and linkage: PASS\n'

# 2. Native C11 normal (-O2, hot-path zero-alloc wrapped) tests + Python oracle.
"${CC}" "${cflags[@]}" -O2 tests/multi_lora_sgmv/test_multi_lora_sgmv_native.c \
  "${archives[@]}" "${wrap[@]}" -lm -o "${scratch}/test_multi_lora_sgmv_native"
"${scratch}/test_multi_lora_sgmv_native"
python3 tests/multi_lora_sgmv/test_multi_lora_sgmv_oracle.py \
  "${scratch}/test_multi_lora_sgmv_native"

# 3. Native C11 (zero-alloc wrapped) + C++17 under ASan/UBSan/LSan.
export ASAN_OPTIONS=detect_leaks=1:halt_on_error=1
export UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1

"${CC}" "${cflags[@]}" "${san[@]}" -c native/multi_lora_sgmv.c \
  -o "${scratch}/multi_lora_sgmv.san.o"
"${CC}" "${cflags[@]}" "${san[@]}" -c native/multi_lora_sgmv_eshkol_bridge.c \
  -o "${scratch}/multi_lora_sgmv_eshkol_bridge.san.o"

"${CC}" "${cflags[@]}" "${san[@]}" \
  tests/multi_lora_sgmv/test_multi_lora_sgmv_native.c \
  "${scratch}/multi_lora_sgmv_eshkol_bridge.san.o" \
  "${scratch}/multi_lora_sgmv.san.o" "${wrap[@]}" -lm \
  -o "${scratch}/test_multi_lora_sgmv_native.san"
"${scratch}/test_multi_lora_sgmv_native.san"

"${CXX}" "${cppflags[@]}" "${san[@]}" \
  tests/multi_lora_sgmv/header_mlora_cpp.cpp \
  "${scratch}/multi_lora_sgmv_eshkol_bridge.san.o" \
  "${scratch}/multi_lora_sgmv.san.o" -lm \
  -o "${scratch}/header_mlora_cpp.san"
"${scratch}/header_mlora_cpp.san"
printf 'MULTI-LORA SGMV ASan/UBSan/LSan tests: PASS\n'

# 4. Compile tests/multi_lora_sgmv/test_multi_lora_sgmv_aot.esk TWICE from fresh
#    dirs and run two separate OS processes per build (train/hotswap/save ->
#    fresh OS process reload/verify/negatives).
for run in 1 2; do
  mkdir -p "${scratch}/cache-${run}" "${scratch}/out-${run}" \
           "${scratch}/libs-${run}" "${scratch}/run-${run}"
  bash scripts/build-multi-lora-sgmv.sh "${scratch}/libs-${run}" >/dev/null
  diagnostic="${scratch}/compile-${run}.log"
  timeout --foreground --kill-after=5s 90s \
    env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR -u ESHKOL_AOT_MODULE_CACHE_DIR \
    ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="${scratch}/cache-${run}" \
    "${runner}" --strict-types --optimize 0 --no-stdlib \
      -L "${scratch}/libs-${run}" \
      --lib eshkol_transformer_multi_lora_sgmv_aot \
      --lib eshkol_transformer_multi_lora_sgmv \
      tests/multi_lora_sgmv/test_multi_lora_sgmv_aot.esk \
      -o "${scratch}/out-${run}/test_multi_lora_sgmv_aot" >"${diagnostic}" 2>&1

  bundle="${scratch}/run-${run}/adapters.etml"
  tamper="${scratch}/run-${run}/tampered.etml"

  # Process 1: parity + 12-step multi-adapter AdamW + FD + hot-swap + save.
  diagnostic="${scratch}/train-${run}.stderr"
  timeout --foreground --kill-after=5s 60s \
    env MLORA_AOT_PHASE=train MLORA_AOT_BUNDLE="${bundle}" \
      MLORA_AOT_TAMPER="${tamper}" \
    "${scratch}/out-${run}/test_multi_lora_sgmv_aot" \
    >"${scratch}/train-${run}.stdout" 2>"${diagnostic}" || {
      cat "${scratch}/train-${run}.stdout" >&2
      die "compiled Multi-LoRA AOT train process failed on run ${run}"
    }
  [[ ! -s "${diagnostic}" ]] || die "compiled Multi-LoRA AOT train emitted stderr"

  # Process 2: fresh OS process reloads bundle and verifies parity + negatives.
  diagnostic="${scratch}/reload-${run}.stderr"
  timeout --foreground --kill-after=5s 60s \
    env MLORA_AOT_PHASE=reload MLORA_AOT_BUNDLE="${bundle}" \
      MLORA_AOT_TAMPER="${tamper}" \
    "${scratch}/out-${run}/test_multi_lora_sgmv_aot" \
    >"${scratch}/reload-${run}.stdout" 2>"${diagnostic}" || {
      cat "${scratch}/reload-${run}.stdout" >&2
      die "compiled Multi-LoRA AOT reload process failed on run ${run}"
    }
  [[ ! -s "${diagnostic}" ]] || die "compiled Multi-LoRA AOT reload emitted stderr"

  cat "${scratch}/train-${run}.stdout" "${scratch}/reload-${run}.stdout" \
    >"${scratch}/combined-${run}.stdout"
done
diagnostic=''

readelf -h "${scratch}/out-1/test_multi_lora_sgmv_aot" >/dev/null
cmp "${scratch}/out-1/test_multi_lora_sgmv_aot" \
    "${scratch}/out-2/test_multi_lora_sgmv_aot"
cmp "${scratch}/combined-1.stdout" "${scratch}/combined-2.stdout"

awk '
  BEGIN { number = "^-?[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$" }
  $0 == "MLORA AOT PARITY m=12 base_rows=3 adapted_rows=9 segments=3 bitwise_diff=0 PASS" {
    ++parity_ok; next
  }
  $1 == "MLORA" && $2 == "AOT" && $3 == "LOSS" {
    if (NF != 7 || $4 !~ /^[0-9]+$/ || $4 + 0 != step) exit 1
    if ($5 !~ number || $6 !~ number || $7 !~ number) exit 1
    if (step > 0 && ($5 + 0 >= p1 || $6 + 0 >= p2 || $7 + 0 >= p3)) exit 1
    p1 = $5 + 0; p2 = $6 + 0; p3 = $7 + 0
    ++step
    next
  }
  $1 == "MLORA" && $2 == "AOT" && $3 == "FD" {
    if (NF != 7 || $4 != "checks=10" || $5 !~ /^max_err=/ ||
        $6 != "idle_bleed=0" || $7 != "PASS") exit 1
    ++fd_ok; next
  }
  $1 == "MLORA" && $2 == "AOT" && $3 == "TRAIN" {
    if (NF != 8 || $4 != "steps=12" || $8 != "base_frozen=PASS") exit 1
    ++train_ok; next
  }
  $0 == "MLORA AOT HOTSWAP old_gen=1 new_gen=2 busy_reject=PASS" {
    ++hotswap_ok; next
  }
  $1 == "MLORA" && $2 == "AOT" && $3 == "Y_SHA256" {
    if (NF != 7) exit 1
    y_sha = $4 " " $5 " " $6 " " $7
    ++y_seen; next
  }
  $1 == "MLORA" && $2 == "AOT" && $3 == "BASE_SHA256" {
    if (NF != 7) exit 1
    b_sha = $4 " " $5 " " $6 " " $7
    ++b_seen; next
  }
  $1 == "MLORA" && $2 == "AOT" && $3 == "RELOAD_Y_SHA256" {
    if (NF != 7) exit 1
    ry_sha = $4 " " $5 " " $6 " " $7
    ++ry_seen; next
  }
  $1 == "MLORA" && $2 == "AOT" && $3 == "RELOAD_BASE_SHA256" {
    if (NF != 7) exit 1
    rb_sha = $4 " " $5 " " $6 " " $7
    ++rb_seen; next
  }
  $0 == "MLORA AOT RELOAD adapters=4 bitwise_parity=PASS" {
    ++reload_ok; next
  }
  $0 == "MLORA AOT NEGATIVE PASS" { ++neg_ok; next }
  { exit 1 }
  END {
    if (parity_ok != 1 || step != 13 || fd_ok != 1 || train_ok != 1 ||
        hotswap_ok != 1 || y_seen != 1 || b_seen != 1 || ry_seen != 1 ||
        rb_seen != 1 || reload_ok != 1 || neg_ok != 1)
      exit 1
    if (y_sha != ry_sha || b_sha != rb_sha)
      exit 1
  }
' "${scratch}/combined-1.stdout" || {
  cat "${scratch}/combined-1.stdout" >&2
  die "invalid compiled Multi-LoRA AOT output"
}

cat "${scratch}/combined-1.stdout"
elf_sha="$(sha256sum "${scratch}/out-1/test_multi_lora_sgmv_aot")"
printf 'MULTI-LORA SGMV AOT ELF SHA-256 %s\n' "${elf_sha%% *}"
printf 'MULTI-LORA SGMV AOT two fresh compilations, byte-identical ELF and stdout: PASS\n'
