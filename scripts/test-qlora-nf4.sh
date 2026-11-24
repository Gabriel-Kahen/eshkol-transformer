#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

cd "${PROJECT_ROOT}"
verify_toolchain
export LC_ALL=C

export CC="${CC:-$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cc_path)}"
export CXX="${CXX:-$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cxx_path)}"
runner="$(eshkol_build_dir)/eshkol-run"

out_dir="${PROJECT_ROOT}/build/qlora_nf4"
bash scripts/build-qlora-nf4.sh "${out_dir}"

scratch="$(mktemp -d "${out_dir}/.test.XXXXXX")"
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
  "${out_dir}/libeshkol_transformer_qlora_nf4_aot.a"
  "${out_dir}/libeshkol_transformer_qlora_nf4.a"
)

# 1. C++17 header compile and linkage check.
"${CXX}" "${cppflags[@]}" -O2 tests/qlora_nf4/header_qlora_cpp.cpp \
  "${archives[@]}" -lm -o "${scratch}/header_qlora_cpp"
"${scratch}/header_qlora_cpp"
printf 'QLORA NF4 C++17 header and linkage: PASS\n'

# 2. Native C11 normal (-O2, zero-alloc wrapped hot kernels) + Python oracle.
"${CC}" "${cflags[@]}" -O2 tests/qlora_nf4/test_qlora_nf4_native.c \
  "${archives[@]}" "${wrap[@]}" -lm -o "${scratch}/test_qlora_nf4_native"
"${scratch}/test_qlora_nf4_native"
python3 tests/qlora_nf4/test_qlora_nf4_oracle.py "${scratch}/test_qlora_nf4_native"

# 3. Native C11 and C++17 tests under ASan/UBSan/LSan.
export ASAN_OPTIONS=detect_leaks=1:halt_on_error=1
export UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1
"${CC}" "${cflags[@]}" "${san[@]}" -c native/qlora_nf4.c \
  -o "${scratch}/qlora_nf4.san.o"
"${CC}" "${cflags[@]}" "${san[@]}" -c native/qlora_nf4_eshkol_bridge.c \
  -o "${scratch}/qlora_nf4_eshkol_bridge.san.o"

"${CC}" "${cflags[@]}" "${san[@]}" tests/qlora_nf4/test_qlora_nf4_native.c \
  "${scratch}/qlora_nf4_eshkol_bridge.san.o" "${scratch}/qlora_nf4.san.o" \
  "${wrap[@]}" -lm -o "${scratch}/test_qlora_nf4_native.san"
"${scratch}/test_qlora_nf4_native.san"

"${CXX}" "${cppflags[@]}" "${san[@]}" tests/qlora_nf4/header_qlora_cpp.cpp \
  "${scratch}/qlora_nf4_eshkol_bridge.san.o" "${scratch}/qlora_nf4.san.o" \
  -lm -o "${scratch}/header_qlora_cpp.san"
"${scratch}/header_qlora_cpp.san"
printf 'QLORA NF4 ASan/UBSan/LSan tests: PASS\n'

# 4. Compile tests/qlora_nf4/test_qlora_nf4_aot.esk TWICE from fresh dirs and
#    run 3 separate OS processes per build (full12, part1 -> save, fresh part2 -> reload + finish + negatives).
for run in 1 2; do
  mkdir -p "${scratch}/cache-${run}" "${scratch}/out-${run}" \
           "${scratch}/libs-${run}" "${scratch}/run-${run}"
  bash scripts/build-qlora-nf4.sh "${scratch}/libs-${run}" >/dev/null
  diagnostic="${scratch}/compile-${run}.log"
  timeout --foreground --kill-after=5s 90s \
    env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR -u ESHKOL_AOT_MODULE_CACHE_DIR \
    ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="${scratch}/cache-${run}" \
    "${runner}" --strict-types --optimize 0 --no-stdlib \
      -L "${scratch}/libs-${run}" \
      --lib eshkol_transformer_qlora_nf4_aot \
      --lib eshkol_transformer_qlora_nf4 \
      tests/qlora_nf4/test_qlora_nf4_aot.esk \
      -o "${scratch}/out-${run}/test_qlora_nf4_aot" >"${diagnostic}" 2>&1

  ckpt="${scratch}/run-${run}/step6.etql"
  tamper="${scratch}/run-${run}/tampered.etql"

  # Process 1: Quantize SQ/DQ, RMSE vs INT4, FD grad checks, full 12-step AdamW, merge parity.
  diagnostic="${scratch}/full12-${run}.stderr"
  timeout --foreground --kill-after=5s 60s \
    env QLORA_AOT_PHASE=full12 QLORA_AOT_CKPT="${ckpt}" QLORA_AOT_TAMPER="${tamper}" \
    "${scratch}/out-${run}/test_qlora_nf4_aot" \
    >"${scratch}/full12-${run}.stdout" 2>"${diagnostic}" || {
      cat "${scratch}/full12-${run}.stdout" >&2
      die "compiled QLoRA NF4 AOT full12 process failed on run ${run}"
    }
  [[ ! -s "${diagnostic}" ]] || die "compiled QLoRA NF4 AOT full12 emitted stderr"

  # Process 2: Part 1 of 6+6 (first 6 AdamW steps + ETQL1000 checkpoint save).
  diagnostic="${scratch}/part1-${run}.stderr"
  timeout --foreground --kill-after=5s 60s \
    env QLORA_AOT_PHASE=part1 QLORA_AOT_CKPT="${ckpt}" QLORA_AOT_TAMPER="${tamper}" \
    "${scratch}/out-${run}/test_qlora_nf4_aot" \
    >"${scratch}/part1-${run}.stdout" 2>"${diagnostic}" || {
      cat "${scratch}/part1-${run}.stdout" >&2
      die "compiled QLoRA NF4 AOT part1 process failed on run ${run}"
    }
  [[ ! -s "${diagnostic}" ]] || die "compiled QLoRA NF4 AOT part1 emitted stderr"

  # Process 3: Fresh OS process reloads ETQL1000, finishes steps 7..12, verifies parity & negatives.
  diagnostic="${scratch}/part2-${run}.stderr"
  timeout --foreground --kill-after=5s 60s \
    env QLORA_AOT_PHASE=part2 QLORA_AOT_CKPT="${ckpt}" QLORA_AOT_TAMPER="${tamper}" \
    "${scratch}/out-${run}/test_qlora_nf4_aot" \
    >"${scratch}/part2-${run}.stdout" 2>"${diagnostic}" || {
      cat "${scratch}/part2-${run}.stdout" >&2
      die "compiled QLoRA NF4 AOT part2 process failed on run ${run}"
    }
  [[ ! -s "${diagnostic}" ]] || die "compiled QLoRA NF4 AOT part2 emitted stderr"

  cat "${scratch}/full12-${run}.stdout" "${scratch}/part1-${run}.stdout" \
      "${scratch}/part2-${run}.stdout" >"${scratch}/combined-${run}.stdout"
done
diagnostic=''

readelf -h "${scratch}/out-1/test_qlora_nf4_aot" >/dev/null
cmp "${scratch}/out-1/test_qlora_nf4_aot" "${scratch}/out-2/test_qlora_nf4_aot"
cmp "${scratch}/combined-1.stdout" "${scratch}/combined-2.stdout"

awk '
  BEGIN { number = "^-?[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$" }
  $1 == "QLORA" && $2 == "AOT" && $3 == "QUANT" {
    if (NF != 9 || $4 != "f32_bytes=98304" || $5 != "sq_bytes=13824" ||
        $6 != "dq_bytes=12688")
      exit 1
    ++quant_seen; next
  }
  $1 == "QLORA" && $2 == "AOT" && $3 == "RMSE" {
    if (NF != 7 || $7 != "nf4_beats_int4=PASS") exit 1
    ++rmse_seen; next
  }
  $1 == "QLORA" && $2 == "AOT" && $3 == "GRAD" {
    if (NF != 6 || $4 != "checks=12" || $6 != "PASS") exit 1
    ++grad_seen; next
  }
  $1 == "QLORA" && $2 == "AOT" && $3 == "FULL12_LOSS" {
    if (NF != 5 || $4 !~ /^[0-9]+$/ || $4 + 0 != fstep || $5 !~ number || $5 + 0 < 0)
      exit 1
    if (fstep > 0 && $5 + 0 >= prev_floss) exit 1
    full_loss[fstep] = $5
    prev_floss = $5 + 0
    ++fstep; next
  }
  $1 == "QLORA" && $2 == "AOT" && $3 == "FULL12" {
    if (NF != 8 || $4 != "steps=12" || $8 != "base_frozen=PASS") exit 1
    ++full12_seen; next
  }
  $1 == "QLORA" && $2 == "AOT" && $3 == "BASE_SHA256" {
    if (NF != 7) exit 1
    base_sha = $4 " " $5 " " $6 " " $7; ++base_seen; next
  }
  $1 == "QLORA" && $2 == "AOT" && $3 == "FULL12_Y_SHA256" {
    if (NF != 7) exit 1
    full_y = $4 " " $5 " " $6 " " $7; ++full_y_seen; next
  }
  $1 == "QLORA" && $2 == "AOT" && $3 == "FULL12_STATE_SHA256" {
    if (NF != 7) exit 1
    full_st = $4 " " $5 " " $6 " " $7; ++full_st_seen; next
  }
  $0 == "QLORA AOT PART1 saved_step=6 base_frozen=PASS" {
    ++part1_seen; next
  }
  $1 == "QLORA" && $2 == "AOT" && $3 == "RESUME_LOSS" {
    if (NF != 5 || $4 !~ /^[0-9]+$/ || $4 + 0 != rstep + 7 || $5 != full_loss[rstep + 7])
      exit 1
    ++rstep; next
  }
  $1 == "QLORA" && $2 == "AOT" && $3 == "RESUME_BASE_SHA256" {
    if (NF != 7) exit 1
    res_base = $4 " " $5 " " $6 " " $7; ++res_base_seen; next
  }
  $1 == "QLORA" && $2 == "AOT" && $3 == "RESUME12_Y_SHA256" {
    if (NF != 7) exit 1
    res_y = $4 " " $5 " " $6 " " $7; ++res_y_seen; next
  }
  $1 == "QLORA" && $2 == "AOT" && $3 == "RESUME12_STATE_SHA256" {
    if (NF != 7) exit 1
    res_st = $4 " " $5 " " $6 " " $7; ++res_st_seen; next
  }
  $1 == "QLORA" && $2 == "AOT" && $3 == "RESUME" {
    if (NF != 6 || $6 != "parity=PASS") exit 1
    ++resume_seen; next
  }
  $0 == "QLORA AOT NEGATIVE checks=13 PASS" {
    ++neg_seen; next
  }
  { exit 1 }
  END {
    if (quant_seen != 1 || rmse_seen != 1 || grad_seen != 1 || fstep != 13 ||
        full12_seen != 1 || base_seen != 1 || full_y_seen != 1 || full_st_seen != 1 ||
        part1_seen != 1 || rstep != 6 || res_base_seen != 1 || res_y_seen != 1 ||
        res_st_seen != 1 || resume_seen != 1 || neg_seen != 1)
      exit 1
    if (base_sha != res_base || full_y != res_y || full_st != res_st)
      exit 1
  }
' "${scratch}/combined-1.stdout" || {
  cat "${scratch}/combined-1.stdout" >&2
  die "invalid compiled QLoRA NF4 AOT output"
}

cat "${scratch}/combined-1.stdout"
elf_sha="$(sha256sum "${scratch}/out-1/test_qlora_nf4_aot")"
printf 'QLORA NF4 AOT ELF SHA-256 %s\n' "${elf_sha%% *}"
printf 'QLORA NF4 AOT two fresh compilations, byte-identical ELF and stdout: PASS\n'
