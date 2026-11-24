#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

cd "${PROJECT_ROOT}"
verify_toolchain
export LC_ALL=C

export CC="${CC:-$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cc_path)}"
export CXX="${CXX:-$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cxx_path)}"
runner="$(eshkol_build_dir)/eshkol-run"

out_dir="${PROJECT_ROOT}/build/w4a32_quant"
bash scripts/build-w4a32-quant.sh "${out_dir}"

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
  "${out_dir}/libeshkol_transformer_w4a32_quant_aot.a"
  "${out_dir}/libeshkol_transformer_w4a32_quant.a"
)

# 1. C++17 header compile and linkage check.
"${CXX}" "${cppflags[@]}" -O2 tests/w4a32_quant/header_w4a32_cpp.cpp \
  "${archives[@]}" -lm -o "${scratch}/header_w4a32_cpp"
"${scratch}/header_w4a32_cpp"
printf 'W4A32 C++17 header and linkage: PASS\n'

# 2. Native C11 normal (-O2, zero-alloc wrapped hot kernels) + Python oracle.
"${CC}" "${cflags[@]}" -O2 tests/w4a32_quant/test_w4a32_native.c \
  "${archives[@]}" "${wrap[@]}" -lm -o "${scratch}/test_w4a32_native"
"${scratch}/test_w4a32_native"
python3 tests/w4a32_quant/test_w4a32_oracle.py "${scratch}/test_w4a32_native"

# 3. Native C11 and C++17 tests under ASan/UBSan/LSan.
export ASAN_OPTIONS=detect_leaks=1:halt_on_error=1
export UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1
"${CC}" "${cflags[@]}" "${san[@]}" -c native/w4a32_quant.c \
  -o "${scratch}/w4a32_quant.san.o"
"${CC}" "${cflags[@]}" "${san[@]}" -c native/w4a32_quant_eshkol_bridge.c \
  -o "${scratch}/w4a32_quant_eshkol_bridge.san.o"

"${CC}" "${cflags[@]}" "${san[@]}" tests/w4a32_quant/test_w4a32_native.c \
  "${scratch}/w4a32_quant_eshkol_bridge.san.o" "${scratch}/w4a32_quant.san.o" \
  "${wrap[@]}" -lm -o "${scratch}/test_w4a32_native.san"
"${scratch}/test_w4a32_native.san"

"${CXX}" "${cppflags[@]}" "${san[@]}" tests/w4a32_quant/header_w4a32_cpp.cpp \
  "${scratch}/w4a32_quant_eshkol_bridge.san.o" "${scratch}/w4a32_quant.san.o" \
  -lm -o "${scratch}/header_w4a32_cpp.san"
"${scratch}/header_w4a32_cpp.san"
printf 'W4A32 ASan/UBSan/LSan tests: PASS\n'

# 4. Compile tests/w4a32_quant/test_w4a32_aot.esk TWICE from fresh dirs and run
#    two separate OS processes per build (save -> fresh reload -> verify).
for run in 1 2; do
  mkdir -p "${scratch}/cache-${run}" "${scratch}/out-${run}" \
           "${scratch}/libs-${run}" "${scratch}/run-${run}"
  bash scripts/build-w4a32-quant.sh "${scratch}/libs-${run}" >/dev/null
  diagnostic="${scratch}/compile-${run}.log"
  timeout --foreground --kill-after=5s 90s \
    env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR -u ESHKOL_AOT_MODULE_CACHE_DIR \
    ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="${scratch}/cache-${run}" \
    "${runner}" --strict-types --optimize 0 --no-stdlib \
      -L "${scratch}/libs-${run}" \
      --lib eshkol_transformer_w4a32_quant_aot \
      --lib eshkol_transformer_w4a32_quant \
      tests/w4a32_quant/test_w4a32_aot.esk \
      -o "${scratch}/out-${run}/test_w4a32_aot" >"${diagnostic}" 2>&1

  # Process 1: quantize 2-layer MLP (Q4_0 + Q4_ASYM), verify parity, save ETQ41000.
  diagnostic="${scratch}/save-${run}.stderr"
  timeout --foreground --kill-after=5s 60s \
    env W4A32_AOT_PHASE=save W4A32_AOT_DIR="${scratch}/run-${run}" \
    "${scratch}/out-${run}/test_w4a32_aot" \
    >"${scratch}/save-${run}.stdout" 2>"${diagnostic}" || {
      cat "${scratch}/save-${run}.stdout" >&2
      die "compiled W4A32 AOT save process failed on run ${run}"
    }
  [[ ! -s "${diagnostic}" ]] || die "compiled W4A32 AOT save emitted stderr"

  # Process 2: fresh OS process reloads ETQ41000 containers and verifies parity + negatives.
  diagnostic="${scratch}/reload-${run}.stderr"
  timeout --foreground --kill-after=5s 60s \
    env W4A32_AOT_PHASE=reload W4A32_AOT_DIR="${scratch}/run-${run}" \
    "${scratch}/out-${run}/test_w4a32_aot" \
    >"${scratch}/reload-${run}.stdout" 2>"${diagnostic}" || {
      cat "${scratch}/reload-${run}.stdout" >&2
      die "compiled W4A32 AOT reload process failed on run ${run}"
    }
  [[ ! -s "${diagnostic}" ]] || die "compiled W4A32 AOT reload emitted stderr"

  cat "${scratch}/save-${run}.stdout" "${scratch}/reload-${run}.stdout" \
    >"${scratch}/combined-${run}.stdout"
done
diagnostic=''

readelf -h "${scratch}/out-1/test_w4a32_aot" >/dev/null
cmp "${scratch}/out-1/test_w4a32_aot" "${scratch}/out-2/test_w4a32_aot"
cmp "${scratch}/combined-1.stdout" "${scratch}/combined-2.stdout"

awk '
  $1 == "W4A32" && $2 == "AOT" && $3 == "SAVE" {
    if (NF != 8 || $4 != "f32_bytes=65536" || $5 != "q4_bytes=9216" ||
        $7 != "asym_bytes=9216")
      exit 1
    ++save_seen; next
  }
  $1 == "W4A32" && $2 == "AOT" && $3 == "PARITY" {
    if (NF != 9) exit 1
    ++parity_seen; next
  }
  $1 == "W4A32" && $2 == "AOT" && $3 == "Q4_PACKED_SHA256" {
    if (NF != 7) exit 1
    q4_p = $4 " " $5 " " $6 " " $7; ++q4_p_seen; next
  }
  $1 == "W4A32" && $2 == "AOT" && $3 == "Q4_Y_SHA256" {
    if (NF != 7) exit 1
    q4_y = $4 " " $5 " " $6 " " $7; ++q4_y_seen; next
  }
  $1 == "W4A32" && $2 == "AOT" && $3 == "ASYM_PACKED_SHA256" {
    if (NF != 7) exit 1
    asym_p = $4 " " $5 " " $6 " " $7; ++asym_p_seen; next
  }
  $1 == "W4A32" && $2 == "AOT" && $3 == "ASYM_Y_SHA256" {
    if (NF != 7) exit 1
    asym_y = $4 " " $5 " " $6 " " $7; ++asym_y_seen; next
  }
  $1 == "W4A32" && $2 == "AOT" && $3 == "RELOAD_Q4_PACKED_SHA256" {
    if (NF != 7) exit 1
    r_q4_p = $4 " " $5 " " $6 " " $7; ++r_q4_p_seen; next
  }
  $1 == "W4A32" && $2 == "AOT" && $3 == "RELOAD_Q4_Y_SHA256" {
    if (NF != 7) exit 1
    r_q4_y = $4 " " $5 " " $6 " " $7; ++r_q4_y_seen; next
  }
  $1 == "W4A32" && $2 == "AOT" && $3 == "RELOAD_ASYM_PACKED_SHA256" {
    if (NF != 7) exit 1
    r_asym_p = $4 " " $5 " " $6 " " $7; ++r_asym_p_seen; next
  }
  $1 == "W4A32" && $2 == "AOT" && $3 == "RELOAD_ASYM_Y_SHA256" {
    if (NF != 7) exit 1
    r_asym_y = $4 " " $5 " " $6 " " $7; ++r_asym_y_seen; next
  }
  $1 == "W4A32" && $2 == "AOT" && $3 == "RELOAD" {
    if (NF != 8 || $8 != "parity=PASS") exit 1
    ++reload_seen; next
  }
  $0 == "W4A32 AOT NEGATIVE checks=12 PASS" { ++neg_seen; next }
  { exit 1 }
  END {
    if (save_seen != 1 || parity_seen != 1 || reload_seen != 1 || neg_seen != 1 ||
        q4_p_seen != 1 || q4_y_seen != 1 || asym_p_seen != 1 || asym_y_seen != 1 ||
        r_q4_p_seen != 1 || r_q4_y_seen != 1 || r_asym_p_seen != 1 || r_asym_y_seen != 1)
      exit 1
    if (q4_p != r_q4_p || q4_y != r_q4_y || asym_p != r_asym_p || asym_y != r_asym_y)
      exit 1
  }
' "${scratch}/combined-1.stdout" || {
  cat "${scratch}/combined-1.stdout" >&2
  die "invalid compiled W4A32 AOT output"
}

cat "${scratch}/combined-1.stdout"
elf_sha="$(sha256sum "${scratch}/out-1/test_w4a32_aot")"
printf 'W4A32 AOT ELF SHA-256 %s\n' "${elf_sha%% *}"
printf 'W4A32 AOT two fresh compilations, byte-identical ELF and stdout: PASS\n'
