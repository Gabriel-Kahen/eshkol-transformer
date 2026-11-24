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
mamba_dir="${PROJECT_ROOT}/build/mamba_ssm"

bash scripts/build-mamba-ssm.sh "${mamba_dir}"

scratch="$(mktemp -d "${mamba_dir}/.test.XXXXXX")"
diagnostic=''
cleanup() {
  local status=$?
  if [[ "${status}" != 0 && -n "${diagnostic}" && -s "${diagnostic}" ]]; then
    cat "${diagnostic}" >&2
  fi
  rm -rf -- "${scratch}"
}
trap cleanup EXIT

# 1. Verify byte-identical deterministic archive builds from two fresh dirs.
bash scripts/build-mamba-ssm.sh "${scratch}/det-1" >/dev/null
bash scripts/build-mamba-ssm.sh "${scratch}/det-2" >/dev/null
cmp -- "${scratch}/det-1/libeshkol_transformer_mamba_ssm.a" \
  "${scratch}/det-2/libeshkol_transformer_mamba_ssm.a"
cmp -- "${mamba_dir}/libeshkol_transformer_mamba_ssm.a" \
  "${scratch}/det-1/libeshkol_transformer_mamba_ssm.a"
cmp -- "${scratch}/det-1/libeshkol_transformer_mamba_ssm_aot.a" \
  "${scratch}/det-2/libeshkol_transformer_mamba_ssm_aot.a"
cmp -- "${mamba_dir}/libeshkol_transformer_mamba_ssm_aot.a" \
  "${scratch}/det-1/libeshkol_transformer_mamba_ssm_aot.a"
core_sha="$(sha256sum "${mamba_dir}/libeshkol_transformer_mamba_ssm.a" | awk '{print $1}')"
aot_sha="$(sha256sum "${mamba_dir}/libeshkol_transformer_mamba_ssm_aot.a" | awk '{print $1}')"
printf 'MAMBA_SSM deterministic archives SHA-256: core=%s aot=%s PASS\n' "${core_sha}" "${aot_sha}"

common=(-Wall -Wextra -Werror -Wpedantic -ffp-contract=off -fno-fast-math
        -I include -I native)
cflags=(-std=c11 "${common[@]}" -fexcess-precision=standard)
cppflags=(-std=c++17 "${common[@]}")
san=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer
     -fno-sanitize-recover=all)
wrap=(-Wl,--wrap=malloc,--wrap=calloc,--wrap=realloc,--wrap=free)

archives=(
  "${mamba_dir}/libeshkol_transformer_mamba_ssm_aot.a"
  "${mamba_dir}/libeshkol_transformer_mamba_ssm.a"
)

# 2. C++17 header and linkage test.
"${CXX}" "${cppflags[@]}" -O2 tests/mamba_ssm/header_mamba_cpp.cpp \
  "${archives[@]}" -lm -o "${scratch}/header-cpp"
"${scratch}/header-cpp"
printf 'MAMBA_SSM C++17 header and linkage: PASS\n'

# 3. Native C11 suite (normal -O2, zero-heap-allocation trapped via --wrap).
"${CC}" "${cflags[@]}" -O2 tests/mamba_ssm/test_mamba_native.c \
  "${archives[@]}" "${wrap[@]}" -lm -o "${scratch}/test_mamba_native"
"${scratch}/test_mamba_native"

# 4. Independent Python binary64 forward/backward/FD/cache-parity oracle.
python3 tests/mamba_ssm/test_mamba_oracle.py "${scratch}/test_mamba_native"

# 5. Native C11 + C++17 under ASan/UBSan/LSan (detect_leaks=1:halt_on_error=1).
"${CC}" "${cflags[@]}" "${san[@]}" -c native/mamba_ssm.c \
  -o "${scratch}/mamba_ssm.san.o"
"${CC}" "${cflags[@]}" "${san[@]}" -c native/mamba_ssm_eshkol_bridge.c \
  -o "${scratch}/mamba_ssm_eshkol_bridge.san.o"
san_objs=("${scratch}/mamba_ssm_eshkol_bridge.san.o" "${scratch}/mamba_ssm.san.o")

"${CC}" "${cflags[@]}" "${san[@]}" tests/mamba_ssm/test_mamba_native.c \
  "${san_objs[@]}" "${wrap[@]}" -lm -o "${scratch}/test_mamba_native.san"
"${CXX}" "${cppflags[@]}" "${san[@]}" tests/mamba_ssm/header_mamba_cpp.cpp \
  "${san_objs[@]}" -lm -o "${scratch}/header-cpp.san"

ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
  UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  "${scratch}/test_mamba_native.san"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
  UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  "${scratch}/header-cpp.san"
printf 'MAMBA_SSM ASan/UBSan/LSan: PASS\n'

# 6. Compiled Eshkol AOT driver compiled TWICE from fresh dirs (byte-identical ELF).
for run in 1 2; do
  run_dir="${scratch}/esk-run-${run}"
  mkdir -p "${run_dir}/cache" "${run_dir}/out"
  diagnostic="${run_dir}/compile.log"
  if ! (
    cd "${run_dir}/out"
    timeout --foreground --kill-after=5s 120s \
      env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR -u ESHKOL_AOT_MODULE_CACHE_DIR \
      ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="${run_dir}/cache" \
      "${runner}" --strict-types --optimize 0 --no-stdlib \
        -L "${mamba_dir}" \
        --lib eshkol_transformer_mamba_ssm_aot \
        --lib eshkol_transformer_mamba_ssm \
        "${PROJECT_ROOT}/tests/mamba_ssm/test_mamba_ssm_aot.esk" \
        -o mamba-ssm-aot >"${diagnostic}" 2>&1
  ) || [[ ! -x "${run_dir}/out/mamba-ssm-aot" ]] || grep -Fq 'ERROR:' "${diagnostic}"; then
    die "Mamba S6 AOT Eshkol compilation ${run} failed"
  fi
  diagnostic="${run_dir}/run.stderr"
  timeout --foreground --kill-after=5s 60s "${run_dir}/out/mamba-ssm-aot" \
    >"${run_dir}/run.stdout" 2>"${diagnostic}" || {
    cat "${run_dir}/run.stdout" >&2
    die "compiled Mamba S6 AOT executable failed on run ${run}"
  }
  [[ ! -s "${diagnostic}" ]] || die "compiled Mamba S6 AOT emitted stderr"
done
diagnostic=''

elf1="${scratch}/esk-run-1/out/mamba-ssm-aot"
elf2="${scratch}/esk-run-2/out/mamba-ssm-aot"
readelf -h "${elf1}" >/dev/null
cmp -- "${elf1}" "${elf2}"
cmp -- "${scratch}/esk-run-1/run.stdout" "${scratch}/esk-run-2/run.stdout"

awk '
  BEGIN { number = "^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$" }
  $0 == "MAMBA_SSM AOT NEGATIVE PASS" { ++neg; next }
  $1 == "MAMBA_SSM" && $2 == "AOT" && $3 == "FD_GRADIENTS" {
    if (NF != 6 || $4 != "checked=27" || $5 !~ /^max_abs_err=/ || $6 != "PASS")
      exit 1
    ++fd_ok
    next
  }
  $1 == "MAMBA_SSM" && $2 == "AOT" && $3 == "LOSS" {
    if (NF != 5 || $4 !~ /^[0-9]+$/ || $4 != steps || $5 !~ number ||
        $5 + 0 < 0 || (steps && $5 + 0 >= previous))
      exit 1
    if (!steps) initial = $5
    final = $5
    previous = $5 + 0
    ++steps
    next
  }
  $1 == "MAMBA_SSM" && $2 == "AOT" && $3 == "O1_CACHE" {
    if (NF != 9 || $4 != "steps=8" || $5 != "state_bytes_t8=352" ||
        $6 != "state_bytes_t32=352" || $7 != "state_bytes_t64=352" ||
        $8 !~ /^max_abs_diff=/ || $9 != "PASS")
      exit 1
    ++cache_ok
    next
  }
  $1 == "MAMBA_SSM" && $2 == "AOT" && $3 == "RESULT" {
    if (steps != 13 || NF != 12 || $4 != "n=1" || $5 != "t=8" ||
        $6 != "d=8" || $7 != "s=4" || $8 != "k=4" || $9 != "steps=12" ||
        $10 != "initial=" initial || $11 != "final=" final ||
        $12 !~ /^checksum=/)
      exit 1
    ++result_ok
    next
  }
  { exit 1 }
  END {
    if (neg != 1 || fd_ok != 1 || cache_ok != 1 || result_ok != 1 || steps != 13)
      exit 1
  }
' "${scratch}/esk-run-1/run.stdout" || {
  cat "${scratch}/esk-run-1/run.stdout" >&2
  die "invalid Mamba S6 AOT output trace"
}

cat "${scratch}/esk-run-1/run.stdout"
elf_sha="$(sha256sum "${elf1}" | awk '{print $1}')"
printf 'MAMBA_SSM AOT two fresh compilations, byte-identical ELF SHA-256: %s PASS\n' "${elf_sha}"
