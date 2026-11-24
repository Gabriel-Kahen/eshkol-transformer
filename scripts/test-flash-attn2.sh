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
fa2_dir="${PROJECT_ROOT}/build/flash_attn2"

bash scripts/build-flash-attn2.sh "${fa2_dir}"

scratch="$(mktemp -d "${fa2_dir}/.test.XXXXXX")"
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
bash scripts/build-flash-attn2.sh "${scratch}/det-1" >/dev/null
bash scripts/build-flash-attn2.sh "${scratch}/det-2" >/dev/null
cmp -- "${scratch}/det-1/libeshkol_transformer_flash_attn2.a" \
  "${scratch}/det-2/libeshkol_transformer_flash_attn2.a"
cmp -- "${fa2_dir}/libeshkol_transformer_flash_attn2.a" \
  "${scratch}/det-1/libeshkol_transformer_flash_attn2.a"
cmp -- "${scratch}/det-1/libeshkol_transformer_flash_attn2_aot.a" \
  "${scratch}/det-2/libeshkol_transformer_flash_attn2_aot.a"
cmp -- "${fa2_dir}/libeshkol_transformer_flash_attn2_aot.a" \
  "${scratch}/det-1/libeshkol_transformer_flash_attn2_aot.a"
core_sha="$(sha256sum "${fa2_dir}/libeshkol_transformer_flash_attn2.a" | awk '{print $1}')"
aot_sha="$(sha256sum "${fa2_dir}/libeshkol_transformer_flash_attn2_aot.a" | awk '{print $1}')"
printf 'FLASH_ATTN2 deterministic archives SHA-256: core=%s aot=%s PASS\n' "${core_sha}" "${aot_sha}"

common=(-Wall -Wextra -Werror -Wpedantic -ffp-contract=off -fno-fast-math
        -I include -I native)
cflags=(-std=c11 "${common[@]}" -fexcess-precision=standard)
cppflags=(-std=c++17 "${common[@]}")
san=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer
     -fno-sanitize-recover=all)
wrap=(-Wl,--wrap=malloc,--wrap=calloc,--wrap=realloc,--wrap=free)

archives=(
  "${fa2_dir}/libeshkol_transformer_flash_attn2_aot.a"
  "${fa2_dir}/libeshkol_transformer_flash_attn2.a"
)

# 2. C++17 header and linkage test.
"${CXX}" "${cppflags[@]}" -O2 tests/flash_attn2/header_fa2_cpp.cpp \
  "${archives[@]}" -lm -o "${scratch}/header-cpp"
"${scratch}/header-cpp"
printf 'FLASH_ATTN2 C++17 header and linkage: PASS\n'

# 3. Native C11 suite (normal -O2, zero-heap-allocation trapped via --wrap).
"${CC}" "${cflags[@]}" -O2 tests/flash_attn2/test_flash_attn2_native.c \
  "${archives[@]}" "${wrap[@]}" -lm -o "${scratch}/test_flash_attn2_native"
"${scratch}/test_flash_attn2_native"

# 4. Independent Python binary64 dense attention & FD gradient oracle.
python3 tests/flash_attn2/test_flash_attn2_oracle.py "${scratch}/test_flash_attn2_native"

# 5. Native C11 + C++17 under ASan/UBSan/LSan (detect_leaks=1:halt_on_error=1).
"${CC}" "${cflags[@]}" "${san[@]}" -c native/flash_attn2.c \
  -o "${scratch}/flash_attn2.san.o"
"${CC}" "${cflags[@]}" "${san[@]}" -c native/flash_attn2_eshkol_bridge.c \
  -o "${scratch}/flash_attn2_eshkol_bridge.san.o"
san_objs=("${scratch}/flash_attn2_eshkol_bridge.san.o" "${scratch}/flash_attn2.san.o")

"${CC}" "${cflags[@]}" "${san[@]}" tests/flash_attn2/test_flash_attn2_native.c \
  "${san_objs[@]}" "${wrap[@]}" -lm -o "${scratch}/test_flash_attn2_native.san"
"${CXX}" "${cppflags[@]}" "${san[@]}" tests/flash_attn2/header_fa2_cpp.cpp \
  "${san_objs[@]}" -lm -o "${scratch}/header-cpp.san"

ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
  UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  "${scratch}/test_flash_attn2_native.san"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
  UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  "${scratch}/header-cpp.san"
printf 'FLASH_ATTN2 ASan/UBSan/LSan: PASS\n'

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
        -L "${fa2_dir}" \
        --lib eshkol_transformer_flash_attn2_aot \
        --lib eshkol_transformer_flash_attn2 \
        "${PROJECT_ROOT}/tests/flash_attn2/test_flash_attn2_aot.esk" \
        -o flash-attn2-aot >"${diagnostic}" 2>&1
  ) || [[ ! -x "${run_dir}/out/flash-attn2-aot" ]] || grep -Fq 'ERROR:' "${diagnostic}"; then
    die "FlashAttention-2 AOT Eshkol compilation ${run} failed"
  fi
  diagnostic="${run_dir}/run.stderr"
  timeout --foreground --kill-after=5s 60s "${run_dir}/out/flash-attn2-aot" \
    >"${run_dir}/run.stdout" 2>"${diagnostic}" || {
    cat "${run_dir}/run.stdout" >&2
    die "compiled FlashAttention-2 AOT executable failed on run ${run}"
  }
  [[ ! -s "${diagnostic}" ]] || die "compiled FlashAttention-2 AOT emitted stderr"
done
diagnostic=''

elf1="${scratch}/esk-run-1/out/flash-attn2-aot"
elf2="${scratch}/esk-run-2/out/flash-attn2-aot"
readelf -h "${elf1}" >/dev/null
cmp -- "${elf1}" "${elf2}"
cmp -- "${scratch}/esk-run-1/run.stdout" "${scratch}/esk-run-2/run.stdout"

awk '
  BEGIN { number = "^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$" }
  $0 == "FLASH_ATTN2 AOT NEGATIVE PASS" { ++neg; next }
  $1 == "FLASH_ATTN2" && $2 == "AOT" && $3 == "TILE_INVARIANCE" {
    if ($4 != "tiles=(8,8),(16,16),(32,16)" || $5 !~ /^skip8=[1-9][0-9]*\/[0-9]+$/ || $NF != "PASS")
      exit 1
    ++tile_inv
    next
  }
  $1 == "FLASH_ATTN2" && $2 == "AOT" && $3 == "CONSTANT_SCRATCH" {
    if ($4 != "br=16" || $5 != "bc=16" || $6 != "t32=4096" ||
        $7 != "t64=4096" || $8 != "t128=4096" || $9 != "t256=4096" || $10 != "PASS")
      exit 1
    ++const_sb
    next
  }
  $1 == "FLASH_ATTN2" && $2 == "AOT" && $3 == "FD_GRADIENTS" {
    if (NF != 6 || $4 != "checked=24" || $5 !~ /^max_abs_err=/ || $6 != "PASS")
      exit 1
    ++fd_ok
    next
  }
  $1 == "FLASH_ATTN2" && $2 == "AOT" && $3 == "LOSS" {
    if (NF != 5 || $4 !~ /^[0-9]+$/ || $4 != steps || $5 !~ number ||
        $5 + 0 < 0 || (steps && $5 + 0 >= previous))
      exit 1
    if (!steps) initial = $5
    final = $5
    previous = $5 + 0
    ++steps
    next
  }
  $1 == "FLASH_ATTN2" && $2 == "AOT" && $3 == "RESULT" {
    if (steps != 13 || $4 != "b=1" || $5 != "hq=4" || $6 != "hkv=2" ||
        $7 != "t=16" || $8 != "dh=4" || $9 != "br=8" || $10 != "bc=8" ||
        $11 != "win=6" || $12 != "cap=15" || $13 != "steps=12" ||
        $14 != "initial=" initial || $15 != "final=" final ||
        $16 !~ /^checksum=/)
      exit 1
    ++result_ok
    next
  }
  { exit 1 }
  END {
    if (neg != 1 || tile_inv != 1 || const_sb != 1 || fd_ok != 1 ||
        result_ok != 1 || steps != 13)
      exit 1
  }
' "${scratch}/esk-run-1/run.stdout" || {
  cat "${scratch}/esk-run-1/run.stdout" >&2
  die "invalid FlashAttention-2 AOT output trace"
}

cat "${scratch}/esk-run-1/run.stdout"
elf_sha="$(sha256sum "${elf1}" | awk '{print $1}')"
printf 'FLASH_ATTN2 AOT two fresh compilations, byte-identical ELF SHA-256: %s PASS\n' "${elf_sha}"
