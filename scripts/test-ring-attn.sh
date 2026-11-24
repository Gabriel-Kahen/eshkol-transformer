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
ring_dir="${PROJECT_ROOT}/build/ring_attn"

bash scripts/build-ring-attn.sh "${ring_dir}"

scratch="$(mktemp -d "${ring_dir}/.test.XXXXXX")"
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
bash scripts/build-ring-attn.sh "${scratch}/det-1" >/dev/null
bash scripts/build-ring-attn.sh "${scratch}/det-2" >/dev/null
cmp -- "${scratch}/det-1/libeshkol_transformer_ring_attn.a" \
  "${scratch}/det-2/libeshkol_transformer_ring_attn.a"
cmp -- "${ring_dir}/libeshkol_transformer_ring_attn.a" \
  "${scratch}/det-1/libeshkol_transformer_ring_attn.a"
cmp -- "${scratch}/det-1/libeshkol_transformer_ring_attn_aot.a" \
  "${scratch}/det-2/libeshkol_transformer_ring_attn_aot.a"
cmp -- "${ring_dir}/libeshkol_transformer_ring_attn_aot.a" \
  "${scratch}/det-1/libeshkol_transformer_ring_attn_aot.a"
core_sha="$(sha256sum "${ring_dir}/libeshkol_transformer_ring_attn.a" | awk '{print $1}')"
aot_sha="$(sha256sum "${ring_dir}/libeshkol_transformer_ring_attn_aot.a" | awk '{print $1}')"
printf 'RING_ATTN deterministic archives SHA-256: core=%s aot=%s PASS\n' "${core_sha}" "${aot_sha}"

common=(-Wall -Wextra -Werror -Wpedantic -ffp-contract=off -fno-fast-math
        -I include -I native)
cflags=(-std=c11 "${common[@]}" -fexcess-precision=standard)
cppflags=(-std=c++17 "${common[@]}")
san=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer
     -fno-sanitize-recover=all)

archives=(
  "${ring_dir}/libeshkol_transformer_ring_attn_aot.a"
  "${ring_dir}/libeshkol_transformer_ring_attn.a"
)

# 2. C++17 header and linkage test.
"${CXX}" "${cppflags[@]}" -O2 tests/ring_attn/header_ring_attn_cpp.cpp \
  "${archives[@]}" -lm -o "${scratch}/header-cpp"
"${scratch}/header-cpp"
printf 'RING_ATTN C++17 header and linkage: PASS\n'

# 3. Native C11 suite (normal -O2).
"${CC}" "${cflags[@]}" -O2 tests/ring_attn/test_ring_attn_native.c \
  "${archives[@]}" -lm -o "${scratch}/test_ring_attn_native"
"${scratch}/test_ring_attn_native"

# 4. Independent Python binary64 dense attention & FD gradient oracle.
python3 tests/ring_attn/test_ring_attn_oracle.py "${scratch}/test_ring_attn_native"

# 5. Native C11 + C++17 under ASan/UBSan/LSan (detect_leaks=1:halt_on_error=1).
"${CC}" "${cflags[@]}" "${san[@]}" -c native/ring_attn.c \
  -o "${scratch}/ring_attn.san.o"
"${CC}" "${cflags[@]}" "${san[@]}" -c native/ring_attn_eshkol_bridge.c \
  -o "${scratch}/ring_attn_eshkol_bridge.san.o"
san_objs=("${scratch}/ring_attn_eshkol_bridge.san.o" "${scratch}/ring_attn.san.o")

"${CC}" "${cflags[@]}" "${san[@]}" tests/ring_attn/test_ring_attn_native.c \
  "${san_objs[@]}" -lm -o "${scratch}/test_ring_attn_native.san"
"${CXX}" "${cppflags[@]}" "${san[@]}" tests/ring_attn/header_ring_attn_cpp.cpp \
  "${san_objs[@]}" -lm -o "${scratch}/header-cpp.san"

ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
  UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  "${scratch}/test_ring_attn_native.san"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
  UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  "${scratch}/header-cpp.san"
printf 'RING_ATTN ASan/UBSan/LSan: PASS\n'

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
        -L "${ring_dir}" \
        --lib eshkol_transformer_ring_attn_aot \
        --lib eshkol_transformer_ring_attn \
        "${PROJECT_ROOT}/tests/ring_attn/test_ring_attn_aot.esk" \
        -o ring-attn-aot >"${diagnostic}" 2>&1
  ) || [[ ! -x "${run_dir}/out/ring-attn-aot" ]] || grep -Fq 'ERROR:' "${diagnostic}"; then
    die "Ring Attention AOT Eshkol compilation ${run} failed"
  fi
  diagnostic="${run_dir}/run.stderr"
  timeout --foreground --kill-after=5s 60s "${run_dir}/out/ring-attn-aot" \
    >"${run_dir}/run.stdout" 2>"${diagnostic}" || {
    cat "${run_dir}/run.stdout" >&2
    die "compiled Ring Attention AOT executable failed on run ${run}"
  }
  [[ ! -s "${diagnostic}" ]] || die "compiled Ring Attention AOT emitted stderr"
done
diagnostic=''

elf1="${scratch}/esk-run-1/out/ring-attn-aot"
elf2="${scratch}/esk-run-2/out/ring-attn-aot"
readelf -h "${elf1}" >/dev/null
cmp -- "${elf1}" "${elf2}"
cmp -- "${scratch}/esk-run-1/run.stdout" "${scratch}/esk-run-2/run.stdout"

awk '
  $0 == "RING_ATTN AOT NEGATIVE PASS" { ++neg; next }
  $1 == "RING_ATTN" && $2 == "AOT" && $3 == "PARITY" {
    if ($NF != "PASS") exit 1
    if ($4 == "world=4" && $5 == "causal=1") {
      if ($6 != "skip=6/16" || $7 != "bskip=6/16") exit 1
      ++w4_causal
    }
    ++parity
    next
  }
  $1 == "RING_ATTN" && $2 == "AOT" && $3 == "FD_GRADIENTS" {
    if (NF != 8 || $4 != "world=4" || $6 != "checked=18" || $8 != "PASS") exit 1
    ++fd
    next
  }
  $0 == "RING_ATTN AOT SUMMARY n=1 t=8 h=4 hkv=2 dh=4 worlds=1,2,4 PASS" {
    ++summary
    next
  }
  { exit 1 }
  END {
    if (neg != 1 || parity != 5 || w4_causal != 1 || fd != 2 || summary != 1)
      exit 1
  }
' "${scratch}/esk-run-1/run.stdout" || {
  cat "${scratch}/esk-run-1/run.stdout" >&2
  die "invalid Ring Attention AOT output trace"
}

cat "${scratch}/esk-run-1/run.stdout"
elf_sha="$(sha256sum "${elf1}" | awk '{print $1}')"
printf 'RING_ATTN AOT two fresh compilations, byte-identical ELF SHA-256: %s PASS\n' "${elf_sha}"
