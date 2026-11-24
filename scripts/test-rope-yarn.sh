#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

cd "${PROJECT_ROOT}"
verify_toolchain
export LC_ALL=C

export CC="${CC:-$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cc_path)}"
export CXX="${CXX:-$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cxx_path)}"
runner="$(eshkol_build_dir)/eshkol-run"

rope_dir="${PROJECT_ROOT}/build/rope_yarn"
bash scripts/build-rope-yarn.sh "${rope_dir}"

scratch="$(mktemp -d "${rope_dir}/.test.XXXXXX")"
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
  "${rope_dir}/libeshkol_transformer_rope_yarn_aot.a"
  "${rope_dir}/libeshkol_transformer_rope_yarn.a"
)

# 1. C++17 header compile and linkage check.
"${CXX}" "${cppflags[@]}" -O2 tests/rope_yarn/header_rope_yarn_cpp.cpp \
  "${archives[@]}" -lm -o "${scratch}/header_rope_yarn_cpp"
"${scratch}/header_rope_yarn_cpp"
printf 'RoPE YaRN C++17 header and linkage: PASS\n'

# 2. Native C11 normal (-O2) tests with allocation trap + Python binary64 oracle.
"${CC}" "${cflags[@]}" -O2 tests/rope_yarn/test_rope_yarn_native.c \
  "${archives[@]}" "${wrap[@]}" -lm -o "${scratch}/test_rope_yarn_native"
"${scratch}/test_rope_yarn_native"
python3 tests/rope_yarn/test_rope_yarn_oracle.py "${scratch}/test_rope_yarn_native"

# 3. Native C11 (with allocation trap) and C++17 tests under ASan/UBSan/LSan.
export ASAN_OPTIONS=detect_leaks=1:halt_on_error=1
export UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1
"${CC}" "${cflags[@]}" "${san[@]}" -c native/rope_yarn.c \
  -o "${scratch}/rope_yarn.san.o"
"${CC}" "${cflags[@]}" "${san[@]}" -c native/rope_yarn_eshkol_bridge.c \
  -o "${scratch}/rope_yarn_eshkol_bridge.san.o"
san_objs=(
  "${scratch}/rope_yarn_eshkol_bridge.san.o"
  "${scratch}/rope_yarn.san.o"
)

"${CC}" "${cflags[@]}" "${san[@]}" tests/rope_yarn/test_rope_yarn_native.c \
  "${san_objs[@]}" "${wrap[@]}" -lm -o "${scratch}/test_rope_yarn_native.san"
"${scratch}/test_rope_yarn_native.san"

"${CXX}" "${cppflags[@]}" "${san[@]}" tests/rope_yarn/header_rope_yarn_cpp.cpp \
  "${san_objs[@]}" -lm -o "${scratch}/header_rope_yarn_cpp.san"
"${scratch}/header_rope_yarn_cpp.san"
printf 'RoPE YaRN ASan/UBSan/LSan tests: PASS\n'

# 4. Compile tests/rope_yarn/test_rope_yarn_aot.esk TWICE from fresh dirs and
#    verify byte-identical ELFs and byte-identical stdout.
for run in 1 2; do
  mkdir -p "${scratch}/cache-${run}" "${scratch}/out-${run}" \
           "${scratch}/libs-${run}"
  bash scripts/build-rope-yarn.sh "${scratch}/libs-${run}" >/dev/null
  diagnostic="${scratch}/compile-${run}.log"
  timeout --foreground --kill-after=5s 90s \
    env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR -u ESHKOL_AOT_MODULE_CACHE_DIR \
    ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="${scratch}/cache-${run}" \
    "${runner}" --strict-types --optimize 0 --no-stdlib \
      -L "${scratch}/libs-${run}" \
      --lib eshkol_transformer_rope_yarn_aot \
      --lib eshkol_transformer_rope_yarn \
      tests/rope_yarn/test_rope_yarn_aot.esk \
      -o "${scratch}/out-${run}/test_rope_yarn_aot" >"${diagnostic}" 2>&1

  diagnostic="${scratch}/run-${run}.stderr"
  timeout --foreground --kill-after=5s 60s \
    "${scratch}/out-${run}/test_rope_yarn_aot" \
    >"${scratch}/run-${run}.stdout" 2>"${diagnostic}" || {
      cat "${scratch}/run-${run}.stdout" >&2
      die "compiled RoPE YaRN AOT executable failed on run ${run}"
    }
  [[ ! -s "${diagnostic}" ]] || die "compiled RoPE YaRN AOT emitted stderr"
done
diagnostic=''

readelf -h "${scratch}/out-1/test_rope_yarn_aot" >/dev/null
cmp "${scratch}/out-1/test_rope_yarn_aot" "${scratch}/out-2/test_rope_yarn_aot"
cmp "${scratch}/run-1.stdout" "${scratch}/run-2.stdout"

awk '
  $0 == "ROPE YARN AOT NEGATIVE checks=8 PASS" { ++neg_ok; next }
  $1 == "ROPE" && $2 == "YARN" && $3 == "AOT" && $4 == "FREQS" {
    if (NF != 10 || $5 != "modes=4" || $6 !~ /^yarn_mscale=/ ||
        $7 !~ /^yarn_f0=/ || $8 !~ /^yarn_f3=/ || $9 !~ /^yarn_f7=/ ||
        $10 != "PASS")
      exit 1
    ++freqs_ok
    next
  }
  $1 == "ROPE" && $2 == "YARN" && $3 == "AOT" && $4 == "INVARIANCE" {
    if (NF != 8 || $5 != "checked=8" || $6 != "max_pos=8192" ||
        $7 !~ /^max_abs_err=/ || $8 != "PASS")
      exit 1
    ++inv_ok
    next
  }
  $1 == "ROPE" && $2 == "YARN" && $3 == "AOT" && $4 == "FD_GRADIENTS" {
    if (NF != 7 || $5 != "checked=48" || $6 !~ /^max_abs_err=/ ||
        $7 != "PASS")
      exit 1
    ++fd_ok
    next
  }
  $0 == "ROPE YARN AOT SUMMARY modes=4 layouts=2 partial_rotary=1 max_pos=8192 PASS" {
    ++summary_ok
    next
  }
  { exit 1 }
  END {
    if (neg_ok != 1 || freqs_ok != 1 || inv_ok != 1 || fd_ok != 1 || summary_ok != 1)
      exit 1
  }
' "${scratch}/run-1.stdout" || {
  cat "${scratch}/run-1.stdout" >&2
  die "invalid compiled RoPE YaRN AOT output"
}

cat "${scratch}/run-1.stdout"
elf_sha="$(sha256sum "${scratch}/out-1/test_rope_yarn_aot")"
printf 'RoPE YaRN AOT ELF SHA-256 %s\n' "${elf_sha%% *}"
printf 'RoPE YaRN AOT two fresh compilations, byte-identical ELF and stdout: PASS\n'
