#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

cd "${PROJECT_ROOT}"
verify_toolchain
export LC_ALL=C
export CC="${CC:-$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cc_path)}"
export CXX="${CXX:-$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cxx_path)}"

runner="$(eshkol_build_dir)/eshkol-run"
rope_dir="${PROJECT_ROOT}/build/mod5_rope"
mod5_dir="${PROJECT_ROOT}/build/mod5"

bash scripts/build-mod5-rope.sh
bash scripts/build-mod5.sh

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

bash scripts/build-k1.sh "${scratch}/k1"

common=(-Wall -Wextra -Werror -Wpedantic -ffp-contract=off -fno-fast-math
        -I include -I native)
cflags=(-std=c11 "${common[@]}" -fexcess-precision=standard)
cppflags=(-std=c++17 "${common[@]}")
san=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer
     -fno-sanitize-recover=all)
wrap=(-Wl,--wrap=malloc,--wrap=calloc,--wrap=realloc,--wrap=free)

# Build MOD5 core bridge for 2-layer AOT composition.
"${CC}" "${cflags[@]}" -O2 -c native/mod5_eshkol_bridge.c \
  -o "${scratch}/mod5_eshkol_bridge.o"

# 1. C++17 header and linkage check.
"${CXX}" "${cppflags[@]}" -O2 tests/mod5_rope/header_rope_cpp.cpp \
  "${rope_dir}/mod5_rope_eshkol_bridge.o" \
  "${rope_dir}/libeshkol_transformer_mod5_rope.a" \
  "${scratch}/k1/libeshkol_transformer_k1.a" -lm -o "${scratch}/header-cpp"
"${scratch}/header-cpp"
printf 'MOD5 RoPE C++17 header and linkage: PASS\n'

# 2. Native C11 suite (normal -O2, allocation-trapped dispatch).
"${CC}" "${cflags[@]}" -O2 tests/mod5_rope/test_mod5_rope_native.c \
  "${rope_dir}/mod5_rope_eshkol_bridge.o" \
  "${rope_dir}/libeshkol_transformer_mod5_rope.a" \
  "${scratch}/k1/libeshkol_transformer_k1.a" "${wrap[@]}" -lm \
  -o "${scratch}/test_mod5_rope_native"
"${scratch}/test_mod5_rope_native"

# 3. Independent binary64 Python oracle.
python3 tests/mod5_rope/test_mod5_rope_oracle.py "${scratch}/test_mod5_rope_native"

# 4. Native C11 + C++17 under ASan/UBSan/LSan.
for src in kernel_abi mod5_rope_provider mod5_rope_eshkol_bridge; do
  "${CC}" "${cflags[@]}" "${san[@]}" -c "native/${src}.c" \
    -o "${scratch}/${src}.san.o"
done
san_objs=("${scratch}/mod5_rope_eshkol_bridge.san.o"
          "${scratch}/mod5_rope_provider.san.o"
          "${scratch}/kernel_abi.san.o")
"${CC}" "${cflags[@]}" "${san[@]}" tests/mod5_rope/test_mod5_rope_native.c \
  "${san_objs[@]}" "${wrap[@]}" -lm -o "${scratch}/test_mod5_rope_native.san"
"${CXX}" "${cppflags[@]}" "${san[@]}" tests/mod5_rope/header_rope_cpp.cpp \
  "${san_objs[@]}" -lm -o "${scratch}/header-cpp.san"

ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
  UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  "${scratch}/test_mod5_rope_native.san"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
  UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  "${scratch}/header-cpp.san"
printf 'MOD5 RoPE ASan/UBSan/LSan: PASS\n'

# 5. Compiled Eshkol AOT 2-layer Llama-style decoder (compiled twice from fresh dirs).
for run in 1 2; do
  mkdir "${scratch}/cache-${run}" "${scratch}/link-${run}"
  # The pinned driver removes positional object inputs after linking.
  cp "${rope_dir}/mod5_rope_eshkol_bridge.o" "${scratch}/link-${run}/"
  cp "${scratch}/mod5_eshkol_bridge.o" "${scratch}/link-${run}/"
  diagnostic="${scratch}/compile-${run}.log"
  timeout --foreground --kill-after=5s 90s \
    env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR -u ESHKOL_AOT_MODULE_CACHE_DIR \
    ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="${scratch}/cache-${run}" \
    "${runner}" --strict-types --optimize 0 --no-stdlib \
      -L "${rope_dir}" -L "${mod5_dir}" -L "${scratch}/k1" \
      --lib eshkol_transformer_mod5_rope \
      --lib eshkol_transformer_mod5 \
      --lib eshkol_transformer_k1 \
      tests/mod5_rope/test_mod5_rope_aot.esk \
      "${scratch}/link-${run}/mod5_rope_eshkol_bridge.o" \
      "${scratch}/link-${run}/mod5_eshkol_bridge.o" \
      -o "${scratch}/rope-aot-${run}" >"${diagnostic}" 2>&1
  diagnostic="${scratch}/run-${run}.stderr"
  timeout --foreground --kill-after=5s 60s "${scratch}/rope-aot-${run}" \
    >"${scratch}/run-${run}.stdout" 2>"${diagnostic}" || {
    cat "${scratch}/run-${run}.stdout" >&2
    die "compiled RoPE AOT executable failed on run ${run}"
  }
  [[ ! -s "${diagnostic}" ]] || die "compiled RoPE AOT emitted stderr"
done
diagnostic=''

readelf -h "${scratch}/rope-aot-1" >/dev/null
cmp "${scratch}/rope-aot-1" "${scratch}/rope-aot-2"
cmp "${scratch}/run-1.stdout" "${scratch}/run-2.stdout"

awk '
  BEGIN { number = "^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$" }
  $0 == "MOD5 ROPE AOT malformed statuses PASS" { ++negative; next }
  $1 == "MOD5" && $2 == "ROPE" && $3 == "AOT" && $4 == "FD_GRADIENTS" {
    if (NF != 7 || $5 != "checked=42" || $6 !~ /^max_abs_err=/ || $7 != "PASS")
      exit 1
    ++fd_ok
    next
  }
  $1 == "MOD5" && $2 == "ROPE" && $3 == "AOT" && $4 == "LOSS" {
    if (NF != 6 || $5 !~ /^[0-9]+$/ || $5 != steps || $6 !~ number ||
        $6 + 0 < 0 || (steps && $6 + 0 >= previous))
      exit 1
    if (!steps) initial = $6
    final = $6
    previous = $6 + 0
    ++steps
    next
  }
  $1 == "MOD5" && $2 == "ROPE" && $3 == "AOT" && $4 == "RESULT" {
    if (steps != 13 || NF != 11 || $5 != "layers=2" || $6 != "hq=4" ||
        $7 != "hkv=2" || $8 != "steps=12" || $9 != "initial=" initial ||
        $10 != "final=" final || $11 !~ /^checksum=/)
      exit 1
    ++result_ok
    next
  }
  { exit 1 }
  END {
    if (negative != 1 || fd_ok != 1 || result_ok != 1 || steps != 13)
      exit 1
  }
' "${scratch}/run-1.stdout" || {
  cat "${scratch}/run-1.stdout" >&2
  die "invalid RoPE AOT output trace"
}

cat "${scratch}/run-1.stdout"
digest="$(sha256sum "${scratch}/rope-aot-1")"
printf 'MOD5 RoPE AOT ELF SHA-256 %s\n' "${digest%% *}"
printf 'MOD5 RoPE AOT two fresh compilations, byte-identical ELF and stdout: PASS\n'
