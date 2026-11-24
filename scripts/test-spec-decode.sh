#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

cd "${PROJECT_ROOT}"
verify_toolchain
export LC_ALL=C

export CC="${CC:-$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cc_path)}"
export CXX="${CXX:-$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cxx_path)}"
runner="$(eshkol_build_dir)/eshkol-run"

build_dir="${PROJECT_ROOT}/build/spec_decode"
bash scripts/build-spec-decode.sh "${build_dir}"

scratch="$(mktemp -d "${build_dir}/.test.XXXXXX")"
diagnostic=''
cleanup() {
  local status=$?
  if [[ "${status}" != 0 && -n "${diagnostic}" && -s "${diagnostic}" ]]; then
    cat "${diagnostic}" >&2
  fi
  rm -rf -- "${scratch}"
}
trap cleanup EXIT

archives=(
  "${build_dir}/libeshkol_transformer_spec_decode_aot.a"
  "${build_dir}/libeshkol_transformer_spec_decode.a"
)

# 1. C++17 header compile and archive linkage check.
"${CXX}" -std=c++17 -Wall -Wextra -Werror -Wpedantic \
  -I include -I native tests/spec_decode/header_spec_cpp.cpp \
  "${archives[@]}" -lm -o "${scratch}/header_spec_cpp"
"${scratch}/header_spec_cpp"
printf 'SPEC_DECODE C++17 header and linkage: PASS\n'

# 2. Native C11 tests in normal (-O2, with alloc-trap during dispatch) AND
#    ASan/UBSan/LSan modes.
cflags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -ffp-contract=off
        -fexcess-precision=standard -fno-fast-math -I include -I native)
wrap=(-Wl,--wrap=malloc,--wrap=calloc,--wrap=realloc,--wrap=free)

"${CC}" "${cflags[@]}" -O2 -DET_SPEC_WRAP_ALLOC \
  tests/spec_decode/test_spec_decode_native.c \
  "${archives[@]}" "${wrap[@]}" -lm -o "${scratch}/test_spec_decode_native"
"${scratch}/test_spec_decode_native"
printf 'SPEC_DECODE native C11 normal (-O2, zero-alloc dispatch): PASS\n'

san=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
"${CC}" "${cflags[@]}" "${san[@]}" \
  tests/spec_decode/test_spec_decode_native.c \
  native/spec_decode_eshkol_bridge.c native/spec_decode.c \
  -lm -o "${scratch}/test_spec_decode_native.san"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  "${scratch}/test_spec_decode_native.san"
printf 'SPEC_DECODE native C11 ASan/UBSan/LSan: PASS\n'

# 3. Independent Python analytical & empirical oracle against compiled C engine.
"${CC}" "${cflags[@]}" -O2 -fPIC -shared \
  native/spec_decode.c -lm -o "${scratch}/libeshkol_transformer_spec_decode.so"
python3 tests/spec_decode/test_spec_decode_oracle.py \
  "${scratch}/libeshkol_transformer_spec_decode.so"
printf 'SPEC_DECODE Python analytical & empirical oracle: PASS\n'

# 4. Compile tests/spec_decode/test_spec_decode_aot.esk TWICE from fresh dirs
#    and verify byte-identical ELFs and identical stdout.
for run in 1 2; do
  mkdir -p "${scratch}/cache-${run}" "${scratch}/out-${run}" "${scratch}/lib-${run}"
  bash scripts/build-spec-decode.sh "${scratch}/lib-${run}" >/dev/null
  diagnostic="${scratch}/compile-${run}.log"
  timeout --foreground --kill-after=5s 90s \
    env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR -u ESHKOL_AOT_MODULE_CACHE_DIR \
    ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="${scratch}/cache-${run}" \
    "${runner}" --strict-types --optimize 0 --no-stdlib \
      -L "${scratch}/lib-${run}" \
      --lib eshkol_transformer_spec_decode_aot \
      --lib eshkol_transformer_spec_decode \
      tests/spec_decode/test_spec_decode_aot.esk \
      -o "${scratch}/out-${run}/test_spec_decode_aot" >"${diagnostic}" 2>&1
  diagnostic="${scratch}/run-${run}.stderr"
  timeout --foreground --kill-after=5s 60s \
    "${scratch}/out-${run}/test_spec_decode_aot" \
    >"${scratch}/run-${run}.stdout" 2>"${diagnostic}" || {
    cat "${scratch}/run-${run}.stdout" >&2
    die "compiled SPEC_DECODE AOT program failed on run ${run}"
  }
  [[ ! -s "${diagnostic}" ]] || die "compiled SPEC_DECODE AOT program emitted stderr"
done
diagnostic=''

readelf -h "${scratch}/out-1/test_spec_decode_aot" >/dev/null
cmp "${scratch}/out-1/test_spec_decode_aot" "${scratch}/out-2/test_spec_decode_aot"
cmp "${scratch}/run-1.stdout" "${scratch}/run-2.stdout"

awk '
  BEGIN { number = "^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$" }
  $0 == "SPEC AOT NEGATIVE PASS" { ++neg; next }
  $1 == "SPEC" && $2 == "AOT" && $3 == "GREEDY" {
    if (NF != 9 || $9 != "exact_match=PASS") exit 1
    sub(/^mode=/, "", $4); mode = $4
    sub(/^tokens=/, "", $5); toks = $5 + 0
    sub(/^target_calls=/, "", $6); calls = $6 + 0
    sub(/^accepted=/, "", $7); acc = $7 + 0
    sub(/^acc_per_step=/, "", $8); aps = $8
    if (toks != 32 || calls <= 0 || calls >= toks || acc <= 0 || aps !~ number) exit 1
    ++greedy[mode]
    next
  }
  $1 == "SPEC" && $2 == "AOT" && $3 == "STOCHASTIC" {
    if (NF != 8 || $8 != "replay=PASS") exit 1
    ++stoch
    next
  }
  $1 == "SPEC" && $2 == "AOT" && $3 == "EOS" {
    if (NF != 7 || $7 != "PASS" || $6 != "eos_hit=1") exit 1
    sub(/^mode=/, "", $4); emode = $4
    ++eos[emode]
    next
  }
  { exit 1 }
  END {
    if (neg != 1 || greedy["identical"] != 1 || greedy["partial"] != 1 ||
        stoch != 1 || eos["greedy"] != 1 || eos["stochastic"] != 1)
      exit 1
  }
' "${scratch}/run-1.stdout" || {
  cat "${scratch}/run-1.stdout" >&2
  die "invalid compiled SPEC_DECODE AOT output"
}

cat "${scratch}/run-1.stdout"
digest="$(sha256sum "${scratch}/out-1/test_spec_decode_aot")"
printf 'SPEC_DECODE AOT ELF SHA-256 %s\n' "${digest%% *}"
printf 'SPEC_DECODE AOT two fresh compilations, byte-identical ELF and stdout: PASS\n'
