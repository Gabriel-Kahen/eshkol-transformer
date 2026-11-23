#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

cd "${PROJECT_ROOT}"
verify_toolchain
export LC_ALL=C

cc="${CC:-$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cc_path)}"
cxx="${CXX:-$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cxx_path)}"
runner="$(eshkol_build_dir)/eshkol-run"

bash scripts/build-ac4.sh
bash scripts/build-mod5.sh
bash scripts/build-ac4-aot.sh

aot_dir="${PROJECT_ROOT}/build/ac4_aot"
ac4_dir="${PROJECT_ROOT}/build/ac4"
mod5_dir="${PROJECT_ROOT}/build/mod5"
scratch="$(mktemp -d "${aot_dir}/.test.XXXXXX")"
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

archives=(
  "${aot_dir}/libeshkol_transformer_ac4_aot.a"
  "${ac4_dir}/libeshkol_ac4.a"
  "${mod5_dir}/libeshkol_transformer_mod5.a"
  "${scratch}/k1/libeshkol_transformer_k1.a"
)

# 1. C++17 header compile and archive linkage check.
"${cxx}" -std=c++17 -Wall -Wextra -Werror -Wpedantic \
  -I include -I native tests/ac4/header_bridge_cpp.cpp \
  "${archives[@]}" -lm -o "${scratch}/header_bridge_cpp"
"${scratch}/header_bridge_cpp"
printf 'AC4 AOT bridge C++17 header and linkage: PASS\n'

# 2. Native C bridge tests in normal (-O2) AND ASan/UBSan/LSan modes.
cflags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -ffp-contract=off
        -fexcess-precision=standard -fno-fast-math -I include -I native)
wrap=(-Wl,--wrap=malloc,--wrap=calloc,--wrap=realloc,--wrap=free)

"${cc}" "${cflags[@]}" -O2 tests/ac4/test_ac4_aot_bridge.c \
  "${archives[@]}" "${wrap[@]}" -lm -o "${scratch}/test_ac4_aot_bridge"
"${scratch}/test_ac4_aot_bridge"
printf 'AC4 AOT bridge normal (-O2) tests: PASS\n'

san=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
"${cc}" "${cflags[@]}" "${san[@]}" tests/ac4/test_ac4_aot_bridge.c \
  native/ac4_eshkol_bridge.c native/ac4_checkpointing.c \
  native/mod5_primitives_provider.c native/kernel_abi.c \
  "${wrap[@]}" -lm -o "${scratch}/test_ac4_aot_bridge.san"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  "${scratch}/test_ac4_aot_bridge.san"
printf 'AC4 AOT bridge ASan/UBSan/LSan tests: PASS\n'

# 3. Compile tests/ac4/test_ac4_aot.esk TWICE from fresh build & cache dirs
#    and verify byte-identical ELFs and identical stdout.
for run in 1 2; do
  mkdir -p "${scratch}/cache-${run}" "${scratch}/out-${run}" "${scratch}/aot-${run}"
  bash scripts/build-ac4-aot.sh "${scratch}/aot-${run}" >/dev/null
  diagnostic="${scratch}/compile-${run}.log"
  timeout --foreground --kill-after=5s 90s \
    env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR -u ESHKOL_AOT_MODULE_CACHE_DIR \
    ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="${scratch}/cache-${run}" \
    "${runner}" --strict-types --optimize 0 --no-stdlib \
      -L "${scratch}/aot-${run}" -L "${ac4_dir}" -L "${mod5_dir}" -L "${scratch}/k1" \
      --lib eshkol_transformer_ac4_aot --lib eshkol_ac4 \
      --lib eshkol_transformer_mod5 --lib eshkol_transformer_k1 \
      tests/ac4/test_ac4_aot.esk \
      -o "${scratch}/out-${run}/test_ac4_aot" >"${diagnostic}" 2>&1
  diagnostic="${scratch}/run-${run}.stderr"
  timeout --foreground --kill-after=5s 60s "${scratch}/out-${run}/test_ac4_aot" \
    >"${scratch}/run-${run}.stdout" 2>"${diagnostic}" || {
    cat "${scratch}/run-${run}.stdout" >&2
    die "compiled AC4 AOT program failed on run ${run}"
  }
  [[ ! -s "${diagnostic}" ]] || die "compiled AC4 AOT program emitted stderr"
done
diagnostic=''

readelf -h "${scratch}/out-1/test_ac4_aot" >/dev/null
cmp "${scratch}/out-1/test_ac4_aot" "${scratch}/out-2/test_ac4_aot"
cmp "${scratch}/run-1.stdout" "${scratch}/run-2.stdout"

awk '
  BEGIN {
    number = "^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$"
    expected_peak["4:NONE"] = 8832;        expected_recomp["4:NONE"] = 0
    expected_peak["4:EVERY_LAYER"] = 3648; expected_recomp["4:EVERY_LAYER"] = 4
    expected_peak["4:SQRT_SEGMENTS"] = 3456; expected_recomp["4:SQRT_SEGMENTS"] = 6
    expected_peak["4:EVERY_K"] = 3264;     expected_recomp["4:EVERY_K"] = 10
    expected_peak["8:NONE"] = 16512;       expected_recomp["8:NONE"] = 0
    expected_peak["8:EVERY_LAYER"] = 4416; expected_recomp["8:EVERY_LAYER"] = 8
    expected_peak["8:SQRT_SEGMENTS"] = 3840; expected_recomp["8:SQRT_SEGMENTS"] = 13
    expected_peak["8:EVERY_K"] = 3264;     expected_recomp["8:EVERY_K"] = 36
  }
  $0 == "AC4 AOT NEGATIVE PASS" { ++negative; next }
  $1 == "AC4" && $2 == "AOT" && $3 == "MEMORY" {
    if (NF != 10) exit 1
    sub(/^L=/, "", $4); layers = $4
    sub(/^dropout=/, "", $5); dropout = $5
    sub(/^policy=/, "", $6); policy = $6
    sub(/^peak_bytes=/, "", $7); peak = $7
    sub(/^workspace_bytes=/, "", $8); ws = $8
    sub(/^replays=/, "", $9); recomp = $9
    sub(/^saved_pct=/, "", $10); saved = $10
    key = layers ":" policy
    if (!(key in expected_peak)) exit 1
    if (peak + 0 != expected_peak[key] || recomp + 0 != expected_recomp[key]) exit 1
    if (ws + 0 <= peak + 0 || saved !~ number) exit 1
    if (policy == "NONE" && saved + 0 != 0) exit 1
    if (policy != "NONE" && saved + 0 <= 50) exit 1
    ++mem[layers ":" dropout ":" policy]
    next
  }
  $1 == "AC4" && $2 == "AOT" && $3 == "TRAIN" {
    if (NF != 9) exit 1
    sub(/^L=/, "", $4); layers = $4
    sub(/^dropout=/, "", $5); dropout = $5
    if ($6 != "steps=8") exit 1
    sub(/^eval_initial=/, "", $7); init_l = $7
    sub(/^eval_final=/, "", $8); fin_l = $8
    if (init_l !~ number || fin_l !~ number || fin_l + 0 >= init_l + 0) exit 1
    if ($9 != "bitwise_parity=PASS") exit 1
    ++train[layers ":" dropout]
    next
  }
  { exit 1 }
  END {
    if (negative != 1) exit 1
    split("4 8", Ls, " ")
    split("0.0 0.2", Ds, " ")
    split("NONE EVERY_LAYER SQRT_SEGMENTS EVERY_K", Ps, " ")
    for (li in Ls) {
      for (di in Ds) {
        if (train[Ls[li] ":" Ds[di]] != 1) exit 1
        for (pi in Ps) {
          if (mem[Ls[li] ":" Ds[di] ":" Ps[pi]] != 1) exit 1
        }
      }
    }
  }
' "${scratch}/run-1.stdout" || {
  cat "${scratch}/run-1.stdout" >&2
  die "invalid compiled AC4 AOT output"
}

cat "${scratch}/run-1.stdout"
digest="$(sha256sum "${scratch}/out-1/test_ac4_aot")"
printf 'AC4 AOT ELF SHA-256 %s\n' "${digest%% *}"
printf 'AC4 AOT two fresh compilations, byte-identical ELF and stdout: PASS\n'
