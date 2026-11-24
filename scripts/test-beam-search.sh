#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

cd "${PROJECT_ROOT}"
for command in ar cmp diff nm readelf sha256sum timeout; do
  require_command "${command}"
done
verify_toolchain
export LC_ALL=C

export CC="${CC:-$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cc_path)}"
export CXX="${CXX:-$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cxx_path)}"

runner="$(eshkol_build_dir)/eshkol-run"
beam_dir="${PROJECT_ROOT}/build/beam_search"

bash scripts/build-beam-search.sh "${beam_dir}"

scratch="$(mktemp -d "${beam_dir}/.test.XXXXXX")"
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
  "${beam_dir}/libeshkol_transformer_beam_search_aot.a"
  "${beam_dir}/libeshkol_transformer_beam_search.a"
)

# 1. C++17 header compile and linkage check.
"${CXX}" "${cppflags[@]}" -O2 tests/beam_search/header_beam_cpp.cpp \
  "${archives[@]}" -lm -o "${scratch}/header_beam_cpp"
"${scratch}/header_beam_cpp"
printf 'BEAM_SEARCH C++17 header and linkage: PASS\n'

# 2. Native C11 suite (normal -O2 with zero-alloc malloc/calloc/realloc/free trap).
"${CC}" "${cflags[@]}" -O2 tests/beam_search/test_beam_search_native.c \
  "${archives[@]}" "${wrap[@]}" -lm -o "${scratch}/test_beam_search_native"
"${scratch}/test_beam_search_native"

# 3. Independent Python binary64 beam-search & logit-processor oracle.
python3 tests/beam_search/test_beam_search_oracle.py "${scratch}/test_beam_search_native"

# 4. Native C11 + C++17 under ASan/UBSan/LSan (detect_leaks=1:halt_on_error=1).
export ASAN_OPTIONS=detect_leaks=1:halt_on_error=1
export UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1

"${CC}" "${cflags[@]}" "${san[@]}" -c native/beam_search.c \
  -o "${scratch}/beam_search.san.o"
"${CC}" "${cflags[@]}" "${san[@]}" -c native/beam_search_eshkol_bridge.c \
  -o "${scratch}/beam_search_eshkol_bridge.san.o"
san_objs=(
  "${scratch}/beam_search_eshkol_bridge.san.o"
  "${scratch}/beam_search.san.o"
)

"${CC}" "${cflags[@]}" "${san[@]}" tests/beam_search/test_beam_search_native.c \
  "${san_objs[@]}" "${wrap[@]}" -lm -o "${scratch}/test_beam_search_native.san"
"${scratch}/test_beam_search_native.san"

"${CXX}" "${cppflags[@]}" "${san[@]}" tests/beam_search/header_beam_cpp.cpp \
  "${san_objs[@]}" -lm -o "${scratch}/header_beam_cpp.san"
"${scratch}/header_beam_cpp.san"
printf 'BEAM_SEARCH ASan/UBSan/LSan (with zero-alloc wrap): PASS\n'

# 5. Compiled Eshkol AOT driver compiled TWICE from fresh dirs (byte-identical ELF).
for run in 1 2; do
  run_dir="${scratch}/esk-run-${run}"
  mkdir -p "${run_dir}/cache" "${run_dir}/out" "${run_dir}/libs"
  bash scripts/build-beam-search.sh "${run_dir}/libs" >/dev/null
  diagnostic="${run_dir}/compile.log"
  if ! (
    cd "${run_dir}/out"
    timeout --foreground --kill-after=5s 120s \
      env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR -u ESHKOL_AOT_MODULE_CACHE_DIR \
      ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="${run_dir}/cache" \
      "${runner}" --strict-types --optimize 0 --no-stdlib \
        -L "${run_dir}/libs" \
        --lib eshkol_transformer_beam_search_aot \
        --lib eshkol_transformer_beam_search \
        "${PROJECT_ROOT}/tests/beam_search/test_beam_search_aot.esk" \
        -o beam-search-aot >"${diagnostic}" 2>&1
  ) || [[ ! -x "${run_dir}/out/beam-search-aot" ]] || grep -Fq 'ERROR:' "${diagnostic}"; then
    die "Beam Search AOT Eshkol compilation ${run} failed"
  fi
  diagnostic="${run_dir}/run.stderr"
  timeout --foreground --kill-after=5s 60s "${run_dir}/out/beam-search-aot" \
    >"${run_dir}/run.stdout" 2>"${diagnostic}" || {
    cat "${run_dir}/run.stdout" >&2
    die "compiled Beam Search AOT executable failed on run ${run}"
  }
  [[ ! -s "${diagnostic}" ]] || die "compiled Beam Search AOT emitted stderr"
done
diagnostic=''

elf1="${scratch}/esk-run-1/out/beam-search-aot"
elf2="${scratch}/esk-run-2/out/beam-search-aot"
readelf -h "${elf1}" >/dev/null
cmp -- "${elf1}" "${elf2}"
cmp -- "${scratch}/esk-run-1/run.stdout" "${scratch}/esk-run-2/run.stdout"

awk '
  $0 == "BEAM_SEARCH AOT TRAP b1_len=4 b4_len=5 gain_gt_3p5=1 PASS" { ++trap_ok; next }
  $0 == "BEAM_SEARCH AOT NGRAM n0_loops=1 n2_repeats=0 n3_repeats=0 PASS" { ++ngram_ok; next }
  $0 == "BEAM_SEARCH AOT MINLEN_WU minlen_gen=5 a0_gen=2 a15_gen=5 PASS" { ++wu_ok; next }
  $0 == "BEAM_SEARCH AOT NEGATIVE PASS" { ++neg_ok; next }
  $0 == "BEAM_SEARCH AOT SUMMARY PASS" { ++sum_ok; next }
  { exit 1 }
  END {
    if (trap_ok != 1 || ngram_ok != 1 || wu_ok != 1 || neg_ok != 1 || sum_ok != 1)
      exit 1
  }
' "${scratch}/esk-run-1/run.stdout" || {
  cat "${scratch}/esk-run-1/run.stdout" >&2
  die "invalid Beam Search AOT output trace"
}

cat "${scratch}/esk-run-1/run.stdout"
elf_sha="$(sha256sum "${elf1}" | awk '{print $1}')"
printf 'BEAM_SEARCH AOT two fresh compilations, byte-identical ELF SHA-256: %s PASS\n' "${elf_sha}"
