#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

cd "${PROJECT_ROOT}"
for command in ar cmp python3 readelf sha256sum timeout; do
  require_command "${command}"
done
verify_toolchain
export LC_ALL=C PYTHONDONTWRITEBYTECODE=1

cc="${CC:-$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cc_path)}"
cxx="${CXX:-$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cxx_path)}"
export CC="${cc}"
export CXX="${cxx}"
runner="$(eshkol_build_dir)/eshkol-run"

mod5_dir="${PROJECT_ROOT}/build/mod5"
adamw_dir="${PROJECT_ROOT}/build/mod5_adamw"
bash scripts/build-mod5.sh
bash scripts/build-mod5-adamw.sh

scratch="$(mktemp -d "${adamw_dir}/.test.XXXXXX")"
diagnostic=''
cleanup() {
  local status=$?
  if [[ "${status}" != 0 && -n "${diagnostic}" && -s "${diagnostic}" ]]; then
    cat "${diagnostic}" >&2
  fi
  rm -rf -- "${scratch}"
}
trap cleanup EXIT

# 1. Verify deterministic archive reproducibility.
cp "${adamw_dir}/libeshkol_transformer_mod5_adamw.a" "${scratch}/first.a"
bash scripts/build-mod5-adamw.sh
cmp "${scratch}/first.a" "${adamw_dir}/libeshkol_transformer_mod5_adamw.a"
printf 'MOD5 AdamW archive reproducibility: PASS\n'

bash scripts/build-k1.sh "${scratch}/k1"

common=(-Wall -Wextra -Werror -Wpedantic -ffp-contract=off -fno-fast-math -I include -I native)
cflags=(-std=c11 "${common[@]}" -fexcess-precision=standard)
cppflags=(-std=c++17 "${common[@]}")
san=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)

# 2. C++17 header and linkage check.
"${cxx}" "${cppflags[@]}" tests/mod5/header_adamw_cpp.cpp \
  "${adamw_dir}/libeshkol_transformer_mod5_adamw.a" -lm \
  -o "${scratch}/header-adamw-cpp"
"${scratch}/header-adamw-cpp" "${scratch}/cpp_smoke.etsnap"
printf 'MOD5 AdamW C++17 header and linkage: PASS\n'

# 3. Native C11 unit + negative tests (normal mode).
"${cc}" "${cflags[@]}" -O2 tests/mod5/test_mod5_adamw_native.c \
  "${adamw_dir}/libeshkol_transformer_mod5_adamw.a" -lm \
  -o "${scratch}/test-native"
"${scratch}/test-native"

# 4. Native C11 + C++17 tests under ASan/UBSan/LSan.
"${cc}" "${cflags[@]}" "${san[@]}" -c native/mod5_adamw_state.c \
  -o "${scratch}/mod5_adamw_state.san.o"
"${cc}" "${cflags[@]}" "${san[@]}" tests/mod5/test_mod5_adamw_native.c \
  "${scratch}/mod5_adamw_state.san.o" -lm -o "${scratch}/test-native-san"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  "${scratch}/test-native-san"
"${cxx}" "${cppflags[@]}" "${san[@]}" tests/mod5/header_adamw_cpp.cpp \
  "${scratch}/mod5_adamw_state.san.o" -lm -o "${scratch}/header-cpp-san"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  "${scratch}/header-cpp-san" "${scratch}/cpp_smoke_san.etsnap"
printf 'MOD5 AdamW ASan/UBSan/LSan: PASS\n'

# 5. Independent Python oracle verification.
"${cc}" "${cflags[@]}" -O2 -fPIC -shared native/mod5_adamw_state.c -lm \
  -o "${scratch}/libmod5_adamw_oracle.so"
MOD5_ADAMW_LIB="${scratch}/libmod5_adamw_oracle.so" \
  python3 tests/mod5/test_mod5_adamw_oracle.py -v

# 6. Build MOD5 K1 bridge object for AOT linking.
"${cc}" "${cflags[@]}" -O2 -fPIC -c native/mod5_eshkol_bridge.c \
  -o "${scratch}/mod5_eshkol_bridge.o"
ar rcsD "${scratch}/libmod5_bridge.a" "${scratch}/mod5_eshkol_bridge.o"

# 7. Compile test_mod5_adamw_resume_aot.esk TWICE from fresh dirs/caches ->
#    byte-identical ELFs.
for run in 1 2; do
  mkdir -p "${scratch}/aot-${run}/cache"
  diagnostic="${scratch}/compile-${run}.log"
  if ! (
    cd "${scratch}/aot-${run}"
    timeout --foreground --kill-after=5s 120s \
      env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR -u ESHKOL_AOT_MODULE_CACHE_DIR \
      ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="${scratch}/aot-${run}/cache" \
      "${runner}" --strict-types --optimize 0 --no-stdlib \
        -L "${adamw_dir}" -L "${scratch}" -L "${mod5_dir}" -L "${scratch}/k1" \
        --lib eshkol_transformer_mod5_adamw \
        --lib mod5_bridge \
        --lib eshkol_transformer_mod5 \
        --lib eshkol_transformer_k1 \
        "${PROJECT_ROOT}/tests/mod5/test_mod5_adamw_resume_aot.esk" \
        -o mod5-adamw-aot >"${diagnostic}" 2>&1
  ) || [[ ! -x "${scratch}/aot-${run}/mod5-adamw-aot" ]] || grep -Fq 'ERROR:' "${diagnostic}"; then
    die "MOD5 AdamW AOT compilation ${run} failed"
  fi
done
diagnostic=''

readelf -h "${scratch}/aot-1/mod5-adamw-aot" >/dev/null
cmp "${scratch}/aot-1/mod5-adamw-aot" "${scratch}/aot-2/mod5-adamw-aot"
elf_sha256="$(sha256sum "${scratch}/aot-1/mod5-adamw-aot" | awk '{print $1}')"
printf 'MOD5 AdamW AOT two fresh builds byte-identical ELF: PASS (sha256=%s)\n' "${elf_sha256}"

# 8. Execute 16-step uninterrupted vs 8+8 fresh-process bitwise resume gate.
bin="${scratch}/aot-1/mod5-adamw-aot"
work="${scratch}/work"
mkdir -p "${work}/corpus"

diagnostic="${work}/stderr.log"
"${bin}" --mode write-corpus "${work}/corpus" >"${work}/corpus.stdout" 2>"${diagnostic}"
[[ ! -s "${diagnostic}" ]] || die "write-corpus emitted stderr"
cat "${work}/corpus.stdout"

"${bin}" --mode run-full "${work}/corpus" "${work}/full.etsnap" \
  >"${work}/full.stdout" 2>"${diagnostic}"
[[ ! -s "${diagnostic}" ]] || die "run-full emitted stderr"

# Process 1: train steps 1..8, save midpoint snapshot, exit.
"${bin}" --mode run-part1 "${work}/corpus" "${work}/mid.etsnap" \
  >"${work}/part1.stdout" 2>"${diagnostic}"
[[ ! -s "${diagnostic}" ]] || die "run-part1 emitted stderr"

# Construct corrupt (1-bit flip) and config-mismatch snapshots for negative load gate.
python3 - "${work}/mid.etsnap" "${work}/corrupt.etsnap" <<'PY'
import pathlib, sys
raw = bytearray(pathlib.Path(sys.argv[1]).read_bytes())
raw[260] ^= 0x01
pathlib.Path(sys.argv[2]).write_bytes(raw)
PY
"${bin}" --mode make-mismatch "${work}/mismatch.etsnap" \
  >"${work}/mismatch.stdout" 2>"${diagnostic}"
[[ ! -s "${diagnostic}" ]] || die "make-mismatch emitted stderr"

# Process 2 (fresh OS process!): load midpoint snapshot, verify corrupt/mismatch
# load rejection leaves state untouched, train steps 9..16, save final snapshot.
"${bin}" --mode run-part2 "${work}/corpus" "${work}/mid.etsnap" \
  "${work}/corrupt.etsnap" "${work}/mismatch.etsnap" "${work}/resumed.etsnap" \
  >"${work}/part2.stdout" 2>"${diagnostic}"
[[ ! -s "${diagnostic}" ]] || die "run-part2 emitted stderr"
diagnostic=''

# Compare steps 1..8 between run-full and run-part1 (exact bitwise stdout).
grep -E '^STEP ([1-8]) ' "${work}/full.stdout" >"${work}/full_1_8.txt"
grep -E '^STEP ([1-8]) ' "${work}/part1.stdout" >"${work}/part1_1_8.txt"
cmp "${work}/full_1_8.txt" "${work}/part1_1_8.txt"

# Compare steps 9..16, FINAL_EVAL, and SUMMARY between run-full and Process 2.
grep -E '^(STEP (9|1[0-6]) |FINAL_EVAL |SUMMARY )' "${work}/full.stdout" >"${work}/full_9_16.txt"
grep -E '^(STEP (9|1[0-6]) |FINAL_EVAL |SUMMARY )' "${work}/part2.stdout" >"${work}/part2_9_16.txt"
cmp "${work}/full_9_16.txt" "${work}/part2_9_16.txt"

# Verify byte-identical final snapshot files and print SHA-256s.
cmp "${work}/full.etsnap" "${work}/resumed.etsnap"
full_snap_sha="$(sha256sum "${work}/full.etsnap" | awk '{print $1}')"
resumed_snap_sha="$(sha256sum "${work}/resumed.etsnap" | awk '{print $1}')"
mid_snap_sha="$(sha256sum "${work}/mid.etsnap" | awk '{print $1}')"
[[ "${full_snap_sha}" == "${resumed_snap_sha}" ]] || die "final snapshot SHA-256 mismatch"

grep -Fq 'NEGATIVE_LOAD_REJECTION corrupt=-5 mismatch=-7 state_untouched=1 PASS' "${work}/part2.stdout"

cat "${work}/full.stdout"
grep -E '^(MID_SUMMARY|NEGATIVE_LOAD_REJECTION)' "${work}/part1.stdout" "${work}/part2.stdout"
printf 'MID_SNAPSHOT_SHA256=%s\n' "${mid_snap_sha}"
printf 'FULL_FINAL_SNAPSHOT_SHA256=%s\n' "${full_snap_sha}"
printf 'RESUMED_FINAL_SNAPSHOT_SHA256=%s\n' "${resumed_snap_sha}"
printf 'MOD5 AdamW 16-step vs 8+8 fresh-process bitwise resume gate: PASS\n'
