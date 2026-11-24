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

mtp_dir="${PROJECT_ROOT}/build/mtp_head"
bash scripts/build-mtp-head.sh "${mtp_dir}"

scratch="$(mktemp -d "${mtp_dir}/.test.XXXXXX")"
diagnostic=''
cleanup() {
  local status=$?
  if [[ "${status}" != 0 && -n "${diagnostic}" && -s "${diagnostic}" ]]; then
    cat "${diagnostic}" >&2
  fi
  rm -rf -- "${scratch}"
}
trap cleanup EXIT

# 1. Verify byte-identical deterministic archives from two independent dirs.
bash scripts/build-mtp-head.sh "${scratch}/det-1" >/dev/null
bash scripts/build-mtp-head.sh "${scratch}/det-2" >/dev/null
cmp -- "${scratch}/det-1/libeshkol_transformer_mtp_head.a" \
       "${scratch}/det-2/libeshkol_transformer_mtp_head.a"
cmp -- "${mtp_dir}/libeshkol_transformer_mtp_head.a" \
       "${scratch}/det-1/libeshkol_transformer_mtp_head.a"
cmp -- "${scratch}/det-1/libeshkol_transformer_mtp_head_aot.a" \
       "${scratch}/det-2/libeshkol_transformer_mtp_head_aot.a"
cmp -- "${mtp_dir}/libeshkol_transformer_mtp_head_aot.a" \
       "${scratch}/det-1/libeshkol_transformer_mtp_head_aot.a"
core_sha="$(sha256sum "${mtp_dir}/libeshkol_transformer_mtp_head.a" | awk '{print $1}')"
aot_sha="$(sha256sum "${mtp_dir}/libeshkol_transformer_mtp_head_aot.a" | awk '{print $1}')"
printf 'MTP_HEAD deterministic archives SHA-256: core=%s aot=%s PASS\n' \
  "${core_sha}" "${aot_sha}"

common=(-Wall -Wextra -Werror -Wpedantic -ffp-contract=off -fno-fast-math
        -I include -I native)
cflags=(-std=c11 "${common[@]}" -fexcess-precision=standard)
cppflags=(-std=c++17 "${common[@]}")
san=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer
     -fno-sanitize-recover=all)

archives=(
  "${mtp_dir}/libeshkol_transformer_mtp_head_aot.a"
  "${mtp_dir}/libeshkol_transformer_mtp_head.a"
)

# 2. C++17 header compile and archive linkage check.
"${CXX}" "${cppflags[@]}" -O2 tests/mtp_head/header_mtp_cpp.cpp \
  "${archives[@]}" -lm -o "${scratch}/header_mtp_cpp"
"${scratch}/header_mtp_cpp"
printf 'MTP_HEAD C++17 header and linkage: PASS\n'

# 3. Native C11 suite (-O2) + independent Python binary64 oracle.
"${CC}" "${cflags[@]}" -O2 tests/mtp_head/test_mtp_head_native.c \
  "${archives[@]}" -lm -o "${scratch}/test_mtp_head_native"
"${scratch}/test_mtp_head_native"
python3 tests/mtp_head/test_mtp_head_oracle.py "${scratch}/test_mtp_head_native"

# 4. Native C11 & C++17 under ASan/UBSan/LSan (detect_leaks=1:halt_on_error=1).
export ASAN_OPTIONS=detect_leaks=1:halt_on_error=1
export UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1

"${CC}" "${cflags[@]}" "${san[@]}" -c native/mtp_head.c \
  -o "${scratch}/mtp_head.san.o"
"${CC}" "${cflags[@]}" "${san[@]}" -c native/mtp_head_eshkol_bridge.c \
  -o "${scratch}/mtp_head_eshkol_bridge.san.o"
san_objs=("${scratch}/mtp_head_eshkol_bridge.san.o" "${scratch}/mtp_head.san.o")

"${CC}" "${cflags[@]}" "${san[@]}" tests/mtp_head/test_mtp_head_native.c \
  "${san_objs[@]}" -lm -o "${scratch}/test_mtp_head_native.san"
"${CXX}" "${cppflags[@]}" "${san[@]}" tests/mtp_head/header_mtp_cpp.cpp \
  "${san_objs[@]}" -lm -o "${scratch}/header_mtp_cpp.san"

"${scratch}/test_mtp_head_native.san" >/dev/null
"${scratch}/header_mtp_cpp.san"
printf 'MTP_HEAD ASan/UBSan/LSan: PASS\n'

# 5. Compiled Eshkol AOT driver compiled TWICE from fresh dirs (byte-identical ELF).
for run in 1 2; do
  run_dir="${scratch}/esk-run-${run}"
  mkdir -p "${run_dir}/cache" "${run_dir}/out" "${run_dir}/libs"
  bash scripts/build-mtp-head.sh "${run_dir}/libs" >/dev/null
  diagnostic="${run_dir}/compile.log"
  if ! (
    cd "${run_dir}/out"
    timeout --foreground --kill-after=5s 120s \
      env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR -u ESHKOL_AOT_MODULE_CACHE_DIR \
      ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="${run_dir}/cache" \
      "${runner}" --strict-types --optimize 0 --no-stdlib \
        -L "${run_dir}/libs" \
        --lib eshkol_transformer_mtp_head_aot \
        --lib eshkol_transformer_mtp_head \
        "${PROJECT_ROOT}/tests/mtp_head/test_mtp_head_aot.esk" \
        -o mtp-head-aot >"${diagnostic}" 2>&1
  ) || [[ ! -x "${run_dir}/out/mtp-head-aot" ]] || grep -Fq 'ERROR:' "${diagnostic}"; then
    die "MTP_HEAD AOT Eshkol compilation ${run} failed"
  fi
done
diagnostic=''

elf1="${scratch}/esk-run-1/out/mtp-head-aot"
elf2="${scratch}/esk-run-2/out/mtp-head-aot"
readelf -h "${elf1}" >/dev/null
cmp -- "${elf1}" "${elf2}"
elf_sha="$(sha256sum "${elf1}" | awk '{print $1}')"
printf 'MTP_HEAD AOT two fresh compilations, byte-identical ELF SHA-256: %s PASS\n' "${elf_sha}"

# 6. Execute compiled AOT ELF: train12 (twice to verify deterministic output).
for run in 1 2; do
  out_f="${scratch}/train12-${run}.stdout"
  err_f="${scratch}/train12-${run}.stderr"
  diagnostic="${err_f}"
  timeout --foreground --kill-after=5s 60s "${elf1}" --mode train12 \
    >"${out_f}" 2>"${err_f}" || {
    cat "${out_f}" >&2 || true
    die "MTP AOT train12 run ${run} failed"
  }
  [[ ! -s "${err_f}" ]] || die "MTP AOT train12 run ${run} emitted stderr"
done
diagnostic=''
cmp -- "${scratch}/train12-1.stdout" "${scratch}/train12-2.stdout"

# 7. Execute 6+6 fresh-process ETMT1000 checkpoint save/reload + tamper checks.
ckpt_dir="${scratch}/ckpt"
mkdir -p "${ckpt_dir}/neg"
ckpt_file="${ckpt_dir}/step6.etmt"

diagnostic="${scratch}/resume6.stderr"
timeout --foreground --kill-after=5s 60s "${elf1}" \
  --mode resume6 --ckpt "${ckpt_file}" \
  >"${scratch}/resume6.stdout" 2>"${diagnostic}" || {
  cat "${scratch}/resume6.stdout" >&2 || true
  die "MTP AOT resume6 failed"
}
[[ ! -s "${diagnostic}" ]] || die "MTP AOT resume6 emitted stderr"

diagnostic="${scratch}/resume6-cont.stderr"
timeout --foreground --kill-after=5s 60s "${elf1}" \
  --mode resume6-cont --ckpt "${ckpt_file}" --tmp-dir "${ckpt_dir}/neg" \
  >"${scratch}/resume6-cont.stdout" 2>"${diagnostic}" || {
  cat "${scratch}/resume6-cont.stdout" >&2 || true
  die "MTP AOT resume6-cont failed"
}
[[ ! -s "${diagnostic}" ]] || die "MTP AOT resume6-cont emitted stderr"
diagnostic=''

# Validate full AOT stdout structure and bitwise 6+6 == 12 state SHA-256 parity.
train12_sha="$(awk '$1=="MTP" && $2=="AOT" && $3=="RESULT" {
  for (i=1; i<=NF; ++i) if ($i ~ /^sha256=/) { sub(/^sha256=/, "", $i); print $i }
}' "${scratch}/train12-1.stdout")"
resume12_sha="$(awk '$1=="MTP" && $2=="AOT" && $3=="RESUME_RESULT" {
  for (i=1; i<=NF; ++i) if ($i ~ /^sha256=/) { sub(/^sha256=/, "", $i); print $i }
}' "${scratch}/resume6-cont.stdout")"

[[ "${#train12_sha}" == 64 ]] || die "missing train12 sha256"
[[ "${train12_sha}" == "${resume12_sha}" ]] || {
  printf 'train12=%s resume6+6=%s\n' "${train12_sha}" "${resume12_sha}" >&2
  die "6+6 fresh-process ETMT1000 checkpoint resume SHA-256 mismatch"
}
grep -Fq 'MTP AOT NEGATIVE PASS' "${scratch}/train12-1.stdout"
grep -Fq 'MTP AOT FD_CHECK ' "${scratch}/train12-1.stdout"
grep -Fq 'MTP AOT SPEC_VERIFY ' "${scratch}/train12-1.stdout"
grep -Fq 'MTP AOT CKPT_NEGATIVE PASS' "${scratch}/resume6-cont.stdout"

cat "${scratch}/train12-1.stdout"
cat "${scratch}/resume6.stdout"
cat "${scratch}/resume6-cont.stdout"
printf 'MTP_HEAD AOT 6+6 fresh-process ETMT1000 bitwise resume parity SHA-256: %s PASS\n' \
  "${train12_sha}"
