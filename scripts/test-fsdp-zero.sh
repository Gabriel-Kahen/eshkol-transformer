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

fsdp_dir="${PROJECT_ROOT}/build/fsdp_zero"
bash scripts/build-fsdp-zero.sh "${fsdp_dir}"

scratch="$(mktemp -d "${fsdp_dir}/.test.XXXXXX")"
sock_root="$(mktemp -d /tmp/fsdp_aot_XXXXXX)"
diagnostic=''
cleanup() {
  local status=$?
  if [[ "${status}" != 0 && -n "${diagnostic}" && -s "${diagnostic}" ]]; then
    cat "${diagnostic}" >&2
  fi
  rm -rf -- "${scratch}" "${sock_root}"
}
trap cleanup EXIT

common=(-Wall -Wextra -Werror -Wpedantic -ffp-contract=off -fno-fast-math
        -I include -I native)
cflags=(-std=c11 "${common[@]}" -fexcess-precision=standard)
cppflags=(-std=c++17 "${common[@]}")
san=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer
     -fno-sanitize-recover=all)

archives=(
  "${fsdp_dir}/libeshkol_transformer_fsdp_zero_aot.a"
  "${fsdp_dir}/libeshkol_transformer_fsdp_zero.a"
)

# 1. C++17 header compile and archive linkage check.
"${CXX}" "${cppflags[@]}" -O2 tests/fsdp_zero/header_fsdp_cpp.cpp \
  "${archives[@]}" -lm -o "${scratch}/header_fsdp_cpp"
"${scratch}/header_fsdp_cpp"
printf 'FSDP_ZERO C++17 header and linkage: PASS\n'

# 2. Native C11 normal (-O2) tests + independent Python binary64 oracle.
"${CC}" "${cflags[@]}" -O2 tests/fsdp_zero/test_fsdp_zero_native.c \
  "${archives[@]}" -lm -o "${scratch}/test_fsdp_zero_native"
"${scratch}/test_fsdp_zero_native"
python3 tests/fsdp_zero/test_fsdp_zero_oracle.py "${scratch}/test_fsdp_zero_native"

# 3. Native C11 and C++17 tests under ASan/UBSan/LSan (detect_leaks=1:halt_on_error=1).
export ASAN_OPTIONS=detect_leaks=1:halt_on_error=1
export UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1

"${CC}" "${cflags[@]}" "${san[@]}" \
  tests/fsdp_zero/test_fsdp_zero_native.c \
  native/fsdp_zero_eshkol_bridge.c native/fsdp_zero.c \
  -lm -o "${scratch}/test_fsdp_zero_native.san"
"${scratch}/test_fsdp_zero_native.san"

"${CC}" "${cflags[@]}" "${san[@]}" -c native/fsdp_zero.c \
  -o "${scratch}/fsdp_zero.san.o"
"${CC}" "${cflags[@]}" "${san[@]}" -c native/fsdp_zero_eshkol_bridge.c \
  -o "${scratch}/fsdp_zero_eshkol_bridge.san.o"
"${CXX}" "${cppflags[@]}" "${san[@]}" tests/fsdp_zero/header_fsdp_cpp.cpp \
  "${scratch}/fsdp_zero_eshkol_bridge.san.o" \
  "${scratch}/fsdp_zero.san.o" \
  -lm -o "${scratch}/header_fsdp_cpp.san"
"${scratch}/header_fsdp_cpp.san"
printf 'FSDP_ZERO ASan/UBSan/LSan tests: PASS\n'

# 4. Compile tests/fsdp_zero/test_fsdp_zero_aot.esk TWICE from fresh dirs and
#    verify byte-identical ELFs (--strict-types --optimize 0 --no-stdlib).
for run in 1 2; do
  run_dir="${scratch}/esk-run-${run}"
  mkdir -p "${run_dir}/cache" "${run_dir}/out" "${run_dir}/libs"
  bash scripts/build-fsdp-zero.sh "${run_dir}/libs" >/dev/null
  diagnostic="${run_dir}/compile.log"
  if ! (
    cd "${run_dir}/out"
    timeout --foreground --kill-after=5s 120s \
      env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR -u ESHKOL_AOT_MODULE_CACHE_DIR \
      ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="${run_dir}/cache" \
      "${runner}" --strict-types --optimize 0 --no-stdlib \
        -L "${run_dir}/libs" \
        --lib eshkol_transformer_fsdp_zero_aot \
        --lib eshkol_transformer_fsdp_zero \
        "${PROJECT_ROOT}/tests/fsdp_zero/test_fsdp_zero_aot.esk" \
        -o fsdp-zero-aot >"${diagnostic}" 2>&1
  ) || [[ ! -x "${run_dir}/out/fsdp-zero-aot" ]] || grep -Fq 'ERROR:' "${diagnostic}"; then
    die "FSDP_ZERO AOT Eshkol compilation ${run} failed"
  fi
done
diagnostic=''

elf="${scratch}/esk-run-1/out/fsdp-zero-aot"
readelf -h "${elf}" >/dev/null
cmp -- "${elf}" "${scratch}/esk-run-2/out/fsdp-zero-aot"
elf_sha256="$(sha256sum "${elf}" | awk '{print $1}')"
printf 'FSDP_ZERO AOT two fresh compilations, byte-identical ELF SHA-256: %s PASS\n' "${elf_sha256}"

# Helper to launch W independent OS processes in reversed rank order.
run_group() {
  local label="$1" world="$2" stage="$3" mode="$4"
  shift 4
  local sdir="${sock_root}/${label}"
  local odir="${scratch}/out-${label}"
  mkdir -p "${sdir}" "${odir}"
  local -a pids=()
  local r
  for ((r = world - 1; r >= 0; --r)); do
    local -a cmd=(
      "${elf}" --sock-dir "${sdir}" --rank "${r}" --world "${world}"
      --stage "${stage}" --mode "${mode}" "$@"
    )
    if [[ "${mode}" == "train12" && "${r}" == 0 ]]; then
      cmd+=(--dump-weights "${odir}/weights.bin")
    fi
    timeout --foreground --kill-after=5s 30s "${cmd[@]}" \
      >"${odir}/rank-${r}.stdout" 2>"${odir}/rank-${r}.stderr" &
    pids+=("$!")
  done

  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  for ((r = 0; r < world; ++r)); do
    if [[ -s "${odir}/rank-${r}.stderr" ]]; then
      cat "${odir}/rank-${r}.stderr" >&2
      failed=1
    fi
  done
  if [[ "${failed}" != 0 ]]; then
    for ((r = 0; r < world; ++r)); do
      printf '=== %s rank %d stdout ===\n' "${label}" "${r}" >&2
      cat "${odir}/rank-${r}.stdout" >&2 || true
    done
    die "FSDP_ZERO AOT group ${label} failed"
  fi
}

extract_result_sha() {
  awk '$1 == "FSDP" && $2 == "AOT" && ($3 == "RESULT" || $3 == "RESUME_RESULT") {
    for (i = 1; i <= NF; ++i) {
      if (index($i, "weights_sha256=") == 1) {
        sub(/^weights_sha256=/, "", $i);
        print $i;
      }
    }
  }' "$1"
}

# 5. Live multi-process ZeRO-3 FSDP 12-step training: W=1, W=2, W=4.
for W in 1 2 4; do
  run_group "train-s3-w${W}" "${W}" 3 train12
  sha0="$(extract_result_sha "${scratch}/out-train-s3-w${W}/rank-0.stdout")"
  [[ "${#sha0}" == 64 ]] || die "missing weights SHA-256 for W=${W}"
  for ((r = 1; r < W; ++r)); do
    shar="$(extract_result_sha "${scratch}/out-train-s3-w${W}/rank-${r}.stdout")"
    [[ "${shar}" == "${sha0}" ]] || die "W=${W} rank ${r} weights diverged from rank 0"
  done
  cat "${scratch}/out-train-s3-w${W}/rank-0.stdout"
done

# Cross-verify compiled Eshkol AOT W=1, W=2, W=4 dumped weights against each
# other and the independent Python binary64 reference (<= 1e-6).
python3 tests/fsdp_zero/test_fsdp_zero_oracle.py --check-aot \
  "${scratch}/out-train-s3-w1/weights.bin" \
  "${scratch}/out-train-s3-w2/weights.bin" \
  "${scratch}/out-train-s3-w4/weights.bin"

# 6. Exact 6+6 fresh-process sharded checkpoint resume for Stage 3 (W=1, 2, 4).
for W in 1 2 4; do
  cdir="${scratch}/ckpt-s3-w${W}"
  mkdir -p "${cdir}"
  run_group "resume6-s3-w${W}" "${W}" 3 resume6 --ckpt-dir "${cdir}"
  run_group "resume6-cont-s3-w${W}" "${W}" 3 resume6-cont --ckpt-dir "${cdir}"
  sha_full="$(extract_result_sha "${scratch}/out-train-s3-w${W}/rank-0.stdout")"
  for ((r = 0; r < W; ++r)); do
    sha_res="$(extract_result_sha "${scratch}/out-resume6-cont-s3-w${W}/rank-${r}.stdout")"
    [[ "${sha_res}" == "${sha_full}" ]] || \
      die "W=${W} rank ${r} 6+6 resume diverged: ${sha_res} vs ${sha_full}"
  done
  cat "${scratch}/out-resume6-cont-s3-w${W}/rank-0.stdout"
done
printf 'FSDP_ZERO AOT 6+6 fresh-process checkpoint resume W=1/2/4 bitwise parity: PASS\n'

# 7. Live multi-process atomic non-finite abort (W=4, Stage 3 FSDP).
run_group "nonfinite-s3-w4" 4 3 nonfinite
for r in 0 1 2 3; do
  line="$(cat "${scratch}/out-nonfinite-s3-w4/rank-${r}.stdout")"
  [[ "${line}" == "FSDP AOT NONFINITE world=4 rank=${r} nan_status=-5 inf_status=-5 shard_preserved=1 recovered=1 PASS" ]] || \
    die "unexpected nonfinite output on rank ${r}: ${line}"
done
cat "${scratch}/out-nonfinite-s3-w4/rank-0.stdout"

# 8. Compiled Eshkol AOT negative checks (W=1 local checkpoint/config + W=2 tag mismatch).
mkdir -p "${scratch}/neg-ckpt"
run_group "negative-w1" 1 3 negative --ckpt-dir "${scratch}/neg-ckpt"
cat "${scratch}/out-negative-w1/rank-0.stdout"

run_group "negative-w2" 2 3 negative
for r in 0 1; do
  line="$(cat "${scratch}/out-negative-w2/rank-${r}.stdout")"
  [[ "${line}" == "FSDP AOT NEGATIVE_MISMATCH world=2 rank=${r} status=-2 sticky=-2 PASS" ]] || \
    die "unexpected mismatch output on rank ${r}: ${line}"
done
cat "${scratch}/out-negative-w2/rank-0.stdout"

printf 'FSDP_ZERO FULL SUITE PASS\n'
