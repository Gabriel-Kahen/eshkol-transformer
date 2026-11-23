#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

cd "${PROJECT_ROOT}"
for command in ar cmp nm readelf sha256sum timeout; do
  require_command "${command}"
done
verify_toolchain
export LC_ALL=C

cc="${CC:-$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cc_path)}"
cxx="${CXX:-$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cxx_path)}"
export CC="${cc}"
export CXX="${cxx}"
runner="$(eshkol_build_dir)/eshkol-run"

dist5_dir="${PROJECT_ROOT}/build/dist5"
mod5_dir="${PROJECT_ROOT}/build/mod5"
aot_dir="${PROJECT_ROOT}/build/dist5_aot"

bash scripts/build-dist5.sh
bash scripts/build-mod5.sh
bash scripts/build-dist5-aot.sh

scratch="$(mktemp -d "${aot_dir}/.test.XXXXXX")"
sock_root="$(mktemp -d /tmp/dist5_aot_XXXXXX)"
diagnostic=''
cleanup() {
  local status=$?
  if [[ "${status}" != 0 && -n "${diagnostic}" && -s "${diagnostic}" ]]; then
    cat "${diagnostic}" >&2
  fi
  rm -rf -- "${scratch}" "${sock_root}"
}
trap cleanup EXIT

# Verify byte-identical archive reproducibility from two fresh builds.
mkdir -p "${scratch}/aot-1" "${scratch}/aot-2"
bash scripts/build-dist5-aot.sh "${scratch}/aot-1" >/dev/null
bash scripts/build-dist5-aot.sh "${scratch}/aot-2" >/dev/null
cmp -- "${scratch}/aot-1/libeshkol_transformer_dist5_aot.a" \
  "${scratch}/aot-2/libeshkol_transformer_dist5_aot.a"
cmp -- "${aot_dir}/libeshkol_transformer_dist5_aot.a" \
  "${scratch}/aot-1/libeshkol_transformer_dist5_aot.a"
printf 'DIST5 AOT archive reproducibility: PASS\n'

bash scripts/build-k1.sh "${scratch}/k1"

archives=(
  "${aot_dir}/libeshkol_transformer_dist5_aot.a"
  "${dist5_dir}/libeshkol_dist5.a"
  "${mod5_dir}/libeshkol_transformer_mod5.a"
  "${scratch}/k1/libeshkol_transformer_k1.a"
)

common=(-Wall -Wextra -Werror -Wpedantic -ffp-contract=off -fno-fast-math -I include -I native)
cflags=(-std=c11 "${common[@]}" -fexcess-precision=standard)
cppflags=(-std=c++17 "${common[@]}")
san=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)

# 1. C++17 header compile and archive linkage check.
"${cxx}" "${cppflags[@]}" tests/dist5/header_bridge_cpp.cpp \
  "${archives[@]}" -lm -o "${scratch}/header-bridge-cpp"
"${scratch}/header-bridge-cpp"
printf 'DIST5 AOT bridge C++17 header and linkage: PASS\n'

# 2. Native C11 bridge tests in normal (-O2) mode.
"${cc}" "${cflags[@]}" -O2 tests/dist5/test_dist5_aot_bridge.c \
  "${archives[@]}" -lm -o "${scratch}/bridge-normal"
"${scratch}/bridge-normal"

# 3. Native C11 bridge tests under ASan/UBSan/LSan.
"${cc}" "${cflags[@]}" "${san[@]}" tests/dist5/test_dist5_aot_bridge.c \
  native/dist5_eshkol_bridge.c native/dist5_collectives.c \
  native/mod5_primitives_provider.c native/kernel_abi.c \
  -lm -o "${scratch}/bridge-san"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  "${scratch}/bridge-san"
printf 'DIST5 AOT bridge ASan/UBSan/LSan: PASS\n'

# 4. Compile tests/dist5/test_dist5_aot.esk TWICE from fresh dirs and verify
#    byte-identical ELFs.
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
        -L "${aot_dir}" -L "${dist5_dir}" -L "${mod5_dir}" -L "${scratch}/k1" \
        --lib eshkol_transformer_dist5_aot \
        --lib eshkol_dist5 \
        --lib eshkol_transformer_mod5 \
        --lib eshkol_transformer_k1 \
        "${PROJECT_ROOT}/tests/dist5/test_dist5_aot.esk" \
        -o dist5-aot >"${diagnostic}" 2>&1
  ) || [[ ! -x "${run_dir}/out/dist5-aot" ]] || grep -Fq 'ERROR:' "${diagnostic}"; then
    die "DIST5 AOT Eshkol compilation ${run} failed"
  fi
done
diagnostic=''

elf="${scratch}/esk-run-1/out/dist5-aot"
readelf -h "${elf}" >/dev/null
cmp -- "${elf}" "${scratch}/esk-run-2/out/dist5-aot"
elf_sha256="$(sha256sum "${elf}" | awk '{print $1}')"
printf 'DIST5 AOT two fresh compilations, byte-identical ELF SHA-256: %s PASS\n' "${elf_sha256}"

# Helper to run W independent OS processes (launching in reversed rank order
# for W > 1 to exercise rendezvous retry).
run_group() {
  local label="$1" world="$2" mode="$3" extra_flag="${4:-}" extra_val="${5:-}"
  local sdir="${sock_root}/${label}"
  local odir="${scratch}/out-${label}"
  mkdir -p "${sdir}" "${odir}"
  local -a pids=()
  local r
  for ((r = world - 1; r >= 0; --r)); do
    local -a cmd=("${elf}" --sock-dir "${sdir}" --rank "${r}" --world "${world}" --mode "${mode}")
    if [[ -n "${extra_flag}" ]]; then
      cmd+=("${extra_flag}" "${extra_val}")
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
    die "DIST5 AOT group ${label} failed"
  fi
}

validate_train_rank() {
  local file="$1" world="$2" rank="$3" accum="$4"
  awk -v W="${world}" -v R="${rank}" -v A="${accum}" '
    BEGIN {
      number = "^-?[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$"
      steps = 0
      result = 0
    }
    $1 == "DIST5" && $2 == "AOT" && $3 == "LOSS" {
      if (NF != 7 || $4 != "world=" W || $5 != "rank=" R || $6 != "step=" steps)
        exit 1
      val = $7
      sub(/^loss=/, "", val)
      if (val !~ number || val + 0 < 0) exit 1
      if (steps > 0 && val + 0 >= prev) exit 1
      if (steps == 0) init_s = val
      fin_s = val
      prev = val + 0
      ++steps
      next
    }
    $1 == "DIST5" && $2 == "AOT" && $3 == "RESULT" {
      sha = $10
      if (index(sha, "weights_sha256=") != 1) exit 1
      sub(/^weights_sha256=/, "", sha)
      if (NF != 11 || $4 != "world=" W || $5 != "rank=" R || $6 != "accum=" A ||
          $7 != "steps=10" || $8 != "initial=" init_s || $9 != "final=" fin_s ||
          length(sha) != 64 || sha !~ /^[0-9a-f]+$/ || $11 != "peer_bitwise=PASS")
        exit 1
      ++result
      next
    }
    { exit 1 }
    END {
      if (steps != 11 || result != 1) exit 1
    }
  ' "${file}" || {
    cat "${file}" >&2
    die "invalid training trace in ${file}"
  }
}

extract_sha() {
  awk '$1 == "DIST5" && $2 == "AOT" && $3 == "RESULT" {
    sub(/^weights_sha256=/, "", $10); print $10
  }' "$1"
}

# 5. Live multi-process training runs: W=1 (accum=4 and accum=2), W=2, W=4.
run_group "train-w1-a4" 1 train --accum 4
validate_train_rank "${scratch}/out-train-w1-a4/rank-0.stdout" 1 0 4
sha_w1_a4="$(extract_sha "${scratch}/out-train-w1-a4/rank-0.stdout")"

run_group "train-w1-a2" 1 train --accum 2
validate_train_rank "${scratch}/out-train-w1-a2/rank-0.stdout" 1 0 2
sha_w1_a2="$(extract_sha "${scratch}/out-train-w1-a2/rank-0.stdout")"

run_group "train-w2" 2 train
for r in 0 1; do
  validate_train_rank "${scratch}/out-train-w2/rank-${r}.stdout" 2 "${r}" 2
done
sha_w2_r0="$(extract_sha "${scratch}/out-train-w2/rank-0.stdout")"
sha_w2_r1="$(extract_sha "${scratch}/out-train-w2/rank-1.stdout")"
[[ "${sha_w2_r0}" == "${sha_w2_r1}" ]] || die "W=2 rank weights diverged: ${sha_w2_r0} vs ${sha_w2_r1}"
[[ "${sha_w2_r0}" == "${sha_w1_a2}" ]] || die "W=2 weights diverged from W=1(accum=2): ${sha_w2_r0} vs ${sha_w1_a2}"

run_group "train-w4" 4 train
for r in 0 1 2 3; do
  validate_train_rank "${scratch}/out-train-w4/rank-${r}.stdout" 4 "${r}" 4
done
sha_w4_r0="$(extract_sha "${scratch}/out-train-w4/rank-0.stdout")"
for r in 1 2 3; do
  sha_r="$(extract_sha "${scratch}/out-train-w4/rank-${r}.stdout")"
  [[ "${sha_w4_r0}" == "${sha_r}" ]] || die "W=4 rank ${r} weights diverged: ${sha_w4_r0} vs ${sha_r}"
done
[[ "${sha_w4_r0}" == "${sha_w1_a4}" ]] || die "W=4 weights diverged from W=1(accum=4): ${sha_w4_r0} vs ${sha_w1_a4}"

# Print rank-0 training traces for W=1(accum=4), W=2, W=4.
cat "${scratch}/out-train-w1-a4/rank-0.stdout"
cat "${scratch}/out-train-w2/rank-0.stdout"
cat "${scratch}/out-train-w4/rank-0.stdout"
printf 'DIST5 AOT multi-process training W=1/2/4 bitwise parity: PASS\n'

# 6. Live multi-process atomic non-finite abort test (W=4, NaN/Inf on rank 3).
run_group "nonfinite-w4" 4 nonfinite
nf_sha0=''
for r in 0 1 2 3; do
  line="$(cat "${scratch}/out-nonfinite-w4/rank-${r}.stdout")"
  [[ "${line}" =~ ^DIST5\ AOT\ NONFINITE\ world=4\ rank=${r}\ nan_status=5\ inf_status=5\ weights_preserved=1\ recovered=1\ weights_sha256=([0-9a-f]{64})\ PASS$ ]] \
    || die "unexpected nonfinite output for rank ${r}: ${line}"
  if [[ -z "${nf_sha0}" ]]; then
    nf_sha0="${BASH_REMATCH[1]}"
  else
    [[ "${BASH_REMATCH[1]}" == "${nf_sha0}" ]] \
      || die "nonfinite preserved weights diverged on rank ${r}"
  fi
done
cat "${scratch}/out-nonfinite-w4/rank-0.stdout"
printf 'DIST5 AOT multi-process nonfinite atomic abort W=4: PASS\n'

# 7. Live multi-process tag mismatch test (W=2).
run_group "mismatch-w2" 2 mismatch
for r in 0 1; do
  line="$(cat "${scratch}/out-mismatch-w2/rank-${r}.stdout")"
  [[ "${line}" == "DIST5 AOT MISMATCH world=2 rank=${r} status=2 sticky_status=2 PASS" ]] \
    || die "unexpected mismatch output for rank ${r}: ${line}"
done
cat "${scratch}/out-mismatch-w2/rank-0.stdout"
printf 'DIST5 AOT multi-process mismatch W=2: PASS\n'
