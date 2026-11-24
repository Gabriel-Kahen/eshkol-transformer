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

mesh_dir="${PROJECT_ROOT}/build/hybrid_mesh"
bash scripts/build-hybrid-mesh.sh "${mesh_dir}"

scratch="$(mktemp -d "${mesh_dir}/.test.XXXXXX")"
sock_root="$(mktemp -d /tmp/hybrid_mesh_aot_XXXXXX)"
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
  "${mesh_dir}/libeshkol_transformer_hybrid_mesh_aot.a"
  "${mesh_dir}/libeshkol_transformer_hybrid_mesh.a"
)

# 1. C++17 header compile and archive linkage check.
"${CXX}" "${cppflags[@]}" -O2 tests/hybrid_mesh/header_mesh_cpp.cpp \
  "${archives[@]}" -lm -o "${scratch}/header_mesh_cpp"
"${scratch}/header_mesh_cpp"
printf 'HYBRID_MESH C++17 header and linkage: PASS\n'

# 2. Native C11 normal (-O2) tests + independent Python binary64 oracle.
"${CC}" "${cflags[@]}" -O2 tests/hybrid_mesh/test_hybrid_mesh_native.c \
  "${archives[@]}" -lm -o "${scratch}/test_hybrid_mesh_native"
"${scratch}/test_hybrid_mesh_native"
python3 tests/hybrid_mesh/test_hybrid_mesh_oracle.py "${scratch}/test_hybrid_mesh_native"

# 3. Native C11 and C++17 tests under ASan/UBSan/LSan (detect_leaks=1:halt_on_error=1).
export ASAN_OPTIONS=detect_leaks=1:halt_on_error=1
export UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1

"${CC}" "${cflags[@]}" "${san[@]}" \
  tests/hybrid_mesh/test_hybrid_mesh_native.c \
  native/hybrid_mesh_eshkol_bridge.c native/hybrid_mesh.c \
  -lm -o "${scratch}/test_hybrid_mesh_native.san"
"${scratch}/test_hybrid_mesh_native.san"

"${CC}" "${cflags[@]}" "${san[@]}" -c native/hybrid_mesh.c \
  -o "${scratch}/hybrid_mesh.san.o"
"${CC}" "${cflags[@]}" "${san[@]}" -c native/hybrid_mesh_eshkol_bridge.c \
  -o "${scratch}/hybrid_mesh_eshkol_bridge.san.o"
"${CXX}" "${cppflags[@]}" "${san[@]}" tests/hybrid_mesh/header_mesh_cpp.cpp \
  "${scratch}/hybrid_mesh_eshkol_bridge.san.o" \
  "${scratch}/hybrid_mesh.san.o" \
  -lm -o "${scratch}/header_mesh_cpp.san"
"${scratch}/header_mesh_cpp.san"
printf 'HYBRID_MESH ASan/UBSan/LSan tests: PASS\n'

# 4. Compile tests/hybrid_mesh/test_hybrid_mesh_aot.esk TWICE from fresh dirs and
#    verify byte-identical ELFs (--strict-types --optimize 0 --no-stdlib).
for run in 1 2; do
  run_dir="${scratch}/esk-run-${run}"
  mkdir -p "${run_dir}/cache" "${run_dir}/out" "${run_dir}/libs"
  bash scripts/build-hybrid-mesh.sh "${run_dir}/libs" >/dev/null
  diagnostic="${run_dir}/compile.log"
  if ! (
    cd "${run_dir}/out"
    timeout --foreground --kill-after=5s 120s \
      env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR -u ESHKOL_AOT_MODULE_CACHE_DIR \
      ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="${run_dir}/cache" \
      "${runner}" --strict-types --optimize 0 --no-stdlib \
        -L "${run_dir}/libs" \
        --lib eshkol_transformer_hybrid_mesh_aot \
        --lib eshkol_transformer_hybrid_mesh \
        "${PROJECT_ROOT}/tests/hybrid_mesh/test_hybrid_mesh_aot.esk" \
        -o hybrid-mesh-aot >"${diagnostic}" 2>&1
  ) || [[ ! -x "${run_dir}/out/hybrid-mesh-aot" ]] || grep -Fq 'ERROR:' "${diagnostic}"; then
    die "HYBRID_MESH AOT Eshkol compilation ${run} failed"
  fi
done
diagnostic=''

elf="${scratch}/esk-run-1/out/hybrid-mesh-aot"
readelf -h "${elf}" >/dev/null
cmp -- "${elf}" "${scratch}/esk-run-2/out/hybrid-mesh-aot"
elf_sha256="$(sha256sum "${elf}" | awk '{print $1}')"
printf 'HYBRID_MESH AOT two fresh compilations, byte-identical ELF SHA-256: %s PASS\n' "${elf_sha256}"

# Helper to launch (dp * tp) independent OS processes in reversed rank order.
run_mesh() {
  local label="$1" dp="$2" tp="$3" mode="$4"
  shift 4
  local world=$((dp * tp))
  local sdir="${sock_root}/${label}"
  local odir="${scratch}/out-${label}"
  mkdir -p "${sdir}" "${odir}"
  local -a pids=()
  local r
  for ((r = world - 1; r >= 0; --r)); do
    local -a cmd=(
      "${elf}" --sock-dir "${sdir}" --rank "${r}" --dp "${dp}" --tp "${tp}"
      --mode "${mode}" "$@"
    )
    if [[ "${mode}" == "train10" && "${r}" == 0 ]]; then
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
    die "HYBRID_MESH AOT group ${label} failed"
  fi
}

extract_field() {
  local key="$1" file="$2"
  awk -v k="${key}=" '$1 == "MESH" && $2 == "AOT" && ($3 == "RESULT" || $3 == "RESUME_RESULT") {
    for (i = 1; i <= NF; ++i) {
      if (index($i, k) == 1) {
        print substr($i, length(k) + 1);
      }
    }
  }' "${file}"
}

# 5. Live compiled Eshkol AOT 10-step training across (1x1), (2x1), (1x2), (2x2).
for topo in "1 1" "2 1" "1 2" "2 2"; do
  read -r dp tp <<< "${topo}"
  world=$((dp * tp))
  label="train-dp${dp}-tp${tp}"
  run_mesh "${label}" "${dp}" "${tp}" train10
  wsha0="$(extract_field weights_sha256 "${scratch}/out-${label}/rank-0.stdout")"
  [[ "${#wsha0}" == 64 ]] || die "missing weights_sha256 for ${label}"
  for ((r = 1; r < world; ++r)); do
    wshar="$(extract_field weights_sha256 "${scratch}/out-${label}/rank-${r}.stdout")"
    [[ "${wshar}" == "${wsha0}" ]] || die "${label} rank ${r} gathered weights diverged from rank 0"
  done
  # Verify bitwise equality of matching TP shards across DP replicas!
  if (( dp > 1 )); then
    for ((d = 1; d < dp; ++d)); do
      for ((tr = 0; tr < tp; ++tr)); do
        r0="${tr}"
        rd=$((d * tp + tr))
        lsha0="$(extract_field local_sha256 "${scratch}/out-${label}/rank-${r0}.stdout")"
        lshad="$(extract_field local_sha256 "${scratch}/out-${label}/rank-${rd}.stdout")"
        [[ "${lsha0}" == "${lshad}" ]] || \
          die "${label} DP replica rank ${rd} shard diverged from rank ${r0}"
      done
    done
  fi
  # Verify distinct TP shards between tp_rank 0 and tp_rank 1 on (2x2).
  if (( dp == 2 && tp == 2 )); then
    lsha_r0="$(extract_field local_sha256 "${scratch}/out-${label}/rank-0.stdout")"
    lsha_r1="$(extract_field local_sha256 "${scratch}/out-${label}/rank-1.stdout")"
    [[ "${lsha_r0}" != "${lsha_r1}" ]] || die "2x2 tp_rank 0 and tp_rank 1 shards unexpectedly identical"
  fi
  cat "${scratch}/out-${label}/rank-0.stdout"
done

# Cross-verify compiled Eshkol AOT (1x1), (2x1), (1x2), (2x2) dumped weights against
# each other and the independent Python binary64 reference (<= 1e-6).
python3 tests/hybrid_mesh/test_hybrid_mesh_oracle.py --check-aot \
  "${scratch}/out-train-dp1-tp1/weights.bin" \
  "${scratch}/out-train-dp2-tp1/weights.bin" \
  "${scratch}/out-train-dp1-tp2/weights.bin" \
  "${scratch}/out-train-dp2-tp2/weights.bin"

# 6. Fresh-process 5+5 ET2D1000 checkpoint resume on the (2x2) 4-process 2D mesh!
cdir="${scratch}/ckpt-dp2-tp2"
mkdir -p "${cdir}"
run_mesh "resume5-dp2-tp2" 2 2 resume5 --ckpt-dir "${cdir}"
run_mesh "resume5-cont-dp2-tp2" 2 2 resume5-cont --ckpt-dir "${cdir}"
for r in 0 1 2 3; do
  w_full="$(extract_field weights_sha256 "${scratch}/out-train-dp2-tp2/rank-${r}.stdout")"
  l_full="$(extract_field local_sha256 "${scratch}/out-train-dp2-tp2/rank-${r}.stdout")"
  w_res="$(extract_field weights_sha256 "${scratch}/out-resume5-cont-dp2-tp2/rank-${r}.stdout")"
  l_res="$(extract_field local_sha256 "${scratch}/out-resume5-cont-dp2-tp2/rank-${r}.stdout")"
  [[ "${w_res}" == "${w_full}" ]] || die "2x2 rank ${r} 5+5 resume weights diverged"
  [[ "${l_res}" == "${l_full}" ]] || die "2x2 rank ${r} 5+5 resume local shard diverged"
done
cat "${scratch}/out-resume5-cont-dp2-tp2/rank-0.stdout"
printf 'HYBRID_MESH AOT 5+5 fresh-process (2x2) checkpoint resume bitwise parity: PASS\n'

# 7. Live 4-process (2x2) single-rank qNaN atomic abort on rank 3 + clean recovery!
run_mesh "nonfinite-dp2-tp2" 2 2 nonfinite
for r in 0 1 2 3; do
  line="$(cat "${scratch}/out-nonfinite-dp2-tp2/rank-${r}.stdout")"
  [[ "${line}" == "MESH AOT NONFINITE dp=2 tp=2 rank=${r} nan_status=-7 shard_preserved=1 recovered=1 PASS" ]] || \
    die "unexpected nonfinite output on rank ${r}: ${line}"
done
cat "${scratch}/out-nonfinite-dp2-tp2/rank-0.stdout"

# 8. Compiled Eshkol AOT negative checks (1x1 local + 1x2 AF_UNIX sticky mismatch).
mkdir -p "${scratch}/neg-ckpt"
run_mesh "negative-dp1-tp1" 1 1 negative --ckpt-dir "${scratch}/neg-ckpt"
cat "${scratch}/out-negative-dp1-tp1/rank-0.stdout"

run_mesh "negative-dp1-tp2" 1 2 negative
for r in 0 1; do
  line="$(cat "${scratch}/out-negative-dp1-tp2/rank-${r}.stdout")"
  [[ "${line}" == "MESH AOT NEGATIVE_MISMATCH dp=1 tp=2 rank=${r} status=-4 sticky=-4 PASS" ]] || \
    die "unexpected mismatch output on rank ${r}: ${line}"
done
cat "${scratch}/out-negative-dp1-tp2/rank-0.stdout"

printf 'HYBRID_MESH FULL SUITE PASS\n'
