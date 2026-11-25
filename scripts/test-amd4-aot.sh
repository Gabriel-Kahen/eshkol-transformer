#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

cd "${PROJECT_ROOT}"
for command in ar cmp diff nm readelf sha256sum timeout; do
  require_command "${command}"
done
verify_toolchain
export LC_ALL=C

cc="${CC:-$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cc_path)}"
cxx="${CXX:-$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cxx_path)}"
runner="$(eshkol_build_dir)/eshkol-run"

mkdir -p "$(project_build_dir)"
build="$(realpath -- "$(project_build_dir)")"
scratch="$(mktemp -d "${build}/.amd4-aot-test.XXXXXX")"
diagnostic=''
cleanup() {
  local status=$?
  if [[ "${status}" != 0 && -n "${diagnostic}" && -s "${diagnostic}" ]]; then
    cat "${diagnostic}" >&2
  fi
  rm -rf -- "${scratch}"
}
trap cleanup EXIT

# 1. Build AMD4 provider, K1 runtime, and AMD4 AOT bridge archive twice for
#    byte-for-byte archive reproducibility and symbol audit.
bash scripts/build-amd4.sh "${build}/amd4"
bash scripts/build-k1.sh "${scratch}/k1"
bash scripts/build-amd4-aot.sh "${scratch}/aot-1"
bash scripts/build-amd4-aot.sh "${scratch}/aot-2"
cmp -- "${scratch}/aot-1/libeshkol_transformer_amd4_aot.a" \
  "${scratch}/aot-2/libeshkol_transformer_amd4_aot.a"
bash scripts/build-amd4-aot.sh "${build}/amd4_aot"
cmp -- "${build}/amd4_aot/libeshkol_transformer_amd4_aot.a" \
  "${scratch}/aot-1/libeshkol_transformer_amd4_aot.a"
(cd -- "${build}/amd4_aot" && sha256sum -c libeshkol_transformer_amd4_aot.a.sha256)

# 2. C++17 header and linkage test.
warnings=(-Wall -Wextra -Werror -Wpedantic -I include -I native)
"${cxx}" -std=c++17 "${warnings[@]}" tests/amd4/header_bridge_cpp.cpp \
  "${build}/amd4_aot/libeshkol_transformer_amd4_aot.a" \
  "${build}/amd4/libeshkol_transformer_amd4.a" \
  "${scratch}/k1/libeshkol_transformer_k1.a" \
  -ldl -o "${scratch}/header-bridge-cpp"
"${scratch}/header-bridge-cpp"
printf 'AMD4 AOT bridge C++17 header and linkage: PASS\n'

# 3. Direct C11 bridge tests in normal mode and ASan/UBSan/LSan mode.
"${cc}" -std=c11 "${warnings[@]}" -O2 tests/amd4/test_amd4_aot_bridge.c \
  "${build}/amd4_aot/libeshkol_transformer_amd4_aot.a" \
  "${build}/amd4/libeshkol_transformer_amd4.a" \
  "${scratch}/k1/libeshkol_transformer_k1.a" \
  -ldl -o "${scratch}/test-amd4-aot-bridge"
AMD4_EXPECT_NO_HIP="${AMD4_EXPECT_NO_HIP:-1}" "${scratch}/test-amd4-aot-bridge"

"${cc}" -std=c11 "${warnings[@]}" -O1 -g -fno-omit-frame-pointer \
  -fsanitize=address,undefined -fno-sanitize-recover=all \
  tests/amd4/test_amd4_aot_bridge.c \
  native/amd4_eshkol_bridge.c \
  native/amd4_hip_observability.c \
  native/kernel_abi.c \
  -ldl -o "${scratch}/test-amd4-aot-bridge-san"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
  UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  AMD4_EXPECT_NO_HIP="${AMD4_EXPECT_NO_HIP:-1}" \
  "${scratch}/test-amd4-aot-bridge-san"
printf 'AMD4 AOT bridge C11 normal + ASan/UBSan/LSan: PASS\n'

# 4. Compile tests/amd4/test_amd4_aot.esk TWICE from fresh build/cache dirs
#    and verify byte-identical ELFs and identical stdout.
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
        -L "${build}/amd4_aot" -L "${build}/amd4" -L "${scratch}/k1" \
        --lib eshkol_transformer_amd4_aot \
        --lib eshkol_transformer_amd4 \
        --lib eshkol_transformer_k1 \
        --lib dl \
        "${PROJECT_ROOT}/tests/amd4/test_amd4_aot.esk" \
        -o amd4-aot >"${diagnostic}" 2>&1
  ) || [[ ! -x "${run_dir}/out/amd4-aot" ]] || grep -Fq 'ERROR:' "${diagnostic}"; then
    die "AMD4 AOT Eshkol compilation ${run} failed"
  fi
  diagnostic="${run_dir}/run.stderr"
  timeout --foreground --kill-after=5s 30s "${run_dir}/out/amd4-aot" \
    >"${run_dir}/run.stdout" 2>"${diagnostic}" || {
    cat "${run_dir}/run.stdout" >&2
    die "AMD4 AOT compiled binary failed on run ${run}"
  }
  [[ ! -s "${diagnostic}" ]] || die "AMD4 AOT compiled binary emitted stderr on run ${run}"
done
diagnostic=''

readelf -h "${scratch}/esk-run-1/out/amd4-aot" >/dev/null
cmp -- "${scratch}/esk-run-1/out/amd4-aot" "${scratch}/esk-run-2/out/amd4-aot"
cmp -- "${scratch}/esk-run-1/run.stdout" "${scratch}/esk-run-2/run.stdout"

awk '
  /^AMD4 AOT LIVE ABSENCE status=unavailable-runtime kfd=(absent|inaccessible|accessible) hip=unavailable rocblas=unavailable devices=0 bytes=[1-9][0-9]*$/ { ++live; next }
  /^AMD4 AOT INJECTED TWO_DEVICE status=ok devices=2 bytes=[1-9][0-9]*$/ { ++twodev; next }
  $0 == "AMD4 AOT INJECTED FAILURES oom=alloc driver=driver-failure rocblas=missing-symbols PASS" { ++fail; next }
  $0 == "AMD4 AOT K1 REJECTION gemm=unsupported attention=unsupported code=15 canary=0xdeadbeef PASS" { ++k1; next }
  $0 == "AMD4 AOT ALL CHECKS PASS" { ++done_line; next }
  { exit 1 }
  END {
    if (live != 1 || twodev != 1 || fail != 1 || k1 != 1 || done_line != 1)
      exit 1
  }
' "${scratch}/esk-run-1/run.stdout" || {
  cat "${scratch}/esk-run-1/run.stdout" >&2
  die "unexpected AMD4 AOT compiled stdout"
}

cat "${scratch}/esk-run-1/run.stdout"
elf_sha="$(sha256sum "${scratch}/esk-run-1/out/amd4-aot" | awk '{ print $1 }')"
printf 'AMD4 AOT ELF SHA-256: %s\n' "${elf_sha}"
printf 'AMD4 AOT two fresh compilations, byte-identical ELF and stdout: PASS\n'
