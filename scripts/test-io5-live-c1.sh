#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

for command in ar cmp python3 rg sha256sum timeout; do
  require_command "${command}"
done
verify_toolchain
runner="$(eshkol_build_dir)/eshkol-run"
cc="$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cc_path)"
compiler_timeout="${IO5_LIVE_C1_COMPILER_TIMEOUT_SECONDS:-480}"
[[ "${compiler_timeout}" =~ ^[1-9][0-9]*$ ]] ||   die "IO5_LIVE_C1_COMPILER_TIMEOUT_SECONDS must be a positive integer"

build="$(project_build_dir)"
mkdir -p "${build}"
build="$(realpath -- "${build}")"
export BUILD_DIR="${build}"
scratch="$(mktemp -d "${build}/.io5-live-c1.XXXXXX")"
trap 'rm -rf -- "${scratch}"' EXIT

bash "${PROJECT_ROOT}/scripts/build-c1.sh"
bash "${PROJECT_ROOT}/scripts/build-p1-identity.sh" "${scratch}/p1" normal trusted

# Link the real I2 carrier helpers and their production support objects into
# the same identity universe as the source-composed C1 and P1 roots.
cflags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic -Wconversion
  -Wsign-conversion -Wshadow -fPIC -fvisibility=hidden -fno-common
  -fstack-protector-all -ffp-contract=off -fexcess-precision=standard
  -frounding-math -I "${PROJECT_ROOT}/include" -I "${PROJECT_ROOT}/native"
)
objects=()
for source in i2_wave2_package_bridge data_io kernel_abi i64_tensor t1_i64_shell f32_tensor; do
  object="${scratch}/${source}.o"
  "${cc}" "${cflags[@]}" -DET_I2_NATIVE_HELPERS_ONLY     -c "${PROJECT_ROOT}/native/${source}.c" -o "${object}"
  objects+=("${object}")
done
ar rcsD "${scratch}/libio5_live_i2.a" "${objects[@]}"

for run in a b; do
  output="${scratch}/${run}"
  mkdir -p "${output}/cache"
  if ! (cd "${output}" &&     timeout --foreground --signal=TERM --kill-after=5s "${compiler_timeout}s"       env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0         XDG_CACHE_HOME="${output}/cache" ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib"         "${runner}" --strict-types --optimize 0 --no-stdlib         -I "${PROJECT_ROOT}/internal/c1/lib" -I "${PROJECT_ROOT}/internal/p1/lib"         -I "${PROJECT_ROOT}/internal/t1/lib" -I "${PROJECT_ROOT}/src"         -I "${PROJECT_ROOT}/lib" -I "${PROJECT_ROOT}/native"         -L "${scratch}/p1/trusted" --lib eshkol_transformer_p1_identity         -L "${build}/c1" --lib eshkol_transformer_checkpoint_io         -L "${scratch}" --lib io5_live_i2         "${PROJECT_ROOT}/tests/io5/live_c1_io5_bridge.esk" -o live-c1-io5         >compile.log 2>&1) ||
    [[ ! -x "${output}/live-c1-io5" ]] ||
    rg -F 'ERROR:' "${output}/compile.log" >/dev/null; then
    sed -n '1,240p' "${output}/compile.log" >&2
    die "IO5 live C1 AOT compilation ${run} failed"
  fi
  for mode in write verify; do
    if ! timeout 60s "${output}/live-c1-io5" "${mode}" "${output}/original.etcp" >"${output}/${mode}.stdout"; then
      cat "${output}/${mode}.stdout" >&2
      die "IO5 live C1 ${mode} ${run} failed"
    fi
  done
done

for artifact in live-c1-io5 original.etcp write.stdout verify.stdout; do
  cmp "${scratch}/a/${artifact}" "${scratch}/b/${artifact}"
done
(cd "${scratch}/a" && sha256sum live-c1-io5 original.etcp)

IO5_LIVE_C1_BRIDGE="${scratch}/a/live-c1-io5"   python3 "${PROJECT_ROOT}/tests/io5/test_live_c1_interop.py" -v
printf 'IO5 LIVE C1 PASS: deterministic AOT and checkpoint, five exact live round trips, corruption rejections\n'
