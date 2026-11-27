#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/common.sh"

[[ "${MOD5_V256_GRADIENT_AOT_SLOT:-0}" == 1 ]] ||
  die "MOD5 V256 gradient AOT requires an assigned build slot"
[[ $# == 1 && ! -e "$1" && ! -L "$1" ]] || die "expected a fresh evidence path"
[[ "${ESHKOL_ALLOW_UNSUPPORTED_HOST:-0}" == 0 ]] || die "expected the supported toolchain"
for command in ar cmp git grep python3 readelf sha256sum timeout; do
  require_command "${command}"
done
verify_toolchain
mkdir -p -- "$1"
out="$(realpath -- "$1")"
cd "${PROJECT_ROOT}"
export LC_ALL=C PYTHONDONTWRITEBYTECODE=1

export BUILD_DIR="${out}/build"
cc= cxx=
resolve_provenance_compilers cc cxx "$(lock_value supported_cc)" "$(lock_value supported_cxx)"
runner="$(eshkol_build_dir)/eshkol-run"
{
  git rev-parse HEAD 'HEAD^{tree}'
  sha256sum toolchain/eshkol.lock "${runner}"
} > "${out}/provenance.txt"
python3 -B -m tests.mod5.mod5_v256_gradient build "${out}"
bash scripts/build-c1.sh > "${out}/c1.log" 2>&1
bash scripts/build-p1-identity.sh "${out}/p1" normal trusted > "${out}/p1.log" 2>&1

cflags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -Wconversion -Wsign-conversion
  -Wshadow -fPIC -fvisibility=hidden -fno-common -fstack-protector-all
  -ffp-contract=off -fexcess-precision=standard -frounding-math -I include -I native)
objects=()
for source in i2_wave2_package_bridge data_io kernel_abi i64_tensor t1_i64_shell f32_tensor mod5_eshkol_bridge; do
  object="${out}/${source}.o"
  "${cc}" "${cflags[@]}" -DET_I2_NATIVE_HELPERS_ONLY -c "native/${source}.c" -o "${object}"
  objects+=("${object}")
done
"${cc}" -std=c11 -Wall -Wextra -Werror -Wpedantic -O2 -ffp-contract=off \
  -fexcess-precision=standard -fno-fast-math -fstack-protector-all -fPIC -I include \
  -c native/mod5_primitives_provider.c -o "${out}/mod5_primitives_provider.o"
ar rcsD "${out}/libmod5_v256_gradient.a" "${objects[@]}" "${out}/mod5_primitives_provider.o"

mkdir "${out}/cache"
(
  cd "${out}"
  timeout --foreground --kill-after=5s 600s \
    env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR -u ESHKOL_AOT_MODULE_CACHE_DIR \
    ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="${out}/cache" \
    ESHKOL_CXX_COMPILER="${cxx}" "${runner}" --strict-types --optimize 0 --no-stdlib \
    -I "${PROJECT_ROOT}/internal/c1/lib" -I "${PROJECT_ROOT}/internal/p1/lib" \
    -I "${PROJECT_ROOT}/internal/t1/lib" -I "${PROJECT_ROOT}/src" \
    -I "${PROJECT_ROOT}/lib" -I "${PROJECT_ROOT}/native" \
    -L "${out}/p1/trusted" --lib eshkol_transformer_p1_identity \
    -L "${BUILD_DIR}/c1" --lib eshkol_transformer_checkpoint_io \
    -L "${out}" --lib mod5_v256_gradient "${out}/witness.esk" -o witness
) > "${out}/compile.log" 2>&1
test -x "${out}/witness"
! grep -Fq 'ERROR:' "${out}/compile.log" || die "MOD5 V256 gradient compiler reported an error"
readelf -h "${out}/witness" > "${out}/elf.txt"
for mode in normal repeat; do
  timeout --foreground --kill-after=5s 120s "${out}/witness" \
    "${out}/fixture/tokenizer.etok" "${out}/fixture/eval-clean" \
    "${out}/fixture/model.etcp" "${out}/tied.etcp" \
    > "${out}/${mode}.stdout" 2> "${out}/${mode}.stderr"
  test ! -s "${out}/${mode}.stderr"
  python3 -B -m tests.mod5.mod5_v256_gradient verify "${out}" "${out}/${mode}.stdout" \
    > "${out}/${mode}.oracle"
done
cmp "${out}/normal.stdout" "${out}/repeat.stdout"
(cd "${out}" && sha256sum provenance.txt fixture/fixture.json tied.etcp reference.json \
  witness.esk witness libmod5_v256_gradient.a compile.log elf.txt \
  normal.stdout normal.stderr normal.oracle repeat.stdout repeat.stderr repeat.oracle > SHA256SUMS)
cat "${out}/normal.oracle"
