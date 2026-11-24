#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

cd "${PROJECT_ROOT}"
for command in ar cmp diff nm readelf sha256sum timeout; do
  require_command "${command}"
done
verify_toolchain
export LC_ALL=C PYTHONDONTWRITEBYTECODE=1

export CC="${CC:-$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cc_path)}"
export CXX="${CXX:-$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cxx_path)}"
runner="$(eshkol_build_dir)/eshkol-run"
compiler_timeout="${MOD5_GEN_COMPILER_TIMEOUT_SECONDS:-480}"
[[ "${compiler_timeout}" =~ ^[1-9][0-9]*$ ]] || die "compiler timeout must be a positive integer"

mkdir -p "$(project_build_dir)"
build="$(realpath -- "$(project_build_dir)")"
export BUILD_DIR="${build}"
scratch="$(mktemp -d "${build}/.mod5-gen.XXXXXX")"
trap 'rm -rf -- "${scratch}"' EXIT

# 1. Build MOD5 K1 provider and MOD5 generation archives (normal + sanitize).
bash scripts/build-mod5.sh
bash scripts/build-mod5-generate.sh "${build}/mod5_gen" normal
bash scripts/build-mod5-generate.sh "${scratch}/mod5_gen_san" sanitize

common_flags=(-Wall -Wextra -Werror -Wpedantic -ffp-contract=off -fno-fast-math -I include)
cflags=(-std=c11 "${common_flags[@]}" -fexcess-precision=standard -frounding-math)
cppflags=(-std=c++17 "${common_flags[@]}")
san_flags=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)

# 2. Normal native tests (C11 + C++17 header/linkage).
"${CC}" "${cflags[@]}" -O2 -c native/kernel_abi.c -o "${scratch}/kernel_abi.o"
normal_libs=(
  "${build}/mod5_gen/libeshkol_transformer_mod5_generate.a"
  "${PROJECT_ROOT}/build/mod5/libeshkol_transformer_mod5.a"
  "${scratch}/kernel_abi.o"
)

"${CXX}" "${cppflags[@]}" -O2 tests/mod5/header_generate_cpp.cpp \
  "${normal_libs[@]}" -lm -o "${scratch}/header_generate_cpp"
"${scratch}/header_generate_cpp"
printf 'MOD5 generate C++17 header & linkage (normal): PASS\n'

"${CC}" "${cflags[@]}" -O2 tests/mod5/test_mod5_generate_native.c \
  "${normal_libs[@]}" -lm -o "${scratch}/test_mod5_generate_native"
"${scratch}/test_mod5_generate_native"
printf 'MOD5 generate C11 native tests (normal): PASS\n'

# 3. ASan/UBSan/LSan native tests (C11 + C++17).
"${CC}" "${cflags[@]}" "${san_flags[@]}" -c native/kernel_abi.c \
  -o "${scratch}/kernel_abi.san.o"
"${CC}" "${cflags[@]}" "${san_flags[@]}" -c native/mod5_primitives_provider.c \
  -o "${scratch}/mod5_primitives_provider.san.o"
san_libs=(
  "${scratch}/mod5_gen_san/libeshkol_transformer_mod5_generate.a"
  "${scratch}/mod5_primitives_provider.san.o"
  "${scratch}/kernel_abi.san.o"
)

"${CXX}" "${cppflags[@]}" "${san_flags[@]}" tests/mod5/header_generate_cpp.cpp \
  "${san_libs[@]}" -lm -o "${scratch}/header_generate_cpp.san"
"${CC}" "${cflags[@]}" "${san_flags[@]}" tests/mod5/test_mod5_generate_native.c \
  "${san_libs[@]}" -lm -o "${scratch}/test_mod5_generate_native.san"

export ASAN_OPTIONS="detect_leaks=1:halt_on_error=1"
export UBSAN_OPTIONS="halt_on_error=1"
"${scratch}/header_generate_cpp.san"
"${scratch}/test_mod5_generate_native.san"
printf 'MOD5 generate C11 + C++17 ASan/UBSan/LSan tests: PASS\n'

# 4. Build C1/P1/I2/MOD5 prerequisites and compile Eshkol AOT executable TWICE.
bash scripts/build-c1.sh
bash scripts/build-p1-identity.sh "${scratch}/p1" normal trusted

native_cflags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -Wconversion -Wsign-conversion
  -Wshadow -fPIC -fvisibility=hidden -fno-common -fstack-protector-all
  -ffp-contract=off -fexcess-precision=standard -frounding-math
  -I include -I native)
objects=()
for source in i2_wave2_package_bridge data_io kernel_abi i64_tensor t1_i64_shell f32_tensor mod5_eshkol_bridge; do
  object="${scratch}/${source}.o"
  "${CC}" "${native_cflags[@]}" -DET_I2_NATIVE_HELPERS_ONLY -c "native/${source}.c" -o "${object}"
  objects+=("${object}")
done
ar rcsD "${scratch}/libmod5_gen_e2e_native.a" "${objects[@]}"

for run in a b; do
  output="${scratch}/${run}"
  mkdir -p "${output}/cache"
  if ! (
    cd "${output}"
    timeout --foreground --kill-after=5s "${compiler_timeout}s" \
      env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR -u ESHKOL_AOT_MODULE_CACHE_DIR \
      ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="${output}/cache" \
      "${runner}" --strict-types --optimize 0 --no-stdlib \
        -I "${PROJECT_ROOT}/internal/c1/lib" -I "${PROJECT_ROOT}/internal/p1/lib" \
        -I "${PROJECT_ROOT}/internal/t1/lib" -I "${PROJECT_ROOT}/src" \
        -I "${PROJECT_ROOT}/lib" -I "${PROJECT_ROOT}/native" \
        -L "${scratch}/p1/trusted" --lib eshkol_transformer_p1_identity \
        -L "${build}/c1" --lib eshkol_transformer_checkpoint_io \
        -L "${build}/mod5_gen" --lib eshkol_transformer_mod5_generate \
        -L "${scratch}" --lib mod5_gen_e2e_native \
        -L "${PROJECT_ROOT}/build/mod5" --lib eshkol_transformer_mod5 \
        "${PROJECT_ROOT}/tests/mod5/test_mod5_generate_aot.esk" \
        -o mod5-generate-aot >compile.log 2>&1
  ) || [[ ! -x "${output}/mod5-generate-aot" ]] || grep -Fq 'ERROR:' "${output}/compile.log"; then
    tail -120 "${output}/compile.log" >&2
    die "MOD5 generate AOT compilation ${run} failed"
  fi
done

readelf -h "${scratch}/a/mod5-generate-aot" >/dev/null
cmp "${scratch}/a/mod5-generate-aot" "${scratch}/b/mod5-generate-aot"
elf_sha="$(sha256sum "${scratch}/a/mod5-generate-aot" | awk '{print $1}')"
printf 'MOD5 generate AOT byte-identical ELF SHA-256: %s\n' "${elf_sha}"

# 5. End-to-end AOT pipeline: D1 corpus -> train & save C1 -> fresh-process reload & generation.
aot_bin="${scratch}/a/mod5-generate-aot"
work="${scratch}/e2e"
mkdir -p "${work}/corpus"

"${aot_bin}" --mode write-corpus "${work}/corpus"
train_out="$("${aot_bin}" --mode train-save "${work}/corpus" "${work}/trained.etcp")"
printf '%s\n' "${train_out}"
grep -q '^TRAIN_GRADIENTS 12$' <<< "${train_out}"
grep -q '^TRAIN_SAVED$' <<< "${train_out}"
[[ -f "${work}/trained.etcp" ]] || die "C1 checkpoint not created"
ckpt_sha="$(sha256sum "${work}/trained.etcp" | awk '{print $1}')"
printf 'MOD5 generate C1 checkpoint SHA-256: %s\n' "${ckpt_sha}"

# Two fresh processes reloading the same C1 checkpoint with seed 42 must match identically.
gen_a="$("${aot_bin}" --mode generate "${work}/trained.etcp" 42)"
gen_b="$("${aot_bin}" --mode generate "${work}/trained.etcp" 42)"
printf '%s\n' "${gen_a}"
[[ "${gen_a}" == "${gen_b}" ]] || die "fresh-process generation mismatch for identical seed 42"
grep -q '^PARITY_T8 max_abs_diff=' <<< "${gen_a}"
grep -q '^PARITY_GEN8 max_abs_diff=' <<< "${gen_a}"
grep -q '^GREEDY tokens=' <<< "${gen_a}"
grep -q '^EOS_STOP target=' <<< "${gen_a}"
grep -q '^SAMPLED seed=42 tokens=' <<< "${gen_a}"
grep -q '^AOT_NEGATIVE_CHECKS PASS$' <<< "${gen_a}"
printf 'MOD5 generate fresh-process same-seed replay (seed=42): PASS\n'

# Changing seed (1337) produces deterministic output across two fresh processes.
gen_c="$("${aot_bin}" --mode generate "${work}/trained.etcp" 1337)"
gen_d="$("${aot_bin}" --mode generate "${work}/trained.etcp" 1337)"
printf '%s\n' "${gen_c}"
[[ "${gen_c}" == "${gen_d}" ]] || die "fresh-process generation mismatch for identical seed 1337"
grep -q '^SAMPLED seed=1337 tokens=' <<< "${gen_c}"

greedy_a="$(grep '^GREEDY ' <<< "${gen_a}")"
greedy_c="$(grep '^GREEDY ' <<< "${gen_c}")"
[[ "${greedy_a}" == "${greedy_c}" ]] || die "greedy output must be independent of seed"

sampled_a="$(grep '^SAMPLED ' <<< "${gen_a}" | sed 's/^SAMPLED seed=42 //')"
sampled_c="$(grep '^SAMPLED ' <<< "${gen_c}" | sed 's/^SAMPLED seed=1337 //')"
[[ "${sampled_a}" != "${sampled_c}" ]] || die "changing seed 42->1337 should change sampled trajectory"
printf 'MOD5 generate different-seed deterministic replay (seed=1337): PASS\n'

# Corrupt C1 checkpoint must reject cleanly in a fresh process.
corrupt="${work}/corrupt.etcp"
cp "${work}/trained.etcp" "${corrupt}"
python3 - "${corrupt}" <<'PY'
import sys
from pathlib import Path
p = Path(sys.argv[1])
b = bytearray(p.read_bytes())
b[len(b) // 2] ^= 0xFF
p.write_bytes(b)
PY
set +e
corrupt_out="$("${aot_bin}" --mode generate "${corrupt}" 42)"
corrupt_rc=$?
set -e
printf '[corrupt C1] rc=%d %s\n' "${corrupt_rc}" "${corrupt_out}"
[[ "${corrupt_rc}" -eq 2 ]] || die "corrupt C1 expected exit 2, got ${corrupt_rc}"
grep -q '^MOD5 GEN ERROR ' <<< "${corrupt_out}"
! grep -q '^GREEDY ' <<< "${corrupt_out}" || die "corrupt C1 produced GREEDY output"
printf 'MOD5 generate corrupt C1 checkpoint rejection: PASS\n'

printf 'MOD5 GENERATION FULL SUITE (native normal+ASan/UBSan/LSan, C++17, byte-identical AOT ELF, C1 E2E): PASS\n'
