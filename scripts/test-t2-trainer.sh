#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

cd "${PROJECT_ROOT}"
for command in ar cmp diff nm python3 readelf sha256sum timeout; do
  require_command "${command}"
done
verify_toolchain
export LC_ALL=C PYTHONDONTWRITEBYTECODE=1

export CC="${CC:-$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cc_path)}"
export CXX="${CXX:-$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cxx_path)}"
runner="$(eshkol_build_dir)/eshkol-run"
compiler_timeout="${T2_TRAINER_COMPILER_TIMEOUT_SECONDS:-180}"
[[ "${compiler_timeout}" =~ ^[1-9][0-9]*$ ]] || die "compiler timeout must be a positive integer"

mkdir -p "$(project_build_dir)"
build="$(realpath -- "$(project_build_dir)")"
export BUILD_DIR="${build}"
scratch="$(mktemp -d "${build}/.t2-trainer-test.XXXXXX")"
trap 'rm -rf -- "${scratch}"' EXIT

# 1. Build deterministic archives (normal + sanitize) with symbol & C++17 header audits.
bash scripts/build-t2-trainer.sh "${build}/t2_trainer" normal
bash scripts/build-t2-trainer.sh "${scratch}/t2_trainer_san" sanitize

# 2. C++17 header & archive linkage test.
"${CXX}" -std=c++17 -Wall -Wextra -Werror -Wpedantic -O2 \
  -I include -I native \
  tests/t2_trainer/header_trainer_cpp.cpp \
  "${build}/t2_trainer/libeshkol_transformer_t2_trainer_aot.a" \
  "${build}/t2_trainer/libeshkol_transformer_t2_trainer.a" \
  -o "${scratch}/header_trainer_cpp"
"${scratch}/header_trainer_cpp"
printf 'T2_TRAINER C++17 header and archive linkage: PASS\n'

# 3. Native C11 tests in normal (-O2) AND ASan/UBSan/LSan modes.
cflags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic -Wconversion -Wsign-conversion
  -Wshadow -fno-fast-math -ffp-contract=off -frounding-math
  -fexcess-precision=standard -I include -I native
)
"${CC}" "${cflags[@]}" -O2 tests/t2_trainer/test_t2_trainer_native.c \
  "${build}/t2_trainer/libeshkol_transformer_t2_trainer_aot.a" \
  "${build}/t2_trainer/libeshkol_transformer_t2_trainer.a" \
  -o "${scratch}/test_t2_trainer_native"
"${CC}" "${cflags[@]}" -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer \
  tests/t2_trainer/test_t2_trainer_native.c \
  "${scratch}/t2_trainer_san/libeshkol_transformer_t2_trainer_aot.a" \
  "${scratch}/t2_trainer_san/libeshkol_transformer_t2_trainer.a" \
  -o "${scratch}/test_t2_trainer_native_san"

mkdir -p "${scratch}/native-normal-dir" "${scratch}/native-san-dir"
"${scratch}/test_t2_trainer_native" "${scratch}/native-normal-dir"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
  "${scratch}/test_t2_trainer_native_san" "${scratch}/native-san-dir"
printf 'T2_TRAINER native C11 normal + ASan/UBSan/LSan tests: PASS\n'

# 4. Independent Python reference BPE oracle parity test.
T2_TRAINER_NATIVE_BIN="${scratch}/test_t2_trainer_native" \
  python3 -m unittest -v tests/t2_trainer/test_t2_trainer_oracle.py
printf 'T2_TRAINER independent Python oracle merge & token parity: PASS\n'

# 5. Compile tests/t2_trainer/test_t2_trainer_aot.esk TWICE from fresh dirs and
#    verify byte-identical ELFs and identical multi-process stdout.
for run in 1 2; do
  out_dir="${scratch}/aot-${run}"
  mkdir -p "${out_dir}/cache" "${out_dir}/lib"
  bash scripts/build-t2-trainer.sh "${out_dir}/lib" normal >/dev/null
  if ! (
    cd "${out_dir}"
    timeout --foreground --kill-after=5s "${compiler_timeout}s" \
      env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR -u ESHKOL_AOT_MODULE_CACHE_DIR \
      ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="${out_dir}/cache" \
      "${runner}" --strict-types --optimize 0 --no-stdlib \
        -L "${out_dir}/lib" \
        --lib eshkol_transformer_t2_trainer_aot \
        --lib eshkol_transformer_t2_trainer \
        "${PROJECT_ROOT}/tests/t2_trainer/test_t2_trainer_aot.esk" \
        -o test_t2_trainer_aot >compile.log 2>&1
  ) || [[ ! -x "${out_dir}/test_t2_trainer_aot" ]] || grep -Fq 'ERROR:' "${out_dir}/compile.log"; then
    tail -120 "${out_dir}/compile.log" >&2
    die "T2 trainer Eshkol AOT compilation ${run} failed"
  fi
done

readelf -h "${scratch}/aot-1/test_t2_trainer_aot" >/dev/null
cmp "${scratch}/aot-1/test_t2_trainer_aot" "${scratch}/aot-2/test_t2_trainer_aot"
elf_sha="$(sha256sum "${scratch}/aot-1/test_t2_trainer_aot" | awk '{print $1}')"
printf 'T2_TRAINER Eshkol AOT byte-identical ELF SHA-256: %s\n' "${elf_sha}"

# 6. Run both AOT binaries through the full multi-process workflow:
#    train -> fresh-process reload + lossless encode/decode -> fresh-process
#    tamper / truncated / bad-magic / version-mismatch / symlink rejection.
for run in 1 2; do
  bin="${scratch}/aot-${run}/test_t2_trainer_aot"
  work="${scratch}/work-${run}"
  mkdir -p "${work}"
  art="${work}/trained.etbpe"

  {
    timeout --foreground --kill-after=5s 30s "${bin}" --mode train "${art}"
    [[ -f "${art}" ]] || die "AOT train did not write artifact"
    timeout --foreground --kill-after=5s 30s "${bin}" --mode reload "${art}"

    # Craft corrupted/invalid artifacts for fresh-process rejection checks.
    python3 - "${art}" "${work}" <<'PY'
from pathlib import Path
import sys

art = Path(sys.argv[1]).read_bytes()
work = Path(sys.argv[2])

# 1. Corrupt checksum (-8): flip 1 bit in merge table payload.
tampered = bytearray(art)
tampered[44] ^= 0x01
(work / "tampered.etbpe").write_bytes(bytes(tampered))

# 2. Truncated (-7): drop last 4 bytes.
(work / "truncated.etbpe").write_bytes(art[:-4])

# 3. Bad magic (-5): alter magic byte 0.
bad_magic = bytearray(art)
bad_magic[0] = ord("X")
(work / "bad_magic.etbpe").write_bytes(bytes(bad_magic))

# 4. Version mismatch (-6): bump version_major to 2.
bad_ver = bytearray(art)
bad_ver[8] = 2
bad_ver[9] = 0
(work / "bad_version.etbpe").write_bytes(bytes(bad_ver))
PY
    ln -s "${art}" "${work}/symlink.etbpe"

    timeout --foreground --kill-after=5s 30s "${bin}" --mode reject "${work}/tampered.etbpe" -8
    timeout --foreground --kill-after=5s 30s "${bin}" --mode reject "${work}/truncated.etbpe" -7
    timeout --foreground --kill-after=5s 30s "${bin}" --mode reject "${work}/bad_magic.etbpe" -5
    timeout --foreground --kill-after=5s 30s "${bin}" --mode reject "${work}/bad_version.etbpe" -6
    timeout --foreground --kill-after=5s 30s "${bin}" --mode reject "${work}/symlink.etbpe" -11
  } >"${scratch}/aot-${run}.stdout" 2>"${scratch}/aot-${run}.stderr"

  [[ ! -s "${scratch}/aot-${run}.stderr" ]] || {
    cat "${scratch}/aot-${run}.stderr" >&2
    die "AOT run ${run} emitted stderr"
  }
done

cmp "${scratch}/aot-1.stdout" "${scratch}/aot-2.stdout"
cat "${scratch}/aot-1.stdout"
grep -q '^T2_BPE_AOT_TRAIN ' "${scratch}/aot-1.stdout"
grep -q '^T2_BPE_AOT_RELOAD .* lossless_all256=PASS ' "${scratch}/aot-1.stdout"
grep -q '^T2_BPE_AOT_REJECT expected=-8 actual=-8 PASS$' "${scratch}/aot-1.stdout"
grep -q '^T2_BPE_AOT_REJECT expected=-7 actual=-7 PASS$' "${scratch}/aot-1.stdout"
grep -q '^T2_BPE_AOT_REJECT expected=-5 actual=-5 PASS$' "${scratch}/aot-1.stdout"
grep -q '^T2_BPE_AOT_REJECT expected=-6 actual=-6 PASS$' "${scratch}/aot-1.stdout"
grep -q '^T2_BPE_AOT_REJECT expected=-11 actual=-11 PASS$' "${scratch}/aot-1.stdout"

printf 'T2_TRAINER two fresh AOT compilations, byte-identical ELF and stdout: PASS\n'
