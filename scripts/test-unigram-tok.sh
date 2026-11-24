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
compiler_timeout="${UNIGRAM_TOK_COMPILER_TIMEOUT_SECONDS:-180}"
[[ "${compiler_timeout}" =~ ^[1-9][0-9]*$ ]] || die "compiler timeout must be a positive integer"

mkdir -p "$(project_build_dir)"
build="$(realpath -- "$(project_build_dir)")"
export BUILD_DIR="${build}"
scratch="$(mktemp -d "${build}/.unigram-tok-test.XXXXXX")"
trap 'rm -rf -- "${scratch}"' EXIT

# 1. Build deterministic archives (normal + sanitize) with symbol & C++17 header audits.
bash scripts/build-unigram-tok.sh "${build}/unigram_tok" normal
bash scripts/build-unigram-tok.sh "${scratch}/unigram_tok_san" sanitize

# 2. C++17 header & archive linkage test.
"${CXX}" -std=c++17 -Wall -Wextra -Werror -Wpedantic -O2 \
  -I include -I native \
  tests/unigram_tok/header_unigram_cpp.cpp \
  "${build}/unigram_tok/libeshkol_transformer_unigram_tok_aot.a" \
  "${build}/unigram_tok/libeshkol_transformer_unigram_tok.a" \
  -lm -o "${scratch}/header_unigram_cpp"
"${scratch}/header_unigram_cpp"
printf 'UNIGRAM_TOK C++17 header and archive linkage: PASS\n'

# 3. Native C11 tests in normal (-O2) AND ASan/UBSan/LSan modes.
cflags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic -Wconversion -Wsign-conversion
  -Wshadow -fno-fast-math -ffp-contract=off -frounding-math
  -fexcess-precision=standard -I include -I native
)
"${CC}" "${cflags[@]}" -O2 tests/unigram_tok/test_unigram_tok_native.c \
  "${build}/unigram_tok/libeshkol_transformer_unigram_tok_aot.a" \
  "${build}/unigram_tok/libeshkol_transformer_unigram_tok.a" \
  -lm -o "${scratch}/test_unigram_tok_native"
"${CC}" "${cflags[@]}" -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer \
  tests/unigram_tok/test_unigram_tok_native.c \
  "${scratch}/unigram_tok_san/libeshkol_transformer_unigram_tok_aot.a" \
  "${scratch}/unigram_tok_san/libeshkol_transformer_unigram_tok.a" \
  -lm -o "${scratch}/test_unigram_tok_native_san"

mkdir -p "${scratch}/native-normal-dir" "${scratch}/native-san-dir"
"${scratch}/test_unigram_tok_native" "${scratch}/native-normal-dir"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
  "${scratch}/test_unigram_tok_native_san" "${scratch}/native-san-dir"
printf 'UNIGRAM_TOK native C11 normal + ASan/UBSan/LSan tests: PASS\n'

# 4. Independent Python binary64 reference Viterbi + FB + EM oracle parity test.
UNIGRAM_TOK_NATIVE_BIN="${scratch}/test_unigram_tok_native" \
  python3 -m unittest -v tests/unigram_tok/test_unigram_tok_oracle.py
printf 'UNIGRAM_TOK independent Python binary64 oracle parity: PASS\n'

# 5. Compile tests/unigram_tok/test_unigram_tok_aot.esk TWICE from fresh dirs
#    and verify byte-identical ELFs and identical multi-process stdout.
for run in 1 2; do
  out_dir="${scratch}/aot-${run}"
  mkdir -p "${out_dir}/cache" "${out_dir}/lib"
  bash scripts/build-unigram-tok.sh "${out_dir}/lib" normal >/dev/null
  if ! (
    cd "${out_dir}"
    timeout --foreground --kill-after=5s "${compiler_timeout}s" \
      env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR -u ESHKOL_AOT_MODULE_CACHE_DIR \
      ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="${out_dir}/cache" \
      "${runner}" --strict-types --optimize 0 --no-stdlib \
        -L "${out_dir}/lib" \
        --lib eshkol_transformer_unigram_tok_aot \
        --lib eshkol_transformer_unigram_tok \
        "${PROJECT_ROOT}/tests/unigram_tok/test_unigram_tok_aot.esk" \
        -o test_unigram_tok_aot >compile.log 2>&1
  ) || [[ ! -x "${out_dir}/test_unigram_tok_aot" ]] || grep -Fq 'ERROR:' "${out_dir}/compile.log"; then
    tail -120 "${out_dir}/compile.log" >&2
    die "UNIGRAM_TOK Eshkol AOT compilation ${run} failed"
  fi
done

readelf -h "${scratch}/aot-1/test_unigram_tok_aot" >/dev/null
cmp "${scratch}/aot-1/test_unigram_tok_aot" "${scratch}/aot-2/test_unigram_tok_aot"
elf_sha="$(sha256sum "${scratch}/aot-1/test_unigram_tok_aot" | awk '{print $1}')"
printf 'UNIGRAM_TOK Eshkol AOT byte-identical ELF SHA-256: %s\n' "${elf_sha}"

# 6. Run both AOT binaries through the full multi-process workflow:
#    train (+ FD + negatives) -> fresh-process reload + all-256/NUL/UTF-8 round-trip
#    -> fresh-process tamper / truncated / bad-magic / version-mismatch / symlink rejection.
for run in 1 2; do
  bin="${scratch}/aot-${run}/test_unigram_tok_aot"
  work="${scratch}/work-${run}"
  mkdir -p "${work}"
  art="${work}/trained.etug"

  {
    timeout --foreground --kill-after=5s 60s "${bin}" --mode train "${art}"
    [[ -f "${art}" ]] || die "AOT train did not write artifact"
    timeout --foreground --kill-after=5s 30s "${bin}" --mode reload "${art}"

    python3 - "${art}" "${work}" <<'PY'
from pathlib import Path
import sys

art = Path(sys.argv[1]).read_bytes()
work = Path(sys.argv[2])

# 1. Corrupt checksum (-8): flip 1 bit in piece payload.
tampered = bytearray(art)
tampered[48] ^= 0x01
(work / "tampered.etug").write_bytes(bytes(tampered))

# 2. Truncated (-7): drop last 4 bytes.
(work / "truncated.etug").write_bytes(art[:-4])

# 3. Bad magic (-5): alter magic byte 0.
bad_magic = bytearray(art)
bad_magic[0] = ord("X")
(work / "bad_magic.etug").write_bytes(bytes(bad_magic))

# 4. Version mismatch (-6): bump version_major to 2.
bad_ver = bytearray(art)
bad_ver[8] = 2
bad_ver[9] = 0
(work / "bad_version.etug").write_bytes(bytes(bad_ver))
PY
    ln -s "${art}" "${work}/symlink.etug"

    timeout --foreground --kill-after=5s 30s "${bin}" --mode reject "${work}/tampered.etug" -8
    timeout --foreground --kill-after=5s 30s "${bin}" --mode reject "${work}/truncated.etug" -7
    timeout --foreground --kill-after=5s 30s "${bin}" --mode reject "${work}/bad_magic.etug" -5
    timeout --foreground --kill-after=5s 30s "${bin}" --mode reject "${work}/bad_version.etug" -6
    timeout --foreground --kill-after=5s 30s "${bin}" --mode reject "${work}/symlink.etug" -11
  } >"${scratch}/aot-${run}.stdout" 2>"${scratch}/aot-${run}.stderr"

  [[ ! -s "${scratch}/aot-${run}.stderr" ]] || {
    cat "${scratch}/aot-${run}.stderr" >&2
    die "AOT run ${run} emitted stderr"
  }
done

cmp "${scratch}/aot-1.stdout" "${scratch}/aot-2.stdout"
cat "${scratch}/aot-1.stdout"

# Verify fresh-process train vs reload produced identical tokens, vscore, fb_log_z, and sha256.
python3 - "${scratch}/aot-1.stdout" <<'PY'
from pathlib import Path
import sys

lines = Path(sys.argv[1]).read_text().splitlines()
train_kv = {}
reload_kv = {}
for ln in lines:
    if ln.startswith("UNIGRAM_AOT_TRAIN "):
        for tok in ln.split()[1:]:
            k, v = tok.split("=", 1)
            train_kv[k] = v
    elif ln.startswith("UNIGRAM_AOT_RELOAD "):
        for tok in ln.split()[1:]:
            k, v = tok.split("=", 1)
            reload_kv[k] = v

assert train_kv and reload_kv, "missing train or reload summary line"
for key in ("vocab", "learned", "raw_bytes", "encoded_tokens", "vscore", "fb_log_z", "tokens", "sha256"):
    assert train_kv[key] == reload_kv[key], f"train vs reload mismatch on {key}: {train_kv[key]} != {reload_kv[key]}"
PY

grep -q '^UNIGRAM_AOT_TRAIN .* neg_checks=PASS ' "${scratch}/aot-1.stdout"
grep -q '^UNIGRAM_AOT_RELOAD .* lossless_all256=PASS ' "${scratch}/aot-1.stdout"
grep -q '^UNIGRAM_AOT_REJECT expected=-8 actual=-8 PASS$' "${scratch}/aot-1.stdout"
grep -q '^UNIGRAM_AOT_REJECT expected=-7 actual=-7 PASS$' "${scratch}/aot-1.stdout"
grep -q '^UNIGRAM_AOT_REJECT expected=-5 actual=-5 PASS$' "${scratch}/aot-1.stdout"
grep -q '^UNIGRAM_AOT_REJECT expected=-6 actual=-6 PASS$' "${scratch}/aot-1.stdout"
grep -q '^UNIGRAM_AOT_REJECT expected=-11 actual=-11 PASS$' "${scratch}/aot-1.stdout"

printf 'UNIGRAM_TOK two fresh AOT compilations, byte-identical ELF and stdout: PASS\n'
