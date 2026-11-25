#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

cd "${PROJECT_ROOT}"
verify_toolchain
export LC_ALL=C

export CC="${CC:-$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cc_path)}"
export CXX="${CXX:-$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cxx_path)}"
runner="$(eshkol_build_dir)/eshkol-run"

out_dir="${1:-${PROJECT_ROOT}/build/data4_pipeline}"
mkdir -p -- "${out_dir}"

# Build normal and sanitized archives (with deterministic archive & symbol audit).
bash scripts/build-data4-pipeline.sh "${out_dir}" normal
bash scripts/build-data4-pipeline.sh "${out_dir}/sanitize" sanitize

scratch="$(mktemp -d "${out_dir}/.data4-test.XXXXXX")"
diagnostic=''
cleanup() {
  local status=$?
  if [[ "${status}" != 0 && -n "${diagnostic}" && -s "${diagnostic}" ]]; then
    cat "${diagnostic}" >&2
  fi
  rm -rf -- "${scratch}"
}
trap cleanup EXIT

common_flags=(
  -Wall -Wextra -Werror -Wpedantic
  -fno-fast-math -ffp-contract=off
  -I "${PROJECT_ROOT}/include" -I "${PROJECT_ROOT}/native"
)
cflags=(-std=c11 "${common_flags[@]}" -frounding-math -fexcess-precision=standard)
cppflags=(-std=c++17 "${common_flags[@]}")
san_flags=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)

# 1. C++17 header and linkage test.
"${CXX}" "${cppflags[@]}" "${PROJECT_ROOT}/tests/data4/header_pipeline_cpp.cpp" \
  "${out_dir}/libeshkol_transformer_data4_pipeline.a" \
  -lm -o "${scratch}/header-pipeline-cpp"
"${scratch}/header-pipeline-cpp"
printf 'DATA4 pipeline C++17 header and linkage: PASS\n'

# 2. Native C11 test driver (normal mode).
"${CC}" "${cflags[@]}" -O2 "${PROJECT_ROOT}/tests/data4/test_data4_pipeline_native.c" \
  "${out_dir}/libeshkol_transformer_data4_pipeline.a" \
  -lm -o "${scratch}/native-normal"
mkdir -p -- "${scratch}/normal-data" "${scratch}/sanitize-data"
"${scratch}/native-normal" --oracle-out "${scratch}/normal-data" \
  > "${scratch}/normal-data/stdout"

# 3. Native C11 test driver under ASan/UBSan/LSan.
"${CC}" "${cflags[@]}" "${san_flags[@]}" \
  "${PROJECT_ROOT}/tests/data4/test_data4_pipeline_native.c" \
  "${out_dir}/sanitize/libeshkol_transformer_data4_pipeline.a" \
  -lm -o "${scratch}/native-san"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  "${scratch}/native-san" --oracle-out "${scratch}/sanitize-data" \
  > "${scratch}/sanitize-data/stdout"

cmp "${scratch}/normal-data/stdout" "${scratch}/sanitize-data/stdout"
cmp "${scratch}/normal-data/report.tsv" "${scratch}/sanitize-data/report.tsv"
cmp "${scratch}/normal-data/minhash.txt" "${scratch}/sanitize-data/minhash.txt"
cat "${scratch}/normal-data/stdout"
printf 'DATA4 pipeline native C11 normal + ASan/UBSan/LSan parity: PASS\n'

# 4. Independent Python oracle verification.
python3 "${PROJECT_ROOT}/tests/data4/test_data4_pipeline_oracle.py" \
  "${scratch}/native-normal" "${scratch}/normal-data"
printf 'DATA4 pipeline independent Python oracle: PASS\n'

# 5. Compiled Eshkol AOT verification: two fresh compilations -> byte-identical
#    ELFs and identical stdout.
for run in 1 2; do
  mkdir -p "${scratch}/aot-build-${run}" "${scratch}/aot-cache-${run}"
  diagnostic="${scratch}/aot-compile-${run}.log"
  timeout --foreground --kill-after=5s 60s \
    env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR -u ESHKOL_AOT_MODULE_CACHE_DIR \
    ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="${scratch}/aot-cache-${run}" \
    "${runner}" --strict-types --optimize 0 --no-stdlib \
      -L "${out_dir}" \
      --lib eshkol_transformer_data4_pipeline \
      "${PROJECT_ROOT}/tests/data4/test_data4_pipeline_aot.esk" \
      -o "${scratch}/aot-build-${run}/data4-pipeline-aot" >"${diagnostic}" 2>&1
  diagnostic="${scratch}/aot-run-${run}.stderr"
  timeout --foreground --kill-after=5s 30s \
    "${scratch}/aot-build-${run}/data4-pipeline-aot" \
    >"${scratch}/aot-run-${run}.stdout" 2>"${diagnostic}"
  [[ ! -s "${diagnostic}" ]] || die "compiled DATA4 AOT binary emitted stderr"
done
diagnostic=''

cmp "${scratch}/aot-build-1/data4-pipeline-aot" \
    "${scratch}/aot-build-2/data4-pipeline-aot"
cmp "${scratch}/aot-run-1.stdout" "${scratch}/aot-run-2.stdout"

awk '
  $0 ~ /^DATA4 AOT CORPUS input=9 kept=2 utf8=1 length=2 quality=1 repetition=1 exact=1 near=1 jaccard=[0-9]+\/[0-9]+ PASS$/ { ++corpus; next }
  $1 == "DATA4" && $2 == "AOT" && $3 == "REPORT" && NF == 7 &&
    $4 ~ /^bytes=[0-9]+$/ && $5 ~ /^sha256=[0-9a-f]+$/ && length($5) == 71 &&
    $6 == "tamper=REJECTED" && $7 == "PASS" { ++report; next }
  { exit 1 }
  END { if (corpus != 1 || report != 1) exit 1 }
' "${scratch}/aot-run-1.stdout" || {
  cat "${scratch}/aot-run-1.stdout" >&2
  die "unexpected DATA4 AOT stdout"
}

elf_sha256="$(sha256sum "${scratch}/aot-build-1/data4-pipeline-aot" | awk '{print $1}')"
cat "${scratch}/aot-run-1.stdout"
printf 'DATA4 AOT compiled ELF SHA-256: %s\n' "${elf_sha256}"
printf 'DATA4 AOT two fresh compilations, byte-identical ELF and stdout: PASS\n'
