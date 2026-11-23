#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

cd "${PROJECT_ROOT}"
for command in ar cmp diff nm readelf sha256sum timeout; do
  require_command "${command}"
done
verify_toolchain
export LC_ALL=C PYTHONDONTWRITEBYTECODE=1

cc="${CC:-$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cc_path)}"
cxx="${CXX:-$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cxx_path)}"
runner="$(eshkol_build_dir)/eshkol-run"
compiler_timeout="${EVAL5_AOT_COMPILER_TIMEOUT_SECONDS:-480}"
[[ "${compiler_timeout}" =~ ^[1-9][0-9]*$ ]] || die "compiler timeout must be a positive integer"

mkdir -p "$(project_build_dir)"
build="$(realpath -- "$(project_build_dir)")"
export BUILD_DIR="${build}"
scratch="$(mktemp -d "${build}/.eval5-aot.XXXXXX")"
trap 'rm -rf -- "${scratch}"' EXIT

# 1. Build EVAL5 core and EVAL5 AOT bridge (normal + sanitize), auditing symbols & C++17 header.
bash scripts/build-eval5.sh "${build}/eval5" normal
bash scripts/build-eval5.sh "${scratch}/eval5-san" sanitize
bash scripts/build-eval5-aot.sh "${build}/eval5_aot" normal
bash scripts/build-eval5-aot.sh "${scratch}/eval5-aot-san" sanitize

# 2. C++17 linkage test against normal archives.
"${cxx}" -std=c++17 -Wall -Wextra -Werror -Wpedantic -O2 \
  -I include -I native \
  tests/eval5/header_bridge_cpp.cpp \
  "${build}/eval5_aot/libeshkol_transformer_eval5_aot.a" \
  "${build}/eval5/libeshkol_eval5.a" -lm \
  -o "${scratch}/header-bridge-cpp"
"${scratch}/header-bridge-cpp"
printf 'EVAL5 AOT C++17 header & linkage test: PASS\n'

# 3. Native C11 bridge tests in normal AND ASan/UBSan/LSan modes.
bridge_cflags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -Wconversion -Wsign-conversion
  -Wshadow -fno-fast-math -ffp-contract=off -frounding-math -fexcess-precision=standard
  -I include -I native)
"${cc}" "${bridge_cflags[@]}" -O2 tests/eval5/test_eval5_aot_bridge.c \
  "${build}/eval5_aot/libeshkol_transformer_eval5_aot.a" \
  "${build}/eval5/libeshkol_eval5.a" -lm -o "${scratch}/bridge-normal"
"${cc}" "${bridge_cflags[@]}" -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer \
  tests/eval5/test_eval5_aot_bridge.c \
  "${scratch}/eval5-aot-san/libeshkol_transformer_eval5_aot.a" \
  "${scratch}/eval5-san/libeshkol_eval5.a" -lm -o "${scratch}/bridge-san"

mkdir -p "${scratch}/bridge-normal-dir" "${scratch}/bridge-san-dir"
"${scratch}/bridge-normal" "${scratch}/bridge-normal-dir"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
  "${scratch}/bridge-san" "${scratch}/bridge-san-dir"
printf 'EVAL5 AOT C11 bridge normal + ASan/UBSan/LSan tests: PASS\n'

# 4. Build live MOD5 / C1 / P1 / I2 prerequisites and compile Eshkol AOT runner TWICE.
bash scripts/build-mod5.sh
bash scripts/build-c1.sh
bash scripts/build-p1-identity.sh "${scratch}/p1" normal trusted

native_cflags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -Wconversion -Wsign-conversion
  -Wshadow -fPIC -fvisibility=hidden -fno-common -fstack-protector-all
  -ffp-contract=off -fexcess-precision=standard -frounding-math
  -I include -I native)
objects=()
for source in i2_wave2_package_bridge data_io kernel_abi i64_tensor t1_i64_shell f32_tensor mod5_eshkol_bridge; do
  object="${scratch}/${source}.o"
  "${cc}" "${native_cflags[@]}" -DET_I2_NATIVE_HELPERS_ONLY -c "native/${source}.c" -o "${object}"
  objects+=("${object}")
done
ar rcsD "${scratch}/libeval5_mod5_native.a" "${objects[@]}"

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
        -L "${scratch}" --lib eval5_mod5_native \
        -L "${PROJECT_ROOT}/build/mod5" --lib eshkol_transformer_mod5 \
        -L "${build}/eval5_aot" --lib eshkol_transformer_eval5_aot \
        -L "${build}/eval5" --lib eshkol_eval5 \
        "${PROJECT_ROOT}/tests/eval5/test_eval5_mod5_runner.esk" \
        -o eval5-mod5-runner >compile.log 2>&1
  ) || [[ ! -x "${output}/eval5-mod5-runner" ]] || grep -Fq 'ERROR:' "${output}/compile.log"; then
    tail -120 "${output}/compile.log" >&2
    die "EVAL5 MOD5 AOT compilation ${run} failed"
  fi
done

readelf -h "${scratch}/a/eval5-mod5-runner" >/dev/null
cmp "${scratch}/a/eval5-mod5-runner" "${scratch}/b/eval5-mod5-runner"
elf_sha="$(sha256sum "${scratch}/a/eval5-mod5-runner" | awk '{print $1}')"
printf 'EVAL5 MOD5 AOT byte-identical ELF SHA-256: %s\n' "${elf_sha}"

# 5. Live pipeline: write D1 corpora -> train MOD5 + save C1 -> contaminated fail-closed ->
#    clean eval + manifest publish -> third-process manifest verify -> tampered & symlink reject.
aot_bin="${scratch}/a/eval5-mod5-runner"
work="${scratch}/pipeline"
mkdir -p "${work}/train" "${work}/eval-clean" "${work}/eval-exact" "${work}/eval-ngram" "${work}/eval-both"

"${aot_bin}" --mode write-corpora "${work}"

train_out="$("${aot_bin}" --mode train-save "${work}/train" "${work}/trained.etcp")"
printf '%s\n' "${train_out}"
grep -q '^TRAIN_GRADIENTS 12$' <<< "${train_out}"
grep -q '^TRAIN_DONE steps=12 ' <<< "${train_out}"
[[ -f "${work}/trained.etcp" ]] || die "C1 checkpoint not written"

# Contaminated splits must fail closed (exit 7) and refuse to publish any manifest file.
for split in eval-exact eval-ngram eval-both; do
  rej_manifest="${work}/${split}.manifest"
  set +e
  contam_out="$("${aot_bin}" --mode eval "${work}/train" "${work}/${split}" "${work}/trained.etcp" "${rej_manifest}")"
  contam_rc=$?
  set -e
  printf '[%s] %s\n' "${split}" "${contam_out}"
  [[ "${contam_rc}" -eq 7 ]] || die "${split} expected fail-closed exit 7, got ${contam_rc}"
  grep -q '^CONTAMINATED_REJECTED .* write_status=-7$' <<< "${contam_out}"
  [[ ! -e "${rej_manifest}" ]] || die "contaminated split ${split} published manifest ${rej_manifest}"
done
printf 'EVAL5 MOD5 contaminated splits (exact, ngram, both) fail-closed rejection: PASS\n'

# Clean held-out split in a fresh process: reloads C1, audits clean, scores LM + MC, publishes manifest.
clean_manifest="${work}/clean.manifest"
clean_out="$("${aot_bin}" --mode eval "${work}/train" "${work}/eval-clean" "${work}/trained.etcp" "${clean_manifest}")"
printf '%s\n' "${clean_out}"
grep -q '^EVAL_CLEAN ' <<< "${clean_out}"
[[ -f "${clean_manifest}" ]] || die "clean manifest not published"

# Third fresh process: reads clean manifest via O_NOFOLLOW, verifies SHA-256 + schema + digests.
verify_out="$("${aot_bin}" --mode verify-manifest "${clean_manifest}" "${work}/train" "${work}/eval-clean" "${work}/trained.etcp")"
printf '%s\n' "${verify_out}"
grep -q '^MANIFEST_VERIFIED ' <<< "${verify_out}"

clean_tail="${clean_out#EVAL_CLEAN }"
verify_tail="${verify_out#MANIFEST_VERIFIED }"
[[ "${clean_tail}" == "${verify_tail}" ]] || die "manifest reload metrics/digest mismatch: '${clean_tail}' vs '${verify_tail}'"
printf 'EVAL5 MOD5 clean eval -> atomic manifest -> third-process reload verification: PASS\n'

# Tampered manifest rejection in fresh process (checksum error = -9).
tampered_manifest="${work}/tampered.manifest"
sed 's/^suite=eval5-mod5-live$/suite=eval5-mod5-fake/' "${clean_manifest}" > "${tampered_manifest}"
set +e
tamper_out="$("${aot_bin}" --mode verify-manifest "${tampered_manifest}" "${work}/train" "${work}/eval-clean" "${work}/trained.etcp")"
tamper_rc=$?
set -e
printf '[tampered] %s\n' "${tamper_out}"
[[ "${tamper_rc}" -eq 9 ]] || die "tampered manifest expected exit 9, got ${tamper_rc}"
grep -q '^MANIFEST_REJECTED status=-9$' <<< "${tamper_out}"

# Symlink manifest rejection in fresh process (O_NOFOLLOW IO error = -11).
ln -s "${clean_manifest}" "${work}/symlink.manifest"
set +e
link_out="$("${aot_bin}" --mode verify-manifest "${work}/symlink.manifest" "${work}/train" "${work}/eval-clean" "${work}/trained.etcp")"
link_rc=$?
set -e
printf '[symlink] %s\n' "${link_out}"
[[ "${link_rc}" -eq 9 ]] || die "symlink manifest expected exit 9, got ${link_rc}"
grep -q '^MANIFEST_REJECTED status=-11$' <<< "${link_out}"

printf 'EVAL5 MOD5 tampered and symlink manifest rejection: PASS\n'
printf 'EVAL5 AOT full pipeline (bridge, ASan/UBSan/LSan, byte-identical ELF, D1/MOD5/C1/EVAL5): PASS\n'
