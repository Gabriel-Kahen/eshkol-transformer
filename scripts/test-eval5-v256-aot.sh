#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

[[ "${EVAL5_V256_AOT_SLOT:-0}" == 1 ]] ||
  die "EVAL5 V256 AOT requires an assigned build slot (EVAL5_V256_AOT_SLOT=1)"
cd "${PROJECT_ROOT}"
for command in ar cmp python3 timeout; do require_command "${command}"; done
verify_toolchain
export LC_ALL=C PYTHONDONTWRITEBYTECODE=1

cc="$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cc_path)"
runner="$(eshkol_build_dir)/eshkol-run"
compiler_timeout="${EVAL5_V256_AOT_COMPILER_TIMEOUT_SECONDS:-480}"
[[ "${compiler_timeout}" =~ ^[1-9][0-9]*$ ]] || die "invalid compiler timeout"
mkdir -p "$(project_build_dir)"
build="$(realpath -- "$(project_build_dir)")"
export BUILD_DIR="${build}"
if [[ -n "${EVAL5_V256_EVIDENCE_DIR:-}" ]]; then
  [[ ! -e "${EVAL5_V256_EVIDENCE_DIR}" && ! -L "${EVAL5_V256_EVIDENCE_DIR}" ]] ||
    die "evidence directory must be fresh"
  mkdir -p -- "${EVAL5_V256_EVIDENCE_DIR}"
  scratch="$(realpath -- "${EVAL5_V256_EVIDENCE_DIR}")"
else
  scratch="$(mktemp -d "${build}/.eval5-v256.XXXXXX")"
  trap 'if [[ -d "${scratch}/fixture" ]]; then chmod -R u+w -- "${scratch}/fixture"; fi; rm -rf -- "${scratch}"' EXIT
fi

fixture="${scratch}/fixture"
python3 -m tests.eval5.mod5_v256_fixture build "${fixture}" > "${scratch}/fixture.json"
python3 -m tests.eval5.mod5_v256_fixture verify "${fixture}" > /dev/null
python3 -m tests.eval5.build_mod5_v256_witness "${fixture}" "${scratch}/witness.esk"
chmod -R a-w -- "${fixture}"

bash scripts/build-eval5.sh "${build}/eval5" normal
bash scripts/build-mod5.sh
bash scripts/build-c1.sh
bash scripts/build-p1-identity.sh "${scratch}/p1" normal trusted

native_cflags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -Wconversion
  -Wsign-conversion -Wshadow -fPIC -fvisibility=hidden -fno-common
  -fstack-protector-all -ffp-contract=off -fexcess-precision=standard
  -frounding-math -I include -I native -I tests/eval5)
objects=()
for source in i2_wave2_package_bridge data_io kernel_abi i64_tensor t1_i64_shell f32_tensor mod5_eshkol_bridge; do
  object="${scratch}/${source}.o"
  "${cc}" "${native_cflags[@]}" -DET_I2_NATIVE_HELPERS_ONLY \
    -c "native/${source}.c" -o "${object}"
  objects+=("${object}")
done
"${cc}" "${native_cflags[@]}" -c tests/eval5/mod5_v256_score_bridge.c \
  -o "${scratch}/mod5_v256_score_bridge.o"
objects+=("${scratch}/mod5_v256_score_bridge.o")
ar rcsD "${scratch}/libeval5_v256_native.a" "${objects[@]}"

mkdir -p "${scratch}/cache"
if ! (
  cd "${scratch}"
  timeout --foreground --kill-after=5s "${compiler_timeout}s" \
    env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR -u ESHKOL_AOT_MODULE_CACHE_DIR \
    ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="${scratch}/cache" \
    "${runner}" --strict-types --optimize 0 --no-stdlib \
      -I "${PROJECT_ROOT}/internal/c1/lib" -I "${PROJECT_ROOT}/internal/p1/lib" \
      -I "${PROJECT_ROOT}/internal/t1/lib" -I "${PROJECT_ROOT}/src" \
      -I "${PROJECT_ROOT}/lib" -I "${PROJECT_ROOT}/native" \
      -L "${scratch}/p1/trusted" --lib eshkol_transformer_p1_identity \
      -L "${build}/c1" --lib eshkol_transformer_checkpoint_io \
      -L "${scratch}" --lib eval5_v256_native \
      -L "${build}/mod5" --lib eshkol_transformer_mod5 \
      -L "${build}/eval5" --lib eshkol_eval5 \
      "${scratch}/witness.esk" -o witness > compile.log 2>&1
) || [[ ! -x "${scratch}/witness" ]] || grep -Fq 'ERROR:' "${scratch}/compile.log"; then
  tail -100 "${scratch}/compile.log" >&2
  die "EVAL5 V256 AOT compilation failed"
fi

"${scratch}/witness" "${fixture}/tokenizer.etok" "${fixture}/train" \
  "${fixture}/eval-clean" "${fixture}/model.etcp" > "${scratch}/score.out"
python3 -m tests.eval5.check_mod5_v256_witness "${fixture}" "${scratch}/score.out"

"${scratch}/witness" --check-c1 "${fixture}/model.etcp" > "${scratch}/c1-good.out"
grep -Fxq C1_CHECKED "${scratch}/c1-good.out"
python3 - "${fixture}/model.etcp" "${scratch}/wrong-shape.etcp" <<'PY'
from dataclasses import replace
from pathlib import Path
import sys
from tools.io5.safetensors_interop import decode_c1, encode_c1
state = decode_c1(Path(sys.argv[1]).read_bytes())
tensors = list(state.tensors)
tensors[9] = replace(tensors[9], shape=(128, 16), data=tensors[9].data[:128 * 16 * 4])
Path(sys.argv[2]).write_bytes(encode_c1(replace(state, tensors=tuple(tensors))))
PY
set +e
"${scratch}/witness" --check-c1 "${scratch}/wrong-shape.etcp" > "${scratch}/c1-bad.out"
shape_status=$?
"${scratch}/witness" "${fixture}/tokenizer.etok" "${fixture}/train" \
  "${fixture}/eval-clean" "${scratch}/wrong-shape.etcp" > "${scratch}/c1-identity-bad.out"
checkpoint_identity_status=$?
"${scratch}/witness" "${fixture}/tokenizer.etok" "${fixture}/eval-clean" \
  "${fixture}/train" "${fixture}/model.etcp" > "${scratch}/split-bad.out"
identity_status=$?
set -e
[[ "${shape_status}" -eq 2 && "${checkpoint_identity_status}" -eq 2 && "${identity_status}" -eq 2 ]] ||
  die "negative C1 shape, C1 identity, or D1 split identity was accepted"
grep -Fq 'unexpected MOD5 parameter metadata' "${scratch}/c1-bad.out"
grep -Fq 'C1 checkpoint SHA-256 mismatch' "${scratch}/c1-identity-bad.out"
grep -Fq 'D1 manifest SHA-256 mismatch' "${scratch}/split-bad.out"
! grep -q '^SCORE' "${scratch}/c1-bad.out" "${scratch}/c1-identity-bad.out" "${scratch}/split-bad.out" ||
  die "rejected input produced a SCORE line"
printf 'EVAL5 V256 AOT live score, C1 shape/identity, and D1 identity: PASS\n'
