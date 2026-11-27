#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

cd "${PROJECT_ROOT}"
for command in ar cmp find git python3 rg sha256sum sort timeout xargs; do
  require_command "${command}"
done
export LC_ALL=C

if [[ -n "${MOD5_C2_EVIDENCE_DIR:-}" ]]; then
  work="${MOD5_C2_EVIDENCE_DIR}"
  [[ "${work}" == /* && -d "${work}" ]] || \
    die "MOD5_C2_EVIDENCE_DIR must be an existing absolute directory"
  [[ -z "$(find "${work}" -mindepth 1 -print -quit)" ]] || \
    die "MOD5_C2_EVIDENCE_DIR must be empty"
else
  mkdir -p "$(project_build_dir)"
  work="$(mktemp -d "$(project_build_dir)/.mod5-c2.XXXXXX")"
fi

diagnostics() {
  local status=$?
  if (( status != 0 )); then
    for log in native-compile.stdout native-compile.stderr \
               compile.stdout compile.stderr runtime.stdout runtime.stderr; do
      if [[ -s "${work}/${log}" ]]; then
        tail -n 80 "${work}/${log}" >&2
      fi
    done
  fi
  if [[ -z "${MOD5_C2_EVIDENCE_DIR:-}" ]]; then
    rm -rf -- "${work}"
  fi
  return "${status}"
}
trap diagnostics EXIT

source_snapshot() {
  local label=$1 first last
  first="$(git rev-parse --verify HEAD)" || return
  GIT_OPTIONAL_LOCKS=0 git status --porcelain=v1 --untracked-files=all \
    --ignore-submodules=none >"${work}/source-${label}.status" || return
  last="$(git rev-parse --verify HEAD)" || return
  if [[ "${first}" != "${last}" ]]; then
    printf '%s\n' "HEAD moved during status observation" \
      >>"${work}/source-${label}.status" || return
  fi
  printf '%s\n' "${first}"
}
if [[ -n "${MOD5_C2_EVIDENCE_DIR:-}" ]]; then
  head_before="$(source_snapshot before)"
fi

verify_toolchain
cc=
cxx=
resolve_provenance_compilers cc cxx \
  "${CC:-$(lock_value supported_cc)}" "${CXX:-$(lock_value supported_cxx)}"
runner="$(eshkol_build_dir)/eshkol-run"

mkdir "${work}/native" "${work}/cache" "${work}/corpus" "${work}/checkpoints"
PYTHONDONTWRITEBYTECODE=1 python3 - "${work}/corpus" <<'PY'
from pathlib import Path
import sys

from tests.d2.test_resources import write_d1_resource
from tests.tr3b.prepare_fixture import FINGERPRINT

write_d1_resource(Path(sys.argv[1]),
                  ((3, 197, 41, 7, 11, 19, 47, 128, 6, 211, 2, 108, 71),),
                  fingerprint=FINGERPRINT, vocab=256)
PY

flags=(-std=c11 -O2 -Wall -Wextra -Werror -Wpedantic -Wconversion -Wshadow
       -Wsign-conversion -ffp-contract=off -fexcess-precision=standard
       -frounding-math -fno-fast-math -fPIC -fvisibility=hidden
       -fno-common -fstack-protector-all -I include -I native)
compile() {
  local source=$1
  shift
  "${cc}" "${flags[@]}" "$@" -c "${source}" \
    -o "${work}/native/$(basename "${source%.c}").o"
}
{
  compile native/i2_wave2_package_bridge.c \
    -DET_I2_NATIVE_HELPERS_ONLY -DET_C2_I2_MODEL_COPY
  compile native/o2_wave2_package_bridge.c \
    -DET_O2_NATIVE_HELPERS_ONLY -DET_C2_O2_RECONSTRUCT_BRIDGE
  compile native/p1_identity.c -DET_P1_TRUSTED_BUILD=1
  compile native/k2_capabilities.c -DET_C2_CARRIER_FACTORIES
  for source in ac4_checkpointing mod5_primitives_provider data_io checkpoint_io \
                kernel_abi i64_tensor t1_i64_shell f32_tensor d2_native o2_optimizer \
                c2_x1_canonical c2_checkpoint_codec c2_checkpoint_format \
                c2_checkpoint_save_bridge c2_checkpoint_reader c2_checkpoint_core \
                c2_checkpoint_load_bridge c2_checkpoint_inspect_bridge; do
    compile "native/${source}.c"
  done
  compile tests/mod5/mod5_c2_test_bridge.c
  ar rcsD "${work}/native/libmod5_c2_witness.a" "${work}/native/"*.o
} >"${work}/native-compile.stdout" 2>"${work}/native-compile.stderr"

env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR -u ESHKOL_AOT_MODULE_CACHE_DIR \
  ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="${work}/cache" \
  ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" ESHKOL_CXX_COMPILER="${cxx}" \
  timeout --foreground --signal=TERM --kill-after=5s 900s "${runner}" \
    --strict-types --no-stdlib -O 0 -I native -I internal/p1/lib \
    -I internal/c1/lib -I internal/t2/lib -I internal/t1/lib -I internal/d2/lib \
    -I src -I lib -L "${work}/native" --lib mod5_c2_witness \
    tests/mod5/mod5_c2_continuation.esk -o "${work}/mod5-c2" \
    >"${work}/compile.stdout" 2>"${work}/compile.stderr"
test ! -s "${work}/compile.stderr"
ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM --kill-after=5s \
  120s "${work}/mod5-c2" "${work}/corpus" "${work}/checkpoints" \
  >"${work}/runtime.stdout" 2>"${work}/runtime.stderr"
test ! -s "${work}/runtime.stderr"
rg -x 'MOD5/C2 PASS [0-9]+ checks, 2 \+ 4 updates, 27 exact tensors' \
  "${work}/runtime.stdout"
cmp "${work}/checkpoints/split.c2" "${work}/checkpoints/loaded.c2"
cmp "${work}/checkpoints/split.c2" "${work}/checkpoints/receiver.c2"
for update in 3 4 5 6; do
  cmp "${work}/checkpoints/source-${update}.c2" \
      "${work}/checkpoints/receiver-${update}.c2"
done

if [[ -n "${MOD5_C2_EVIDENCE_DIR:-}" ]]; then
  head_after="$(source_snapshot after)"
  runner_checksum="$(sha256sum <"${runner}")"
  result="$(cat "${work}/runtime.stdout")"
  {
    if [[ "${head_before}" == "${head_after}" && \
          ! -s "${work}/source-before.status" && ! -s "${work}/source-after.status" ]]; then
      source_tree="$(git rev-parse --verify "${head_before}^{tree}")"
      printf 'source_status\tclean\n'
      printf 'source_commit\t%s\n' "${head_before}"
      printf 'source_tree\t%s\n' "${source_tree}"
    else
      printf 'source_status\tworking-tree\n'
      printf 'observed_head_before\t%s\n' "${head_before}"
      printf 'observed_head_after\t%s\n' "${head_after}"
    fi
    printf 'runner_sha256\t%s\n' "${runner_checksum%% *}"
    printf 'result\t%s\n' "${result}"
  } >"${work}/manifest.tsv"
  (
    cd "${work}"
    {
      printf '%s\0' manifest.tsv source-before.status source-after.status \
        native/libmod5_c2_witness.a native-compile.stdout native-compile.stderr \
        mod5-c2 compile.stdout compile.stderr runtime.stdout runtime.stderr
      find corpus checkpoints -type f -print0
    } | sort -z | xargs -0 sha256sum >SHA256SUMS
  )
  seal_checksum="$(sha256sum <"${work}/SHA256SUMS")"
  printf 'MOD5/C2 evidence: %s\n' "${work}"
  printf 'MOD5/C2 evidence seal: %s\n' "${seal_checksum%% *}"
fi
