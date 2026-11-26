#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
for command in flock mktemp python3 sha256sum timeout; do
  require_command "${command}"
done

evidence="${DATA4_Q_T2_D1_EVIDENCE_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/eshkol-data4-q-t2-d1.XXXXXX")}"
if [[ -d "${evidence}" && -n "$(ls -A -- "${evidence}")" ]]; then
  die "evidence directory must be empty: ${evidence}"
fi
mkdir -p -- "${evidence}/build"
evidence="$(realpath -- "${evidence}")"

# The compiler and DATA4 archive can each use substantial memory.
exec 9>/home/gabe/.cache/eshkol-transformer-orchestration-build.lock
flock -x 9

runner="$(eshkol_build_dir)/eshkol-run"
provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
cc="$(tsv_value "${provenance}" cc_path)"
export CC="${cc}"
export CXX="$(tsv_value "${provenance}" cxx_path)"
bash "${PROJECT_ROOT}/scripts/build-data4-pipeline.sh" \
  "${evidence}/build/pipeline" normal >"${evidence}/archive.log" 2>&1

native_objects=()
for source in data_io checkpoint_io kernel_abi i64_tensor t1_i64_shell p1_identity; do
  object="${evidence}/build/${source}.o"
  extra_flags=()
  if [[ "${source}" == p1_identity ]]; then
    extra_flags=(-DET_P1_TRUSTED_BUILD=1)
  fi
  "${cc}" -std=c11 -Wall -Wextra -Werror -Wpedantic \
    -fstack-protector-all -fPIC -fvisibility=hidden -fno-common \
    "${extra_flags[@]}" \
    -I "${PROJECT_ROOT}/include" -I "${PROJECT_ROOT}/native" \
    -I "${PROJECT_ROOT}/src" \
    -c "${PROJECT_ROOT}/native/${source}.c" -o "${object}"
  native_objects+=("${object}")
done

timeout --foreground --kill-after=5s 300s "${runner}" \
  --strict-types --optimize 0 --no-stdlib \
  -I "${PROJECT_ROOT}/internal/p1/lib" \
  -I "${PROJECT_ROOT}/internal/c1/lib" \
  -I "${PROJECT_ROOT}/internal/t2/lib" \
  -I "${PROJECT_ROOT}/internal/t1/lib" \
  -I "${PROJECT_ROOT}/internal/data4/lib" \
  -I "${PROJECT_ROOT}/src" -I "${PROJECT_ROOT}/lib" \
  -I "${PROJECT_ROOT}/native" \
  -L "${evidence}/build/pipeline" \
  --lib eshkol_transformer_data4_pipeline \
  "${PROJECT_ROOT}/tests/data4/quality_t2_d1_witness.esk" \
  "${native_objects[@]}" -o "${evidence}/build/witness" \
  >"${evidence}/compile.stdout" 2>"${evidence}/compile.stderr"

"${evidence}/build/witness" reject "${evidence}/rejected-corpus" \
  >"${evidence}/reject.stdout" 2>"${evidence}/reject.stderr"
[[ ! -e "${evidence}/rejected-corpus" ]] || \
  die "rejected source IDs published a D1 corpus"
"${evidence}/build/witness" write "${evidence}/corpus" \
  >"${evidence}/write.stdout" 2>"${evidence}/write.stderr"
[[ -s "${evidence}/corpus/manifest.etm" ]] || die "missing D1 manifest"
[[ "$(cat "${evidence}/reject.stdout")" == \
   "duplicate-source rejected before D1 publication" ]] || die "wrong negative result"
[[ "$(cat "${evidence}/write.stdout")" == \
   "quality-selected alpha bravo charlie; T2/D1 tokens=20 readback=20 PASS" ]] || \
  die "wrong positive result"
[[ ! -s "${evidence}/reject.stderr" && ! -s "${evidence}/write.stderr" ]] || \
  die "witness emitted stderr"

(
  cd -- "${evidence}"
  sha256sum build/witness corpus/manifest.etm corpus/shard-*.ets \
    reject.stdout write.stdout > SHA256SUMS
)
printf 'DATA4-Q -> T2/D1 witness passed; evidence: %s\n' "${evidence}"
