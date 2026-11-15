#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
for command in cmp flock mktemp python3 sha256sum timeout; do
  require_command "${command}"
done

if [[ -n "${DATA4_EVIDENCE_DIR:-}" ]]; then
  evidence="${DATA4_EVIDENCE_DIR}"
  if [[ -d "${evidence}" && -n "$(ls -A -- "${evidence}")" ]]; then
    die "DATA4 evidence directory must be empty: ${evidence}"
  fi
else
  evidence="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-transformer-data4-native.XXXXXX")"
fi
mkdir -p -- "${evidence}"
evidence="$(realpath -- "${evidence}")"
build="${evidence}/build"
mkdir -p -- "${build}" "${evidence}/corpus-a" "${evidence}/corpus-b"

# This gate compiles the genuine D1/T2 trusted source closure, which can use
# multiple GiB. It never mutates the canonical installed package artifacts.
exec 9>/tmp/eshkol-transformer-heavy-build.lock
flock -x 9

runner="$(eshkol_build_dir)/eshkol-run"
provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
cc="$(tsv_value "${provenance}" cc_path)"
native_objects=()
for source in data_io checkpoint_io kernel_abi i64_tensor t1_i64_shell \
              p1_identity; do
  object="${build}/${source}.o"
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

/usr/bin/time -v timeout 300 "${runner}" --strict-types --optimize 0 \
  --no-stdlib \
  -I "${PROJECT_ROOT}/internal/p1/lib" \
  -I "${PROJECT_ROOT}/internal/c1/lib" \
  -I "${PROJECT_ROOT}/internal/t2/lib" \
  -I "${PROJECT_ROOT}/internal/t1/lib" \
  -I "${PROJECT_ROOT}/internal/data4/lib" \
  -I "${PROJECT_ROOT}/src" -I "${PROJECT_ROOT}/lib" \
  -I "${PROJECT_ROOT}/native" \
  "${PROJECT_ROOT}/tests/data4/data4_same_aggregate_runtime.esk" \
  "${native_objects[@]}" -o "${build}/data4-native" \
  >"${evidence}/compile.stdout" 2>"${evidence}/compile.stderr"

"${build}/data4-native" write "${evidence}/corpus-a" \
  >"${evidence}/report-a.tsv" 2>"${evidence}/write-a.stderr"
"${build}/data4-native" write "${evidence}/corpus-b" \
  >"${evidence}/report-b.tsv" 2>"${evidence}/write-b.stderr"
cmp "${evidence}/report-a.tsv" "${evidence}/report-b.tsv"
for leaf in manifest.etm shard-0000000000000000.ets \
            shard-0000000000000001.ets; do
  cmp "${evidence}/corpus-a/${leaf}" "${evidence}/corpus-b/${leaf}"
done
if "${build}/data4-native" write "${evidence}/corpus-a" \
    >"${evidence}/rewrite.stdout" 2>"${evidence}/rewrite.stderr"; then
  die "D1 accepted a preexisting corpus target"
fi
[[ ! -s "${evidence}/rewrite.stdout" ]] || \
  die "failed D1 rewrite emitted a report"
cmp "${evidence}/corpus-a/manifest.etm" \
    "${evidence}/corpus-b/manifest.etm"

PYTHONDONTWRITEBYTECODE=1 python3 \
  "${PROJECT_ROOT}/tests/data4/check_native_report.py" \
  "${evidence}/report-b.tsv" "${evidence}/corpus-b"
"${build}/data4-native" validate-corrupt "${evidence}/corpus-b" \
  >"${evidence}/corrupt.stdout" 2>"${evidence}/corrupt.stderr"
[[ "$(cat "${evidence}/corrupt.stdout")" == "D1 corrupt-data confirmed" ]]

(
  cd -- "${evidence}"
  sha256sum report-a.tsv report-b.tsv corpus-a/manifest.etm \
    corpus-a/shard-0000000000000000.ets \
    corpus-a/shard-0000000000000001.ets \
    build/data4-native > SHA256SUMS
)
printf 'DATA4 native gate passed; evidence: %s\n' "${evidence}"
