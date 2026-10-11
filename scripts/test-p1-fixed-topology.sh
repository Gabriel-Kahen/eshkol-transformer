#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

evidence="${1:-$(project_build_dir)/p1-fixed-topology}"
mkdir -p "${evidence}/overlay/transformer"
python3 "${PROJECT_ROOT}/scripts/check-p1-fixed-topology-source.py" \
  --output "${evidence}/overlay/transformer/module.esk" \
  >"${evidence}/source.stdout"
if [[ "${P1_FIXED_SOURCE_ONLY:-0}" == 1 ]]; then
  cat "${evidence}/source.stdout"
  exit 0
fi

verify_toolchain
for command in ar cmp timeout; do require_command "${command}"; done
runner="$(eshkol_build_dir)/eshkol-run"
provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
cc="$(tsv_value "${provenance}" cc_path)"
cxx="$(tsv_value "${provenance}" cxx_path)"
"${cc}" -std=c11 -Wall -Wextra -Werror -Wpedantic -Wconversion \
  -Wsign-conversion -Wshadow -fPIC -fvisibility=hidden -fno-common \
  -DET_P1_TRUSTED_BUILD=1 -I "${PROJECT_ROOT}/native" \
  -c "${PROJECT_ROOT}/native/p1_identity.c" \
  -o "${evidence}/p1_identity.o"
ar rcsD "${evidence}/libeshkol_transformer_p1_identity.a" \
  "${evidence}/p1_identity.o"
for run in a b; do
  env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
    XDG_CACHE_HOME="${evidence}/cache-${run}" \
    ESHKOL_CXX_COMPILER="${cxx}" \
    timeout --foreground --signal=TERM --kill-after=5s 600s \
    "${runner}" --strict-types --no-stdlib \
    -I "${evidence}/overlay" -I "${PROJECT_ROOT}/internal/p1/lib" \
    -I "${PROJECT_ROOT}/lib" -I "${PROJECT_ROOT}/native" \
    -I "${PROJECT_ROOT}/tests/p1/providers" \
    -L "${evidence}" --lib eshkol_transformer_p1_identity \
    "${PROJECT_ROOT}/tests/p1/fixed_topology_slot73_test.esk" \
    -o "${evidence}/fixed-${run}" \
    >"${evidence}/compile-${run}.stdout" \
    2>"${evidence}/compile-${run}.stderr"
  ESHKOL_ARENA_POISON=1 "${evidence}/fixed-${run}" \
    >"${evidence}/run-${run}.stdout" \
    2>"${evidence}/run-${run}.stderr"
  test ! -s "${evidence}/run-${run}.stderr"
  grep -E '^P1 fixed topology PASS: [1-9][0-9]*$' \
    "${evidence}/run-${run}.stdout" >/dev/null
done
cmp "${evidence}/run-a.stdout" "${evidence}/run-b.stdout"
cat "${evidence}/run-a.stdout"
