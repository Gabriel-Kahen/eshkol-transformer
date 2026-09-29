#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
for command in ar cmp python3 timeout; do require_command "${command}"; done
[[ -x /usr/bin/time ]] || die "GNU time is required for horizon measurements"
[[ $# -le 1 ]] || die "usage: $0 [FRESH_OUTPUT_DIRECTORY]"
out="${1:-$(project_build_dir)/e3-single-invocation}"
[[ ! -e "${out}" ]] || die "horizon output directory must be fresh: ${out}"
mkdir -p "${out}"

artifact="${out}/installed"
/usr/bin/bash "${PROJECT_ROOT}/scripts/build-e3-diagnostic-public.sh" "${artifact}" \
  >"${out}/package.stdout" 2>"${out}/package.stderr"
cmp "${PROJECT_ROOT}/native/e3_diagnostic_public_package_defined_symbols.txt" \
  "${artifact}/e3_diagnostic_public_package.o.evidence/global-defined.txt"

provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
cc="$(tsv_value "${provenance}" cc_path)"
cxx="$(tsv_value "${provenance}" cxx_path)"
"${cc}" -std=c11 -O2 -Wall -Wextra -Werror \
  -c "${PROJECT_ROOT}/tests/e3_horizon/late_shard.c" \
  -o "${out}/late_shard.o"
ar p "${artifact}/libeshkol_transformer_e3_diagnostic.a" \
  e3_diagnostic_public_package.o >"${out}/package.o"
cmp "${artifact}/e3_diagnostic_public_package.o" "${out}/package.o"
ar rcsD "${out}/libeshkol_transformer_e3_horizon_test.a" \
  "${out}/package.o" "${out}/late_shard.o"

runner="$(eshkol_build_dir)/eshkol-run"
compiler_timeout="${E3_COMPILER_TIMEOUT_SECONDS:-900}"
# The pinned compiler emits depfiles for object compilation, not executable links.
env -u CPATH -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH -u OBJC_INCLUDE_PATH \
  -u DEPENDENCIES_OUTPUT -u SUNPRO_DEPENDENCIES -u GCC_EXEC_PREFIX \
  -u COMPILER_PATH -u LIBRARY_PATH -u CLANG_CONFIG_FILE \
  -u CCC_OVERRIDE_OPTIONS -u CCC_CC -u CCC_CXX \
  -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
  XDG_CACHE_HOME="${out}/cache-closure" ESHKOL_LIB_DIR="${artifact}/facades" \
  ESHKOL_CXX_COMPILER="${cxx}" \
  timeout --foreground --signal=TERM --kill-after=5s "${compiler_timeout}s" \
  "${runner}" --strict-types --no-stdlib -O 2 \
  -I "${artifact}/facades" --compile-only \
  --emit-depfile "${out}/runtime.d" \
  "${PROJECT_ROOT}/tests/e3_horizon/runtime.esk" \
  -o "${out}/runtime-closure.o" \
  >"${out}/closure.stdout" 2>"${out}/closure.stderr"
env -u CPATH -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH -u OBJC_INCLUDE_PATH \
  -u DEPENDENCIES_OUTPUT -u SUNPRO_DEPENDENCIES -u GCC_EXEC_PREFIX \
  -u COMPILER_PATH -u LIBRARY_PATH -u CLANG_CONFIG_FILE \
  -u CCC_OVERRIDE_OPTIONS -u CCC_CC -u CCC_CXX \
  -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
  XDG_CACHE_HOME="${out}/cache" ESHKOL_LIB_DIR="${artifact}/facades" \
  ESHKOL_CXX_COMPILER="${cxx}" \
  timeout --foreground --signal=TERM --kill-after=5s "${compiler_timeout}s" \
  "${runner}" --strict-types --no-stdlib -O 2 \
  -I "${artifact}/facades" -L "${out}" \
  --lib eshkol_transformer_e3_horizon_test \
  "${PROJECT_ROOT}/tests/e3_horizon/runtime.esk" \
  -o "${out}/runtime" >"${out}/compile.stdout" 2>"${out}/compile.stderr"
python3 - "${artifact}" "${PROJECT_ROOT}/tests/e3_horizon/runtime.esk" \
  "${out}/runtime.d" <<'PY'
from pathlib import Path
import shlex
import sys
installed, source, depfile = map(Path, sys.argv[1:])
paths = shlex.split(depfile.read_text().partition(':')[2].replace('\\\n', ' '))
allowed = {str(source)} | {str(path) for path in (installed / 'facades').rglob('*.esk')}
assert paths.count(str(source)) == 1 and len(paths) == len(set(paths))
assert all(path in allowed and str(Path(path).resolve(strict=True)) == path for path in paths)
assert len(paths) > 1
PY

for horizon in 1024 8192; do
  corpus="${out}/corpus-${horizon}"
  late="$(PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${PROJECT_ROOT}" \
    python3 -m tests.e3_horizon.generate \
    --output "${corpus}" --horizon "${horizon}")"
  expected_late="$(printf '%s/shard-%016d.ets' "${corpus}" "$((horizon / 1024))")"
  [[ "${late}" == "${expected_late}" ]] || \
    die "generated late shard path differs from the admitted canonical layout"
  runtime_timeout="${E3_HORIZON_RUNTIME_TIMEOUT_SECONDS:-3600}"
  ESHKOL_ARENA_POISON=1 E3_HORIZON_LATE_SHARD="${late}" \
    /usr/bin/time -f 'elapsed_seconds=%e peak_rss_kib=%M' \
    -o "${out}/runtime-${horizon}.time" \
    timeout --foreground --signal=TERM --kill-after=5s \
      "${runtime_timeout}s" "${out}/runtime" "${corpus}" "${horizon}" \
      >"${out}/runtime-${horizon}.stdout" \
      2>"${out}/runtime-${horizon}.stderr"
  grep -Eq "^E3-HORIZON-PASS horizon=${horizon} failed-stage=traversal retry-batches=${horizon} retry-tokens=$((2 * horizon)) arena-failure-delta=[0-9]+ arena-retry-delta=[0-9]+$" \
    "${out}/runtime-${horizon}.stdout"
  [[ ! -s "${out}/runtime-${horizon}.stderr" ]] || \
    die "horizon runtime wrote stderr: ${horizon}"
  [[ -f "${late}" && ! -e "${late}.held" ]] || \
    die "late shard not restored after horizon ${horizon}"
  peak="$(sed -n 's/^.*peak_rss_kib=\([0-9][0-9]*\)$/\1/p' "${out}/runtime-${horizon}.time")"
  [[ -n "${peak}" && "${peak}" -le 524288 ]] || \
    die "E3 horizon ${horizon} exceeds proposed 512 MiB runtime RSS ceiling: ${peak:-missing} KiB"
done
printf 'E3 single-invocation horizons passed: %s\n' "${out}"
