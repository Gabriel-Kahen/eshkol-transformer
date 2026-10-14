#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
for command in ar cmp python3 timeout; do require_command "${command}"; done
[[ -x /usr/bin/time ]] || die "GNU time is required for cap measurements"
[[ $# -le 1 ]] || die "usage: $0 [FRESH_OUTPUT_DIRECTORY]"
out="${1:-$(project_build_dir)/e3-cap-preflight}"
[[ "${out}" == /* ]] || die "cap-preflight output directory must be absolute"
[[ ! -e "${out}" ]] || die "cap-preflight output directory must be fresh: ${out}"
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
  -o "${out}/shard_hook.o"
ar p "${artifact}/libeshkol_transformer_e3_diagnostic.a" \
  e3_diagnostic_public_package.o >"${out}/package.o"
cmp "${artifact}/e3_diagnostic_public_package.o" "${out}/package.o"
ar rcsD "${out}/libeshkol_transformer_e3_cap_test.a" \
  "${out}/package.o" "${out}/shard_hook.o"
[[ "$(ar t "${out}/libeshkol_transformer_e3_cap_test.a")" == \
   $'package.o\nshard_hook.o' ]] || die "cap caller archive membership differs"

runner="$(eshkol_build_dir)/eshkol-run"
compiler_timeout="${E3_COMPILER_TIMEOUT_SECONDS:-900}"
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
  "${PROJECT_ROOT}/tests/e3_horizon/cap_runtime.esk" \
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
  --lib eshkol_transformer_e3_cap_test \
  "${PROJECT_ROOT}/tests/e3_horizon/cap_runtime.esk" \
  -o "${out}/runtime" >"${out}/compile.stdout" 2>"${out}/compile.stderr"
python3 - "${artifact}" "${PROJECT_ROOT}/tests/e3_horizon/cap_runtime.esk" \
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

corpus="${out}/corpus-8193"
first="$(PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${PROJECT_ROOT}" \
  python3 -m tests.e3_horizon.cap_generate --output "${corpus}")"
[[ "${first}" == "${corpus}/shard-0000000000000000.ets" ]] || \
  die "generated first shard path differs from the authenticated layout"
for poison in 0 1; do
  ESHKOL_ARENA_POISON="${poison}" E3_HORIZON_LATE_SHARD="${first}" \
    /usr/bin/time -f 'elapsed_seconds=%e peak_rss_kib=%M' \
    -o "${out}/runtime-${poison}.time" \
    timeout --foreground --signal=TERM --kill-after=5s 120s \
      "${out}/runtime" "${corpus}" \
      >"${out}/runtime-${poison}.stdout" \
      2>"${out}/runtime-${poison}.stderr"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${PROJECT_ROOT}" \
    python3 -m tests.e3_horizon.parse_cap \
      --stdout "${out}/runtime-${poison}.stdout" \
      --stderr "${out}/runtime-${poison}.stderr" \
      --time "${out}/runtime-${poison}.time" --first-shard "${first}"
done
printf 'E3 public cap preflight passed: %s\n' "${out}"
