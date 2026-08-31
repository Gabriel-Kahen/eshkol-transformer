#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

for command in awk cmp grep nm readelf; do
  require_command "${command}"
done

output="${1:-$(project_build_dir)/p1/libeshkol_transformer_p1.o}"

"${PROJECT_ROOT}/scripts/build-e1b-consumer.sh" \
  "${PROJECT_ROOT}/internal/p1/lib/transformer/module.esk" \
  "${PROJECT_ROOT}/native/p1_package_bridge.c" \
  "${PROJECT_ROOT}/native/p1_package_renames.txt" \
  "${PROJECT_ROOT}/native/p1_package_public_exports.txt" \
  "${output}"

cmp "${PROJECT_ROOT}/native/p1_package_defined_symbols.txt" \
  "${output}.evidence/global-defined.txt" || \
  die "P1 package differs from the exact defined-symbol manifest"
cmp "${PROJECT_ROOT}/native/p1_package_undefined_symbols.txt" \
  "${output}.evidence/undefined.txt" || \
  die "P1 package differs from the exact undefined-symbol manifest"

if nm -u --format=posix "${output}" | awk '{ print $1 }' | \
    grep -E '^(et_e1b_|et_p1_|transformer-error-|e1-internal-dispatch)' \
      >/dev/null; then
  die "P1 package retains an unresolved private boundary symbol"
fi
if readelf --wide --syms "${output}" | awk '
  /et_p1_(private|public)_|et_e1b_(private|consumer|box|ensure)|transformer-error-(make|raise|wrap-foreign)|e1-internal-dispatch/ {
    if ($5 != "LOCAL" && $7 != "UND") bad = 1
  }
  END { exit bad ? 0 : 1 }
'; then
  die "P1 package retained a nonlocal private boundary definition"
fi
