#!/usr/bin/env bash

set -euo pipefail

if [[ "$#" != 1 ]]; then
  printf 'usage: %s ARTIFACT\n' "$0" >&2
  exit 2
fi

artifact=$1
[[ -r "${artifact}" ]] || {
  printf 'error: D2 dependency artifact is not readable: %s\n' \
    "${artifact}" >&2
  exit 2
}

match_file="$(mktemp "${TMPDIR:-/tmp}/eshkol-d2-dependency.XXXXXX")"
trap 'rm -f -- "${match_file}"' EXIT

report_match() {
  local channel=$1
  printf 'error: D2 development-oracle dependency match: artifact=%s channel=%s\n' \
    "${artifact}" "${channel}" >&2
  sed -n '1,20p' "${match_file}" >&2
  exit 1
}

# Raw executable bytes can form short printable fragments accidentally. Match
# only bounded library, header, package, or interpreter-path markers here.
strings_pattern='(^|[^[:alnum:]_])(libpython[[:alnum:]._-]*|libtorch[[:alnum:]._-]*|pytorch|python\.h)([^[:alnum:]_]|$)|(^|[\\/])python[0-9.]*([\\/]|$)'
if strings -a "${artifact}" | grep -Ei "${strings_pattern}" >"${match_file}"; then
  report_match strings
fi

# Symbol and dynamic-link inspection remain intentionally broader: these
# channels have structure and do not interpret arbitrary instruction bytes.
if nm -a "${artifact}" 2>/dev/null | \
    grep -Ei 'python|pytorch|torch|libpython|(^|[[:space:]])Py_' \
      >"${match_file}"; then
  report_match nm
fi
if ldd "${artifact}" 2>/dev/null | \
    grep -Ei 'python|pytorch|torch|libpython' >"${match_file}"; then
  report_match ldd
fi
