#!/usr/bin/env bash
# Finalize a retained capture, including failures, without hashing the seal itself.
set -euo pipefail
[[ $# -eq 2 && -d "$1" && "$2" =~ ^([0-9]|[1-9][0-9]{1,2})$ ]] || {
  echo "usage: $0 CAPTURE_DIRECTORY EXIT_STATUS" >&2; exit 2;
}
out=$1
status=$2
tmp="${out}/SHA256SUMS.tmp.$$"
fail() {
  chmod u+w -- "${out}" "${out}/capture.exit" 2>/dev/null || true
  printf '1\n' >"${out}/capture.exit" 2>/dev/null || true
  rm -f -- "${tmp}" "${out}/SHA256SUMS" 2>/dev/null || true
  exit 1
}
printf '%s\n' "${status}" >"${out}/capture.exit" || fail
if ! (cd "${out}" && find . -type f ! -name SHA256SUMS \
      ! -name 'SHA256SUMS.tmp.*' -print0 | LC_ALL=C sort -z | \
      xargs -0 -r sha256sum) >"${tmp}"; then
  fail
fi
mv -- "${tmp}" "${out}/SHA256SUMS" || fail
(cd "${out}" && sha256sum --quiet -c SHA256SUMS) || fail
chmod -R a-w -- "${out}" || fail
