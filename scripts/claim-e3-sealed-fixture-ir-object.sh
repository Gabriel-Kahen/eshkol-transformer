#!/usr/bin/env bash
# Atomic, persistent one-use claim for the accepted J IR tuple.
set -euo pipefail
[[ $# -eq 2 ]] || { echo "usage: $0 CLAIM_ROOT OUTPUT" >&2; exit 2; }
root=$1
output=$2
[[ -d "${root}" && ! -L "${root}" ]] || {
  echo "claim root must be an existing directory" >&2; exit 1;
}
claim="${root}/e3-section9-j-ir-object-fb31-fe9"
mkdir -- "${claim}" || {
  echo "accepted J IR-object claim was already consumed" >&2; exit 1;
}
printf '%s\n' "${output}" >"${claim}/output-path.txt"
printf '%s\n' "fb31fac32530e0ff9b00ed3458e9080eaba9643a" \
  >"${claim}/j-source-head.txt"
chmod a-w -- "${claim}/output-path.txt" "${claim}/j-source-head.txt" "${claim}"
