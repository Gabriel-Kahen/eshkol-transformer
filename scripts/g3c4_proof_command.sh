#!/usr/bin/env bash
# Opt-in command boundary. Disabled execution is the original argv in place.
proof_command() {
  if [[ ${G3C4_PROOF_RECEIPTS:-} == '' ]]; then
    "$@"
  else
    local caller=${BASH_SOURCE[1]:-${BASH_SOURCE[0]}}
    G3C4_PROOF_SITE="${G3C4_PROOF_SITE_OVERRIDE:-${caller##*/}:${BASH_LINENO[0]}}" \
      G3C4_PROOF_HELPER=1 \
      python3 "$PROJECT_ROOT/scripts/g3c4_proof_command.py" \
      "$G3C4_PROOF_RECEIPTS" -- "$@"
  fi
}
