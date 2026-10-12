#!/usr/bin/env bash
# Opt-in command boundary. Disabled execution is the original argv in place.
proof_command() {
  if [[ ${G3C4_PROOF_RECEIPTS:-} == '' ]]; then
    "$@"
  else
    G3C4_PROOF_SITE="${BASH_SOURCE[1]##*/}:${BASH_LINENO[0]}" \
      python3 "$PROJECT_ROOT/scripts/g3c4_proof_command.py" \
      "$G3C4_PROOF_RECEIPTS" -- "$@"
  fi
}
