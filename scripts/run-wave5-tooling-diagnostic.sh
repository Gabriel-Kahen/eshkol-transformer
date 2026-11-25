#!/usr/bin/env bash
set -euo pipefail

evidence=${1:?usage: run-wave5-tooling-diagnostic.sh EVIDENCE_DIR}
[[ -d "$evidence" && ! -L "$evidence" ]] || {
  printf 'evidence directory is missing or a symlink\n' >&2
  exit 2
}
evidence=$(realpath -- "$evidence")
printf 'gate\tresult\texit_code\n' > "$evidence/gates.tsv"

failed=0
for gate in test-t2-trainer test-data4-pipeline test-beam-search \
            test-constrained-decode test-paged-kv test-spec-decode; do
  mkdir -m 0700 "$evidence/$gate"
  if timeout --kill-after=10s 2400s make "$gate" \
      > "$evidence/$gate/stdout" 2> "$evidence/$gate/stderr"; then
    result=pass
    status=0
  else
    status=$?
    result=fail
    failed=1
  fi
  printf '%s\t%s\t%s\n' "$gate" "$result" "$status" \
    >> "$evidence/gates.tsv"
done
exit "$failed"
