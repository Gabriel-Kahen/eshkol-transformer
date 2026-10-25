#!/usr/bin/env bash
# Run fault and nearest control in separate processes, even if either fails.
set -euo pipefail
[[ $# == 7 && -d "$1" && -x "$2" && -d "$3" && "$5" =~ ^[01]$ &&
   "$6" =~ ^[1-9][0-9]*$ && -x "$7" ]] ||
  { printf 'usage: %s OUT EXECUTABLE CORPUS PROFILE POISON TIMEOUT GNU_TIME\n' "$0" >&2; exit 2; }
out=$1 executable=$2 corpus=$3 profile=$4 poison=$5 runtime_timeout=$6 time_tool=$7
overall=0
for mode in fault control; do
  name="diagnostic-$profile-$mode"
  status=0
  ESHKOL_ARENA_POISON="$poison" ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
    UBSAN_OPTIONS=halt_on_error=1 \
    "$time_tool" -f 'elapsed_seconds=%e peak_rss_kib=%M' -o "$out/$name.time" \
    timeout --foreground --signal=TERM --kill-after=5s "${runtime_timeout}s" \
    "$executable" "$corpus" "$mode" \
    >"$out/$name.stdout" 2>"$out/$name.stderr" || status=$?
  printf '%s\n' "$status" >"$out/$name.exit"
  sample=0 postcheck=0
  if [[ "$(grep -Ec "^E3-RETENTION-ARENA-DIAGNOSTIC mode=$mode prepared=[0-9]+ before=[0-9]+ after-arm=[0-9]+ after-call=[0-9]+ after-checks=[0-9]+ after-retry=[0-9]+$" "$out/$name.stdout")" == 1 ]]; then
    sample=1
  fi
  if [[ "$(grep -Ec "^E3-RETENTION-ARENA-POSTCHECK-PASS mode=$mode$" "$out/$name.stdout")" == 1 ]]; then
    postcheck=1
  fi
  printf '%s\t%s\t%s\t%s\t%s\n' "$profile" "$mode" "$status" "$sample" "$postcheck"
  [[ "$status" == 0 && "$sample" == 1 && "$postcheck" == 1 ]] || overall=1
done
exit "$overall"
