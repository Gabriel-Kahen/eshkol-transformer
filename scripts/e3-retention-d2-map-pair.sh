#!/usr/bin/env bash
# Diagnostic only: compare direct D2, E3 mapper, and successful control in fresh processes.
set -euo pipefail
[[ $# == 7 && -d "$1" && -x "$2" && -d "$3" && "$5" =~ ^[01]$ &&
   "$6" =~ ^[1-9][0-9]*$ && -x "$7" ]] ||
  { printf 'usage: %s OUT EXECUTABLE CORPUS PROFILE POISON TIMEOUT GNU_TIME\n' "$0" >&2; exit 2; }
out=$1 executable=$2 corpus=$3 profile=$4 poison=$5 runtime_timeout=$6 time_tool=$7
overall=0
for mode in d2-fault d2-map-fault d2-control; do
  name="map-$profile-$mode"
  status=0
  ESHKOL_ARENA_POISON="$poison" ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
    UBSAN_OPTIONS=halt_on_error=1 \
    "$time_tool" -f 'elapsed_seconds=%e peak_rss_kib=%M' -o "$out/$name.time" \
    timeout --foreground --signal=TERM --kill-after=5s "${runtime_timeout}s" \
    "$executable" "$corpus" "$mode" \
    >"$out/$name.stdout" 2>"$out/$name.stderr" || status=$?
  printf '%s\n' "$status" >"$out/$name.exit"
  sample=0
  if [[ "$mode" == d2-map-fault ]]; then
    pattern='^E3-RETENTION-D2-MAP before=[0-9]+ after-arm=[0-9]+ caught=[0-9]+ before-map=[0-9]+ after-map=[0-9]+ after-call=[0-9]+ after-checks=[0-9]+ after-retry=[0-9]+$'
  else
    pattern="^E3-RETENTION-D2-ARENA mode=$mode before=[0-9]+ after-arm=[0-9]+ after-call=[0-9]+ after-checks=[0-9]+ after-retry=[0-9]+$"
  fi
  [[ "$(grep -Ec "$pattern" "$out/$name.stdout")" == 1 ]] && sample=1
  printf '%s\t%s\t%s\t%s\n' "$profile" "$mode" "$status" "$sample"
  [[ "$status" == 0 && "$sample" == 1 ]] || overall=1
done
exit "$overall"
