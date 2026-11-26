#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/p1-compile-pair.sh"
p1_tmp="$(mktemp -d)"
p1_timing="${p1_tmp}/timing.log"
trap 'p1_stop_pair; rm -rf -- "${p1_tmp}"' EXIT

mock_compile() {
  local name="$1" peer="$2"
  : >"${p1_tmp}/${name}.started"
  while [[ ! -e "${p1_tmp}/${peer}.started" ]]; do sleep 0.01; done
  printf '%s\n' "${name}" >>"${p1_timing}"
}
p1_compile_pair mock_compile first second -- mock_compile second first
[[ "$(cat "${p1_timing}")" == $'first\nsecond' ]]
[[ "${#p1_pair_pids[@]}" == 0 ]]

mock_fail() {
  while [[ ! -s "${p1_tmp}/descendant.pid" ]]; do sleep 0.01; done
  printf '%s\n' failure >>"${p1_timing}"
  return 37
}
mock_slow() {
  sleep 30 &
  printf '%s\n' "$!" >"${p1_tmp}/descendant.pid"
  wait "$!"
}
status=0
p1_compile_pair mock_fail -- mock_slow || status=$?
[[ "${status}" == 37 ]]
[[ "${#p1_pair_pids[@]}" == 0 ]]
pid="$(cat "${p1_tmp}/descendant.pid")"
if ps -o stat= -p "${pid}" | grep -Eq '^[[:space:]]*[^Z[:space:]]'; then
  printf 'P1 pair left a live compiler descendant: %s\n' "${pid}" >&2
  exit 1
fi
[[ "$(tail -n 1 "${p1_timing}")" == failure ]]
printf 'P1 compile pair PASS: overlap, timing order, exit status, child cleanup\n'
