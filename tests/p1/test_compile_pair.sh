#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/p1-compile-pair.sh"
p1_tmp="$(mktemp -d)"
p1_timing="${p1_tmp}/timing.log"
trap 'p1_stop_pair; rm -rf -- "${p1_tmp}"' EXIT

p1_require_pair_memory() {
  printf '%s\n' checked >>"${p1_tmp}/memory-checks"
  return "${mock_memory_status}"
}
: >"${p1_tmp}/memory-checks"
p1_two_compilers=0
if p1_pair_selected serial; then exit 1; fi
p1_two_compilers=1
p1_pair_selected force
[[ ! -s "${p1_tmp}/memory-checks" ]]

p1_two_compilers=auto
mock_memory_status=0
p1_pair_selected public-objects-1 >"${p1_tmp}/pair-schedule"
mock_memory_status=1
if p1_pair_selected trusted-object-registry-1 >>"${p1_tmp}/pair-schedule"; then exit 1; fi
[[ "$(cat "${p1_tmp}/memory-checks")" == $'checked\nchecked' ]]
[[ "$(cat "${p1_tmp}/pair-schedule")" == $'P1 compile schedule stage=public-objects-1 mode=paired\nP1 compile schedule stage=trusted-object-registry-1 mode=serial reason=below-8GiB-headroom' ]]

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
  sleep 30 &
  printf '%s\n' "$!" >"${p1_tmp}/failed-descendant.pid"
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
for marker in descendant.pid failed-descendant.pid; do
  pid="$(cat "${p1_tmp}/${marker}")"
  if ps -o stat= -p "${pid}" | grep -Eq '^[[:space:]]*[^Z[:space:]]'; then
    printf 'P1 pair left a live compiler descendant: %s\n' "${pid}" >&2
    exit 1
  fi
done
[[ "$(tail -n 1 "${p1_timing}")" == failure ]]
printf 'P1 compile pair PASS: memory selection, overlap, timing order, exit status, child cleanup\n'
