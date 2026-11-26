#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/p1-compile-pair.sh"
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
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

p1_pair_memory_trace=1
python3 -B "${PROJECT_ROOT}/scripts/p1-pair-memory.py" --check
mock_memory_compile() {
  python3 -B -c 'import time; payload = bytearray(b"x" * (8 * 1024 * 1024)); time.sleep(0.8); assert payload[0] == 120'
}
p1_compile_pair mock_memory_compile -- mock_memory_compile
python3 -B - "${p1_tmp}/pair-1-memory.tsv" <<'PY'
import json
from pathlib import Path
import sys

trace = Path(sys.argv[1])
rows = trace.read_text().splitlines()
summary = json.loads(trace.with_suffix('.summary.json').read_text())
assert len(rows) >= 2 and rows[0].startswith('utc\tmonotonic_ns\t')
assert summary['both_active_samples'] > 0, summary
assert summary['max_sampled_pair_rss_bytes'] >= 16 * 1024 * 1024, summary
PY
[[ -z "${p1_pair_memory_pid}" && "${#p1_pair_pids[@]}" == 0 ]]
mock_memory_fail() {
  while [[ ! -e "${p1_tmp}/memory-peer.started" ]]; do sleep 0.01; done
  sleep 0.1
  return 37
}
mock_memory_slow() {
  : >"${p1_tmp}/memory-peer.started"
  sleep 30 &
  printf '%s\n' "$!" >"${p1_tmp}/memory-descendant.pid"
  wait "$!"
}
status=0
p1_compile_pair mock_memory_fail -- mock_memory_slow || status=$?
[[ "${status}" == 37 && -s "${p1_tmp}/pair-2-memory.summary.json" ]]
[[ -z "${p1_pair_memory_pid}" && "${#p1_pair_pids[@]}" == 0 ]]
pid="$(cat "${p1_tmp}/memory-descendant.pid")"
if ps -o stat= -p "${pid}" | grep -Eq '^[[:space:]]*[^Z[:space:]]'; then
  printf 'P1 pair left a live compiler descendant after traced failure: %s\n' "${pid}" >&2
  exit 1
fi
printf 'P1 compile pair PASS: overlap, timing order, exit status, child cleanup, scoped RSS trace\n'
