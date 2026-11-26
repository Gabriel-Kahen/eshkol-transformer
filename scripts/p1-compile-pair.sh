#!/usr/bin/env bash

# Sourced by test-p1.sh. Each background shell gets its own process group so
# failure or interruption also terminates timeout and compiler descendants.
p1_pair_pids=()
p1_pair_timing_files=()
p1_pair_memory_pid=
p1_pair_memory_index=0

p1_stop_pair_memory() {
  local status=0
  [[ -n "${p1_pair_memory_pid}" ]] || return 0
  kill -TERM "${p1_pair_memory_pid}" 2>/dev/null || true
  wait "${p1_pair_memory_pid}" || status=$?
  p1_pair_memory_pid=
  return "${status}"
}

p1_require_pair_memory() {
  local available_kib limit used
  available_kib="$(awk '$1 == "MemAvailable:" { print $2 }' /proc/meminfo)"
  [[ "${available_kib}" =~ ^[0-9]+$ ]] || return 1
  ((available_kib >= 8 * 1024 * 1024)) || return 1
  if [[ -r /sys/fs/cgroup/memory.max && -r /sys/fs/cgroup/memory.current ]]; then
    read -r limit </sys/fs/cgroup/memory.max
    read -r used </sys/fs/cgroup/memory.current
    if [[ "${limit}" =~ ^[0-9]+$ && "${used}" =~ ^[0-9]+$ ]]; then
      ((limit - used >= 8 * 1024 * 1024 * 1024)) || return 1
    fi
  elif [[ -r /sys/fs/cgroup/memory/memory.limit_in_bytes &&
          -r /sys/fs/cgroup/memory/memory.usage_in_bytes ]]; then
    read -r limit </sys/fs/cgroup/memory/memory.limit_in_bytes
    read -r used </sys/fs/cgroup/memory/memory.usage_in_bytes
    ((limit - used >= 8 * 1024 * 1024 * 1024)) || return 1
  fi
}

p1_flush_pair_timing() {
  if ((${#p1_pair_timing_files[@]})); then
    cat "${p1_pair_timing_files[@]}" >>"${p1_timing}"
    rm -f -- "${p1_pair_timing_files[@]}"
    p1_pair_timing_files=()
  fi
}

p1_stop_pair() {
  local pid
  for pid in "${p1_pair_pids[@]}"; do
    kill -TERM -- "-${pid}" 2>/dev/null || true
  done
  for pid in "${p1_pair_pids[@]}"; do
    kill -KILL -- "-${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
  done
  p1_pair_pids=()
}

p1_compile_pair() {
  local -a first=() second=() remaining=()
  local first_timing second_timing memory_trace pid finished status=0
  while [[ "$1" != -- ]]; do first+=("$1"); shift; done
  shift
  second=("$@")
  first_timing="$(mktemp "${p1_tmp}/pair-timing.XXXXXX")"
  second_timing="$(mktemp "${p1_tmp}/pair-timing.XXXXXX")"
  p1_pair_timing_files=("${first_timing}" "${second_timing}")

  set -m
  (p1_timing="${first_timing}"; "${first[@]}") &
  p1_pair_pids+=("$!")
  (p1_timing="${second_timing}"; "${second[@]}") &
  p1_pair_pids+=("$!")
  set +m

  if [[ "${p1_pair_memory_trace:-0}" == 1 ]]; then
    p1_pair_memory_index=$((p1_pair_memory_index + 1))
    memory_trace="${p1_tmp}/pair-${p1_pair_memory_index}-memory.tsv"
    python3 -B "${PROJECT_ROOT}/scripts/p1-pair-memory.py" \
      "${memory_trace}" \
      "${p1_pair_pids[0]}" "${p1_pair_pids[1]}" &
    p1_pair_memory_pid=$!
    for ((pid=0; pid<100; ++pid)); do
      [[ -s "${memory_trace}" ]] && break
      kill -0 "${p1_pair_memory_pid}" 2>/dev/null || break
      sleep 0.01
    done
    if [[ ! -s "${memory_trace}" ]]; then
      p1_stop_pair
      p1_stop_pair_memory || true
      p1_flush_pair_timing
      return 1
    fi
  fi

  while ((${#p1_pair_pids[@]})); do
    finished=
    if wait -n -p finished "${p1_pair_pids[@]}"; then
      :
    else
      status=$?
    fi
    # A signal can interrupt wait before it assigns a completed child.
    if [[ -z "${finished}" ]]; then
      [[ "${status}" != 0 ]] || status=1
      break
    fi
    remaining=()
    for pid in "${p1_pair_pids[@]}"; do
      [[ "${pid}" == "${finished}" ]] || remaining+=("${pid}")
    done
    p1_pair_pids=("${remaining[@]}")
    [[ "${status}" == 0 ]] || break
  done
  if [[ "${status}" != 0 ]]; then p1_stop_pair; fi
  if [[ "${p1_pair_memory_trace:-0}" == 1 ]]; then
    p1_stop_pair_memory || { [[ "${status}" != 0 ]] || status=1; }
  fi
  # Deterministic planned order; each time invocation keeps its existing line.
  p1_flush_pair_timing
  return "${status}"
}
