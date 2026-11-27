#!/usr/bin/env bash

# Sourced by test-p1.sh. Each background shell gets its own process group so
# failure or interruption also terminates timeout and compiler descendants.
p1_pair_pids=()
p1_pair_timing_files=()

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

p1_pair_selected() {
  local stage="$1"
  case "${p1_two_compilers}" in
    0) return 1 ;;
    1) return 0 ;;
  esac
  if p1_require_pair_memory; then
    printf 'P1 compile schedule stage=%s mode=paired\n' "${stage}"
    return 0
  fi
  printf 'P1 compile schedule stage=%s mode=serial reason=below-8GiB-headroom\n' "${stage}"
  return 1
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
  local first_timing second_timing pid finished status=0
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
  # Deterministic planned order; each time invocation keeps its existing line.
  p1_flush_pair_timing
  return "${status}"
}
