#!/usr/bin/env bash
# The Python SIGABRT capture wrapper itself must be diagnostically silent.
e3_overflow_capture_streams_empty() {
  local prefix=$1 suffix path
  for suffix in stdout stderr; do
    path="${prefix}.${suffix}"
    [[ -f "${path}" && ! -L "${path}" && ! -s "${path}" ]] || return 1
  done
}
