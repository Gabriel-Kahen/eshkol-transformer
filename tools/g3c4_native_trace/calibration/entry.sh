#!/usr/bin/bash
set -euo pipefail
export PATH=/usr/bin:/bin
export G3C4_TRACE_SOURCE=/workspace
mkdir -p /out/bin
/usr/bin/clang-21 -std=c11 -Wall -Wextra -Werror -Wpedantic -O2 \
  /workspace/tools/g3c4_native_trace/trace.c \
  /workspace/tools/g3c4_native_trace/sha256.c -o /out/bin/trace \
  >/out/trace-build.stdout 2>/out/trace-build.stderr
/usr/bin/clang-21 -std=c11 -Wall -Wextra -Werror -Wpedantic -O2 -pthread \
  /workspace/tools/g3c4_native_trace/tests/canary.c -o /out/bin/canary \
  >/out/canary-build.stdout 2>/out/canary-build.stderr

run_case() {
  local name=$1
  shift
  set +e
  /out/bin/trace --log "/out/$name.events" -- "$@" \
    >"/out/$name.stdout" 2>"/out/$name.stderr"
  local status=$?
  set -e
  printf '%s\n' "$status" >"/out/$name.exit"
}
run_case fast /out/bin/canary fast
run_case spawn /out/bin/canary spawn
run_case thread /out/bin/canary thread
run_case threadexec /out/bin/canary threadexec
set +e
/out/bin/trace --log /out/fault.events --fault-after-exec 2 -- \
  /out/bin/canary faultchild /out/fault-unwanted \
  >/out/fault.stdout 2>/out/fault.stderr
fault_status=$?
/out/bin/trace --log /out/untraced.events -- \
  /out/bin/canary untraced /out/untraced-unwanted \
  >/out/untraced.stdout 2>/out/untraced.stderr
untraced_status=$?
set -e
printf '%s\n' "$fault_status" >/out/fault.exit
printf '%s\n' "$untraced_status" >/out/untraced.exit
/usr/bin/python3 -B /instrument/postflight.py /out \
  >/out/postflight.json 2>/out/postflight.stderr
