#!/usr/bin/env bash
set -u

usage() {
  echo "usage: $0 --eshkol-source DIR --work-dir DIR --results-dir DIR" >&2
  exit 64
}

eshkol_source=
work_dir=
results_dir=
while (( $# > 0 )); do
  case "$1" in
    --eshkol-source) eshkol_source=${2:-}; shift 2 ;;
    --work-dir) work_dir=${2:-}; shift 2 ;;
    --results-dir) results_dir=${2:-}; shift 2 ;;
    *) usage ;;
  esac
done
[[ -n "$eshkol_source" && -n "$work_dir" && -n "$results_dir" ]] || usage
[[ "$eshkol_source" = /* && "$work_dir" = /* && "$results_dir" = /* ]] || {
  echo "all directory arguments must be absolute" >&2
  exit 64
}
if [[ -d "$results_dir" ]] && [[ -n "$(find "$results_dir" -mindepth 1 -print -quit)" ]]; then
  echo "results directory must be absent or empty: $results_dir" >&2
  exit 64
fi

probe_dir=$(cd "$(dirname "$0")" && pwd)
pin=$(tr -d '[:space:]' < "$probe_dir/eshkol.rev")
mkdir -p "$work_dir" "$results_dir/commands"
manifest="$results_dir/manifest.tsv"
printf 'name\texit_code\texpectation\tcommand\n' > "$manifest"
: > "$results_dir/assertion-failures.txt"
: > "$results_dir/parity-failures.txt"

record() {
  local name=$1 classification=$2
  shift 2
  local timeout_seconds=30 stream_limit_bytes=2097152 virtual_kib=2097152
  case "$classification" in
    build) timeout_seconds=900; virtual_kib=0 ;;
    upstream-tests) timeout_seconds=300; virtual_kib=0 ;;
    discovery) timeout_seconds=60; virtual_kib=0 ;;
    *compile*) timeout_seconds=180; virtual_kib=4194304 ;;
  esac
  local stdout="$results_dir/commands/$name.stdout"
  local stderr="$results_dir/commands/$name.stderr"
  local command_file="$results_dir/commands/$name.command"
  printf 'timeout=%ss stream_limit_bytes=%s virtual_kib=%s\n' \
    "$timeout_seconds" "$stream_limit_bytes" "$virtual_kib" > "$command_file"
  local argument first=1
  for argument in "$@"; do
    if (( first == 0 )); then printf ' ' >> "$command_file"; fi
    printf '%q' "$argument" >> "$command_file"
    first=0
  done
  printf '\n' >> "$command_file"
  (
    ulimit -c 0
    if (( virtual_kib > 0 )); then ulimit -v "$virtual_kib"; fi
    /usr/bin/timeout --signal=TERM --kill-after=5s "$timeout_seconds" "$@" \
      > >(head -c "$stream_limit_bytes" > "$stdout") \
      2> >(head -c "$stream_limit_bytes" > "$stderr")
    command_status=$?
    wait
    exit "$command_status"
  )
  local status=$?
  printf '%s\t%s\t%s\t' "$name" "$status" "$classification" >> "$manifest"
  tail -n 1 "$command_file" | tr '\n' ' ' >> "$manifest"
  printf '\n' >> "$manifest"
  return "$status"
}

actual=$(git -C "$eshkol_source" rev-parse HEAD 2>"$results_dir/pin.stderr") || {
  echo "cannot read Eshkol revision" >&2
  exit 65
}
printf '%s\n' "$pin" > "$results_dir/expected-eshkol-revision.txt"
printf '%s\n' "$actual" > "$results_dir/actual-eshkol-revision.txt"
[[ "$actual" == "$pin" ]] || {
  echo "Eshkol revision mismatch: expected $pin, got $actual" >&2
  exit 65
}
dirty=$(git -C "$eshkol_source" status --porcelain)
[[ -z "$dirty" ]] || {
  printf '%s\n' "$dirty" > "$results_dir/dirty-upstream.txt"
  echo "Eshkol checkout has modifications or untracked files" >&2
  exit 65
}
export ESHKOL_PATH="$eshkol_source/lib${ESHKOL_PATH:+:$ESHKOL_PATH}"

{
  echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "eshkol_revision=$actual"
  echo "transformer_revision=$(git -C "$probe_dir/../.." rev-parse HEAD 2>/dev/null || echo unavailable)"
  echo "transformer_status=$(git -C "$probe_dir/../.." status --porcelain 2>/dev/null | tr '\n' ';' || echo unavailable)"
  echo "uname=$(uname -a)"
  echo "cmake=$(cmake --version | sed -n '1p')"
  echo "ninja=$(ninja --version 2>/dev/null || echo unavailable)"
  echo "cc=$(cc --version | sed -n '1p')"
  echo "cxx=$(c++ --version | sed -n '1p')"
  selected_llvm_config=${ESHKOL_LLVM_CONFIG:-llvm-config}
  echo "llvm_config_path=$(command -v "$selected_llvm_config" 2>/dev/null || echo "$selected_llvm_config")"
  echo "llvm_config=$($selected_llvm_config --version 2>/dev/null || echo unavailable)"
  echo "cpu=$(lscpu 2>/dev/null | tr '\n' ';' || echo unavailable)"
  echo "nvidia=$(nvidia-smi -L 2>/dev/null | tr '\n' ';' || echo unavailable)"
  echo "rocm=$(rocminfo 2>/dev/null | sed -n '1,80p' | tr '\n' ';' || echo unavailable)"
} > "$results_dir/environment.txt"

build_dir="$work_dir/build"
cmake_args=(-S "$eshkol_source" -B "$build_dir"
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DESHKOL_BUILD_TESTS=ON)
if [[ -n "${ESHKOL_LLVM_CONFIG:-}" ]]; then
  cmake_args+=("-DLLVM_CONFIG_EXECUTABLE=$ESHKOL_LLVM_CONFIG")
fi
record configure build cmake "${cmake_args[@]}" || exit 1
record build build cmake --build "$build_dir" --parallel 2 || exit 1
record upstream_ctest upstream-tests ctest --test-dir "$build_dir" --output-on-failure
ctest_status=$?
compiler="$build_dir/eshkol-run"
[[ -x "$compiler" ]] || { echo "missing executable: $compiler" >&2; exit 1; }
record compiler_help discovery "$compiler" --help

failures=0
(( ctest_status == 0 )) || failures=$((failures + 1))
if grep -q 'No tests were found' "$results_dir/commands/upstream_ctest.stdout" \
  "$results_dir/commands/upstream_ctest.stderr"; then
  printf '%s\n' "upstream ctest: zero tests registered" >> "$results_dir/assertion-failures.txt"
  failures=$((failures + 1))
fi

has_error() {
  grep -Eiq 'ERROR:|Failed to|terminate called|Assertion' "$@"
}

validate_capability() {
  local name=$1 phase=$2
  local stdout="$results_dir/commands/$name.$phase.stdout"
  local stderr="$results_dir/commands/$name.$phase.stderr"
  if has_error "$stdout" "$stderr"; then
    printf '%s\n' "$name: $phase emitted an error diagnostic" >> "$results_dir/assertion-failures.txt"
    failures=$((failures + 1))
  fi
  if grep -q 'FAIL' "$stdout"; then
    printf '%s\n' "$name: $phase emitted a FAIL marker" >> "$results_dir/assertion-failures.txt"
    failures=$((failures + 1))
  fi
  if ! grep -qx "R0_DONE $name" "$stdout"; then
    printf '%s\n' "$name: $phase did not reach its completion marker" >> "$results_dir/assertion-failures.txt"
    failures=$((failures + 1))
  fi
}

for source in "$probe_dir"/*.esk; do
  name=$(basename "$source" .esk)
  binary="$work_dir/$name.aot"
  rm -f -- "$binary"
  aot_ready=1
  record "$name.aot_compile" capability-compile "$compiler" "$source" -o "$binary" -L "$build_dir" || {
    failures=$((failures + 1)); aot_ready=0;
  }
  if (( aot_ready == 1 )) && has_error \
    "$results_dir/commands/$name.aot_compile.stdout" \
    "$results_dir/commands/$name.aot_compile.stderr"; then
    printf '%s\n' "$name: AOT compilation emitted an error diagnostic" >> "$results_dir/assertion-failures.txt"
    failures=$((failures + 1)); aot_ready=0
  fi
  if (( aot_ready == 1 )) && [[ ! -x "$binary" ]]; then
    printf '%s\n' "$name: AOT compilation returned zero without an executable" >> "$results_dir/assertion-failures.txt"
    failures=$((failures + 1)); aot_ready=0
  fi
  if (( aot_ready == 1 )); then
    if [[ "$name" == memory_loop && -x /usr/bin/time ]]; then
      record "$name.aot_run_1" capability-run /usr/bin/time -v "$binary" || failures=$((failures + 1))
      record "$name.aot_run_2" deterministic-repeat /usr/bin/time -v "$binary" || failures=$((failures + 1))
    else
      record "$name.aot_run_1" capability-run "$binary" || failures=$((failures + 1))
      record "$name.aot_run_2" deterministic-repeat "$binary" || failures=$((failures + 1))
    fi
    validate_capability "$name" aot_run_1
    validate_capability "$name" aot_run_2
    cmp -s "$results_dir/commands/$name.aot_run_1.stdout" "$results_dir/commands/$name.aot_run_2.stdout" || {
      printf '%s\n' "$name: repeated AOT output differs" >> "$results_dir/parity-failures.txt"
      failures=$((failures + 1))
    }
  fi
  record "$name.jit_run" jit "$compiler" -r "$source" -L "$build_dir" || failures=$((failures + 1))
  record "$name.jit_run_2" jit-repeat "$compiler" -r "$source" -L "$build_dir" || failures=$((failures + 1))
  validate_capability "$name" jit_run
  validate_capability "$name" jit_run_2
  if (( aot_ready == 1 )); then
    cmp -s "$results_dir/commands/$name.aot_run_1.stdout" "$results_dir/commands/$name.jit_run.stdout" || {
      printf '%s\n' "$name: AOT/JIT stdout differs" >> "$results_dir/parity-failures.txt"
      failures=$((failures + 1))
    }
  fi
  cmp -s "$results_dir/commands/$name.jit_run.stdout" "$results_dir/commands/$name.jit_run_2.stdout" || {
    printf '%s\n' "$name: repeated JIT output differs" >> "$results_dir/parity-failures.txt"
    failures=$((failures + 1))
  }
done

has_diagnostic() {
  grep -Eiq 'error|invalid|mismatch|bound|range|shape|dimension|cannot|missing|not found' "$@"
}

for source in "$probe_dir"/negative/*.esk; do
  name=negative.$(basename "$source" .esk)
  binary="$work_dir/$name.aot"
  rm -f -- "$binary"
  record "$name.aot_compile" expected-failure-compile "$compiler" "$source" -o "$binary" -L "$build_dir"
  compile_status=$?
  if (( compile_status == 0 )) && ! has_error \
    "$results_dir/commands/$name.aot_compile.stdout" \
    "$results_dir/commands/$name.aot_compile.stderr" && [[ -x "$binary" ]]; then
    record "$name.aot_run" expected-failure "$binary"
    run_status=$?
    if (( run_status == 0 )); then
      printf '%s\n' "$name: malformed input succeeded" >> "$results_dir/assertion-failures.txt"
      failures=$((failures + 1))
    elif (( run_status >= 128 )); then
      printf '%s\n' "$name: AOT terminated by signal or timeout instead of explicit error" >> "$results_dir/assertion-failures.txt"
      failures=$((failures + 1))
    elif ! has_diagnostic "$results_dir/commands/$name.aot_run.stdout" \
      "$results_dir/commands/$name.aot_run.stderr"; then
      printf '%s\n' "$name: AOT failed without an actionable diagnostic" >> "$results_dir/assertion-failures.txt"
      failures=$((failures + 1))
    fi
  else
    printf '%s\n' "$name: negative case did not compile cleanly" >> "$results_dir/assertion-failures.txt"
    failures=$((failures + 1))
  fi
  record "$name.jit_run" expected-failure "$compiler" -r "$source" -L "$build_dir"
  jit_status=$?
  if (( jit_status == 0 )); then
    printf '%s\n' "$name: malformed JIT input succeeded" >> "$results_dir/assertion-failures.txt"
    failures=$((failures + 1))
  elif (( jit_status >= 128 )); then
    printf '%s\n' "$name: JIT terminated by signal or timeout instead of explicit error" >> "$results_dir/assertion-failures.txt"
    failures=$((failures + 1))
  elif ! has_diagnostic "$results_dir/commands/$name.jit_run.stdout" \
    "$results_dir/commands/$name.jit_run.stderr"; then
    printf '%s\n' "$name: JIT failed without an actionable diagnostic" >> "$results_dir/assertion-failures.txt"
    failures=$((failures + 1))
  fi
done

printf 'failures=%s\n' "$failures" > "$results_dir/summary.txt"
exit "$(( failures == 0 ? 0 : 1 ))"
