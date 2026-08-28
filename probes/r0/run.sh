#!/usr/bin/env bash
set -u

readonly CANONICAL_REPOSITORY="https://github.com/tsotchke/eshkol.git"
readonly CANONICAL_VERSION="Eshkol Compiler v1.3.4-evolve"

usage() {
  cat >&2 <<'EOF'
usage: run.sh --eshkol-source DIR --work-dir DIR --results-dir DIR
              [--existing-build DIR] [--probe NAME]
              [--run-timeout SECONDS] [--compile-timeout SECONDS]
              [--build-timeout SECONDS]
EOF
  exit 64
}

eshkol_source= work_dir= results_dir= existing_build= selected_probe=
run_timeout=${R0_RUN_TIMEOUT_SECONDS:-90}
compile_timeout=${R0_COMPILE_TIMEOUT_SECONDS:-300}
build_timeout=${R0_BUILD_TIMEOUT_SECONDS:-1200}
while (( $# > 0 )); do
  case "$1" in
    --eshkol-source) eshkol_source=${2:-}; shift 2 ;;
    --work-dir) work_dir=${2:-}; shift 2 ;;
    --results-dir) results_dir=${2:-}; shift 2 ;;
    --existing-build) existing_build=${2:-}; shift 2 ;;
    --probe) selected_probe=${2:-}; shift 2 ;;
    --run-timeout) run_timeout=${2:-}; shift 2 ;;
    --compile-timeout) compile_timeout=${2:-}; shift 2 ;;
    --build-timeout) build_timeout=${2:-}; shift 2 ;;
    *) usage ;;
  esac
done
[[ -n "$eshkol_source" && -n "$work_dir" && -n "$results_dir" ]] || usage
for directory in "$eshkol_source" "$work_dir" "$results_dir"; do
  [[ "$directory" = /* ]] || { echo "all directory arguments must be absolute" >&2; exit 64; }
done
[[ -z "$existing_build" || "$existing_build" = /* ]] || {
  echo "--existing-build must be absolute" >&2; exit 64;
}
for seconds in "$run_timeout" "$compile_timeout" "$build_timeout"; do
  [[ "$seconds" =~ ^[1-9][0-9]*$ ]] || { echo "timeouts must be positive integers" >&2; exit 64; }
done
if [[ -d "$results_dir" && -n "$(find "$results_dir" -mindepth 1 -print -quit)" ]]; then
  echo "results directory must be absent or empty: $results_dir" >&2
  exit 64
fi

probe_dir=$(cd "$(dirname "$0")" && pwd)
pin=$(tr -d '[:space:]' < "$probe_dir/eshkol.rev")
[[ "$pin" =~ ^[0-9a-f]{40}$ ]] || { echo "invalid pinned Eshkol revision: $pin" >&2; exit 65; }
if [[ -n "$selected_probe" ]]; then
  [[ "$selected_probe" =~ ^[a-z0-9_]+$ && -f "$probe_dir/$selected_probe.esk" ]] || {
    echo "unknown positive probe: $selected_probe" >&2; exit 64;
  }
fi

mkdir -p "$work_dir" "$work_dir/tmp" "$work_dir/run" "$results_dir/commands"
export TMPDIR="$work_dir/tmp"
manifest="$results_dir/manifest.tsv"
printf 'name\texit_code\texpectation\tcommand\n' > "$manifest"
: > "$results_dir/assertion-failures.txt"
: > "$results_dir/parity-failures.txt"
record_cwd="$work_dir/run"

record() {
  local name=$1 classification=$2 timeout_seconds=$run_timeout virtual_kib=2097152
  shift 2
  case "$classification" in
    build|upstream-tests) timeout_seconds=$build_timeout; virtual_kib=0 ;;
    discovery) virtual_kib=0 ;;
    *compile*) timeout_seconds=$compile_timeout; virtual_kib=4194304 ;;
  esac
  local stdout="$results_dir/commands/$name.stdout"
  local stderr="$results_dir/commands/$name.stderr"
  local command_file="$results_dir/commands/$name.command"
  printf 'cwd=%q timeout=%ss stream_limit_bytes=2097152 virtual_kib=%s\n' \
    "$record_cwd" "$timeout_seconds" "$virtual_kib" > "$command_file"
  printf '%q ' "$@" >> "$command_file"; printf '\n' >> "$command_file"
  (
    cd "$record_cwd" || exit 70
    ulimit -c 0
    (( virtual_kib == 0 )) || ulimit -v "$virtual_kib"
    /usr/bin/timeout --signal=TERM --kill-after=5s "$timeout_seconds" "$@" \
      > >(head -c 2097152 > "$stdout") 2> >(head -c 2097152 > "$stderr")
    status=$?; wait; exit "$status"
  )
  local status=$?
  printf '%s\t%s\t%s\t' "$name" "$status" "$classification" >> "$manifest"
  tail -n 1 "$command_file" | tr '\n' ' ' >> "$manifest"; printf '\n' >> "$manifest"
  return "$status"
}

actual=$(git -C "$eshkol_source" rev-parse HEAD 2>"$results_dir/pin.stderr") || {
  echo "cannot read Eshkol revision" >&2; exit 65;
}
origin=$(git -C "$eshkol_source" remote get-url origin 2>>"$results_dir/pin.stderr") || {
  echo "cannot read Eshkol origin" >&2; exit 65;
}
printf '%s\n' "$pin" > "$results_dir/expected-eshkol-revision.txt"
printf '%s\n' "$actual" > "$results_dir/actual-eshkol-revision.txt"
printf '%s\n' "$CANONICAL_REPOSITORY" > "$results_dir/expected-eshkol-repository.txt"
printf '%s\n' "$origin" > "$results_dir/actual-eshkol-repository.txt"
[[ "$actual" == "$pin" ]] || {
  echo "Eshkol revision mismatch: expected $pin, got $actual" >&2; exit 65;
}
[[ "$origin" == "$CANONICAL_REPOSITORY" ]] || {
  echo "Eshkol repository mismatch: expected $CANONICAL_REPOSITORY, got $origin" >&2; exit 65;
}
dirty=$(git -C "$eshkol_source" status --porcelain)
[[ -z "$dirty" ]] || {
  printf '%s\n' "$dirty" > "$results_dir/dirty-upstream.txt"
  echo "Eshkol checkout has modifications or untracked files" >&2; exit 65;
}
export ESHKOL_PATH="$eshkol_source/lib${ESHKOL_PATH:+:$ESHKOL_PATH}"

if [[ -n "$existing_build" ]]; then
  build_dir=$existing_build
  compiler="$build_dir/eshkol-run"
  build_mode=supplied-existing-build
  [[ -x "$compiler" ]] || { echo "missing executable: $compiler" >&2; exit 65; }
else
  build_dir="$work_dir/build"
  compiler="$build_dir/eshkol-run"
  build_mode=harness-full-build
  if [[ -d "$build_dir" && -n "$(find "$build_dir" -mindepth 1 -print -quit)" ]]; then
    echo "full-build directory must be absent or empty: $build_dir" >&2; exit 65
  fi
fi

if [[ "$build_mode" == harness-full-build ]]; then
  cmake_args=(-S "$eshkol_source" -B "$build_dir" -DCMAKE_BUILD_TYPE=RelWithDebInfo -DESHKOL_BUILD_TESTS=ON)
  [[ -z "${ESHKOL_LLVM_CONFIG:-}" ]] || cmake_args+=("-DLLVM_CONFIG_EXECUTABLE=$ESHKOL_LLVM_CONFIG")
  record configure build cmake "${cmake_args[@]}" || exit 1
  record build build cmake --build "$build_dir" --parallel 2 || exit 1
  record upstream_ctest upstream-tests ctest --test-dir "$build_dir" --output-on-failure
  ctest_status=$?
else
  ctest_status=0
fi

compiler_version=$($compiler --version 2>&1 | sed -n '1p')
compiler_sha256=$(sha256sum "$compiler" | awk '{print $1}')
printf '%s\n' "$CANONICAL_VERSION" > "$results_dir/expected-compiler-version.txt"
printf '%s\n' "$compiler_version" > "$results_dir/actual-compiler-version.txt"
printf '%s\n' "$compiler_sha256" > "$results_dir/compiler.sha256"
[[ "$compiler_version" == "$CANONICAL_VERSION" ]] || {
  echo "Eshkol compiler identity mismatch: expected '$CANONICAL_VERSION', got '$compiler_version'" >&2; exit 65;
}
build_provenance=not-applicable-full-build
if [[ "$build_mode" == supplied-existing-build ]]; then
  provenance="$build_dir/eshkol-transformer-provenance.tsv"
  [[ -f "$provenance" ]] || { echo "missing F0 build provenance: $provenance" >&2; exit 65; }
  provenance_value() {
    awk -F '\t' -v key="$1" '$1 == key { print $2; found=1; exit } END { if (!found) exit 1 }' "$provenance"
  }
  provenance_format=$(provenance_value format_version) || { echo "invalid F0 provenance format" >&2; exit 65; }
  provenance_repository=$(provenance_value eshkol_repository) || { echo "missing repository in F0 provenance" >&2; exit 65; }
  provenance_commit=$(provenance_value eshkol_commit) || { echo "missing commit in F0 provenance" >&2; exit 65; }
  provenance_source=$(provenance_value eshkol_source_dir) || { echo "missing source in F0 provenance" >&2; exit 65; }
  provenance_sha256=$(provenance_value eshkol_binary_sha256) || { echo "missing binary hash in F0 provenance" >&2; exit 65; }
  [[ "$provenance_format" == 1 ]] || { echo "unsupported F0 provenance format: $provenance_format" >&2; exit 65; }
  [[ "$provenance_repository" == "$CANONICAL_REPOSITORY" ]] || { echo "F0 provenance repository mismatch" >&2; exit 65; }
  [[ "$provenance_commit" == "$pin" ]] || { echo "F0 provenance commit mismatch" >&2; exit 65; }
  [[ "$provenance_sha256" == "$compiler_sha256" ]] || { echo "F0 provenance binary hash mismatch" >&2; exit 65; }
  [[ "$(realpath "$provenance_source")" == "$(realpath "$eshkol_source")" ]] || { echo "F0 provenance source mismatch" >&2; exit 65; }
  cp -- "$provenance" "$results_dir/f0-build-provenance.tsv"
  build_provenance="$results_dir/f0-build-provenance.tsv"
fi

{
  echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "eshkol_repository=$origin"
  echo "eshkol_revision=$actual"
  echo "compiler_version=$compiler_version"
  echo "compiler_sha256=$compiler_sha256"
  echo "build_mode=$build_mode"
  echo "build_provenance=$build_provenance"
  echo "build_dir=$build_dir"
  echo "run_timeout_seconds=$run_timeout"
  echo "compile_timeout_seconds=$compile_timeout"
  echo "build_timeout_seconds=$build_timeout"
  echo "tmpdir=$TMPDIR"
  echo "transformer_revision=$(git -C "$probe_dir/../.." rev-parse HEAD 2>/dev/null || echo unavailable)"
  echo "transformer_status=$(git -C "$probe_dir/../.." status --porcelain 2>/dev/null | tr '\n' ';' || echo unavailable)"
  echo "uname=$(uname -a)"
  echo "cmake=$(cmake --version | sed -n '1p')"
  echo "ninja=$(ninja --version 2>/dev/null || echo unavailable)"
  selected_llvm_config=${ESHKOL_LLVM_CONFIG:-llvm-config}
  echo "llvm_config_path=$(command -v "$selected_llvm_config" 2>/dev/null || echo "$selected_llvm_config")"
  echo "llvm_config=$($selected_llvm_config --version 2>/dev/null || echo unavailable)"
  echo "cpu=$(lscpu 2>/dev/null | tr '\n' ';' || echo unavailable)"
  echo "nvidia=$(nvidia-smi -L 2>/dev/null | tr '\n' ';' || echo unavailable)"
  echo "rocm=$(rocminfo 2>/dev/null | sed -n '1,80p' | tr '\n' ';' || echo unavailable)"
} > "$results_dir/environment.txt"

record compiler_help discovery "$compiler" --help
failures=0
(( ctest_status == 0 )) || failures=$((failures + 1))
if [[ "$build_mode" == harness-full-build ]] && grep -q 'No tests were found' "$results_dir/commands/upstream_ctest.stderr"; then
  printf '%s\n' 'upstream ctest: zero tests registered' >> "$results_dir/assertion-failures.txt"
  failures=$((failures + 1))
fi

has_diagnostic() {
  grep -Eiq 'error|invalid|mismatch|bound|range|shape|dimension|cannot|missing|not found|unsupported|failed' "$@"
}
validate_positive() {
  local name=$1 phase=$2 stdout="$results_dir/commands/$1.$2.stdout" stderr="$results_dir/commands/$1.$2.stderr"
  if grep -Eiq 'ERROR:|Failed to|terminate called|Assertion' "$stderr"; then
    echo "$name: $phase emitted an error diagnostic" >> "$results_dir/assertion-failures.txt"; failures=$((failures + 1))
  fi
  if grep -q FAIL "$stdout"; then
    echo "$name: $phase emitted a FAIL marker" >> "$results_dir/assertion-failures.txt"; failures=$((failures + 1))
  fi
  if ! grep -qx "R0_DONE $name" "$stdout"; then
    echo "$name: $phase did not reach its completion marker" >> "$results_dir/assertion-failures.txt"; failures=$((failures + 1))
  fi
}

if [[ -n "$selected_probe" ]]; then positive_sources=("$probe_dir/$selected_probe.esk"); else positive_sources=("$probe_dir"/*.esk); fi
for source in "${positive_sources[@]}"; do
  name=$(basename "$source" .esk); binary="$work_dir/$name.aot"; rm -f -- "$binary"; aot_ready=1
  record "$name.aot_compile" capability-compile "$compiler" "$source" -o "$binary" -L "$build_dir" || {
    failures=$((failures + 1)); aot_ready=0;
  }
  if (( aot_ready == 1 )) && grep -Eiq 'ERROR:|Failed to|terminate called|Assertion' \
    "$results_dir/commands/$name.aot_compile.stdout" "$results_dir/commands/$name.aot_compile.stderr"; then
    echo "$name: AOT compilation emitted an error diagnostic" >> "$results_dir/assertion-failures.txt"
    failures=$((failures + 1)); aot_ready=0
  fi
  if (( aot_ready == 1 )) && [[ ! -x "$binary" ]]; then
    echo "$name: AOT compilation returned zero without an executable" >> "$results_dir/assertion-failures.txt"
    failures=$((failures + 1)); aot_ready=0
  fi
  if (( aot_ready == 1 )); then
    if [[ "$name" == memory_loop && -x /usr/bin/time ]]; then runner=(/usr/bin/time -v "$binary"); else runner=("$binary"); fi
    record "$name.aot_run_1" capability-run "${runner[@]}" || failures=$((failures + 1))
    record "$name.aot_run_2" deterministic-repeat "${runner[@]}" || failures=$((failures + 1))
    validate_positive "$name" aot_run_1; validate_positive "$name" aot_run_2
    cmp -s "$results_dir/commands/$name.aot_run_1.stdout" "$results_dir/commands/$name.aot_run_2.stdout" || {
      echo "$name: repeated AOT output differs" >> "$results_dir/parity-failures.txt"; failures=$((failures + 1));
    }
  fi
  record "$name.jit_run" jit "$compiler" -r "$source" -L "$build_dir" || failures=$((failures + 1))
  record "$name.jit_run_2" jit-repeat "$compiler" -r "$source" -L "$build_dir" || failures=$((failures + 1))
  validate_positive "$name" jit_run; validate_positive "$name" jit_run_2
  if (( aot_ready == 1 )); then
    cmp -s "$results_dir/commands/$name.aot_run_1.stdout" "$results_dir/commands/$name.jit_run.stdout" || {
      echo "$name: AOT/JIT stdout differs" >> "$results_dir/parity-failures.txt"; failures=$((failures + 1));
    }
  fi
  cmp -s "$results_dir/commands/$name.jit_run.stdout" "$results_dir/commands/$name.jit_run_2.stdout" || {
    echo "$name: repeated JIT output differs" >> "$results_dir/parity-failures.txt"; failures=$((failures + 1));
  }
done

if [[ -z "$selected_probe" ]]; then
  for source in "$probe_dir"/negative/*.esk; do
    name=negative.$(basename "$source" .esk); binary="$work_dir/$name.aot"; rm -f -- "$binary"
    record "$name.aot_compile" expected-failure-compile "$compiler" "$source" -o "$binary" -L "$build_dir"
    compile_status=$?; compile_rejected=0
    if (( compile_status == 124 || compile_status >= 128 )); then
      echo "$name: AOT rejection timed out or terminated by signal" >> "$results_dir/assertion-failures.txt"; failures=$((failures + 1))
    elif (( compile_status != 0 )); then
      if has_diagnostic "$results_dir/commands/$name.aot_compile.stdout" "$results_dir/commands/$name.aot_compile.stderr"; then
        compile_rejected=1
      else
        echo "$name: AOT compile failed without an actionable diagnostic" >> "$results_dir/assertion-failures.txt"; failures=$((failures + 1))
      fi
    elif [[ ! -x "$binary" ]]; then
      echo "$name: AOT compile returned zero without executable or diagnostic" >> "$results_dir/assertion-failures.txt"; failures=$((failures + 1))
    fi
    if (( compile_rejected == 0 )) && [[ -x "$binary" ]]; then
      record "$name.aot_run" expected-failure-run "$binary"; run_status=$?
      if (( run_status == 124 || run_status >= 128 )); then
        echo "$name: AOT rejection timed out or terminated by signal" >> "$results_dir/assertion-failures.txt"; failures=$((failures + 1))
      elif (( run_status == 0 )); then
        echo "$name: malformed AOT input succeeded" >> "$results_dir/assertion-failures.txt"; failures=$((failures + 1))
      elif ! has_diagnostic "$results_dir/commands/$name.aot_run.stdout" "$results_dir/commands/$name.aot_run.stderr"; then
        echo "$name: AOT failed without an actionable diagnostic" >> "$results_dir/assertion-failures.txt"; failures=$((failures + 1))
      fi
    fi
    record "$name.jit_run" expected-failure-run "$compiler" -r "$source" -L "$build_dir"; jit_status=$?
    if (( jit_status == 124 || jit_status >= 128 )); then
      echo "$name: JIT rejection timed out or terminated by signal" >> "$results_dir/assertion-failures.txt"; failures=$((failures + 1))
    elif (( jit_status == 0 )); then
      echo "$name: malformed JIT input succeeded" >> "$results_dir/assertion-failures.txt"; failures=$((failures + 1))
    elif ! has_diagnostic "$results_dir/commands/$name.jit_run.stdout" "$results_dir/commands/$name.jit_run.stderr"; then
      echo "$name: JIT failed without an actionable diagnostic" >> "$results_dir/assertion-failures.txt"; failures=$((failures + 1))
    fi
  done
fi

printf 'failures=%s\n' "$failures" > "$results_dir/summary.txt"
exit "$(( failures == 0 ? 0 : 1 ))"
