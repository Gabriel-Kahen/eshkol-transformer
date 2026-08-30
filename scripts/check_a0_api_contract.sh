#!/usr/bin/env bash
set -euo pipefail

A0_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "${A0_ROOT}/scripts/common.sh"

require_command timeout
require_command ar
verify_toolchain
A0_RUNNER="$(eshkol_build_dir)/eshkol-run"
A0_D1_ARTIFACT_DIR="$(project_build_dir)/d1"
A0_D1_LIBRARY="${A0_D1_ARTIFACT_DIR}/libeshkol_transformer_d1.a"
A0_COMPILER_TIMEOUT_SECONDS=${A0_COMPILER_TIMEOUT_SECONDS:-60}
[[ "${A0_COMPILER_TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*$ ]] || \
    die "A0_COMPILER_TIMEOUT_SECONDS must be a positive integer"

A0_TMP=$(mktemp -d "${TMPDIR:-/tmp}/eshkol-transformer-a0.XXXXXX")
trap 'rm -rf -- "$A0_TMP"' EXIT
A0_JIT_INVOCATION=0

[[ -r "${A0_D1_LIBRARY}" ]] || die "canonical D1 native archive is missing"
[[ "$(ar t "${A0_D1_LIBRARY}")" == "stdlib.o" ]] || \
    die "canonical D1 combined archive has unexpected members"

run_compiler() {
    timeout --foreground --signal=TERM --kill-after=5s \
        "${A0_COMPILER_TIMEOUT_SECONDS}s" "${A0_RUNNER}" "$@"
}

run_fixture() {
    local source=$1
    local jit_cache
    A0_JIT_INVOCATION=$((A0_JIT_INVOCATION + 1))
    jit_cache="$A0_TMP/jit-cache-$A0_JIT_INVOCATION"
    mkdir -p "$jit_cache"
    # A0 is a declaration/import gate.  Use a fresh cache for each repeated JIT
    # invocation so both diagnostics cover the same compilation phase.  Its
    # test-only accessor shim keeps this upstream declaration gate independent of
    # E1B's AOT-only artifact; compile_only_fixture still compiles the real facade.
    XDG_CACHE_HOME="$jit_cache" run_compiler --optimize 0 --no-stdlib \
        -I "$A0_ROOT/tests/fixtures/a0_runtime" \
        -I "$A0_ROOT/lib" \
        -I "$A0_ROOT/tests/fixtures/a0" \
        -L "${A0_D1_ARTIFACT_DIR}" --lib eshkol_transformer_d1 \
        -r "$source"
}

compile_only_fixture() {
    local source=$1
    local output=$2
    run_compiler --strict-types --emit-object --no-stdlib \
        -I "$A0_ROOT/lib" \
        -I "$A0_ROOT/tests/fixtures/a0" \
        "$source" -o "$output"
}

for A0_SOURCE in compile_public_api compile_module_imports; do
    for A0_RUN in 1 2; do
        compile_only_fixture \
            "$A0_ROOT/tests/fixtures/a0/$A0_SOURCE.esk" \
            "$A0_TMP/$A0_SOURCE-$A0_RUN.o" \
            >"$A0_TMP/$A0_SOURCE-$A0_RUN.compile.log" 2>&1
    done
    cmp "$A0_TMP/$A0_SOURCE-1.compile.log" \
        "$A0_TMP/$A0_SOURCE-2.compile.log"
    cmp "$A0_TMP/$A0_SOURCE-1.o" "$A0_TMP/$A0_SOURCE-2.o"
done

for A0_RUN in 1 2; do
    run_fixture \
        "$A0_ROOT/tests/fixtures/a0/compile_public_api.esk" \
        >"$A0_TMP/public-api-$A0_RUN.stdout"
done

cmp "$A0_TMP/public-api-1.stdout" "$A0_TMP/public-api-2.stdout"
grep -Fx "0.1.0-draft" "$A0_TMP/public-api-1.stdout" >/dev/null

for A0_RUN in 1 2; do
    if run_compiler --strict-types --emit-object --no-stdlib \
        -I "$A0_ROOT/lib" \
        -I "$A0_ROOT/tests/fixtures/a0" \
        "$A0_ROOT/tests/fixtures/a0/negative_wrong_arity.esk" \
        -o "$A0_TMP/wrong-arity-$A0_RUN.o" \
        >"$A0_TMP/wrong-arity-$A0_RUN.log" 2>&1; then
        echo "A0 FAIL: wrong-arity fixture unexpectedly compiled" >&2
        exit 1
    fi
    grep -F "Arity mismatch: tokenizer-encode expects 2 arguments but got 1" \
        "$A0_TMP/wrong-arity-$A0_RUN.log" >/dev/null
    test ! -e "$A0_TMP/wrong-arity-$A0_RUN.o"

    if run_fixture \
        "$A0_ROOT/tests/fixtures/a0/negative_unsupported_capability.esk" \
        >"$A0_TMP/unsupported-capability-$A0_RUN.log" 2>&1; then
        echo "A0 FAIL: unsupported-capability fixture unexpectedly succeeded" >&2
        exit 1
    fi
    grep -Fx "unsupported: capability-require declaration guard" \
        "$A0_TMP/unsupported-capability-$A0_RUN.log" >/dev/null
done

cmp "$A0_TMP/wrong-arity-1.log" "$A0_TMP/wrong-arity-2.log"
cmp "$A0_TMP/unsupported-capability-1.log" \
    "$A0_TMP/unsupported-capability-2.log"

for A0_RUN in 1 2; do
    if run_compiler --strict-types --emit-object --no-stdlib \
        -I "$A0_ROOT/lib" \
        -I "$A0_ROOT/tests/fixtures/a0" \
        "$A0_ROOT/tests/fixtures/a0/negative_d1_wrong_arity.esk" \
        -o "$A0_TMP/d1-wrong-arity-$A0_RUN.o" \
        >"$A0_TMP/d1-wrong-arity-$A0_RUN.log" 2>&1; then
        echo "A0 FAIL: D1 wrong-arity fixture unexpectedly compiled" >&2
        exit 1
    fi
    grep -F "Arity mismatch: token-corpus-write! expects 5 arguments but got 4" \
        "$A0_TMP/d1-wrong-arity-$A0_RUN.log" >/dev/null
    test ! -e "$A0_TMP/d1-wrong-arity-$A0_RUN.o"
done
cmp "$A0_TMP/d1-wrong-arity-1.log" "$A0_TMP/d1-wrong-arity-2.log"

echo "A0 PASS: declarations/imports repeated; expected negatives rejected"
