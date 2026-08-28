#!/usr/bin/env bash
set -euo pipefail

A0_ROOT=$(cd "$(dirname "$0")/.." && pwd)
A0_RUNNER=${ESHKOL_RUNNER:-}

if [[ -z "$A0_RUNNER" ]]; then
    for A0_CANDIDATE in \
        "$A0_ROOT/../eshkol/build/eshkol-run" \
        "$A0_ROOT/../eshkol/build-poet/eshkol-run"; do
        if [[ -x "$A0_CANDIDATE" ]]; then
            A0_RUNNER=$A0_CANDIDATE
            break
        fi
    done
fi

if [[ -z "$A0_RUNNER" || ! -x "$A0_RUNNER" ]]; then
    echo "A0 BLOCKED: set ESHKOL_RUNNER to an executable eshkol-run" >&2
    exit 2
fi

A0_TMP=$(mktemp -d "${TMPDIR:-/tmp}/eshkol-transformer-a0.XXXXXX")
trap 'rm -rf -- "$A0_TMP"' EXIT

run_fixture() {
    local source=$1
    "$A0_RUNNER" --no-stdlib \
        -I "$A0_ROOT/lib" \
        -I "$A0_ROOT/tests/fixtures/a0" \
        -r "$source"
}

compile_only_fixture() {
    local source=$1
    local output=$2
    "$A0_RUNNER" --strict-types --emit-object --no-stdlib \
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
    if "$A0_RUNNER" --strict-types --emit-object --no-stdlib \
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

echo "A0 PASS: declarations/imports repeated; expected negatives rejected"
