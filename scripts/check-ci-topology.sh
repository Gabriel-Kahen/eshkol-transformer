#!/usr/bin/env bash

set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

python3 - "${project_root}" <<'PY'
from collections import Counter
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
makefile = (root / "Makefile").read_text(encoding="utf-8")
ci = (root / ".github/workflows/ci.yml").read_text(encoding="utf-8")
acceptance = (root / ".github/workflows/acceptance.yml").read_text(encoding="utf-8")

NATIVE_SUITE = "native-numerics"
BLOCKING_TIMEOUT = "${{ matrix.suite == 'native-numerics' && 105 || 75 }}"


def job_body(workflow: str, name: str, next_name: str | None = None) -> str:
    end = rf"(?=^  {re.escape(next_name)}:|\Z)" if next_name else r"\Z"
    match = re.search(
        rf"^  {re.escape(name)}:\n(?P<body>.*?){end}",
        workflow,
        re.MULTILINE | re.DOTALL,
    )
    assert match is not None, name
    return match.group("body")


def job_timeout(body: str) -> str:
    matches = re.findall(r"^    timeout-minutes: (.+)$", body, re.MULTILINE)
    assert len(matches) == 1, matches
    return matches[0]


def job_field(body: str, name: str) -> str:
    matches = re.findall(
        rf"^    {re.escape(name)}: (.+)$", body, re.MULTILINE
    )
    assert len(matches) == 1, (name, matches)
    return matches[0]


def job_env(body: str, name: str) -> str:
    matches = re.findall(
        rf"^      {re.escape(name)}: (.+)$", body, re.MULTILINE
    )
    assert len(matches) == 1, (name, matches)
    return matches[0]


def step_body(body: str, name: str) -> str:
    match = re.search(
        rf"^      - name: {re.escape(name)}\n(?P<body>.*?)(?=^      - |\Z)",
        body,
        re.MULTILINE | re.DOTALL,
    )
    assert match is not None, name
    return match.group("body")


def step_field(body: str, name: str) -> str:
    matches = re.findall(
        rf"^        {re.escape(name)}: (.+)$", body, re.MULTILINE
    )
    assert len(matches) == 1, (name, matches)
    return matches[0]


def step_run(body: str) -> list[str]:
    scalar = re.findall(r"^        run: ([^|].*)$", body, re.MULTILINE)
    block = re.search(
        r"^        run: \|\n(?P<commands>(?:^          .*\n?)+)",
        body,
        re.MULTILINE,
    )
    assert len(scalar) + (block is not None) == 1, (scalar, block)
    if scalar:
        return scalar
    assert block is not None
    return [line[10:] for line in block.group("commands").splitlines()]


def check_workflow_timeouts(workflow: str) -> None:
    assert job_timeout(job_body(workflow, "topology", "blocking")) == "2"
    assert job_timeout(job_body(workflow, "blocking", "f0-linux")) == BLOCKING_TIMEOUT
    assert job_timeout(job_body(workflow, "f0-linux")) == "2"


def check_ci_workflow_contract(workflow: str) -> None:
    check_workflow_timeouts(workflow)
    topology = job_body(workflow, "topology", "blocking")
    blocking = job_body(workflow, "blocking", "f0-linux")
    final = job_body(workflow, "f0-linux")

    assert re.findall(r"^      - run: (.+)$", topology, re.MULTILINE) == [
        "make test-ci-topology"
    ]
    assert step_run(step_body(topology, "Test change selection")) == [
        "python3 -m unittest discover -v -s tests/ci -p 'test_*.py'",
        "python3 -m unittest -v tests.q0.test_python_isolation",
    ]
    assert job_env(blocking, "A0_COMPILER_TIMEOUT_SECONDS") == "'60'"
    assert step_run(step_body(blocking, "Build suite prerequisites")) == [
        'make clean && make "${{ matrix.build_target }}"'
    ]
    assert step_run(step_body(blocking, "Run full suite")) == [
        'make "${{ matrix.test_target }}"'
    ]
    assert step_run(step_body(blocking, "Smoke and benchmark")) == [
        "make smoke-after-build benchmark-after-build"
    ]
    for name in (
        "Select pinned oracle Python",
        "Install pinned development oracle",
        "Smoke and benchmark",
    ):
        assert step_field(step_body(blocking, name), "if") == (
            "matrix.suite == 'native-numerics'"
        )

    assert job_field(final, "if") == "${{ always() }}"
    assert job_field(final, "needs") == "[topology, blocking]"
    assert step_run(
        step_body(final, "Require successful checks for the selected scope")
    ) == [
        'test "$TOPOLOGY_RESULT" = success',
        'case "$RUN_FULL:$BLOCKING_RESULT" in',
        "  true:success|false:skipped) ;;",
        "  *) exit 1 ;;",
        "esac",
    ]


def check_acceptance_workflow_contract(workflow: str) -> None:
    supported = job_body(workflow, "supported-linux")
    assert job_timeout(supported) == "240"
    assert job_env(supported, "A0_COMPILER_TIMEOUT_SECONDS") == "'60'"
    assert step_run(step_body(supported, "Clean build")) == [
        "make clean && make build"
    ]
    assert step_run(step_body(supported, "Full test suite")) == [
        "make test-after-build"
    ]
    assert step_run(step_body(supported, "Smoke and reproducible benchmark")) == [
        "make smoke-after-build benchmark-after-build"
    ]


def assert_timeout_mutation_rejected(workflow: str, replacement: str) -> None:
    mutated = workflow.replace(BLOCKING_TIMEOUT, replacement, 1)
    assert mutated != workflow, replacement
    try:
        check_workflow_timeouts(mutated)
    except AssertionError:
        return
    raise AssertionError(f"CI timeout mutation was admitted: {replacement}")


def assert_mutation_rejected(
    workflow: str,
    old: str,
    new: str,
    checker,
    expected_count: int = 1,
) -> None:
    assert workflow.count(old) == expected_count, (old, workflow.count(old))
    mutated = workflow.replace(old, new)
    try:
        checker(mutated)
    except AssertionError:
        return
    raise AssertionError(f"workflow mutation was admitted: {old!r} -> {new!r}")


def recipes(source: str) -> dict[str, list[str]]:
    parsed: dict[str, list[str]] = {}
    current: str | None = None
    for line in source.splitlines():
        if line and not line[0].isspace() and ":" in line:
            target, _ = line.split(":", 1)
            current = target if re.fullmatch(r"[a-z0-9-]+", target) else None
            if current is not None:
                parsed[current] = []
        elif current is not None and line.startswith("\t"):
            parsed[current].append(line.strip())
        elif line.strip() and not line.startswith("\t"):
            current = None
    return parsed


expected_suites = [
    ("native-numerics", "build-ci-core", "test-ci-core-after-build"),
    ("contracts-data", "build-ci-contracts", "test-ci-contracts-after-build"),
    ("checkpoint-io", "build-ci-checkpoint", "test-ci-checkpoint-after-build"),
    ("parameter-state", "build-ci-parameters", "test-ci-parameters-after-build"),
    ("byte-tokenizer", "build-ci-tokenizer-byte", "test-ci-tokenizer-byte-after-build"),
    ("bpe-tokenizer", "build-ci-tokenizer-bpe", "test-ci-tokenizer-bpe-after-build"),
    (
        "bpe-boundary",
        "build-ci-tokenizer-bpe-boundary",
        "test-ci-tokenizer-bpe-boundary-after-build",
    ),
    ("shard-loader", "build-ci-dataset", "test-ci-dataset-after-build"),
]
actual_suites = re.findall(
    r"- suite: ([a-z0-9-]+)\n\s+build_target: ([a-z0-9-]+)\n"
    r"\s+test_target: ([a-z0-9-]+)",
    ci,
)
assert actual_suites == expected_suites, (actual_suites, expected_suites)

check_ci_workflow_contract(ci)
check_acceptance_workflow_contract(acceptance)
assert ci.count(BLOCKING_TIMEOUT) == 1
expected_suite_timeouts = {
    suite: 105 if suite == NATIVE_SUITE else 75 for suite, _, _ in expected_suites
}
assert expected_suite_timeouts[NATIVE_SUITE] == 105
assert all(
    timeout == 75
    for suite, timeout in expected_suite_timeouts.items()
    if suite != NATIVE_SUITE
)
assert_timeout_mutation_rejected(
    ci, "${{ matrix.suite == 'native-numerics' && 75 || 75 }}"
)
assert_timeout_mutation_rejected(
    ci, "${{ matrix.suite == 'native-numerics' && 105 || 105 }}"
)
assert_mutation_rejected(
    ci,
    '        run: make "${{ matrix.test_target }}"',
    '        run: make "${{ matrix.test_target }}" || true',
    check_ci_workflow_contract,
)
assert_mutation_rejected(
    ci,
    '          test "$TOPOLOGY_RESULT" = success',
    '          # test "$TOPOLOGY_RESULT" = success',
    check_ci_workflow_contract,
)
assert_mutation_rejected(
    ci,
    "        if: matrix.suite == 'native-numerics'",
    "        if: false",
    check_ci_workflow_contract,
    expected_count=3,
)
assert_mutation_rejected(
    acceptance,
    "        run: make test-after-build",
    "        run: make test-after-build || true",
    check_acceptance_workflow_contract,
)
assert_mutation_rejected(
    ci,
    "      A0_COMPILER_TIMEOUT_SECONDS: '60'",
    "      A0_COMPILER_TIMEOUT_SECONDS: '600'",
    check_ci_workflow_contract,
)
assert_mutation_rejected(
    acceptance,
    "      A0_COMPILER_TIMEOUT_SECONDS: '60'",
    "      A0_COMPILER_TIMEOUT_SECONDS: '600'",
    check_acceptance_workflow_contract,
)

targets = recipes(makefile)
for _, build_target, test_target in expected_suites:
    assert build_target in targets, build_target
    assert test_target in targets, test_target
    build_header = re.search(
        rf"^{re.escape(build_target)}:(.*)$", makefile, re.MULTILINE
    )
    assert build_header is not None, build_target
    assert build_header.group(1).strip() == "configure", build_target
    header = re.search(rf"^{re.escape(test_target)}:(.*)$", makefile, re.MULTILINE)
    assert header is not None and not header.group(1).strip(), test_target

expected_ci_build_commands = {
    "build-ci-core": {
        "/usr/bin/bash scripts/generate-p1-roots.sh --check",
        "/usr/bin/bash scripts/compile-smoke.sh \"$${BUILD_DIR:-$$(pwd)/build}\"",
        "/usr/bin/bash scripts/build-k1.sh",
        "/usr/bin/bash scripts/build-l2.sh",
        "/usr/bin/bash scripts/build-i1.sh",
        "/usr/bin/bash scripts/build-a2.sh",
        "/usr/bin/bash scripts/build-i2.sh",
        "/usr/bin/bash scripts/build-k2.sh",
        "/usr/bin/bash scripts/build-n2.sh",
        "/usr/bin/bash scripts/build-n3k.sh",
        "/usr/bin/bash scripts/build-t2.sh",
        "/usr/bin/bash scripts/build-d2.sh",
        "/usr/bin/bash scripts/build-o2.sh",
        "/usr/bin/bash scripts/build-p1-package.sh",
    },
    "build-ci-contracts": {
        "/usr/bin/bash scripts/build-x1.sh",
        "/usr/bin/bash scripts/build-d1.sh",
    },
    "build-ci-checkpoint": {"/usr/bin/bash scripts/build-c1.sh"},
    "build-ci-parameters": set(),
    "build-ci-tokenizer-byte": {"/usr/bin/bash scripts/build-d2.sh"},
    "build-ci-tokenizer-bpe": set(),
    "build-ci-tokenizer-bpe-boundary": {
        "/usr/bin/bash scripts/build-d2.sh",
    },
    "build-ci-dataset": {
        "/usr/bin/bash scripts/build-k1.sh",
        "/usr/bin/bash scripts/build-i1.sh",
        "/usr/bin/bash scripts/build-d2.sh",
    },
}
for target, expected in expected_ci_build_commands.items():
    assert set(targets[target]) == expected, (target, targets[target], expected)

required_build_commands_by_test = {
    "/usr/bin/bash scripts/test-k2.sh": {
        "/usr/bin/bash scripts/build-k1.sh",
        "/usr/bin/bash scripts/build-i2.sh",
        "/usr/bin/bash scripts/build-k2.sh",
    },
    "/usr/bin/bash scripts/test-o2.sh": {
        "/usr/bin/bash scripts/build-k1.sh",
        "/usr/bin/bash scripts/build-i2.sh",
        "/usr/bin/bash scripts/build-t2.sh",
        "/usr/bin/bash scripts/build-d2.sh",
        "/usr/bin/bash scripts/build-o2.sh",
    },
    "/usr/bin/bash scripts/test-t1.sh": {
        "/usr/bin/bash scripts/build-d2.sh",
    },
    "/usr/bin/bash scripts/test-t2-boundary.sh": {
        "/usr/bin/bash scripts/build-d2.sh",
    },
}
for _, build_target, test_target in expected_suites:
    for test_command in targets[test_target]:
        required = required_build_commands_by_test.get(test_command, set())
        missing = required - set(targets[build_target])
        assert not missing, (test_command, build_target, missing)

full_commands = targets["test-after-build"]
sharded_commands = [
    command
    for _, _, test_target in expected_suites
    for command in targets[test_target]
]
assert Counter(sharded_commands) == Counter(full_commands), (
    Counter(full_commands) - Counter(sharded_commands),
    Counter(sharded_commands) - Counter(full_commands),
)
assert len(sharded_commands) == len(full_commands)

required_build_commands = {
    "/usr/bin/bash scripts/generate-p1-roots.sh --check",
    "/usr/bin/bash scripts/build.sh",
    "/usr/bin/bash scripts/build-a2.sh",
    "/usr/bin/bash scripts/build-p1-identity.sh",
    "/usr/bin/bash scripts/build-p1-package.sh",
    "/usr/bin/bash scripts/build-c1.sh",
    "/usr/bin/bash scripts/build-t1.sh",
    "/usr/bin/bash scripts/build-t2.sh",
    "/usr/bin/bash scripts/build-d2.sh",
}
assert required_build_commands.issubset(targets["build"])
build_driver = (root / "scripts/build.sh").read_text(encoding="utf-8")
assert "scripts/build-n2.sh" in build_driver
assert "scripts/build-n3k.sh" in build_driver
assert "scripts/build-o2.sh" in build_driver
assert "scripts/build-k2.sh" in build_driver

for workflow in (ci, acceptance):
    assert "O2_ASAN_DETECT_LEAKS: '1'" in workflow
    assert "K2_ASAN_DETECT_LEAKS: '1'" in workflow
    assert "ATEN_CPU_CAPABILITY=default" in workflow
    assert 'torch.backends.cpu.get_cpu_capability() == "DEFAULT"' in workflow
    assert "N2_ORACLE_PYTHON=$n2_oracle_python" in workflow
    assert "O2_ORACLE_PYTHON=$oracle_python" in workflow
    assert "A2_ORACLE_PYTHON=$oracle_python" in workflow
    assert "Q0_PYTHON=$oracle_python" in workflow
assert job_timeout(job_body(acceptance, "supported-linux")) == "240"

print(
    f"CI TOPOLOGY PASS: {len(expected_suites)} blocking suites cover "
    f"all {len(full_commands)} full test commands exactly once"
)
print(
    "CI BUDGET PASS: native-numerics=105, other blocking suites=75, "
    "control jobs=2, exhaustive=240, A0 compiler timeout=60; "
    "command, condition, gate, and timeout mutations rejected"
)
PY
