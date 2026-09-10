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
    ("parameter-state-public", "build-ci-parameters", "test-ci-parameters-public-after-build"),
    ("parameter-state-state", "build-ci-parameters", "test-ci-parameters-state-after-build"),
    ("parameter-state-registry", "build-ci-parameters", "test-ci-parameters-registry-after-build"),
    ("byte-tokenizer", "build-ci-tokenizer-byte", "test-ci-tokenizer-byte-after-build"),
    ("bpe-tokenizer", "build-ci-tokenizer-bpe", "test-ci-tokenizer-bpe-after-build"),
    (
        "bpe-boundary-production",
        "build-ci-tokenizer-bpe-boundary-production",
        "test-ci-tokenizer-bpe-boundary-production-after-build",
    ),
    (
        "bpe-boundary-d1-test",
        "build-ci-tokenizer-bpe-boundary-d1-test",
        "test-ci-tokenizer-bpe-boundary-d1-test-after-build",
    ),
    (
        "bpe-boundary-public-caller",
        "build-ci-tokenizer-bpe-boundary",
        "test-ci-tokenizer-bpe-boundary-public-caller-after-build",
    ),
    ("shard-loader-semantics", "build-ci-dataset", "test-ci-dataset-semantics-after-build"),
    ("shard-loader-resources", "build-ci-dataset", "test-ci-dataset-resources-after-build"),
    ("shard-loader-packaging", "build-ci-dataset", "test-ci-dataset-packaging-after-build"),
]
actual_suites = re.findall(
    r"- suite: ([a-z0-9-]+)\n\s+build_target: ([a-z0-9-]+)\n"
    r"\s+test_target: ([a-z0-9-]+)",
    ci,
)
assert actual_suites == expected_suites, (actual_suites, expected_suites)

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
        "/usr/bin/bash scripts/build-n2.sh",
        "/usr/bin/bash scripts/build-n3k.sh",
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
    "build-ci-tokenizer-bpe-boundary-production": set(),
    "build-ci-tokenizer-bpe-boundary-d1-test": set(),
    "build-ci-dataset": {
        "/usr/bin/bash scripts/build-k1.sh",
        "/usr/bin/bash scripts/build-i1.sh",
        "/usr/bin/bash scripts/build-d2.sh",
    },
}
for target, expected in expected_ci_build_commands.items():
    assert set(targets[target]) == expected, (target, targets[target], expected)

required_build_commands_by_test = {
    "/usr/bin/bash scripts/test-t1.sh": {
        "/usr/bin/bash scripts/build-d2.sh",
    },
    "/usr/bin/bash scripts/test-t2-boundary.sh --phase public-caller": {
        "/usr/bin/bash scripts/build-d2.sh",
    },
}
for _, build_target, test_target in expected_suites:
    for test_command in targets[test_target]:
        required = required_build_commands_by_test.get(test_command, set())
        missing = required - set(targets[build_target])
        assert not missing, (test_command, build_target, missing)

# Full local/acceptance scripts prepare shared artifacts once. CI runs their
# complete phase union independently; phase-dispatch tests enforce that contract.
phase_commands = {
    "/usr/bin/bash scripts/test-t2-boundary.sh": [
        f"/usr/bin/bash scripts/test-t2-boundary.sh --phase {phase}"
        for phase in ("production", "d1-test", "public-caller")
    ],
    "/usr/bin/bash scripts/test-p1.sh": [
        f"/usr/bin/bash scripts/test-p1.sh --phase {phase}"
        for phase in ("public", "state", "registry")
    ],
    "/usr/bin/bash scripts/test-d2.sh": [
        f"/usr/bin/bash scripts/test-d2.sh --phase {phase}"
        for phase in ("semantics", "resources", "packaging")
    ],
}
for command in phase_commands:
    assert targets["test-after-build"].count(command) == 1, command
full_commands = [
    phase
    for command in targets["test-after-build"]
    for phase in phase_commands.get(command, [command])
]
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

for command in (
    "make clean && make build",
    "make test-after-build",
    "make smoke-after-build benchmark-after-build",
):
    assert command in acceptance, command

print(
    f"CI TOPOLOGY PASS: {len(expected_suites)} blocking suites cover "
    f"all {len(full_commands)} test phases exactly once"
)
PY
