"""Fail-closed structural coverage checks for the shared full CI engine."""
from collections import Counter
from pathlib import Path
import re

GROUPS = ("format", "state", "save", "load", "public", "operational")
SUITES = [
    ("canonical-build", "test-ci-clean-build-after-build", 75),
    ("native-numerics", "test-ci-core-after-build", 105),
    ("contracts-data", "test-ci-contracts-after-build", 75),
    ("checkpoint-io", "test-ci-checkpoint-after-build", 75),
    ("parameter-state", "test-ci-parameters-after-build", 75),
    ("byte-tokenizer", "test-ci-tokenizer-byte-after-build", 75),
    ("bpe-tokenizer", "test-ci-tokenizer-bpe-after-build", 75),
    ("bpe-boundary", "test-ci-tokenizer-bpe-boundary-after-build", 75),
    ("shard-loader", "test-ci-dataset-after-build", 75),
] + [(f"c2-{g}", f"test-ci-c2-{g}-after-build", 240) for g in GROUPS]
FULL = ["/usr/bin/bash scripts/" + name for name in (
    "test.sh", "check_a0_api_contract.sh", "test-k1.sh", "test-a2.sh",
    "test-l2.sh", "test-e1.sh", "test-e1b.sh", "test-i1.sh", "test-i2.sh",
    "test-k2.sh", "test-n2.sh", "test-n3k.sh", "test-o2.sh", "test-x1.sh",
    "test-p1.sh", "test-d1.sh", "test-d2.sh", "test-c1.sh", "test-c2.sh",
    "test-t1.sh", "test-t2.sh --runtime-only", "test-t2-boundary.sh", "test-q0.sh",
)]
CHECKOUT = "actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683"


def jobs(text):
    text = text.split("\njobs:\n", 1)[1]
    return {m[1]: m[2] for m in re.finditer(
        r"^  ([a-z0-9-]+):\n(.*?)(?=^  [a-z0-9-]+:\n|\Z)", text, re.M | re.S)}


def field(text, indent, name):
    values = re.findall(rf"^{' ' * indent}{re.escape(name)}: (.+)$", text, re.M)
    assert len(values) == 1, (name, values)
    return values[0]


def steps(body):
    result = {}
    for m in re.finditer(r"^      - name: ([^\n]+)\n(.*?)(?=^      - |\Z)", body, re.M | re.S):
        assert m[1] not in result
        result[m[1]] = m[2]
    return result


def run(body):
    scalar = re.findall(r"^        run: ([^|].*)$", body, re.M)
    block = re.search(r"^        run: \|\n((?:^          .*\n?)+)", body, re.M)
    assert len(scalar) + (block is not None) == 1
    return scalar if scalar else [line[10:] for line in block[1].splitlines()]


def recipes(text):
    result = {}
    current = None
    for line in text.splitlines():
        match = re.fullmatch(r"([a-z0-9-]+):(.*)", line)
        if match:
            current = match[1]
            result[current] = []
        elif current and line.startswith("\t"):
            result[current].append(line.strip())
        elif line.strip():
            current = None
    return result


def check(root, overrides=None):
    overrides = overrides or {}
    def read(path):
        return overrides[path] if path in overrides else (root / path).read_text()
    ci = read(".github/workflows/ci.yml")
    acceptance = read(".github/workflows/acceptance.yml")
    engine = read(".github/workflows/full-coverage.yml")
    make = read("Makefile")
    cj, aj, ej = jobs(ci), jobs(acceptance), jobs(engine)
    assert set(cj) == {"topology", "blocking", "f0-linux"}
    assert set(aj) == {"select", "full", "accepted"}
    assert set(ej) == {"suites", "evidence"}
    for workflow in (ci, acceptance, engine):
        assert "continue-on-error" not in workflow
        assert "pull_request_target" not in workflow
    assert "  workflow_call:" in engine
    assert "  workflow_dispatch:" in ci
    assert "  actions: read\n" in acceptance.split("\njobs:\n", 1)[0]
    for body in (cj["blocking"], aj["full"]):
        assert field(body, 4, "uses") == "./.github/workflows/full-coverage.yml"
    assert field(cj["blocking"], 4, "needs") == "topology"
    assert field(cj["blocking"], 4, "if") == "needs.topology.outputs.run_full == 'true'"
    assert field(aj["full"], 4, "needs") == "select"
    assert field(aj["full"], 4, "if") == "needs.select.outputs.reused == 'false'"
    assert field(cj["f0-linux"], 4, "needs") == "[topology, blocking]"
    assert field(aj["accepted"], 4, "needs") == "[select, full]"
    assert field(cj["f0-linux"], 4, "if") == "${{ always() }}"
    assert field(aj["accepted"], 4, "if") == "${{ always() }}"
    assert run(steps(cj["f0-linux"])["Require successful checks for the selected scope"]) == [
        'test "$TOPOLOGY_RESULT" = success', 'case "$RUN_FULL:$BLOCKING_RESULT" in',
        "  true:success|false:skipped) ;;", "  *) exit 1 ;;", "esac"]
    assert run(steps(aj["accepted"])["Require verified evidence or fresh complete coverage"]) == [
        'test "$SELECTION_RESULT" = success', 'case "$REUSED:$FULL_RESULT" in',
        "  true:skipped|false:success) ;;", "  *) exit 1 ;;", "esac"]
    assert run(steps(aj["select"])["Find completed exact-tree full CI"]) == [
        "python3 tests/ci/evidence.py select"]
    assert field(steps(aj["select"])["Find completed exact-tree full CI"], 10, "CI_EVENT") == "${{ github.event_name }}"
    assert run(steps(aj["select"])["Validate topology and selection"]) == [
        "make test-ci-topology",
        "python3 -m unittest discover -v -s tests/ci -p 'test_*.py'",
        "python3 -m unittest -v tests.q0.test_python_isolation"]
    assert run(steps(cj["topology"])["Test change selection"]) == [
        "python3 -m unittest discover -v -s tests/ci -p 'test_*.py'",
        "python3 -m unittest -v tests.q0.test_python_isolation"]
    assert "      - run: make test-ci-topology\n" in cj["topology"]
    suite = ej["suites"]
    assert field(suite, 4, "runs-on") == "ubuntu-22.04"
    assert field(suite, 4, "timeout-minutes") == "${{ matrix.timeout }}"
    assert field(suite, 6, "fail-fast") == "false"
    matrix = re.findall(r"- suite: ([a-z0-9-]+)\n\s+test_target: ([a-z0-9-]+)\n\s+timeout: ([0-9]+)", suite)
    assert [(s, t, int(v)) for s, t, v in matrix] == SUITES
    for name, value in {
        "CC": "clang-21", "CXX": "clang++-21", "ESHKOL_CXX_COMPILER": "clang++-21",
        "ESHKOL_JOBS": "'2'",
        "LLVM_CONFIG_EXECUTABLE": "llvm-config-21", "A0_COMPILER_TIMEOUT_SECONDS": "'60'",
        **{k: "'1'" for k in ("P1_LSAN", "C1_LSAN", "I2_ASAN_DETECT_LEAKS",
           "K2_ASAN_DETECT_LEAKS", "D2_ASAN_DETECT_LEAKS", "N2_ASAN_DETECT_LEAKS",
           "O2_ASAN_DETECT_LEAKS", "N3K_ASAN_DETECT_LEAKS")}
    }.items():
        assert field(suite, 6, name) == value
    ss = steps(suite)
    assert set(ss) == {"Install host dependencies", "Restore pinned Eshkol toolchain",
        "Select pinned oracle Python", "Install pinned development oracle",
        "Verify pinned Eshkol toolchain", "Build suite prerequisites", "Run full suite"}
    assert suite.count("        if:") == 2
    for name in ("Select pinned oracle Python", "Install pinned development oracle"):
        assert field(ss[name], 8, "if") == "matrix.suite == 'native-numerics'"
    assert run(ss["Build suite prerequisites"]) == [
        "started=$SECONDS", 'if [[ "${{ matrix.suite }}" == canonical-build ]]; then',
        "  make clean && make build", "else",
        '  /usr/bin/bash scripts/ci-build-prerequisites.sh "${{ matrix.suite }}"', "fi",
        'echo "prerequisite_seconds=$((SECONDS - started))" >> "$GITHUB_STEP_SUMMARY"']
    assert run(ss["Run full suite"]) == [
        "started=$SECONDS", 'make "${{ matrix.test_target }}"',
        'echo "suite_seconds=$((SECONDS - started))" >> "$GITHUB_STEP_SUMMARY"']
    for name in ("Install host dependencies", "Install pinned development oracle",
                 "Verify pinned Eshkol toolchain", "Build suite prerequisites", "Run full suite"):
        assert field(ss[name], 8, "shell") == "/usr/bin/bash -euo pipefail {0}"
    assert run(ss["Verify pinned Eshkol toolchain"]) == ["make toolchain"]
    assert 'test "$(llvm-config-21 --version)" = \'21.1.8\'' in run(ss["Install host dependencies"])
    assert "          python-version: '3.14.6'" in ss["Select pinned oracle Python"]
    assert field(ss["Select pinned oracle Python"], 8, "uses") == "actions/setup-python@ece7cb06caefa5fff74198d8649806c4678c61a1"
    assert field(ss["Restore pinned Eshkol toolchain"], 8, "uses") == "actions/cache@0057852bfaa89a56745cba8c7296529d2fc39830"
    assert 'ATEN_CPU_CAPABILITY=default' in ss["Install pinned development oracle"]
    assert 'MKL_CBWR=COMPATIBLE' in ss["Install pinned development oracle"]
    assert suite.count("MKL_CBWR") == 2
    assert 'os.environ.get("ATEN_CPU_CAPABILITY") == "default"' in ss["Install pinned development oracle"]
    assert 'os.environ.get("MKL_CBWR") == "COMPATIBLE"' in ss["Install pinned development oracle"]
    assert 'torch.backends.cpu.get_cpu_capability() == "DEFAULT"' in ss["Install pinned development oracle"]
    for variable, value in (("N2", "n2_oracle_python"), ("O2", "oracle_python"), ("A2", "oracle_python")):
        assert f'echo "{variable}_ORACLE_PYTHON=${value}"' in ss["Install pinned development oracle"]
    assert 'echo "Q0_PYTHON=$oracle_python"' in ss["Install pinned development oracle"]
    assert field(ej["evidence"], 4, "needs") == "suites"
    assert field(ej["evidence"], 4, "if") == "${{ always() }}"
    assert run(steps(ej["evidence"])["Require complete coverage and record tested tree"]) == [
        'test "$SUITES_RESULT" = success', "python3 tests/ci/evidence.py emit"]
    for body in (suite, ej["evidence"], cj["topology"], aj["select"]):
        assert f"      - uses: {CHECKOUT}" in body
    for body in (cj["topology"], cj["f0-linux"], aj["accepted"], ej["evidence"]):
        assert field(body, 4, "timeout-minutes") == "2"
    assert field(aj["select"], 4, "timeout-minutes") == "3"
    targets = recipes(make)
    assert targets["test-after-build"] == FULL
    assert targets["test-ci-clean-build-after-build"] == ["$(MAKE) smoke-after-build benchmark-after-build"]
    assert targets["test-acceptance-predecessors-after-build"] == [c for c in FULL if c != "/usr/bin/bash scripts/test-c2.sh"]
    assert targets["test-acceptance-c2-after-build"] == ["/usr/bin/bash scripts/test-c2.sh"]
    leaves = []
    for suite_id, target, _ in SUITES:
        assert re.search(rf"^{target}:\s*$", make, re.M), target
        if suite_id == "canonical-build":
            continue
        if suite_id.startswith("c2-"):
            assert targets[target] == [f"/usr/bin/bash scripts/test-c2.sh --group {suite_id}"]
        else:
            leaves += targets[target]
    assert Counter(leaves) == Counter(c for c in FULL if c != "/usr/bin/bash scripts/test-c2.sh")
    assert len(leaves) == 22
    canonical = ["/usr/bin/bash scripts/" + s for s in (
        "generate-p1-roots.sh --check", "build.sh", "build-a2.sh",
        "build-p1-identity.sh", "build-p1-package.sh", "build-c1.sh",
        "build-t1.sh", "build-t2.sh", "build-d2.sh", "build-c2.sh")]
    assert targets["build"] == canonical
    return len(SUITES)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    print(f"CI TOPOLOGY PASS: {check(root)} shared suites; full 23-command coverage, six C2 groups, clean build, strict evidence gates")
