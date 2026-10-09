"""Exact source-only guard for the private split C4 native/Q0 evidence route."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "native/g3c4_c4_native_capture_source_closure.txt"
CLOSURE = ROOT / "native/g3c4_c4_split_capture_source_closure.txt"
SOURCE = ROOT / "tests/g3c4/c4_split_capture.py"
BASE_SHA = "6ed741bb2ffc6073a6fbcd14ec9c865d5156c0898bd54dcfcbda903fc8159c9c"
ADDITIONS = (
    "tests/g3c4/c4_split_capture.py",
    "tests/g3c4/check_c4_split_capture.py",
    "tests/g3c4/test_c4_split_capture.py",
    "native/g3c4_c4_split_capture_source_closure.txt",
    "docs/g3/G3_C4_SPLIT_NATIVE_CAPTURE_SOURCE_LEAF.md",
)
EXPECTED_SOURCES = (
    "tests/g3c4/c4_native_capture.c",
    "src/eshkol_transformer/m3_i64_integration.c",
    "native/t1_i64_shell.c",
    "native/n3k_primitives_provider.c",
    "native/g3c4_primitives_provider.c",
    "native/g3s_sampling_provider.c",
    "native/g3n_primitives_provider.c",
    "native/n2_primitives_provider.c",
    "native/a2_kv_cache.c",
)


def check(source: str, base: list[str], closure: list[str]) -> None:
    ast.parse(source, feature_version=(3, 10))
    assert len(base) == 212 and closure == base + list(ADDITIONS)
    assert len(closure) == 217 and len(set(closure)) == 217
    assert all((ROOT / member).is_file() for member in closure)
    required = (
        'BASE_COMMIT = "6b18976d1f3f900a333c9e99122a1a603664b37d"',
        'BASE_TREE = "2d238af456dc5657a21806fcadda12dfae637cdf"',
        "TOOL_PINS_SHA", "RUNTIME_PINS_SHA", "LOCK_SHA",
        'git("status", "--porcelain=v1", "--untracked-files=all")',
        "source_state(expected_head, expected_tree)",
        "guest_state(fixture, lock, tools, runtimes,",
        "before == after", "GUEST-INVENTORY.json", "verify_closed_tree",
        "transcript.run_native", "transcript.parse(raw, frozen)",
        "streams[0] == streams[1] == streams[2]",
        "raws[0] == raws[1] == raws[2]", "oracle.check_source_pins()",
        "oracle.compare_observation", "q0_before == q0_state()",
        '"-fsanitize=address,undefined"',
        '"detect_leaks=1:halt_on_error=1"',
        '"-Wl,--wrap=et_a2_kv_cache_transaction_abort_v1"',
        '"/usr/bin/clang-21"', '"/usr/bin/python3.10"',
        '"wait_status"] == 0', '"quiescent"] is True',
        '"stderr_bytes"] == len(stderr) == 0', '"overflow"] == ""',
        "file_digest(fixture) == FIXTURE_SHA", "--source-head",
        "--source-tree",
    )
    for phrase in required:
        assert phrase in source, phrase
    tree = ast.parse(source)
    source_assignment = next(
        statement for statement in tree.body
        if isinstance(statement, ast.Assign) and
        any(isinstance(target, ast.Name) and target.id == "SOURCES"
            for target in statement.targets))
    assert ast.literal_eval(source_assignment.value) == EXPECTED_SOURCES
    imports = {alias.name.split(".")[0] for statement in tree.body
               if isinstance(statement, ast.Import)
               for alias in statement.names}
    assert "torch" not in imports
    assert "docker" not in source.lower()
    assert '"eshkol-run"' not in source
    assert source.count("transcript.run_native(") == 2


def main() -> None:
    assert hashlib.sha256(BASE.read_bytes()).hexdigest() == BASE_SHA
    check(SOURCE.read_text(), BASE.read_text().splitlines(),
          CLOSURE.read_text().splitlines())
    print("C4 split native capture source/closure PASS")


if __name__ == "__main__":
    main()
