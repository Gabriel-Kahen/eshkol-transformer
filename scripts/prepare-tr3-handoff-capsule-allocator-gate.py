#!/usr/bin/env python3
"""Source-only preflight for the distinct, unadmitted capsule allocator matrix."""
from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tests/tr3_handoff_capsule/allocator_prefix_candidate.tsv"
HASH_PATHS = {
    "capsule_root_sha256": "native/tr3_c_handoff_capsule_root.esk",
    "capsule_source_sha256": "native/tr3_c_handoff_capsule_extension.esk",
    "lease_core_sha256": "native/tr3_lease_core_extension.esk",
    "fixture_sha256": "tests/tr3_handoff_capsule/allocator_prefix_runtime.esk",
    "shim_sha256": "tests/tr3_handoff_capsule/capsule_allocator_shim.cpp",
    "gate_sha256": "scripts/test-tr3-handoff-capsule-allocator-gate.sh",
    "historical_failure_manifest_sha256":
        "tests/tr3_lease/failure_runtime_candidate.tsv",
}
RUNTIME_KEYS = (
    "runtime_source_commit", "runtime_source_tree",
    "runner_relative_path", "runner_sha256", "runner_role",
    "runtime_archive_relative_path", "runtime_archive_sha256",
    "container_image", "container_digest",
)
REQUIRED = set(HASH_PATHS) | set(RUNTIME_KEYS) | {
    "format_version", "admission", "source_basis_commit", "source_basis_tree",
    "allocation_classes",
}


def parse_manifest(source: str) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in source.splitlines():
        parts = line.split("\t")
        if len(parts) != 2 or not parts[0] or not parts[1] or parts[0] in rows:
            raise ValueError("malformed or duplicate capsule allocator manifest row")
        rows[parts[0]] = parts[1]
    if set(rows) != REQUIRED:
        raise ValueError(f"capsule allocator manifest keys differ: {sorted(set(rows) ^ REQUIRED)}")
    if rows["format_version"] != "1" or rows["admission"] != "unadmitted_source_recipe":
        raise ValueError("capsule allocator admission marker changed")
    if rows["allocation_classes"] != "object,all,promotion,target,handler":
        raise ValueError("five-class allocator matrix declaration changed")
    if rows["source_basis_commit"] != "3ea8f80ba330d5574975cfb4e5ca4187db094b9b":
        raise ValueError("accepted source basis changed")
    if rows["source_basis_tree"] != "cc19a332d5e8c324f507a1b6039e261b0808b990":
        raise ValueError("accepted source tree changed")
    historical = dict(
        line.split("\t", 1) for line in
        (ROOT / "tests/tr3_lease/failure_runtime_candidate.tsv").read_text().splitlines()
    )
    for key in RUNTIME_KEYS:
        if rows[key] != historical[key]:
            raise ValueError(f"repaired allocator runtime provenance differs: {key}")
    for key, path in HASH_PATHS.items():
        actual = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        if rows[key] != actual:
            raise ValueError(f"source digest differs: {path}")
    # The separate allocator runtime is a descendant of the accepted
    # functional runner source. Its reviewed delta is confined to generated
    # allocator hardening and tests; no language or C header ABI changed.
    runtime_source = Path(
        "/home/gabe/.codex/worktrees/wave3-checked-arena-calls/eshkol"
    )
    if subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=runtime_source, check=True,
        capture_output=True, text=True,
    ).stdout.strip() != rows["runtime_source_commit"]:
        raise ValueError("repaired runtime source checkout differs")
    if subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"], cwd=runtime_source, check=True,
        capture_output=True, text=True,
    ).stdout.strip() != rows["runtime_source_tree"]:
        raise ValueError("repaired runtime source tree differs")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor",
         "81298b4a9608fb92eb6f351a2eabd8392da7d9ef",
         rows["runtime_source_commit"]],
        cwd=runtime_source, check=True,
    )
    changed = set(subprocess.run(
        ["git", "diff", "--name-only",
         "81298b4a9608fb92eb6f351a2eabd8392da7d9ef",
         rows["runtime_source_commit"]],
        cwd=runtime_source, check=True, capture_output=True, text=True,
    ).stdout.splitlines())
    allowed = {
        "cmake/CheckedPromotionTests.cmake",
        "lib/backend/codegen_context.cpp",
        "lib/backend/complex_codegen.cpp",
        "lib/backend/function_codegen.cpp",
        "lib/backend/llvm_codegen.cpp",
        "lib/core/runtime_closure_alloc.cpp",
        "tests/core/check_constructor_emergency_ir.py",
        "tests/core/constructor_emergency_jit_test.cpp",
        "tests/core/constructor_emergency_shim.cpp",
        "tests/core/constructor_emergency_test.esk",
        "tests/core/runtime_closure_alloc_test.cpp",
    }
    if changed != allowed:
        raise ValueError("repaired runtime delta exceeds reviewed allocator paths")
    for key in ("source_basis_commit", "source_basis_tree",
                "runtime_source_commit", "runtime_source_tree"):
        if not re.fullmatch(r"[0-9a-f]{40}", rows[key]):
            raise ValueError(f"invalid git identity: {key}")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", rows["source_basis_commit"], "HEAD"],
        cwd=ROOT, check=True,
    )
    subprocess.run(
        ["git", "diff", "--quiet", rows["source_basis_commit"], "--",
         "include", "internal", "lib", "native", "src", "templates"],
        cwd=ROOT, check=True,
    )
    return rows


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true", required=True)
    parser.parse_args()
    parse_manifest(MANIFEST.read_text())
    print("TR3 capsule allocator source recipe PASS; compiled matrix remains unadmitted")
