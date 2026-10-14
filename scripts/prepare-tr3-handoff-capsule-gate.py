#!/usr/bin/env python3
"""Validate and render a held, unadmitted private capsule runtime candidate."""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tests/tr3_handoff_capsule/current_tree_candidate.tsv"
HASH_PATHS = {
    "capsule_root_sha256": "native/tr3_c_handoff_capsule_root.esk",
    "capsule_source_sha256": "native/tr3_c_handoff_capsule_extension.esk",
    "lease_core_sha256": "native/tr3_lease_core_extension.esk",
    "snapshot_root_sha256": "native/tr3_c_snapshot_root.esk",
    "snapshot_source_sha256": "native/tr3_c_snapshot_extension.esk",
    "fixture_sha256": "tests/tr3_handoff_capsule/runtime_candidate.esk",
    "inherited_gate_sha256": "scripts/test-tr3-c-joint-runtime.sh",
    "recipe_sha256": "scripts/prepare-tr3-handoff-capsule-gate.py",
    "allocator_shim_sha256": "tests/tr3_lease_failure/lease_failure_shim.cpp",
    "historical_functional_manifest_sha256": "tests/tr3_lease/runtime_candidate.tsv",
    "historical_failure_manifest_sha256": "tests/tr3_lease/failure_runtime_candidate.tsv",
}
REQUIRED = set(HASH_PATHS) | {
    "format_version", "admission", "source_basis_commit", "source_basis_tree",
    "runner_relative_path", "runner_sha256", "compiler_source_commit",
    "container_image", "container_digest", "allocation_classes",
}

def parse_manifest(source: str) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in source.splitlines():
        parts = line.split("\t")
        if len(parts) != 2 or not parts[0] or not parts[1] or parts[0] in rows:
            raise ValueError("candidate manifest has malformed or duplicate row")
        rows[parts[0]] = parts[1]
    if set(rows) != REQUIRED:
        raise ValueError(f"candidate keys differ: {sorted(set(rows) ^ REQUIRED)}")
    if rows["format_version"] != "1" or rows["admission"] != "unadmitted_source_recipe":
        raise ValueError("candidate admission marker changed")
    if rows["allocation_classes"] != "object,all,promotion,target,handler":
        raise ValueError("real allocator failure matrix is not fully declared")
    historical = dict(line.split("\t", 1) for line in
                      (ROOT / "tests/tr3_lease/runtime_candidate.tsv").read_text().splitlines())
    for key in ("runner_relative_path", "runner_sha256", "compiler_source_commit",
                "container_image", "container_digest"):
        if rows[key] != historical[key]:
            raise ValueError(f"inherited runner provenance differs: {key}")
    if not re.fullmatch(r"[0-9a-f]{40}", rows["source_basis_tree"]):
        raise ValueError("invalid source basis tree")
    for key in HASH_PATHS:
        if not re.fullmatch(r"[0-9a-f]{64}", rows[key]):
            raise ValueError(f"invalid digest: {key}")
        actual = hashlib.sha256((ROOT / HASH_PATHS[key]).read_bytes()).hexdigest()
        if actual != rows[key]:
            raise ValueError(f"source digest differs: {HASH_PATHS[key]}")
    if not re.fullmatch(r"[0-9a-f]{40}", rows["source_basis_commit"]):
        raise ValueError("invalid source basis")
    result = subprocess.run(
        ["git", "rev-parse", f"{rows['source_basis_commit']}^{{tree}}"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    if result.stdout.strip() != rows["source_basis_tree"]:
        raise ValueError("source basis tree differs")
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

def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise ValueError(f"inherited recipe marker count changed: {old}")
    return source.replace(old, new)

def render(rows: dict[str, str]) -> str:
    gate = (ROOT / "scripts/test-tr3-c-joint-runtime.sh").read_text()
    gate = replace_once(
        gate,
        'source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"',
        f'source "{ROOT}/scripts/common.sh"',
    )
    gate = replace_once(
        gate,
        'candidate_manifest="${PROJECT_ROOT}/tests/tr3_lease/runtime_candidate.tsv"',
        'candidate_manifest="${PROJECT_ROOT}/tests/tr3_handoff_capsule/current_tree_candidate.tsv"',
    )
    gate = replace_once(
        gate, 'cd "${PROJECT_ROOT}"',
        'cd "${PROJECT_ROOT}"\nPYTHONDONTWRITEBYTECODE=1 python3 scripts/prepare-tr3-handoff-capsule-gate.py --verify-only',
    )
    substitutions = {
        "TR3_JOINT_EVIDENCE_DIR": "TR3_HANDOFF_CANDIDATE_EVIDENCE_DIR",
        "tests/tr3_joint_restore/runtime_smoke.esk":
            "tests/tr3_handoff_capsule/runtime_candidate.esk",
        "tr3_joint_runtime": "tr3_handoff_candidate_runtime",
        "joint-runtime": "handoff-candidate-runtime",
        "TR3-C JOINT RUNTIME PASS: [0-9]+ checks, 42 tensors, abort/retry":
            "TR3-HANDOFF-CANDIDATE-PASS checks=[0-9]+",
    }
    for old, new in substitutions.items():
        if old not in gate:
            raise ValueError(f"missing inherited recipe marker: {old}")
        gate = gate.replace(old, new)
    return gate

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.verify_only == (args.output is not None):
        parser.error("choose exactly one of --verify-only or --output")
    rows = parse_manifest(MANIFEST.read_text())
    if args.output is not None:
        args.output.write_text(render(rows))
        args.output.chmod(0o755)
    else:
        print("TR3 handoff source manifest PASS; compiled gate remains unadmitted")

if __name__ == "__main__":
    main()
