#!/usr/bin/env python3
"""Verify the accepted J capture before reusing its retained IR."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

SOURCE_HEAD = "fb31fac32530e0ff9b00ed3458e9080eaba9643a"
SOURCE_TREE = "5ca77e78a6f798f677a0b23a14e763d957f830bd"
RUNTIME_HEAD = "fe9dfd5241a1f4c4f58dee8442f44e4ff95e55b9"
ROOT_SEAL_SHA256 = "b09b7992e6be107eed9f531def85dc1e65195396f6090cebbe36b124cd9ed29e"
ROOT_REVIEW_SHA256 = "c5fc1d4c48ca165d0983c399b3175fc8fa0500fab7b17fc642385c8189ee96fc"
RUN_MANIFEST_SHA256 = "a5240c3e46a542ad067aa628ff43f0cd56636a3c0037f79c8a39c7f874947cc9"
SOURCE_CLOSURE_SHA256 = "a968591c58b8804d8e20c4b438330e2bb21d23d05cc0e944c23513db7d330733"
IR_SHA256 = "ae7c7aa16610b62d7e242bfde76a7cec9bec1e9e75f497bd932a39e2f7f0ce58"
LIB_SHA256 = "95fd238d9c8c073df0be98e040e331b3f0b2f1b661035cac9f8aa53c435ef434"
RUNNER_SHA256 = "7dd254bab761fe41142a0e3777338f41b3b9f03a5ce4c0bba419f1e2b22a99aa"
PROVENANCE_SHA256 = "4b2e87315b91dd27f6453862a4c4ebcd9fa1f577619d3bfa6ceca92964a9824c"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def reject_symlink_ancestors(path: Path) -> None:
    current = Path(path.anchor)
    for part in path.absolute().parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"symlinked J input path: {path}")


def require_hash(path: Path, expected: str) -> None:
    reject_symlink_ancestors(path)
    if not path.is_file() or digest(path) != expected:
        raise ValueError(f"hash or file type differs: {path}")


def manifest_rows(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in path.read_text().splitlines():
        pieces = line.split("  ", 1)
        if len(pieces) != 2 or len(pieces[0]) != 64 or any(
            ch not in "0123456789abcdef" for ch in pieces[0]
        ):
            raise ValueError("malformed run manifest")
        name = pieces[1]
        parts = name[2:].split("/")
        if not name.startswith("./") or any(
            part in ("", "..", ".") for part in parts
        ) or name in rows:
            raise ValueError("unsafe or duplicate run manifest path")
        rows[name] = pieces[0]
    return rows


def check_pins(source: bytes, runtime: bytes, closure: bytes) -> None:
    if source != f"{SOURCE_HEAD}\n{SOURCE_TREE}\n".encode():
        raise ValueError("J source pin differs")
    if runtime != f"{RUNTIME_HEAD}\n".encode():
        raise ValueError("J runtime pin differs")
    if hashlib.sha256(closure).hexdigest() != SOURCE_CLOSURE_SHA256:
        raise ValueError("J source closure differs")


def verify_run_manifest(run: Path, manifest: Path) -> None:
    rows = manifest_rows(manifest)
    required = {
        "./source-once/compiler/private.ll": IR_SHA256,
        "./source-once/compiler/libprivate.so": LIB_SHA256,
        "./source-once/observed-source-closure.txt": SOURCE_CLOSURE_SHA256,
    }
    if any(rows.get(name) != expected for name, expected in required.items()):
        raise ValueError("J manifest omits exact retained inputs")
    if not rows:
        raise ValueError("empty J run manifest")
    for name, expected in rows.items():
        target = run / name[2:]
        if not target.resolve().is_relative_to(run.resolve()):
            raise ValueError("J manifest escaped run directory")
        require_hash(target, expected)


def verify(run: Path, root_review: Path) -> None:
    reject_symlink_ancestors(run)
    reject_symlink_ancestors(root_review)
    if not run.is_dir() or not root_review.is_dir():
        raise ValueError("J evidence directory is absent or symlinked")
    seal = root_review / "SHA256SUMS-ROOT"
    review = root_review / "ROOT-REVIEW.tsv"
    require_hash(seal, ROOT_SEAL_SHA256)
    if seal.read_text() != f"{ROOT_REVIEW_SHA256}  ROOT-REVIEW.tsv\n":
        raise ValueError("J ROOT seal does not identify review")
    require_hash(review, ROOT_REVIEW_SHA256)
    fields = dict(line.split("\t", 1) for line in review.read_text().splitlines())
    for key, expected in (
        ("source_head", SOURCE_HEAD), ("source_tree", SOURCE_TREE),
        ("run_manifest_sha256", RUN_MANIFEST_SHA256),
    ):
        if fields.get(key) != expected:
            raise ValueError(f"J ROOT review {key} differs")
    if fields.get("decision") != "ACCEPT_COMPILED_SOURCE_CAPTURE_ONLY_NO_SECTION9_ACCEPTANCE":
        raise ValueError("J ROOT decision differs")
    manifest = run.parent / f"{run.name}.claim" / "attempt" / "RUN-SHA256SUMS"
    require_hash(manifest, RUN_MANIFEST_SHA256)
    verify_run_manifest(run, manifest)
    source_once = run / "source-once"
    check_pins((source_once / "source-pin.txt").read_bytes(),
               (source_once / "runtime-pin.txt").read_bytes(),
               (source_once / "observed-source-closure.txt").read_bytes())
    if (source_once / "capture.exit").read_bytes() != b"0\n":
        raise ValueError("J compiler capture did not exit zero")
    toolchain = (source_once / "toolchain.sha256").read_text().splitlines()
    expected = (RUNNER_SHA256, PROVENANCE_SHA256)
    if len(toolchain) != 2 or any(not row.startswith(hash_ + "  ")
                                  for row, hash_ in zip(toolchain, expected)):
        raise ValueError("J toolchain pins differ")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("root_review", type=Path)
    args = parser.parse_args()
    try:
        verify(args.run, args.root_review)
    except (OSError, ValueError) as exc:
        print(f"J input rejected: {exc}", file=sys.stderr)
        return 1
    print("accepted J inputs verified; source closure only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
