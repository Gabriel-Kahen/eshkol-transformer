"""Test-only external lock admission for the C4 native capture runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess

ROOT = Path(__file__).resolve().parents[2]
COMMON_SHA = "2749689e6301cad6e489de4b23e0f9b1ec7c9627f47dd1c7007d9ed8e5955e83"
TRACKED_LOCK_SHA = "8c168f5d560781245f6d4c73f695e72cbafa413b606d50bc7c36d2119acd9efe"
CANDIDATE_LOCK_SHA = "b798822b18535405092b201248d8e37682d5ddb9555b6080ee32131659bfbea5"
COMPILER_COMMIT = "6803dbe24fb65eaad2901767bf010787517946ef"
COMPILER_TREE = "539d22b1b6a89c290aa13c91263982332b01b87b"
OLD_ROW = b"eshkol_commit\tfe9dfd5241a1f4c4f58dee8442f44e4ff95e55b9\n"
NEW_ROW = b"eshkol_commit\t6803dbe24fb65eaad2901767bf010787517946ef\n"


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def digest_file(path: Path) -> str:
    state = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            state.update(block)
    return state.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "--no-optional-locks", "-C", str(root), *args],
                          check=True, capture_output=True, timeout=30).stdout.decode().strip()


def file_identity(path: Path) -> dict[str, int]:
    entry = path.lstat()
    return {"dev": entry.st_dev, "ino": entry.st_ino, "uid": entry.st_uid,
            "mode": entry.st_mode, "nlink": entry.st_nlink,
            "size": entry.st_size}


def ordinary_external_lock(path: Path, root: Path) -> tuple[bytes, dict]:
    require(path.is_absolute() and ".." not in path.parts and "." not in path.parts,
            "candidate lock needs a normalized absolute path")
    require(not path.is_relative_to(root.resolve()),
            "candidate lock must be outside the tracked C4 checkout")
    current = Path(path.anchor)
    for component in path.parts[1:-1]:
        current /= component
        entry = current.lstat()
        require(stat.S_ISDIR(entry.st_mode) and not stat.S_ISLNK(entry.st_mode),
                "candidate lock parent contains a symlink or non-directory")
    parent_before = file_identity(path.parent)
    leaf_before = path.lstat()
    require(stat.S_ISREG(leaf_before.st_mode) and leaf_before.st_nlink == 1,
            "candidate lock must be an ordinary unique file")
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        opened = os.fstat(descriptor)
        require(stat.S_ISREG(opened.st_mode) and opened.st_nlink == 1 and
                (opened.st_dev, opened.st_ino) ==
                (leaf_before.st_dev, leaf_before.st_ino) and opened.st_size == 477,
                "candidate lock changed during nofollow open")
        raw = os.read(descriptor, 478)
        require(len(raw) == 477 and os.read(descriptor, 1) == b"",
                "candidate lock length changed during read")
    finally:
        os.close(descriptor)
    require(parent_before == file_identity(path.parent) and
            file_identity(path) == file_identity_from_stat(opened),
            "candidate lock path or parent identity changed")
    return raw, {"path": str(path), "parent": str(path.parent),
                 "parent_identity": parent_before,
                 "file_identity": file_identity_from_stat(opened),
                 "sha256": digest(raw)}


def file_identity_from_stat(entry: os.stat_result) -> dict[str, int]:
    return {"dev": entry.st_dev, "ino": entry.st_ino, "uid": entry.st_uid,
            "mode": entry.st_mode, "nlink": entry.st_nlink,
            "size": entry.st_size}


def tsv_value(raw: bytes, key: str) -> str:
    rows = [line.split("\t", 1)[1] for line in raw.decode().splitlines()
            if line.startswith(key + "\t") and "\t" in line]
    require(len(rows) == 1 and bool(rows[0]), f"provenance {key} missing/duplicate")
    return rows[0]


def observe(candidate: Path, source: Path, build: Path) -> dict:
    require(os.environ.get("ESHKOL_ALLOW_UNSUPPORTED_HOST", "0") == "0",
            "unsupported-host bypass forbidden")
    require(source.is_absolute() and build.is_absolute() and
            source.is_dir() and build.is_dir(),
            "selected compiler source/build need absolute existing directories")
    require(git(ROOT, "status", "--porcelain=v1", "--untracked-files=all") == "",
            "C4 source checkout must be clean")
    common = (ROOT / "scripts/common.sh").read_bytes()
    tracked = (ROOT / "toolchain/eshkol.lock").read_bytes()
    require(digest(common) == COMMON_SHA and digest(tracked) == TRACKED_LOCK_SHA,
            "C4 common or production lock drift")
    require(tracked.count(OLD_ROW) == 1 and tracked.count(b"eshkol_commit\t") == 1,
            "production lock lacks unique fe9 row")
    expected = tracked.replace(OLD_ROW, NEW_ROW)
    require(len(expected) == 477 and digest(expected) == CANDIDATE_LOCK_SHA,
            "candidate lock derivation drift")
    raw, lock = ordinary_external_lock(candidate, ROOT)
    require(raw == expected, "candidate lock differs from sole fe9-to-6803 replacement")
    commit = git(source, "rev-parse", "HEAD")
    tree = git(source, "rev-parse", "HEAD^{tree}")
    require(commit == COMPILER_COMMIT and tree == COMPILER_TREE and
            git(source, "status", "--porcelain=v1", "--untracked-files=all") == "",
            "selected 6803 compiler source commit/tree/cleanliness mismatch")
    provenance = (build / "eshkol-transformer-provenance.tsv").read_bytes()
    cache = (build / "CMakeCache.txt").read_bytes()
    compiler = build / "eshkol-run"
    require(compiler.is_file() and os.access(compiler, os.X_OK),
            "selected Eshkol compiler executable missing")
    require(tsv_value(provenance, "eshkol_commit") == COMPILER_COMMIT and
            tsv_value(provenance, "eshkol_source_dir") == str(source.resolve()),
            "selected compiler provenance mismatch")
    cache_home = [line.split("=", 1)[1] for line in cache.decode().splitlines()
                  if line.startswith("CMAKE_HOME_DIRECTORY:INTERNAL=")]
    require(len(cache_home) == 1 and Path(cache_home[0]).resolve() == source.resolve(),
            "selected compiler CMake source mismatch")
    return {"mode": "c4-external-6803-test-lock", "candidate_lock": lock,
            "common_sha256": digest(common), "tracked_lock_sha256": digest(tracked),
            "c4_commit": git(ROOT, "rev-parse", "HEAD"),
            "c4_tree": git(ROOT, "rev-parse", "HEAD^{tree}"),
            "compiler_source": str(source), "compiler_commit": commit,
            "compiler_tree": tree, "compiler_build": str(build),
            "provenance_sha256": digest(provenance),
            "cmake_cache_sha256": digest(cache),
            "eshkol_run_sha256": digest_file(compiler)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("build", type=Path)
    args = parser.parse_args()
    print(json.dumps(observe(args.candidate, args.source, args.build),
                     sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
