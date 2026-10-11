#!/usr/bin/env python3
"""Source-only preflight for the two private one-root union fixtures.

This command has no compiler, loader, Docker, or native launch path. A later
launcher needs a separately reviewed full pin table and resource decision.
"""

from __future__ import annotations

import os
from pathlib import Path
import stat
import subprocess
import sys


ROOT = Path(__file__).resolve().parent.parent
BASE = "4ab9b5c0cb3ff98aaa2c69298cd17bb8495a1a5b"
PACKAGE = "native/g3g_manual_package"
HISTORY = "tests/g3c4/p2g2_public_history_private_runtime.esk"
T1 = "tests/g3g_manual/private_t1_retirement_runtime.esk"
COMPANION = "tests/t1/test_sealed_retire.c"
INCLUDES = ("internal/p1/lib", "internal/c1/lib", "internal/t1/lib", "src")
TUPLE = tuple(PACKAGE + suffix for suffix in
              ("_root.esk", "_bridge.c", "_private_renames.txt",
               "_public_exports.txt"))
MANIFESTS = tuple(PACKAGE + suffix for suffix in
                  ("_source_closure.txt", "_native_source_closure.txt",
                   "_native_objects.txt", "_archive_members.txt"))
FIXED = TUPLE + MANIFESTS + (HISTORY, T1, COMPANION,
                            "native/t1_i64_shell.c", "native/t1_i64_shell.h",
                            "native/i64_tensor.c",
                            "include/eshkol_transformer/i64_tensor.h",
                            "scripts/build-g3g-manual.sh",
                            "scripts/build-e1b-consumer.sh",
                            "scripts/g3g-manual-package-policy.sh",
                            "scripts/g3g-manual-native-inputs.sh")
PROVIDERS = (
    "n2/n2_primitives_provider.o", "n3k/n3k_primitives_provider.o",
    "a2/a2_attention_provider.o", "g3n/g3n_primitives_provider.o",
    "g3s/g3s_sampling_provider.o", "g3c4/g3c4_primitives_provider.o",
)
T1_TEST_FLAGS = (
    "ET_T1_SEALED_RELEASE_TEST_ONLY", "ET_T1_I64_SHELL_TESTING",
    "ET_G3C4_T1_I1_EXACT_PAIR_PRIVATE",
    "ET_I64_TENSOR_STORAGE_QUERY_PRIVATE",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def git(root: Path, *args: str) -> bytes:
    return subprocess.run(("git", "--no-optional-locks", "-C", str(root), *args),
                          check=True, capture_output=True, timeout=30).stdout


def ordinary(root: Path, relative: str) -> bytes:
    require(relative and not relative.startswith("/") and
            all(part not in ("", ".", "..") for part in relative.split("/")),
            f"malformed source path: {relative}")
    path = root
    for part in relative.split("/")[:-1]:
        path = path / part
        mode = path.lstat().st_mode
        require(stat.S_ISDIR(mode) and not stat.S_ISLNK(mode),
                f"nonordinary source parent: {relative}")
    path = path / relative.split("/")[-1]
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1,
            f"nonordinary source: {relative}")
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        opened = os.fstat(fd)
        require((opened.st_dev, opened.st_ino, opened.st_mode, opened.st_size) ==
                (before.st_dev, before.st_ino, before.st_mode, before.st_size),
                f"source changed before read: {relative}")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            data = stream.read()
        after = path.lstat()
        require((after.st_dev, after.st_ino, after.st_mode, after.st_size,
                 after.st_mtime_ns, after.st_ctime_ns) ==
                (before.st_dev, before.st_ino, before.st_mode, before.st_size,
                 before.st_mtime_ns, before.st_ctime_ns),
                f"source changed during read: {relative}")
        return data
    finally:
        os.close(fd)


def closure(raw: bytes) -> tuple[str, ...]:
    names = tuple(raw.decode("utf-8").splitlines())
    require(names and len(names) == len(set(names)) and
            all(name and not name.startswith("/") and
                all(part not in ("", ".", "..")
                    for part in name.split("/")) for name in names),
            "malformed or duplicate package closure entry")
    return names


def fixed_manifest(raw: bytes, expected: tuple[str, ...]) -> None:
    require(closure(raw) == expected, "fixed package manifest drifted")


def one_root(raw: bytes, name: str) -> None:
    # These fixtures are fixed source inputs, so exact lexical load is required.
    loads = tuple(line.strip() for line in raw.decode("utf-8").splitlines()
                  if line.lstrip().startswith("(load "))
    require(loads == ('(load "g3g_manual_package_root.esk")',),
            f"{name} must load the fixed package root exactly once")
    require(b"libeshkol_transformer_g3g_manual" not in raw and
            b"(require transformer." not in raw,
            f"{name} mixes an installed package with its private root")


def source_topology(root: Path) -> None:
    require(git(root, "merge-base", "--is-ancestor", BASE, "HEAD") == b"",
            "reviewed source base is not an ancestor")
    require(git(root, "status", "--porcelain=v1", "--untracked-files=all") == b"",
            "source checkout is dirty")
    # Compare the reviewed source union bytes, including modes through Git.
    names = set(FIXED)
    for manifest in MANIFESTS[:2]:
        # The reviewed closure also names generated Eshkol headers under
        # .deps; those require a later actual-header pin, not a Git blob.
        names.update(name for name in closure(ordinary(root, manifest))
                     if not name.startswith(".deps/"))
    require(set(TUPLE).issubset(names), "fixed package tuple omitted")
    fixed_manifest(ordinary(root, MANIFESTS[2]), PROVIDERS)
    fixed_manifest(ordinary(root, MANIFESTS[3]),
                   ("g3g_manual_package.o",))
    for name in sorted(names):
        actual = ordinary(root, name)
        expected = git(root, "show", f"{BASE}:{name}")
        require(actual == expected, f"reviewed source bytes drifted: {name}")
        mode = git(root, "ls-tree", BASE, "--", name).split(None, 1)
        require(mode and mode[0] in (b"100644", b"100755") and
                stat.S_IMODE((root / name).stat().st_mode) ==
                int(mode[0][-3:], 8),
                f"reviewed source mode drifted: {name}")
    for name in (HISTORY, T1):
        one_root(ordinary(root, name), name)
    build = ordinary(root, "scripts/build-g3g-manual.sh").decode()
    for name in TUPLE:
        require(Path(name).name in build,
                f"installed package build omitted {name}")
    for name in INCLUDES:
        require(f'"${{PROJECT_ROOT}}/{name}"' in build,
                f"ordered trusted include missing: {name}")
    policy = ordinary(root, "scripts/g3g-manual-package-policy.sh").decode()
    require("G3-G manual requires the exact repository tuple" in policy and
            "G3-G manual requires exact ordered trusted include roots" in policy,
            "fixed package input policy absent")
    companion = ordinary(root, COMPANION).decode()
    require(all(f"defined({flag})" in companion for flag in T1_TEST_FLAGS),
            "T1 native companion omitted a test-only gate")
    production = ordinary(root, "scripts/build-e1b-consumer.sh").decode()
    require(all(flag not in production for flag in T1_TEST_FLAGS[:2]),
            "release-only T1 test macro escaped into public package recipe")


def main() -> int:
    try:
        source_topology(ROOT)
    except (OSError, ValueError, subprocess.CalledProcessError,
            subprocess.TimeoutExpired, UnicodeDecodeError) as exc:
        print(f"HOLD: source preflight failed: {exc}", file=sys.stderr)
        return 2
    print("HOLD: source topology verified; no admitted full external seals, "
          "selected tool/header/provider/loader pins or package resource bounds. "
          "No compiler, loader, Docker or native child started.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
