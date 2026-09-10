#!/usr/bin/env python3
"""Decide whether a Git change requires the full CI suite.

The selector is intentionally fail-closed: only a nonempty, complete diff made up
solely of the small prose allowlist below may skip the full suite.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path
from typing import Sequence


MAX_DIFF_BYTES = 1024 * 1024
GIT_TIMEOUT_SECONDS = 30
ROOT_PROSE_PATHS = {"README.md", "CONTRIBUTING.md"}
COMMIT_HASH = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")


class SelectionError(Exception):
    """Raised when the selector cannot prove that the diff is prose-only."""


def _git(arguments: Sequence[str], repository: Path) -> bytes:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repository,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=GIT_TIMEOUT_SECONDS,
        )
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        raise SelectionError("git invocation failed") from error
    if completed.returncode != 0:
        raise SelectionError("git rejected the requested comparison")
    if len(completed.stdout) > MAX_DIFF_BYTES:
        raise SelectionError("git output exceeds the selector bound")
    return completed.stdout


def _resolve_commit(revision: str, repository: Path) -> str:
    if not revision:
        raise SelectionError("missing revision")
    output = _git(
        ["rev-parse", "--verify", "--end-of-options", f"{revision}^{{commit}}"],
        repository,
    )
    try:
        resolved = output.decode("ascii").strip()
    except UnicodeDecodeError as error:
        raise SelectionError("git returned a malformed object ID") from error
    if not COMMIT_HASH.fullmatch(resolved):
        raise SelectionError("git returned a malformed object ID")
    return resolved


def _changed_paths(base: str, head: str, repository: Path) -> tuple[str, ...]:
    base_hash = _resolve_commit(base, repository)
    head_hash = _resolve_commit(head, repository)
    output = _git(
        [
            "diff",
            "--name-only",
            "-z",
            "--no-renames",
            base_hash,
            head_hash,
            "--",
        ],
        repository,
    )
    if not output or not output.endswith(b"\0"):
        raise SelectionError("empty or incomplete changed-path list")

    encoded_paths = output[:-1].split(b"\0")
    if not encoded_paths or any(not path for path in encoded_paths):
        raise SelectionError("malformed changed-path list")
    try:
        return tuple(path.decode("utf-8") for path in encoded_paths)
    except UnicodeDecodeError as error:
        raise SelectionError("changed path is not UTF-8") from error


def _is_prose_path(path: str) -> bool:
    if path in ROOT_PROSE_PATHS:
        return True

    parts = path.split("/")
    return (
        len(parts) >= 2
        and parts[0] == "docs"
        and all(part not in {"", ".", ".."} for part in parts)
        and parts[-1] != "AGENTS.md"
        and parts[-1].endswith(".md")
    )


def requires_full_suite(base: str, head: str, repository: Path | None = None) -> bool:
    """Return false only when the complete Git diff is allowlisted prose."""
    root = Path.cwd() if repository is None else repository
    try:
        paths = _changed_paths(base, head, root)
    except SelectionError:
        return True
    return not all(_is_prose_path(path) for path in paths)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="")
    parser.add_argument("--head", default="")
    arguments = parser.parse_args(argv)
    print("true" if requires_full_suite(arguments.base, arguments.head) else "false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
