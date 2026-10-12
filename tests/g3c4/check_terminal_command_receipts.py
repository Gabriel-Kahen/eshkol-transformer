#!/usr/bin/env python3
"""Reject incomplete or ambiguous terminal-runner command receipts."""

from pathlib import Path
import re
import stat
import sys

FILES = frozenset({"label", "cwd", "argv.nul", "exit", "stdout.raw", "stderr.raw"})
NAME = re.compile(r"(?:[0-9]{4}-[A-Za-z0-9_-]+|[A-Za-z0-9_-]+-clang-link\.[A-Za-z0-9]+)")
FIXTURES = ("", "feature-off", "parity-original", "parity-overlay", "unmarked",
            "decode-first", "decode-second", "precommit", "postzero", "lease-i1", "lease-a2")


def check_link_mapping(parent_labels: set[str], nested_prefixes: list[str], full: bool) -> None:
    counts = {prefix: nested_prefixes.count(prefix) for prefix in set(nested_prefixes)}
    if set(counts) - parent_labels:
        raise ValueError("orphan nested linker receipt")
    links = {label for label in parent_labels if label.endswith("-link")}
    if full:
        expected = {f"{mode}{('-' + fixture) if fixture else ''}-link"
                    for mode in ("normal", "sanitize") for fixture in FIXTURES}
        if links and links != expected:
            raise ValueError("fixture link receipt set differs")
        if not links:
            expected_compile = {label.removesuffix("-link") + "-compile" for label in expected}
            if not expected_compile <= parent_labels:
                raise ValueError("ordinary fixture compile receipt set differs")
            for label in expected_compile:
                if counts.get(label) != 1:
                    raise ValueError(f"missing/duplicate ordinary C++ link: {label}")
    for label in links:
        if counts.get(label) != 1:
            raise ValueError(f"missing/duplicate emitted C++ link: {label}")


def regular(path: Path) -> None:
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError(f"unsafe or missing receipt file: {path}")


def verify(evidence: Path, allow_current: bool = False, full: bool = False) -> int:
    pointer = evidence / "commands-path.txt"
    if pointer.exists():
        regular(pointer)
        value = pointer.read_text()
        commands = Path(value.rstrip("\n"))
        if value != f"{commands}\n" or commands.parent != evidence or \
                not commands.name.startswith("commands."):
            raise ValueError("command receipt pointer differs")
    else:
        commands = evidence / "commands"  # isolated source-only fixtures
    if not stat.S_ISDIR(commands.lstat().st_mode):
        raise ValueError("command receipt root is unsafe")
    receipts = sorted(commands.iterdir())
    if not receipts:
        raise ValueError("no command receipts")
    parent_numbers = []
    parent_labels = set()
    nested_prefixes = []
    current = 0
    for receipt in receipts:
        if not NAME.fullmatch(receipt.name) or not stat.S_ISDIR(receipt.lstat().st_mode):
            raise ValueError(f"unsafe receipt: {receipt}")
        files = {part.name for part in receipt.iterdir()}
        incomplete_current = allow_current and receipt.name.endswith("-receipts-verify")
        if incomplete_current:
            current += 1
            if files != FILES - {"exit"}:
                raise ValueError("current audit receipt shape differs")
        elif files != FILES:
            raise ValueError(f"receipt files differ: {receipt}")
        for name in files:
            regular(receipt / name)
        if not incomplete_current:
            exit_bytes = (receipt / "exit").read_bytes()
            if not re.fullmatch(rb"(?:0|[1-9][0-9]{0,2})\n", exit_bytes) or int(exit_bytes) > 255:
                raise ValueError(f"invalid numeric exit: {receipt}")
        argv = (receipt / "argv.nul").read_bytes()
        if not argv or not argv.endswith(b"\0") or b"\0\0" in argv:
            raise ValueError(f"invalid NUL argv: {receipt}")
        cwd = (receipt / "cwd").read_text()
        if not cwd.startswith("/") or not cwd.endswith("\n") or "\n" in cwd[:-1]:
            raise ValueError(f"invalid cwd: {receipt}")
        if receipt.name[0].isdigit():
            parent_numbers.append(int(receipt.name[:4]))
            parent_labels.add(receipt.name[5:])
        else:
            nested_prefixes.append(receipt.name.split("-clang-link.", 1)[0])
    if parent_numbers != list(range(1, len(parent_numbers) + 1)) or current > 1:
        raise ValueError("missing/duplicate parent receipt index or audit")
    check_link_mapping(parent_labels, nested_prefixes, full)
    return len(receipts)


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3) or (len(sys.argv) == 3 and sys.argv[2] not in ("--allow-current", "--full")):
        raise SystemExit("usage: check_terminal_command_receipts.py EVIDENCE [--allow-current|--full]")
    option = sys.argv[2] if len(sys.argv) == 3 else ""
    print(f"terminal command receipts PASS: {verify(Path(sys.argv[1]), option == '--allow-current', bool(option))}")
