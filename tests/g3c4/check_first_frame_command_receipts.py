#!/usr/bin/env python3
"""Audit the native first-frame leaf's ordered raw command receipts."""

from hashlib import sha256
from pathlib import Path
import re
import stat
import sys

FILES = ("label", "cwd", "argv.nul", "exit", "stdout.raw", "stderr.raw")
LABELS = (
    "toolchain-verify", "static",
    *(f"owner-{mode}-{step}" for mode in ("off", "on")
      for step in ("compile", "nm", "awk", "sort")),
    "symbol-parity", "invalid-compile", "invalid-diagnostic",
    "feature-off-compile", "feature-off-run", "feature-off-stderr-empty",
    "feature-off-pass",
    *(f"{mode}-{step}" for mode in ("normal", "sanitize")
      for step in ("compile", "run", "stderr-empty", "pass")),
    "oracle", "repeat-run", "repeat-stderr-empty",
    "repeat-parity", "sanitize-parity", "receipts-audit",
)
ALIASES = {
    "static": ("static.stdout", "static.stderr"),
    "owner-off-sort": ("off.symbols", None),
    "owner-on-sort": ("on.symbols", None),
    "invalid-compile": ("invalid.stdout", "invalid.stderr"),
    "feature-off-run": ("feature-off.stdout", "feature-off.stderr"),
    "normal-run": ("normal.stdout", "normal.stderr"),
    "sanitize-run": ("sanitize.stdout", "sanitize.stderr"),
    "oracle": ("oracle.stdout", "oracle.stderr"),
    "repeat-run": ("repeat.stdout", "repeat.stderr"),
    "receipts-audit": ("receipts-audit.stdout", "receipts-audit.stderr"),
}
MANIFEST = "SHA256SUMS-RECEIPTS"


def regular(path: Path) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise ValueError(f"missing receipt file: {path}") from exc
    if not stat.S_ISREG(mode):
        raise ValueError(f"missing or unsafe receipt file: {path}")


def receipt_root(evidence: Path) -> Path:
    pointer = evidence / "commands-path.txt"
    regular(pointer)
    value = pointer.read_text()
    root = Path(value.rstrip("\n"))
    if value != f"{root}\n" or root.parent != evidence or \
            not root.name.startswith("commands.") or root.is_symlink() or \
            not root.is_dir():
        raise ValueError("invalid command root pointer")
    return root


def expected_program(label: str) -> str:
    if label == "toolchain-verify":
        return "bash"
    if label in ("static", "oracle", "receipts-audit"):
        return "python3"
    if label.endswith("-compile"):
        return "clang-21"
    if label.endswith("-nm"):
        return "nm"
    if label.endswith("-awk"):
        return "awk"
    if label.endswith("-sort") or label in ("normal-run", "sanitize-run"):
        return "env"
    if label.endswith("-stderr-empty"):
        return "test"
    if label.endswith("-pass") or label == "invalid-diagnostic":
        return "grep"
    if label in ("symbol-parity", "repeat-parity", "sanitize-parity"):
        return "cmp"
    if label in ("feature-off-run", "repeat-run"):
        return label.removesuffix("-run").replace("repeat", "normal")
    raise ValueError(f"unexpected label: {label}")


def inspect(evidence: Path, seal_current: bool = False) -> int:
    root = receipt_root(evidence)
    manifest = root / MANIFEST
    children = sorted(path for path in root.iterdir() if path.name != MANIFEST)
    if len(children) != len(LABELS):
        raise ValueError("missing or extra command receipt")
    records = []
    cwd_seen = set()
    for number, (child, label) in enumerate(zip(children, LABELS), 1):
        if child.name != f"{number:04d}-{label}" or child.is_symlink() or \
                not child.is_dir():
            raise ValueError(f"receipt order or label differs: {child}")
        current = seal_current and label == "receipts-audit"
        expected_files = set(FILES) - ({"exit"} if current else set())
        if {part.name for part in child.iterdir()} != expected_files:
            raise ValueError(f"receipt files differ: {label}")
        for filename in expected_files:
            regular(child / filename)
        if (child / "label").read_bytes() != (label + "\n").encode():
            raise ValueError(f"receipt label differs: {label}")
        cwd_bytes = (child / "cwd").read_bytes()
        try:
            cwd = cwd_bytes.decode()
        except UnicodeDecodeError as exc:
            raise ValueError("invalid cwd encoding") from exc
        if not cwd.startswith("/") or not cwd.endswith("\n") or \
                "\n" in cwd[:-1] or Path(cwd[:-1]).resolve() != Path(cwd[:-1]):
            raise ValueError(f"nonphysical cwd: {label}")
        cwd_seen.add(cwd)
        argv = (child / "argv.nul").read_bytes()
        if not argv.endswith(b"\0") or b"\0\0" in argv or not argv:
            raise ValueError(f"malformed NUL argv: {label}")
        try:
            args = [piece.decode() for piece in argv[:-1].split(b"\0")]
        except UnicodeDecodeError as exc:
            raise ValueError(f"invalid argv encoding: {label}") from exc
        if Path(args[0]).name != expected_program(label):
            raise ValueError(f"wrong executable in argv: {label}")
        if not current:
            raw_exit = (child / "exit").read_bytes()
            if not re.fullmatch(rb"(?:0|[1-9][0-9]{0,2})\n", raw_exit) or \
                    int(raw_exit) > 255:
                raise ValueError(f"malformed numeric exit: {label}")
            status = int(raw_exit)
            if label == "invalid-compile":
                if not 1 <= status <= 123:
                    raise ValueError("invalid compile was not an ordinary rejection")
            elif status != 0:
                raise ValueError(f"unexpected command failure: {label}")
            aliases = ALIASES.get(label, (None, None))
            for name, stream in zip(aliases, ("stdout.raw", "stderr.raw")):
                if name is not None:
                    target = evidence / name
                    regular(target)
                    if not target.samefile(child / stream):
                        raise ValueError(f"raw alias differs: {label}/{name}")
            if label == "invalid-compile" and \
                    b"C4 P2/G2 first frame requires pending output and token forward" \
                    not in (child / "stderr.raw").read_bytes():
                raise ValueError("current invalid compiler diagnostic absent")
            if label != "receipts-audit":
                records.extend((child / filename, filename) for filename in FILES)
    if len(cwd_seen) != 1:
        raise ValueError("command cwd changed")
    lines = "".join(
        f"{sha256(path.read_bytes()).hexdigest()}  {path.parent.name}/{name}\n"
        for path, name in records
    )
    if seal_current:
        if manifest.exists():
            raise ValueError("receipt seal already exists")
        with manifest.open("x") as output:
            output.write(lines)
    else:
        regular(manifest)
        if manifest.read_text() != lines:
            raise ValueError("receipt raw or metadata changed")
    return len(children)


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3) or (len(sys.argv) == 3 and sys.argv[2] != "--seal-current"):
        raise SystemExit("usage: check_first_frame_command_receipts.py EVIDENCE [--seal-current]")
    print(f"first-frame command receipts PASS: {inspect(Path(sys.argv[1]), len(sys.argv) == 3)}")
