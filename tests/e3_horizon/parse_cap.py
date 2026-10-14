"""Strictly admit the public 8,193-row preflight runtime record."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

PASS = re.compile(
    r"E3-CAP-PASS rows=8193 denials=2 cursor=unchanged "
    r"first-shard=unread arena-delta=([0-9]+)\n"
)
TIME = re.compile(r"elapsed_seconds=([0-9]+(?:\.[0-9]+)?) peak_rss_kib=([0-9]+)\n")


def verify(stdout: str, stderr: str, timing: str, first_shard: Path) -> None:
    if PASS.fullmatch(stdout) is None or stderr:
        raise ValueError("cap runtime transcript differs or stderr is nonempty")
    match = TIME.fullmatch(timing)
    if match is None or float(match[1]) > 120 or not (0 < int(match[2]) <= 524288):
        raise ValueError("cap runtime measurement differs or exceeds the limit")
    if not first_shard.is_file() or first_shard.with_name(first_shard.name + ".held").exists():
        raise ValueError("first shard was not restored")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stdout", type=Path, required=True)
    parser.add_argument("--stderr", type=Path, required=True)
    parser.add_argument("--time", type=Path, required=True)
    parser.add_argument("--first-shard", type=Path, required=True)
    args = parser.parse_args()
    verify(args.stdout.read_text(), args.stderr.read_text(), args.time.read_text(), args.first_shard)


if __name__ == "__main__":
    main()
