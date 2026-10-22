"""Make the test-only E3 D2 restore-generation mutant for a compiled gate."""

import argparse
import hashlib
from pathlib import Path

SOURCE_SHA256 = "5f26e48410a9567f612fb5bab1fe982b6e95642d149d36f0f89278cc91d27217"
COMMIT_START = "(define (e3-d2-restore-commit! frame)"
COMMIT_END = "(define (e3-d2-cancel-prepare! frame)"
ANCHOR = "    (vector-set! core 15 (vector-ref record 7))\n"
MUTATION = "    (vector-set! state 8 (vector-ref record 13))\n"


def mutate(source: str) -> str:
    if hashlib.sha256(source.encode()).hexdigest() != SOURCE_SHA256:
        raise ValueError("E3 D2 restore source differs from the accepted N4 base")
    prefix, marker, remainder = source.partition(COMMIT_START)
    commit, end, suffix = remainder.partition(COMMIT_END)
    if not marker or not end or commit.count(ANCHOR) != 1:
        raise ValueError("E3 D2 restore commit anchor is not unique")
    return prefix + marker + commit.replace(ANCHOR, ANCHOR + MUTATION, 1) + end + suffix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        parser.error("mutant output must differ from production source")
    args.output.write_text(mutate(args.source.read_text()))


if __name__ == "__main__":
    main()
