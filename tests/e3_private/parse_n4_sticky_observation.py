"""Check test-only fenv status around genuine sparse private K1 dispatch."""

from __future__ import annotations

import argparse
from pathlib import Path

from tests.e3_private.parse_n4_sparse_observation import parse as parse_sparse

MODES = {"clear": 0, "seed": 1, "clear-before": 2, "clear-after": 3}


def parse(raw: bytes, gradient: str, mode: str) -> None:
    if mode not in MODES or not raw.endswith(b"\n") or len(raw) > 262144:
        raise ValueError("N4 sticky transcript or mode is invalid")
    lines = raw.splitlines(keepends=True)
    if len(lines) < 2 or not lines[0].startswith(b"E3N4K1\t"):
        raise ValueError("N4 sticky K1 record is missing")
    try:
        fields = lines[1].decode("ascii").rstrip("\n").split("\t")
        if len(fields) != 8 or fields[0] != "E3N4FENV" or \
                fields[1] != str(MODES[mode]):
            raise ValueError("N4 sticky status record or mode differs")
        widths = (4, 4, 4, 4, 8, 8)
        for field, width in zip(fields[2:], widths, strict=True):
            if len(field) != width or any(ch not in "0123456789abcdef" for ch in field):
                raise ValueError("N4 sticky status field is malformed")
        cw_before, cw_after, sw_before, sw_after, mx_before, mx_after = \
            (int(field, 16) for field in fields[2:])
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError("N4 sticky status record is malformed") from error
    if (cw_before != cw_after or (cw_before & 0x003f) != 0x003f or
            (cw_before & 0x0c00) != 0 or
            (mx_before & ~0x003f) != (mx_after & ~0x003f) or
            (mx_before & 0x1f80) != 0x1f80 or (mx_before & 0xe040) != 0):
        raise ValueError("N4 sticky controls changed or are unsupported")
    if mode == "clear":
        if (sw_before & 0x003f) or (mx_before & 0x003f):
            raise ValueError("N4 clear control entered K1 with sticky flags")
    elif ((sw_before & 0x0004) != 0x0004 or
          (mx_before & 0x0021) != 0x0021 or
          (sw_after & 0x0004) != 0x0004 or
          (mx_after & 0x0021) != 0x0021):
        raise ValueError("N4 seeded sticky flags were absent or cleared")
    # The existing parser still checks all 512 original K1 words, every M3
    # role, CE, batch/state, exact D2 row, and the Eshkol result/footer.
    parse_sparse(b"".join(lines[:1] + lines[2:]), gradient)


def main() -> None:
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--stdout", required=True, type=Path)
    args.add_argument("--gradient", required=True, choices=("absent", "present"))
    args.add_argument("--mode", required=True, choices=tuple(MODES))
    parsed = args.parse_args()
    parse(parsed.stdout.read_bytes(), parsed.gradient, parsed.mode)
    print("verified_sparse_m3_k1_sticky=1")


if __name__ == "__main__":
    main()
