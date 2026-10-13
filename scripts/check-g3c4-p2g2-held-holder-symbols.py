#!/usr/bin/env python3
"""Check exact global holder definitions in one feature-on/off owner object."""

from pathlib import Path
import sys


PREFIX = "et_g3c4_private_p2g2_holder_"
EXPECTED = {f"{PREFIX}{name}_v1" for name in (
    "new", "generation", "bind", "begin_capture", "witness",
    "normal_close", "abort_attempt", "end_after_busy", "dispose",
)}


def check(mode: str, output: str) -> list[str]:
    if mode not in {"on", "off"}:
        raise ValueError(f"invalid feature mode: {mode}")
    definitions = {}
    for line in output.splitlines():
        fields = line.split()
        if len(fields) < 3:
            continue
        symbol = fields[-1]
        if symbol.startswith(PREFIX):
            definitions.setdefault(symbol, []).append(line)
    expected = EXPECTED if mode == "on" else set()
    actual = set(definitions)
    missing, extra = expected - actual, actual - expected
    duplicates = {name for name, lines in definitions.items() if len(lines) != 1}
    if missing or extra or duplicates:
        raise ValueError(
            f"holder {mode} ABI mismatch: missing={sorted(missing)}, "
            f"extra={sorted(extra)}, duplicates={sorted(duplicates)}")
    return [definitions[name][0] for name in sorted(expected)]


def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} on|off NM_OUTPUT", file=sys.stderr)
        return 2
    try:
        lines = check(sys.argv[1], Path(sys.argv[2]).read_text())
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    if lines:
        print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
