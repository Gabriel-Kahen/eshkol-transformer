#!/usr/bin/env python3
"""Generate deterministic real-parser fixtures for the private C2 inspector."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tests.c2.test_checkpoint_format import make_fixture, resign_outer


def main() -> None:
    root = Path(sys.argv[1])
    root.mkdir(parents=True, exist_ok=True)
    valid, offsets = make_fixture()
    (root / "valid.c2").write_bytes(valid)

    compiler = valid.copy()
    compiler[offsets["compiler"] + 18] = ord("9")
    resign_outer(compiler)
    (root / "compiler.c2").write_bytes(compiler)

    checksum = valid.copy()
    checksum[offsets["x1"]] ^= 1
    (root / "checksum.c2").write_bytes(checksum)

    hostile = valid.copy()
    hostile[offsets["tokenizer"]] = ord("@")
    resign_outer(hostile)
    (root / "hostile.c2").write_bytes(hostile)

    profile, _ = make_fixture(8_388_612, 3)
    (root / "profile.c2").write_bytes(profile)


if __name__ == "__main__":
    main()
