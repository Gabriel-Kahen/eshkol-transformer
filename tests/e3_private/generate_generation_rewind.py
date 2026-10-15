"""Generate one noninstalled E3 D2 rollback mutant from the pinned source."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "internal/e3/lib/e3_d2_restore.esk"
SOURCE_SHA256 = "5f26e48410a9567f612fb5bab1fe982b6e95642d149d36f0f89278cc91d27217"
MUTANT_SHA256 = "670a3e5cabd2eba86bfaf8125301b51b22ed29ff9985791e667fb7c1cfe1c125"
ANCHOR = """    (vector-set! core 15 (vector-ref record 7))
    (vector-set! state 10 'none)
"""
MUTANT = """    (vector-set! core 15 (vector-ref record 7))
    (vector-set! state 8 (vector-ref record 13))
    (vector-set! state 10 'none)
"""


def generate(source: bytes) -> bytes:
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise ValueError("E3 D2 source differs from reviewed rollback contract")
    text = source.decode("utf-8")
    if text.count(ANCHOR) != 1 or "(vector-set! state 8" in text:
        raise ValueError("E3 D2 restore anchor is not unique or already rewinds generation")
    result = text.replace(ANCHOR, MUTANT, 1).encode("utf-8")
    if hashlib.sha256(result).hexdigest() != MUTANT_SHA256:
        raise ValueError("generation rewind mutant bytes differ")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.resolve() == SOURCE.resolve():
        parser.error("mutant output must not replace the canonical source")
    payload = generate(SOURCE.read_bytes())
    with args.output.open("xb") as output:
        output.write(payload)


if __name__ == "__main__":
    main()
