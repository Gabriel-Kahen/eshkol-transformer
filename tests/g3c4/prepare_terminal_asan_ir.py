#!/usr/bin/env python3
"""Make an exact reversible ASan copy of pinned Eshkol LLVM IR."""

from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
import sys

DEFINITION = re.compile(rb"^define [^\r\n]* \{\n$")
DEFINE_TOKEN = re.compile(rb"^\s*define(?=\s|$)")


def regular(path: Path) -> None:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or not info.st_size:
        raise ValueError(f"unsafe or empty IR: {path}")


def lines(source):
    count = 0
    for original in source:
        if not original.endswith(b"\n") or b"\r" in original or b"sanitize_address" in original:
            raise ValueError("unsupported IR byte format or existing ASan marker")
        if DEFINE_TOKEN.match(original):
            if not DEFINITION.fullmatch(original) or b"@" not in original or b"(" not in original:
                raise ValueError("unsupported function definition")
            count += 1
            yield original, original[:-3] + b" sanitize_address {\n", True
        else:
            yield original, original, False
    if count == 0:
        raise ValueError("IR has no definitions")


def scan_pair(raw: Path, derived: Path) -> dict:
    regular(raw)
    regular(derived)
    raw_hash = sha256()
    derived_hash = sha256()
    definitions = []
    with raw.open("rb") as source, derived.open("rb") as output:
        for original, expected, marked in lines(source):
            actual = output.read(len(expected))
            if actual != expected:
                raise ValueError("ASan IR differs from exact reversible rewrite")
            raw_hash.update(original)
            derived_hash.update(actual)
            if marked:
                definitions.append(sha256(original).hexdigest())
        if output.read(1):
            raise ValueError("ASan IR has extra bytes")
    return {
        "raw_sha256": raw_hash.hexdigest(),
        "derived_sha256": derived_hash.hexdigest(),
        "definitions": len(definitions),
        "definition_line_sha256": definitions,
    }


def prepare(raw: Path, derived: Path, receipt: Path) -> None:
    regular(raw)
    if len({raw.absolute(), derived.absolute(), receipt.absolute()}) != 3:
        raise ValueError("IR paths alias")
    for path in (derived, receipt):
        try:
            path.lstat()
        except FileNotFoundError:
            pass
        else:
            raise ValueError(f"output already exists: {path}")
    with raw.open("rb") as source, os.fdopen(
        os.open(derived, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600),
        "wb",
    ) as output:
        for _, rewritten, _ in lines(source):
            output.write(rewritten)
    result = scan_pair(raw, derived)
    with os.fdopen(
        os.open(receipt, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600),
        "w",
    ) as output:
        json.dump(result, output, sort_keys=True, separators=(",", ":"))
        output.write("\n")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("usage: prepare_terminal_asan_ir.py RAW_LL ASAN_LL RECEIPT")
    prepare(*(Path(argument) for argument in sys.argv[1:]))
