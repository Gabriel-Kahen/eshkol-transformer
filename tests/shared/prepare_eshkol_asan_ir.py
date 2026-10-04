#!/usr/bin/env python3
"""Make a retained, reversible ASan IR copy without loading the module.

LLVM IR input lacks Clang frontend ASan attributes. This test-only recipe
marks each supported Eshkol function definition; unexpected syntax fails.
"""

import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
import sys


DEFINITION = re.compile(rb"^define [^\r\n]* \{\n$")
DEFINE_TOKEN = re.compile(rb"^\s*define(?=\s|$)")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def transformed(lines):
    count = 0
    for line in lines:
        if not line.endswith(b"\n") or b"\r" in line or b"sanitize_address" in line:
            raise ValueError("unsupported IR byte format or existing ASan attribute")
        if DEFINE_TOKEN.match(line):
            if (not DEFINITION.fullmatch(line) or b"{" in line[:-2]
                    or b"}" in line[:-2] or b"@" not in line
                    or b"(" not in line or b")" not in line):
                raise ValueError("unsupported function definition syntax")
            count += 1
            yield line, line[:-3] + b" sanitize_address {\n", sha(line)
        else:
            yield line, line, None
    if count == 0:
        raise ValueError("IR has no definitions")


def prepare(raw: bytes) -> tuple[bytes, list[str]]:
    """Small-fixture interface; supported gates use streaming file functions."""
    if not raw:
        raise ValueError("empty IR")
    output = bytearray()
    definitions = []
    for _, derived, definition in transformed(io.BytesIO(raw)):
        output.extend(derived)
        if definition is not None:
            definitions.append(definition)
    return bytes(output), definitions


def regular(path: Path) -> None:
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size == 0:
        raise ValueError(f"unsafe or empty artifact: {path}")


def real_directory(path: Path) -> None:
    if not path.is_dir() or path.is_symlink() or path.resolve() != path.absolute():
        raise ValueError(f"unsafe directory: {path}")


def fresh(path: Path) -> None:
    try:
        path.lstat()  # Also rejects dangling symlinks.
    except FileNotFoundError:
        pass
    else:
        raise ValueError(f"output already exists: {path}")
    real_directory(path.parent)


def scan_pair(raw_path: Path, derived_path: Path) -> dict:
    """Stream-compare every derived byte against the exact raw line rewrite."""
    real_directory(raw_path.parent)
    real_directory(derived_path.parent)
    regular(raw_path)
    regular(derived_path)
    raw_hash = hashlib.sha256()
    derived_hash = hashlib.sha256()
    definitions = []
    with raw_path.open("rb") as raw_file, derived_path.open("rb") as derived_file:
        for original, expected, definition in transformed(raw_file):
            actual = derived_file.read(len(expected))
            if actual != expected:
                raise ValueError("derived IR differs from exact reversible rewrite")
            raw_hash.update(original)
            derived_hash.update(actual)
            if definition is not None:
                definitions.append(definition)
        if derived_file.read(1):
            raise ValueError("derived IR has extra bytes")
    return {"raw_sha256": raw_hash.hexdigest(),
            "derived_sha256": derived_hash.hexdigest(),
            "definitions": len(definitions),
            "definition_line_sha256": definitions}


def main(raw_path: Path, derived_path: Path, receipt_path: Path) -> None:
    real_directory(raw_path.parent)
    regular(raw_path)
    fresh(derived_path)
    fresh(receipt_path)
    if len({raw_path.absolute(), derived_path.absolute(), receipt_path.absolute()}) != 3:
        raise ValueError("IR paths alias")
    with raw_path.open("rb") as raw_file, os.fdopen(
        os.open(derived_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600),
        "wb",
    ) as derived_file:
        for _, derived, _ in transformed(raw_file):
            derived_file.write(derived)
    receipt = scan_pair(raw_path, derived_path)
    with os.fdopen(
        os.open(receipt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600),
        "w",
    ) as output:
        json.dump(receipt, output, sort_keys=True, separators=(",", ":"))
        output.write("\n")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("usage: prepare_eshkol_asan_ir.py RAW_LL DERIVED_LL RECEIPT")
    main(*(Path(arg) for arg in sys.argv[1:]))
