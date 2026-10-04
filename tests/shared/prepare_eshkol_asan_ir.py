#!/usr/bin/env python3
"""Add ASan function attributes to an exact, retained Eshkol IR copy.

Clang's IR input path does not add address checks to unmarked definitions.
This transformation is deliberately narrower than a general LLVM IR parser:
unexpected definition syntax is an unsupported gate result, not a rewrite.
"""

import hashlib
import json
from pathlib import Path
import re
import sys


DEFINITION = re.compile(rb"^define [^\r\n]* \{\n$")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def prepare(raw: bytes) -> tuple[bytes, list[str]]:
    if (not raw or b"\r" in raw or not raw.endswith(b"\n")
            or b"sanitize_address" in raw):
        raise ValueError("unsupported IR byte format")
    output = []
    definitions = []
    for line in raw.splitlines(keepends=True):
        if line.lstrip().startswith(b"define ") and not line.startswith(b"define "):
            raise ValueError("indented definition is unsupported")
        if line.startswith(b"define "):
            if (not DEFINITION.fullmatch(line) or b"sanitize_address" in line
                    or b"{" in line[:-2] or b"}" in line[:-2]
                    or b"@" not in line or b"(" not in line or b")" not in line):
                raise ValueError("unsupported or already instrumented definition")
            definitions.append(sha(line))
            output.append(line[:-3] + b" sanitize_address {\n")
        else:
            output.append(line)
    if not definitions:
        raise ValueError("IR has no definitions")
    derived = b"".join(output)
    if derived == raw or derived.count(b" sanitize_address {\n") != len(definitions):
        raise ValueError("incomplete ASan definition coverage")
    restored = derived.replace(b" sanitize_address {\n", b" {\n")
    if restored != raw:
        raise ValueError("derived IR does not restore to exact raw bytes")
    return derived, definitions


def main(raw_path: Path, derived_path: Path, receipt_path: Path) -> None:
    if derived_path.exists() or receipt_path.exists():
        raise ValueError("derived IR or receipt already exists")
    raw = raw_path.read_bytes()
    derived, definitions = prepare(raw)
    derived_path.write_bytes(derived)
    receipt_path.write_text(json.dumps({
        "raw_sha256": sha(raw),
        "derived_sha256": sha(derived),
        "definitions": len(definitions),
        "definition_line_sha256": definitions,
    }, sort_keys=True, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("usage: prepare_eshkol_asan_ir.py RAW_LL DERIVED_LL RECEIPT")
    main(*(Path(arg) for arg in sys.argv[1:]))
