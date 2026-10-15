#!/usr/bin/env python3
"""Generate the closed numerical tuple's in-call D2 end-cursor observer."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "native/e3_private_root.esk"
SOURCE_EXTENSION = ROOT / "native/e3_private_extension.esk"
SOURCE_DRIVER = ROOT / "tests/e3_private/driver.esk"
PINS = {
    SOURCE_ROOT: "c5a0cdbceff21aea36b68a4b438e76962250a56cbf5259e2b765c92ce50c63b0",
    SOURCE_EXTENSION: "571f878aaa8b77ece6474c8c184abbd96aa8863581b989d2e1fef465cab361d6",
    SOURCE_DRIVER: "5538c33183760a3522933ab35c7eff9684950be92238d8170de95a1014b4ef0b",
}


def _pinned(path: Path) -> str:
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != PINS[path]:
        raise ValueError(f"numerical cursor predecessor changed: {path.name}")
    return source.decode("ascii")


def _replace(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise ValueError("numerical cursor insertion anchor changed")
    return source.replace(old, new, 1)


def generate() -> dict[str, bytes]:
    root = _replace(_pinned(SOURCE_ROOT),
                    '(load "e3_private_extension.esk")',
                    '(load "e3_numerical_extension.esk")')
    extension = _pinned(SOURCE_EXTENSION)
    # Inline the observer at the sole successful traversal site. A new top-level
    # function would alter the exact package symbol contract for this tuple.
    observer = '''(let ((cursor (d2-token-dataset-cursor dataset))
                                                     (digits "0123456789abcdef"))
                                                 (display "E3ENDCURSOR\\t")
                                                 (let loop ((index 0))
                                                   (if (= index (bytevector-length cursor)) #t
                                                       (let ((byte (bytevector-u8-ref cursor index)))
                                                         (display (string-ref digits (quotient byte 16)))
                                                         (display (string-ref digits (remainder byte 16)))
                                                         (loop (+ index 1)))))
                                                 (newline))'''
    extension = _replace(
        extension,
        '(if (e3-traverse! frame dataset)\n'
        '                                             (e3-restore-closed! frame #t)',
        '(if (e3-traverse! frame dataset)\n'
        '                                             (begin\n'
        '                                               ' + observer + '\n'
        '                                               (e3-restore-closed! frame #t))',
    )
    driver = _replace(_pinned(SOURCE_DRIVER),
                      ":maximum-batch-bytes 34",
                      ":maximum-batch-bytes 1048576")
    return {
        "e3_numerical_private_root.esk": root.encode("ascii"),
        "e3_numerical_extension.esk": extension.encode("ascii"),
        "e3_numerical_driver.esk": driver.encode("ascii"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.output_dir.is_symlink() or not args.output_dir.is_dir():
        raise ValueError("numerical generated source directory is absent or a symlink")
    for name, data in generate().items():
        path = args.output_dir / name
        if path.is_symlink():
            raise ValueError("numerical generated source is a symlink")
        if args.check:
            if path.read_bytes() != data:
                raise ValueError(f"numerical generated source differs: {name}")
        else:
            with path.open("xb") as stream:
                stream.write(data)
        print(f"{name}\t{hashlib.sha256(data).hexdigest()}")


if __name__ == "__main__":
    main()
