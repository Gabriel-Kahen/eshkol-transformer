#!/usr/bin/env python3
"""Stage the exact noninstalled I2 abort-removal control for the E3 zero tuple."""

import argparse
import hashlib
import json
import os
from pathlib import Path


SOURCES = (
    ("tests/e3_private/zero_control_driver_root.esk", "ea54f57a8f345bac2618b82367ddb9c08a48cc0f9571359a7a95a45b2af69f15"),
    ("native/e3_private_driver_root.esk", "a27a9a3390225b08cdbf2ad0ad80fd49536736f1fc06d5197451cfde839a82cd"),
    ("native/e3_private_root.esk", "c5a0cdbceff21aea36b68a4b438e76962250a56cbf5259e2b765c92ce50c63b0"),
    ("native/m3_package_root.esk", "02bac9a961beea7d01229df1a745e037e9698a64620c6308d8d2a17ed2697a6b"),
    ("native/m3t_package_root.esk", "f3f804cedff58b87c0ba08d3a30a3559bd2d0850d53558a7b3a21b4fd7e01c98"),
    ("native/i2_wave2_root.esk", "f1314ccd4c5a382733115005c93d8712290bd63c5373eac2509093989fefad33"),
)
TOP_LOAD = (b'(load "../../native/e3_private_driver_root.esk")',
            b'(load "e3_private_driver_root.esk")')
ABORT_ARM = (b'                 (i2-native-copy-builder-abort native-plan)\n'
             b'                 (raise caught)',
             b'                 (raise caught)')


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def stage(root: Path, output: Path) -> list[dict[str, str]]:
    if not output.is_dir() or output.is_symlink() or any(output.iterdir()):
        raise ValueError("mutant source stage must be an empty owned directory")
    records = []
    for index, (relative, expected) in enumerate(SOURCES):
        original = root / relative
        if original.is_symlink() or not original.is_file():
            raise ValueError(f"invalid canonical source: {relative}")
        raw = original.read_bytes()
        if sha256(raw) != expected:
            raise ValueError(f"canonical source SHA drifted: {relative}")
        staged = raw
        if index == 0:
            before, after = TOP_LOAD
            if staged.count(before) != 1:
                raise ValueError("zero root load rewrite is not unique")
            staged = staged.replace(before, after, 1)
            if staged.replace(after, before, 1) != raw:
                raise ValueError("zero root reversal differs from canonical bytes")
        elif index == 5:
            before, after = ABORT_ARM
            if staged.count(before) != 1:
                raise ValueError("I2 exception-arm abort is not unique")
            staged = staged.replace(before, after, 1)
            if staged.replace(after, before, 1) != raw:
                raise ValueError("I2 mutant reversal differs from canonical bytes")
        target = output / original.name
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(staged)
        records.append({"canonical": relative, "original_sha256": expected,
                        "staged": target.name, "staged_sha256": sha256(staged)})
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = stage(args.root, args.output)
    print(json.dumps(records, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
