#!/usr/bin/env python3
"""Validate and sync four unique raw extraction receipts without following links."""

from __future__ import annotations

import os
from pathlib import Path
import re
import stat
import sys


def check(paths: list[Path]) -> None:
    if len(paths) != 4 or len(set(paths)) != 4:
        raise ValueError("four unique raw receipt paths required")
    for index, path in enumerate(paths):
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_uid != os.geteuid():
            raise ValueError(f"nonordinary raw receipt: {path.name}")
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            opened = os.fstat(fd)
            after = path.lstat()
            fields = lambda x: (x.st_dev, x.st_ino, x.st_mode, x.st_nlink, x.st_uid, x.st_size)
            if fields(opened) != fields(before) or fields(after) != fields(before):
                raise ValueError(f"raw receipt changed: {path.name}")
            if index == 3:
                raw = os.read(fd, 16)
                if os.read(fd, 1):
                    raise ValueError("raw receipt has excess child-exit bytes")
                if not re.fullmatch(rb"(?:0|[1-9][0-9]{0,2})\n", raw) or int(raw) > 255:
                    raise ValueError("raw receipt lacks genuine bounded numeric child exit")
            os.fsync(fd)
        finally:
            os.close(fd)


def _write_checked(fd: int, raw: bytes) -> None:
    if os.write(fd, raw) != len(raw):
        raise OSError("short child-exit receipt write")
    os.fsync(fd)


def write_exit(stage: Path, result: int) -> None:
    if not 0 <= result <= 255:
        raise ValueError("child exit outside numeric wait range")
    directory = os.open(stage, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        fd = os.open("compiler.exit.raw", os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                     os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=directory)
        try:
            raw = f"{result}\n".encode("ascii")
            _write_checked(fd, raw)
        finally:
            os.close(fd)
        os.fsync(directory)
    finally:
        os.close(directory)


if __name__ == "__main__":
    try:
        if len(sys.argv) == 4 and sys.argv[1] == "--write-exit":
            write_exit(Path(sys.argv[2]), int(sys.argv[3]))
        else:
            check([Path(value) for value in sys.argv[1:]])
    except (OSError, ValueError) as error:
        print(f"sealed-fixture raw receipt rejected: {error}", file=sys.stderr)
        raise SystemExit(1)
