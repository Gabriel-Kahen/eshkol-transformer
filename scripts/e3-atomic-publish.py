#!/usr/bin/env python3
"""Atomically publish one complete E3 artifact directory."""

import ctypes
import errno
import os
from pathlib import Path
import sys


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: e3-atomic-publish.py STAGING TARGET")
    staging = Path(sys.argv[1]).absolute()
    target = Path(sys.argv[2]).absolute()
    if staging.is_symlink() or not staging.is_dir():
        raise SystemExit(f"E3 artifact staging path must be a directory: {staging}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or (target.exists() and not target.is_dir()):
        raise SystemExit(f"E3 artifact target must be a directory, not {target}")
    if not target.exists():
        os.rename(staging, target)
        return 0

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = libc.renameat2
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100, os.fsencode(staging), -100, os.fsencode(target), 2
    )
    if result == 0:
        return 0
    error = ctypes.get_errno()
    if error in (errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP):
        raise SystemExit("atomic E3 artifact replacement is unsupported")
    raise OSError(error, os.strerror(error), str(target))


if __name__ == "__main__":
    raise SystemExit(main())
