"""Capture a mutant SIGABRT without a shell-generated job diagnostic."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import resource
import signal
import subprocess


def capture(prefix: Path, command: list[str]) -> int:
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if not command or not prefix.is_absolute() or prefix.parent.is_symlink():
        raise ValueError("expected an absolute fresh abort evidence prefix")
    paths = [prefix.with_name(prefix.name + suffix)
             for suffix in (".stdout", ".stderr", ".exit", ".raw-exit")]
    if any(path.exists() or path.is_symlink() for path in paths):
        raise ValueError("abort evidence path already exists")
    with paths[0].open("xb") as stdout, paths[1].open("xb") as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr,
                                   start_new_session=True)
        try:
            raw = process.wait(timeout=135)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
    canonical = 128 - raw if raw < 0 else raw
    paths[2].write_text(f"{canonical}\n")
    paths[3].write_text(f"{raw}\n")
    if canonical == 134 and raw != -signal.SIGABRT:
        raise ValueError("status 134 was returned without a SIGABRT signal")
    return canonical


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True, type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if capture(args.prefix, command) != 134:
        raise SystemExit("expected only canonical SIGABRT status 134")


if __name__ == "__main__":
    main()
