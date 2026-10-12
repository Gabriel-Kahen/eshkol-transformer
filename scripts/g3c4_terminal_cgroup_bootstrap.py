#!/usr/bin/env python3
"""Join a private v2 cgroup, wait for parent authentication, then exec one CLI."""

import os
from pathlib import Path
import sys


def starttime(pid: int) -> int:
    value = Path(f"/proc/{pid}/stat").read_text()
    return int(value[value.rindex(")") + 2:].split()[19])


def main() -> None:
    if len(sys.argv) < 6 or sys.argv[4] != "--":
        raise SystemExit(125)
    cgroup, ready_fd, go_fd = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    command = sys.argv[5:]
    if not command or not cgroup.is_absolute():
        raise SystemExit(125)
    with (cgroup / "cgroup.procs").open("w") as output:
        output.write(str(os.getpid()))
    os.write(ready_fd, f"{os.getpid()}:{starttime(os.getpid())}\n".encode())
    os.close(ready_fd)
    if os.read(go_fd, 1) != b"G":
        raise SystemExit(125)
    os.close(go_fd)
    os.execv(command[0], command)


if __name__ == "__main__":
    main()
