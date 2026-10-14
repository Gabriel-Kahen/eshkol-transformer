#!/usr/bin/env python3
"""Generated-byte cgroup placement diagnostic; protected custody stays HOLD."""

import ctypes
import json
import os
import selectors
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e3_n1_finite_closure as closure

SCHEMA = "e3-n1-scope-custody-synthetic-v1"
ROLES = ("broker", "sealer")
CLONE_PIDFD = 0x00001000
CLONE_INTO_CGROUP = 0x200000000
SYS_CLONE3_X86_64 = 435
MAX_RECORD = 512


class Hold(Exception):
    pass


class CloneArgs(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in (
        "flags", "pidfd", "child_tid", "parent_tid", "exit_signal", "stack",
        "stack_size", "tls", "set_tid", "set_tid_size", "cgroup")]


def _cgroup_path():
    lines = Path("/proc/self/cgroup").read_text().splitlines()
    matches = [line[3:] for line in lines if line.startswith("0::/")]
    if len(matches) != 1:
        raise Hold("cgroup-v2-unavailable")
    return Path("/sys/fs/cgroup" + matches[0])


def _events_empty(path):
    fields = dict(line.split() for line in (path / "cgroup.events").read_text().splitlines())
    return fields.get("populated") == "0"


def _clone_role(cgroup_fd, expected, role):
    if role not in ROLES:
        raise Hold("unknown-role")
    read_fd, write_fd = os.pipe2(os.O_CLOEXEC)
    pidfd = ctypes.c_int(-1)
    args = CloneArgs(flags=CLONE_PIDFD | CLONE_INTO_CGROUP,
                     pidfd=ctypes.addressof(pidfd), exit_signal=signal.SIGCHLD,
                     cgroup=cgroup_fd)
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    result = libc.syscall(SYS_CLONE3_X86_64, ctypes.byref(args), ctypes.sizeof(args))
    if result == 0:
        os.close(read_fd)
        try:
            actual = _cgroup_path()
            os.write(write_fd, (role + ":" + str(actual) + "\n").encode())
            os._exit(0)
        except BaseException:
            os._exit(9)
    os.close(write_fd)
    if result < 0:
        error = ctypes.get_errno()
        os.close(read_fd)
        raise Hold("clone3-into-cgroup:" + str(error))
    observed = bytearray()
    selector = None
    reaped = None
    eof = False
    try:
        selector = selectors.DefaultSelector()
        os.set_blocking(read_fd, False)
        selector.register(read_fd, selectors.EVENT_READ)
        deadline = closure._boot_ns() + 1_000_000_000
        while closure._boot_ns() < deadline:
            for _, _ in selector.select(0.01):
                block = os.read(read_fd, MAX_RECORD + 1)
                if not block:
                    selector.unregister(read_fd)
                    eof = True
                else:
                    observed.extend(block[:max(0, MAX_RECORD + 1 - len(observed))])
            if reaped is None:
                pid, status, usage = os.wait4(result, os.WNOHANG)
                if pid:
                    reaped = (os.waitstatus_to_exitcode(status), usage.ru_maxrss * 1024)
            if reaped is not None and eof:
                break
    finally:
        if reaped is None:
            try:
                signal.pidfd_send_signal(pidfd.value, signal.SIGKILL)
            except OSError:
                pass  # A child that already exited still needs wait4 below.
            cleanup_end = closure._boot_ns() + 1_000_000_000
            while closure._boot_ns() < cleanup_end:
                try:
                    pid, status, usage = os.wait4(result, os.WNOHANG)
                except OSError:
                    break
                if pid:
                    reaped = (os.waitstatus_to_exitcode(status), usage.ru_maxrss * 1024)
                    break
                time.sleep(0.01)
        if selector is not None:
            selector.close()
        os.close(read_fd)
        os.close(pidfd.value)
        if reaped is None:
            raise Hold("role-closure-uncertain")
    if reaped is None or not eof or reaped[0] != 0 or len(observed) > MAX_RECORD:
        raise Hold("role-closure-uncertain")
    if bytes(observed) != (role + ":" + str(expected) + "\n").encode():
        raise Hold("role-cgroup-mismatch")
    return {"exit_code": reaped[0], "peak_rss_bytes": reaped[1],
            "pipe_eof": eof, "direct_cgroup": str(expected)}


def _scope_worker(unit):
    if sys.platform != "linux" or os.uname().machine != "x86_64":
        raise Hold("clone3-architecture-unsupported")
    current = _cgroup_path()
    if current.name != unit or not unit.endswith(".scope"):
        raise Hold("unexpected-scope")
    closure._no_inherited_socket()
    closure._isolate()  # Network syscalls denied before any actor is cloned.
    child = current / "n1-generated-actors"
    child.mkdir(mode=0o700)
    fd = None
    try:
        fd = os.open(child, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        actors = {role: _clone_role(fd, child, role) for role in ROLES}
        if not _events_empty(child):
            raise Hold("actor-cgroup-populated")
    finally:
        if fd is not None:
            os.close(fd)
        try:
            child.rmdir()
        except OSError as error:
            raise Hold("actor-cgroup-removal-uncertain") from error
    return {"scope_status": "fixed-actor-direct-placement-observed",
            "actors": actors, "actor_scope_empty": True,
            "protected_pre_read_custody": "HOLD",
            "source_opened": False, "broker_issued_receipt": False,
            "sealer_issued_seal": False,
            "reason": "no-independent-distinct-uid-authority"}


def probe():
    """Run only generated role markers; no request/source path is accepted."""
    started = closure._boot_ns()
    unit = "codex-e3-n1-scope-" + uuid.uuid4().hex[:16] + ".scope"
    command = ["systemd-run", "--user", "--scope", "--quiet", "-p", "Delegate=yes",
               "--unit=" + unit, sys.executable, "-I", "-B",
               str(Path(__file__).resolve()), "worker", unit]
    try:
        process = subprocess.run(command, stdin=subprocess.DEVNULL,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 close_fds=True, timeout=4,
                                 env={"LC_ALL": "C", "TZ": "UTC", "PYTHONNOUSERSITE": "1",
                                      "XDG_RUNTIME_DIR": os.environ.get("XDG_RUNTIME_DIR", "")})
    except subprocess.TimeoutExpired:
        # The direct systemd-run process is gone, but its unique scope may persist.
        try:
            subprocess.run(["systemctl", "--user", "stop", unit],
                           stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, close_fds=True, timeout=2)
        except (OSError, subprocess.TimeoutExpired):
            pass
        return {"schema": SCHEMA, "status": "HOLD", "resource_proof": "HOLD",
                "reason": "scope-timeout-closure-unverified", "unit": unit,
                "source_opened": False, "protected_pre_read_custody": "HOLD"}
    except OSError as error:
        return {"schema": SCHEMA, "status": "HOLD", "resource_proof": "HOLD",
                "reason": "delegated-scope-unavailable:" + type(error).__name__,
                "source_opened": False, "protected_pre_read_custody": "HOLD"}
    try:
        if process.returncode != 0 or process.stderr:
            raise ValueError("delegated-scope-failed")
        record = json.loads(process.stdout)
        if record.get("schema") != SCHEMA or record.get("status") != "HOLD":
            raise ValueError("worker-record")
    except (ValueError, UnicodeDecodeError):
        return {"schema": SCHEMA, "status": "HOLD", "resource_proof": "HOLD",
                "reason": "scope-worker-unverified", "source_opened": False,
                "protected_pre_read_custody": "HOLD"}
    record["outer_elapsed_ns"] = closure._boot_ns() - started
    return record


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "worker":
        try:
            body = _scope_worker(sys.argv[2])
        except (Hold, OSError, ValueError) as error:
            body = {"scope_status": "HOLD", "reason": str(error),
                    "source_opened": False, "protected_pre_read_custody": "HOLD"}
        record = {"schema": SCHEMA, "status": "HOLD", "resource_proof": "HOLD", **body}
    elif len(sys.argv) == 2 and sys.argv[1] == "probe":
        record = probe()
    else:
        return 2
    sys.stdout.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
