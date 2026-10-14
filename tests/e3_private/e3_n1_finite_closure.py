#!/usr/bin/env python3
"""Generated-byte Linux subreaper/pidfd/pipe diagnostic; never N1 authority."""

import ctypes
import errno
import json
import os
import selectors
import signal
import socket
import stat
import subprocess
import sys
import time
from pathlib import Path

SCHEMA = "e3-n1-finite-closure-synthetic-v1"
RSS_CAP = 268435456
SECONDS = 30
STREAM_CAP = 4096
NETWORK_CALLS = (b"socket", b"socketpair", b"connect", b"bind", b"listen",
                 b"accept", b"accept4", b"sendto", b"sendmsg", b"sendmmsg",
                 b"recvfrom", b"recvmsg", b"recvmmsg", b"shutdown",
                 b"getsockopt", b"setsockopt", b"io_uring_setup")


class Hold(Exception):
    pass


def _boot_ns():
    if sys.platform != "linux" or not hasattr(time, "CLOCK_BOOTTIME"):
        raise Hold("boottime-unavailable")
    return time.clock_gettime_ns(time.CLOCK_BOOTTIME)


def _isolate():
    """One process only; children inherit the network syscall filter."""
    if not (hasattr(os, "wait4") and hasattr(os, "pidfd_open") and
            hasattr(signal, "pidfd_send_signal")):
        raise Hold("linux-process-api-unavailable")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise Hold("subreaper-unavailable")
    flag = ctypes.c_int()
    if libc.prctl(37, ctypes.byref(flag), 0, 0, 0) != 0 or flag.value != 1:
        raise Hold("subreaper-unverified")
    try:
        seccomp = ctypes.CDLL("libseccomp.so.2", use_errno=True)
    except OSError as error:
        raise Hold("network-filter-unavailable") from error
    seccomp.seccomp_init.argtypes = [ctypes.c_uint32]
    seccomp.seccomp_init.restype = ctypes.c_void_p
    seccomp.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    seccomp.seccomp_syscall_resolve_name.restype = ctypes.c_int
    seccomp.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32,
                                       ctypes.c_int, ctypes.c_uint]
    seccomp.seccomp_load.argtypes = [ctypes.c_void_p]
    seccomp.seccomp_release.argtypes = [ctypes.c_void_p]
    if libc.prctl(38, 1, 0, 0, 0) != 0:  # PR_SET_NO_NEW_PRIVS
        raise Hold("no-new-privs-unavailable")
    context = seccomp.seccomp_init(0x7fff0000)  # SCMP_ACT_ALLOW
    if not context:
        raise Hold("network-filter-init")
    try:
        deny = 0x00050000 | errno.EPERM  # SCMP_ACT_ERRNO(EPERM)
        for name in NETWORK_CALLS:
            number = seccomp.seccomp_syscall_resolve_name(name)
            if number < 0 or seccomp.seccomp_rule_add(context, deny, number, 0) != 0:
                raise Hold("network-filter-rule:" + name.decode())
        if seccomp.seccomp_load(context) != 0:
            raise Hold("network-filter-load")
    finally:
        seccomp.seccomp_release(context)
    try:
        connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except OSError as error:
        if error.errno != errno.EPERM:
            raise Hold("network-filter-unverified") from error
    else:
        connection.close()
        raise Hold("network-filter-unverified")


def _actor(mode, ack_fd):
    if os.read(0, 1) != b"G":
        os._exit(6)
    os.write(1, b"L")
    holder = os.fork()
    if holder == 0:
        os.setsid()
        os.write(2, (str(os.getpid()) + "\n").encode())
        if os.read(ack_fd, 1) != b"A":
            os._exit(7)
        os.close(ack_fd)
        try:
            connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        except OSError as error:
            if error.errno != errno.EPERM:
                os._exit(4)
        else:
            connection.close()
            os._exit(5)
        os.write(1, b"N")
        time.sleep(2 if mode == "late" else 0.06)
        os.write(1, b"G")
        os._exit(0)
    os.close(ack_fd)
    os._exit(0)


def _no_inherited_socket():
    for name in os.listdir("/proc/self/fd"):
        try:
            if stat.S_ISSOCK(os.fstat(int(name)).st_mode):
                raise Hold("inherited-socket")
        except OSError:
            continue


def _probe(mode, seconds):
    if mode not in ("normal", "late") or type(seconds) not in (int, float) or not (0 < seconds <= SECONDS):
        raise Hold("synthetic-arguments")
    started = _boot_ns()
    _no_inherited_socket()
    _isolate()
    ack_reader, ack_writer = os.pipe2(os.O_CLOEXEC)
    try:
        process = subprocess.Popen([sys.executable, "-I", "-B", str(Path(__file__).resolve()),
                                    "actor", mode, str(ack_reader)], stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   close_fds=True, pass_fds=(ack_reader,),
                                   start_new_session=True,
                                   env={"LC_ALL": "C", "TZ": "UTC", "PYTHONNOUSERSITE": "1"})
    except BaseException:
        os.close(ack_writer)
        raise
    finally:
        os.close(ack_reader)
    try:
        fds = {"leader": os.pidfd_open(process.pid)}
    except OSError as error:
        os.close(ack_writer)
        process.kill()
        os.wait4(process.pid, 0)
        process.stdin.close()
        process.stdout.close()
        process.stderr.close()
        raise Hold("leader-pidfd-unavailable") from error
    try:
        process.stdin.write(b"G")
        process.stdin.close()
    except OSError as error:
        os.close(ack_writer)
        try:
            signal.pidfd_send_signal(fds["leader"], signal.SIGKILL)
        except ProcessLookupError:
            pass
        os.wait4(process.pid, 0)
        process.stdout.close()
        process.stderr.close()
        os.close(fds["leader"])
        raise Hold("leader-start-failed") from error
    reaped = {}
    streams = {"stdout": bytearray(), "stderr": bytearray()}
    selector = selectors.DefaultSelector()
    for name, pipe in (("stdout", process.stdout), ("stderr", process.stderr)):
        os.set_blocking(pipe.fileno(), False)
        selector.register(pipe, selectors.EVENT_READ, name)
    deadline = started + int(seconds * 1e9)
    cleanup_end = deadline + 1_000_000_000
    timed_out = False
    reason = None
    holder_pid = None
    pipes_eof = False
    try:
        while True:
            for key, _ in selector.select(0.01):
                block = os.read(key.fileobj.fileno(), STREAM_CAP + 1)
                if not block:
                    selector.unregister(key.fileobj)
                    continue
                stream = streams[key.data]
                stream.extend(block[:max(0, STREAM_CAP + 1 - len(stream))])
                if len(stream) > STREAM_CAP:
                    reason = "stream-cap"
            if holder_pid is None and b"\n" in streams["stderr"]:
                try:
                    holder_pid = int(streams["stderr"].split(b"\n", 1)[0])
                    if holder_pid <= 0 or holder_pid == process.pid:
                        raise ValueError("holder-pid")
                    fds["holder"] = os.pidfd_open(holder_pid)
                    if timed_out:
                        if ack_writer is not None:
                            os.close(ack_writer)
                            ack_writer = None
                    else:
                        os.write(ack_writer, b"A")
                        os.close(ack_writer)
                        ack_writer = None
                except (ValueError, OSError):
                    reason = "holder-identity-unavailable"
                    if ack_writer is not None:
                        os.close(ack_writer)
                        ack_writer = None
            while True:
                try:
                    pid, status, usage = os.wait4(-1, os.WNOHANG)
                except ChildProcessError:
                    break
                if pid == 0:
                    break
                role = "leader" if pid == process.pid else "holder" if pid == holder_pid else "unknown"
                if role in reaped or role == "unknown" or usage is None:
                    reason = "unexpected-or-unmeasured-actor"
                else:
                    reaped[role] = {"exit_code": os.waitstatus_to_exitcode(status),
                                    "peak_rss_bytes": usage.ru_maxrss * 1024}
                if pid == process.pid:
                    process.returncode = os.waitstatus_to_exitcode(status)
            now = _boot_ns()
            if now >= deadline and not timed_out:
                timed_out = True
                reason = reason or "deadline"
                if ack_writer is not None:
                    os.close(ack_writer)
                    ack_writer = None
                for role, fd in fds.items():
                    if role not in reaped:
                        try:
                            signal.pidfd_send_signal(fd, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
            if len(reaped) == 2 and not selector.get_map():
                pipes_eof = True
                break
            if now >= cleanup_end:
                if len(reaped) != 2 or selector.get_map():
                    reason = "closure-uncertain"
                break
    finally:
        if ack_writer is not None:
            os.close(ack_writer)
        selector.close()
        process.stdout.close()
        process.stderr.close()
        for fd in fds.values():
            os.close(fd)
    complete = (reason is None and set(reaped) == {"leader", "holder"} and
                all(item["exit_code"] == 0 for item in reaped.values()) and
                bytes(streams["stdout"]) == b"LNG" and
                bytes(streams["stderr"]) == (str(holder_pid) + "\n").encode())
    return {"schema": SCHEMA, "status": "HOLD", "resource_proof": "HOLD",
            "reason": "unconfined-process-set-and-no-custody" if complete else (reason or "actor-failure"),
            "known_actors_reaped": set(reaped) == {"leader", "holder"},
            "pipes_eof": pipes_eof, "network_filter_loaded": True,
            "actor_network_denied": b"N" in streams["stdout"],
            "actor_peaks": {key: item["peak_rss_bytes"] for key, item in reaped.items()},
            "actor_exit_codes": {key: item["exit_code"] for key, item in reaped.items()},
            "elapsed_ns": _boot_ns() - started, "cap_ns": int(seconds * 1e9),
            "rss_cap_bytes": RSS_CAP,
            "scope": "fixed-generated-bytes;subreaper-pidfd-diagnostic-only"}


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "actor" and sys.argv[2] in ("normal", "late"):
        _actor(sys.argv[2], int(sys.argv[3]))
        return 0
    if len(sys.argv) != 4 or sys.argv[1] != "probe":
        return 2
    try:
        result = _probe(sys.argv[2], float(sys.argv[3]))
    except (Hold, OSError, ValueError) as error:
        result = {"schema": SCHEMA, "status": "HOLD", "resource_proof": "HOLD",
                  "reason": str(error), "known_actors_reaped": False,
                  "scope": "fixed-generated-bytes;no-real-read"}
    sys.stdout.write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
