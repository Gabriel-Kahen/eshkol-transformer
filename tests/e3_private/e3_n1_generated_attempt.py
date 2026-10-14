#!/usr/bin/env python3
"""Generated-only finite attempt diagnostic. Every result retains custody HOLD."""

import hashlib
import json
import os
import select
import signal
import tempfile
import time
from pathlib import Path

SCHEMA = "e3-n1-generated-attempt-v1"
FAULTS = {None, "before_hold", "after_hold", "after_receipt", "after_seal",
          "hold_sync", "receipt_sync", "seal_sync", "timeout",
          "inherited_fd", "cgroup_migration"}
GENERATED = b"e3-n1-generated-stream-v1\n"
MAX_REPORT = 8192


class Hold(Exception):
    pass


def _now():
    return time.clock_gettime_ns(time.CLOCK_BOOTTIME)


def _cgroup():
    lines = Path("/proc/self/cgroup").read_text().splitlines()
    matches = [line[3:] for line in lines if line.startswith("0::/")]
    if len(matches) != 1:
        raise Hold("cgroup-v2-unavailable")
    return matches[0]


def _close_except(allowed):
    for item in os.listdir("/proc/self/fd"):
        fd = int(item)
        if fd not in allowed:
            try:
                os.close(fd)
            except OSError:
                pass


def _assert_fds(allowed):
    for item in os.listdir("/proc/self/fd"):
        fd = int(item)
        if fd not in allowed:
            try:
                os.fstat(fd)
            except OSError:
                continue  # The temporary /proc directory descriptor closed.
            raise Hold("unexpected-inherited-fd")


def _sync_json(path, body, fault=False):
    raw = json.dumps(body, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        offset = 0
        while offset < len(raw):
            written = os.write(fd, raw[offset:])
            if written <= 0:
                raise Hold("short-record-write")
            offset += written
        if fault:
            raise Hold("injected-record-sync")
        os.fsync(fd)
    finally:
        os.close(fd)
    parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)
    if path.read_bytes() != raw:
        raise Hold("readback-mismatch")
    return hashlib.sha256(raw).hexdigest()


def _phase(fd, name, fault=None):
    group = "/synthetic-migration" if fault == "cgroup_migration" else _cgroup()
    os.write(fd, (name + "|" + group + "\n").encode())


def _broker(root, report_fd, data_fd, fault, extra_fd):
    allowed = {0, 1, 2, report_fd, data_fd}
    _close_except(allowed | ({extra_fd} if extra_fd is not None else set()))
    _assert_fds(allowed)
    if fault == "before_hold":
        raise Hold("injected-before-hold")
    _sync_json(root / "HOLD.json", {"schema": SCHEMA, "state": "HOLD"}, fault == "hold_sync")
    _phase(report_fd, "HOLD")
    if fault == "timeout":
        time.sleep(2)
    chunks = []
    while True:
        block = os.read(data_fd, 128)
        if not block:
            break
        chunks.append(block)
        if sum(map(len, chunks)) > len(GENERATED):
            raise Hold("generated-overflow")
    if b"".join(chunks) != GENERATED:
        raise Hold("generated-stream-mismatch")
    _sync_json(root / "stage.json", {"generated_sha256": hashlib.sha256(GENERATED).hexdigest()})
    receipt_hash = _sync_json(root / "receipt-stub.json", {"schema": SCHEMA, "kind": "unsigned-receipt-stub"}, fault == "receipt_sync")
    _sync_json(root / "commit-stub.json", {"receipt_sha256": receipt_hash})
    _phase(report_fd, "RECEIPT")


def _reader(data_fd):
    allowed = {0, 1, 2, data_fd}
    _close_except(allowed)
    _assert_fds(allowed)
    os.write(data_fd, GENERATED)


def _sealer(root, report_fd, fault):
    allowed = {0, 1, 2, report_fd}
    _close_except(allowed)
    _assert_fds(allowed)
    if not (root / "HOLD.json").is_file() or not (root / "commit-stub.json").is_file():
        raise Hold("predecessor-missing")
    _sync_json(root / "seal-stub.json", {"schema": SCHEMA, "kind": "unsigned-seal-stub"}, fault == "seal_sync")
    _phase(report_fd, "SEAL", fault)


def _spawn(fn, *args):
    pid = os.fork()
    if pid == 0:
        try:
            fn(*args)
            os._exit(0)
        except BaseException:
            os._exit(9)
    return pid


def _line(fd, deadline):
    data = bytearray()
    while _now() < deadline and len(data) <= 256:
        ready, _, _ = select.select([fd], [], [], max(0, min(0.01, (deadline - _now()) / 1e9)))
        if ready:
            byte = os.read(fd, 1)
            if not byte:
                raise Hold("phase-eof")
            if byte == b"\n":
                name, group = data.decode().split("|", 1)
                return name, group
            data.extend(byte)
    raise Hold("phase-timeout")


def _expect_eof(fd, deadline):
    while _now() < deadline:
        ready, _, _ = select.select([fd], [], [], 0.01)
        if ready:
            if os.read(fd, 1):
                raise Hold("unexpected-phase-byte")
            return
    raise Hold("phase-pipe-not-eof")


def _unique_object(pairs):
    body = {}
    for key, value in pairs:
        if key in body:
            raise ValueError("duplicate-report-key")
        body[key] = value
    return body


def _reap(roles, force):
    peaks = {}
    unresolved = []
    for role, pid in roles.items():
        deadline = _now() + (0 if force else 200_000_000)
        while _now() < deadline:
            try:
                waited, status, usage = os.wait4(pid, os.WNOHANG)
            except ChildProcessError:
                unresolved.append(role)
                break
            if waited:
                peaks[role] = {"exit_code": os.waitstatus_to_exitcode(status),
                               "peak_rss_bytes": usage.ru_maxrss * 1024}
                break
            time.sleep(0.005)
        else:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            cleanup_end = _now() + 1_000_000_000
            while _now() < cleanup_end:
                try:
                    waited, status, usage = os.wait4(pid, os.WNOHANG)
                except ChildProcessError:
                    break
                if waited:
                    peaks[role] = {"exit_code": os.waitstatus_to_exitcode(status),
                                   "peak_rss_bytes": usage.ru_maxrss * 1024}
                    break
                time.sleep(0.005)
            if role not in peaks:
                unresolved.append(role)
    return peaks, unresolved


def _setup(fault):
    roles = {}
    fds = []
    phases = []
    reason = "generated-order-observed"
    expected = _cgroup()
    with tempfile.TemporaryDirectory(prefix="e3-n1-generated-") as directory:
        root = Path(directory)
        broker_r, broker_w = os.pipe2(os.O_CLOEXEC)
        data_r, data_w = os.pipe2(os.O_CLOEXEC)
        fds.extend((broker_r, broker_w, data_r, data_w))
        extra = os.open("/dev/null", os.O_RDONLY) if fault == "inherited_fd" else None
        if extra is not None:
            fds.append(extra)
        try:
            roles["broker"] = _spawn(_broker, root, broker_w, data_r, fault, extra)
            os.close(broker_w); fds.remove(broker_w)
            os.close(data_r); fds.remove(data_r)
            deadline = _now() + (150_000_000 if fault == "timeout" else 1_000_000_000)
            phase, group = _line(broker_r, deadline)
            if (phase, group) != ("HOLD", expected):
                raise Hold("hold-or-cgroup-mismatch")
            phases.append(phase)
            if fault == "after_hold":
                raise Hold("injected-after-hold")
            roles["reader"] = _spawn(_reader, data_w)
            os.close(data_w); fds.remove(data_w)
            phase, group = _line(broker_r, deadline)
            if (phase, group) != ("RECEIPT", expected):
                raise Hold("receipt-or-cgroup-mismatch")
            phases.append(phase)
            if fault == "after_receipt":
                raise Hold("injected-after-receipt")
            seal_r, seal_w = os.pipe2(os.O_CLOEXEC)
            fds.extend((seal_r, seal_w))
            roles["sealer"] = _spawn(_sealer, root, seal_w, fault)
            os.close(seal_w); fds.remove(seal_w)
            phase, group = _line(seal_r, deadline)
            if (phase, group) != ("SEAL", expected):
                raise Hold("seal-or-cgroup-mismatch")
            phases.append(phase)
            if fault == "after_seal":
                raise Hold("injected-after-seal")
            _expect_eof(seal_r, deadline)
            _expect_eof(broker_r, deadline)
        except (Hold, OSError, ValueError) as error:
            reason = str(error)
        finally:
            for fd in fds:
                os.close(fd)
            peaks, unresolved = _reap(roles, reason != "generated-order-observed")
    if unresolved:
        reason = "actor-closure-unverified"
    elif reason == "generated-order-observed" and (
            set(peaks) != {"broker", "reader", "sealer"} or
            any(item["exit_code"] != 0 for item in peaks.values())):
        reason = "actor-exit-unverified"
    return {"reason": reason, "phases": phases, "roles": peaks,
            "unresolved": unresolved, "uid_transition": "STUB-SAME-UID",
            "protected_channel": "STUB-SAME-UID",
            "private_cgroup": "HOLD", "v3_signatures": "HOLD"}


def probe(fault=None):
    """No source path or real-read mode is accepted."""
    if fault not in FAULTS:
        raise ValueError("unknown-generated-fault")
    start = _now()
    read_fd, write_fd = os.pipe2(os.O_CLOEXEC)
    pid = os.fork()
    if pid == 0:
        os.close(read_fd)
        os.setpgid(0, 0)
        try:
            body = _setup(fault)
            os.write(write_fd, json.dumps(body, sort_keys=True).encode() + b"\n")
            os._exit(0)
        except BaseException:
            os._exit(9)
    os.close(write_fd)
    data = bytearray()
    reaped = None
    eof = False
    deadline = start + 3_000_000_000
    while _now() < deadline and (reaped is None or not eof):
        ready, _, _ = select.select([read_fd], [], [], 0.01)
        if ready:
            block = os.read(read_fd, MAX_REPORT + 1)
            if block:
                data.extend(block)
            else:
                eof = True
        if reaped is None:
            waited, status, usage = os.wait4(pid, os.WNOHANG)
            if waited:
                reaped = (os.waitstatus_to_exitcode(status), usage.ru_maxrss * 1024)
        if len(data) > MAX_REPORT:
            break
    if reaped is None:
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    if reaped is None:
        cleanup_end = _now() + 1_000_000_000
        while _now() < cleanup_end:
            waited, status, usage = os.wait4(pid, os.WNOHANG)
            if waited:
                reaped = (os.waitstatus_to_exitcode(status), usage.ru_maxrss * 1024)
                break
            time.sleep(0.005)
    os.close(read_fd)
    end = _now()
    result = {"schema": SCHEMA, "status": "HOLD", "resource_proof": "HOLD",
              "protected_pre_read_custody": "HOLD", "source_opened": False,
              "elapsed_boottime_ns": end - start,
              "setup_peak_rss_bytes": reaped[1] if reaped is not None else None}
    if reaped is None or not eof or reaped[0] != 0 or len(data) > MAX_REPORT:
        result["reason"] = "setup-or-report-closure-unverified"
        return result
    try:
        body = json.loads(data, object_pairs_hook=_unique_object)
        if not isinstance(body, dict) or set(body) != {
                "reason", "phases", "roles", "unresolved", "uid_transition",
                "protected_channel", "private_cgroup", "v3_signatures"}:
            raise ValueError
        if (not isinstance(body["reason"], str) or
                not isinstance(body["phases"], list) or
                any(phase not in ("HOLD", "RECEIPT", "SEAL") for phase in body["phases"]) or
                not isinstance(body["roles"], dict) or
                not set(body["roles"]) <= {"broker", "reader", "sealer"} or
                not isinstance(body["unresolved"], list) or
                body["uid_transition"] != "STUB-SAME-UID" or
                body["protected_channel"] != "STUB-SAME-UID" or
                body["private_cgroup"] != "HOLD" or body["v3_signatures"] != "HOLD"):
            raise ValueError
        for item in body["roles"].values():
            if (not isinstance(item, dict) or set(item) != {"exit_code", "peak_rss_bytes"}
                    or type(item["exit_code"]) is not int
                    or type(item["peak_rss_bytes"]) is not int
                    or item["peak_rss_bytes"] < 0):
                raise ValueError
    except (ValueError, UnicodeDecodeError, TypeError):
        result["reason"] = "setup-report-unverified"
        return result
    result.update(body)
    complete = (body["reason"] == "generated-order-observed" and
                body["phases"] == ["HOLD", "RECEIPT", "SEAL"] and
                set(body["roles"]) == {"broker", "reader", "sealer"} and
                body["unresolved"] == [] and
                all(item["exit_code"] == 0 for item in body["roles"].values()))
    if complete:
        result["observed_peak_sum_bytes"] = reaped[1] + sum(
            item["peak_rss_bytes"] for item in body["roles"].values())
    elif body["reason"] == "generated-order-observed":
        result["reason"] = "finite-graph-unverified"
    return result
