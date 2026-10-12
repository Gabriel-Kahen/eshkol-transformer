#!/usr/bin/env python3
"""One-use, source-only HOLD capture for the small-metadata CLI."""

import argparse
import base64
import binascii
import hashlib
import json
import math
import os
import selectors
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import e3_n1_small_metadata as small

CLI = Path(__file__).with_name("e3_n1_small_metadata.py")
CAPTURE_RUN_ENABLED = False  # External deadline, whole-process RSS and custody pending.
STDOUT_CAP = 8 * 1024 * 1024
STDERR_CAP = 1024 * 1024
RSS_CAP = 268435456
SECONDS = 30
CHUNK = 65536


class CaptureError(Exception):
    pass


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode()


def sha(raw):
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _write(fd, raw):
    view = memoryview(raw)
    while view:
        count = os.write(fd, view)
        if count <= 0:
            raise OSError("short-write")
        view = view[count:]


def _new_file(directory, name):
    return os.open(name, os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW |
                   os.O_CLOEXEC | os.O_RDWR, 0o600, dir_fd=directory)


def _record(directory, name, value):
    body = dict(value)
    body["checksum_sha256"] = sha(canonical(value))
    raw = canonical(body)
    fd = _new_file(directory, name)
    try:
        _write(fd, raw)
        os.fsync(fd)
    finally:
        os.close(fd)
    return raw


def _destination(path):
    small.absolute(path)
    parts = path.split("/")[1:]
    parent = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY |
                            os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=parent)
            os.close(parent)
            parent = child
        os.mkdir(parts[-1], 0o700, dir_fd=parent)
        directory = os.open(parts[-1], os.O_RDONLY | os.O_DIRECTORY |
                            os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=parent)
        try:
            os.fsync(parent)
            return directory
        except OSError:
            os.close(directory)
            raise
    finally:
        os.close(parent)


def _kill_group(pid):
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _pump(command, directory, child_deadline, hard_deadline,
          stdout_cap, stderr_cap):
    streams = {}
    output = {}
    process = None
    selector = None
    status = usage = None
    reason = None
    try:
        for name, cap in (("stdout", stdout_cap), ("stderr", stderr_cap)):
            fd = _new_file(directory, name + ".bin")
            streams[name] = {"fd": fd, "cap": cap, "bytes": bytearray(),
                             "truncated": False}
        selector = selectors.DefaultSelector()
        if time.monotonic() >= child_deadline:
            raise CaptureError("deadline-before-child")
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   close_fds=True, start_new_session=True,
                                   env={"LC_ALL": "C", "TZ": "UTC", "PYTHONNOUSERSITE": "1"})
        selector.register(process.stdout, selectors.EVENT_READ, "stdout")
        selector.register(process.stderr, selectors.EVENT_READ, "stderr")
        while True:
            if status is None:
                pid, observed, resource = os.wait4(process.pid, os.WNOHANG)
                if pid:
                    status, usage = observed, resource
                    process.returncode = os.waitstatus_to_exitcode(status)
            if not selector.get_map() and status is not None:
                break
            remaining = child_deadline - time.monotonic()
            if remaining <= 0 and reason is None:
                reason = "deadline"
                if status is None:
                    _kill_group(process.pid)
            hard_remaining = hard_deadline - time.monotonic()
            if hard_remaining <= 0:
                break
            for key, _ in selector.select(min(hard_remaining, 0.02)):
                stream = streams[key.data]
                remaining_bytes = stream["cap"] - len(stream["bytes"])
                block = os.read(key.fileobj.fileno(), min(CHUNK, remaining_bytes + 1))
                if not block:
                    selector.unregister(key.fileobj)
                    continue
                retained = block[:remaining_bytes]
                _write(stream["fd"], retained)
                stream["bytes"].extend(retained)
                if len(block) > remaining_bytes:
                    stream["truncated"] = True
                    reason = reason or key.data + "-cap"
                    if status is None:
                        _kill_group(process.pid)
        if status is None:
            reason = (reason + "-child-not-reaped") if reason else "child-not-reaped"
            _kill_group(process.pid)
        for name, stream in streams.items():
            os.fsync(stream["fd"])
            output[name] = {"size": len(stream["bytes"]),
                            "sha256": sha(stream["bytes"]),
                            "truncated": stream["truncated"]}
        peak_rss = None if usage is None else usage.ru_maxrss * 1024
        return (process.returncode, status is not None, peak_rss, reason,
                bytes(streams["stdout"]["bytes"]), output)
    finally:
        if selector is not None:
            selector.close()
        if process is not None:
            if status is None:
                _kill_group(process.pid)
                try:
                    pid, observed, resource = os.wait4(process.pid, os.WNOHANG)
                    if pid:
                        process.returncode = os.waitstatus_to_exitcode(observed)
                except ChildProcessError:
                    pass
            process.stdout.close()
            process.stderr.close()
        for stream in streams.values():
            os.close(stream["fd"])


def _validate(raw, request_sha256, index_digest):
    try:
        record = small.parse(raw, require_canonical=True)
        small.fields(record, ("schema", "status", "stage", "candidate",
                              "request_sha256", "raw_sha256", "raw_size",
                              "raw_base64", "source_path", "read_time_identity",
                              "claims", "limits", "scope", "checksum_sha256"),
                     "evidence-fields")
        small.need(record["schema"] == "e3-n1-small-metadata-v1" and
                   record["status"] == "HOLD" and record["stage"] == "index" and
                   record["request_sha256"] == request_sha256 and
                   record["candidate"] == {"name": small.F32_NAME,
                                           "index_digest": index_digest} and
                   record["raw_sha256"] == index_digest and
                   record["scope"] == "small-json-read-only;no-layer-or-custody-authority",
                   "evidence-identity")
        small.absolute(record["source_path"])
        small.evidence_identity(record["read_time_identity"], record["source_path"])
        small.fields(record["limits"], ("max_bytes", "max_seconds", "max_rss_bytes"),
                     "evidence-limits")
        small.positive(record["limits"]["max_bytes"], 4096, "evidence-byte-limit")
        small.positive(record["limits"]["max_seconds"], SECONDS,
                       "evidence-time-limit")
        small.positive(record["limits"]["max_rss_bytes"], RSS_CAP,
                       "evidence-rss-limit")
        checksum = record.pop("checksum_sha256")
        small.need(checksum == sha(canonical(record)), "evidence-checksum")
        encoded = record["raw_base64"]
        small.need(type(encoded) is str, "raw-encoding")
        content = base64.b64decode(encoded, validate=True)
        small.need(type(record["raw_size"]) is int and
                   base64.b64encode(content).decode() == encoded and
                   len(content) == record["raw_size"] and
                   len(content) <= record["limits"]["max_bytes"] and
                   record["read_time_identity"]["file"]["size"] == len(content) and
                   sha(content) == index_digest,
                   "raw-drift")
        small.need(record["claims"] ==
                   {"selected_manifest": small.selected_manifest(small.parse(content))},
                   "index-claims")
    except (small.Hold, ValueError, TypeError, KeyError, binascii.Error) as error:
        raise CaptureError("malformed-child-evidence") from error


def _seal_result(directory, result):
    raw = _record(directory, "result.json.partial", result)
    os.link("result.json.partial", "result.json", src_dir_fd=directory,
            dst_dir_fd=directory, follow_symlinks=False)
    os.fsync(directory)
    return raw


def capture(command, destination, request_sha256, index_digest,
            *, seconds=SECONDS, stdout_cap=STDOUT_CAP, stderr_cap=STDERR_CAP,
            rss_cap=RSS_CAP):
    """Retain one bounded attempt; every result is HOLD, never custody authority."""
    if not hasattr(os, "wait4"):
        raise CaptureError("child-rss-measurement-unsupported")
    if not (type(seconds) in (int, float) and math.isfinite(seconds) and
            type(stdout_cap) is int and type(stderr_cap) is int and
            type(rss_cap) is int and
            0 < seconds <= SECONDS and 0 < stdout_cap <= STDOUT_CAP and
            0 < stderr_cap <= STDERR_CAP and 0 < rss_cap <= RSS_CAP):
        raise CaptureError("capture-limits")
    small.digest(request_sha256)
    small.digest(index_digest)
    if (threading.current_thread() is not threading.main_thread() or
            not hasattr(signal, "setitimer") or
            signal.getitimer(signal.ITIMER_REAL)[0] != 0):
        raise CaptureError("hard-deadline-unsupported")

    def expired(_signum, _frame):
        raise CaptureError("deadline")

    previous = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return _capture(command, destination, request_sha256, index_digest,
                        seconds, stdout_cap, stderr_cap, rss_cap)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _capture(command, destination, request_sha256, index_digest,
             seconds, stdout_cap, stderr_cap, rss_cap):
    started_ns = time.time_ns()
    started = time.monotonic()
    deadline = started + seconds
    directory = _destination(str(destination))
    try:
        _record(directory, "intent.json", {"schema": "e3-n1-small-capture-intent-v1",
                   "status": "HOLD", "request_sha256": request_sha256,
                   "started_ns": started_ns,
                   "limits": {"seconds": seconds, "stdout_bytes": stdout_cap,
                              "stderr_bytes": stderr_cap, "child_rss_bytes": rss_cap}})
        os.fsync(directory)
        reserve = min(0.2, seconds / 4)
        exit_code, child_reaped, peak_rss, reason, stdout, output = _pump(
            command, directory, deadline - reserve, deadline,
            stdout_cap, stderr_cap)
        if reason is None and exit_code != 0:
            reason = "child-exit"
        if reason is None and peak_rss is None:
            reason = "child-rss-unobserved"
        if reason is None and peak_rss > rss_cap:
            reason = "child-rss-cap"
        if reason is None:
            try:
                _validate(stdout, request_sha256, index_digest)
            except CaptureError:
                reason = "malformed-child-evidence"
        if time.monotonic() > deadline:
            reason = "deadline"
        result = {"schema": "e3-n1-small-capture-v1", "status": "HOLD",
                  "outcome": "captured-hold" if reason is None else "failed-hold",
                  "reason": reason, "request_sha256": request_sha256,
                  "index_digest": index_digest, "started_ns": started_ns,
                  "finished_ns": time.time_ns(),
                  "elapsed_ns": int((time.monotonic() - started) * 1e9),
                  "child_exit_code": exit_code, "child_reaped": child_reaped,
                  "child_peak_rss_bytes": peak_rss,
                  "streams": output, "scope": "same-uid-capture;no-custody-or-admission-authority"}
        _seal_result(directory, result)
        return result
    finally:
        os.close(directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request")
    parser.add_argument("fresh_capture_directory")
    args = parser.parse_args()
    try:
        if not CAPTURE_RUN_ENABLED:
            raise CaptureError("capture-procedure-not-admitted")
        small.absolute(args.request)
        result = capture([sys.executable, str(CLI), args.request],
                         args.fresh_capture_directory,
                         small.REVIEWED_REQUEST_SHA256, small.F32_INDEX)
    except (CaptureError, small.Hold, OSError) as error:
        print("HOLD: " + str(error), file=sys.stderr)
        return 1
    print("HOLD: " + result["outcome"] +
          ("" if result["reason"] is None else ":" + result["reason"]), file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
