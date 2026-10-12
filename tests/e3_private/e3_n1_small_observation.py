#!/usr/bin/env python3
"""Synthetic-only expected-actor diagnostics; never an acquisition proof."""

import argparse
import base64
import json
import os
import selectors
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import e3_n1_small_capture as capture

HERE = Path(__file__).resolve()
LIMIT_NS = 30_000_000_000
RSS_CAP = 268435456
PIPE_CAP = 65536
MODES = frozenset(("valid", "malformed", "failed", "timeout", "overflow",
                   "observer-delay"))


class Hold(Exception):
    pass


def _checked(raw, schema):
    try:
        value = json.loads(raw)
        if not isinstance(value, dict) or capture.canonical(value) != raw:
            raise ValueError("noncanonical")
        checksum = value.pop("checksum_sha256")
        if checksum != capture.sha(capture.canonical(value)) or value.get("schema") != schema:
            raise ValueError("checksum-or-schema")
        return value
    except (TypeError, ValueError, KeyError, UnicodeDecodeError) as error:
        raise Hold("malformed-report") from error


def _emit(value):
    record = dict(value)
    record["checksum_sha256"] = capture.sha(capture.canonical(value))
    sys.stdout.buffer.write(capture.canonical(record))
    sys.stdout.buffer.flush()


def _kill(pid):
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _run(argv, directory, prefix, deadline_ns):
    """One fixed child, bounded pipes and wait4; uncertainty never passes."""
    streams = {}
    selector = None
    process = None
    reaped = False
    usage = None
    status = None
    reason = None
    try:
        for name in ("stdout", "stderr"):
            streams[name] = {"fd": capture._new_file(directory, prefix + "." + name + ".bin"),
                             "raw": bytearray(), "truncated": False}
        selector = selectors.DefaultSelector()
        if time.monotonic_ns() >= deadline_ns:
            raise Hold("deadline-before-child")
        process = subprocess.Popen(argv, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   close_fds=True, start_new_session=True,
                                   env={"LC_ALL": "C", "TZ": "UTC", "PYTHONNOUSERSITE": "1"})
        selector.register(process.stdout, selectors.EVENT_READ, "stdout")
        selector.register(process.stderr, selectors.EVENT_READ, "stderr")
        while True:
            if not reaped:
                pid, observed, rusage = os.wait4(process.pid, os.WNOHANG)
                if pid:
                    reaped, status, usage = True, observed, rusage
                    process.returncode = os.waitstatus_to_exitcode(status)
            if reaped and not selector.get_map():
                break
            if time.monotonic_ns() >= deadline_ns:
                reason = "deadline"
                if not reaped:
                    _kill(process.pid)
                break
            for key, _ in selector.select(min(0.02, max(0, (deadline_ns - time.monotonic_ns()) / 1e9))):
                stream = streams[key.data]
                remaining = PIPE_CAP - len(stream["raw"])
                block = os.read(key.fileobj.fileno(), min(65536, remaining + 1))
                if not block:
                    selector.unregister(key.fileobj)
                    continue
                retained = block[:remaining]
                capture._write(stream["fd"], retained)
                stream["raw"].extend(retained)
                if len(block) > remaining:
                    stream["truncated"] = True
                    reason = key.data + "-cap"
                    if not reaped:
                        _kill(process.pid)
                    break
            if reason:
                break
        for stream in streams.values():
            os.fsync(stream["fd"])
        return {"reason": reason, "reaped": reaped,
                "exit_code": process.returncode,
                "peak_rss_bytes": None if usage is None else usage.ru_maxrss * 1024,
                "stdout": bytes(streams["stdout"]["raw"]),
                "streams": {name: {"size": len(stream["raw"]),
                                   "sha256": capture.sha(stream["raw"]),
                                   "truncated": stream["truncated"]}
                            for name, stream in streams.items()}}
    finally:
        if selector is not None:
            selector.close()
        if process is not None:
            if not reaped:
                _kill(process.pid)
                cleanup_deadline = time.monotonic() + 0.5
                while time.monotonic() < cleanup_deadline:
                    try:
                        pid, observed, _ = os.wait4(process.pid, os.WNOHANG)
                    except ChildProcessError:
                        break
                    if pid:
                        process.returncode = os.waitstatus_to_exitcode(observed)
                        break
                    time.sleep(0.01)
            process.stdout.close()
            process.stderr.close()
        for stream in streams.values():
            os.close(stream["fd"])


def _fixture(directory):
    manifest = {"mediaType": capture.small.MANIFEST_TYPE,
                "digest": capture.sha(b"synthetic manifest"), "size": 18,
                "platform": {"os": "linux", "architecture": "amd64"}}
    index = capture.canonical({"schemaVersion": 2,
                               "mediaType": capture.small.INDEX_TYPE,
                               "manifests": [manifest]})
    source = directory / "synthetic-index.bin"
    fd = os.open(source, os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_WRONLY, 0o600)
    try:
        capture._write(fd, index)
        os.fsync(fd)
    finally:
        os.close(fd)
    fd, ancestors = capture.small.open_nofollow(str(source))
    try:
        identity = {"file": capture.small.identity(os.fstat(fd)),
                    "ancestors": ancestors}
    finally:
        os.close(fd)
    request_sha = capture.sha(b"synthetic request")
    index_sha = capture.sha(index)
    body = {"schema": "e3-n1-small-metadata-v1", "status": "HOLD",
            "stage": "index", "candidate": {"name": capture.small.F32_NAME,
                                            "index_digest": index_sha},
            "request_sha256": request_sha, "raw_sha256": index_sha,
            "raw_size": len(index), "raw_base64": base64.b64encode(index).decode(),
            "source_path": str(source), "read_time_identity": identity,
            "claims": {"selected_manifest": manifest},
            "limits": {"max_bytes": 4096, "max_seconds": 30,
                       "max_rss_bytes": RSS_CAP},
            "scope": "small-json-read-only;no-layer-or-custody-authority"}
    body["checksum_sha256"] = capture.sha(capture.canonical(body))
    return request_sha, index_sha, capture.canonical(body)


def _child(mode, raw):
    if mode == "malformed":
        raw = b"{bad"
    elif mode == "overflow":
        raw = b"x" * 4096
    elif mode == "timeout":
        raw = b"partial"
    sys.stdout.buffer.write(raw)
    sys.stdout.buffer.flush()
    if mode == "timeout":
        time.sleep(2)
    return 7 if mode == "failed" else 0


def _worker(root, mode, deadline_ns, rss_cap):
    request_sha, index_sha, raw = _fixture(root)
    remaining = (deadline_ns - time.monotonic_ns()) / 1e9
    if remaining <= 0:
        raise Hold("deadline-before-capture")
    command = [sys.executable, "-I", "-B", str(HERE), "child", mode, raw.hex()]
    result = capture.capture(command, root / "capture", request_sha, index_sha,
                             seconds=min(remaining, 0.2 if mode == "timeout" else capture.SECONDS),
                             stdout_cap=64 if mode == "overflow" else capture.STDOUT_CAP,
                             rss_cap=rss_cap)
    _emit({"schema": "e3-n1-observation-worker-v1", "status": "HOLD",
           "child_peak_rss_bytes": result["child_peak_rss_bytes"],
           "child_reaped": result["child_reaped"],
           "child_exit_code": result["child_exit_code"],
           "capture_outcome": result["outcome"], "capture_reason": result["reason"]})


def _observer(root, mode, deadline_ns, rss_cap):
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        if mode == "observer-delay":
            time.sleep(2)
        command = [sys.executable, "-I", "-B", str(HERE), "worker",
                   str(root), mode, str(deadline_ns), str(rss_cap)]
        result = _run(command, directory, "worker", deadline_ns)
        worker = _checked(result["stdout"], "e3-n1-observation-worker-v1") if (
            result["reaped"] and result["exit_code"] == 0 and
            result["reason"] is None) else None
        _emit({"schema": "e3-n1-observation-observer-v1", "status": "HOLD",
               "worker_peak_rss_bytes": result["peak_rss_bytes"],
               "worker_reaped": result["reaped"],
               "worker_exit_code": result["exit_code"],
               "worker_reason": result["reason"], "worker": worker})
    finally:
        os.close(directory)


def observe_synthetic(destination, *, mode="valid", seconds=30,
                      rss_cap=RSS_CAP):
    """Fixed synthetic actors only. Retain diagnostics; proof always HOLD."""
    if not hasattr(os, "wait4") or sys.platform != "linux":
        raise Hold("wait4-unsupported")
    if mode not in MODES or type(seconds) not in (int, float) or not (0 < seconds <= 30):
        raise Hold("synthetic-contract")
    if type(rss_cap) is not int or not (0 < rss_cap <= RSS_CAP):
        raise Hold("rss-cap")
    raw_destination = str(destination)
    capture.small.absolute(raw_destination)
    if not raw_destination.startswith("/tmp/"):
        raise Hold("synthetic-tmp-only")
    started_ns = time.monotonic_ns()
    deadline_ns = started_ns + int(seconds * 1e9)
    root = Path(raw_destination)
    directory = capture._destination(raw_destination)
    try:
        capture._record(directory, "intent.json", {
            "schema": "e3-n1-observation-intent-v1", "status": "HOLD",
            "scope": "fixed-synthetic-only", "started_monotonic_ns": started_ns,
            "seconds_cap": seconds, "rss_cap_bytes": rss_cap})
        os.fsync(directory)
        command = [sys.executable, "-I", "-B", str(HERE), "observer",
                   str(root), mode, str(deadline_ns), str(rss_cap)]
        run = _run(command, directory, "observer", deadline_ns)
        ended_ns = time.monotonic_ns()
        report = _checked(run["stdout"], "e3-n1-observation-observer-v1") if (
            run["reaped"] and run["exit_code"] == 0 and run["reason"] is None) else None
        worker = None if report is None else report["worker"]
        child_peak = None if worker is None else worker["child_peak_rss_bytes"]
        worker_peak = None if report is None else report["worker_peak_rss_bytes"]
        observer_peak = run["peak_rss_bytes"]
        peaks = (observer_peak, worker_peak, child_peak)
        expected_complete = (mode == "valid" and report is not None and worker is not None and
                  report["worker_reaped"] is True and report["worker_exit_code"] == 0 and
                  report["worker_reason"] is None and worker["child_reaped"] is True and
                  worker["child_exit_code"] == 0 and
                  worker["capture_outcome"] == "captured-hold" and
                  worker["capture_reason"] is None)
        measured = (expected_complete and run["reaped"] and run["exit_code"] == 0 and
                    run["reason"] is None and ended_ns <= deadline_ns and
                    all(type(peak) is int and peak > 0 for peak in peaks))
        expected_sum = sum(peaks) if measured else None
        result = {"schema": "e3-n1-synthetic-observation-v1", "status": "HOLD",
                  "resource_proof": "HOLD",
                  "reason": "unconfined-process-set-and-unmeasured-finalization",
                  "expected_actors": ["observer", "worker", "direct-child"],
                  "closure": "source-assumption-only;not-runtime-confinement",
                  "started_monotonic_ns": started_ns, "ended_monotonic_ns": ended_ns,
                  "expected_actor_elapsed_ns": ended_ns - started_ns,
                  "elapsed_cap_ns": int(seconds * 1e9),
                  "peak_rss_bytes": {"observer": observer_peak, "worker": worker_peak,
                                     "direct_child": child_peak},
                  "expected_actor_peak_sum_bytes": expected_sum,
                  "nominal_expected_actor_caps_met": bool(measured and expected_sum <= rss_cap),
                  "rss_cap_bytes": rss_cap,
                  "observer_reaped": run["reaped"], "observer_exit_code": run["exit_code"],
                  "observer_reason": run["reason"],
                  "streams": run["streams"],
                  "preservation": {"status": "HOLD",
                                   "reason": "distinct-uid-protected-custody-unavailable",
                                   "interface": "reviewed-broker-intent-copy-seal-receipt-required"},
                  "scope": "synthetic-measurement;no-real-read-or-admission"}
        capture._seal_result(directory, result)
        return result
    finally:
        os.close(directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("observer", "worker", "child"))
    parser.add_argument("args", nargs="*")
    args = parser.parse_args()
    try:
        if args.role == "child" and len(args.args) == 2:
            mode, raw = args.args
            if mode not in MODES:
                raise Hold("mode")
            return _child(mode, bytes.fromhex(raw))
        if args.role in ("observer", "worker") and len(args.args) == 4:
            root, mode, deadline, rss = args.args
            if mode not in MODES:
                raise Hold("mode")
            if args.role == "observer":
                _observer(Path(root), mode, int(deadline), int(rss))
            else:
                _worker(Path(root), mode, int(deadline), int(rss))
            return 0
        raise Hold("role-arguments")
    except (Hold, capture.CaptureError, capture.small.Hold, OSError, ValueError) as error:
        print("HOLD: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
