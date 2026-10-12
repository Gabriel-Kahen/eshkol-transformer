#!/usr/bin/env python3
"""Opt-in raw command receipt; outputs are replayed while they are captured."""
import hashlib
import json
import os
import selectors
import subprocess
import sys
import tempfile
import time
from pathlib import Path

STREAM_LIMIT = 16 * 1024 * 1024
COMMAND_LIMIT_NS = 7200 * 1_000_000_000


def run(root: Path, argv: list[str]) -> int:
    if not argv or not root.is_dir() or root.is_symlink():
        return 86
    try:
        requested_ms = int(os.environ.get("G3C4_PROOF_COMMAND_LIMIT_MS", "7200000"))
    except ValueError:
        return 86
    if requested_ms <= 0 or requested_ms > 7_200_000:
        return 86
    command_limit_ns = min(COMMAND_LIMIT_NS, requested_ms * 1_000_000)
    receipt = Path(tempfile.mkdtemp(prefix="command.", dir=root))
    start_ns = time.monotonic_ns()
    (receipt / "cwd").write_bytes(os.fsencode(os.getcwd()) + b"\n")
    (receipt / "argv.nul").write_bytes(b"\0".join(os.fsencode(x) for x in argv) + b"\0")
    (receipt / "environment.sha256").write_text(hashlib.sha256(
        b"\0".join(os.fsencode(k + "=" + v) for k, v in sorted(os.environ.items()))
    ).hexdigest() + "\n")
    streams = [open(receipt / "stdout.raw", "wb"), open(receipt / "stderr.raw", "wb")]
    sizes = [0, 0]
    hashes = [hashlib.sha256(), hashlib.sha256()]
    overflow = False
    child = None
    status = 86
    try:
        # Preserve caller-owned marker fds, including fd 3 used by overlays.
        child = subprocess.Popen(argv, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, close_fds=False)
        selector = selectors.DefaultSelector()
        for i, pipe in enumerate((child.stdout, child.stderr)):
            os.set_blocking(pipe.fileno(), False)
            selector.register(pipe, selectors.EVENT_READ, i)
        while selector.get_map():
            if time.monotonic_ns() - start_ns > command_limit_ns:
                overflow = True
                break
            for key, _ in selector.select(timeout=0.25):
                i = key.data
                chunk = os.read(key.fileobj.fileno(), 65536)
                if not chunk:
                    selector.unregister(key.fileobj)
                    key.fileobj.close()
                    continue
                remaining = STREAM_LIMIT - sizes[i]
                kept = chunk[:max(remaining, 0)]
                if kept:
                    streams[i].write(kept)
                    hashes[i].update(kept)
                    sizes[i] += len(kept)
                    view = memoryview(kept)
                    try:
                        while view:
                            written = os.write(1 + i, view)
                            if written == 0:
                                raise BrokenPipeError("zero-byte replay")
                            view = view[written:]
                    except OSError:
                        overflow = True
                if len(kept) != len(chunk):
                    overflow = True
            if overflow:
                if child.poll() is None:
                    child.kill()
                break
        for key in list(selector.get_map().values()):
            selector.unregister(key.fileobj)
            key.fileobj.close()
        selector.close()
        if overflow and child.poll() is None:
            child.kill()
        status = child.wait()
        if status < 0:
            status = 128 - status
        if overflow:
            status = 86
    except OSError as exc:
        (receipt / "spawn-error").write_text(str(exc) + "\n")
        if child is not None and child.poll() is None:
            child.kill()
            child.wait()
    finally:
        for stream in streams:
            stream.close()
        end_ns = time.monotonic_ns()
        manifest = {
            "exit": status,
            "start_monotonic_ns": start_ns,
            "end_monotonic_ns": end_ns,
            "wall_ns": end_ns - start_ns,
            "stdout_bytes": sizes[0],
            "stderr_bytes": sizes[1],
            "stdout_sha256": hashes[0].hexdigest(),
            "stderr_sha256": hashes[1].hexdigest(),
            "argv_sha256": hashlib.sha256((receipt / "argv.nul").read_bytes()).hexdigest(),
            "script": os.environ.get("G3C4_PROOF_SCRIPT", ""),
            "site": os.environ.get("G3C4_PROOF_SITE", ""),
        }
        (receipt / "receipt.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
    return status


if __name__ == "__main__":
    if len(sys.argv) < 4 or sys.argv[2] != "--":
        raise SystemExit(86)
    raise SystemExit(run(Path(sys.argv[1]), sys.argv[3:]))
