#!/usr/bin/env python3
"""Verify v2 calibration events, including expected incomplete failure logs."""
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(os.environ.get("G3C4_TRACE_SOURCE", "/workspace")) / "scripts"))
from g3c4_native_trace_receipt import verify


def require(value, reason):
    if not value:
        raise ValueError(reason)


def events(path):
    lines = path.read_text(encoding="ascii").splitlines()
    require(lines and lines[0] == "g3c4-native-exec-v2", "bad v2 header")
    return [line.split("\t") for line in lines[1:]]


def audit(root):
    require(root.is_dir() and not root.is_symlink(), "bad output root")
    report = {"status": "V2_CALIBRATION_ONLY_UNADMITTED", "cases": {}, "binaries": {}}
    for name in ("trace", "canary"):
        path = root / "bin" / name
        require(path.is_file() and not path.is_symlink(), "missing compiled canary")
        report["binaries"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ("fast", "spawn", "thread", "threadexec"):
        require((root / f"{name}.exit").read_text() == "0\n", f"{name} exit")
        summary = verify(root / f"{name}.events", [])
        data = events(root / f"{name}.events")
        require(data[-1][0] == "complete", f"{name} completion")
        if name == "fast":
            require((summary["native_spawns"], summary["native_execs"]) == (32, 33),
                    "fast fork/exec coverage")
            require(all(event[4] == "fork" and event[5] == event[3]
                        for event in data if event[0] == "spawn"), "fast fork groups")
            children = {event[3] for event in data if event[0] == "spawn"}
            require(children == {event[2] for event in data[1:] if event[0] == "exec"},
                    "fast child exec coverage")
            require(all(bytes.fromhex(event[7]) == b"/bin/true\0" for event in data[1:]
                        if event[0] == "exec"), "fast child image differs")
        elif name == "spawn":
            require(summary["native_spawns"] >= 1 and summary["native_execs"] >= 2,
                    "posix_spawn descendant absent")
            require(any(event[0] == "exec" and
                        bytes.fromhex(event[7]) == b"/bin/true\0" for event in data),
                    "posix_spawn child image differs")
        elif name == "thread":
            require((summary["native_spawns"], summary["native_execs"]) == (1, 1) and
                    any(event[0] == "spawn" and event[4] == "clone" and
                        event[5] == event[2] for event in data), "thread clone group absent")
        else:
            require((summary["native_spawns"], summary["native_execs"]) == (1, 2) and
                    any(event[0] == "merge" for event in data),
                    "nonleader exec/merge absent")
        report["cases"][name] = summary
    for name, reason in (("fault", "injected supervisor failure"),
                         ("untraced", "forbidden clone route")):
        require((root / f"{name}.exit").read_text() == "86\n", f"{name} did not HOLD")
        require(not (root / f"{name}-unwanted").exists(), f"{name} child escaped")
        data = events(root / f"{name}.events")
        require(data and not any(event[0] == "complete" for event in data),
                f"{name} wrote success footer")
        require(reason.encode() in (root / f"{name}.stderr").read_bytes(),
                f"{name} failed for another reason")
        try:
            verify(root / f"{name}.events", [])
        except ValueError:
            pass
        else:
            raise ValueError(f"{name} incomplete log was accepted")
        report["cases"][name] = {"status": "EXPECTED_HOLD", "events": len(data)}
    return report


def main():
    if len(sys.argv) != 2:
        raise SystemExit(86)
    try:
        print(json.dumps(audit(Path(sys.argv[1])), sort_keys=True))
    except (OSError, ValueError) as exc:
        print(f"v2 calibration HOLD: {exc}", file=sys.stderr)
        raise SystemExit(86)


if __name__ == "__main__":
    main()
