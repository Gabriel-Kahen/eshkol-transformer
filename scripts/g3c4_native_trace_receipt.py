"""Offline structural check and receipt correlation for native exec logs."""
from collections import Counter, defaultdict
from pathlib import Path
import re

HEADER = "g3c4-native-exec-v1"
MAX_LOG = 32 * 1024 * 1024
HEX = re.compile(r"(?:[0-9a-f]{2})+")
SHA256 = re.compile(r"[0-9a-f]{64}")


def require(condition, message):
    if not condition:
        raise ValueError(f"native trace: {message}")


def number(value, label, *, minimum=0):
    require(value.isascii() and value.isdecimal(), f"bad {label}")
    parsed = int(value)
    require(parsed >= minimum, f"bad {label}")
    return parsed


def hex_bytes(value, label):
    require(bool(HEX.fullmatch(value)), f"bad {label}")
    return bytes.fromhex(value)


def verify(log_path: Path, receipts: list[tuple[bytes, int, int]]) -> dict:
    require(log_path.is_file() and not log_path.is_symlink(), "missing or linked log")
    require(log_path.stat().st_size <= MAX_LOG, "log cap exceeded")
    raw = log_path.read_bytes()
    require(len(raw) <= MAX_LOG, "log cap exceeded")
    require(raw.endswith(b"\n"), "incomplete final record")
    try:
        lines = raw.decode("ascii").split("\n")[:-1]
    except UnicodeDecodeError as exc:
        raise ValueError("native trace: non-ASCII log") from exc
    require(lines and lines[0] == HEADER, "bad header")
    live = {}
    counts = Counter()
    execs = defaultdict(list)
    root = None
    root_status = None
    last_time = 0
    require(len(lines) >= 3, "missing exec or completion")
    for index, line in enumerate(lines[1:], 1):
        fields = line.split("\t")
        kind = fields[0]
        require(kind in {"spawn", "exec", "exit", "merge", "complete"},
                f"unknown event at line {index + 1}")
        widths = {"spawn": 5, "exec": 11, "exit": 5, "merge": 4, "complete": 8}
        require(len(fields) == widths[kind], f"bad {kind} width")
        when = number(fields[1], "event time", minimum=1)
        require(when >= last_time, "event time reversed")
        last_time = when
        if kind == "complete":
            require(index == len(lines) - 1, "completion is not last")
            require(root is not None and not live, "live task at completion")
            require(number(fields[2], "root pid", minimum=1) == root and
                    number(fields[3], "root status") == root_status == 0,
                    "root completion mismatch")
            require([number(x, "event count") for x in fields[4:]] ==
                    [counts[x] for x in ("spawn", "exec", "exit", "merge")],
                    "event count mismatch")
            require(counts["spawn"] + 1 == counts["exit"] + counts["merge"],
                    "task ledger mismatch")
            continue
        require(index < len(lines) - 1, "missing completion")
        if kind == "spawn":
            parent = number(fields[2], "spawn parent", minimum=1)
            child = number(fields[3], "spawn child", minimum=1)
            require(root is not None and parent in live and child not in live and
                    child != parent, "invalid spawn lineage")
            require(fields[4] in {"fork", "vfork", "clone"}, "bad spawn kind")
            live[child] = parent
        elif kind == "exec":
            pid = number(fields[2], "exec pid", minimum=1)
            parent = number(fields[3], "exec parent")
            if root is None:
                require(index == 1 and parent == 0, "root exec missing")
                root = pid
                live[pid] = 0
            require(live.get(pid) == parent, "invalid exec lineage")
            for value, label in zip(fields[4:7], ("start ticks", "device", "inode")):
                number(value, label, minimum=1)
            argv = hex_bytes(fields[7], "exec argv")
            require(argv.endswith(b"\0") and argv != b"\0", "bad exec argv")
            require(hex_bytes(fields[8], "exec path").startswith(b"/"),
                    "bad exec path")
            require(hex_bytes(fields[9], "exec cwd").startswith(b"/"),
                    "bad exec cwd")
            require(bool(SHA256.fullmatch(fields[10])), "bad environment hash")
            execs[argv].append(when)
        elif kind == "exit":
            pid = number(fields[2], "exit pid", minimum=1)
            signal = number(fields[3], "exit signal")
            status = number(fields[4], "exit status")
            require(pid in live and signal <= 64 and status <= 255 and
                    (signal == 0 or status == 128 + signal), "invalid task exit")
            if pid == root:
                root_status = status
            del live[pid]
        else:
            former = number(fields[2], "former tid", minimum=1)
            pid = number(fields[3], "merged pid", minimum=1)
            require(former != pid and former in live and pid in live,
                    "invalid thread merge")
            del live[former]
        counts[kind] += 1
    require(lines[-1].startswith("complete\t"), "missing completion")
    unmatched = sum(map(len, execs.values()))
    for argv, start, end in sorted(receipts, key=lambda item: item[2]):
        require(isinstance(argv, bytes) and argv.endswith(b"\0") and
                type(start) is int and type(end) is int and 0 <= start <= end,
                "bad receipt interval or argv")
        times = execs.get(argv, [])
        match = next((i for i, when in enumerate(times) if start <= when <= end), None)
        require(match is not None, "command receipt has no traced exec")
        times.pop(match)
        unmatched -= 1
    return {"native_execs": counts["exec"], "matched_receipts": len(receipts),
            "unmatched_native_execs": unmatched, "native_spawns": counts["spawn"]}
