#!/usr/bin/env python3
"""Sample RSS of the two P1 compiler process groups; development evidence only."""

import datetime
import json
import os
from pathlib import Path
import signal
import sys
import time

INTERVAL_SECONDS = 0.25
PAGE_BYTES = os.sysconf("SC_PAGE_SIZE")
stopping = False


def process_group(pid: int) -> int | None:
    try:
        raw = (Path("/proc") / str(pid) / "stat").read_bytes()
    except (FileNotFoundError, ProcessLookupError):
        return None
    # The command name in parentheses can itself contain spaces or parentheses.
    end = raw.rfind(b") ")
    if end < 0:
        raise RuntimeError("malformed /proc stat")
    return int(raw[end + 2:].split()[2])


def sample(groups: tuple[int, int]) -> tuple[list[int], list[int]]:
    rss = [0, 0]
    counts = [0, 0]
    with os.scandir("/proc") as entries:
        for entry in entries:
            if not entry.name.isdecimal():
                continue
            pid = int(entry.name)
            group = process_group(pid)
            if group not in groups:
                continue
            try:
                pages = int((Path("/proc") / entry.name / "statm").read_text().split()[1])
            except (FileNotFoundError, ProcessLookupError):
                continue  # The process exited between stat and statm.
            index = groups.index(group)
            rss[index] += pages * PAGE_BYTES
            counts[index] += 1
    return rss, counts


def check() -> None:
    if PAGE_BYTES <= 0 or process_group(os.getpid()) != os.getpgrp():
        raise RuntimeError("Linux /proc process-group RSS sampling unavailable")
    sample((os.getpgrp(), -1))


def stop(_signum, _frame) -> None:
    global stopping
    stopping = True


def monitor(path: Path, groups: tuple[int, int]) -> None:
    check()
    if len(set(groups)) != 2 or any(group <= 0 for group in groups):
        raise ValueError("two distinct positive process groups required")
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    samples = both_active = max_rss = 0
    with path.open("x", encoding="ascii") as output:
        output.write("utc\tmonotonic_ns\tpgid_first\tpgid_second\t"
                     "rss_first_bytes\trss_second_bytes\trss_sum_bytes\t"
                     "processes_first\tprocesses_second\n")
        while True:
            rss, counts = sample(groups)
            now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds")
            output.write(f"{now}\t{time.monotonic_ns()}\t{groups[0]}\t{groups[1]}\t"
                         f"{rss[0]}\t{rss[1]}\t{sum(rss)}\t{counts[0]}\t{counts[1]}\n")
            output.flush()
            samples += 1
            both_active += int(all(counts))
            max_rss = max(max_rss, sum(rss))
            if stopping:
                break
            time.sleep(INTERVAL_SECONDS)
    summary = {"scope": "sampled RSS of two P1 compiler process groups only",
               "interval_seconds": INTERVAL_SECONDS, "samples": samples,
               "both_active_samples": both_active,
               "max_sampled_pair_rss_bytes": max_rss,
               "note": "Sampling can miss short peaks; no cgroup or hosted memory claim."}
    path.with_suffix(".summary.json").write_text(json.dumps(summary, sort_keys=True) + "\n")


def main(argv: list[str]) -> None:
    if argv == ["--check"]:
        check()
    elif len(argv) == 3:
        monitor(Path(argv[0]), (int(argv[1]), int(argv[2])))
    else:
        raise SystemExit("usage: p1-pair-memory.py --check | TRACE.tsv PGID_FIRST PGID_SECOND")


if __name__ == "__main__":
    main(sys.argv[1:])
