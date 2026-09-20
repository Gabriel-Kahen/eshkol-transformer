"""Report measured cumulative retention; never treat release as flat-memory proof."""
import csv
from pathlib import Path
import sys

MODES = {"forward", "logits", "vjp", "reset", "failure"}


def check(path):
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != ["mode", "horizon", "arena_bytes", "peak_rss_kib"]:
            raise ValueError("retention report columns differ")
        rows = {}
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("retention row width differs")
            mode = row["mode"]
            numbers = [row[name] for name in ("horizon", "arena_bytes", "peak_rss_kib")]
            if mode not in MODES or any(not n or not n.isascii() or not n.isdecimal() for n in numbers):
                raise ValueError("invalid retention row")
            horizon, arena, rss = map(int, numbers)
            key = mode, horizon
            if horizon not in (1024, 8192) or key in rows or arena <= 0 or rss <= 0:
                raise ValueError("invalid or duplicate retention measurement")
            rows[key] = arena, rss
    if set(rows) != {(mode, count) for mode in MODES for count in (1024, 8192)}:
        raise ValueError("retention measurement missing")
    for mode in sorted(MODES):
        small, small_rss = rows[mode, 1024]
        large, large_rss = rows[mode, 8192]
        if large < small:
            raise ValueError(f"{mode}: cumulative arena accounting decreased")
        print(f"M3 retention {mode}: arena1024={small} arena8192={large} "
              f"bytes_per_additional_iteration={(large-small)/7168:.6f} "
              f"peak_rss_kib1024={small_rss} peak_rss_kib8192={large_rss}")
    print("M3 retained identity and plan costs are cumulative; no flat training-memory claim.")


if __name__ == "__main__":
    check(sys.argv[1])
