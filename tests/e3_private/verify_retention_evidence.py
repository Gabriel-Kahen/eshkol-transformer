"""Check the complete E3 retention run's measured records as one evidence set.

The compiled witness owns the in-call assertions. This checker rejects missing,
malformed, or mutually inconsistent records after all 20 processes exit.
"""

from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
from pathlib import Path
import re


HORIZONS = (1024, 8192)
GRADIENTS = ("absent", "present")
PROFILES = ("normal-a", "normal-b", "poisoned-a", "poisoned-b", "sanitized")
NUMBER = r"(?:0|[1-9][0-9]*)"
FIELDS = (
    "arena_used", "arena_total", "arena_blocks", "caller_region_used",
    "failure_region_used", "retry_region_used", "borrow_begins",
    "borrow_views", "retired_borrows", "retired_control_bytes",
    "d2_live_allocations", "fd_peak_growth",
)
RESULT = re.compile(
    r"E3-COMPLETE-CALL-RETENTION-PASS horizon=(1024|8192) "
    r"gradient=(absent|present) "
    + " ".join(f"{name}=({NUMBER})" for name in FIELDS)
    + r"\n\Z"
)
RSS = re.compile(rf"rss_before_kib=({NUMBER}) rss_after_kib=({NUMBER})\n\Z")
TIME = re.compile(rf"elapsed_seconds=([0-9]+(?:\.[0-9]+)?) peak_rss_kib=({NUMBER})\n\Z")


def _read(path: Path) -> str:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 2048:
        raise ValueError(f"missing, linked, or oversized retention record: {path}")
    try:
        return path.read_bytes().decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError(f"non-ASCII retention record: {path}") from error


def verify(directory: Path) -> dict[str, int]:
    records: dict[tuple[int, str], dict[str, int]] = {}
    control_bytes: int | None = None
    maximum_peak = 0
    for horizon in HORIZONS:
        for gradient in GRADIENTS:
            baseline: dict[str, int] | None = None
            for profile in PROFILES:
                stem = directory / f"{horizon}-{gradient}-{profile}"
                match = RESULT.fullmatch(_read(Path(f"{stem}.stdout")))
                if match is None or int(match[1]) != horizon or match[2] != gradient:
                    raise ValueError(f"malformed or misidentified retention result: {stem}")
                values = dict(zip(FIELDS, map(int, match.groups()[2:]), strict=True))
                if baseline is None:
                    baseline = values
                elif values != baseline:
                    raise ValueError(f"different retention result for one horizon: {stem}")

                rss = RSS.fullmatch(_read(Path(f"{stem}.rss")))
                timing = TIME.fullmatch(_read(Path(f"{stem}.time")))
                if rss is None or timing is None:
                    raise ValueError(f"malformed RSS or timing record: {stem}")
                before, after = map(int, rss.groups())
                peak = int(timing[2])
                try:
                    elapsed = Decimal(timing[1])
                except InvalidOperation as error:
                    raise ValueError(f"invalid elapsed time: {stem}") from error
                if min(before, after, peak) <= 0 or peak < max(before, after) or elapsed <= 0:
                    raise ValueError(f"inconsistent process resource measurement: {stem}")
                if _read(Path(f"{stem}.stderr")):
                    raise ValueError(f"retention runtime wrote stderr: {stem}")
                maximum_peak = max(maximum_peak, peak)

            assert baseline is not None
            if baseline["arena_used"] > baseline["arena_total"] or baseline["arena_blocks"] == 0:
                raise ValueError(f"invalid retained arena census: {horizon}/{gradient}")
            if (baseline["caller_region_used"] != baseline["retry_region_used"]
                    or max(baseline["caller_region_used"], baseline["failure_region_used"]) > 262144):
                raise ValueError(f"unreconciled caller region: {horizon}/{gradient}")
            begins = baseline["borrow_begins"]
            if (begins == 0 or baseline["borrow_views"] < begins
                    or baseline["retired_borrows"] != begins
                    or baseline["retired_control_bytes"] == 0
                    or baseline["retired_control_bytes"] % begins):
                raise ValueError(f"unreconciled I2 borrow controls: {horizon}/{gradient}")
            bytes_per_control = baseline["retired_control_bytes"] // begins
            if control_bytes is None:
                control_bytes = bytes_per_control
            elif bytes_per_control != control_bytes:
                raise ValueError("retired I2 control size changes across runs")
            records[horizon, gradient] = baseline

    for gradient in GRADIENTS:
        short = records[1024, gradient]
        long = records[8192, gradient]
        for field in ("arena_used", "arena_total", "arena_blocks", "caller_region_used",
                      "retry_region_used", "d2_live_allocations"):
            if short[field] != long[field]:
                raise ValueError(f"retained baseline changes with horizon: {gradient}/{field}")
        if long["borrow_begins"] <= short["borrow_begins"]:
            raise ValueError(f"long horizon did not perform additional I2 borrows: {gradient}")
    return {"runs": 20, "retired_control_bytes_each": control_bytes or 0,
            "maximum_peak_rss_kib": maximum_peak}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    args = parser.parse_args()
    result = verify(args.directory)
    print("E3 retention evidence verified: " + " ".join(
        f"{key}={value}" for key, value in result.items()))


if __name__ == "__main__":
    main()
