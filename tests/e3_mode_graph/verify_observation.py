"""Check actual in-call mode masks and real graph registry event deltas."""

from __future__ import annotations

import argparse
from pathlib import Path


LABELS = (("success", 4), ("later-failure", 1), ("retry", 4))
ALL_EVAL = (1 << 17) - 1
MISSED_CHILD = ALL_EVAL ^ (1 << 2)


def verify(text: str, expected: str) -> None:
    rows = text.splitlines()
    offset = 0
    for label, count in LABELS:
        for ordinal in range(count):
            if offset >= len(rows):
                raise ValueError("in-call note missing")
            fields = rows[offset].split()
            offset += 1
            if len(fields) != 8 or fields[:3] != ["E3MG-NOTE", label,
                                                  str(ordinal)]:
                raise ValueError("in-call note identity differs")
            mask, attempts, enrollments, releases, live = map(int, fields[3:])
            wanted_mask = MISSED_CHILD if expected == "missed-child" else ALL_EVAL
            if mask != wanted_mask:
                raise ValueError(f"in-call mode-mask differs at {label}/{ordinal}")
            graph = 1 if expected == "graph-create" else 0
            if (attempts, enrollments, releases, live) != (graph, graph,
                                                              graph, 0):
                raise ValueError(f"in-call graph-event differs at {label}/{ordinal}")
        if offset >= len(rows):
            raise ValueError("completed call summary missing")
        fields = rows[offset].split()
        offset += 1
        graph = 1 if expected == "graph-create" else 0
        if fields != ["E3MG-END", label, str(count), str(graph), str(graph),
                      str(graph), "0"]:
            raise ValueError(f"completed graph-event differs at {label}")
    if rows[offset:] != ["E3-MODE-GRAPH-RUNTIME-PASS"]:
        raise ValueError("test driver did not complete exactly")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stdout", type=Path, required=True)
    parser.add_argument("--expect", choices=("normal", "missed-child",
                                             "graph-create"), required=True)
    args = parser.parse_args()
    verify(args.stdout.read_text(), args.expect)


if __name__ == "__main__":
    main()
