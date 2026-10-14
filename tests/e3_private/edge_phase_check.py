"""Validate flushed edge diagnostics without admitting a partial traversal."""

import argparse
import re
from pathlib import Path


CASES = ("empty", "singleton", "entry-eos", "one-token-tail")
PHASES = (
    "parameter", "scenario", "fixture", "d2-open", "mode-seed",
    "gradient-seed", "entry-skip", "cursor-snapshot", "outputs-snapshot",
    "modes-snapshot", "model-snapshot", "rng-snapshot", "probes-snapshot",
    "evaluate", "post-eval", "outcome-check", "cursor-check", "owners-check",
    "modes-check", "model-check", "rng-check", "close", "result-read", "done",
)
PHASE_LINE = re.compile(r"E3-EDGE-PHASE case=([a-z-]+) phase=([a-z0-9-]+)\Z")
def check(text: str, mode: str) -> None:
    lines = text.splitlines()
    if mode == "pre" and lines and lines[-1] == "Unhandled exception: user exception":
        lines.pop()
    if mode == "post" and lines and lines[-1] == "timeout: the monitored command dumped core":
        lines.pop()
    markers = []
    exceptions = []
    for line in lines:
        match = PHASE_LINE.fullmatch(line)
        if match:
            markers.append(match.groups())
        elif line.startswith("E3-EDGE-EXCEPTION "):
            exceptions.append(line)
        else:
            raise ValueError(f"unexpected edge stderr line: {line!r}")
    if mode == "normal":
        expected = [(case, phase) for case in CASES for phase in PHASES]
        if markers != expected or exceptions:
            raise ValueError("normal edge traversal lacks four complete phase traces")
        return
    if mode == "pre":
        expected = [("empty", item) for item in PHASES[:PHASES.index("d2-open") + 1]]
        provenance = ('E3-EDGE-EXCEPTION case=empty phase=d2-open '
                      'category=io operation=token-dataset-open '
                      'message="required file is missing"')
        if markers != expected or exceptions != [provenance]:
            raise ValueError("missing D2 corpus must fail at open with IO provenance")
        return
    expected = [("empty", item) for item in PHASES[:PHASES.index("outcome-check") + 1]]
    expected.append(("empty", "postcondition-control"))
    if markers != expected or exceptions:
        raise ValueError("post control must fail after checked evaluator outcome")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stderr", type=Path, required=True)
    parser.add_argument("--mode", choices=("normal", "pre", "post"), required=True)
    args = parser.parse_args()
    check(args.stderr.read_text(), args.mode)


if __name__ == "__main__":
    main()
