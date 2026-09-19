"""Fail closed on every native field in the fixed M3 public arena fixture."""
from pathlib import Path
import re
import sys

MODES = ("forward", "logits", "vjp", "reset", "failure")
HORIZONS = (1024, 8192)


def expected(mode, horizon):
    if mode not in MODES or horizon not in HORIZONS:
        raise ValueError("unknown native retention case")
    fresh = horizon if mode in ("forward", "logits") else 0
    copies = horizon if mode == "logits" else 0
    contributions = horizon if mode in ("vjp", "reset") else 0
    resets = horizon if mode == "reset" else 0
    # Package provider admission creates four scratch tensors (two assignments)
    # and four detached source/destination baselines, plus one copy plan/builder:
    # native/i2_wave2_extension.esk:i2-register-provider!, P1:tensor-provider-admit!.
    # Model construction retires 14 temporaries; fixture teardown retires the
    # initial graph's 15 tensors and one mutable seed tensor. These fixed costs
    # precede the loop and are independently derived, not fitted per horizon.
    retired_tensors = 8 + 14 + 15 + 1 + 15 * fresh + copies
    return {
        "graphs": 0, "logits": 0, "frames": 0, "graph-data": 0,
        "m3-controls": 448 * (fresh + 1) + 32 * copies + 32,
        # Two initializers, model owner, input shell, seed shell, one workspace.
        "m3t-controls": 2 * 48 + 272 + 24 + 24 + 760,
        "m3-f32-tensors": 108, "m3-f32-payload": 32868,
        "m3-f32-metadata": 3952, "i2-tensors": 108, "i2-parameters": 14,
        "i2-owned-clones": 0, "i2-borrows": 0, "i2-copy-plans": 0,
        "i2-gradient-plans": 0, "i2-reset-plans": 0,
        "i2-retired": 88 * retired_tensors + 40 + 48 * contributions + 40 * resets,
        "i1-tensors": 2, "i1-borrows": 0, "i1-payload": 32,
        "i1-controls": 144, "i1-metadata": 64,
        "i2-retired-tensors": retired_tensors, "i2-retired-parameters": 0,
        "i2-retired-borrows": 0, "i2-retired-copy-plans": 1,
        "i2-retired-gradient-plans": contributions, "i2-retired-reset-plans": resets,
        "bridge-observed": 1, "bridge-live-copy": 0, "bridge-live-reset": 0,
        "bridge-live-decode": 0, "bridge-retired-copy": 1,
        "bridge-retired-reset": resets, "bridge-retired-decode": 0,
        "bridge-retired-bytes": 40 * (resets + 1),
    }


def parse(text, mode, horizon):
    marker = f"M3-ARENA-RETENTION-PASS {mode} {horizon}"
    lines = text.splitlines()
    if len(lines) != 2:
        raise ValueError("native stdout must contain exactly marker then report")
    markers = [line for line in lines if line.startswith("M3-ARENA-RETENTION-PASS")]
    reports = [line for line in lines if line.startswith("M3 native")]
    if markers != [marker] or len(reports) != 1:
        raise ValueError("missing, duplicate or mismatched native fixture marker/report")
    if lines.index(marker) >= lines.index(reports[0]):
        raise ValueError("native report precedes fixture teardown")
    if not reports[0].startswith("M3 native "):
        raise ValueError("malformed native report prefix")
    fields = {}
    for token in reports[0][len("M3 native "):].split(" "):
        if not re.fullmatch(r"[a-z0-9-]+=(0|[1-9][0-9]*)", token):
            raise ValueError("malformed native report field")
        key, value = token.split("=")
        if key in fields:
            raise ValueError("duplicate native report field")
        fields[key] = int(value)
    wanted = expected(mode, horizon)
    if fields.keys() != wanted.keys():
        raise ValueError(f"native report schema differs: {fields.keys() ^ wanted.keys()}")
    for key, value in wanted.items():
        if fields[key] != value:
            raise ValueError(f"{mode}/{horizon}: {key}={fields[key]}, expected {value}")
    return fields


def check(directory):
    directory = Path(directory)
    for mode in MODES:
        rows = []
        for horizon in HORIZONS:
            rows.append(parse((directory / f"arena-{mode}-{horizon}.stdout").read_text(), mode, horizon))
        def retained(row):
            return row["m3-controls"] + row["m3t-controls"] + row["i2-retired"] + row["bridge-retired-bytes"]
        delta = retained(rows[1]) - retained(rows[0])
        divisor = HORIZONS[1] - HORIZONS[0]
        if delta % divisor:
            raise ValueError("nonintegral native retained-control trajectory")
        print(f"M3 native retention {mode}: scoped_control_bytes_per_iteration={delta // divisor} "
              "live_graphs=0 live_frames=0 live_plans=0 live_f32_payload=32868")
    print("M3 NATIVE RETENTION PASS: all 5 modes and both horizons match exact source contracts; "
          "fixed K1/P1 allocations and Eshkol arena are separate accounting.")


if __name__ == "__main__":
    check(sys.argv[1])
