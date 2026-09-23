"""Independent pinned-PyTorch check of the compiled TR3-B witness."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import struct

import torch

from tests.m3.reference import PARAMETERS, TIED, forward, parameters


ABSOLUTE_BOUND = 2e-8
RELATIVE_BOUND = 6e-4
SIZES = (64, 64, 64, 64, 128, 128, 16, 16, 16, 16, 4096, 16, 16, 32)
CASES = (
    ((3, 197), (197, 41), (1.0, 1.0)),   # genuine D2 mask 11
    ((41, 0), (7, 0), (1.0, 0.0)),       # genuine padded D2 mask 10
    ((3, 197), (197, 41), (0.0, 1.0)),   # explicitly test-only mask 01
)


def decode_bytes(fields: list[str], count: int) -> bytes:
    if len(fields) != count:
        raise ValueError(f"expected {count} bytes, observed {len(fields)}")
    values = tuple(map(int, fields))
    if any(not 0 <= value < 256 for value in values):
        raise ValueError("non-byte result")
    return bytes(values)


def bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def independent_records():
    contributions: list[dict[str, torch.Tensor]] = []
    edge_contributions: list[tuple[torch.Tensor, torch.Tensor]] = []
    observations: list[tuple[float, float, float]] = []
    for tokens, targets, mask_values in CASES:
        p, _ = parameters(1729)
        logits64 = forward(p, tokens)
        logits = logits64.to(torch.float32)
        targets_tensor = torch.tensor(targets, dtype=torch.int64).reshape(1, 2)
        mask = torch.tensor(mask_values, dtype=torch.float32).reshape(1, 2)
        losses = torch.logsumexp(logits, dim=-1) - logits.gather(
            -1, targets_tensor.unsqueeze(-1)
        ).squeeze(-1)
        numerator = mask[0, 0] * losses[0, 0]
        numerator = numerator + mask[0, 1] * losses[0, 1]
        weight = mask[0, 0] + mask[0, 1]
        mean = numerator / weight
        seed = torch.softmax(logits, dim=-1)
        seed = seed.scatter_add(
            -1, targets_tensor.unsqueeze(-1),
            -torch.ones((1, 2, 1), dtype=torch.float32),
        ) * mask.unsqueeze(-1)
        seed = seed.detach()
        values = torch.autograd.grad(
            (logits64 * seed.to(torch.float64)).sum(), tuple(p.values())
        )
        contribution = dict(zip(p, values))
        contributions.append(contribution)
        observations.append((numerator.item(), weight.item(), mean.item()))

        embedding = p[TIED].detach().clone().requires_grad_()
        head = p[TIED].detach().clone().requires_grad_()
        untied = forward(p, tokens, embedding_weight=embedding, head_weight=head)
        head_edge, embedding_edge = torch.autograd.grad(
            (untied * seed.to(torch.float64)).sum(), (head, embedding)
        )
        if not torch.equal(contribution[TIED], head_edge + embedding_edge):
            raise ValueError("independent tied-edge sum is not the unique gradient")
        edge_contributions.append((head_edge, embedding_edge))

    cumulative = []
    cumulative_edges = []
    running = {path: torch.zeros_like(contributions[0][path]) for path, _ in PARAMETERS}
    running_head = torch.zeros_like(edge_contributions[0][0])
    running_embedding = torch.zeros_like(edge_contributions[0][1])
    for contribution, (head, embedding) in zip(contributions, edge_contributions):
        running = {path: running[path] + contribution[path] for path, _ in PARAMETERS}
        running_head = running_head + head
        running_embedding = running_embedding + embedding
        cumulative.append({path: value.detach().clone() for path, value in running.items()})
        cumulative_edges.append((running_head.detach().clone(),
                                 running_embedding.detach().clone()))
    return cumulative, observations, cumulative_edges


def compare(label: str, raw: bytes, expected: torch.Tensor) -> None:
    observed = struct.unpack("<" + "f" * (len(raw) // 4), raw)
    wanted = expected.detach().flatten().tolist()
    if (len(observed) != len(wanted)
            or any(not math.isfinite(x) for x in observed)
            or any(not math.isfinite(x) for x in wanted)):
        raise ValueError(f"{label}: malformed or nonfinite gradient")
    maximum = max(map(abs, wanted), default=0.0)
    error = max((abs(a - b) for a, b in zip(observed, wanted)), default=0.0)
    bound = ABSOLUTE_BOUND + RELATIVE_BOUND * maximum
    if error > bound:
        raise ValueError(f"{label}: max error {error:.9g} exceeds {bound:.9g}")


def require_distinct(label: str, raw: bytes, forbidden: torch.Tensor) -> None:
    try:
        compare(label, raw, forbidden)
    except ValueError:
        return
    raise ValueError(f"{label}: unique tied gradient collapsed to one logical edge")


def parse_report(fields: list[str]) -> dict[str, int]:
    report: dict[str, int] = {}
    for field in fields:
        key, separator, encoded = field.partition("=")
        if not separator or not key or not encoded.isdecimal() or key in report:
            raise ValueError("malformed or duplicate M3 report field")
        report[key] = int(encoded)
    return report


def check(path: Path) -> None:
    gradients: dict[tuple[int, int], bytes] = {}
    metadata: dict[tuple[int, int], tuple[int, int, int]] = {}
    observations: dict[int, bytes] = {}
    late_observation = None
    native_report = None
    lines = path.read_text().splitlines()
    if not lines or lines[-1] != "TR3B-WITNESS-PASS":
        raise ValueError("missing exact witness completion marker")
    for line in lines[:-1]:
        fields = line.split()
        if fields[:1] == ["TR3B-GRADIENT"]:
            case, index = map(int, fields[1:3])
            key = (case, index)
            if key in gradients or not 0 <= case <= 5 or not 0 <= index < 14:
                raise ValueError("duplicate or invalid gradient row")
            gradients[key] = decode_bytes(fields[3:], SIZES[index])
        elif fields[:1] == ["TR3B-METADATA"]:
            case, index = map(int, fields[1:3])
            key = (case, index)
            if key in metadata or len(fields) != 6:
                raise ValueError("duplicate or malformed metadata row")
            metadata[key] = tuple(map(int, fields[3:6]))
        elif fields[:1] == ["TR3B-OBSERVATION"]:
            case = int(fields[1])
            if case in observations:
                raise ValueError("duplicate objective observation")
            observations[case] = decode_bytes(fields[2:], 12)
        elif fields[:1] == ["TR3B-LATE-OBSERVATION"]:
            if late_observation is not None:
                raise ValueError("duplicate late observation")
            late_observation = decode_bytes(fields[2:], 12)
        elif fields[:2] == ["M3", "native"]:
            if native_report is not None:
                raise ValueError("duplicate M3 native report")
            native_report = parse_report(fields[2:])
        else:
            raise ValueError(f"unexpected witness output: {line}")

    expected_keys = {(case, index) for case in range(6) for index in range(14)}
    if set(gradients) != expected_keys or set(metadata) != expected_keys:
        raise ValueError("missing gradient or metadata rows")
    if set(observations) != {0, 1, 2} or late_observation is None:
        raise ValueError("missing objective observation")
    if native_report is None:
        raise ValueError("missing M3 native report")

    cumulative, expected_observations, cumulative_edges = independent_records()
    expected_snapshot = {0: 0, 1: 1, 4: 2}
    for snapshot, contribution in expected_snapshot.items():
        for index, (path_name, _) in enumerate(PARAMETERS):
            compare(f"snapshot {snapshot} {path_name}", gradients[(snapshot, index)],
                    cumulative[contribution][path_name])
    for failed in (2, 3):
        for index, (path_name, _) in enumerate(PARAMETERS):
            if gradients[(failed, index)] != gradients[(1, index)]:
                raise ValueError(f"failure snapshot {failed} changed {path_name} bytes")
            if metadata[(failed, index)] != metadata[(1, index)]:
                raise ValueError(f"failure snapshot {failed} changed {path_name} metadata")
    expected_metadata = {
        0: (1, 1, bits(2.0)),
        1: (1, 2, bits(3.0)),
        2: (1, 2, bits(3.0)),
        3: (1, 2, bits(3.0)),
        4: (1, 3, bits(4.0)),
        5: (0, 0, 0),
    }
    for snapshot in range(6):
        for index in range(14):
            if metadata[(snapshot, index)] != expected_metadata[snapshot]:
                raise ValueError(
                    f"snapshot {snapshot} parameter {index}: metadata mismatch"
                )
            if snapshot == 5 and gradients[(snapshot, index)] != bytes(SIZES[index]):
                raise ValueError(f"reset left nonzero gradient bytes for parameter {index}")

    for case, expected in enumerate(expected_observations):
        observed = struct.unpack("<fff", observations[case])
        if any(not math.isfinite(value) for value in observed + expected):
            raise ValueError(f"case {case}: nonfinite observation")
        if bits(observed[1]) != bits(expected[1]):
            raise ValueError(f"case {case}: weight bits changed")
        for name, actual, wanted in zip(("numerator", "weight", "mean"),
                                        observed, expected):
            if abs(actual - wanted) > 2e-5 + 3e-5 * abs(wanted):
                raise ValueError(
                    f"case {case}: {name} differs from independent reference"
                )
    if late_observation != observations[0]:
        raise ValueError(
            "successful prepare before late VJP failure differs from case 0"
        )

    for snapshot, contribution in expected_snapshot.items():
        tied_raw = gradients[(snapshot, 10)]
        head, embedding = cumulative_edges[contribution]
        compare(f"snapshot {snapshot} tied unique sum", tied_raw,
                head + embedding)
        require_distinct(f"snapshot {snapshot} forbidden head-only", tied_raw, head)
        require_distinct(f"snapshot {snapshot} forbidden embedding-only",
                         tied_raw, embedding)

    # Exact live persistent model/workspace state is separated from zero live
    # transient borrows/plans and from cumulative released shells/tombstones.
    # Both ordinary and sanitizer witnesses must produce this entire schema.
    expected_report = {
        "graphs": 0, "logits": 0, "frames": 0, "graph-data": 0,
        "m3-controls": 2912, "m3t-controls": 1200,
        "m3-f32-tensors": 108, "m3-f32-payload": 32868,
        "m3-f32-metadata": 3952,
        "i2-tensors": 108, "i2-parameters": 14, "i2-owned-clones": 0,
        "i2-borrows": 0, "i2-copy-plans": 0, "i2-gradient-plans": 0,
        "i2-reset-plans": 0, "i2-retired": 10784,
        "i1-tensors": 2, "i1-borrows": 0, "i1-payload": 32,
        "i1-controls": 144, "i1-metadata": 64,
        "i2-retired-tensors": 120, "i2-retired-parameters": 0,
        "i2-retired-borrows": 0, "i2-retired-copy-plans": 1,
        "i2-retired-gradient-plans": 3, "i2-retired-reset-plans": 1,
        "bridge-observed": 0, "bridge-live-copy": 0,
        "bridge-live-reset": 0, "bridge-live-decode": 0,
        "bridge-retired-copy": 0, "bridge-retired-reset": 0,
        "bridge-retired-decode": 0, "bridge-retired-bytes": 0,
    }
    if native_report != expected_report:
        raise ValueError(f"M3 native report differs: observed {native_report}")

    print("TR3B-LIFETIME-PASS persistent-f32=108/32868/3952 "
          "live-graphs/logits/frames/borrows/copy/gradient/reset=0 "
          "retired-tensors/gradient/reset=120/3/1 controls=10784")
    print("TR3B-NUMERICAL-METADATA-PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    check(parser.parse_args().output)
