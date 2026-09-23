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
    observations: list[tuple[float, float, float]] = []
    tie_edges = None
    for case_index, (tokens, targets, mask_values) in enumerate(CASES):
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
        # L2 consumes a detached copy of M3's graph logits and stages a new
        # genuine seed owner. Its softmax construction is outside the M3 VJP.
        seed = seed.detach()
        values = torch.autograd.grad(
            (logits64 * seed.to(torch.float64)).sum(), tuple(p.values())
        )
        contributions.append(dict(zip(p, values)))
        observations.append((numerator.item(), weight.item(), mean.item()))

        if case_index == 0:
            embedding = p[TIED].detach().clone().requires_grad_()
            head = p[TIED].detach().clone().requires_grad_()
            untied = forward(
                p, tokens, embedding_weight=embedding, head_weight=head
            )
            head_edge, embedding_edge = torch.autograd.grad(
                (untied * seed.to(torch.float64)).sum(), (head, embedding)
            )
            unique = contributions[0][TIED]
            if not torch.equal(unique, head_edge + embedding_edge):
                raise ValueError("independent tied-edge sum is not the unique gradient")
            tie_edges = (head_edge, embedding_edge)

    cumulative = []
    running = {path: torch.zeros_like(contributions[0][path]) for path, _ in PARAMETERS}
    for contribution in contributions:
        running = {path: running[path] + contribution[path] for path, _ in PARAMETERS}
        cumulative.append({path: value.detach().clone() for path, value in running.items()})
    return cumulative, observations, tie_edges


def compare(label: str, raw: bytes, expected: torch.Tensor) -> None:
    observed = struct.unpack("<" + "f" * (len(raw) // 4), raw)
    wanted = expected.detach().flatten().tolist()
    if len(observed) != len(wanted) or any(not math.isfinite(x) for x in observed):
        raise ValueError(f"{label}: malformed or nonfinite gradient")
    maximum = max(map(abs, wanted), default=0.0)
    error = max((abs(a - b) for a, b in zip(observed, wanted)), default=0.0)
    bound = ABSOLUTE_BOUND + RELATIVE_BOUND * maximum
    if error > bound:
        raise ValueError(f"{label}: max error {error:.9g} exceeds {bound:.9g}")


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
            if key in gradients or not 0 <= case <= 2 or not 0 <= index < 14:
                raise ValueError("duplicate or invalid gradient row")
            gradients[key] = decode_bytes(fields[3:], SIZES[index])
        elif fields[:1] == ["TR3B-METADATA"]:
            case, index = map(int, fields[1:3])
            key = (case, index)
            if key in metadata:
                raise ValueError("duplicate metadata row")
            metadata[key] = tuple(map(int, fields[3:6]))
        elif fields[:1] == ["TR3B-OBSERVATION"]:
            case = int(fields[1])
            observations[case] = decode_bytes(fields[2:], 12)
        elif fields[:1] == ["TR3B-LATE-OBSERVATION"]:
            late_observation = decode_bytes(fields[2:], 12)
        elif fields[:2] == ["M3", "native"]:
            native_report = dict(field.split("=", 1) for field in fields[2:])
        else:
            raise ValueError(f"unexpected witness output: {line}")

    expected_keys = {(case, index) for case in range(3) for index in range(14)}
    if set(gradients) != expected_keys or set(metadata) != expected_keys:
        raise ValueError("missing gradient or metadata rows")
    if set(observations) != {0, 1, 2} or late_observation is None:
        raise ValueError("missing objective observation")
    if native_report is None or any(native_report.get(key) != "0" for key in
                                    ("graphs", "logits", "frames")):
        raise ValueError("M3 live graph/logits/frame baseline was not restored")

    cumulative, expected_observations, tie_edges = independent_records()
    # Snapshot 0 follows genuine 11 + 10. Snapshot 1 is the failed retry and
    # must be byte-identical. Snapshot 2 additionally contains injected 01.
    for index, (path_name, _) in enumerate(PARAMETERS):
        compare(f"after two contributions {path_name}", gradients[(0, index)],
                cumulative[1][path_name])
        if gradients[(1, index)] != gradients[(0, index)]:
            raise ValueError(f"late failure changed {path_name} bytes")
        compare(f"after injected contribution {path_name}", gradients[(2, index)],
                cumulative[2][path_name])
    for index in range(14):
        if metadata[(0, index)] != (1, 2, bits(3.0)):
            raise ValueError("two-contribution metadata mismatch")
        if metadata[(1, index)] != metadata[(0, index)]:
            raise ValueError("late failure changed gradient metadata")
        if metadata[(2, index)] != (1, 3, bits(4.0)):
            raise ValueError("three-contribution metadata mismatch")

    for case, expected in enumerate(expected_observations):
        observed = struct.unpack("<fff", observations[case])
        if bits(observed[1]) != bits(expected[1]):
            raise ValueError(f"case {case}: weight bits changed")
        for name, actual, wanted in zip(("numerator", "weight", "mean"), observed, expected):
            if abs(actual - wanted) > 2e-5 + 3e-5 * abs(wanted):
                raise ValueError(f"case {case}: {name} differs from independent reference")
    if late_observation != observations[0]:
        raise ValueError("successful prepare before late failure produced different observation")

    # The compiled unique tied row must match the head+embedding sum and must
    # be distinguishable from either individual logical path.
    tied_raw = gradients[(0, 10)]
    compare("tied unique sum", tied_raw, cumulative[1][TIED])
    head, embedding = tie_edges
    try:
        compare("forbidden head-only tie", tied_raw, head)
    except ValueError:
        pass
    else:
        raise ValueError("tied unique gradient collapsed to head-only contribution")
    try:
        compare("forbidden embedding-only tie", tied_raw, embedding)
    except ValueError:
        pass
    else:
        raise ValueError("tied unique gradient collapsed to embedding-only contribution")

    print("TR3B-NUMERICAL-METADATA-PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    check(parser.parse_args().output)
