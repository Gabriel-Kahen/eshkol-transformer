"""Development-only PyTorch check of the compiled private one-update witness."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import struct

import torch

from tests.m3.reference import parameters
from tests.tr3b.check_witness import independent_records


KEY = "blocks/0/attention/key/weight"


def _f32(word: str) -> float:
    return struct.unpack(">f", bytes.fromhex(word))[0]


def _raw(fields: list[str]) -> tuple[float, ...]:
    if len(fields) != 64:
        raise ValueError("key/weight must be exact CPU f32[4,4]")
    values = [int(field) for field in fields]
    if any(not 0 <= value <= 255 for value in values):
        raise ValueError("parameter contains a non-byte value")
    result = struct.unpack("<16f", bytes(values))
    if not all(math.isfinite(value) for value in result):
        raise ValueError("parameter contains nonfinite f32")
    return result


def check(path: Path) -> None:
    lines = path.read_text().splitlines()
    if len(lines) != 5 or lines[-1] != "TR3-STEP-LEAF-PASS checks=41":
        raise ValueError("incomplete or extra witness output")
    before = lines[0].split()
    after = lines[3].split()
    if before[:1] != ["TR3-STEP-KEY-BEFORE"] or after[:1] != ["TR3-STEP-KEY-AFTER"]:
        raise ValueError("missing exact parameter rows")
    observed_before = _raw(before[1:])
    observed_after = _raw(after[1:])
    cumulative, observations, _ = independent_records()
    for index in range(2):
        fields = lines[index + 1].split()
        if fields[:2] != ["TR3-STEP-OBS", str(index)] or len(fields) != 5:
            raise ValueError("missing exact objective observation")
        actual = struct.unpack("<fff", struct.pack("<III", *(int(v) for v in fields[2:])))
        wanted = observations[index]
        if not all(math.isfinite(value) for value in actual):
            raise ValueError("nonfinite objective observation")
        if fields[3] != str(struct.unpack("<I", struct.pack("<f", wanted[1]))[0]):
            raise ValueError("mask weight bits differ")
        for a, b in zip(actual, wanted):
            if abs(a - b) > 2e-5 + 3e-5 * abs(b):
                raise ValueError("objective differs from independent reference")

    expected_before = parameters(1729)[0][KEY].detach().to(torch.float32)
    observed_before_tensor = torch.tensor(observed_before, dtype=torch.float32).reshape(4, 4)
    before_delta = (observed_before_tensor - expected_before).abs().max().item()
    if before_delta > 2e-6:
        raise ValueError(f"initial parameter differs: {before_delta}")

    # The accepted O2 logical config is AdamW, constant schedule, no clipping
    # or decay, lr=0.1, beta=(0.5,0.5), epsilon=0.001. Its input is the
    # numerator-gradient sum divided once by the exact mask weight three.
    expected_after = expected_before.clone().detach().requires_grad_()
    expected_after.grad = cumulative[1][KEY].detach().to(torch.float32) / 3.0
    optimizer = torch.optim.AdamW(
        [expected_after], lr=_f32("3dcccccd"),
        betas=(_f32("3f000000"), _f32("3f000000")),
        eps=_f32("3a83126f"), weight_decay=0.0,
        foreach=False, fused=False,
    )
    optimizer.step()
    observed_after_tensor = torch.tensor(observed_after, dtype=torch.float32).reshape(4, 4)
    after_delta = (observed_after_tensor - expected_after.detach()).abs().max().item()
    if after_delta > 2e-6 + 2e-5 * expected_after.detach().abs().max().item():
        raise ValueError(f"updated parameter differs: {after_delta}")
    if torch.equal(observed_before_tensor, observed_after_tensor):
        raise ValueError("optimizer update made no parameter change")
    print(f"TR3-STEP-NUMERICAL-PASS before_max_abs={before_delta:.9g} after_max_abs={after_delta:.9g}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("witness", type=Path)
    check(parser.parse_args().witness)
