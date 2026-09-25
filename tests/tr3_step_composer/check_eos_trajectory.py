"""Development-only independent PyTorch oracle for two finite-D2 updates."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import struct

import torch

from tests.m3.reference import forward, parameters
from tests.tr3b.check_witness import CASES
from tests.tr3_step_leaf.check_numeric import KEY, _f32, _raw


def _observation(p: dict[str, torch.Tensor], case: int):
    tokens, targets, mask_values = CASES[case]
    logits64 = forward(p, tokens)
    logits = logits64.to(torch.float32)
    target = torch.tensor(targets, dtype=torch.int64).reshape(1, 2)
    mask = torch.tensor(mask_values, dtype=torch.float32).reshape(1, 2)
    losses = torch.logsumexp(logits, dim=-1) - logits.gather(
        -1, target.unsqueeze(-1)
    ).squeeze(-1)
    numerator = mask[0, 0] * losses[0, 0] + mask[0, 1] * losses[0, 1]
    weight = mask.sum()
    seed = (torch.softmax(logits, dim=-1).scatter_add(
        -1, target.unsqueeze(-1),
        -torch.ones((1, 2, 1), dtype=torch.float32),
    ) * mask.unsqueeze(-1)).detach()
    gradient = torch.autograd.grad(
        (logits64 * seed.to(torch.float64)).sum(), tuple(p.values())
    )
    return (numerator.item(), weight.item(), (numerator / weight).item()), dict(zip(p, gradient))


def _observed(words: list[str]) -> tuple[float, float, float]:
    if len(words) != 5 or words[:2] not in (["TR3-STEP-COMPOSER-OBS", "0"],
                                             ["TR3-STEP-COMPOSER-OBS", "1"]):
        raise ValueError("missing exact finite-D2 observation")
    values = struct.unpack("<fff", struct.pack("<III", *(int(w) for w in words[2:])))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("nonfinite finite-D2 observation")
    return values


def check(path: Path) -> None:
    lines = path.read_text().splitlines()
    before = next(i for i, line in enumerate(lines) if line.startswith("TR3-STEP-KEY-BEFORE "))
    after1 = next(i for i, line in enumerate(lines) if line.startswith("TR3-STEP-KEY-AFTER "))
    after2 = next(i for i, line in enumerate(lines) if line.startswith("TR3-STEP-KEY-AFTER-2 "))
    if not before < after1 < after2:
        raise ValueError("parameter trajectory is not ordered")
    observed = (
        _raw(lines[before].split()[1:]),
        _raw(lines[after1].split()[1:]),
        _raw(lines[after2].split()[1:]),
    )
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    p, _ = parameters(1729, dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        list(p.values()), lr=_f32("3dcccccd"),
        betas=(_f32("3f000000"), _f32("3f000000")),
        eps=_f32("3a83126f"), weight_decay=0.0,
        foreach=False, fused=False,
    )
    errors = []
    for step in range(3):
        expected = p[KEY].detach().flatten()
        actual = torch.tensor(observed[step], dtype=torch.float32)
        error = (actual - expected).abs().max().item()
        if error > 2e-6 + 2e-5 * expected.abs().max().item():
            raise ValueError(f"update {step} key/weight trajectory differs: {error:.9g}")
        errors.append(error)
        if step == 2:
            break
        p64 = {name: tensor.detach().to(torch.float64).requires_grad_()
               for name, tensor in p.items()}
        cumulative = {name: torch.zeros_like(value) for name, value in p64.items()}
        wanted = []
        for case in (0, 1):
            obs, gradients = _observation(p64, case)
            wanted.append(obs)
            cumulative = {name: cumulative[name] + gradients[name]
                          for name in cumulative}
        if step == 1:
            rows = [line.split() for line in lines[after1 + 1:after2]
                    if line.startswith("TR3-STEP-COMPOSER-OBS ")]
            if len(rows) != 2 or [row[1] for row in rows] != ["0", "1"]:
                raise ValueError("second update lacks two ordered observations")
            for row, expected_obs in zip(rows, wanted):
                actual_obs = _observed(row)
                if any(abs(a - b) > 2e-5 + 3e-5 * abs(b)
                       for a, b in zip(actual_obs, expected_obs)):
                    raise ValueError("post-rewind objective differs from independent reference")
        for name, tensor in p.items():
            tensor.grad = (cumulative[name] / 3.0).to(torch.float32)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    print("TR3-STEP-EOS-TRAJECTORY-PASS max_abs="
          f"{max(errors):.9g} after1={errors[1]:.9g} after2={errors[2]:.9g}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("witness", type=Path)
    check(parser.parse_args().witness)
