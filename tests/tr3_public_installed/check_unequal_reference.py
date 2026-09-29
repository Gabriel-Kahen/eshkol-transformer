"""Independent fixed-profile D2/M3/O2 reference for the unequal train witness."""
from __future__ import annotations

import struct

import torch

from tests.d2.reference import logical_rows
from tests.m3.reference import forward, parameters


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def main() -> None:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rows = logical_rows(((3, 197, 41, 7, 11, 19),), 2, True)
    assert len(rows) == 3
    assert [sum(row.loss_mask) for row in rows] == [2, 2, 1]

    model, _ = parameters(1729, dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        list(model.values()), lr=f32(0.1), betas=(0.5, 0.5),
        eps=f32(0.001), weight_decay=0.0, foreach=False, fused=False,
    )
    numerators: list[float] = []
    weights: list[float] = []
    means: list[float] = []
    for order in ((0, 1), (2, 0)):
        gradients = {name: torch.zeros_like(value, dtype=torch.float64)
                     for name, value in model.items()}
        for ordinal in order:
            row = rows[ordinal]
            model64 = {name: value.detach().to(torch.float64).requires_grad_()
                       for name, value in model.items()}
            logits64 = forward(model64, row.inputs)
            logits = logits64.to(torch.float32)
            target = torch.tensor(row.targets, dtype=torch.int64).reshape(1, 2)
            mask = torch.tensor(row.loss_mask, dtype=torch.float32).reshape(1, 2)
            losses = (torch.logsumexp(logits, dim=-1)
                      - logits.gather(-1, target.unsqueeze(-1)).squeeze(-1))
            numerator = (mask[0, 0] * losses[0, 0]
                         + mask[0, 1] * losses[0, 1]).item()
            numerators.append(numerator)
            weights.append(mask.sum().item())
            seed = (torch.softmax(logits, dim=-1).scatter_add(
                -1, target.unsqueeze(-1),
                -torch.ones((1, 2, 1), dtype=torch.float32)
            ) * mask.unsqueeze(-1)).detach()
            contribution = torch.autograd.grad(
                (logits64 * seed.to(torch.float64)).sum(),
                tuple(model64.values()),
            )
            gradients = {name: gradients[name] + gradient
                         for (name, _), gradient in zip(model64.items(), contribution)}
        step_numerator = f32(numerators[-2] + numerators[-1])
        step_weight = f32(weights[-2] + weights[-1])
        means.append(f32(step_numerator / step_weight))
        for name, value in model.items():
            value.grad = (gradients[name] / step_weight).to(torch.float32)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    running = 0.0
    for numerator in numerators:
        running = f32(running + numerator)
    train_mean = f32(running / f32(sum(weights)))
    mean_of_means = f32(f32(means[0] / 2) + f32(means[1] / 2))
    assert [bits(value) for value in numerators] == [
        1093788954, 1093813638, 1085364985, 1093109014,
    ]
    assert [bits(value) for value in means] == [1085412688, 1084935265]
    assert bits(train_mean) == 1085208078
    assert bits(mean_of_means) == 1085173976
    assert bits(train_mean) != bits(mean_of_means)
    print("TR3-TRAIN-UNEQUAL-REFERENCE-PASS weights=4,3 loss-bits=1085208078")


if __name__ == "__main__":
    main()
