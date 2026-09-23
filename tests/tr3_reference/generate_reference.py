#!/usr/bin/env python3
"""Generate the deterministic, development-only TR3 training reference.

The output contains the complete 33-point loss/parameter/AdamW-state trajectory.
It belongs in a temporary or build directory and is intentionally not checked in.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys

from tests.m3 import reference as m3
import torch
import torch.nn.functional as F

from tests.tr3_reference.constants import (
    FORMAT,
    INITIALIZER_SEED,
    OPTIMIZER,
    PROFILE,
    PROPOSAL,
    SOURCE_PATHS,
    TOLERANCES,
    UPDATE_COUNT,
    VERSION,
)
from tests.tr3_reference.corpus import materialize


ROOT = Path(__file__).resolve().parents[2]


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")


def _f32_bits(value: float) -> str:
    return struct.pack(">f", float(value)).hex()


def _tensor(value: torch.Tensor) -> dict[str, object]:
    flat = value.detach().to(dtype=torch.float32, device="cpu").contiguous().view(-1)
    return {
        "shape": list(value.shape),
        "encoding": "ieee754-f32-hex-be",
        "data": [_f32_bits(item) for item in flat.tolist()],
    }


def _digest_record(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def _batch_tensors(batch: dict[str, object]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens = torch.tensor(batch["inputs"], dtype=torch.int64).reshape(1, 2)
    targets = torch.tensor(batch["targets"], dtype=torch.int64).reshape(1, 2)
    mask = torch.tensor(batch["mask"], dtype=torch.bool).reshape(1, 2)
    return tokens, targets, mask


def _loss(p: dict[str, torch.Tensor], batch: dict[str, object]):
    tokens, targets, mask = _batch_tensors(batch)
    logits = m3.forward(p, tuple(tokens[0].tolist()))
    per_token = F.cross_entropy(logits.reshape(-1, 256), targets.reshape(-1), reduction="none").reshape(1, 2)
    numerator = (per_token * mask.to(torch.float32)).sum()
    weight = int(mask.sum().item())
    return logits, per_token, numerator, weight


def _evaluation(p: dict[str, torch.Tensor], batches: list[dict[str, object]]) -> dict[str, object]:
    numerator = 0.0
    weight = 0
    records = []
    with torch.no_grad():
        for batch in batches:
            logits, per_token, batch_numerator, batch_weight = _loss(p, batch)
            predictions = logits.argmax(dim=-1)[0].tolist()
            unique = [int((row == row.max()).sum().item()) == 1 for row in logits[0]]
            records.append(
                {
                    "inputs": batch["inputs"],
                    "targets": batch["targets"],
                    "mask": batch["mask"],
                    "weight": batch_weight,
                    "numerator_bits": _f32_bits(batch_numerator.item()),
                    "per_token_loss_bits": [_f32_bits(item) for item in per_token[0].tolist()],
                    "argmax": predictions,
                    "unique_argmax": unique,
                }
            )
            numerator += float(batch_numerator.item())
            weight += batch_weight
    mean = numerator / weight
    return {
        "batch_count": len(batches),
        "token_count": weight,
        "weight": weight,
        "numerator": numerator,
        "mean": mean,
        "numerator_bits_after_f64_sum": struct.pack(">d", numerator).hex(),
        "mean_bits_f64": struct.pack(">d", mean).hex(),
        "batches": records,
    }


def _optimizer(parameters) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        list(parameters),
        lr=OPTIMIZER["learning_rate"],
        betas=tuple(OPTIMIZER["betas"]),
        eps=OPTIMIZER["eps"],
        weight_decay=OPTIMIZER["weight_decay"],
        foreach=False,
        fused=False,
        capturable=False,
        differentiable=False,
        maximize=False,
        amsgrad=False,
    )


def _state_entry(
    update: int,
    p: dict[str, torch.Tensor],
    optimizer: torch.optim.AdamW,
    evaluation: dict[str, object],
) -> dict[str, object]:
    moments = {}
    for path, value in p.items():
        state = optimizer.state.get(value)
        if not state:
            zeros = torch.zeros_like(value)
            moments[path] = {"step": 0, "exp_avg": _tensor(zeros), "exp_avg_sq": _tensor(zeros)}
        else:
            moments[path] = {
                "step": int(state["step"].item()),
                "exp_avg": _tensor(state["exp_avg"]),
                "exp_avg_sq": _tensor(state["exp_avg_sq"]),
            }
    return {
        "completed_updates": update,
        "training_evaluation": evaluation,
        "parameters": {path: _tensor(value) for path, value in p.items()},
        "optimizer_state": moments,
    }


def _accumulation_proof(batches: list[dict[str, object]]) -> dict[str, object]:
    if [item["weight"] for item in batches] != [2, 1]:
        raise AssertionError("A=2 proof requires unequal mask weights [2,1]")
    separate, _ = m3.parameters(INITIALIZER_SEED, dtype=torch.float32)
    combined, _ = m3.parameters(INITIALIZER_SEED, dtype=torch.float32)
    wrong, _ = m3.parameters(INITIALIZER_SEED, dtype=torch.float32)
    total_weight = sum(item["weight"] for item in batches)
    for batch in batches:
        _loss(separate, batch)[2].backward()
    combined_numerators = [_loss(combined, batch)[2] for batch in batches]
    sum(combined_numerators).backward()
    for batch in batches:
        numerator = _loss(wrong, batch)[2]
        (numerator / batch["weight"]).backward()
    max_abs = max(
        float((separate[path].grad - combined[path].grad).abs().max().item())
        for path in separate
    )
    wrong_max_abs = max(
        float((separate[path].grad / total_weight - wrong[path].grad / len(batches)).abs().max().item())
        for path in separate
    )
    separate_record = {path: _tensor(value.grad) for path, value in separate.items()}
    combined_record = {path: _tensor(value.grad) for path, value in combined.items()}
    wrong_record = {path: _tensor(value.grad) for path, value in wrong.items()}
    exact = separate_record == combined_record
    return {
        "contribution_ordinals": [0, 1],
        "weights": [2, 1],
        "total_weight": total_weight,
        "parameter_count": len(separate),
        "summed_numerator_gradient_sha256": _digest_record(separate_record),
        "combined_numerator_gradient_sha256": _digest_record(combined_record),
        "summed_numerator_gradients": separate_record,
        "combined_numerator_gradients": combined_record,
        "wrong_per_batch_mean_gradients": wrong_record,
        "bit_identical": exact,
        "within_parameter_tolerance": max_abs <= TOLERANCES["parameter"]["absolute"],
        "max_abs_difference": max_abs,
        "wrong_per_batch_mean_max_abs_difference": wrong_max_abs,
    }


def _source_hashes() -> dict[str, str]:
    return {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in SOURCE_PATHS}


def build(output: Path) -> dict[str, object]:
    m3.setup()
    corpus = materialize(output)
    train_batches = corpus["datasets"]["train"]["batches"]
    heldout_batches = corpus["datasets"]["heldout"]["batches"]
    if len(train_batches) != 1 or train_batches[0]["weight"] != 2:
        raise AssertionError("training corpus must be one fully active batch")
    if [batch["weight"] for batch in heldout_batches] != [2, 1]:
        raise AssertionError("held-out corpus must have two unequal-mask batches")
    train_transitions = {
        (left, right)
        for batch in train_batches
        for left, right, active in zip(batch["inputs"], batch["targets"], batch["mask"])
        if active
    }
    heldout_transitions = {
        (left, right)
        for batch in heldout_batches
        for left, right, active in zip(batch["inputs"], batch["targets"], batch["mask"])
        if active
    }
    if train_transitions & heldout_transitions:
        raise AssertionError("held-out active transitions must differ from training transitions")

    p, initializer_successor = m3.parameters(INITIALIZER_SEED, dtype=torch.float32)
    optimizer = _optimizer(p.values())
    before_train = _evaluation(p, train_batches)
    before_heldout = _evaluation(p, heldout_batches)
    trajectory = [_state_entry(0, p, optimizer, before_train)]
    for update in range(1, UPDATE_COUNT + 1):
        optimizer.zero_grad(set_to_none=True)
        _, _, numerator, weight = _loss(p, train_batches[0])
        (numerator / weight).backward()
        optimizer.step()
        trajectory.append(_state_entry(update, p, optimizer, _evaluation(p, train_batches)))
    after_train = trajectory[-1]["training_evaluation"]
    after_heldout = _evaluation(p, heldout_batches)
    active_targets = [
        target
        for batch in after_train["batches"]
        for target, active in zip(batch["targets"], batch["mask"])
        if active
    ]
    active_predictions = [
        prediction
        for batch in after_train["batches"]
        for prediction, active in zip(batch["argmax"], batch["mask"])
        if active
    ]
    active_unique = [
        unique
        for batch in after_train["batches"]
        for unique, active in zip(batch["unique_argmax"], batch["mask"])
        if active
    ]
    if active_predictions != active_targets or not all(active_unique):
        raise AssertionError("final active targets are not unique logits argmax")
    if not math.isfinite(after_train["mean"]) or after_train["mean"] >= 0.05:
        raise AssertionError("one-batch overfit threshold was not met")
    if after_heldout["mean"] >= before_heldout["mean"]:
        raise AssertionError("held-out global numerator/weight loss did not decrease")

    corpus_payload = {
        "format": FORMAT,
        "version": VERSION,
        "kind": "corpora-and-cursors",
        "profile": PROFILE,
        **corpus,
    }
    trajectory_payload = {
        "format": FORMAT,
        "version": VERSION,
        "kind": "complete-training-trajectory",
        "profile": PROFILE,
        "parameter_paths": [path for path, _ in m3.PARAMETERS],
        "parameter_shapes": {path: list(shape) for path, shape in m3.PARAMETERS},
        "logical_alias": {"token_embedding/weight": m3.TIED},
        "entries": trajectory,
    }
    summary_payload = {
        "format": FORMAT,
        "version": VERSION,
        "kind": "acceptance-summary",
        "profile": PROFILE,
        "initializer": {"seed": INITIALIZER_SEED, "successor": initializer_successor},
        "optimizer": OPTIMIZER,
        "updates": UPDATE_COUNT,
        "training": {"before": before_train, "after": after_train},
        "heldout": {
            "before": before_heldout,
            "after": after_heldout,
            "train_transitions": [list(item) for item in sorted(train_transitions)],
            "heldout_transitions": [list(item) for item in sorted(heldout_transitions)],
        },
        "accumulation_reference": _accumulation_proof(corpus["datasets"]["resume-a2"]["batches"]),
        "thresholds": {"final_training_mean_ce_strict_upper": 0.05, "tolerances": TOLERANCES},
        "provenance": {
            "python": ".".join(map(str, m3.PINNED_PYTHON)),
            "torch": str(torch.__version__),
            "environment": {name: os.environ[name] for name in ("ATEN_CPU_CAPABILITY", "MKL_CBWR")},
            "proposal": PROPOSAL,
            "source_sha256": _source_hashes(),
        },
        "scope": {
            "development_reference_only": True,
            "native_acceptance": False,
            "native_resume_proof": False,
            "checkpoint_format_or_api": False,
            "runtime_or_production_dependency": False,
        },
    }
    payloads = {
        "corpora.json": corpus_payload,
        "trajectory.json": trajectory_payload,
        "summary.json": summary_payload,
    }
    for name, payload in payloads.items():
        (output / name).write_bytes(canonical(payload))
    artifacts = {
        name: {"bytes": (output / name).stat().st_size, "sha256": _sha256(output / name)}
        for name in payloads
    }
    return {
        "format": "eshkol-tr3-training-reference-manifest",
        "version": VERSION,
        "profile": PROFILE,
        "artifacts": artifacts,
        "configuration": {
            "initializer_seed": INITIALIZER_SEED,
            "optimizer": OPTIMIZER,
            "updates": UPDATE_COUNT,
        },
        "measurements": {
            "initial_training_mean_ce": before_train["mean"],
            "final_training_mean_ce": after_train["mean"],
            "final_active_targets": active_targets,
            "final_active_argmax": active_predictions,
            "final_active_unique_argmax": active_unique,
            "initial_heldout_global_mean_ce": before_heldout["mean"],
            "final_heldout_global_mean_ce": after_heldout["mean"],
            "heldout_weight": after_heldout["weight"],
            "heldout_batches": after_heldout["batch_count"],
            "heldout_mask_weights": [batch["weight"] for batch in heldout_batches],
        },
        "provenance": summary_payload["provenance"],
        "scope": summary_payload["scope"],
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--candidate-manifest", type=Path)
    args = parser.parse_args()
    manifest = build(args.output)
    from tests.tr3_reference.schema import validate_output

    validate_output(args.output, manifest)
    if args.candidate_manifest is not None:
        args.candidate_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.candidate_manifest.write_bytes(canonical(manifest))
    else:
        expected = json.loads((Path(__file__).with_name("reference_manifest.json")).read_text())
        if manifest != expected:
            raise RuntimeError("TR3 reference differs from the frozen manifest; use --candidate-manifest for independent review")
    print(hashlib.sha256(canonical(manifest)).hexdigest())


if __name__ == "__main__":
    main()
