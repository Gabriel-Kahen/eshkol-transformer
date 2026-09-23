"""Frozen constants for the development-only TR3 mathematical reference."""

from __future__ import annotations

import struct


def _f32(word: str) -> float:
    return struct.unpack(">f", bytes.fromhex(word))[0]


FORMAT = "eshkol-tr3-training-reference"
VERSION = 1
PROFILE = "tr3-fixed-n1-t2-v256-d4-v1"
TOKENIZER_FINGERPRINT = "sha256:eshkol-byte-tokenizer-v1:aabb31f49216963582383a00fd2e85d7c20b658d5bef0e7945344ed6bfe50704"
VOCAB_SIZE = 256
INITIALIZER_SEED = 1729
UPDATE_COUNT = 32
TRAIN_SHARDS = ((3, 197, 41),)
HELDOUT_SHARDS = ((5, 197, 197), (6, 41))
RESUME_A1_SHARDS = TRAIN_SHARDS
RESUME_A2_SHARDS = ((9, 197, 197), (10, 41))

OPTIMIZER = {
    "algorithm": "adamw",
    "amsgrad": False,
    "betas": [_f32("3f666666"), _f32("3f7fbe77")],
    "betas_bits": ["3f666666", "3f7fbe77"],
    "capturable": False,
    "differentiable": False,
    "eps": _f32("322bcc77"),
    "eps_bits": "322bcc77",
    "foreach": False,
    "fused": False,
    "learning_rate": _f32("3d4ccccd"),
    "learning_rate_bits": "3d4ccccd",
    "maximize": False,
    "schedule": {"kind": "constant", "factor": 1.0, "factor_bits": "3f800000"},
    "weight_decay": 0.0,
    "weight_decay_bits": "00000000",
}

TOLERANCES = {
    "loss": {"absolute": 2.0e-5, "relative": 2.0e-5},
    "parameter": {"absolute": 2.0e-6, "relative": 2.0e-5},
    "optimizer_moment": {"absolute": 2.0e-6, "relative": 2.0e-5},
    "exact": [
        "configuration",
        "cursor_bytes",
        "mask_weights",
        "parameter_paths_and_shapes",
        "target_ids",
        "update_and_contribution_ordinals",
    ],
}

PROPOSAL = {
    "commit": "3e28db6",
    "path": "docs/TR3_TRAINER_PROPOSAL.md",
    "sha256": "08e6e23d1606e3db561b496af1a4b5b6acad7ea737b778b9c100411f8016d5d5",
    "section": 13,
}

SOURCE_PATHS = (
    "tests/tr3_reference/constants.py",
    "tests/tr3_reference/corpus.py",
    "tests/tr3_reference/schema.py",
    "tests/tr3_reference/generate_reference.py",
    "tests/m3/reference.py",
    "tests/n3k/test_initializer_reference.py",
    "tests/d2/reference.py",
    "tests/d2/test_resources.py",
    "tests/t1/reference.py",
    "tests/q0/requirements-oracle.lock",
)

PARAMETER_SHAPES = {
    "blocks/0/attention/key/weight": [4, 4],
    "blocks/0/attention/output/weight": [4, 4],
    "blocks/0/attention/query/weight": [4, 4],
    "blocks/0/attention/value/weight": [4, 4],
    "blocks/0/ffn/down/weight": [4, 8],
    "blocks/0/ffn/up/weight": [8, 4],
    "blocks/0/norm1/bias": [4],
    "blocks/0/norm1/weight": [4],
    "blocks/0/norm2/bias": [4],
    "blocks/0/norm2/weight": [4],
    "head/weight": [256, 4],
    "norm_final/bias": [4],
    "norm_final/weight": [4],
    "position_embedding/weight": [2, 4],
}
