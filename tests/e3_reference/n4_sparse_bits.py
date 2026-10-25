"""Development-only C1 state and independent sparse M3 forward bit oracle.

No production schedule or provider is imported. This state is a candidate for
the private compiled gate, not evidence that the gate ran.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path
import hashlib
import json

from tests.c1.test_checkpoint_format import build_container, parse_golden
from tests.e3_reference.uniform_tie import ALIAS, PROVIDER, SHAPES
from tests.e3_reference.finite_overflow import PROFILE, RAW
from tests.e3_reference.corpus import FINGERPRINT, VOCAB_SIZE
from tests.d2.reference import ReferenceDataset
from tests.d2.test_resources import load_d1_resource, write_d1_resource


AXIS = (0x3F800000, 0xBF800000, 0x3F800000, 0xBF800000)
MIN_SUBNORMAL = 0x00000001
EPSILON = 0x3727C5AC


def bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def value(word: int) -> float:
    return struct.unpack("<f", struct.pack("<I", word))[0]


def f32(number: float) -> float:
    return value(bits(number))


def parameter_words(path: str, shape: tuple[int, ...]) -> tuple[int, ...]:
    count = math.prod(shape)
    words = [0] * count
    if path in ("head/weight", "token_embedding/weight"):
        assert shape == (256, 4)
        words[3 * 4:4 * 4] = AXIS
        words[4 * 4] = MIN_SUBNORMAL
    elif path == "position_embedding/weight":
        assert shape == (2, 4)
        words[4:8] = AXIS
    elif path == "norm_final/weight":
        assert shape == (4,)
        words[:] = [0x3F800000] * 4
    return tuple(words)


def checkpoint() -> bytes:
    entries = tuple(
        (tuple(part.encode("ascii") for part in path.split("/")),
         1, 3, shape, struct.pack("<" + "I" * math.prod(shape),
                                *parameter_words(path, shape)))
        for path, shape in SHAPES
    )
    assert entries[ALIAS[0]][-1] == entries[ALIAS[1]][-1]
    encoded = build_container(entries, (ALIAS,), PROVIDER)
    parsed = parse_golden(encoded, PROVIDER)
    assert len(parsed.records) == 15 and len(parsed.aliases) == 1
    assert tuple(parsed.aliases[0].members) == ALIAS
    for record, expected in zip(parsed.records, entries, strict=True):
        assert encoded[record.payload_start:record.payload_start + record.payload_bytes] == expected[-1]
    return encoded


def _norm_axis() -> tuple[float, ...]:
    # The sparse graph leaves X=R=Y=(1,-1,1,-1) at both positions.
    x = tuple(map(value, AXIS))
    mean = f32(sum(x) / 4)
    variance = f32(sum(f32(f32(v - mean) ** 2) for v in x) / 4)
    reciprocal = f32(1 / f32(math.sqrt(f32(variance + value(EPSILON)))))
    return tuple(f32(f32(f32(v - mean) * reciprocal) * 1.0) for v in x)


def forward_bits() -> tuple[int, ...]:
    """Full 1x2x256 logit shape for the sparse, zero-projection M3 graph."""
    normalized = _norm_axis()
    out = [0] * 512
    for position in range(2):
        # Every attention and FFN projection is exactly zero. The three
        # residuals preserve AXIS; final LN feeds the tied head matmul.
        head3 = 0.0
        for x, word in zip(normalized, AXIS, strict=True):
            head3 = f32(head3 + f32(x * value(word)))
        tiny = f32(0.0 + f32(normalized[0] * value(MIN_SUBNORMAL)))
        out[position * 256 + 3] = bits(head3)
        out[position * 256 + 4] = bits(tiny)
    return tuple(out)


def materialize(root: Path) -> dict:
    if root.exists() or root.is_symlink() or root.parent.resolve() != root.parent.absolute():
        raise ValueError("N4 fixture directory must be fresh and canonical")
    root.mkdir()
    encoded = checkpoint()
    (root / "sparse.etcp").write_bytes(encoded)
    corpus = root / "sparse-corpus"
    corpus.mkdir()
    write_d1_resource(corpus, (tuple(RAW),),
                      fingerprint=FINGERPRINT, vocab=VOCAB_SIZE)
    loaded, digest = load_d1_resource(
        corpus, expected_fingerprint=FINGERPRINT,
        expected_vocab=VOCAB_SIZE, maximum_total_tokens=2)
    dataset = ReferenceDataset(loaded, PROFILE, manifest_digest=digest,
                               tokenizer_fingerprint=FINGERPRINT,
                               vocab_size=VOCAB_SIZE)
    row = dataset.next_batch()
    assert row is not None and dataset.next_batch() is None
    assert list(row.inputs[0]) == [3, 0]
    assert list(row.targets[0]) == [4, 0]
    assert list(row.loss_mask[0]) == [True, False]
    manifest = {
        "kind": "e3-n4-sparse-subnormal-v1",
        "checkpoint_sha256": hashlib.sha256(encoded).hexdigest(),
        "corpus": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in sorted(corpus.iterdir())},
        "inputs": [3, 0], "targets": [4, 0], "mask": [1, 0],
        "logit_words_sha256": hashlib.sha256(struct.pack("<512I", *forward_bits())).hexdigest(),
    }
    (root / "fixture.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
    return manifest


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m tests.e3_reference.n4_sparse_bits FRESH_DIR")
    print(json.dumps(materialize(Path(sys.argv[1])), sort_keys=True))
