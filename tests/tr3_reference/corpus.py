"""Create authentic tiny T1/D1/D2 resources and derive EOS cursor traces."""

from __future__ import annotations

import hashlib
from pathlib import Path

from tests.d1.reference_format import parse_corpus
from tests.d2.reference import ReferenceConfig, ReferenceDataset, decode_reference_cursor
from tests.d2.test_resources import load_d1_resource, write_d1_resource
from tests.t1.reference import Identity, encode_artifact, fingerprint
from tests.tr3_reference.constants import (
    HELDOUT_SHARDS,
    RESUME_A1_SHARDS,
    RESUME_A2_SHARDS,
    TOKENIZER_FINGERPRINT,
    TRAIN_SHARDS,
    VOCAB_SIZE,
)


IDENTITY = Identity("raw", (), (), ())
CASES = (
    ("train", TRAIN_SHARDS, True),
    ("heldout", HELDOUT_SHARDS, False),
    ("resume-a1", RESUME_A1_SHARDS, True),
    ("resume-a2", RESUME_A2_SHARDS, False),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _batch(batch) -> dict[str, object]:
    return {
        "inputs": list(batch.inputs[0]),
        "targets": list(batch.targets[0]),
        "mask": list(batch.loss_mask[0]),
        "weight": sum(batch.loss_mask[0]),
    }


def _files(directory: Path) -> list[dict[str, object]]:
    return [
        {"path": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in sorted(directory.iterdir(), key=lambda item: item.name)
    ]


def _new_dataset(shards, digest: bytes, packing: bool) -> ReferenceDataset:
    return ReferenceDataset(
        shards,
        ReferenceConfig(1, 2, None, 1, packing),
        manifest_digest=digest,
        tokenizer_fingerprint=TOKENIZER_FINGERPRINT,
        vocab_size=VOCAB_SIZE,
    )


def _resume_trace(dataset: ReferenceDataset, *, start_ordinal: int, accumulation: int) -> dict[str, object]:
    epoch_start = dataset.snapshot()
    for _ in range(start_ordinal):
        if dataset.next_batch() is None:
            raise AssertionError("resume start exceeds rows")
    saved = dataset.snapshot()
    contributions = []
    crossings = 0
    epoch = 0
    for ordinal in range(accumulation):
        batch = dataset.next_batch()
        if batch is None:
            crossings += 1
            epoch += 1
            dataset.seek(epoch_start)
            batch = dataset.next_batch()
        if batch is None:
            raise AssertionError("nonempty resume corpus remained exhausted after reset")
        contributions.append(
            {"contribution_ordinal": ordinal, "batch": _batch(batch), "epoch": epoch}
        )
    return {
        "accumulation_steps": accumulation,
        "epoch_crossings": crossings,
        "epoch_start_cursor": epoch_start.hex(),
        "saved_cursor": saved.hex(),
        "saved_next_ordinal": decode_reference_cursor(saved).next_ordinal,
        "end_cursor": dataset.snapshot().hex(),
        "contributions": contributions,
        "total_weight": sum(item["batch"]["weight"] for item in contributions),
    }


def materialize(root: Path) -> dict[str, object]:
    """Populate a fresh output directory and return its strict corpus record."""

    if root.exists() or root.is_symlink():
        raise FileExistsError("TR3 reference output must be a fresh path")
    root.mkdir(parents=True)
    if fingerprint(IDENTITY) != TOKENIZER_FINGERPRINT:
        raise AssertionError("raw tokenizer fingerprint changed")
    tokenizer = encode_artifact(IDENTITY)
    (root / "tokenizer.etok").write_bytes(tokenizer)
    records: dict[str, object] = {}
    datasets: dict[str, ReferenceDataset] = {}
    for name, source_shards, packing in CASES:
        directory = root / name
        directory.mkdir()
        manifest = write_d1_resource(
            directory,
            source_shards,
            fingerprint=TOKENIZER_FINGERPRINT,
            vocab=VOCAB_SIZE,
        )
        shards, digest = load_d1_resource(
            directory,
            expected_fingerprint=TOKENIZER_FINGERPRINT,
            expected_vocab=VOCAB_SIZE,
            maximum_total_tokens=sum(map(len, source_shards)),
        )
        parsed = parse_corpus(directory)
        if tuple(parsed["tokens"]) != tuple(token for shard in source_shards for token in shard):
            raise AssertionError("D1 resource token sequence changed")
        dataset = _new_dataset(shards, digest, packing)
        datasets[name] = dataset
        start = dataset.snapshot()
        batches = []
        while (batch := dataset.next_batch()) is not None:
            batches.append(_batch(batch))
        records[name] = {
            "packing": packing,
            "shards": [list(shard) for shard in shards],
            "batches": batches,
            "start_cursor": start.hex(),
            "end_cursor": dataset.snapshot().hex(),
            "manifest_trailer_sha256": manifest[-32:].hex(),
            "files": _files(directory),
        }
    # The state at K=1 is deliberately the first resumed cursor.  R=3 is the
    # minimum suffix required by the proposal; this records cursor math only.
    resume_a1 = _new_dataset(RESUME_A1_SHARDS, bytes.fromhex(records["resume-a1"]["manifest_trailer_sha256"]), True)
    resume_a2 = _new_dataset(RESUME_A2_SHARDS, bytes.fromhex(records["resume-a2"]["manifest_trailer_sha256"]), False)
    return {
        "tokenizer": {
            "path": "tokenizer.etok",
            "sha256": hashlib.sha256(tokenizer).hexdigest(),
            "fingerprint": TOKENIZER_FINGERPRINT,
            "vocab_size": VOCAB_SIZE,
            "identity": {"utf8_policy": "raw", "specials": [], "prefix": [], "suffix": []},
        },
        "datasets": records,
        "resume_cases": {
            "accumulation-1": {"K": 1, "R": 3} | _resume_trace(resume_a1, start_ordinal=1, accumulation=1),
            "accumulation-2-unequal-mask": {"K": 1, "R": 3} | _resume_trace(resume_a2, start_ordinal=1, accumulation=2),
        },
    }
