from __future__ import annotations

import hashlib
from pathlib import Path
import struct
import tempfile
import unittest

from tests.d1.reference_format import parse_corpus, parse_shard
from tests.d2.reference import MAX_I64, ReferenceConfig, ReferenceDataset


MANIFEST_FIXED = 96
SHARD_FIXED = 80
RECORD_BYTES = 56


class ResourceError(ValueError):
    def __init__(self, category: str, message: str) -> None:
        self.category = category
        super().__init__(f"{category}: {message}")


def _shard_bytes(index: int, tokens: tuple[int, ...], fingerprint: bytes, vocab: int) -> bytes:
    header_bytes = SHARD_FIXED + len(fingerprint)
    file_bytes = header_bytes + 8 * len(tokens) + 32
    data = bytearray(file_bytes)
    struct.pack_into("<8sHHIIII4xQQQQI4xQ", data, 0, b"ESHKTSH1", 1, 0,
                     header_bytes, 0, 1, 1, index, len(tokens), 8 * len(tokens),
                     vocab, len(fingerprint), file_bytes)
    data[SHARD_FIXED:header_bytes] = fingerprint
    for offset, token in enumerate(tokens):
        struct.pack_into("<Q", data, header_bytes + 8 * offset, token)
    data[-32:] = hashlib.sha256(data[:-32]).digest()
    return bytes(data)


def write_d1_resource(
    directory: Path,
    shards: tuple[tuple[int, ...], ...],
    *,
    fingerprint: str = "opaque-tokenizer-identity",
    vocab: int = 257,
) -> bytes:
    """Write a tiny deterministic accepted-D1 resource for D2 test consumption."""

    encoded_fingerprint = fingerprint.encode("utf-8")
    shard_limit = len(shards[0]) if shards else 1
    if any(not shard or len(shard) != shard_limit for shard in shards[:-1]):
        raise ValueError("nonfinal test shards must be nonempty and canonical-sized")
    if shards and (not shards[-1] or len(shards[-1]) > shard_limit):
        raise ValueError("final test shard must contain 1..shard-limit tokens")

    encoded_shards = []
    for index, tokens in enumerate(shards):
        raw = _shard_bytes(index, tokens, encoded_fingerprint, vocab)
        (directory / f"shard-{index:016d}.ets").write_bytes(raw)
        encoded_shards.append(raw)

    header_bytes = MANIFEST_FIXED + len(encoded_fingerprint)
    manifest_bytes = header_bytes + RECORD_BYTES * len(shards) + 32
    data = bytearray(manifest_bytes)
    struct.pack_into(
        "<8sHHIIII4xQQQQQIIQ8x", data, 0, b"ESHKTCM1", 1, 0,
        header_bytes, 0, 1, 1, len(shards), sum(map(len, shards)),
        sum(map(len, encoded_shards)), vocab, shard_limit, len(encoded_fingerprint),
        RECORD_BYTES, manifest_bytes,
    )
    data[MANIFEST_FIXED:header_bytes] = encoded_fingerprint
    for index, (tokens, raw) in enumerate(zip(shards, encoded_shards)):
        offset = header_bytes + index * RECORD_BYTES
        struct.pack_into("<QQQ", data, offset, index, len(tokens), len(raw))
        data[offset + 24 : offset + 56] = raw[-32:]
    data[-32:] = hashlib.sha256(data[:-32]).digest()
    (directory / "manifest.etm").write_bytes(data)
    return bytes(data)


def load_d1_resource(
    directory: Path,
    *,
    expected_fingerprint: str,
    expected_vocab: int,
    maximum_manifest_bytes: int = MAX_I64,
    maximum_shard_bytes: int = MAX_I64,
    maximum_total_tokens: int = MAX_I64,
) -> tuple[tuple[tuple[int, ...], ...], bytes]:
    """Fully validate a tiny D1 resource before exposing any oracle input."""

    try:
        if (
            isinstance(maximum_manifest_bytes, bool)
            or not isinstance(maximum_manifest_bytes, int)
            or not 0 < maximum_manifest_bytes <= MAX_I64
            or isinstance(maximum_shard_bytes, bool)
            or not isinstance(maximum_shard_bytes, int)
            or not 0 < maximum_shard_bytes <= MAX_I64
            or isinstance(maximum_total_tokens, bool)
            or not isinstance(maximum_total_tokens, int)
            or not 0 <= maximum_total_tokens <= MAX_I64
        ):
            raise ResourceError("invalid-argument", "invalid D1 policy limit")
        manifest = (directory / "manifest.etm").read_bytes()
        if len(manifest) > maximum_manifest_bytes:
            raise ResourceError("corrupt-data", "manifest exceeds policy limit")
        corpus = parse_corpus(directory)
        if corpus["fingerprint"] != expected_fingerprint or corpus["vocab_size"] != expected_vocab:
            raise ResourceError("invalid-argument", "tokenizer identity or vocabulary mismatch")
        if len(corpus["tokens"]) > maximum_total_tokens:
            raise ResourceError("corrupt-data", "token total exceeds policy limit")
        shards = tuple(
            tuple(parse_shard(directory / f"shard-{index:016d}.ets")["tokens"])
            for index in range(int(corpus["shard_count"]))
        )
        if any(
            (directory / f"shard-{index:016d}.ets").stat().st_size > maximum_shard_bytes
            for index in range(int(corpus["shard_count"]))
        ):
            raise ResourceError("corrupt-data", "shard exceeds policy limit")
        # Cursor corpus identity is D1's stored trailer: SHA-256 of the bytes
        # preceding the trailer, not SHA-256 of the complete manifest.
        return shards, manifest[-32:]
    except ResourceError:
        raise
    except FileNotFoundError as error:
        raise ResourceError("io", "missing manifest or shard") from error
    except (OSError, ValueError) as error:
        raise ResourceError("corrupt-data", "invalid D1 resource") from error


class D1ResourceAdversarialTests(unittest.TestCase):
    def test_deterministic_resource_load_and_cross_shard_reference(self) -> None:
        with tempfile.TemporaryDirectory(prefix="d2-resource-") as raw_root:
            root = Path(raw_root)
            manifest = write_d1_resource(root, ((10,), (11,), (12,), (13,), (14,)))
            shards, digest = load_d1_resource(
                root, expected_fingerprint="opaque-tokenizer-identity", expected_vocab=257
            )
            self.assertEqual(digest, hashlib.sha256(manifest[:-32]).digest())
            self.assertEqual(digest, manifest[-32:])
            dataset = ReferenceDataset(
                shards,
                ReferenceConfig(1, 4, 0, 1, True),
                manifest_digest=digest,
                tokenizer_fingerprint="opaque-tokenizer-identity",
                vocab_size=257,
            )
            batch = dataset.next_batch()
            self.assertIsNotNone(batch)
            self.assertEqual(batch.inputs, ((10, 11, 12, 13),))
            self.assertEqual(batch.targets, ((11, 12, 13, 14),))
            self.assertEqual(batch.loss_mask, ((True, True, True, True),))

    def test_empty_resource_is_valid_and_stably_exhausted(self) -> None:
        with tempfile.TemporaryDirectory(prefix="d2-empty-") as raw_root:
            root = Path(raw_root)
            write_d1_resource(root, ())
            shards, digest = load_d1_resource(
                root, expected_fingerprint="opaque-tokenizer-identity", expected_vocab=257
            )
            dataset = ReferenceDataset(
                shards,
                ReferenceConfig(2, 4),
                manifest_digest=digest,
                tokenizer_fingerprint="opaque-tokenizer-identity",
                vocab_size=257,
            )
            cursor = dataset.snapshot()
            self.assertIsNone(dataset.next_batch())
            self.assertIsNone(dataset.next_batch())
            self.assertEqual(dataset.snapshot(), cursor)

    def test_missing_corrupt_and_identity_mismatch_are_distinct(self) -> None:
        with tempfile.TemporaryDirectory(prefix="d2-adversarial-") as raw_root:
            root = Path(raw_root)
            with self.assertRaisesRegex(ResourceError, "io"):
                load_d1_resource(root, expected_fingerprint="fp", expected_vocab=2)

            # Orphan shards are not a publication without manifest.etm.
            (root / "shard-0000000000000000.ets").write_bytes(b"orphan")
            with self.assertRaisesRegex(ResourceError, "io"):
                load_d1_resource(root, expected_fingerprint="fp", expected_vocab=2)
            (root / "shard-0000000000000000.ets").unlink()

            write_d1_resource(root, ((1, 2), (3, 4)))
            with self.assertRaisesRegex(ResourceError, "invalid-argument"):
                load_d1_resource(root, expected_fingerprint="wrong", expected_vocab=257)
            with self.assertRaisesRegex(ResourceError, "invalid-argument"):
                load_d1_resource(
                    root, expected_fingerprint="opaque-tokenizer-identity", expected_vocab=258
                )

            shard = root / "shard-0000000000000001.ets"
            damaged = bytearray(shard.read_bytes())
            damaged[-33] ^= 1
            shard.write_bytes(damaged)
            with self.assertRaisesRegex(ResourceError, "corrupt-data"):
                load_d1_resource(
                    root, expected_fingerprint="opaque-tokenizer-identity", expected_vocab=257
                )

    def test_manifest_shard_and_token_policy_limits_are_exact(self) -> None:
        with tempfile.TemporaryDirectory(prefix="d2-limits-") as raw_root:
            root = Path(raw_root)
            manifest = write_d1_resource(root, ((1, 2), (3, 4)))
            shard_bytes = (root / "shard-0000000000000000.ets").stat().st_size
            common = {
                "expected_fingerprint": "opaque-tokenizer-identity",
                "expected_vocab": 257,
            }
            load_d1_resource(
                root,
                **common,
                maximum_manifest_bytes=len(manifest),
                maximum_shard_bytes=shard_bytes,
                maximum_total_tokens=4,
            )
            mutations = (
                {"maximum_manifest_bytes": len(manifest) - 1},
                {"maximum_shard_bytes": shard_bytes - 1},
                {"maximum_total_tokens": 3},
            )
            for limits in mutations:
                with self.subTest(limits=limits), self.assertRaisesRegex(
                    ResourceError, "corrupt-data"
                ):
                    load_d1_resource(root, **common, **limits)
            for limits in (
                {"maximum_manifest_bytes": 0},
                {"maximum_shard_bytes": True},
                {"maximum_total_tokens": -1},
            ):
                with self.subTest(limits=limits), self.assertRaisesRegex(
                    ResourceError, "invalid-argument"
                ):
                    load_d1_resource(root, **common, **limits)

    def test_missing_late_shard_exposes_no_partial_rows(self) -> None:
        with tempfile.TemporaryDirectory(prefix="d2-late-missing-") as raw_root:
            root = Path(raw_root)
            write_d1_resource(root, ((1, 2), (3, 4), (5, 6)))
            (root / "shard-0000000000000002.ets").unlink()
            with self.assertRaisesRegex(ResourceError, "io"):
                load_d1_resource(
                    root, expected_fingerprint="opaque-tokenizer-identity", expected_vocab=257
                )

    def test_unpacked_zero_row_singleton_corruption_is_still_rejected_at_open(self) -> None:
        with tempfile.TemporaryDirectory(prefix="d2-singleton-corrupt-") as raw_root:
            root = Path(raw_root)
            write_d1_resource(root, ((1,), (2,), (3,)))
            shard = root / "shard-0000000000000001.ets"
            damaged = bytearray(shard.read_bytes())
            damaged[-33] ^= 1
            shard.write_bytes(damaged)
            with self.assertRaisesRegex(ResourceError, "corrupt-data"):
                load_d1_resource(
                    root,
                    expected_fingerprint="opaque-tokenizer-identity",
                    expected_vocab=257,
                )


if __name__ == "__main__":
    unittest.main()
