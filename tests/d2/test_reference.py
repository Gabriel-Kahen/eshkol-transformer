from __future__ import annotations

import struct
import unittest

from tests.d2.reference import (
    CURSOR_BYTES,
    D2ReferenceError,
    MAX_I64,
    ReferenceConfig,
    ReferenceDataset,
    Row,
    batch_payload_bytes,
    decode_reference_cursor,
    encode_reference_cursor,
    logical_rows,
    packed_row_count,
    rejection_sample_words,
    resign_reference_cursor,
    unpacked_row_count,
    window_permutation,
    working_payload_bytes,
)


DIGEST = bytes.fromhex("10" * 32)


class RowReferenceTests(unittest.TestCase):
    def test_empty_singleton_short_exact_and_padded_shift_mask(self) -> None:
        cases = (
            ((), 4, ()),
            ((7,), 4, ()),
            ((7, 8), 4, (Row((7, 0, 0, 0), (8, 0, 0, 0), (True, False, False, False)),)),
            ((1, 2, 3, 4), 4, (Row((1, 2, 3, 0), (2, 3, 4, 0), (True, True, True, False)),)),
            ((1, 2, 3, 4, 5), 4, (Row((1, 2, 3, 4), (2, 3, 4, 5), (True, True, True, True)),)),
            (
                tuple(range(1, 10)),
                4,
                (
                    Row((1, 2, 3, 4), (2, 3, 4, 5), (True,) * 4),
                    Row((5, 6, 7, 8), (6, 7, 8, 9), (True,) * 4),
                ),
            ),
        )
        for tokens, width, expected in cases:
            with self.subTest(tokens=tokens):
                self.assertEqual(logical_rows((tokens,), width, True), expected)

    def test_packed_crosses_arbitrarily_many_shards_unpacked_does_not(self) -> None:
        shards = ((10,), (11,), (12,), (13,), (14,))
        expected = (Row((10, 11, 12, 13), (11, 12, 13, 14), (True,) * 4),)
        self.assertEqual(logical_rows(shards, 4, True), expected)
        self.assertEqual(logical_rows(shards, 4, False), ())

        shards = ((1, 2, 3), (4, 5))
        self.assertEqual(
            logical_rows(shards, 3, False),
            (
                Row((1, 2, 0), (2, 3, 0), (True, True, False)),
                Row((4, 0, 0), (5, 0, 0), (True, False, False)),
            ),
        )

    def test_row_count_formula_exact_and_overflow_edges(self) -> None:
        self.assertEqual([packed_row_count(length, 4) for length in range(11)],
                         [0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3])
        self.assertEqual(packed_row_count(MAX_I64, 1), MAX_I64 - 1)
        self.assertEqual(unpacked_row_count((1, 2, 5), 4), 2)
        with self.assertRaisesRegex(D2ReferenceError, "addition overflow"):
            unpacked_row_count((MAX_I64, MAX_I64), 1)


class ShuffleReferenceTests(unittest.TestCase):
    def test_frozen_window_vectors_and_window_membership(self) -> None:
        vectors = {
            (0, 0, 4): (),
            (1, 123, 1): (0,),
            (12, 7, 1): tuple(range(12)),
            (12, 0, 4): (3, 1, 0, 2, 6, 4, 7, 5, 10, 9, 11, 8),
            (12, 7, 5): (0, 2, 3, 4, 1, 8, 9, 5, 6, 7, 10, 11),
        }
        for (total, seed, window), expected in vectors.items():
            with self.subTest(total=total, seed=seed, window=window):
                actual = window_permutation(total, seed, window)
                self.assertEqual(actual, expected)
                self.assertEqual(sorted(actual), list(range(total)))
                for start in range(0, total, window):
                    self.assertEqual(
                        set(actual[start : start + window]),
                        set(range(start, min(total, start + window))),
                    )

    def test_determinism_seed_difference_and_invalid_values(self) -> None:
        self.assertEqual(window_permutation(100, None, 13), tuple(range(100)))
        expected = window_permutation(100, 314159, 13)
        self.assertEqual(window_permutation(100, 314159, 13), expected)
        self.assertNotEqual(window_permutation(100, 314160, 13), expected)
        for arguments in ((-1, 0, 1), (1, -1, 1), (1, 0, 0), (True, 0, 1)):
            with self.subTest(arguments=arguments), self.assertRaises(D2ReferenceError):
                window_permutation(*arguments)

    def test_rejection_sampling_consumes_high_tail_without_modulo_bias(self) -> None:
        # 2^64 mod 3 = 1, so UINT64_MAX is the sole rejected high-tail word.
        self.assertEqual(rejection_sample_words(3, ((1 << 64) - 1, 4)), (1, 2))
        # A power-of-two modulus divides the whole u64 domain; the maximum is valid.
        self.assertEqual(rejection_sample_words(8, ((1 << 64) - 1,)), (7, 1))
        with self.assertRaisesRegex(D2ReferenceError, "unsupported"):
            rejection_sample_words(3, ((1 << 64) - 1,))


class CursorResumeTests(unittest.TestCase):
    @staticmethod
    def dataset(*, digest: bytes = DIGEST, seed: int = 7) -> ReferenceDataset:
        return ReferenceDataset(
            ((1, 2), (3,), (4, 5, 6), (7, 8, 9, 10)),
            ReferenceConfig(2, 3, seed, 3, True),
            manifest_digest=digest,
            tokenizer_fingerprint="opaque-tokenizer-identity",
            vocab_size=257,
        )

    def test_cursor_resume_at_start_batch_boundary_final_batch_and_eos(self) -> None:
        uninterrupted = self.dataset()
        start = uninterrupted.snapshot()
        first = uninterrupted.next_batch()
        boundary = uninterrupted.snapshot()
        second = uninterrupted.next_batch()
        final = uninterrupted.snapshot()
        self.assertIsNone(uninterrupted.next_batch())
        self.assertEqual(uninterrupted.snapshot(), final)

        replay = self.dataset()
        replay.seek(start)
        self.assertEqual(replay.next_batch(), first)
        replay.seek(boundary)
        self.assertEqual(replay.next_batch(), second)
        self.assertIsNone(replay.next_batch())
        replay.seek(final)
        self.assertIsNone(replay.next_batch())

        # N=2 and M=3 makes the final physical row canonical masked zero filler.
        self.assertEqual(second.loss_mask[-1], (False, False, False))
        self.assertEqual(second.inputs[-1], (0, 0, 0))
        self.assertEqual(second.targets[-1], (0, 0, 0))

    def test_resume_at_packed_crossing_and_unpacked_shard_boundary(self) -> None:
        common = {
            "manifest_digest": DIGEST,
            "tokenizer_fingerprint": "opaque-tokenizer-identity",
            "vocab_size": 257,
        }
        packed = ReferenceDataset(
            ((1, 2), (3, 4), (5, 6)), ReferenceConfig(1, 2), **common
        )
        self.assertEqual(packed.next_batch().targets, ((2, 3),))
        packed_crossing = packed.snapshot()
        expected = packed.next_batch()
        replay = ReferenceDataset(
            ((1, 2), (3, 4), (5, 6)), ReferenceConfig(1, 2), **common
        )
        replay.seek(packed_crossing)
        self.assertEqual(replay.next_batch(), expected)

        unpacked_config = ReferenceConfig(1, 2, packing=False)
        unpacked = ReferenceDataset(((1, 2, 3), (4, 5, 6)), unpacked_config, **common)
        self.assertEqual(unpacked.next_batch().inputs, ((1, 2),))
        shard_boundary = unpacked.snapshot()
        expected = unpacked.next_batch()
        replay = ReferenceDataset(
            ((1, 2, 3), (4, 5, 6)), unpacked_config, **common
        )
        replay.seek(shard_boundary)
        self.assertEqual(replay.next_batch(), expected)

    def test_cursor_corruption_noncanonical_version_range_and_mismatch_are_atomic(self) -> None:
        dataset = self.dataset()
        dataset.next_batch()
        before = dataset.snapshot()

        mutations = [b"", before[:-1], before + b"x"]
        flipped = bytearray(before)
        flipped[20] ^= 1
        mutations.append(bytes(flipped))
        for raw in mutations:
            with self.subTest(length=len(raw)), self.assertRaises(D2ReferenceError):
                dataset.seek(raw)
            self.assertEqual(dataset.snapshot(), before)

        wrong_version = bytearray(before)
        struct.pack_into("<H", wrong_version, 8, 2)
        with self.assertRaisesRegex(D2ReferenceError, "version-mismatch"):
            dataset.seek(resign_reference_cursor(bytes(wrong_version)))
        self.assertEqual(dataset.snapshot(), before)

        one_over = bytearray(before)
        total = struct.unpack_from("<Q", one_over, 48)[0]
        struct.pack_into("<Q", one_over, 56, total + 1)
        with self.assertRaisesRegex(D2ReferenceError, "noncanonical ordinal"):
            dataset.seek(resign_reference_cursor(bytes(one_over)))
        self.assertEqual(dataset.snapshot(), before)

        other_corpus = self.dataset(digest=bytes.fromhex("20" * 32))
        with self.assertRaisesRegex(D2ReferenceError, "mismatch"):
            other_corpus.seek(before)
        self.assertEqual(other_corpus.next_ordinal, 0)

        other_config = self.dataset(seed=8)
        with self.assertRaisesRegex(D2ReferenceError, "mismatch"):
            other_config.seek(before)
        self.assertEqual(other_config.next_ordinal, 0)

    def test_private_envelope_is_fresh_deterministic_and_exact_length(self) -> None:
        dataset = self.dataset()
        first = dataset.snapshot()
        second = dataset.snapshot()
        self.assertIsNot(first, second)
        self.assertEqual(first, second)
        self.assertEqual(len(first), CURSOR_BYTES)
        identity, total, ordinal = decode_reference_cursor(first)
        self.assertEqual(identity, dataset.identity)
        self.assertEqual(total, len(dataset.rows))
        self.assertEqual(ordinal, 0)
        self.assertEqual(encode_reference_cursor(identity, total, ordinal), first)


class ArithmeticAndMemoryTests(unittest.TestCase):
    def test_batch_payload_exact_limit_one_under_and_one_over(self) -> None:
        self.assertEqual(batch_payload_bytes(3, 5, 255), 255)
        with self.assertRaisesRegex(D2ReferenceError, "configured limit"):
            batch_payload_bytes(3, 5, 254)

        exact_t = MAX_I64 // 17
        self.assertEqual(batch_payload_bytes(1, exact_t, MAX_I64), exact_t * 17)
        with self.assertRaisesRegex(D2ReferenceError, "multiplication overflow"):
            batch_payload_bytes(1, exact_t + 1, MAX_I64)
        with self.assertRaisesRegex(D2ReferenceError, "multiplication overflow"):
            batch_payload_bytes(1, 4, MAX_I64, size_max=67)

    def test_working_payload_formula_exact_and_one_over(self) -> None:
        # one 200-byte manifest window + one 1000-byte shard + 17*2*4 batch
        # + min(5,7)*8 row ordinals.
        expected = 200 + 1000 + 136 + 40
        self.assertEqual(
            working_payload_bytes(200, 1000, 2, 4, 5, 7, maximum_batch_bytes=136),
            expected,
        )
        with self.assertRaisesRegex(D2ReferenceError, "addition overflow"):
            working_payload_bytes(
                MAX_I64 - 1, 1, 1, 1, 1, 1,
                maximum_batch_bytes=17,
            )
        self.assertEqual(
            working_payload_bytes(200, 1000, 2, 4, 50, 0, maximum_batch_bytes=136),
            1336,
        )


if __name__ == "__main__":
    unittest.main()
