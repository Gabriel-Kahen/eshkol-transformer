"""Independent integer/IEEE-bit oracle for the proposed N3K initializer.

This is development-only. Abstract path/storage fixtures specify a caller draw
schedule; they neither define nor enforce a public model lifecycle or registry.
"""
from __future__ import annotations

import hashlib
import struct
import unittest

U32 = (1 << 32) - 1
U64 = (1 << 64) - 1
U128 = (1 << 128) - 1
MATRIX_SHAPES = ((256, 4), (2, 4), (4, 4), (8, 4), (4, 8))
# Seed 1729, zero counter, row-major little-endian f32 payload. Independently
# cross-checked against the existing N2 development Philox uniform generator.
WHOLE_TENSOR_SHA256 = {
    (256, 4): "25d05727f2212d51ed0f840c48ce0788332ed3a22390419f898a37f005a40b96",
    (2, 4): "36df30436c9310f3ed7833c24b3cc5f279d8a03ae215e8a7d399eac6c5b93e12",
    (4, 4): "13931cd4e2b7fb296d0bd9e284f6e1d60fc90168d3282b4a62beb1f95f5e0ce6",
    (8, 4): "23bcc930b4fd34c348dd310292c00aa9d69bd68f3132b763fc8dbcbc91645910",
    (4, 8): "23bcc930b4fd34c348dd310292c00aa9d69bd68f3132b763fc8dbcbc91645910",
}


def philox(counter: int, key: int) -> tuple[int, ...]:
    """Integer-only Philox4x32-10, independent of native/N2 test helpers."""
    if not 0 <= counter <= U128 or not 0 <= key <= U64:
        raise ValueError("counter/key out of range")
    words = [(counter >> shift) & U32 for shift in (0, 32, 64, 96)]
    for round_number in range(10):
        products = (words[0] * 0xd2511f53, words[2] * 0xcd9e8d57)
        keys = ((key + round_number * 0x9e3779b9) & U32,
                ((key >> 32) + round_number * 0xbb67ae85) & U32)
        words = [(products[1] >> 32) ^ words[1] ^ keys[0], products[1] & U32,
                 (products[0] >> 32) ^ words[3] ^ keys[1], products[0] & U32]
    return tuple(words)


def weight_bits(lane: int) -> int:
    """Exact rational mapping (lane>>8 - 2**23) * 2**-28 to binary32.

    Numerator has at most 24 significant bits, so the mathematical result is
    exactly representable; computing its bits avoids native float evaluation.
    """
    if not 0 <= lane <= U32:
        raise ValueError("lane out of range")
    numerator = (lane >> 8) - (1 << 23)
    if numerator == 0:
        return 0
    magnitude = abs(numerator)
    leading = magnitude.bit_length() - 1
    exponent = leading - 28 + 127
    fraction = (magnitude << (23 - leading)) & 0x7fffff
    return (0x80000000 if numerator < 0 else 0) | (exponent << 23) | fraction


def signed(word: int) -> int:
    return word if word < (1 << 63) else word - (1 << 64)


def initialize(shape: tuple[int, int], state: tuple[int, ...]) -> tuple[list[int], tuple[int, ...]]:
    """Return row-major weight words and successor exact signed i64[4]."""
    if shape not in MATRIX_SHAPES:
        raise ValueError("unsupported exact matrix row")
    if len(state) != 4 or any(type(x) is not int or not -(1 << 63) <= x < (1 << 63)
                              for x in state):
        raise ValueError("expected exact signed i64[4]")
    if state[0] != 1:
        raise ValueError("version mismatch")
    key = state[1] & U64
    counter = (state[2] & U64) | ((state[3] & U64) << 64)
    count = shape[0] * shape[1]
    blocks = (count + 3) // 4
    if counter == U128 or counter + blocks > U128:
        raise OverflowError("counter exhausted")
    words = [weight_bits(lane) for offset in range(blocks)
             for lane in philox(counter + offset, key)][:count]
    successor = counter + blocks
    return words, (1, state[1], signed(successor & U64), signed(successor >> 64))


def unique_schedule(entries: list[tuple[str, str, tuple[int, int]]]) -> list[tuple[str, str, tuple[int, int]]]:
    """Test-only minimum UTF-8 path representative per abstract storage identity."""
    seen_paths: set[str] = set()
    groups: dict[str, tuple[str, str, tuple[int, int]]] = {}
    for path, identity, shape in entries:
        if path in seen_paths:
            raise ValueError("duplicate path")
        seen_paths.add(path)
        if shape not in MATRIX_SHAPES:
            raise ValueError("unsupported exact matrix row")
        if identity in groups and groups[identity][2] != shape:
            raise ValueError("tied shape mismatch")
        if identity not in groups or path.encode("utf-8") < groups[identity][0].encode("utf-8"):
            groups[identity] = (path, identity, shape)
    return sorted(groups.values(), key=lambda item: item[0].encode("utf-8"))


class InitializerReferenceTests(unittest.TestCase):
    def test_normative_philox_vectors(self) -> None:
        cases = ((0, 0, (0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8)),
                 (U128, U64, (0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd)),
                 (0x0370734413198a2e85a308d3243f6a88, 0x299f31d0a4093822,
                  (0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1)))
        for counter, key, expected in cases:
            self.assertEqual(philox(counter, key), expected)

    def test_mapping_endpoints_zero_sign_and_discarded_low_bits(self) -> None:
        for lane, expected in ((0, 0xbd000000), (255, 0xbd000000),
                               (0x7fffffff, 0xb1800000), (0x80000000, 0),
                               (0x800000ff, 0), (0xffffffff, 0x3cfffffe)):
            self.assertEqual(weight_bits(lane), expected)
        for lane in philox(0, 0) + philox(19, 0xfedcba9876543210):
            numeric = ((lane >> 8) * 2.**-24 - 0.5) * 2.**-4
            self.assertEqual(weight_bits(lane), struct.unpack("<I", struct.pack("<f", numeric))[0])
            self.assertEqual(weight_bits(lane), weight_bits(lane & 0xffffff00))
        counter = 0x2841331e2da21113bbb7d390f60f4efa
        self.assertEqual(philox(counter, 0), (0, 0xffffffff, 0x80000000, 0x7fffffff))
        words, _ = initialize((2, 4), (1, 0, signed(counter & U64), signed(counter >> 64)))
        self.assertEqual(words[:4], [0xbd000000, 0x3cfffffe, 0, 0xb1800000])

    def test_every_exact_row_whole_tensor_and_continuation(self) -> None:
        for shape in MATRIX_SHAPES:
            state = (1, signed(0xfedcba9876543210), 0, 0)
            first, after = initialize(shape, state)
            self.assertEqual((first, after), initialize(shape, state))
            second, final = initialize(shape, after)
            count = shape[0] * shape[1]
            whole = [weight_bits(lane) for c in range(count // 2)
                     for lane in philox(c, state[1] & U64)]
            self.assertEqual(first + second, whole)
            self.assertEqual(final, (1, state[1], count // 2, 0))
            self.assertNotEqual(first, second)

    def test_known_weight_words_and_whole_tensor_digests(self) -> None:
        expected = [0xbbcec0c0, 0x3cc2d38a, 0x3c715eb0, 0x3bd806d8,
                    0x3cf1c998, 0xbc0d3800, 0x3c4695d0, 0xbced0202]
        self.assertEqual(initialize((2, 4), (1, 0, 0, 0))[0], expected)
        for shape, digest in WHOLE_TENSOR_SHA256.items():
            words, _ = initialize(shape, (1, 1729, 0, 0))
            payload = struct.pack("<" + "I" * len(words), *words)
            self.assertEqual(hashlib.sha256(payload).hexdigest(), digest)

    def test_counter_carry_last_admitted_call_and_overflow(self) -> None:
        shape = (2, 4)
        _, next_state = initialize(shape, (1, 7, -1, 19))
        self.assertEqual(next_state, (1, 7, 1, 20))
        _, exhausted = initialize(shape, (1, 7, -3, -1))
        self.assertEqual(exhausted, (1, 7, -1, -1))
        for state in (exhausted, (1, 7, -2, -1)):
            with self.assertRaises(OverflowError):
                initialize(shape, state)

    def test_full_seed_word_signed_encoding(self) -> None:
        for seed in (0, 1729, (1 << 63) - 1, -(1 << 63), -1):
            words, next_state = initialize((2, 4), (1, seed, 0, 0))
            expected = [weight_bits(lane) for counter in range(2)
                        for lane in philox(counter, seed & U64)]
            self.assertEqual(words, expected)
            self.assertEqual(next_state, (1, seed, 2, 0))

    def test_unique_storage_order_is_input_order_invariant_and_draws_once(self) -> None:
        entries = [("z.token", "tied", (256, 4)), ("b.position", "position", (2, 4)),
                   ("a.head", "tied", (256, 4)), ("c.query", "query", (4, 4)),
                   ("d.key", "key", (4, 4)), ("e.value", "value", (4, 4)),
                   ("f.attention", "attention", (4, 4)),
                   ("g.expand", "expand", (8, 4)), ("h.contract", "contract", (4, 8))]
        schedule = unique_schedule(entries)
        self.assertEqual(schedule, unique_schedule(list(reversed(entries))))
        self.assertEqual([row[0] for row in schedule],
                         ["a.head", "b.position", "c.query", "d.key", "e.value",
                          "f.attention", "g.expand", "h.contract"])
        state = (1, 1729, 0, 0)
        values = {}
        starts = []
        for _, identity, shape in schedule:
            starts.append(state[2])
            values[identity], state = initialize(shape, state)
        self.assertEqual(starts, [0, 256, 258, 262, 266, 270, 274, 282])
        self.assertEqual(state, (1, 1729, 290, 0))
        self.assertEqual(len(values), 8)
        self.assertEqual(sum(len(x) for x in values.values()), 1160)
        duplicated_state = (1, 1729, 0, 0)
        wrong_values = {}
        for _, identity, shape in sorted(entries):
            wrong_values[identity], duplicated_state = initialize(shape, duplicated_state)
        self.assertNotEqual(values["tied"], wrong_values["tied"])
        self.assertNotEqual(state, duplicated_state)

    def test_utf8_path_bytes_choose_tied_representative(self) -> None:
        entries = [("é", "tie", (4, 4)), ("a", "tie", (4, 4)),
                   ("😀", "other", (2, 4)), ("z", "third", (4, 8))]
        self.assertEqual([row[0] for row in unique_schedule(entries)], ["a", "z", "😀"])

    def test_state_shape_and_tie_negatives(self) -> None:
        for shape in ((255, 4), (257, 4), (2, 3), (2, 5), (4, 2), (0, 4)):
            with self.assertRaises(ValueError):
                initialize(shape, (1, 0, 0, 0))
        for state in ((2, 0, 0, 0), (0, 0, 0, 0), (1, 0, 0),
                      (1, 1 << 63, 0, 0), (True, 0, 0, 0), (1, 0., 0, 0)):
            with self.assertRaises(ValueError):
                initialize((2, 4), state)
        for entries in ([("a", "x", (2, 4)), ("a", "y", (2, 4))],
                        [("a", "x", (2, 4)), ("b", "x", (4, 4))]):
            with self.assertRaises(ValueError):
                unique_schedule(entries)


if __name__ == "__main__":
    unittest.main()
