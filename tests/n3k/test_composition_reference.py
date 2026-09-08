"""Independent development oracles for the bounded N3K composition proposal.

These functions are test-only math, not a runtime or a public model interface.
Native conformance tests must compare actual provider results with these oracles.
"""
from __future__ import annotations

import math
import struct
import unittest


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def token_to_head(values: list[float]) -> list[float]:
    """[1,2,4] -> [1,2,2,2], y[n,h,t,d]=x[n,t,h*2+d]."""
    if len(values) != 8:
        raise ValueError("expected eight elements")
    tokens = [values[:4], values[4:]]
    return [tokens[t][2 * h + d] for h in range(2)
            for t in range(2) for d in range(2)]


def head_to_token(values: list[float]) -> list[float]:
    """[1,2,2,2] -> [1,2,4], inverse of token_to_head."""
    if len(values) != 8:
        raise ValueError("expected eight elements")
    heads = [[values[4 * h + 2 * t:4 * h + 2 * t + 2]
              for t in range(2)] for h in range(2)]
    return [heads[h][t][d] for t in range(2)
            for h in range(2) for d in range(2)]


def ordered_sum(*operands: list[float]) -> list[float]:
    """Round each edge addition to f32 in supplied left-to-right order."""
    if len(operands) not in (2, 3) or not operands[0]:
        raise ValueError("requires two or three nonempty operands")
    if any(len(x) != len(operands[0]) for x in operands):
        raise ValueError("shape mismatch")
    if any(not math.isfinite(x) for operand in operands for x in operand):
        raise ValueError("nonfinite input")
    result = operands[0][:]
    for operand in operands[1:]:
        try:
            result = [f32(x + y) for x, y in zip(result, operand)]
        except OverflowError as error:
            raise ValueError("nonfinite intermediate") from error
        if any(not math.isfinite(x) for x in result):
            raise ValueError("nonfinite intermediate")
    return result


class CompositionReferenceTests(unittest.TestCase):
    def test_marked_layout_and_inverse(self) -> None:
        token = [0., 1., 10., 11., 100., 101., 110., 111.]
        head = [0., 1., 100., 101., 10., 11., 110., 111.]
        self.assertEqual(token_to_head(token), head)
        self.assertEqual(head_to_token(head), token)
        self.assertNotEqual(token, head)  # Detect relabel-without-permutation.

    def test_permutation_analytic_vjps_central_differences(self) -> None:
        primal = [0.125 * (i - 3) for i in range(8)]
        upstream = [0.0625 * (3 * i - 7) for i in range(8)]
        for forward, vjp in ((token_to_head, head_to_token),
                             (head_to_token, token_to_head)):
            expected = vjp(upstream)
            for index in range(8):
                plus, minus = primal[:], primal[:]
                plus[index] += 0.03125
                minus[index] -= 0.03125
                objective = lambda x: sum(a * b for a, b in zip(forward(x), upstream))
                numerical = (objective(plus) - objective(minus)) / 0.0625
                self.assertEqual(numerical, expected[index])
            self.assertNotEqual(expected, upstream)

    def test_finite_payload_bits_survive_layout_roundtrip(self) -> None:
        words = [0, 0x80000000, 1, 0x80000001, 0x00800000,
                 0x80800000, 0x7f7fffff, 0xff7fffff]
        values = [struct.unpack("<f", struct.pack("<I", x))[0] for x in words]
        self.assertEqual([bits(x) for x in head_to_token(token_to_head(values))], words)

    def test_binary_activation_and_tied_sum_every_element(self) -> None:
        for count in (8, 1024):
            lhs = [f32((i - count // 2) / 256) for i in range(count)]
            rhs = [f32((7 * i % 23 - 11) / 32) for i in range(count)]
            expected = [f32(a + b) for a, b in zip(lhs, rhs)]
            self.assertEqual(ordered_sum(lhs, rhs), expected)
            self.assertNotEqual(expected, lhs)  # Omitted head/embedding edge.
            self.assertNotEqual(expected, rhs)
            self.assertNotEqual(expected, ordered_sum(lhs, lhs))  # Wrong edge.

    def test_three_branch_order_and_edge_sensitivity(self) -> None:
        q = [16777216., 3., 0.5, 7., 2., 4., -2., 1.]
        k = [-16777216., -4., 1.5, -1., 9., -3., 5., 2.]
        v = [1., 2., -4., 0.25, -7., 1., 3., -9.]
        expected = [1., 1., -2., 6.25, 4., 2., 6., -6.]
        self.assertEqual(ordered_sum(q, k, v), expected)
        self.assertEqual(ordered_sum(k, v)[0], -16777215.)
        self.assertNotEqual(ordered_sum(q, v, k), expected)
        for operands in ((q, k), (q, v), (k, v), (q, k, k), (q, q, v)):
            self.assertNotEqual(ordered_sum(*operands), expected)
        self.assertEqual(ordered_sum([1.], [16777216.], [-16777216.]), [0.])
        self.assertEqual(ordered_sum([1.], ordered_sum([16777216.], [-16777216.])), [1.])

    def test_sum_analytic_vjp_all_edges(self) -> None:
        upstream = [0.125 * (i - 2) for i in range(8)]
        for count in (2, 3):
            primals = [[0.25 * (i + edge) for i in range(8)] for edge in range(count)]
            for edge in range(count):
                for index in range(8):
                    plus = [p[:] for p in primals]
                    minus = [p[:] for p in primals]
                    plus[edge][index] += 0.03125
                    minus[edge][index] -= 0.03125
                    objective = lambda p: sum(x * y for x, y in zip(ordered_sum(*p), upstream))
                    self.assertEqual((objective(plus) - objective(minus)) / 0.0625,
                                     upstream[index])

    def test_oracle_rejects_shapes_nonfinite_and_intermediate_overflow(self) -> None:
        for bad in ([], [1.] * 7, [1.] * 9):
            for permutation in (token_to_head, head_to_token):
                with self.assertRaises(ValueError):
                    permutation(bad)
        for bad in (math.inf, -math.inf, math.nan):
            with self.assertRaises(ValueError):
                ordered_sum([bad], [0.])
        maximum = struct.unpack("<f", bytes.fromhex("ffff7f7f"))[0]
        with self.assertRaises(ValueError):
            ordered_sum([maximum], [maximum], [-maximum])
        with self.assertRaises(ValueError):
            ordered_sum([1.], [1., 2.])


if __name__ == "__main__":
    unittest.main()
