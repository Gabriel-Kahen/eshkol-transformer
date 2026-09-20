"""Non-freezing L3S design probe; exact rational binary32, no runtime ABI.

This development-only program explores a proposed numerical domain. It is not
an implementation oracle for transcendental L2 expf/logf, an installed module,
or evidence that any L3S provider exists.
"""

from fractions import Fraction
import random
import unittest


MAX_BITS = 0x7F7FFFFF


def value(bits: int) -> Fraction:
    """Decode a finite positive binary32 bit pattern exactly."""
    if not 0 <= bits <= MAX_BITS:
        raise ValueError("expected nonnegative finite binary32")
    exponent, mantissa = bits >> 23, bits & 0x7FFFFF
    power = exponent - 150 if exponent else -149
    significand = mantissa + (0x800000 if exponent else 0)
    return Fraction(significand * (1 << max(power, 0)), 1 << max(-power, 0))


def rounded(number: Fraction | int) -> int:
    """Round an exact nonnegative rational to binary32, ties to even."""
    number = Fraction(number)
    if number < 0:
        raise ValueError("negative value")
    # At this midpoint RN-even selects the unrepresentable 2**128.
    if number >= Fraction(2**128 - 2**103):
        raise OverflowError("binary32 overflow")
    low, high = 0, MAX_BITS
    while low < high:
        middle = (low + high + 1) // 2
        if value(middle) <= number:
            low = middle
        else:
            high = middle - 1
    if low == MAX_BITS or value(low) == number:
        return low
    below, above = number - value(low), value(low + 1) - number
    return low + (above < below or (above == below and low % 2 == 1))


def add(left: int, right: int) -> int:
    return rounded(value(left) + value(right))


def multiply(left: int, right: int) -> int:
    return rounded(value(left) * value(right))


def divide(left: int, right: int) -> int:
    return rounded(value(left) / value(right))


def propose(losses: tuple[int, int], weights: tuple[int, int]):
    """Return proposed (numerator, W, mean, numerator seed, mean seed)."""
    weight = add(add(0, weights[0]), weights[1])
    if weight == 0:
        raise ValueError("positive total weight required")
    numerator = add(add(0, multiply(losses[0], weights[0])),
                    multiply(losses[1], weights[1]))
    return numerator, weight, divide(numerator, weight), weights, tuple(
        divide(item, weight) for item in weights
    )


class NumericalDesignProbe(unittest.TestCase):
    def test_exact_rounding_boundaries(self):
        self.assertEqual(rounded(Fraction(1, 2**150)), 0)
        self.assertEqual(rounded(Fraction(3, 2**150)), 2)
        self.assertEqual(rounded(value(0x007FFFFF)), 0x007FFFFF)
        self.assertEqual(rounded(value(0x00800000)), 0x00800000)
        self.assertEqual(rounded(value(MAX_BITS)), MAX_BITS)
        with self.assertRaises(OverflowError):
            rounded(2**128 - 2**103)

    def test_distinct_losses_and_denominator(self):
        result = propose((rounded(2), rounded(10)),
                         (rounded(Fraction(1, 2)), rounded(2)))
        self.assertEqual(result[:3], (rounded(21), rounded(Fraction(5, 2)),
                                     rounded(Fraction(42, 5))))
        self.assertEqual(result[4], (rounded(Fraction(1, 5)),
                                     rounded(Fraction(4, 5))))
        # Missing, repeated, count-based and reversed-pair normalization fail.
        numerator, weight, mean = result[:3]
        self.assertNotEqual(mean, numerator)
        self.assertNotEqual(mean, divide(mean, weight))
        self.assertNotEqual(mean, divide(numerator, rounded(2)))
        self.assertNotEqual(mean, propose((rounded(10), rounded(2)),
                                          result[3])[2])

    def test_bool_f32_and_zero_entries(self):
        losses = (rounded(2), rounded(10))
        ones = propose(losses, (rounded(1), rounded(1)))
        self.assertEqual(ones[2], rounded(6))
        self.assertEqual(ones[4], (rounded(Fraction(1, 2)),) * 2)
        selected = propose(losses, (0, rounded(1)))
        self.assertEqual(selected[2], rounded(10))
        self.assertEqual(selected[4], (0, rounded(1)))

    def test_zero_total_rejects(self):
        with self.assertRaises(ValueError):
            propose((rounded(1), rounded(2)), (0, 0))

    def test_all_zero_objective_and_positive_subnormal_weight(self):
        result = propose((0, 0), (1, 1))
        self.assertEqual(result[:3], (0, 2, 0))
        self.assertEqual(result[4], (rounded(Fraction(1, 2)),) * 2)

    def test_underflow_is_permitted(self):
        result = propose((1, 0), (rounded(Fraction(1, 2)), 0))
        self.assertEqual(result[0], 0)
        self.assertEqual(result[2], 0)
        self.assertEqual(result[4][0], rounded(1))
        # A positive mask entry may have a zero mathematical-mean seed in f32.
        result = propose((0, 0), (1, MAX_BITS))
        self.assertEqual(result[4], (0, rounded(1)))

    def test_overflow_rejects_even_if_real_mean_is_finite(self):
        with self.assertRaises(OverflowError):
            propose((rounded(1), rounded(1)), (MAX_BITS, MAX_BITS))
        with self.assertRaises(OverflowError):
            propose((MAX_BITS, 0), (rounded(2), 0))
        with self.assertRaises(OverflowError):
            propose((MAX_BITS, MAX_BITS), (rounded(1), rounded(1)))

    def test_seed_division_cannot_be_replaced_with_reciprocal(self):
        result = propose((0, 0), (rounded(3), rounded(4)))
        direct = result[4][0]
        reciprocal = multiply(rounded(3), divide(rounded(1), rounded(7)))
        self.assertEqual(direct, 0x3EDB6DB7)
        self.assertEqual(reciprocal, 0x3EDB6DB8)

    def test_only_final_mean_overflows(self):
        weights = (0x3DF9528A, 0x3E1C9EE4)
        products = tuple(multiply(MAX_BITS, item) for item in weights)
        self.assertEqual(products, (0x7DF95289, 0x7E1C9EE3))
        weight, numerator = add(*weights), add(*products)
        self.assertEqual((weight, numerator), (0x3E8CA414, 0x7E8CA414))
        self.assertEqual(value(numerator) / value(weight), 2**128)
        with self.assertRaises(OverflowError):
            propose((MAX_BITS, MAX_BITS), weights)

    def test_contraction_mutation_witness(self):
        small, large = rounded(Fraction(1, 2**24)), 0x3F800001
        separate = add(small, multiply(large, large))
        fused = rounded(value(small) + value(large) * value(large))
        self.assertEqual(separate, 0x3F800002)
        self.assertEqual(fused, 0x3F800003)

    def test_two_term_reversal_is_observationally_equivalent(self):
        rng = random.Random(8701)
        for _ in range(256):
            left, right = rng.randrange(MAX_BITS + 1), rng.randrange(MAX_BITS + 1)
            try:
                forward = add(add(0, left), right)
            except OverflowError:
                with self.assertRaises(OverflowError):
                    add(add(0, right), left)
            else:
                self.assertEqual(forward, add(add(0, right), left))
        # This three-term witness explains why widening would be necessary.
        big, one = rounded(2**24), rounded(1)
        self.assertNotEqual(add(add(big, one), one), add(add(one, one), big))


if __name__ == "__main__":
    unittest.main()
