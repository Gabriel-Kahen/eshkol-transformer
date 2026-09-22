"""Independent development oracle: rational RN-even binary32 and Decimal exp.

No host float arithmetic, host libm, native provider, or NumPy is used to decide
rounding or the exp overflow boundary. Decimal precision is checked twice.
"""
from decimal import Decimal, localcontext
from fractions import Fraction
import unittest

MAX_BITS = 0x7F7FFFFF
INF_BITS = 0x7F800000
# Exact f32 inputs immediately surrounding the independently derived threshold.
EXP_NEIGHBORS = (0x42B17216, 0x42B17217, 0x42B17218, 0x42B17219)


def value(bits):
    if bits == 0x80000000:
        bits = 0
    if not 0 <= bits <= MAX_BITS:
        raise ValueError("requires nonnegative finite binary32")
    exponent, fraction = bits >> 23, bits & 0x7FFFFF
    power = exponent - 150 if exponent else -149
    significand = fraction + (0x800000 if exponent else 0)
    return Fraction(significand * (1 << max(power, 0)), 1 << max(-power, 0))


def rounded(number):
    number = Fraction(number)
    if number < 0:
        raise ValueError("negative rational")
    if number >= 2**128 - 2**103:
        return INF_BITS
    lo, hi = 0, MAX_BITS
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if value(mid) <= number:
            lo = mid
        else:
            hi = mid - 1
    if lo == MAX_BITS or value(lo) == number:
        return lo
    lower, upper = number - value(lo), value(lo + 1) - number
    return lo + int(upper < lower or (upper == lower and lo & 1))


def add(a, b):
    return rounded(value(a) + value(b))


def multiply(a, b):
    return rounded(value(a) * value(b))


def divide(a, b):
    return rounded(value(a) / value(b))


def exp_bits(bits):
    exact = value(bits)
    answers = []
    for precision in (100, 180):
        with localcontext() as context:
            context.prec = precision
            argument = Decimal(exact.numerator) / Decimal(exact.denominator)
            answers.append(rounded(Fraction(argument.exp())))
    if answers[0] != answers[1]:
        raise AssertionError("Decimal precision insufficient for binary32 rounding")
    return answers[0]


def reduce_bool(losses, mask):
    weights = tuple(rounded(x) for x in mask)
    return add(add(0, multiply(losses[0], weights[0])), multiply(losses[1], weights[1])), rounded(sum(mask))


def accumulate(state, batch):
    n, w, c, tokens, batches = state
    bn, bw, bc, active = batch
    return add(n, bn), add(w, bw), add(c, bc), tokens + active, batches + 1


def finalize(state):
    n, w, c = state[:3]
    loss, accuracy = divide(n, w), divide(c, w)
    return loss, w, exp_bits(loss), accuracy


class RationalReference(unittest.TestCase):
    def test_round_even_and_subnormal(self):
        self.assertEqual(rounded(Fraction(1, 2**150)), 0)
        self.assertEqual(rounded(Fraction(3, 2**150)), 2)
        self.assertEqual(rounded(value(0x3F800000) + Fraction(1, 2**24)), 0x3F800000)
        self.assertEqual(rounded(value(0x3F800001) + Fraction(1, 2**24)), 0x3F800002)
        for bits in (1, 0x007FFFFF, 0x00800000, MAX_BITS):
            self.assertEqual(rounded(value(bits)), bits)
        self.assertEqual(rounded(2**128 - 2**103), INF_BITS)
        self.assertEqual(divide(1, rounded(16384)), 0)

    def test_three_batch_order_witness(self):
        batches = [(rounded(64), rounded(1), 0, 1),
                   (rounded(Fraction(1, 2**18)), rounded(1), 0, 1)]
        state = (0, 0, 0, 0, 0)
        for batch in (batches[0], batches[1], batches[1]):
            state = accumulate(state, batch)
        wider = rounded(value(batches[0][0]) + 2 * value(batches[1][0]))
        reordered = add(batches[0][0], add(batches[1][0], batches[1][0]))
        self.assertEqual(state[0], 0x42800000)
        self.assertEqual(wider, 0x42800001)
        self.assertEqual(reordered, wider)
        self.assertEqual(finalize(state)[0], 0x41AAAAAB)
        self.assertEqual(divide(wider, rounded(3)), 0x41AAAAAC)
        self.assertLess(finalize(state)[2], INF_BITS)

    def test_mean_of_means(self):
        state = accumulate(accumulate((0, 0, 0, 0, 0),
                                      (rounded(1), rounded(1), 0, 1)),
                           (rounded(6), rounded(2), 0, 2))
        self.assertEqual(finalize(state)[0], rounded(Fraction(7, 3)))
        self.assertNotEqual(finalize(state)[0], rounded(2))

    def test_exp_independent_frozen_neighbors(self):
        expected = (0x7F7FFF04, 0x7F7FFF84, INF_BITS, INF_BITS)
        self.assertEqual(tuple(exp_bits(x) for x in EXP_NEIGHBORS), expected)
        self.assertEqual(exp_bits(0), rounded(1))

    def test_equivalent_controls(self):
        # Canonical Boolean products are exact; a fused product cannot differ.
        for a in (0, 1):
            for b in (0, 1):
                self.assertEqual(add(rounded(a), rounded(b)), rounded(a + b))
        for a, b in ((1, 2), (0x42800000, 0x36800000), (MAX_BITS, 0)):
            self.assertEqual(add(a, b), add(b, a))
        for tokens in range(16385):
            self.assertEqual(value(rounded(tokens)), tokens)


if __name__ == "__main__":
    unittest.main()
