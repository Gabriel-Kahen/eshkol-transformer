"""Independent exact-rational cast oracle exercised through native K1 dispatch."""

from bisect import bisect_left
from fractions import Fraction
from random import Random
import struct
import subprocess
import sys


def power_of_two(exponent: int) -> Fraction:
    if exponent < 0:
        return Fraction(1, 1 << -exponent)
    return Fraction(1 << exponent)


def magnitude(word: int, fraction_width: int, bias: int) -> Fraction:
    exponent, fraction = divmod(word, 1 << fraction_width)
    if not exponent:
        return fraction * power_of_two(1 - bias - fraction_width)
    return ((1 << fraction_width) + fraction) * power_of_two(
        exponent - bias - fraction_width
    )


def f32_word(value: float) -> int:
    return int.from_bytes(struct.pack(">f", value), "big")


def f32_value(word: int) -> float:
    return struct.unpack(">f", word.to_bytes(4, "big"))[0]


def narrow_oracle(word: int, kind: str, neighbors: list[Fraction]) -> int | None:
    sign = (word >> 31) << 15
    absolute = word & 0x7FFFFFFF
    if absolute >= 0x7F800000:
        return None
    value = magnitude(absolute, 23, 127)
    if kind == "h" and value > neighbors[-2]:
        return None  # MP4 requires the original magnitude to fit finite f16.

    upper = bisect_left(neighbors, value)
    if neighbors[upper] == value:
        rounded = upper
    else:
        lower = upper - 1
        down = value - neighbors[lower]
        up = neighbors[upper] - value
        rounded = lower if (down < up or (down == up and not lower % 2)) else upper
    if rounded == len(neighbors) - 1:
        return None  # Rounding carried into the target's infinity encoding.

    result = sign | rounded
    if kind == "h":
        stdlib = int.from_bytes(struct.pack(">e", f32_value(word)), "big")
        assert result == stdlib, (hex(word), hex(result), hex(stdlib))
    return result


def widen_oracle(word: int, kind: str, neighbors: list[Fraction]) -> int | None:
    sign = (word >> 15) << 31
    absolute = word & 0x7FFF
    if absolute >= len(neighbors) - 1:
        return None
    result = sign | f32_word(float(neighbors[absolute]))
    if kind == "H":
        stdlib = f32_word(struct.unpack(">e", word.to_bytes(2, "big"))[0])
        assert result == stdlib, (hex(word), hex(result), hex(stdlib))
    return result


def cases(seed: int = 0x4D5034):
    random = Random(seed)
    for kind, fraction_width, bias, infinity in (
        ("h", 10, 15, 0x7C00),
        ("b", 7, 127, 0x7F80),
    ):
        neighbors = [
            magnitude(word, fraction_width, bias) for word in range(infinity + 1)
        ]
        # Last neighbor is the first unrepresentable power of two and gives
        # the overflow midpoint when rounded with the same exponent lattice.
        source_words = {random.getrandbits(32) for _ in range(8_192)}
        pairs = set(random.sample(range(infinity), 2_048))
        for exponent_boundary in range(0, infinity + 1, 1 << fraction_width):
            pairs.update(
                i
                for i in range(exponent_boundary - 2, exponent_boundary + 2)
                if 0 <= i < infinity
            )
        pairs.update(range(infinity - 5, infinity))
        for lower in pairs:
            midpoint = (neighbors[lower] + neighbors[lower + 1]) / 2
            encoded = f32_word(float(midpoint))
            for bits in (encoded - 1, encoded, encoded + 1):
                source_words.update((bits, bits | 0x80000000))
        for bits in (0, 1, 0x007FFFFF, 0x00800000, 0x477FE000, 0x477FE001,
                     0x477FF000, 0x7F7FFFFF, 0x7F800000, 0x7F800001, 0x7FC00000):
            source_words.update((bits, bits | 0x80000000))
        for bits in sorted(source_words):
            yield kind, bits, narrow_oracle(bits, kind, neighbors)

        widened_kind = kind.upper()
        storage_words = set(random.sample(range(1 << 16), 4_096))
        storage_words.update((0, 1, 0x8000, 0x8001, infinity - 1, infinity,
                              infinity + 1, 0x8000 | infinity, 0xFFFF))
        for bits in sorted(storage_words):
            yield widened_kind, bits, widen_oracle(bits, widened_kind, neighbors)


def main() -> None:
    test_cases = list(cases())
    actual = subprocess.run(
        [sys.argv[1], "--oracle"],
        input="".join(f"{kind} {bits:08x}\n" for kind, bits, _ in test_cases),
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()
    assert len(actual) == len(test_cases), (len(actual), len(test_cases))
    for (kind, bits, reference), observed in zip(test_cases, actual, strict=True):
        expected = "reject" if reference is None else f"{reference:08x}"
        assert observed == expected, (kind, f"{bits:08x}", observed, expected)
    print(f"MP4 Python Fraction and struct oracle: PASS ({len(test_cases):,} casts)")


if __name__ == "__main__":
    main()
