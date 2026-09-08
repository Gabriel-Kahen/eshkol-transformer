#!/usr/bin/env python3
"""Emit build-local native test vectors from the independent integer oracle."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.n3k.test_initializer_reference import MATRIX_SHAPES, initialize, signed


def cases() -> list[tuple[str, tuple[int, int], tuple[int, ...]]]:
    rows = [(f"row-{r}-{c}", (r, c), (1, 1729, 0, 0)) for r, c in MATRIX_SHAPES]
    rows += [(f"seed-{index}", (2, 4), (1, seed, 0, 0))
             for index, seed in enumerate((0, (1 << 63) - 1, -(1 << 63), -1))]
    rows += [("normative-custom", (2, 4),
              (1, signed(0x299f31d0a4093822), signed(0x85a308d3243f6a88),
               signed(0x0370734413198a2e))),
             ("carry", (2, 4), (1, -1, -1, 19)),
             ("last-admitted", (2, 4), (1, -1, -3, -1)),
             # Constructed by reversing the ten integer Philox rounds at key 0.
             # First lanes are exactly 0, ffffffff, 80000000, 7fffffff; this gives
             # native endpoint, center +0, and neighboring-center evidence.
             ("mapping-boundaries", (2, 4),
              (1, 0, signed(0xbbb7d390f60f4efa), signed(0x2841331e2da21113)))]
    # Abstract byte-sorted unique-storage sequence, not a public model path API.
    sequence = ((256, 4), (2, 4), (4, 4), (4, 4), (4, 4), (4, 4), (8, 4), (4, 8))
    state = (1, 1729, 0, 0)
    for index, shape in enumerate(sequence):
        rows.append((f"sequence-{index}", shape, state))
        _, state = initialize(shape, state)
    for shape in MATRIX_SHAPES:
        _, continued = initialize(shape, (1, 1729, 0, 0))
        rows.append((f"continued-{shape[0]}-{shape[1]}", shape, continued))
        blocks = shape[0] * shape[1] // 4
        rows.append((f"last-{shape[0]}-{shape[1]}", shape, (1, -1, -blocks-1, -1)))
    return rows


def i64(value: int) -> str:
    if value == -(1 << 63):
        return "INT64_MIN"
    return f"-INT64_C({-value})" if value < 0 else f"INT64_C({value})"


def render() -> str:
    output = ["/* Generated development oracle; never a production input. */",
              "#ifndef ET_N3K_COMPOSITION_REFERENCE_H",
              "#define ET_N3K_COMPOSITION_REFERENCE_H", "#include <stdint.h>",
              "#include <stddef.h>",
              "typedef struct n3k_init_reference_case {",
              "  const char *name; uint64_t shape[2]; int64_t state[4];",
              "  int64_t successor[4]; const uint32_t *words; size_t count;",
              "} n3k_init_reference_case;"]
    rows = cases()
    for index, (_, shape, state) in enumerate(rows):
        words, _ = initialize(shape, state)
        output.append(f"static const uint32_t n3k_init_words_{index}[] = {{")
        for start in range(0, len(words), 8):
            output.append("  " + ", ".join(f"UINT32_C(0x{x:08x})" for x in words[start:start+8]) + ",")
        output.append("};")
    output.append("static const n3k_init_reference_case n3k_init_references[] = {")
    for index, (name, shape, state) in enumerate(rows):
        words, successor = initialize(shape, state)
        output.append(f'  {{"{name}", {{{shape[0]}, {shape[1]}}}, '
                      f'{{{", ".join(map(i64, state))}}}, '
                      f'{{{", ".join(map(i64, successor))}}}, '
                      f'n3k_init_words_{index}, {len(words)}}},')
    output += ["};", "#endif", ""]
    return "\n".join(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(render(), encoding="ascii")
