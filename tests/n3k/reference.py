"""Development-only independent mathematical oracles for bounded N3K rows.

These references use binary64 math.fsum/erf and explicit analytic derivatives,
not the provider's serial binary32 kernels. Inputs are rounded once to binary32;
native parity must use tolerances, while bit/order tests remain a separate gate.
Generated headers are inert test fixtures and are never production inputs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import struct
from typing import Callable, Sequence


EMBEDDING_ROWS = ((1, 2, 256, 4), (1, 2, 2, 4))
LINEAR_ROWS = ((1, 2, 4, 4), (1, 2, 4, 8), (1, 2, 8, 4), (1, 2, 4, 256))
GELU_ROW = (1, 2, 8)
RESIDUAL_ROW = (1, 2, 4)


def f32(value: float) -> float:
    return struct.unpack("!f", struct.pack("!f", value))[0]


def values(count: int, salt: int = 0) -> tuple[float, ...]:
    """Asymmetric, non-dyadic, finite fixtures with deterministic integer inputs."""
    return tuple(f32((((i * 37 + salt * 19) % 127) - 63) / 43.0)
                 for i in range(count))


def embedding(ids: Sequence[int], weight: Sequence[float], width: int) -> tuple[float, ...]:
    return tuple(weight[token * width + column] for token in ids for column in range(width))


def embedding_vjp(ids: Sequence[int], upstream: Sequence[float],
                  vocabulary: int, width: int) -> tuple[float, ...]:
    return tuple(math.fsum(upstream[position * width + column]
                          for position, token in enumerate(ids) if token == row)
                 for row in range(vocabulary) for column in range(width))


def linear(x: Sequence[float], weight: Sequence[float],
           din: int, dout: int) -> tuple[float, ...]:
    return tuple(math.fsum(a * b for a, b in zip(x[start:start + din], weight[row:row + din]))
                 for start in range(0, len(x), din) for row in range(0, dout * din, din))


def linear_vjp(x: Sequence[float], weight: Sequence[float], upstream: Sequence[float],
               din: int, dout: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    tokens = len(x) // din
    dx = tuple(math.fsum(upstream[token * dout + output] * weight[output * din + column]
                         for output in range(dout))
               for token in range(tokens) for column in range(din))
    dw = tuple(math.fsum(upstream[token * dout + output] * x[token * din + column]
                         for token in range(tokens))
               for output in range(dout) for column in range(din))
    return dx, dw


def gelu(x: Sequence[float]) -> tuple[float, ...]:
    return tuple(0.5 * value * (1.0 + math.erf(value / math.sqrt(2.0))) for value in x)


def gelu_vjp(x: Sequence[float], upstream: Sequence[float]) -> tuple[float, ...]:
    return tuple(gradient * (0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))
                            + value * math.exp(-0.5 * value * value) / math.sqrt(2.0 * math.pi))
                 for value, gradient in zip(x, upstream))


def residual(x: Sequence[float], branch: Sequence[float]) -> tuple[float, ...]:
    return tuple(a + b for a, b in zip(x, branch))


def central_vjp(function: Callable[..., Sequence[float]], operands: Sequence[Sequence[float]],
                upstream: Sequence[float], operand_index: int, step: float = 1e-5) -> tuple[float, ...]:
    """Differentiate the scalar dot(output, upstream), perturbing every element."""
    result = []
    for element in range(len(operands[operand_index])):
        plus = [list(value) for value in operands]
        minus = [list(value) for value in operands]
        plus[operand_index][element] += step
        minus[operand_index][element] -= step
        positive = math.fsum(a * b for a, b in zip(function(*plus), upstream))
        negative = math.fsum(a * b for a, b in zip(function(*minus), upstream))
        result.append((positive - negative) / (2.0 * step))
    return tuple(result)


@dataclass(frozen=True)
class Tensor:
    dtype: str
    shape: tuple[int, ...]
    data: tuple[int | float, ...]

    def __post_init__(self) -> None:
        if self.dtype not in ("f32", "i64") or math.prod(self.shape) != len(self.data):
            raise ValueError("invalid reference tensor dtype or shape")


@dataclass(frozen=True)
class Case:
    name: str
    operation: str
    row: tuple[int, ...]
    inputs: tuple[Tensor, ...]
    outputs: tuple[Tensor, ...]


def tensor(shape: Sequence[int], data: Sequence[float]) -> Tensor:
    return Tensor("f32", tuple(shape), tuple(f32(value) for value in data))


def cases() -> tuple[Case, ...]:
    result = []
    for row in EMBEDDING_ROWS:
        n, t, v, d = row
        for label, ids in (("boundary", (0, v - 1)), ("repeated", (min(7, v - 1),) * 2)):
            name = f"embedding_v{v}_{label}"
            index = Tensor("i64", (n, t), ids)
            weight, upstream = tensor((v, d), values(v * d, 1)), tensor((n, t, d), values(n * t * d, 2))
            result.append(Case(name + "_forward", "embedding.forward", row, (index, weight),
                               (tensor((n, t, d), embedding(ids, weight.data, d)),)))
            result.append(Case(name + "_backward", "embedding.backward", row, (index, upstream),
                               (tensor((v, d), embedding_vjp(ids, upstream.data, v, d)),)))
    for row in LINEAR_ROWS:
        n, t, din, dout = row
        name = f"linear_{din}_{dout}"
        x = tensor((n, t, din), values(n * t * din, 3))
        weight = tensor((dout, din), values(dout * din, 4))
        upstream = tensor((n, t, dout), values(n * t * dout, 5))
        y = tensor((n, t, dout), linear(x.data, weight.data, din, dout))
        dx, dw = linear_vjp(x.data, weight.data, upstream.data, din, dout)
        result.append(Case(name + "_forward", "linear.forward-no-bias", row, (x, weight), (y,)))
        result.append(Case(name + "_backward", "linear.backward-no-bias", row, (x, weight, upstream),
                           (tensor(x.shape, dx), tensor(weight.shape, dw))))
    x = tensor(GELU_ROW, (-8, -3, -1.75, -0.75, -0.25, -0.0, 0.0, 0.125,
                          0.5, 0.75, 1.25, 2, 3, 5, 7, 9))
    upstream = tensor(GELU_ROW, values(16, 6))
    result.append(Case("gelu_forward", "gelu.forward", GELU_ROW, (x,), (tensor(GELU_ROW, gelu(x.data)),)))
    result.append(Case("gelu_backward", "gelu.backward", GELU_ROW, (x, upstream),
                       (tensor(GELU_ROW, gelu_vjp(x.data, upstream.data)),)))
    x, branch, upstream = (tensor(RESIDUAL_ROW, values(8, salt)) for salt in (7, 8, 9))
    result.append(Case("residual_forward", "residual.forward", RESIDUAL_ROW, (x, branch),
                       (tensor(RESIDUAL_ROW, residual(x.data, branch.data)),)))
    result.append(Case("residual_backward", "residual.backward", RESIDUAL_ROW, (upstream,), (upstream, upstream)))
    return tuple(result)


def render_header(records: Sequence[Case] | None = None) -> bytes:
    """Emit ABI-independent dense tensor records; native tests adapt their views."""
    records = cases() if records is None else records
    lines = ["/* Generated development-only mathematical references. Do not commit. */",
             "#ifndef ET_N3K_REFERENCE_VECTORS_H", "#define ET_N3K_REFERENCE_VECTORS_H",
             "#include <stddef.h>", "#include <stdint.h>",
             "struct et_n3k_ref_tensor { const char *dtype; size_t rank; uint64_t shape[4]; size_t count; const void *data; };",
             "struct et_n3k_ref_case { const char *name; const char *operation; size_t rank; uint64_t row[4]; size_t input_count; size_t output_count; struct et_n3k_ref_tensor inputs[3]; struct et_n3k_ref_tensor outputs[2]; };", ""]
    tables = []
    for case_index, case in enumerate(records):
        groups = []
        for role, tensors in (("input", case.inputs), ("output", case.outputs)):
            entries = []
            for index, item in enumerate(tensors):
                symbol = f"et_n3k_ref_{case_index}_{role}_{index}"
                ctype = "float" if item.dtype == "f32" else "int64_t"
                literals = [float(value).hex() + "f" if item.dtype == "f32" else f"INT64_C({value})"
                            for value in item.data]
                lines.append(f"static const {ctype} {symbol}[] = {{")
                lines.extend("  " + ", ".join(literals[start:start + 8]) + ","
                             for start in range(0, len(literals), 8))
                lines.append("};")
                shape = ", ".join(map(str, item.shape))
                entries.append(f'{{"{item.dtype}", {len(item.shape)}, {{{shape}}}, {len(item.data)}, {symbol}}}')
            groups.append("{" + ", ".join(entries) + "}")
        row = ", ".join(map(str, case.row))
        tables.append(f'  {{"{case.name}", "{case.operation}", {len(case.row)}, {{{row}}}, '
                      f'{len(case.inputs)}, {len(case.outputs)}, {groups[0]}, {groups[1]}}},')
    lines.extend(("static const struct et_n3k_ref_case et_n3k_reference_cases[] = {", *tables, "};",
                  f"#define ET_N3K_REFERENCE_CASE_COUNT {len(records)}u", "#endif", ""))
    return "\n".join(lines).encode("ascii")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--header", required=True, type=Path)
    args = parser.parse_args()
    args.header.parent.mkdir(parents=True, exist_ok=True)
    args.header.write_bytes(render_header())
