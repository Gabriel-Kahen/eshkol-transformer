#!/usr/bin/env python3
"""Convert one strict frozen N2 Q0 fixture to an inert native test header."""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.q0.oracle_format import load_fixture

DTYPE_ENUM = {"float32": "ET_N2_REFERENCE_F32", "int64": "ET_N2_REFERENCE_I64"}
ROLE_ENUM = {
    "input": "ET_N2_REFERENCE_INPUT",
    "expected": "ET_N2_REFERENCE_EXPECTED",
    "analytic_gradient": "ET_N2_REFERENCE_ANALYTIC_GRADIENT",
}


def tensor_symbol(name: str) -> str:
    return "et_n2_ref_" + re.sub(r"[^A-Za-z0-9_]", "_", name)


def _signed_i64(encoded: str) -> int:
    return int.from_bytes(bytes.fromhex(encoded), "big", signed=True)


def _i64_literal(value: int) -> str:
    if value == -(1 << 63):
        return "INT64_MIN"
    if value < 0:
        return f"-INT64_C({-value})"
    return f"INT64_C({value})"


def _array(declaration: str, values: Sequence[str], per_line: int) -> list[str]:
    lines = [declaration + " = {"]
    for start in range(0, len(values), per_line):
        lines.append("  " + ", ".join(values[start:start + per_line]) + ",")
    lines.append("};")
    return lines


def _referenced_names(payload: dict[str, Any]) -> set[str]:
    return {
        name
        for case in payload["cases"]
        for name in case["inputs"] + case["expectation"]["outputs"]
    }


def render_header(fixture: bytes) -> bytes:
    payload = load_fixture(fixture)
    referenced = _referenced_names(payload)
    tensors = [tensor for tensor in payload["tensors"] if tensor["name"] in referenced]
    if len(tensors) != len(referenced):
        raise RuntimeError("fixture case references were not completely resolved")
    unsupported = sorted({tensor["dtype"] for tensor in tensors} - set(DTYPE_ENUM))
    if unsupported:
        raise RuntimeError(f"unsupported native reference dtype(s): {unsupported}")
    symbols = [tensor_symbol(tensor["name"]) for tensor in tensors]
    if len(symbols) != len(set(symbols)):
        raise RuntimeError("tensor names collide after conversion to C identifiers")
    indices = {tensor["name"]: index for index, tensor in enumerate(tensors)}

    lines = [
        "/* Generated from the strict Q0 fixture; do not edit by hand. */",
        "#ifndef ESHKOL_TRANSFORMER_N2_REFERENCE_VECTORS_H",
        "#define ESHKOL_TRANSFORMER_N2_REFERENCE_VECTORS_H",
        "",
        "#include <stddef.h>",
        "#include <stdint.h>",
        "",
        "#define ET_N2_REFERENCE_FIXTURE_SHA256 \\",
        f'  "{hashlib.sha256(fixture).hexdigest()}"',
        f"#define ET_N2_REFERENCE_TENSOR_COUNT UINT32_C({len(tensors)})",
        f"#define ET_N2_REFERENCE_CASE_COUNT UINT32_C({len(payload['cases'])})",
        "",
        "enum et_n2_reference_dtype {",
        "  ET_N2_REFERENCE_F32 = 1,",
        "  ET_N2_REFERENCE_I64 = 2,",
        "};",
        "",
        "enum et_n2_reference_role {",
        "  ET_N2_REFERENCE_INPUT = 1,",
        "  ET_N2_REFERENCE_EXPECTED = 2,",
        "  ET_N2_REFERENCE_ANALYTIC_GRADIENT = 3,",
        "};",
        "",
        "struct et_n2_reference_tensor {",
        "  const char *name;",
        "  uint32_t dtype;",
        "  uint32_t role;",
        "  uint32_t rank;",
        "  const int64_t *shape;",
        "  uint64_t element_count;",
        "  const void *words;",
        "};",
        "",
        "struct et_n2_reference_case {",
        "  const char *name;",
        "  const char *operation;",
        "  const char *kind;",
        "  uint32_t input_count;",
        "  const uint16_t *inputs;",
        "  uint32_t output_count;",
        "  const uint16_t *outputs;",
        "};",
        "",
    ]

    for tensor, symbol in zip(tensors, symbols):
        shape = tensor["shape"]
        if shape:
            lines.extend(_array(
                f"static const int64_t {symbol}_shape[]",
                [_i64_literal(dimension) for dimension in shape], 8,
            ))
        if tensor["dtype"] == "float32":
            words = [f"UINT32_C(0x{encoded})" for encoded in tensor["data"]]
            declaration = f"static const uint32_t {symbol}_words[]"
            per_line = 8
        else:
            words = [_i64_literal(_signed_i64(encoded)) for encoded in tensor["data"]]
            declaration = f"static const int64_t {symbol}_words[]"
            per_line = 4
        lines.extend(_array(declaration, words, per_line))
        lines.append("")

    lines.append("static const struct et_n2_reference_tensor et_n2_reference_tensors[] = {")
    for tensor, symbol in zip(tensors, symbols):
        count = len(tensor["data"])
        shape = f"{symbol}_shape" if tensor["shape"] else "NULL"
        lines.extend((
            "  {",
            f'    "{tensor["name"]}", {DTYPE_ENUM[tensor["dtype"]]},',
            f"    {ROLE_ENUM[tensor['role']]}, UINT32_C({len(tensor['shape'])}), {shape},",
            f"    UINT64_C({count}), {symbol}_words,",
            "  },",
        ))
    lines.extend(("};", ""))

    for case in payload["cases"]:
        symbol = "et_n2_ref_case_" + re.sub(r"[^A-Za-z0-9_]", "_", case["name"])
        lines.extend(_array(
            f"static const uint16_t {symbol}_inputs[]",
            [f"UINT16_C({indices[name]})" for name in case["inputs"]], 8,
        ))
        lines.extend(_array(
            f"static const uint16_t {symbol}_outputs[]",
            [f"UINT16_C({indices[name]})" for name in case["expectation"]["outputs"]], 8,
        ))
        lines.append("")

    lines.append("static const struct et_n2_reference_case et_n2_reference_cases[] = {")
    for case in payload["cases"]:
        symbol = "et_n2_ref_case_" + re.sub(r"[^A-Za-z0-9_]", "_", case["name"])
        lines.extend((
            "  {",
            f'    "{case["name"]}", "{case["operation"]}", "{case["kind"]}",',
            f"    UINT32_C({len(case['inputs'])}), {symbol}_inputs,",
            f"    UINT32_C({len(case['expectation']['outputs'])}), {symbol}_outputs,",
            "  },",
        ))
    lines.extend(("};", "", "#endif", ""))
    return "\n".join(lines).encode("ascii")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_bytes(render_header(arguments.fixture.read_bytes()))


if __name__ == "__main__":
    main()
