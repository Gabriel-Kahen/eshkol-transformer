"""Emit native input rows from an accepted E3 reference bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

from tests.e3_reference.transcript import decode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()

    bundle = decode(arguments.reference.read_bytes())
    if bundle.get("kind") != "reference-bundle":
        raise ValueError("fixture source is not an E3 reference bundle")
    cases = bundle.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("reference bundle has no cases")

    lines = [
        "#ifndef ET_E3_NATIVE_PARITY_FIXTURE_H",
        "#define ET_E3_NATIVE_PARITY_FIXTURE_H",
        "typedef struct et_e3_parity_batch {",
        "  int64_t inputs[2];",
        "  int64_t targets[2];",
        "  unsigned char mask[2];",
        "} et_e3_parity_batch;",
        "typedef struct et_e3_parity_case {",
        "  const char *name;",
        "  size_t batch_count;",
        "  const et_e3_parity_batch *batches;",
        "} et_e3_parity_case;",
    ]
    for case_index, case in enumerate(cases):
        if not isinstance(case, dict) or not isinstance(case.get("corpus"), dict):
            raise ValueError("reference case is malformed")
        batches = case["corpus"].get("batches")
        if not isinstance(batches, list) or not batches:
            raise ValueError("native success fixture requires at least one batch")
        lines.append(f"static const et_e3_parity_batch e3_parity_batches_{case_index}[] = {{")
        for batch in batches:
            inputs = batch.get("inputs")
            targets = batch.get("targets")
            mask = batch.get("mask")
            if (
                not isinstance(inputs, list) or len(inputs) != 2
                or not isinstance(targets, list) or len(targets) != 2
                or not isinstance(mask, list) or len(mask) != 2
                or any(isinstance(value, bool) or not isinstance(value, int)
                       for value in inputs + targets)
                or any(type(value) is not bool for value in mask)
            ):
                raise ValueError("reference batch is malformed")
            lines.append(
                "  {{{{{}, {}}}, {{{}, {}}}, {{{}, {}}}}},".format(
                    inputs[0], inputs[1], targets[0], targets[1],
                    int(mask[0]), int(mask[1])
                )
            )
        lines.append("};")
    lines.append("static const et_e3_parity_case e3_parity_cases[] = {")
    for case_index, case in enumerate(cases):
        name = case.get("name")
        if not isinstance(name, str) or not name or any(
            character not in "abcdefghijklmnopqrstuvwxyz-" for character in name
        ):
            raise ValueError("reference case name is not a fixed lowercase label")
        lines.append(
            f'  {{"{name}", sizeof(e3_parity_batches_{case_index}) / '
            f'sizeof(e3_parity_batches_{case_index}[0]), e3_parity_batches_{case_index}}},'
        )
    lines.extend([
        "};",
        "#define ET_E3_PARITY_CASE_COUNT \\",
        "  (sizeof(e3_parity_cases) / sizeof(e3_parity_cases[0]))",
        "#endif",
        "",
    ])
    arguments.output.write_text("\n".join(lines), encoding="ascii", newline="\n")


if __name__ == "__main__":
    main()
