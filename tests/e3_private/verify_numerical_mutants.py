"""Require genuine frozen D2 witnesses and locate each compiled mutant's first change."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import re
import struct

from tests.e3_reference.transcript import decode


EXPECT = {
    "wrong-target": ("E3CE", 0, 2, 0),
    "mean-of-means": ("E3STATE", 0, 2, 0),
    "capacity-denominator": ("E3BATCH", 2, 2, 1),
    "padded-count": ("E3BATCH", 2, 3, 0),
    "missing-batch": ("E3STATE", 2, 2, 0),
    "doubled-batch": ("E3STATE", 2, 2, 0),
    "second-normalization": ("E3FINAL", None, 1, 2),
    "exp-before-aggregate": ("E3STATE", 0, 2, 0),
}
WORD = re.compile(r"[0-9a-f]{8}\Z")


def _f32(word: str) -> float:
    return struct.unpack(">f", bytes.fromhex(word))[0]


def witness(reference: Path) -> None:
    bundle = decode(reference.read_bytes())
    cases = [case for case in bundle["cases"] if case["name"] == "packed-single"]
    if len(cases) != 1:
        raise ValueError("frozen packed-single case is absent or duplicated")
    batches = cases[0]["synthetic"]["batches"]
    masks = [batch["mask"] for batch in batches]
    if masks != [[True, True], [True, True], [True, False]]:
        raise ValueError("frozen D2 rows do not reach the padded third batch")
    contributions = [_f32(batch["reduction_f32"][0]) for batch in batches]
    if not all(math.isfinite(value) and value > 0 for value in contributions):
        raise ValueError("frozen CE numerator is not positive and finite")
    means = [value / sum(mask) for value, mask in zip(contributions, masks)]
    if len(set(means)) == 1:
        raise ValueError("frozen batch losses cannot distinguish aggregation")
    weighted_exp = sum(sum(mask) * math.exp(mean)
                       for mask, mean in zip(masks, means)) / 5
    global_exp = math.exp(sum(contributions) / 5)
    if math.isclose(weighted_exp, global_exp, rel_tol=1e-6):
        raise ValueError("frozen batch losses do not distinguish exp ordering")


def first_change(baseline: bytes, mutant: bytes, mode: str) -> str:
    if mode not in EXPECT:
        raise ValueError("unknown compiled numerical mutation")
    ordinary = baseline.decode("ascii").splitlines()
    changed = mutant.decode("ascii").splitlines()
    if not ordinary or len(ordinary) != len(changed) or \
            ordinary[-1] != changed[-1] or ordinary[-1] != "E3-NUMERICAL-RUNTIME-PASS":
        raise ValueError("mutant did not complete the same compiled observation")
    for before, after in zip(ordinary, changed):
        if before == after:
            continue
        left, right = before.split("\t"), after.split("\t")
        if len(left) != len(right) or left[0] != right[0]:
            raise ValueError("mutant changed transcript structure")
        tag, ordinal, field, word = EXPECT[mode]
        actual_ordinal = int(left[1]) if tag != "E3FINAL" else None
        if left[0] != tag or actual_ordinal != ordinal:
            raise ValueError(f"{mode}: first change was {left[0]} row {actual_ordinal}")
        for index, (prior_field, new_field) in enumerate(zip(left, right)):
            if prior_field == new_field:
                continue
            before_words, after_words = prior_field.split(","), new_field.split(",")
            if index != field or len(before_words) != len(after_words):
                raise ValueError(f"{mode}: first change was field {index}")
            for position, (prior, new) in enumerate(zip(before_words, after_words)):
                if prior != new:
                    if position != word:
                        raise ValueError(f"{mode}: first change was word {position}")
                    return (f"{mode}: first difference {tag} row {ordinal} "
                            f"field {field} word {word}: {prior} -> {new}")
        raise ValueError("mutant line differed without an observed word")
    raise ValueError(f"{mode}: compiled mutant survived the frozen witness")


def _observation(raw: bytes) -> dict[str, object]:
    """Read only the downstream words used by M2, with exact tag/cardinality checks."""
    lines = raw.decode("ascii").splitlines()
    if not lines or lines[-1] != "E3-NUMERICAL-RUNTIME-PASS":
        raise ValueError("M2 transcript did not complete")
    batches: dict[int, tuple[list[str], int]] = {}
    states: dict[int, tuple[list[str], list[int]]] = {}
    final = None
    boxed = None
    for line in lines[:-1]:
        fields = line.split("\t")
        tag = fields[0]
        if tag in {"E3BATCH", "E3STATE"}:
            if len(fields) != 4 or fields[1] not in {"0", "1", "2"}:
                raise ValueError("M2 batch/state tag or ordinal is malformed")
            ordinal = int(fields[1])
            words = fields[2].split(",")
            if len(words) != (4 if tag == "E3BATCH" else 3) or \
                    any(not WORD.fullmatch(word) for word in words):
                raise ValueError("M2 batch/state words are malformed")
            counts = fields[3].split(",")
            if len(counts) != (1 if tag == "E3BATCH" else 2) or \
                    any(not item.isascii() or not item.isdecimal() or
                        str(int(item)) != item for item in counts):
                raise ValueError("M2 batch/state counts are malformed")
            target = batches if tag == "E3BATCH" else states
            if ordinal in target:
                raise ValueError("M2 batch/state ordinal is duplicated")
            target[ordinal] = (words, int(counts[0]) if tag == "E3BATCH" else
                               [int(item) for item in counts])
        elif tag == "E3FINAL":
            if final is not None or len(fields) != 3:
                raise ValueError("M2 final tag is malformed or duplicated")
            words = fields[1].split(",")
            counters = fields[2].split(",")
            if len(words) != 4 or any(not WORD.fullmatch(word) for word in words) or \
                    len(counters) != 2 or any(not item.isascii() or
                    not item.isdecimal() or str(int(item)) != item for item in counters):
                raise ValueError("M2 final words are malformed")
            final = (words, [int(item) for item in counters])
        elif tag == "E3RESULT":
            if boxed is not None or len(fields) != 4:
                raise ValueError("M2 boxed result is malformed or duplicated")
            outputs = fields[1].split(",")
            if len(outputs) != 6 or any(not item.isascii() or
                    not item.isdecimal() or str(int(item)) != item or
                    int(item) > (1 << 32) - 1 for item in outputs):
                raise ValueError("M2 boxed outputs are malformed")
            boxed = [int(item) for item in outputs]
    if set(batches) != {0, 1, 2} or set(states) != {0, 1, 2} or \
            final is None or boxed is None:
        raise ValueError("M2 transcript omitted a downstream row")
    if boxed[:4] != [int(word, 16) for word in final[0]] or \
            boxed[4:] != final[1] or states[2][1] != final[1]:
        raise ValueError("M2 final publication or boxed counters differ from state")
    return {"batches": batches, "states": states, "final": final}


def downstream(baseline: bytes, mutant: bytes, mode: str) -> None:
    ordinary, changed = _observation(baseline), _observation(mutant)
    batch0, batch1 = ordinary["batches"][2], changed["batches"][2]
    state0, state1 = ordinary["states"][2], changed["states"][2]
    final0, final1 = ordinary["final"], changed["final"]
    weight0, weight1 = _f32(final0[0][1]), _f32(final1[0][1])
    if mode == "capacity-denominator":
        valid = (batch0[0][1] == "3f800000" and batch1[0][1] == "40000000" and
                 batch0[1] == batch1[1] == 1 and batch0[0][3] == batch1[0][3] and
                 final0[1] == final1[1] and weight1 == weight0 + 1)
    elif mode == "padded-count":
        valid = (batch0[1] == 1 and batch1[1] == 2 and
                 batch0[0][1] == batch1[0][1] and
                 batch0[0][3] == batch1[0][3] and weight1 == weight0 and
                 final1[1] == [final0[1][0] + 1, final0[1][1]])
    elif mode == "missing-batch":
        valid = (state1 == ordinary["states"][1] and
                 final1[1] == [final0[1][0] - 1, final0[1][1] - 1] and
                 weight1 == weight0 - 1)
    elif mode == "doubled-batch":
        valid = (final1[1] == [final0[1][0] + 1, final0[1][1] + 1] and
                 weight1 == weight0 + 1 and state1[0][0] != state0[0][0])
    elif mode == "second-normalization":
        valid = (final1[0][0:2] == final0[0][0:2] and
                 final1[0][3] == final0[0][3] and final1[1] == final0[1] and
                 final1[0][2] != final0[0][2])
    elif mode == "exp-before-aggregate":
        valid = (final1[0][1] == final0[0][1] and
                 final1[0][3] == final0[0][3] and final1[1] == final0[1] and
                 final1[0][2] != final0[0][2] and
                 final1[0][2] == final1[0][0])
    else:
        return
    if not valid:
        raise ValueError(f"{mode}: downstream batch/state/final assertion failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--mutant", type=Path)
    parser.add_argument("--mode", choices=tuple(EXPECT))
    args = parser.parse_args()
    witness(args.reference)
    if args.mode is None:
        if args.baseline or args.mutant:
            parser.error("runtime paths require a mode")
        print("frozen packed D2 mutation witnesses admitted")
        return
    if not args.baseline or not args.mutant:
        parser.error("mode requires baseline and mutant stdout")
    baseline, mutant = args.baseline.read_bytes(), args.mutant.read_bytes()
    difference = first_change(baseline, mutant, args.mode)
    downstream(baseline, mutant, args.mode)
    print(difference)


if __name__ == "__main__":
    main()
