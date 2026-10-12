"""Require genuine frozen D2 witnesses and locate each compiled mutant's first change."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
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
    print(first_change(args.baseline.read_bytes(), args.mutant.read_bytes(), args.mode))


if __name__ == "__main__":
    main()
