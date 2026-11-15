"""Require the private second-normalization mutant's first value difference.

The enclosing numerical gate first admits the normal transcript, including
independent mathematical M3/CE and role checks, and requires the mutant's
ordinary strict verifier to reject at final.metrics_f32. This check then
classifies that rejection using the same frozen reference and actual words.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from tests.e3_metrics.reference_probe import accumulate, divide, exp_bits, reduce_bool
from tests.e3_reference.transcript import decode, validate_observed_case


def check(reference: dict, normal: dict, mutant: dict,
          normal_roles: bytes, mutant_roles: bytes) -> None:
    cases = [case for case in reference["cases"] if case["name"] == "packed-single"]
    if len(cases) != 1:
        raise ValueError("packed-single reference is missing or duplicated")
    rows = tuple(cases[0]["corpus"]["batches"])
    validate_observed_case(normal, rows)
    if normal_roles != mutant_roles:
        raise ValueError("second-normalization changed M3 role words")
    if (set(normal) != set(mutant) or mutant.get("kind") != "observed-f32"
            or mutant.get("profile") != normal["profile"]
            or len(mutant.get("batches", [])) != len(rows)
            or mutant["batches"] != normal["batches"]):
        raise ValueError("second-normalization diverged before final publication")

    # Recompute the exact binary32 state from independently admitted D2 rows
    # and the already validated CE words, rather than trusting reported state.
    state = (0, 0, 0, 0, 0)
    for row, batch in zip(rows, normal["batches"]):
        numerator, weight = reduce_bool(
            tuple(int(word, 16) for word in batch["ce_f32"]), row["mask"])
        state = accumulate(state, (numerator, weight,
                                   int(batch["correct_f32"], 16),
                                   sum(row["mask"])))
    n, w, correct, tokens, batches = state
    loss = divide(n, w)
    doubled = divide(loss, w)
    if doubled == loss:
        raise ValueError("authentic final state does not distinguish second normalization")
    expected = (f"{doubled:08x}", f"{w:08x}",
                f"{exp_bits(doubled):08x}", f"{divide(correct, w):08x}")
    final = mutant.get("final")
    if not isinstance(final, dict) or set(final) != {"counters", "metrics_f32"}:
        raise ValueError("second-normalization final schema differs")
    counters = final["counters"]
    if (type(counters) is not list or len(counters) != 2
            or any(type(counter) is not int for counter in counters)):
        raise ValueError("second-normalization final counters are malformed")
    words = final["metrics_f32"]
    if (not isinstance(words, list) or len(words) != 4
            or any(type(word) is not str or len(word) != 8 or
                   any(digit not in "0123456789abcdef" for digit in word)
                   for word in words)):
        raise ValueError("second-normalization final words are malformed")
    if words[0] != expected[0]:
        raise ValueError("second-normalization loss is not the second exact division")
    if words[0] == normal["final"]["metrics_f32"][0]:
        raise ValueError("second-normalization did not change final loss")
    if (counters != [tokens, batches]
            or words[1] != expected[1] or words[3] != expected[3]):
        raise ValueError("second-normalization changed final weight, accuracy or counters")
    if abs(int(words[2], 16) - int(expected[2], 16)) > 2:
        raise ValueError("second-normalization perplexity exceeds two binary32 ULP")
    # The ordinary validator checks every field's strict type. Restore only
    # the two deliberately mutated words for this structural admission check.
    projected = copy.deepcopy(mutant)
    for index in (0, 2):
        projected["final"]["metrics_f32"][index] = normal["final"]["metrics_f32"][index]
    validate_observed_case(projected, rows)


def main() -> None:
    from tests.e3_reference.reference import validate_bundle

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "normal", "mutant", "normal-roles", "mutant-roles"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    reference = decode(args.reference.read_bytes())
    validate_bundle(reference)
    check(reference, json.loads(args.normal.read_bytes()),
          json.loads(args.mutant.read_bytes()), args.normal_roles.read_bytes(),
          args.mutant_roles.read_bytes())
    print("E3-SECOND-NORMALIZATION-CAUSAL-KILL-PASS")


if __name__ == "__main__":
    main()
