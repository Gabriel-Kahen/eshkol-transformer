"""Compare live audit and score lines with the exact C1/D1 reference fixture."""

import math
from pathlib import Path
import struct
import sys

from tests.eval5 import mod5_reference_oracle as oracle
from tests.eval5 import mod5_v256_fixture as fixture

REL_TOL = 5e-4
METRIC_ABS_TOL = 5e-3
PROBE_ABS_TOL = 2e-3
CONTEXTS = ((10, 11, 12, 13, 14, 15, 0, 3),
            (3, 0, 15, 14, 13, 12, 11, 10))


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def expected_probes(root):
    weights = oracle.weights_from_c1((root / "model.etcp").read_bytes())
    contexts = [doc[:8] for doc in fixture.SOURCE["eval-clean"]] + list(CONTEXTS)
    probes = []
    for context in contexts:
        rows = oracle.logits(weights, context)
        probes.extend((f32(rows[0][0]), f32(rows[7][255])))
    return probes


def check(root, stdout):
    record = fixture.verify(root)
    lines = stdout.splitlines()
    if len(lines) != 2:
        raise ValueError("expected live AUDIT then SCORE lines")
    audit = record["score_reference"]["audit"]
    wanted_audit = [0, *(audit[key] for key in (
        "eval_documents", "contaminated_documents", "exact_doc_matches",
        "ngram_contaminated_docs", "total_eval_ngrams", "matched_eval_ngrams")),
        0, 0]
    audit_fields = lines[0].split()
    if len(audit_fields) != 10 or audit_fields[0] != "AUDIT":
        raise ValueError("malformed live AUDIT line")
    try:
        actual_audit = [int(value) for value in audit_fields[1:]]
    except ValueError as error:
        raise ValueError("noninteger live audit field") from error
    if actual_audit != wanted_audit:
        raise ValueError("live contamination audit differs from reference")
    fields = lines[1].split()
    if len(fields) != 16 or fields[0] != "SCORE":
        raise ValueError("malformed live SCORE line")
    metrics = [float(value) for value in fields[1:6]]
    counts = [int(value) for value in fields[6:8]]
    probes = [float(value) for value in fields[8:16]]
    expected = record["score_reference"]
    wanted_metrics = [expected["lm"][key] for key in
                      ("total_nll", "mean_nll", "perplexity", "bits_per_byte")]
    wanted_metrics.append(expected["mc"]["accuracy"])
    for index, (actual, wanted) in enumerate(zip(metrics, wanted_metrics, strict=True)):
        if not math.isfinite(actual) or not math.isclose(actual, wanted,
                rel_tol=REL_TOL, abs_tol=METRIC_ABS_TOL):
            raise ValueError(f"live metric {index} differs from reference")
    if counts != [expected["mc"]["correct_count"], expected["mc"]["total_count"]]:
        raise ValueError("live MC counts differ from reference")
    wanted_probes = expected_probes(root)
    for index, (actual, wanted) in enumerate(zip(probes, wanted_probes, strict=True)):
        if not math.isfinite(actual) or not math.isclose(actual, wanted,
                rel_tol=REL_TOL, abs_tol=PROBE_ABS_TOL):
            raise ValueError(f"live f32 logit probe {index} differs from reference")
    return max(abs(actual - wanted) for actual, wanted in zip(metrics, wanted_metrics))


def main(argv):
    if len(argv) != 3:
        raise SystemExit("usage: python3 -m tests.eval5.check_mod5_v256_witness FIXTURE STDOUT")
    delta = check(Path(argv[1]), Path(argv[2]).read_text())
    print(f"EVAL5 V256 live score/reference comparison: PASS max_metric_abs_delta={delta:.9g}")


if __name__ == "__main__":
    main(sys.argv)
