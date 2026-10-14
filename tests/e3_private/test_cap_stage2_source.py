"""Offline controls for the noninstalled N5 private stage-2 source candidate."""

from pathlib import Path
import unittest

from tests.e3_private.test_retention_first_baseline import form


ROOT = Path(__file__).resolve().parents[2]
DRIVER = ROOT / "tests/e3_private/retention_driver.esk"
RUNTIME = ROOT / "tests/e3_private/cap_stage2_runtime.esk"
D2 = ROOT / "internal/e3/lib/e3_d2_restore.esk"
E3 = ROOT / "native/e3_private_extension.esk"


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def audit(source: str) -> None:
    witness = form(source, "(define (e3-test-cap-stage2 present? paths)")
    invariant = form(source, "(define (e3-test-cap-stage2-check!")
    dispatch = form(source, "(define (e3-test-retention-dispatch")
    require(witness.index("(e3-test-run (vector #t 0)") <
            witness.index("'cap-genuine-baseline") <
            witness.index("(d2-token-dataset-open") <
            witness.index("(e3-test-read-fail-stage 1)") <
            witness.index("(e3-evaluate-into!"),
            "genuine published baseline and cap open must precede armed denial")
    require(witness.index("(e3-test-read-fail-stage 1)") <
            witness.index("(e3-test-read-fail-stage 0)") <
            witness.index("'cap-typed-unsupported") <
            witness.index("'cap-retry-unsupported"),
            "read-fault control must be armed only for first denial")
    require("(= (vector-ref core 14) 8193)" in witness and
            "(= (vector-ref core 15) 0)" in witness and
            "(eq? (vector-ref core 12) #t)" in witness,
            "authentic packed D2 row count must be checked")
    require("(if present? 1 0)" in witness and
            "(e3-test-seed-gradient" in witness and
            "1056964608" in witness,
            "absent and present nonzero-gradient choices must be checked")
    require("(= (vector-ref (vector-ref entry 12) 0) 2)" in invariant and
            "(transformer-error-category" in invariant and
            "'unsupported" in invariant and
            "'e3-evaluate-into!" in invariant,
            "exact stage and typed error must survive frame cleanup")
    for name in ("cap-six-caller-outputs", "cap-cursor", "cap-mixed17-modes",
                 "cap-14-model-gradient-images", "cap-initializer",
                 "cap-no-native-live-growth"):
        require("'" + name in invariant, name + " observer missing")
    require("(e3-test-retention-model=? model model-image)" in invariant and
            "(e3-test-retention-output=? frame destinations)" in invariant and
            "(e3-test-retention-modes=? frame modes)" in invariant and
            "(d2-core-bytes=? cursor (d2-token-dataset-cursor dataset))" in invariant,
            "private caller state comparisons must use live owners")
    require("(e3-test-cap-stage2-check!" in witness and
            witness.count("(e3-test-cap-stage2-check! model dataset") == 2,
            "both denied calls must check the same private state")
    require("(eq? (vector-ref scenario 0) 'cap-stage2)" in dispatch and
            "(eq? (vector-ref scenario 0) 'retention)" in dispatch,
            "cap branch must preserve ordinary retention dispatch")


class CapStage2SourceTests(unittest.TestCase):
    def test_private_source_and_upstream_order(self) -> None:
        audit(DRIVER.read_text())
        d2 = D2.read_text()
        e3 = E3.read_text()
        self.assertLess(d2.index("(> remaining 8192)"),
                        d2.index("(vector-set! record 4 dataset)"))
        self.assertLess(e3.index("(e3-stage-set! entry 2)"),
                        e3.index("(e3-d2-prepare! frame dataset)"))
        self.assertLess(e3.index("(e3-d2-prepare! frame dataset)"),
                        e3.index("(e3-native-acquire (vector-ref entry 4))"))
        runtime = RUNTIME.read_text()
        self.assertIn("(vector 'cap-stage2 present?)", runtime)
        self.assertIn("(vector baseline-path cap-path)", runtime)
        self.assertIn("E3-CAP-STAGE2-PASS", runtime)

    def test_causal_negative_mutants(self) -> None:
        source = DRIVER.read_text()
        mutations = (
            ("(define (e3-test-cap-stage2 present? paths)",
             "(e3-test-read-fail-stage 1)", "#t"),
            ("(define (e3-test-cap-stage2 present? paths)",
             "(= (vector-ref core 14) 8193)",
             "(= (vector-ref core 14) 8192)"),
            ("(define (e3-test-cap-stage2-check!",
             "(= (vector-ref (vector-ref entry 12) 0) 2)",
             "(= (vector-ref (vector-ref entry 12) 0) 3)"),
            ("(define (e3-test-cap-stage2-check!",
             "(e3-test-retention-model=? model model-image)",
             "(e3-test-retention-model=? model model-image-missing)"),
            ("(define (e3-test-cap-stage2-check!",
             "(e3-test-retention-output=? frame destinations)",
             "(e3-test-retention-output=? frame destinations-missing)"),
            ("(define (e3-test-cap-stage2-check!",
             "(e3-test-retention-modes=? frame modes)",
             "(e3-test-retention-modes=? frame modes-missing)"),
            ("(define (e3-test-cap-stage2 present? paths)",
             "(if present? 1 0)", "(if present? 0 0)"),
        )
        for marker, before, after in mutations:
            with self.subTest(before=before):
                point = source.index(before, source.index(marker))
                with self.assertRaises(ValueError):
                    audit(source[:point] + after + source[point + len(before):])


if __name__ == "__main__":
    unittest.main()
