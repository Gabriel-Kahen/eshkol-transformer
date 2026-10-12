"""Source-only controls for the noninstalled E3 R1 whole-call candidate."""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
DRIVER = ROOT / "tests/e3_private/r1_source_retry_driver.esk"
LOADER = ROOT / "tests/e3_private/r1_source_retry_driver_root.esk"


def source_contract_errors(source: str) -> list[str]:
    cut = source[source.find("(define (e3-r1-cut-retry!"):
                 source.find("(define (e3-r1-source-retry-run")]
    required = {
        "authenticated native frame": "(vector-ref entry 4) stage role",
        "genuine evaluator": "(e3-evaluate-into! model dataset frame)",
        "source cursor": "(d2-core-bytes=? cursor (d2-token-dataset-cursor dataset))",
        "six output words": "(e3-r1-output=? frame destinations outputs)",
        "P1 mode": "(e3-test-retention-modes=? frame modes)",
        "M3 values and gradients": "(e3-test-retention-model=? model model-image)",
        "first E1 object": "(eq? caught (vector-ref entry 11))",
        "first diagnostic index": "(vector-ref entry 15)",
        "native error code": "(= (vector-ref (vector-ref entry 12) 3) 10)",
        "same-authority retry": "(e3-test-retention-call! model dataset frame destinations cursor modes",
    }
    errors = [name for name, fragment in required.items() if fragment not in cut]
    if source.count(":real et_e3_test_preinvoke_arm_v1") != 1:
        errors.append("test-only native arm")
    for stage, role in ((7, 0), (7, 20), (8, -1), (9, -1),
                        (10, -1), (11, -1), (12, -1)):
        if len(re.findall(rf"\(e3-r1-cut-retry! model dataset frame destinations {stage} {role}\)",
                          source)) != 1:
            errors.append(f"stage {stage}/{role}")
    arm = cut.find("(e3-r1-preinvoke-arm (vector-ref entry 4) stage role)")
    failure = cut.find("(e3-evaluate-into! model dataset frame)", arm)
    checks = cut.find("(e3-test-retention-check 'failure-cursor", failure)
    retry = cut.find("(e3-test-retention-call! model dataset frame destinations cursor modes",
                        checks)
    if min(arm, failure, checks, retry) < 0 or not arm < failure < checks < retry:
        errors.append("arm/failure/rollback/retry order")
    if "(d2-token-dataset-open" not in source or "(e3-test-fixture)" not in source:
        errors.append("authentic fixture")
    return errors


class R1SourceRetryTests(unittest.TestCase):
    def test_closed_source_composition(self) -> None:
        self.assertEqual(LOADER.read_text().count('(load "retention_driver_root.esk")'), 1)
        self.assertEqual(LOADER.read_text().count('(load "r1_source_retry_driver.esk")'), 1)
        canonical = (ROOT / "native/e3_private_driver_root.esk").read_text()
        retention = (ROOT / "tests/e3_private/retention_driver_root.esk").read_text()
        self.assertNotIn("r1_source_retry", canonical + retention)
        self.assertNotIn("r1_source_retry_driver_root.esk", canonical + retention)

    def test_whole_call_contract(self) -> None:
        self.assertEqual(source_contract_errors(DRIVER.read_text()), [])

    def test_missing_cut_or_rollback_assertion_is_rejected(self) -> None:
        source = DRIVER.read_text()
        mutants = [
            source.replace("(e3-r1-cut-retry! model dataset frame destinations 7 0)",
                           "#t", 1),
            source.replace("(e3-r1-cut-retry! model dataset frame destinations 10 -1)",
                           "#t", 1),
            source.replace("(e3-r1-cut-retry! model dataset frame destinations 7 20)",
                           "#t", 1),
            source.replace("(e3-r1-cut-retry! model dataset frame destinations 8 -1)",
                           "#t", 1),
            source.replace("(e3-r1-cut-retry! model dataset frame destinations 9 -1)",
                           "#t", 1),
            source.replace("(e3-r1-cut-retry! model dataset frame destinations 12 -1)",
                           "#t", 1),
            source.replace("(e3-r1-cut-retry! model dataset frame destinations 11 -1)",
                           "#t", 1),
            source.replace("(e3-test-retention-model=? model model-image)", "#t", 1),
            source.replace("(eq? caught (vector-ref entry 11))", "#t", 1),
            source.replace("(e3-r1-preinvoke-arm (vector-ref entry 4) stage role)",
                           "(e3-r1-preinvoke-arm #f stage role)", 1),
            source.replace("(e3-test-retention-call! model dataset frame destinations cursor modes",
                           "(e3-test-retention-call! model replacement frame destinations cursor modes", 1),
        ]
        for mutant in mutants:
            with self.subTest(mutant=mutants.index(mutant)):
                self.assertTrue(source_contract_errors(mutant))


if __name__ == "__main__":
    unittest.main()
