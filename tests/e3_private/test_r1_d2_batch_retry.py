"""Source-only guard for genuine E3 stage-5 D2 batch fault/retry candidate."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
DRIVER = ROOT / "tests/e3_private/r1_d2_batch_retry_driver.esk"
LOADER = ROOT / "tests/e3_private/r1_d2_batch_retry_driver_root.esk"


def source_errors(source: str) -> list[str]:
    cut = source.split("(define (e3-r1-d2-batch-cut-retry!", 1)[1].split(
        "(define (e3-r1-d2-batch-retry-run", 1)[0]
    required = {
        "authentic fixture": "(e3-test-fixture)",
        "authentic D2": "(d2-token-dataset-open",
        "existing native selector": ":real et_d2_batch_test_fail_stage_v1",
        "one-shot selector": ":real et_d2_batch_test_fail_occurrence_v1",
        "occurrence observer": ":real et_d2_batch_test_occurrences_seen_v1",
        "all seven construction stages":
            "(let stage-loop ((stage 1))\n         (if (= stage 8) #t\n"
            "             (begin\n"
            "               (e3-r1-d2-batch-cut-retry! model dataset frame destinations\n"
            "                                           stage 1)\n"
            "               (stage-loop (+ stage 1)))))",
        "all seven late-batch stages":
            "(let late-stage-loop ((stage 1))\n"
            "         (if (= stage 8) #t\n"
            "             (begin\n"
            "               (e3-r1-d2-batch-cut-retry! model dataset frame destinations\n"
            "                                           stage 2)\n"
            "               (late-stage-loop (+ stage 1)))))",
        "real E3 call": "(e3-evaluate-into! model dataset frame)",
        "late arm": "(e3-r1-d2-batch-fail-occurrence stage occurrence)",
        "late exact count": "(= (e3-r1-d2-batch-occurrences-seen) 2)",
        "selector cleared": "(e3-r1-d2-batch-fail-stage 0)",
        "first mapped E1": "(eq? caught (vector-ref entry 11))",
        "stage five": "(= (vector-ref (vector-ref entry 12) 0) 5)",
        "cursor rollback": "(d2-core-bytes=? cursor (d2-token-dataset-cursor dataset))",
        "six outputs": "(e3-r1-output=? frame destinations outputs)",
        "17 modes": "(e3-test-retention-modes=? frame modes)",
        "model and gradients": "(e3-test-retention-model=? model model-image)",
        "rng": "(m3t-public-model-initializer model)",
        "owner census": "(e3-test-retention-probe index)",
        "same-authority retry":
            "(e3-test-retention-call! model dataset frame destinations cursor modes",
    }
    errors = [name for name, fragment in required.items()
              if fragment not in (source if name in {
                  "authentic fixture", "authentic D2", "existing native selector",
                  "all seven construction stages", "all seven late-batch stages",
                  "one-shot selector", "occurrence observer"} else cut)]
    arm = cut.find("(e3-r1-d2-batch-fail-stage stage)")
    call = cut.find("(e3-evaluate-into! model dataset frame)", arm)
    clear = cut.find("(e3-r1-d2-batch-fail-stage 0)", call)
    rollback = cut.find("(d2-core-bytes=? cursor", clear)
    retry = cut.find("(e3-test-retention-call! model dataset frame destinations cursor modes", rollback)
    if min(arm, call, clear, rollback, retry) < 0 or not arm < call < clear < rollback < retry:
        errors.append("arm/failure/clear/rollback/retry order")
    if cut.count("(e3-r1-output=? frame destinations outputs)") != 2:
        errors.append("rollback and retry output images")
    return errors


class R1D2BatchRetryTest(unittest.TestCase):
    def test_closed_source_and_real_native_seam(self) -> None:
        self.assertEqual(LOADER.read_text().count('(load "r1_source_retry_driver_root.esk")'), 1)
        self.assertEqual(LOADER.read_text().count('(load "r1_d2_batch_retry_driver.esk")'), 1)
        canonical = (ROOT / "native/e3_private_driver_root.esk").read_text()
        accepted = (ROOT / "tests/e3_private/r1_source_retry_driver_root.esk").read_text()
        self.assertNotIn("r1_d2_batch_retry", canonical + accepted)
        native = (ROOT / "native/d2_native.c").read_text()
        dataset = (ROOT / "internal/d2/lib/d2_dataset.esk").read_text()
        e3 = (ROOT / "native/e3_private_extension.esk").read_text()
        self.assertIn("(d2-token-dataset-next-batch dataset)", e3)
        self.assertIn("void et_d2_batch_test_fail_stage_v1(int64_t stage)", native)
        self.assertIn("#ifdef ET_D2_NATIVE_TESTING\nvoid et_d2_batch_test_fail_stage_v1", native)
        self.assertIn("int64_t et_d2_batch_test_fail_occurrence_v1(", native)
        self.assertIn("fail_occurrences_seen < fail_occurrence_target", native)
        self.assertIn("fail_occurrences_seen == fail_occurrence_target", native)
        self.assertIn("fail_occurrence_active = fail_occurrence_stage", native)
        self.assertIn("if (fail_occurrence_active == stage)", native)
        self.assertIn("if (fail_occurrence_active != 0) {", native)
        self.assertIn("fail_occurrence_stage = 0;", native)
        self.assertIn("int64_t et_d2_batch_test_occurrences_seen_v1(void)", native)
        self.assertIn("(d2-native-batch-create dataset batch-size sequence-length", dataset)
        for seam in ("ET_D2_TEST_FAIL_CONTROL", "ET_D2_TEST_FAIL_MASK",
                     "ET_D2_TEST_FAIL_BEFORE_PUBLISH"):
            self.assertIn(f"should_fail({seam})", native)
        for seam in ("ET_D2_TEST_FAIL_AFTER_INPUT_TENSOR",
                     "ET_D2_TEST_FAIL_AFTER_INPUT_BORROW",
                     "ET_D2_TEST_FAIL_AFTER_TARGET_TENSOR",
                     "ET_D2_TEST_FAIL_AFTER_TARGET_BORROW"):
            self.assertIn(seam, native)
        self.assertIn("should_fail(after_tensor_stage)", native)
        self.assertIn("should_fail(after_borrow_stage)", native)

    def test_failure_and_retry_contract(self) -> None:
        self.assertEqual(source_errors(DRIVER.read_text()), [])

    def test_missing_stage_or_assertion_is_rejected(self) -> None:
        source = DRIVER.read_text()
        mutants = [
            source.replace("(if (= stage 8) #t", "(if (= stage 7) #t", 1),
            source.replace("(stage-loop (+ stage 1))", "(stage-loop (+ stage 2))", 1),
            source.replace("(late-stage-loop (+ stage 1))", "(late-stage-loop (+ stage 2))", 1),
            source.replace("stage 2)", "stage 1)", 1),
            source.replace("(= (e3-r1-d2-batch-occurrences-seen) 2)", "#t", 1),
            source.replace("(e3-r1-d2-batch-fail-occurrence stage occurrence)",
                           "(e3-r1-d2-batch-fail-occurrence stage 1)", 1),
            source.replace("(e3-r1-d2-batch-fail-stage 0)", "#t", 1),
            source.replace("(eq? caught (vector-ref entry 11))", "#t", 1),
            source.replace("(e3-r1-output=? frame destinations outputs)", "#t", 1),
            source.replace("(e3-test-retention-model=? model model-image)", "#t", 1),
            source.replace("(e3-test-retention-call! model dataset frame destinations cursor modes",
                           "(e3-test-retention-call! replacement dataset frame destinations cursor modes", 1),
        ]
        for mutant in mutants:
            with self.subTest(mutant=mutants.index(mutant)):
                self.assertTrue(source_errors(mutant))


if __name__ == "__main__":
    unittest.main()
