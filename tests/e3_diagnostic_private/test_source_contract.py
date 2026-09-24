"""Static contract checks for the source-private E3 diagnostic successor."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


class DiagnosticSourceContract(unittest.TestCase):
    def test_root_composes_after_private_transaction(self) -> None:
        source = (ROOT / "native/e3_diagnostic_private_driver_root.esk").read_text()
        self.assertLess(
            source.index('(load "e3_private_driver_root.esk")'),
            source.index('(load "e3_diagnostic_private_extension.esk")'),
        )

    def test_candidate_does_not_change_shared_exports(self) -> None:
        names = (
            "diagnostic-evaluate-fixed",
            "diagnostic-evaluation-f32-bits",
            "diagnostic-evaluation-count",
        )
        manifests = list((ROOT / "native").glob("*public_exports.txt"))
        for manifest in manifests:
            text = manifest.read_text()
            for name in names:
                self.assertNotIn(name, text, manifest)

    def test_report_is_capped_and_uses_accepted_seams(self) -> None:
        source = (ROOT / "native/e3_diagnostic_private_extension.esk").read_text()
        for required in (
            "(define e3-diagnostic-report-limit 8192)",
            "(e3-frame-bind! model destinations)",
            "(e3-evaluate-into! model dataset frame)",
            "et_e3_private_selected_metric_bits_ref_v1",
            "(e3-counter-ref frame 0)",
            "(e3-counter-ref frame 1)",
            "(e3-frame-unbind! frame)",
        ):
            self.assertIn(required, source)

        driver = (ROOT / "tests/e3_diagnostic_private/driver.esk").read_text()
        self.assertIn("(g3t-checked-m3t-model-create config initial)", driver)
        self.assertIn("(d2-token-dataset-open", driver)
        self.assertIn("(e3-diagnostic-test-fixture)", driver)
        self.assertNotIn("(e3-test-fixture)", driver)

        self.assertIn("(define-syntax e3-diagnostic-fail", source)
        # One definition and one macro-body call; runtime sites use the
        # non-tail syntax wrapper.
        self.assertEqual(source.count("(e3-diagnostic-raise "), 2)

        # The selected-bit C seam takes the private native frame pointer, not
        # the public Eshkol frame shell accepted by e3-evaluate-into!.
        selected = source[source.index("(define (e3-diagnostic-selected-bits") :]
        selected = selected[: selected.index("(define (e3-diagnostic-read-results!")]
        self.assertIn("(e3-frame-entry-live frame operation)", selected)
        self.assertIn("(vector-ref entry 4) selector", selected)
        self.assertNotIn("(e3-diagnostic-native-selected-bits frame", selected)

    def test_failed_path_abandons_without_publication(self) -> None:
        source = (ROOT / "native/e3_diagnostic_private_extension.esk").read_text()
        handler = source[source.index("(define (e3-diagnostic-evaluate-fixed!") :]
        self.assertLess(handler.index("(e3-diagnostic-abandon-report!)"),
                        handler.index("(raise caught)"))
        self.assertGreater(handler.index("(e3-diagnostic-publish-report! record)"),
                           handler.index("(set! destinations #f)"))

        witness = (ROOT / "tests/e3_diagnostic_private/driver.esk").read_text()
        self.assertIn("(define (e3-diagnostic-test-failure", witness)
        self.assertEqual(witness.count("(d2-core-bytes=?"), 2)
        self.assertIn("stats-before", witness)
        self.assertIn("stats-after", witness)

        runtime = (ROOT / "tests/e3_diagnostic_private/runtime.esk").read_text()
        self.assertNotIn("with-region ('e3-diagnostic-runtime-call", runtime)
        self.assertIn("constructed in the true root context", runtime)

    def test_full_horizons_have_a_fixed_supported_timeout(self) -> None:
        script = (ROOT / "scripts/test-e3-diagnostic-private-runtime.sh").read_text()
        self.assertIn("runtime_timeout=3600", script)
        self.assertIn("success-8192|failure-8192) runtime_timeout=7200", script)
        self.assertIn('"${runtime_timeout}s"', script)
        self.assertNotIn("E3_DIAGNOSTIC_HORIZON_TIMEOUT", script)
        self.assertLess(script.index("success-1024 success-2048 success-4096"),
                        script.index("success-8192"))

        runtime = (ROOT / "tests/e3_diagnostic_private/runtime.esk").read_text()
        self.assertIn('(run-horizon path 2048 #f)', runtime)
        self.assertIn('(run-horizon path 4096 #f)', runtime)

    def test_retention_attribution_uses_exact_transaction_boundaries(self) -> None:
        source = (ROOT / "native/e3_diagnostic_private_extension.esk").read_text()
        self.assertIn(
            "(define e3-diagnostic-report-root (vector '() #f 0 0 0 0))",
            source,
        )
        reserve = source[source.index("(define (e3-diagnostic-reserve-report!") :]
        reserve = reserve[:reserve.index("(define (e3-diagnostic-abandon-report!")]
        self.assertLess(reserve.index("(let ((arena-before (__arena-used)))"),
                        reserve.index("(vector-set! e3-diagnostic-report-root 1 envelope)"))
        self.assertLess(reserve.index("(vector-set! e3-diagnostic-report-root 1 envelope)"),
                        reserve.index("(promoted-bytes (- (__arena-used) arena-before))"))
        self.assertIn("(vector-set! e3-diagnostic-report-root 4", reserve)

        evaluate = source[source.index("(define (e3-diagnostic-evaluate-fixed!") :]
        evaluate = evaluate[:evaluate.index("(define (e3-diagnostic-f32-bits")]
        self.assertLess(evaluate.index("(e3-diagnostic-abandon-report!)"),
                        evaluate.index("(e3-diagnostic-record-transaction-retention!"))
        self.assertLess(evaluate.index("(e3-diagnostic-record-transaction-retention!"),
                        evaluate.index("(raise caught)"))
        self.assertLess(evaluate.rindex("(e3-diagnostic-publish-report! record)"),
                        evaluate.rindex("(e3-diagnostic-record-transaction-retention!"))

        driver = (ROOT / "tests/e3_diagnostic_private/driver.esk").read_text()
        calibration = driver[
            driver.index("(define (e3-diagnostic-test-calibrate-envelope!") :
        ]
        calibration = calibration[
            :calibration.index("(define (e3-diagnostic-test-retention-stats")
        ]
        self.assertIn("(vector e3-diagnostic-report-tag #f '())", calibration)
        self.assertIn("(vector-set! e3-diagnostic-test-retention-root 0 #f)",
                      calibration)
        stats = driver[
            driver.index("(define (e3-diagnostic-test-retention-stats") :
        ]
        stats = stats[:stats.index("(define (e3-diagnostic-test-expect-error")]
        self.assertIn(
            "(if (= (vector-ref e3-diagnostic-test-retention-root 1) 0)",
            stats,
        )
        self.assertIn("(e3-diagnostic-test-calibrate-envelope!)", stats)

        runtime = (ROOT / "tests/e3_diagnostic_private/runtime.esk").read_text()
        for field in (
            "reachable_authority_bytes=",
            "unreachable_staging_bytes=",
            "inherited_transaction_bytes=",
            "witness_other_bytes=",
            "reserve_promoted_bytes=",
            "envelope_bytes_per_call=",
            "calibration_bytes=",
        ):
            self.assertIn(field, runtime)
        self.assertIn("(if failure? last-promoted", runtime)
        self.assertIn("(* envelope-bytes last-reservations)", runtime)
        self.assertIn("(- arena-delta attributed calibration-bytes)", runtime)

        script = (ROOT / "scripts/test-e3-diagnostic-private-runtime.sh").read_text()
        self.assertIn("reachable_authority_bytes\\tunreachable_staging_bytes", script)
        self.assertIn("inherited_transaction_bytes\\twitness_other_bytes", script)
        self.assertIn("envelope_bytes_per_call\\tcalibration_bytes", script)

    def test_mode_enrollment_uses_only_the_active_chain(self) -> None:
        source = (ROOT / "internal/p1/lib/transformer/module.esk").read_text()
        enrolled = source[source.index("(define (e3-mode-enrolled?") :]
        enrolled = enrolled[: enrolled.index("(define (e3-mode-parent")]
        self.assertIn("(vector-ref e3-mode-registry-root 1)", enrolled)
        self.assertIn("(vector-ref record 9)", enrolled)
        self.assertNotIn("(vector-ref e3-mode-registry-root 0)", enrolled)

        retire = source[source.index("(define (e3-mode-active-remove!") :]
        retire = retire[: retire.index("(define (e3-mode-bind!")]
        self.assertIn("(vector-set! e3-mode-registry-root 1 next)", retire)
        self.assertIn("(vector-set! record 9 #f)", retire)
        self.assertIn("(vector-set! record 10 #f)", retire)
        self.assertNotIn("let loop", retire)


if __name__ == "__main__":
    unittest.main()
