"""Stage-3 report isolation from the separate 1,024-call retention gate."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
DRIVER = ROOT / 'tests/e3_private/fault_driver.esk'
RUNTIME = ROOT / 'tests/e3_private/r1_stage3_runtime.esk'


def stage3_report(source: str) -> str:
    start = source.index('(define (e3-test-fault-stage3-report ')
    return source[start:source.index('(define (e3-test-fault-dispatch ', start)]


def isolated(source: str) -> bool:
    report = stage3_report(source)
    dispatch = source[source.index('(define (e3-test-fault-dispatch '):]
    return ("(= (vector-ref scenario 3) 4)" in dispatch and
            "(= (vector-ref scenario 4) 7)" in dispatch and
            "(= (vector-ref scenario 5) 1)" in dispatch and
            '(e3-test-fault-stage3-report present?)' in dispatch and
            '(e3-test-retention-report 1024 present?)' in dispatch and
            'e3-test-retention-reconcile!' not in report and
            'e3-test-retention-report' not in report and
            report.index('(d2-token-dataset-close!') <
            report.index('(vector-set! record 18 #t)'))


class Stage3OneShotSourceTest(unittest.TestCase):
    def test_stage3_site4_ordinal7_uses_bounded_report(self):
        driver = DRIVER.read_text()
        self.assertTrue(isolated(driver))
        runtime = RUNTIME.read_text()
        self.assertIn("(define scenario (vector 'fault 'prepare present? 4 7 1))", runtime)
        self.assertIn("'e3-r1-stage3-one-shot-v1", runtime)
        self.assertNotIn("'e3-retention-result-v1", runtime)
        self.assertIn("(e3-retention-run teardown input output)", runtime)

    def test_retention_reconciliation_mutant_is_rejected(self):
        original = DRIVER.read_text()
        report = stage3_report(original)
        mutant = original.replace(report, report.replace(
            '(d2-token-dataset-close!',
            '(e3-test-retention-reconcile! before after)\n      (d2-token-dataset-close!', 1), 1)
        self.assertFalse(isolated(mutant))

    def test_stage3_fallback_to_retention_report_is_rejected(self):
        original = DRIVER.read_text()
        dispatch_at = original.index('(define (e3-test-fault-dispatch ')
        mutant = original[:dispatch_at] + original[dispatch_at:].replace(
            '(e3-test-fault-stage3-report present?)',
            '(e3-test-retention-report 1024 present?)', 1)
        self.assertFalse(isolated(mutant))


if __name__ == '__main__':
    unittest.main(verbosity=2)
