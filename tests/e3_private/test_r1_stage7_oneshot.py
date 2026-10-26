"""Stage-7 role-one report isolation without weakening the failure/retry proof."""
import hashlib
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
DRIVER = ROOT / 'tests/e3_private/fault_driver.esk'
RUNTIME = ROOT / 'tests/e3_private/r1_stage7_runtime.esk'
EXECUTE_SHA256 = '59e048219506c1b5ea7cbeeea2596e101427fa600d5e422aeeb3a24aeed3cad8'


def forms(source):
    execute = source[source.index('(define (e3-test-fault-execute '):
                     source.index(';;; The role-1 cut')]
    report = source[source.index('(define (e3-test-fault-stage7-report '):
                    source.index('(define (e3-test-fault-dispatch ')]
    dispatch = source[source.index('(define (e3-test-fault-dispatch '):]
    return execute, report, dispatch


def isolated(driver, runtime):
    execute, report, dispatch = forms(driver)
    branch = dispatch[dispatch.rindex('(let ((result'):]
    return (hashlib.sha256(execute.encode()).hexdigest() == EXECUTE_SHA256 and
            '(= (vector-ref scenario 3) 1)' in branch and
            '(= (vector-ref scenario 4) 1)' in branch and
            '(= (vector-ref scenario 5) 1)' in branch and
            '(e3-test-fault-stage7-report present?)' in branch and
            '(e3-test-retention-report 1024 present?)' in branch and
            'e3-test-retention-reconcile!' not in report and
            'e3-test-retention-report' not in report and
            "'stage7-report-identity" in report and
            "'stage7-report-idle" in report and
            "'stage7-report-dataset-close" in report and
            0 <= report.find('(d2-token-dataset-close!') <
            report.find('(vector-set! record 17') <
            report.find('(vector-set! record 18 #t)') and
            "'e3-r1-stage7-one-shot-v1 1 1 present? before after" in report and
            "(define scenario (vector 'fault 'prepare present? 1 1 1))" in runtime and
            "'e3-r1-stage7-one-shot-v1" in runtime and
            "'e3-retention-result-v1" not in runtime and
            '(e3-retention-run teardown input output)' in runtime)


class Stage7OneShotSourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.driver = DRIVER.read_text()
        cls.runtime = RUNTIME.read_text()

    def test_exact_role_one_report_and_existing_execute(self):
        self.assertTrue(isolated(self.driver, self.runtime))
        execute, _, _ = forms(self.driver)
        for check in ("'first-error", "'cursor", "'output", "'modes",
                      "'model", "'rng", "'owners", "'independent-twin",
                      "'retry-output", "'retry-rng"):
            self.assertIn(check, execute)

    def test_retention_reconciliation_mutant_rejected(self):
        _, report, _ = forms(self.driver)
        mutant = self.driver.replace(report, report.replace(
            '(d2-token-dataset-close!',
            '(e3-test-retention-reconcile! before after)\n'
            '      (d2-token-dataset-close!', 1), 1)
        self.assertFalse(isolated(mutant, self.runtime))

    def test_wrong_selector_and_fallback_mutants_rejected(self):
        _, _, dispatch = forms(self.driver)
        branch_start = dispatch.rindex('(let ((result')
        prefix = self.driver[:self.driver.index('(define (e3-test-fault-dispatch ')]
        branch = dispatch[branch_start:]
        for old, new in (('(= (vector-ref scenario 3) 1)', '(= (vector-ref scenario 3) 5)'),
                         ('(= (vector-ref scenario 4) 1)', '(= (vector-ref scenario 4) 0)'),
                         ('(= (vector-ref scenario 5) 1)', '(= (vector-ref scenario 5) 2)'),
                         ('(e3-test-fault-stage7-report present?)',
                          '(e3-test-retention-report 1024 present?)')):
            self.assertIn(old, branch)
            self.assertFalse(isolated(prefix + dispatch[:branch_start] +
                                      branch.replace(old, new, 1), self.runtime))

    def test_close_and_teardown_mutants_rejected(self):
        _, report, _ = forms(self.driver)
        for old, new in (('(d2-token-dataset-close!', '(begin '),
                         ('(vector-set! record 17', '(vector-set! record 16'),
                         ('(vector-set! record 18 #t)', '(vector-set! record 18 #f)')):
            self.assertFalse(isolated(self.driver.replace(
                report, report.replace(old, new, 1), 1), self.runtime))
        self.assertFalse(isolated(self.driver, self.runtime.replace(
            '(e3-retention-run teardown input output)', '(begin teardown input output)')))

    def test_first_error_and_runtime_marker_mutants_rejected(self):
        self.assertFalse(isolated(self.driver.replace("'first-error", "'last-error", 1),
                                  self.runtime))
        self.assertFalse(isolated(self.driver, self.runtime.replace(
            "'e3-r1-stage7-one-shot-v1", "'e3-retention-result-v1", 1)))


if __name__ == '__main__':
    unittest.main(verbosity=2)
