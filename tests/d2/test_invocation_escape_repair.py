"""Source-only fault/control ownership contract for D2's shard escape candidate."""
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'internal/d2/lib/d2_dataset.esk'
BASE = '2e941bb769511430e3b600d53207d99b2cd89e75'


def section(text, start, end):
    return text[text.index(start):text.index(end, text.index(start))]


def contract(text):
    macro = section(text, '(define-syntax d2-with-shard-region', '(define (d2-shuffle-ref!')
    packed = section(text, '(define (d2-fill-packed-row!', '(define (d2-fill-unpacked-row!')
    unpacked = section(text, '(define (d2-fill-unpacked-row!', '(define (d2-fill-row!')
    call = section(text, '(define (d2-token-dataset-next-batch', '(define (d2-token-dataset-cursor')
    return all((
        '((_ operation body ...)' in macro,
        '(let ((promoted (region-close region caught)))\n'
        '                    (vector d2-region-exception-marker promoted))' in macro,
        '(raise (vector-ref result 1))' in macro,
        '((eq? result d2-region-close-failure-marker)' in macro,
        '(region-close region)\n                   #t' in macro,
        '(vector-set! escape 0 promoted)' not in macro,
        '(vector-ref state 11)' not in packed + unpacked + call,
        "(d2-with-shard-region 'token-dataset-next-batch\n" in packed,
        "(d2-with-shard-region 'token-dataset-next-batch\n" in unpacked,
        packed.count("(d2-with-shard-region 'token-dataset-next-batch\n") == 3,
        unpacked.count("(d2-with-shard-region 'token-dataset-next-batch\n") == 1,
        "(d2-with-shard-region 'token-dataset-next-batch escape" not in packed + unpacked,
        '(vector-set! state 11 #f)' in text,
        '(caller-escape (vector #f))' not in call,
    ))


class InvocationEscapeRepairTest(unittest.TestCase):
    def test_fault_only_envelope_and_both_row_modes(self):
        self.assertTrue(contract(SOURCE.read_text()))

    def test_rooted_or_success_path_regressions_are_rejected(self):
        original = SOURCE.read_text()
        for old, new in (
            ('(vector d2-region-exception-marker promoted)',
             '(vector-set! (vector-ref state 11) 0 promoted)'),
            ("(d2-with-shard-region 'token-dataset-next-batch\n",
             "(d2-with-shard-region 'token-dataset-next-batch escape\n"),
            ('(vector-set! state 11 #f)', '(vector-set! state 11 (vector #f))'),
            ('(let ((promoted (region-close region caught)))',
             '(let ((promoted caught))'),
        ):
            self.assertIn(old, original)
            self.assertFalse(contract(original.replace(old, new, 1)), old)

    def test_typed_error_cleanup_and_cursor_commit_match_predecessor(self):
        current = SOURCE.read_text()
        prior = subprocess.check_output(['git', '-C', str(ROOT), 'show',
                                         f'{BASE}:internal/d2/lib/d2_dataset.esk'],
                                        text=True)
        self.assertEqual(section(current, '(define (d2-exact-read-fail',
                                 '(define (d2-exact-file-bytes'),
                         section(prior, '(define (d2-exact-read-fail',
                                 '(define (d2-exact-file-bytes'))
        for seam in ("(d2-native-batch-release dataset generation)",
                     "(raise caught)",
                     "(vector-set! (vector-ref state 6) 15 (vector-ref range 2))"):
            self.assertIn(seam, current)
            self.assertEqual(current.count(seam), prior.count(seam))

    def test_repaired_fault_probe_and_control_run_in_fresh_processes(self):
        driver = (ROOT / 'tests/e3_private/retention_arena_diagnostic_driver.esk').read_text()
        pair = (ROOT / 'scripts/e3-retention-d2-map-pair.sh').read_text()
        self.assertIn("'d2-map-no-dataset-escape", driver)
        self.assertIn('for mode in d2-fault d2-map-fault d2-control', pair)
        self.assertIn('[[ "$status" == 0 && "$sample" == 1 ]] || overall=1', pair)


if __name__ == '__main__':
    unittest.main()
