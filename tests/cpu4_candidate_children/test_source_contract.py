"""Cheap source/recipe guards; execution proof belongs to the compiled gate."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]

class CustodyContract(unittest.TestCase):
    def test_exact_native_abi_and_status(self):
        text = (ROOT / 'native/cpu4_candidate_children_extension.esk').read_text()
        self.assertIn('(extern i32 cpu4-owned-native-o2-preflight ptr ptr', text)
        self.assertIn('(cpu4-owned-native-o2-preflight (vector-ref row 8) \'())', text)
        tail = text[text.index('(let ((status (cpu4-owned-native-o2-preflight'):]
        self.assertNotIn('(o2-native-fail operation)', tail)

    def test_no_public_optimizer_publication_tail(self):
        text = (ROOT / 'native/cpu4_candidate_children_extension.esk').read_text()
        self.assertNotIn('(o2-public-optimizer-create ', text)
        constructor = text[text.index('(define (cpu4-owned-o2-create!'):text.index('(define (cpu4-owned-optimizer-create-internal!')]
        self.assertLess(constructor.index('(guard (caught'), constructor.index('(set! o2-optimizers next)'))
        self.assertLess(constructor.index('(vector-set! row 7 record)'), constructor.index('(o2-native-builder-create'))
        tail = constructor[constructor.index('(vector-set! row 14 \'finished)'):constructor.index('(let* ((handle')]
        self.assertNotIn('(set! ', tail)
        self.assertNotIn('(cons ', tail)
        self.assertNotIn('(vector ', tail)
        self.assertIn('(vector-set! row 8 native)', tail)

    def test_retry_and_irreversible_boundary(self):
        text = (ROOT / 'native/cpu4_candidate_children_extension.esk').read_text()
        retire = text[text.index('(define (cpu4-owned-candidate-retire-internal!'):]
        self.assertLess(retire.index('(cpu4-owned-o2-unlink-plan'), retire.index('(cpu4-trainer-detach-internal!'))
        self.assertLess(retire.index('(vector-set! row 11 #t)'), retire.index('(cpu4-candidate-retire-owned-optimizer-internal!'))
        self.assertIn("(eq? (vector-ref row 3) 'payload-retired)", retire)
        self.assertIn('(set-cdr! (vector-ref plan 0) (vector-ref plan 2))', retire)
        for index in range(6):
            self.assertIn(f'(vector-set! (vector-ref row 7) {index} #f)', retire)

    def test_real_fixture_and_instrumentation(self):
        recipe = (ROOT / 'scripts/test-cpu4-candidate-children.sh').read_text()
        fixture = (ROOT / 'tests/cpu4_candidate_children/runtime_test.esk').read_text()
        self.assertIn('-DET_CPU4_CANDIDATE_PRIVATE -DET_O2_TESTING -DET_F32_TENSOR_TESTING', recipe)
        self.assertIn('tests/cpu4_candidate_children/native_observer.c', recipe)
        self.assertIn('(with-region (\'cpu4-o2-owned-promotion 1048576)', fixture)
        self.assertIn('(allocation-cuts #f)', fixture)
        self.assertIn('(allocation-cuts #t)', fixture)
        self.assertIn('live O2 snapshot blocks before detach', fixture)

if __name__ == '__main__':
    unittest.main()
