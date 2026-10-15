"""Host source contract for the private trainer-to-G3 held admission."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
ADMISSION = (ROOT / "native/g3r_trainer_g3_save_admission_extension.esk").read_text()
ORDINARY = (ROOT / "native/g3t_model_admission_extension.esk").read_text()
CONSTRUCTOR = (ROOT / "native/g3t_generator_constructor_extension.esk").read_text()
ROOT_SOURCE = (ROOT / "native/g3r_trainer_g3_save_root.esk").read_text()
T2_BRIDGE = (ROOT / "native/cli3_generate_t2_bridge.esk").read_text()


class TrainerG3SaveSource(unittest.TestCase):
    def test_causal_lease_and_cleanup_mutations(self):
        begin = ADMISSION.split("(define (g3r-trainer-g3-save-begin!", 1)[1].split(
            "(define (g3r-trainer-g3-save-end!", 1
        )[0]

        def preserves_contract(source):
            try:
                lease = source.index("(tr3-lease-snapshot-enter-internal! trainer parent)")
                hold = source.index("(g3r-trainer-g3-native-begin")
                publish = source.index("(vector-set! ledger 6 'held)")
            except ValueError:
                return False
            return (lease < hold < publish
                    and "g3r-trainer-g3-native-end" in source[:hold]
                    and "tr3-lease-snapshot-abort-internal!" in source[:hold])

        self.assertTrue(preserves_contract(begin))
        for old, new in (
            ("(tr3-lease-snapshot-enter-internal! trainer parent)", "#f"),
            ("(g3r-trainer-g3-native-end", "(g3r-trainer-g3-native-check"),
            ("(tr3-lease-snapshot-abort-internal!", "(tr3-lease-snapshot-recheck-internal"),
        ):
            self.assertIn(old, begin)
            self.assertFalse(preserves_contract(begin.replace(old, new, 1)))

    def test_real_trainer_lease_precedes_native_hold(self):
        begin = ADMISSION.split("(define (g3r-trainer-g3-save-begin!", 1)[1].split(
            "(define (g3r-trainer-g3-save-end!", 1
        )[0]
        enter = begin.index("(tr3-lease-snapshot-enter-internal! trainer parent)")
        first_recheck = begin.index("(tr3-lease-snapshot-recheck-internal record parent)")
        native = begin.index("(g3r-trainer-g3-native-begin")
        second_recheck = begin.rindex("(tr3-lease-snapshot-recheck-internal record parent)")
        publish = begin.index("(vector-set! ledger 6 'held)")
        self.assertLess(enter, first_recheck)
        self.assertLess(first_recheck, native)
        self.assertLess(native, second_recheck)
        self.assertLess(second_recheck, publish)
        self.assertIn("tr3-lease-snapshot-abort-internal!", begin[:native])
        self.assertIn("g3r-trainer-g3-native-end", begin[:native])

    def test_foreign_stale_busy_and_unleased_inputs_fail_before_begin(self):
        entry = ADMISSION.split("(define (g3r-trainer-g3-save-entry", 1)[1].split(
            "(define (g3r-trainer-g3-save-begin!", 1
        )[0]
        for guard in (
            "tr3-lease-record-for-shell-internal trainer operation",
            "tr3-lease-restore-enter-admission! record operation",
            "(eq? (vector-ref record 3) 'idle)",
            "(eq? (vector-ref entry 2) 'live)",
            "(null? (vector-ref entry 3))",
            "(vector-ref entry 4)",
            "(eq? (vector-ref entry 5) model-entry)",
            "(m3t-find (vector-ref record 8))",
        ):
            self.assertIn(guard, entry)
        self.assertNotIn("g3r-trainer-g3-native-begin", entry)

    def test_no_public_eval_relaxation_or_new_tokenizer_registry(self):
        self.assertIn("(module-mode-internal model) 'eval", ORDINARY)
        self.assertIn("(g3t-model-entry-live model operation #f)", CONSTRUCTOR)
        self.assertIn("(load \"cli3_generate_root.esk\")", ROOT_SOURCE)
        self.assertIn("(define t1-private-entry t2-private-entry)", T2_BRIDGE)
        self.assertNotIn("(provide", ADMISSION)
        self.assertNotIn("(provide", ROOT_SOURCE)

    def test_no_result_or_c2_publication(self):
        for forbidden in (
            "c2-training-state-compose-internal",
            "c2-save-image-receipt",
            "vector-set! result-cell",
            "set! tr3-trainers",
        ):
            self.assertNotIn(forbidden, ADMISSION)


if __name__ == "__main__":
    unittest.main()
