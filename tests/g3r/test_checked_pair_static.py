"""Source-closure checks for the bounded private checked-pair leaf."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = (ROOT / "native/g3r_checked_pair_extension.esk").read_text()
FIXTURE = (ROOT / "tests/g3r/checked_pair_test.esk").read_text()
COMPOSITION = (ROOT / "native/g3r_checked_pair_root.esk").read_text()
RECEIPT_CLOSURE = (ROOT / "native/g3r_c2_sha_receipt_source_closure.txt").read_text().splitlines()
PAIR_CLOSURE = (ROOT / "native/g3r_checked_pair_source_closure.txt").read_text().splitlines()
PAIR_EXTRA = [
    "native/g3r_record_codec_extension.esk",
    "native/g3r_checked_pair_extension.esk",
    "native/g3r_checked_pair_root.esk",
    "tests/g3r/checked_pair_test.esk",
    "tests/g3r/test_checked_pair_static.py",
    "scripts/test-g3r-checked-pair.sh",
    "docs/g3/G3_R_CHECKED_PAIR_LEAF.md",
    "native/g3r_checked_pair_source_closure.txt",
]


class CheckedPairSourceTest(unittest.TestCase):
    def test_private_composition(self):
        self.assertIn('(load "g3r_c2_sha_receipt_root.esk")', COMPOSITION)
        self.assertIn('(load "g3r_record_codec_extension.esk")', COMPOSITION)
        self.assertIn('(load "g3r_checked_pair_extension.esk")', COMPOSITION)
        self.assertLess(COMPOSITION.index('g3r_c2_sha_receipt_root.esk'),
                        COMPOSITION.index('g3r_checked_pair_extension.esk'))
        self.assertEqual(PAIR_CLOSURE, RECEIPT_CLOSURE + PAIR_EXTRA)
        self.assertEqual(len(PAIR_CLOSURE), len(set(PAIR_CLOSURE)))
        self.assertTrue(all((ROOT / path).is_file() for path in PAIR_CLOSURE))
        self.assertIn("internal/c1/lib/c1_sha256.esk", PAIR_CLOSURE)

    def test_borrow_and_codec_precede_registry_publication(self):
        admission = SOURCE.split("(define (g3r-checked-pair-admit-internal", 1)[1]
        admission = admission.split("(define (g3r-checked-pair-recheck-internal", 1)[0]
        self.assertLess(admission.index("g3r-c2-receipt-borrow-find"),
                        admission.index("g3r-record-decode!"))
        self.assertLess(admission.index("g3r-record-decode!"),
                        admission.index("(set! g3r-checked-pairs next)"))
        self.assertIn("(c1-sha256 owned)", admission)
        self.assertIn("(bytevector-copy source 0 g3r-record-size)", admission)
        self.assertIn("g3r-c2-receipt-borrow-sha-internal", admission)

    def test_recheck_seals_record_and_exact_receipt(self):
        recheck = SOURCE.split("(define (g3r-checked-pair-recheck-internal", 1)[1]
        for token in ("g3r-checked-pair-find", "g3r-c2-receipt-find",
                      "(eq? (vector-ref entry 1) receipt-shell)",
                      "(eq? (vector-ref entry 2) (vector-ref receipt 2))",
                      "(c1-sha256 (vector-ref entry 3))",
                      "(vector-ref receipt 3)"):
            self.assertIn(token, recheck)
        for token in ("(open ", "(read ", "(write ", "checkpoint-save",
                      "tr3-c-joint-restore", "generator-generate!"):
            self.assertNotIn(token, SOURCE)

    def test_retirement_clears_all_retained_authority(self):
        retire = SOURCE.split("(define (g3r-checked-pair-retire-internal!", 1)[1]
        self.assertIn("(vector-set! entry 6 'dead)", retire)
        for slot in (3, 4, 5):
            self.assertIn(f"(g3r-checked-pair-zero! (vector-ref entry {slot}))", retire)
        for slot in (1, 2, 3, 4, 5):
            self.assertIn(f"(vector-set! entry {slot} #f)", retire)

    def test_negative_fixture_covers_mutation_and_foreign_identity(self):
        for label in ("short record", "foreign borrow", "torn caller record",
                      "valid record for another C2 image", "forged pair shell",
                      "foreign receipt", "foreign active pair",
                      "mutated sealed bytes", "dead receipt",
                      "dead receipt cannot copy out",
                      "retired pair cannot copy out", "retired pair cannot recheck"):
            self.assertIn(f'"{label}"', FIXTURE)


if __name__ == "__main__":
    unittest.main()
