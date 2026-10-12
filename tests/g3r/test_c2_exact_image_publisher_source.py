"""Focused source gate for the private exact-image publication ordering."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = (ROOT / "native/g3r_c2_exact_image_publisher_extension.esk").read_text()
C2_SAVE = (ROOT / "native/c2_checkpoint_save_extension.esk").read_text()


def definition(source: str, name: str) -> str:
    start = source.index(f"(define ({name} ")
    end = source.find("\n(define (", start + 1)
    return source[start:] if end < 0 else source[start:end]


def publisher_contract(source: str) -> bool:
    image = definition(source, "g3r-pair-publish-encoded-image-internal!")
    entry = definition(source, "g3r-c2-exact-image-pair-publish-internal!")
    owner = definition(C2_SAVE, "c2-save-with-owner-borrow")
    ordered = (
        "(c1-sha256 bytes)",
        "(g3r-record-encode!",
        "(g3r-record-decode! record sha roundtrip)",
        "(g3r-pair-write-internal\n              c2-path bytes (bytevector-length bytes) 0)",
        "(g3r-pair-write-internal\n              companion-path record g3r-record-size 0)",
    )
    if not all(part in image for part in ordered):
        return False
    if [image.index(part) for part in ordered] != sorted(image.index(part) for part in ordered):
        return False
    if "(if (not (= c2-status 0))" not in image or "(if (not (= record-status 0))" not in image:
        return False
    if not all(part in entry for part in (
        "(c1-valid-file-path? c2-path)",
        "(c1-valid-file-path? companion-path)",
        "(not (string=? c2-path companion-path))",
        "(file-exists? c2-path)",
        "(file-exists? companion-path)",
        "(g3r-pair-publish-fields-copy fields)",
        "(c2-save-with-owner-borrow",
        "(lambda (bytes)",
        "(g3r-pair-publish-encoded-image-internal!",
    )):
        return False
    if entry.index("(g3r-pair-publish-fields-copy fields)") > entry.index("(c2-save-with-owner-borrow"):
        return False
    if not all(part in owner for part in (
        "(c2-save-encode-staged-into-internal!",
        "(c2-training-state-borrow-end-internal",
        "(publish destination)",
    )):
        return False
    if owner.index("(c2-save-encode-staged-into-internal!") > owner.index("(publish destination)"):
        return False
    if owner.index("(c2-training-state-borrow-end-internal") > owner.index("(publish destination)"):
        return False
    return True


class ExactImagePublisherSourceTest(unittest.TestCase):
    def test_validated_exact_image_precedes_both_no_overwrite_writes(self) -> None:
        self.assertTrue(publisher_contract(SOURCE))

    def test_wrong_image_or_order_is_rejected_by_gate(self) -> None:
        mutations = (
            ("(c1-sha256 bytes)", "(c1-sha256 record)"),
            ("(g3r-record-decode! record sha roundtrip)", "#t"),
            ("c2-path bytes (bytevector-length bytes) 0", "c2-path record g3r-record-size 0"),
            ("companion-path record g3r-record-size 0", "companion-path bytes (bytevector-length bytes) 0"),
            ("(if (not (= c2-status 0))", "(if #f"),
            ("(if (not (= record-status 0))", "(if #f"),
            ("(not (string=? c2-path companion-path))", "#t"),
            ("(file-exists? companion-path)", "#f"),
            ("(copied (g3r-pair-publish-fields-copy fields))", "(copied fields)"),
        )
        for before, after in mutations:
            with self.subTest(before=before):
                self.assertIn(before, SOURCE)
                self.assertFalse(publisher_contract(SOURCE.replace(before, after, 1)))

    def test_fields_are_copied_and_no_record_does_file_io_during_preflight(self) -> None:
        copy = definition(SOURCE, "g3r-pair-publish-fields-copy")
        entry = definition(SOURCE, "g3r-c2-exact-image-pair-publish-internal!")
        self.assertIn("(g3r-record-fields?", copy)
        self.assertIn("g3r-save-tokenizer-fingerprint", copy)
        self.assertIn("(vector-copy policy)", copy)
        self.assertIn("(vector-copy rng)", copy)
        self.assertEqual(copy.count("(bytevector-copy "), 2)
        self.assertLess(entry.index("(file-exists? companion-path)"),
                        entry.index("(c2-save-with-owner-borrow"))


if __name__ == "__main__":
    unittest.main()
