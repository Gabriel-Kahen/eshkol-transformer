"""Static gate for the private receipt's source closure and admission order."""
from pathlib import Path
import unittest

from tests.c2.test_handler_order import parse


ROOT = Path(__file__).resolve().parents[2]
SAVE = (ROOT / "native/c2_checkpoint_save_extension.esk").read_text()
RECEIPT = (ROOT / "native/c2_save_image_receipt_extension.esk").read_text()


class ReceiptSource(unittest.TestCase):
    def test_sources_parse_and_feature_is_private(self):
        for name in ("c2_checkpoint_save_extension.esk",
                     "c2_save_image_receipt_extension.esk",
                     "c2_save_image_receipt_root.esk"):
            self.assertTrue(parse((ROOT / "native" / name).read_text()))
        source_closure = (ROOT / "native/c2_save_image_receipt_source_closure.txt").read_text()
        native_closure = (ROOT / "native/c2_save_image_receipt_native_source_closure.txt").read_text()
        self.assertIn("native/c2_save_image_receipt_extension.esk\n", source_closure)
        self.assertIn("native/c2_save_image_receipt.c\n", native_closure)
        for public in ("native/c2_wave2_root.esk",
                       "native/c2_wave2_source_closure.txt",
                       "native/tr3_public_installed_source_closure.txt"):
            self.assertNotIn("c2_save_image_receipt", (ROOT / public).read_text())

    def test_measured_cap_precedes_destination_and_encode(self):
        measure = SAVE.index("(c2-save-measure-staged-internal\n                                staged limits)")
        cap = SAVE.index("(> file-bytes max-image-bytes)", measure)
        destination = SAVE.index("(c2-save-destination-allocate-internal file-bytes)", cap)
        encode = SAVE.index("(c2-save-encode-staged-into-internal!", destination)
        self.assertLess(measure, cap)
        self.assertLess(cap, destination)
        self.assertLess(destination, encode)
        self.assertIn("(publish destination file-bytes)", SAVE)
        self.assertIn("(not (= (bytevector-length bytes) measured-bytes))", RECEIPT)
        self.assertLess(RECEIPT.index("(not (= (bytevector-length bytes) measured-bytes))"),
                        RECEIPT.index("(c2-image-adopt bytes"))

    def test_closed_wrappers_and_no_overwrite(self):
        self.assertIn("(c2-image-publish (c2-image-token cell) path 0)", RECEIPT)
        self.assertIn("(c2-image-release token)", RECEIPT)
        self.assertIn("(else (loop (car remaining) (cdr remaining)))", RECEIPT)
        self.assertIn("(c2-image-fail 'invalid-state \"C2 image receipt is consumed\")", RECEIPT)
        self.assertNotIn("(define-public", RECEIPT)


if __name__ == "__main__":
    unittest.main()
