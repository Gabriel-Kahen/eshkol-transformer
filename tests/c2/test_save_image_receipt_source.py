"""Static gate for the private receipt's source closure and admission order."""
from pathlib import Path
import re
import unittest

from tests.c2.test_handler_order import parse


ROOT = Path(__file__).resolve().parents[2]
SAVE = (ROOT / "native/c2_checkpoint_save_extension.esk").read_text()
RECEIPT = (ROOT / "native/c2_save_image_receipt_extension.esk").read_text()
DRIVER = (ROOT / "tests/c2/c2_save_image_receipt_runtime.esk").read_text()
C1 = (ROOT / "internal/c1/lib/transformer/checkpoint_internal.esk").read_text()


def assert_aot_contract(save, receipt, driver, c1):
    """Guard the four source forms rejected by the first private AOT attempt."""
    guard = save.split("(> file-bytes max-image-bytes)", 1)[1].split(
        "(c2-save-destination-allocate-internal file-bytes)", 1)[0]
    assert "(c2-save-fail-raw" in guard
    assert "(c2-save-fail\n" not in guard
    assert "'transformer.c2-save-image-receipt-v1" in receipt
    assert not re.search(r"'transformer\.c2-save-image-receipt:v1", receipt)
    private_exports = c1.split("(provide", 1)[1].split(")", 1)[0]
    assert "c1-native-fail-raw" in private_exports.split()
    assert "(c1-native-fail-raw\n" in receipt
    assert "(length c2-image-receipts)" not in driver
    assert driver.count("(c1-list-length-bounded c2-image-receipts 64)") == 3
    assert '(check "receipt registry baseline is bounded" (integer? baseline))' in driver


class ReceiptSource(unittest.TestCase):
    def test_compiler_sensitive_private_contract(self):
        assert_aot_contract(SAVE, RECEIPT, DRIVER, C1)
        for mutated in (
            (SAVE.replace("(c2-save-fail-raw\n", "(c2-save-fail\n", 1),
             RECEIPT, DRIVER, C1),
            (SAVE, RECEIPT.replace("'transformer.c2-save-image-receipt-v1",
                                   "'transformer.c2-save-image-receipt:v1", 1),
             DRIVER, C1),
            (SAVE, RECEIPT, DRIVER,
             C1.replace("c1-native-fail-raw c1-write-u16!", "c1-write-u16!", 1)),
            (SAVE, RECEIPT.replace("(c1-native-fail-raw\n", "(c1-native-fail-raw-missing\n", 1),
             DRIVER, C1),
            (SAVE, RECEIPT,
             DRIVER.replace("(c1-list-length-bounded c2-image-receipts 64)",
                            "(length c2-image-receipts)", 1), C1),
        ):
            with self.assertRaises(AssertionError):
                assert_aot_contract(*mutated)

    def test_raw_io_failure_seam_stays_outside_installed_facades(self):
        for root in (ROOT / "src", ROOT / "lib"):
            for source in root.rglob("*.esk"):
                self.assertNotIn("c1-native-fail-raw", source.read_text(), source)

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
