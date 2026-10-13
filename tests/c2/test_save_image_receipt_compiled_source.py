"""Static admission for the pending compiled C2 receipt gate."""
from pathlib import Path
import unittest

from tests.c2.test_handler_order import parse


ROOT = Path(__file__).resolve().parents[2]
DRIVER = ROOT / "tests/c2/c2_save_image_receipt_runtime.esk"
SCRIPT = ROOT / "scripts/test-c2-save-image-receipt-compiled.sh"


class ReceiptCompiledSource(unittest.TestCase):
    def test_driver_parses_and_uses_real_private_root(self):
        source = DRIVER.read_text()
        self.assertTrue(parse(source))
        self.assertIn('(load "c2_save_image_receipt_root.esk")', source)
        for call in (
            "(c2-checkpoint-save-internal! owner public-path public-policy '())",
            "(c2-save-image-receipt-create-internal owner policy)",
            "(c2-save-image-receipt-digest-copy-internal receipt)",
            "(c2-save-image-receipt-publish-internal! receipt image-path)",
            "(c2-save-image-receipt-release-internal! receipt)",
        ):
            self.assertIn(call, source)

    def test_failure_cuts_and_cleanup_are_present(self):
        source = DRIVER.read_text()
        for cut in (
            "held owner rejects receipt",
            "measured over cap before destination",
            "token backing rejected before FFI",
            "malformed destination rejected before FFI",
            "post-adopt exception releases native image",
            "no-overwrite collision",
            "precommit writer failure",
            "committed writer failure",
            "retained SHA mismatch",
            "receipt registry returns to baseline",
            "uncertain ownership",
        ):
            self.assertIn(cut, source)

    def test_source_only_mode_precedes_toolchain_and_aot(self):
        script = SCRIPT.read_text()
        self.assertLess(script.index("C2_IMAGE_SOURCE_ONLY"),
                        script.index("resolve_provenance_compilers"))
        self.assertIn("-DET_C2_SAVE_IMAGE_RECEIPT_PRIVATE", script)
        self.assertIn("-DET_C2_SAVE_IMAGE_RECEIPT_TESTING", script)
        self.assertIn("ESHKOL_ARENA_POISON=1", script)
        self.assertIn("hashlib.sha256", script)
        self.assertIn("cmp \"$tmp/a/image.c2\" \"$tmp/b/image.c2\"", script)


if __name__ == "__main__":
    unittest.main()
