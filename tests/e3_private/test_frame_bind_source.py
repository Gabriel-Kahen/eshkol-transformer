"""Guard the E3 frame bind's native-token-safe root readback shape."""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "native/e3_private_extension.esk"


class FrameBindSourceTests(unittest.TestCase):
    def test_canonical_readback_uses_fresh_bindings(self) -> None:
        source = SOURCE.read_text()
        bind = source.split("(define (e3-frame-bind! model destinations)", 1)[1]
        bind = bind.split("(define (e3-frame-d2-record-internal frame)", 1)[0]
        self.assertIn("(set! canonical (car (vector-ref e3-frame-registry-root 0)))", bind)
        self.assertIn("(let* ((root-token (vector-ref canonical 0))", bind)
        for name, slot in (("root-model", 2), ("root-model-entry", 3),
                           ("root-destinations", 5), ("root-setup-box", 8)):
            self.assertIn(f"({name} (vector-ref canonical {slot}))", bind)
        self.assertIn("(p1-e3-mode-bind! root-model root-setup-box)", bind)
        self.assertIn("(vector-ref root-model-entry 3)", bind)
        self.assertIn("(vector-ref root-setup-box 1)", bind)
        self.assertIn("(vector-ref (vector-ref canonical 8) 1)", bind)
        self.assertTrue(bind.rstrip().endswith("root-token))))))))"))
        for name in ("token", "model", "model-entry", "destinations", "setup-box"):
            self.assertIsNone(re.search(r"\(set!\s+" + re.escape(name) + r"\b", bind))


if __name__ == "__main__":
    unittest.main()
