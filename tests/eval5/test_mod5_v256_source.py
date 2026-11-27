"""Source-only checks for the bounded Eshkol AOT witness composition."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from tests.eval5 import build_mod5_v256_witness as source_builder
from tests.eval5.mod5_v256_fixture import build


def balanced(source):
    depth = 0
    quoted = escaped = comment = False
    for ch in source:
        if comment:
            if ch == "\n":
                comment = False
        elif quoted:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                quoted = False
        elif ch == ";":
            comment = True
        elif ch == '"':
            quoted = True
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0 and not quoted


class Mod5V256SourceTests(unittest.TestCase):
    def test_pinned_owner_and_generated_contract(self):
        with TemporaryDirectory(prefix="eval5-v256-source-") as temp:
            root = Path(temp) / "fixture"
            record = build(root)
            source = source_builder.generate(root)
            self.assertTrue(balanced(source))
            self.assertEqual(source.count("(define (decoder weights input target gradients tied?)"), 1)
            self.assertEqual(source.count("(define (load-weights path)"), 1)
            self.assertEqual(source.count("(define (read-split directory expected-sha)"), 1)
            self.assertIn("(c1-sha256 manifest) expected-sha", source)
            self.assertNotIn('directory "/manifest.etm"', source)
            self.assertIn("(c1-sha256 bytes) expected-model-sha", source)
            self.assertIn("(c1-checkpoint-decode-state-internal\n                    bytes policy", source)
            self.assertIn("(weights (load-checked-weights model-path))", source)
            self.assertNotIn("(check-file-sha model-path", source)
            self.assertIn("(load-weights (caddr args))", source)
            self.assertEqual(source.count("(define vocab 256)"), 1)
            self.assertEqual(source.count("(define span-bytes (* 4 vocab width))"), 1)
            self.assertIn(record["tokenizer_fingerprint"], source)
            self.assertNotIn("@@", source)
            self.assertNotIn("volatile-load f64", source)
            self.assertNotIn("cddddr", source)
            self.assertIn("score-metric (ffi-pointer metrics) i", source)
            with patch.object(source_builder, "OWNER_SHA256", "0" * 64):
                with self.assertRaises(ValueError):
                    source_builder.generate(root)


if __name__ == "__main__":
    unittest.main()
