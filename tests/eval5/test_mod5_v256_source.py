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
            self.assertIn("(manifest-sha (c1-sha256 manifest))", source)
            self.assertIn("(c1-bytes=? manifest-sha expected-sha)", source)
            self.assertNotIn('directory "/manifest.etm"', source)
            self.assertIn("(digest (c1-sha256 bytes))", source)
            self.assertIn("(c1-bytes=? digest expected-model-sha)", source)
            self.assertIn("(c1-checkpoint-decode-state-internal\n                    bytes policy", source)
            self.assertIn("(audit-result (audit-splits train evaluation 0))", source)
            self.assertIn("(loaded (load-checked-weights model-path))", source)
            self.assertLess(source.index("(audit-result (audit-splits train evaluation 0))"),
                            source.index("(loaded (load-checked-weights model-path))"))
            self.assertIn("et_eval5_v256_manifest_roundtrip_live", source)
            self.assertLess(source.index("(score-v256 (ffi-pointer"),
                            source.rindex("(emit-manifest tokenizer-fingerprint train evaluation model-sha"))
            self.assertIn("et_eval5_v256_compare_runner", source)
            emit = source.split("(define (emit-manifest", 1)[1].split(
                "(define (release-score-buffers", 1)[0]
            self.assertIn("(train-fingerprint (string->utf8 (vector-ref train 3)))", emit)
            self.assertIn("(eval-fingerprint (string->utf8 (vector-ref evaluation 3)))", emit)
            self.assertIn("(copy-digest (vector-ref train 2) digests 0)", emit)
            self.assertIn("(copy-digest (vector-ref evaluation 2) digests 32)", emit)
            self.assertIn("(copy-digest model-sha digests 64)", emit)
            self.assertLess(emit.index("(compare-runner-v256"), emit.index("(emit-score"))
            self.assertLess(emit.index("(if (and valid-manifest (= runner-status 0))"),
                            emit.index('(display "MANIFEST_BEGIN")'))
            self.assertIn("(vector all original manifest-sha (vector-ref metadata 4))", source)
            self.assertIn("(audit-splits train train -7)", source)
            control = source.split("(define (audit-control", 1)[1].split("(define (witness-main", 1)[0]
            self.assertNotIn("emit-manifest", control)
            self.assertNotIn("score-v256", control)
            self.assertNotIn("compare-runner-v256", control)
            self.assertNotIn("et_eval5_manifest_", source)
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
