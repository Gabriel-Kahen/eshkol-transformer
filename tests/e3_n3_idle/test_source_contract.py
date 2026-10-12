"""Source-only N3 control checks; package admission and AOT are still held."""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
DRIVER = ROOT / "tests/e3_n3_idle/driver.esk"


class IdleModelSourceTest(unittest.TestCase):
    def test_separate_entry_and_unchanged_base_abi(self) -> None:
        source = DRIVER.read_text()
        original = (ROOT / "tests/e3_private/driver.esk").read_text()
        root = (ROOT / "tests/e3_n3_idle/driver_root.esk").read_text()
        bridge = (ROOT / "tests/e3_n3_idle/bridge.c").read_text()
        self.assertIn("(define (e3-test-run scenario corpus-path)", original)
        self.assertIn("(= (vector-length scenario) 2)", original)
        self.assertNotIn("e3-n3-", original)
        self.assertIn("(define (e3-n3-zero-test-run scenario corpus-path)", source)
        self.assertIn("(= (vector-length scenario) 3)", source)
        self.assertEqual(root.count('(load "../e3_private/driver.esk")'), 1)
        self.assertEqual(root.count('(load "driver.esk")'), 1)
        self.assertIn("et_e3_n3_zero_test_run_cabi_v1", bridge)
        self.assertIn("et_e3_n3_zero_test_run_v1", bridge)

    def test_load_order_and_authentic_sources(self) -> None:
        source = DRIVER.read_text()
        constructor = source[source.index("(define (e3-n3-zero-test-run") :]
        self.assertLess(constructor.index("g3t-checked-m3t-model-create"),
                        constructor.index("(vector-set! e3-n3-model-roots"))
        self.assertLess(constructor.index("(e3-n3-zero-load! model handles)"),
                        constructor.index("(e3-frame-bind! model destinations)"))
        self.assertIn("(parameter-tree-paths tree)", source)
        self.assertIn("(parameter-tree-tie-groups tree)", source)
        self.assertIn("(parameter-handle-tensor-internal handle)", source)
        self.assertIn("(i2-checkpoint-decode", source)
        self.assertIn("(vector-ref decoded 7)", source)
        self.assertIn("(i2-prepare-load (vector rows e3-n3-operation))", source)
        self.assertRegex(source, re.compile(
            r"\(i2-prepare-load \(vector rows e3-n3-operation\)\)\)+\s*"
            r"\(i2-commit-load! \(vector-ref plan 0\)\)"))
        self.assertIn("(i2-release-owned!", source)
        self.assertIn("(e3-n3-seed-gradient", source)
        self.assertIn("(e3-n3-gradient-image before)", source)
        self.assertIn("(e3-n3-zero-image? loaded)", source)
        self.assertIn("(if frame\n                (begin (e3-frame-unbind! frame)", source)
        self.assertLess(source.rindex("(e3-frame-unbind! frame)"),
                        source.rindex("(e3-test-destinations-destroy! destinations)"))

    def test_build_policy_explicitly_holds_candidate(self) -> None:
        policy = (ROOT / "scripts/e3-private-package-policy.sh").read_text()
        builder = (ROOT / "scripts/build-e1b-consumer.sh").read_text()
        self.assertNotIn("e3_n3_idle", policy)
        self.assertNotIn("e3_n3_idle", builder)
        self.assertNotIn("et_e3_n3_zero_test_run_v1", builder)


if __name__ == "__main__":
    unittest.main()
