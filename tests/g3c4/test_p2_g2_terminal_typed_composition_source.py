"""Static gate for the private same-registry terminal-to-typed composition.

Compiled generation, fault, ownership and feature-off proof is still required.
"""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
PRIVATE = ROOT / "native/g3c4_p2g2_terminal_typed_private_root.esk"
BRIDGE = ROOT / "native/g3c4_p2g2_terminal_typed_extension.esk"
INSTALLED = ROOT / "native/g3g_manual_package_root.esk"
NATIVE = ROOT / "src/eshkol_transformer/g3c4_model_owner.c"


def balanced(source):
    depth = 0
    for line in source.splitlines():
        line = line.split(";;;", 1)[0]
        depth += line.count("(") - line.count(")")
        if depth < 0:
            return False
    return depth == 0


class TerminalTypedCompositionSource(unittest.TestCase):
    def test_private_root_has_one_ordered_registry(self):
        loads = [line.strip() for line in PRIVATE.read_text().splitlines()
                 if line.startswith("(load ")]
        self.assertEqual(loads, [
            '(load "g3g_manual_package_root.esk")',
            '(load "g3c4_p2g2_ownership_extension.esk")',
            '(load "g3c4_p2g2_protected_scope_extension.esk")',
            '(load "g3c4_p2g2_t1_prefix_coordinator_extension.esk")',
            '(load "g3c4_p2g2_terminal_close_extension.esk")',
            '(load "g3c4_p2g2_terminal_detached_extension.esk")',
            '(load "g3c4_p2g2_terminal_typed_extension.esk")',
        ])
        for name in ("terminal_close", "terminal_detached", "terminal_typed"):
            self.assertNotIn(name, INSTALLED.read_text())
        self.assertTrue(balanced(BRIDGE.read_text()))

    def test_authentic_clone_precedes_typed_owners_and_cleanup(self):
        source = BRIDGE.read_text()
        self.assertIn("g3c4-p2g2-terminal-snapshot-internal", source)
        self.assertIn("g3c4-p2g2-terminal-ids-clone-internal", source)
        self.assertIn("g3c4-p2g2-terminal-ids-entry shell operation #f", source)
        self.assertIn("g3c4-p2g2-terminal-ids-copy-internal shell", source)
        self.assertIn("g3c4-p2g2-native-typed-ids-create native", source)
        self.assertIn("g3c4-p2g2-native-typed-scalar-create 11 count", source)
        self.assertIn("g3c4-p2g2-native-typed-scalar-create 12 cache", source)
        self.assertLess(source.index("(set! clone-shell shell)"),
                        source.index("(set! ids"))
        self.assertIn("(= (vector-ref snapshot 3) (+ 2 count))", source)
        self.assertGreaterEqual(source.count(
            "(g3c4-p2g2-terminal-typed-source-check"), 5)
        self.assertIn("g3c4-p2g2-terminal-typed-length ids operation", source)
        self.assertIn("g3c4-p2g2-terminal-typed-kind ids operation", source)
        self.assertIn("g3c4-p2g2-terminal-typed-kind lengths operation", source)
        self.assertIn("g3c4-p2g2-terminal-typed-value ids 0 operation", source)
        self.assertIn("g3c4-p2g2-terminal-typed-value ids 1 operation", source)
        self.assertIn("g3c4-p2g2-terminal-typed-value\n                           lengths 0 operation", source)
        self.assertIn("g3c4-p2g2-terminal-typed-value\n                           cache-lengths 0 operation", source)
        self.assertLess(source.index("(set! cache-lengths"),
                        source.rindex("(g3c4-p2g2-terminal-ids-release-internal! shell)"))
        self.assertIn("(exit 134)", source)
        self.assertNotIn("(m3-call", source)
        self.assertNotIn("g3c4-public-", source)

    def test_native_typed_feature_remains_separate(self):
        source = NATIVE.read_text()
        self.assertIn("#ifdef ET_G3C4_P2_G2_TYPED_RESULT_PRIVATE", source)
        self.assertIn("et_g3c4_private_p2g2_typed_ids_create_v1", source)
        self.assertIn("et_g3c4_private_p2g2_typed_scalar_create_v1", source)
        self.assertIn("et_g3c4_private_p2g2_typed_release_v1", source)
        self.assertIn("et_g3c4_private_p2g2_typed_value_v1", source)
        self.assertIn("et_g3c4_private_p2g2_typed_kind_v1", source)


if __name__ == "__main__":
    unittest.main()
