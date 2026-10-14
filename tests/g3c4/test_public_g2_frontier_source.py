"""Causal mutations of the current public C2/P2G1 source boundary."""

import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "g2_frontier", ROOT / "scripts/check-g3c4-public-g2-frontier.py")
frontier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frontier)


class FrontierMutationTests(unittest.TestCase):
    def setUp(self):
        self.sources = frontier.current_sources()
        self.assertEqual(frontier.issues(self.sources), set())

    def mutate(self, name, before, after, expected):
        self.assertIn(before, self.sources[name])
        self.sources[name] = self.sources[name].replace(before, after, 1)
        self.assertIn(expected, frontier.issues(self.sources))

    def test_budget_widening_rejected(self):
        self.mutate("native/g3c4_public_generation_dispatch_extension.esk",
                    "(= (vector-ref policy 5) 1)",
                    "(= (vector-ref policy 5) 2)", "p2g1-profile-budget-eos")

    def test_eos_widening_rejected(self):
        self.mutate("native/g3c4_public_generation_dispatch_extension.esk",
                    "(eq? (vector-ref policy 6) #f)",
                    "(eq? (vector-ref policy 6) #t)", "p2g1-profile-budget-eos")

    def test_c4_fallthrough_to_c2_rejected(self):
        self.mutate("native/g3g_manual_public_extension.esk",
                    "(if c4-member", "(if #f", "c4-c2-fallback-order")

    def test_premature_terminal_package_load_rejected(self):
        self.sources["native/g3g_manual_package_root.esk"] += (
            '\n(load "g3c4_p2g2_terminal_commit_extension.esk")\n')
        self.assertIn("premature-terminal-package-load",
                      frontier.issues(self.sources))

    def test_scalar_output_relabel_rejected(self):
        self.mutate("native/g3c4_public_output_adapter_extension.esk",
                    "(g3c4-p2g1-output-copy-internal shell)",
                    "(g3c4-p2g2-terminal-snapshot-private shell)",
                    "scalar-adapter-not-g2")

    def test_private_clone_export_rejected(self):
        self.sources["native/g3g_manual_package_public_exports.txt"] += (
            "et_g3c4_private_p2g2_terminal_ids_clone_v1\n")
        self.assertIn("private-clone-public-exposure",
                      frontier.issues(self.sources))

    def test_native_first_release_rejected(self):
        name = "native/g3c4_public_output_adapter_extension.esk"
        before = """            (g3c4-native-check
              (if (eq? expected 'result-rng)
                  (g3c4-native-rng-release (vector-ref live 3))
                  (g3c4-native-result-tensor-release (vector-ref live 3)))
              operation)
            (g3c4-entry-dead! live)"""
        after = """            (g3c4-entry-dead! live)
            (g3c4-native-check
              (if (eq? expected 'result-rng)
                  (g3c4-native-rng-release (vector-ref live 3))
                  (g3c4-native-result-tensor-release (vector-ref live 3)))
              operation)"""
        self.mutate(name, before, after, "exact-owner-native-first-release")

    def test_p2g1_native_release_order_rejected(self):
        self.mutate(
            "native/g3c4_p2g1_shell_publication_extension.esk",
            """              (g3c4-native-check
                (g3c4-native-output-release (vector-ref live 3)) operation)
              (g3c4-entry-dead! live)""",
            """              (g3c4-entry-dead! live)
              (g3c4-native-check
                (g3c4-native-output-release (vector-ref live 3)) operation)""",
            "exact-owner-native-first-release")

    def test_c4_output_fallthrough_to_c2_rejected(self):
        self.mutate("native/g3g_manual_public_extension.esk",
                    "(g3c4-public-generation-output-ids output)",
                    "(g3t-generation-output-ids output)",
                    "c4-c2-fallback-order")

    def test_public_input_c2_budget_widening_rejected(self):
        self.mutate("native/g3g_manual_public_extension.esk",
                    "(= (vector-ref (vector-ref normalized 0) 5) 1)",
                    "(= (vector-ref (vector-ref normalized 0) 5) 2)",
                    "c4-c2-fallback-order")


if __name__ == "__main__":
    unittest.main()
