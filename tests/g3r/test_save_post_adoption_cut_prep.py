"""Source-only checks for the future compiled same-owner failure cut."""
from pathlib import Path
import tempfile
import unittest

from tests.c2.test_handler_order import parse
from tests.g3r.prepare_save_post_adoption_cut import (
    ROOT, CALLER, ADOPTION_RETURN, ABANDON, F31, FE9,
    FE9_RUNNER_SHA256, FE9_RUNTIME_SHA256, frozen_caller, prepare,
)


class PostAdoptionPreparation(unittest.TestCase):
    def test_generated_cut_preserves_real_caller_and_one_cleanup(self):
        original = (ROOT / CALLER).read_text()
        generated = frozen_caller(original)
        self.assertIn(ADOPTION_RETURN +
                      "      (g3r-save-cut-after-publication! (vector-ref shell-cell 0))\n",
                      generated)
        self.assertEqual(generated.count("(g3r-save-cut-abandon! owned-shell)"), 1)
        self.assertNotIn(ABANDON, generated)
        self.assertEqual(len(parse(generated)), len(parse(original)))
        self.assertEqual((ROOT / CALLER).read_text(), original)

    def test_mutated_authority_and_publication_rejected(self):
        original = (ROOT / CALLER).read_text()
        for changed in (original.replace(ADOPTION_RETURN, "", 1),
                        original.replace(ABANDON, "(exit 134)", 1),
                        original.replace("(vector-ref shell-cell 0)",
                                         "shell", 1)):
            with self.subTest(changed=changed[:25]), self.assertRaises(ValueError):
                frozen_caller(changed)

    def test_frozen_closure_contains_real_p1_o2_c2_and_test_cut(self):
        with tempfile.TemporaryDirectory(prefix="g3r-cut-prep-") as directory:
            destination = Path(directory) / "prepared"
            prepare(destination)
            sums = destination / "SOURCE-SHA256SUMS"
            self.assertTrue(sums.is_file())
            self.assertIn("source-prepared; compiled gate not run",
                          (destination / "PREPARED.tsv").read_text())
            root = (destination / "source/tests/g3r/save_post_adoption_cut_root.esk").read_text()
            self.assertIn('(load "g3r_training_controls_root.esk")', root)
            self.assertIn('(load "g3m_prefill_p1_extension.esk")', root)
            self.assertIn('(load "g3r_same_owner_o2_adoption_extension.esk")', root)
            self.assertIn('(load "g3r_same_owner_save_cut_caller.esk")', root)
            self.assertTrue((destination / "source/native/c2_training_state_extension.esk").is_file())
            self.assertTrue((destination / "source/internal/p1/lib/transformer/module.esk").is_file())
            self.assertTrue((destination / "source/tests/g3r/save_model_bits_bridge.c").is_file())
            runtime = (destination / "source/tests/g3r/save_post_adoption_cut_runtime.esk").read_text()
            self.assertIn("(g3m-prefill-p1! generator input)", runtime)
            self.assertNotIn("g3t-with-call", runtime)
            self.assertIn("'((source-code 77)\n                        "
                          "(source-domain g3r-save-o2-adoption))", runtime)
            with self.assertRaises(ValueError):
                prepare(destination)

    def test_supported_runtime_pins_match_existing_gate(self):
        candidate = (ROOT / "tests/tr3_lease/runtime_candidate.tsv").read_text()
        existing = (ROOT / "scripts/test-tr3-c-trainer-load-state.sh").read_text()
        self.assertIn("container_digest\t" + F31, candidate)
        for pin in (FE9, FE9_RUNNER_SHA256, FE9_RUNTIME_SHA256):
            self.assertIn(pin, existing)


if __name__ == "__main__":
    unittest.main()
