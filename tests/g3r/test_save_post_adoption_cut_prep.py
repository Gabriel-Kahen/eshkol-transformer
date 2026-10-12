"""Source-only checks for the future compiled same-owner failure cut."""
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tests.c2.test_handler_order import parse
from tests.g3r import prepare_save_post_adoption_cut as prep
from tests.g3r.prepare_save_post_adoption_cut import (
    ROOT, CALLER, ADOPTION_RETURN, ABANDON, F31, FE9,
    FE9_RUNNER_SHA256, FE9_RUNTIME_SHA256, E3_LOAD_ALIAS, E3_LOAD_SOURCE,
    REPAIRED_CORE, frozen_caller, prepare,
)
from tests.g3r.test_training_controls_capsule_source import branch, descendants


class PostAdoptionPreparation(unittest.TestCase):
    def test_lease_create_guard_has_handler_and_body(self):
        source = (ROOT / REPAIRED_CORE).read_text()
        def construction_guard(text):
            route = parse(text)[-1]
            create = branch(route, "create")
            return next(g for g in descendants(create, "guard")
                        if "tr3-lease-unenroll-internal!" in str(g[1]))
        guard = construction_guard(source)
        self.assertEqual(len(guard), 3)
        self.assertEqual(guard[1][0], "caught")
        self.assertIn("['vector-set!', 'pending', '0', '#f']", str(guard[1]))
        self.assertIn("['raise', 'caught']", str(guard[1]))
        self.assertEqual(guard[2][0], "let")
        self.assertIn("['tr3-lease-create-internal', 'a', 'b', 'c', 'd', 'e']",
                      str(guard[2]))
        broken = source.replace("(raise caught))))\n               ;; The first operation",
                                "(raise caught)))\n               ;; The first operation", 1)
        broken = broken.replace("                   shell)))))\n          ((eq? mode 'originals)",
                                "                   shell))))))\n          ((eq? mode 'originals)", 1)
        self.assertEqual(len(construction_guard(broken)), 2)

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
            self.assertIn("repaired_lease_core_sha256\t" + prep.REPAIRED_CORE_SHA256,
                          (destination / "PREPARED.tsv").read_text())
            root = (destination / "source/tests/g3r/save_post_adoption_cut_root.esk").read_text()
            self.assertIn('(load "g3r_training_controls_root.esk")', root)
            self.assertIn('(load "g3m_prefill_p1_extension.esk")', root)
            self.assertIn('(load "g3r_same_owner_o2_adoption_extension.esk")', root)
            self.assertIn('(load "g3r_same_owner_save_cut_caller.esk")', root)
            self.assertIn(":real et_g3r_test_save_idle_ledger_copy_v1", root)
            self.assertTrue((destination / "source/native/c2_training_state_extension.esk").is_file())
            self.assertTrue((destination / "source/internal/p1/lib/transformer/module.esk").is_file())
            self.assertEqual((destination / "source" / E3_LOAD_ALIAS).read_bytes(),
                             (destination / "source" / E3_LOAD_SOURCE).read_bytes())
            self.assertIn(f"source/{E3_LOAD_ALIAS}", sums.read_text())
            expected = destination / "expected-eshkol-closure.txt"
            closure = expected.read_text().splitlines()
            self.assertEqual(len(closure), 99)
            self.assertEqual(closure, sorted(set(closure)))
            self.assertIn(E3_LOAD_ALIAS, closure)
            self.assertNotIn(E3_LOAD_SOURCE, closure)
            self.assertNotIn(prep.CALLER, closure)
            self.assertNotIn("native/g3r_same_owner_o2_adoption_root.esk", closure)
            self.assertIn("tests/g3r/g3r_same_owner_save_cut_caller.esk", closure)
            self.assertIn("expected-eshkol-closure.txt", sums.read_text())
            self.assertIn('(load "e3_d2_restore.esk")',
                          (destination / "source/native/tr3_public_installed_root.esk").read_text())
            self.assertTrue((destination / "source/tests/g3r/save_model_bits_bridge.c").is_file())
            self.assertTrue((destination / "source/tests/g3r/save_idle_ledger_bridge.c").is_file())
            self.assertEqual(
                (destination / "source/src/eshkol_transformer/g3t_transport.h").read_bytes(),
                (ROOT / "src/eshkol_transformer/g3t_transport.h").read_bytes())
            for name in ("native/f32_parameter_internal.h", "native/f32_tensor.c"):
                self.assertEqual((destination / "source" / name).read_bytes(),
                                 (ROOT / name).read_bytes())
            runtime = (destination / "source/tests/g3r/save_post_adoption_cut_runtime.esk").read_text()
            self.assertIn("(g3m-prefill-p1! generator input)", runtime)
            self.assertLess(runtime.index("(define idle-ledger-before"),
                            runtime.index('"typed post-adoption failure"'))
            self.assertLess(runtime.index('"typed post-adoption failure"'),
                            runtime.index('"original G3 RNG and committed history unchanged"'))
            self.assertIn("(vector-ref entry 3) image 144", runtime)
            self.assertIn("d2-core-bytes=? idle-ledger-before (capture-original-idle-ledger)", runtime)
            self.assertNotIn("g3t-with-call", runtime)
            self.assertIn("'((source-code 77)\n                        "
                          "(source-domain g3r-save-o2-adoption))", runtime)
            with self.assertRaises(ValueError):
                prepare(destination)

    def test_repaired_core_pin_rejects_changed_bytes(self):
        with tempfile.TemporaryDirectory(prefix="g3r-cut-pin-") as directory:
            with mock.patch.object(prep, "REPAIRED_CORE_SHA256", "0" * 64):
                with self.assertRaisesRegex(ValueError, "repaired lease core changed"):
                    prepare(Path(directory) / "prepared")

    def test_supported_runtime_pins_match_existing_gate(self):
        candidate = (ROOT / "tests/tr3_lease/runtime_candidate.tsv").read_text()
        existing = (ROOT / "scripts/test-tr3-c-trainer-load-state.sh").read_text()
        self.assertIn("container_digest\t" + F31, candidate)
        for pin in (FE9, FE9_RUNNER_SHA256, FE9_RUNTIME_SHA256):
            self.assertIn(pin, existing)


if __name__ == "__main__":
    unittest.main()
