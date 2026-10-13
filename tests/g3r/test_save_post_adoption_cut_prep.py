"""Source-only checks for the future compiled same-owner failure cut."""
from pathlib import Path
import inspect
import os
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from tests.c2.test_handler_order import parse
from tests.g3r import prepare_save_post_adoption_cut as preparation
from tests.g3r.prepare_save_post_adoption_cut import (
    ROOT, CALLER, ADOPTION_RETURN, ABANDON, BASE, BASE_TREE, F31, FE9,
    FE9_RUNNER_SHA256, FE9_RUNTIME_SHA256, COMPILER_MANIFEST_SHA256,
    committed_bytes, frozen_caller, parse_unique_tsv, prepare, validate_closure,
    validate_source, validate_toolchain_manifest,
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
            manifest = (destination / "PREPARED.tsv").read_text()
            self.assertEqual(len(sums.read_text().splitlines()), 103)
            self.assertEqual((destination / "COMPILER.tsv").read_bytes(),
                             committed_bytes(subprocess.check_output(
                                 ["git", "rev-parse", "HEAD"], cwd=ROOT,
                                 text=True).strip(), preparation.COMPILER_MANIFEST))
            self.assertEqual(preparation.sha256(
                (destination / "COMPILER.tsv").read_bytes()),
                COMPILER_MANIFEST_SHA256)
            self.assertIn("accepted_source_commit\t" + BASE, manifest)
            self.assertIn("accepted_source_tree\t" + BASE_TREE, manifest)
            self.assertIn("closure_count\t96", manifest)
            self.assertIn("supported_llvm_clang\t21.1.8", manifest)
            self.assertIn("post-adoption longjmp and fail-stop abandon only", manifest)
            self.assertIn("fail_stop_expected_exit\t134", manifest)
            self.assertIn("compiler_manifest_sha256\t" + COMPILER_MANIFEST_SHA256,
                          manifest)
            root = (destination / "source/tests/g3r/save_post_adoption_cut_root.esk").read_text()
            self.assertIn('(load "g3r_training_controls_root.esk")', root)
            self.assertIn('(load "g3m_prefill_p1_extension.esk")', root)
            self.assertIn('(load "g3r_same_owner_o2_adoption_extension.esk")', root)
            self.assertIn('(load "g3r_same_owner_save_cut_caller.esk")', root)
            self.assertTrue((destination / "source/native/c2_training_state_extension.esk").is_file())
            self.assertTrue((destination / "source/internal/p1/lib/transformer/module.esk").is_file())
            runtime = (destination / "source/tests/g3r/save_post_adoption_cut_runtime.esk").read_text()
            self.assertIn("(g3m-prefill-p1! generator input)", runtime)
            self.assertNotIn("g3t-with-call", runtime)
            self.assertIn('"fail-stop"', runtime)
            self.assertIn("G3-R post-adoption RECOVERED ERROR", runtime)
            self.assertIn("G3-R post-adoption RETURNED", runtime)
            self.assertIn("(= (vector-length words) 17)",
                          (ROOT / CALLER).read_text())
            self.assertIn("'((source-code 77)\n                        "
                          "(source-domain g3r-save-o2-adoption))", runtime)
            with self.assertRaises(ValueError):
                prepare(destination)

    def test_dedicated_fe9_manifest_excludes_recovered_candidate(self):
        pins = validate_toolchain_manifest()
        self.assertEqual(pins["compiler_source_commit"], FE9)
        self.assertEqual(pins["runner_sha256"], FE9_RUNNER_SHA256)
        self.assertEqual(pins["runtime_archive_sha256"], FE9_RUNTIME_SHA256)
        self.assertEqual(pins["container_digest"], F31)
        self.assertNotIn("81298b4a", (ROOT / preparation.COMPILER_MANIFEST).read_text())
        self.assertNotIn("runtime_candidate",
                         inspect.getsource(preparation.validate_toolchain_manifest))

    def test_manifest_parser_rejects_duplicate_and_malformed_rows(self):
        for malformed in ("key\ta\nkey\tb\n", "key\n", "key\t\n", "\n", ""):
            with self.subTest(malformed=malformed), self.assertRaises(ValueError):
                parse_unique_tsv(malformed)

    def test_copy_time_compiler_drift_denied_without_prepared_artifact(self):
        original_write = Path.write_bytes
        with tempfile.TemporaryDirectory(prefix="g3r-cut-copy-drift-") as directory:
            parent = Path(directory)
            destination = parent / "prepared"

            def drift_on_stage(path, data):
                if path.name == "COMPILER.tsv":
                    data += b"\n# copy-time drift\n"
                return original_write(path, data)

            with patch.object(Path, "write_bytes", drift_on_stage):
                with self.assertRaisesRegex(ValueError,
                                            "staged compiler manifest differs"):
                    prepare(destination)
            self.assertFalse(destination.exists())
            self.assertEqual(list(parent.iterdir()), [])

    def test_old_base_and_changed_pins_fail_closed_without_output(self):
        with tempfile.TemporaryDirectory(prefix="g3r-cut-negative-") as directory:
            for name, bad in (("BASE_TREE", "0" * 40),
                              ("CALLER_SHA256", "0" * 64),
                              ("ADOPTION_SHA256", "0" * 64),
                              ("CLOSURE_SHA256", "0" * 64),
                              ("LOCK_SHA256", "0" * 64),
                              ("COMPILER_MANIFEST_SHA256", "0" * 64)):
                destination = Path(directory) / name
                with self.subTest(pin=name), patch.object(preparation, name, bad):
                    with self.assertRaises(ValueError):
                        prepare(destination)
                self.assertFalse(destination.exists())

    def test_closure_negative_paths_duplicates_and_missing_leaf(self):
        files = (ROOT / preparation.CLOSURE).read_text().splitlines()
        validate_closure(files)
        for changed in ([*files[:-1], files[0]],
                        [*files[:-1], "../escape.esk"],
                        [*files[:-1], "/absolute.esk"],
                        [*files[:-1], "native/unknown.txt"],
                        [name for name in files if name != CALLER]):
            with self.subTest(last=changed[-1]), self.assertRaises(ValueError):
                validate_closure(changed)

    def test_source_guard_accepts_exact_contract(self):
        self.assertEqual(len(validate_source()), 96)

    def test_fail_stop_reaches_real_release_and_isolated_exit_gate(self):
        root = (ROOT / "tests/g3r/save_post_adoption_cut_root.esk").read_text()
        adoption = (ROOT / "native/g3r_same_owner_o2_adoption_extension.esk").read_text()
        caller = (ROOT / CALLER).read_text()
        launcher = ROOT / "scripts/prepare-g3r-post-adoption-cut.sh"
        release = root.index("(g3r-save-cut-real-o2-release shell)")
        fault = root.index("(raise 'g3r-save-test-o2-release-failure)")
        self.assertLess(release, fault)
        self.assertIn("(eq? (vector-ref g3r-save-cut-state 5)\n"
                      "                             c2-training-state-states)", root)
        self.assertIn("(if (not (o2-public-optimizer-state-release! shell))", adoption)
        self.assertIn("(guard (caught (#t (exit 134)))", adoption)
        self.assertIn("(guard (cleanup (#t (exit 134)))", caller)
        launcher_source = launcher.read_text()
        self.assertIn("runtime verification disabled", launcher_source)
        self.assertNotIn('timeout --foreground', launcher_source)
        for release in ("0", "1"):
            denied = subprocess.run([str(launcher), "--verify-runtime", "/missing",
                                     "/missing", "/missing", "/missing"],
                                    cwd=ROOT,
                                    env={**os.environ,
                                         "G3R_POST_ADOPTION_RUNTIME_RELEASE": release},
                                    capture_output=True, text=True)
            self.assertEqual(denied.returncode, 74)
            self.assertIn("runtime verification disabled", denied.stderr)


if __name__ == "__main__":
    unittest.main()
