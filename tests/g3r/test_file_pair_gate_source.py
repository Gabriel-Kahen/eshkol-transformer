"""Source-only checks for the deferred exact-head G3-R file-pair gate."""

from pathlib import Path
from hashlib import sha256
import subprocess
import tempfile
import unittest

from tests.c2.test_handler_order import parse

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/test-g3r-file-pair.sh"
CLOSURE = ROOT / "native/g3r_file_pair_source_closure.txt"
PREDECESSOR = ROOT / "native/g3r_c2_sha_receipt_source_closure.txt"
OFF = ROOT / "tests/g3r/file_pair_feature_off_test.esk"


class FilePairGateSourceTest(unittest.TestCase):
    def test_exact_root_closure_and_feature_off(self):
        expected = ["native/g3r_file_pair_root.esk",
                    *PREDECESSOR.read_text().splitlines(),
                    "native/g3r_file_pair_extension.esk"]
        self.assertEqual(CLOSURE.read_text().splitlines(), expected)
        self.assertEqual(len(expected), len(set(expected)))
        self.assertTrue(all((ROOT / name).is_file() for name in expected))
        self.assertEqual(parse(OFF.read_text())[0],
                         ["load", '"g3r_c2_sha_receipt_root.esk"'])
        self.assertNotIn("g3r_file_pair_root.esk", OFF.read_text())

    def test_gate_recipe_is_syntactic_and_pinned(self):
        subprocess.run(["bash", "-n", str(RUNNER)], check=True)
        recipe = RUNNER.read_text()
        for required in (
            '[[ $# -eq 3 ]]', '"$expected_commit"', '"$expected_tree"',
            'git -C "$PROJECT_ROOT" rev-parse HEAD',
            "git -C \"$PROJECT_ROOT\" rev-parse 'HEAD^{tree}'",
            'status --porcelain --untracked-files=all',
            'pin_source "$out/pre-source-pins.tsv"',
            'pin_source "$out/post-source-pins.tsv"',
            'cmp "$out/pre-source-pins.tsv" "$out/post-source-pins.tsv"',
            'verify_toolchain', 'eshkol_runner_sha256',
            'compiler_provenance_sha256', 'cc_sha256', 'cxx_sha256',
            'fixture-sha256.txt', 'verify_fixtures "$out/pre-fixture-check.stdout"',
            'verify_fixtures "$out/post-fixture-check.stdout"',
            'cmp "$out/pre-fixture-check.stdout" "$out/post-fixture-check.stdout"',
            'found="$(find . -type l -print -quit)" || die',
            'find . -type f ! -path ./SHA256SUMS-RUN -print0 | LC_ALL=C sort -z',
            'sha256sum -c SHA256SUMS-RUN',
            'prepare_checkpoint_load_fixtures.py',
            'prepare_file_pair_fixtures.py',
            'g3r_file_pair_source_closure.txt',
            'g3r_c2_sha_receipt_source_closure.txt',
            'rg -v \'^tests/g3r/file_pair_test\\.esk$\'',
            'rg -v \'^tests/g3r/file_pair_feature_off_test\\.esk$\'',
        ):
            self.assertIn(required, recipe)
        self.assertEqual(recipe.count('pin_source "$out/'), 2)
        self.assertEqual(recipe.count('build_mode normal'), 1)
        self.assertEqual(recipe.count('build_mode repeat'), 1)
        self.assertEqual(recipe.count('build_mode sanitize'), 1)
        self.assertEqual(recipe.count('  reject_evidence_symlinks'), 2)
        first_check = recipe.index('  reject_evidence_symlinks\n')
        manifest = recipe.index('    xargs -0 sha256sum >SHA256SUMS-RUN')
        verification = recipe.index('  sha256sum -c SHA256SUMS-RUN >/dev/null')
        second_check = recipe.rindex('  reject_evidence_symlinks)')
        self.assertLess(first_check, manifest)
        self.assertLess(manifest, verification)
        self.assertLess(verification, second_check)

    def test_gate_modes_and_fixtures_are_complete(self):
        recipe = RUNNER.read_text()
        for required in (
            'file_pair_test.esk', 'file_pair_feature_off_test.esk',
            'g3r-file-pair-', 'G3-R file pair PASS: checks=14',
            'G3-R file pair off PASS: checks=1',
            'cmp - "$directory/run.stdout"',
            'cmp - "$directory/off-run.stdout"',
            'ASAN_OPTIONS=detect_leaks=1:halt_on_error=1',
            'UBSAN_OPTIONS=halt_on_error=1',
            'LSAN_OPTIONS=exitcode=23:report_objects=1',
            'test ! -s "$directory/build.stderr"',
            'test ! -s "$directory/off-build.stderr"',
            'test ! -s "$directory/off-object-build.stderr"',
            'test ! -s "$directory/run.stderr"',
            'test ! -s "$directory/off-run.stderr"',
            'cmp "$out/normal/run.stdout" "$out/repeat/run.stdout"',
            'cmp "$out/normal/run.stdout" "$out/sanitize/run.stdout"',
            'cmp "$out/normal/off-run.stdout" "$out/sanitize/off-run.stdout"',
            'nm -u --format=posix "$directory/file-pair-test.o"',
            'et_c2_private_checkpoint_load_stage_with_sha_v1',
        ):
            self.assertIn(required, recipe)
        for path in ("good", "short", "long", "torn", "mismatched",
                     "wrong-tokenizer"):
            self.assertIn(f'"$out/pair/{path}.g3r"', recipe)
        self.assertIn('"$out/fixtures/missing.c2"', recipe)
        self.assertIn('"$out/pair/missing.g3r"', recipe)
        fixture = (ROOT / "tests/g3r/file_pair_test.esk").read_text()
        self.assertEqual(fixture.count('(check "') + fixture.count('(reject "'), 14)

    def test_fixture_commands_bind_exact_c2_image(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            c2 = root / "c2"
            pair = root / "pair"
            subprocess.run(["python3", str(ROOT / "tests/c2/prepare_checkpoint_load_fixtures.py"),
                            str(c2)], check=True, capture_output=True)
            subprocess.run(["python3", str(ROOT / "tests/g3r/prepare_file_pair_fixtures.py"),
                            str(c2 / "valid.c2"), str(pair)],
                           check=True, capture_output=True)
            good = (pair / "good.g3r").read_bytes()
            self.assertEqual(good[224:256], sha256((c2 / "valid.c2").read_bytes()).digest())
            self.assertEqual(good[256:288], sha256(good[:256]).digest())
            self.assertEqual((pair / "short.g3r").stat().st_size, 287)
            self.assertEqual((pair / "long.g3r").stat().st_size, 289)
            self.assertFalse((pair / "missing.g3r").exists())


if __name__ == "__main__":
    unittest.main()
