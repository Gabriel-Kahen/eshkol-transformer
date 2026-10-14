"""Static admission for the pending compiled C2 receipt gate."""
from pathlib import Path
import re
import unittest

from tests.c2.test_handler_order import parse


ROOT = Path(__file__).resolve().parents[2]
DRIVER = ROOT / "tests/c2/c2_save_image_receipt_runtime.esk"
SCRIPT = ROOT / "scripts/test-c2-save-image-receipt-compiled.sh"
NATIVE_SOURCES = ROOT / "native/c2_save_image_receipt_compiled_native_sources.txt"


def native_compiler_inputs(script):
    """Expand the gate's literal C compile inputs; reject unknown source forms."""
    marker = 'native="$tmp/native"'
    assert script.count(marker) == 1
    build = script.split(marker, 1)[1].split('ar rcsD ', 1)[0]
    assert build and 'ar rcsD ' in script
    build = build.replace('\\\n', ' ')
    loops = re.findall(r'for source in ([a-z0-9_\s]+); do', build)
    assert len(loops) == 1
    names = loops[0].split()
    assert names and len(names) == len(set(names))
    paths = re.findall(r'-c\s+"\$PROJECT_ROOT/([^"\s]+\.c)"', build)
    references = re.findall(r'"\$PROJECT_ROOT/([^"\s]+\.c)"', build)
    assert paths == references and build.count('.c"') == len(paths)
    assert len(paths) == len(re.findall(r'(?<!\S)-c\s+', build))
    assert paths.count('native/$source.c') == 1
    inputs = []
    for path in paths:
        if path == 'native/$source.c':
            inputs.extend(f'native/{name}.c' for name in names)
        else:
            assert '$' not in path
            inputs.append(path)
    return inputs


class ReceiptCompiledSource(unittest.TestCase):
    def test_exact_native_compiler_inputs_match_tracked_manifest(self):
        manifest = NATIVE_SOURCES.read_text()
        expected = manifest.splitlines()
        self.assertTrue(manifest.endswith('\n'))
        self.assertEqual(len(expected), len(set(expected)))
        self.assertTrue(all(path.endswith('.c') and (ROOT / path).is_file()
                            for path in expected))
        script = SCRIPT.read_text()
        self.assertEqual(native_compiler_inputs(script), expected)
        self.assertLess(script.index('tests.c2.test_save_image_receipt_compiled_source'),
                        script.index('if [[ "${C2_IMAGE_SOURCE_ONLY:-0}"'))

    def test_native_manifest_rejects_build_list_drift(self):
        script = SCRIPT.read_text()
        expected = NATIVE_SOURCES.read_text().splitlines()
        self.assertNotEqual(native_compiler_inputs(
            script.replace('native/c2_save_image_receipt.c',
                           'native/c2_checkpoint_format.c', 1)), expected)
        self.assertNotEqual(native_compiler_inputs(
            script.replace('data_io kernel_abi', 'kernel_abi data_io', 1)), expected)
        self.assertNotEqual(native_compiler_inputs(
            script.replace('data_io kernel_abi', 'kernel_abi', 1)), expected)
        with self.assertRaises(AssertionError):
            native_compiler_inputs(script.replace(
                'c2_checkpoint_format; do',
                'c2_checkpoint_format data_io; do', 1))
        with self.assertRaises(AssertionError):
            native_compiler_inputs(script.replace(
                '"$PROJECT_ROOT/native/c2_save_image_receipt.c"',
                '"${PROJECT_ROOT}/native/c2_save_image_receipt.c"', 1))

    def test_driver_parses_and_uses_real_private_root(self):
        source = DRIVER.read_text()
        self.assertTrue(parse(source))
        self.assertIn('(load "c2_save_image_receipt_root.esk")', source)
        for call in (
            "(c2-checkpoint-save-internal! owner public-path public-policy '())",
            "(c2-save-image-receipt-create-internal owner policy)",
            "(c2-save-image-receipt-digest-copy-internal receipt)",
            "(c2-save-image-receipt-publish-internal! receipt image-path)",
            "(c2-save-image-receipt-release-internal! receipt)",
        ):
            self.assertIn(call, source)

    def test_failure_cuts_and_cleanup_are_present(self):
        source = DRIVER.read_text()
        for cut in (
            "held owner rejects receipt",
            "measured over cap before destination",
            "token backing rejected before FFI",
            "malformed destination rejected before FFI",
            "post-adopt exception releases native image",
            "no-overwrite collision",
            "precommit writer failure",
            "committed writer failure",
            "retained SHA mismatch",
            "receipt registry returns to baseline",
            "repeat native bytes baseline",
            "repeat native roots baseline",
            "repeat Eshkol roots baseline",
            "(c2-image-test-fail-next 4)",
            "uncertain ownership",
        ):
            self.assertIn(cut, source)

    def test_source_only_mode_precedes_toolchain_and_aot(self):
        script = SCRIPT.read_text()
        self.assertLess(script.index("C2_IMAGE_SOURCE_ONLY"),
                        script.index("resolve_provenance_compilers"))
        self.assertIn("-DET_C2_SAVE_IMAGE_RECEIPT_PRIVATE", script)
        self.assertIn("-DET_C2_SAVE_IMAGE_RECEIPT_TESTING", script)
        self.assertIn("ESHKOL_ARENA_POISON=1", script)
        self.assertIn("hashlib.sha256", script)
        self.assertIn("cmp \"$tmp/a/image.c2\" \"$tmp/b/image.c2\"", script)


if __name__ == "__main__":
    unittest.main()
