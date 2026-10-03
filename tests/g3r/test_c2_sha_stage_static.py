"""Closed private C2 SHA stage topology checks; no runtime claim."""
from pathlib import Path
import unittest

from tests.c2.test_handler_order import parse

ROOT = Path(__file__).resolve().parents[2]
PRIVATE = ROOT / "native/g3r_c2_sha_stage_root.esk"
MANIFEST = ROOT / "native/g3r_c2_sha_stage_source_closure.txt"
BASE = ROOT / "native/c2_checkpoint_load_source_closure.txt"
RUNNER = ROOT / "scripts/test-g3r-c2-sha-stage.sh"
EXTENSION = ROOT / "native/g3r_c2_sha_stage_extension.esk"

LOADS = ["d2_wave2_root.esk", "c2_d2_cursor_pair_extension.esk",
         "o2_wave2_extension.esk", "c2_o2_reconstruct_extension.esk",
         "c2_x1_canonical_extension.esk", "c2_training_state_extension.esk",
         "c2_persistence_policy_extension.esk", "c2_checkpoint_load_extension.esk",
         "g3r_c2_sha_stage_extension.esk"]


def verify_root_and_manifest(root: str, manifest: str) -> None:
    forms = parse(root)
    actual_loads = [form[1].strip('"') for form in forms
                    if isinstance(form, list) and len(form) == 2
                    and form[0] == "load"]
    if len(forms) != len(LOADS) or actual_loads != LOADS:
        raise ValueError("private owning load order differs")
    expected = ["native/g3r_c2_sha_stage_root.esk",
                *BASE.read_text().splitlines(),
                "native/g3r_c2_sha_stage_extension.esk"]
    if manifest.splitlines() != expected or len(set(expected)) != len(expected):
        raise ValueError("private source closure differs")


def verify_runner(runner: str) -> None:
    if (runner.count("-DET_G3R_C2_STAGE_SHA_PRIVATE") != 1
            or '"$out/load-off.o"' not in runner
            or '"$out/objects/load.o"' not in runner
            or '"$out/objects"/*.o' not in runner
            or "build-e1b-consumer.sh" in runner):
        raise ValueError("private native feature build differs")


class C2ShaStageStaticTest(unittest.TestCase):
    def test_exact_private_ownership_and_symbols(self) -> None:
        verify_root_and_manifest(PRIVATE.read_text(), MANIFEST.read_text())
        verify_runner(RUNNER.read_text())
        extension = EXTENSION.read_text()
        parse(extension)
        parse((ROOT / "tests/g3r/c2_sha_stage_test.esk").read_text())
        self.assertIn(":real et_c2_private_checkpoint_load_stage_with_sha_v1",
                      extension)
        self.assertNotIn("c2-checkpoint-load-reconstruct-internal", extension)
        self.assertNotIn("g3r-c2-load-with-sha-internal", extension)

    def test_near_misses_reject(self) -> None:
        root = PRIVATE.read_text()
        manifest = MANIFEST.read_text()
        runner = RUNNER.read_text()
        with self.assertRaisesRegex(ValueError, "load order"):
            verify_root_and_manifest(root.replace(
                '(load "c2_checkpoint_load_extension.esk")',
                '(load "c2_public_extension.esk")'), manifest)
        with self.assertRaisesRegex(ValueError, "closure"):
            verify_root_and_manifest(root, manifest.replace(
                "native/g3r_c2_sha_stage_extension.esk",
                "native/c2_public_extension.esk"))
        with self.assertRaisesRegex(ValueError, "feature"):
            verify_runner(runner.replace("-DET_G3R_C2_STAGE_SHA_PRIVATE", ""))
        with self.assertRaisesRegex(ValueError, "feature"):
            verify_runner(runner.replace('"$out/load-off.o"',
                                         '"$out/objects/load-off.o"'))


if __name__ == "__main__":
    unittest.main()
