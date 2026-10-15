"""Host checks for the separate private C1/D2 reachability tuple."""

from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
import unittest

from tests.c2.test_handler_order import parse as parse_source
from tests.e3_private.parse_minsubnormal_observation import (
    MARKER, parse_minsubnormal, require_first_logit,
)
from tests.e3_private.test_numerical_tuple import frame_bytes, frame_lines


ROOT = Path(__file__).resolve().parents[2]
PREFIX = ROOT / "tests/e3_private/minsubnormal_package"
UNIFORM = ROOT / "tests/e3_private/uniform_package"


def manifest(prefix: Path, suffix: str) -> list[str]:
    return Path(f"{prefix}_{suffix}.txt").read_text().splitlines()


class MinSubnormalDriverTest(unittest.TestCase):
    def test_private_tuple_closure_and_source_order(self) -> None:
        driver = (ROOT / "tests/e3_private/minsubnormal_driver.esk").read_text()
        runtime = (ROOT / "tests/e3_private/minsubnormal_runtime.esk").read_text()
        parse_source(driver)
        parse_source(runtime)
        self.assertLess(driver.index("(c1-checkpoint-load-module-internal!"),
                        driver.index("(e3-frame-bind! model destinations)"))
        self.assertLess(driver.index("(e3-frame-bind! model destinations)"),
                        driver.index("(e3-evaluate-into! model dataset frame)"))
        self.assertIn("(e3-minsubnormal-check-owners (module-parameters model) #t)",
                      driver)
        self.assertIn("(= owner 10)", driver)
        self.assertIn("(= owner 11)", driver)
        self.assertIn("(not e3-minsubnormal-bias-clear?)", driver)
        self.assertIn("(e3-minsubnormal-assert-post model", driver)
        self.assertIn("(d2-token-dataset-open", driver)
        self.assertIn("(d2-token-dataset-cursor dataset)", driver)
        self.assertNotIn("(vector-set! logits", driver)
        source = manifest(PREFIX, "source_closure")
        native = manifest(PREFIX, "native_source_closure")
        self.assertEqual(source[0], "tests/e3_private/minsubnormal_driver_root.esk")
        self.assertEqual(source[-1], "tests/e3_private/minsubnormal_driver.esk")
        self.assertIn("tests/e3_private/minsubnormal_bridge.c", native)
        self.assertNotIn("tests/e3_private/uniform_driver.esk", source)
        self.assertNotIn("tests/e3_private/uniform_bridge.c", native)
        self.assertEqual(manifest(PREFIX, "archive_members"),
                         ["e3_minsubnormal_package.o"])
        for suffix in ("private_renames", "public_exports", "native_objects"):
            self.assertEqual(manifest(PREFIX, suffix), manifest(UNIFORM, suffix))

    def test_exact_tuple_and_generated_stage_reject_cross_wiring(self) -> None:
        policy = ROOT / "scripts/e3-private-package-policy.sh"
        script = '''
PROJECT_ROOT=$1
raw_private_root=$2
raw_package_bridge=$3
raw_package_renames=$4
raw_public_exports=$5
raw_include_dirs=("${PROJECT_ROOT}/internal/p1/lib" "${PROJECT_ROOT}/internal/c1/lib" "${PROJECT_ROOT}/internal/t1/lib" "${PROJECT_ROOT}/src" "${PROJECT_ROOT}/internal/d2/lib" "${PROJECT_ROOT}/internal/e3/lib")
die() { printf '%s\\n' "$*" >&2; exit 77; }
source "$6"
printf '%s\\n' "$e3_tuple_kind"
'''
        def choose(bridge: Path, exports: Path):
            return subprocess.run(
                ["bash", "-c", script, "bash", str(ROOT),
                 str(ROOT / "tests/e3_private/minsubnormal_driver_root.esk"),
                 str(bridge), str(Path(f"{PREFIX}_private_renames.txt")),
                 str(exports), str(policy)], capture_output=True, text=True,
            )
        correct = choose(ROOT / "tests/e3_private/minsubnormal_bridge.c",
                         Path(f"{PREFIX}_public_exports.txt"))
        self.assertEqual((correct.returncode, correct.stdout), (0, "minsubnormal\n"))
        wrong = choose(ROOT / "tests/e3_private/uniform_bridge.c",
                       Path(f"{PREFIX}_public_exports.txt"))
        self.assertEqual(wrong.returncode, 77)
        with tempfile.TemporaryDirectory() as temporary:
            stage = Path(temporary) / ".e3-minsubnormal.ABC123"
            stage.mkdir()
            helper = ROOT / "scripts/e3-generated-source-stage.sh"
            def select(name: str):
                return subprocess.run(
                    ["bash", "-c", 'source "$1"; e3_minsubnormal_generated_stage "$2"',
                     "bash", str(helper), str(stage / name)], capture_output=True,
                    text=True,
                )
            self.assertEqual(select("e3_minsubnormal_package.o").returncode, 0)
            self.assertNotEqual(select("e3_uniform_package.o").returncode, 0)

    def test_first_physical_logit_and_bias_clear_mutation(self) -> None:
        lines = frame_lines(3)
        for index, line in enumerate(lines):
            if "\tZ\t1,2,256\t" in line:
                fields = line.split("\t")
                words = ["00000000"] * 512
                words[0] = words[256] = "00000001"
                fields[-1] = ",".join(words)
                lines[index] = "\t".join(fields)
        frame = frame_bytes(lines)
        raw = MARKER + frame[:-len(b"E3-NUMERICAL-RUNTIME-PASS\n")] + (
            b"E3CURSOR\t00\t00\nE3-NUMERICAL-RUNTIME-PASS\n")
        observed, _ = parse_minsubnormal(raw, 3, "00")
        require_first_logit(observed)
        observed["batches"][0]["logits_f32"][0] = "00000000"
        with self.assertRaisesRegex(ValueError, "first physical logit differs"):
            require_first_logit(observed)
        with self.assertRaisesRegex(ValueError, "cursor differs"):
            parse_minsubnormal(raw, 3, "01")


if __name__ == "__main__":
    unittest.main()
