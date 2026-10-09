"""Source-only guard checks for the staged E3/zero-control composition."""

import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src/eshkol_transformer/e3_frame.c"
BASE = [
    "cc", "-E", "-P", "-DET_M3T_PACKAGE_BUILD",
    "-Iinclude", "-Inative", "-Isrc",
]


def expanded(*defines: str) -> str:
    result = subprocess.run(
        [*BASE, *defines, str(SOURCE)], cwd=ROOT,
        capture_output=True, text=True, check=True,
    )
    return result.stdout


class StagedZeroUnion(unittest.TestCase):
    def test_production_and_predecessor_guards(self) -> None:
        off = expanded()
        numerical = expanded("-DET_E3_NUMERICAL_TESTING")
        retention = expanded("-DET_E3_RETENTION_TESTING")
        for source in (off, numerical, retention):
            self.assertNotIn("et_e3_zero_control_anchor_v1(", source)
        self.assertNotIn("e3_numerical_batch(", off)
        self.assertIn("e3_numerical_batch(", numerical)
        self.assertIn("et_e3_test_retention_probe_v1(", retention)
        self.assertIn("ET_E3_NUMERICAL_WRONG_TARGET", SOURCE.read_text())

    def test_zero_only_enables_real_model_observation(self) -> None:
        zero = expanded(
            "-DET_E3_ZERO_CONTROL_TESTING", "-DET_F32_TENSOR_TESTING",
            "-DET_I2_PRIVATE_OWNED_CLONE_MATCH",
        )
        self.assertIn("et_e3_zero_control_anchor_v1(", zero)
        self.assertIn("e3_numerical_batch(", zero)
        self.assertNotIn("et_e3_test_retention_probe_v1(", zero)
        self.assertIn("et_i2_private_owned_clone_match_v1(", zero)


if __name__ == "__main__":
    unittest.main()
