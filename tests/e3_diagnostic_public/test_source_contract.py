"""Installed E3 facade shape and authority-boundary checks."""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[2]


class PublicSourceContract(unittest.TestCase):
    def test_facade_exports_only_reviewed_three_operations(self) -> None:
        source = (ROOT / "lib/transformer/evaluation.esk").read_text()
        provided = re.search(r"\(provide ([^)]+)\)", source, re.S)
        self.assertIsNotNone(provided)
        self.assertEqual(provided.group(1).split(), [
            "diagnostic-evaluate-fixed!", "diagnostic-evaluation-f32-bits",
            "diagnostic-evaluation-count",
        ])
        for symbol in (
            "et_e1b_public_e3_diagnostic_evaluate_fixed_v1",
            "et_e1b_public_e3_diagnostic_evaluation_f32_bits_v1",
            "et_e1b_public_e3_diagnostic_evaluation_count_v1",
        ):
            self.assertEqual(source.count(symbol), 1)
        self.assertNotIn("report-stats", source)
        self.assertNotIn("retention", source)

    def test_bridge_boxes_only_existing_two_argument_operations(self) -> None:
        source = (ROOT / "native/e3_diagnostic_public_bridge.c").read_text()
        self.assertEqual(source.count("ET_E3_PUBLIC_BINARY("), 4)
        self.assertIn('#include "m3_package_bridge.c"', source)
        self.assertIn('#include "e3_d2_public_wrappers.inc"', source)
        self.assertNotIn("et_e3_test", source)
        self.assertNotIn("report_stats", source)

    def test_public_runtime_proves_numeric_and_failure_boundary(self) -> None:
        source = (ROOT / "tests/e3_diagnostic_public/public_runtime.esk").read_text()
        for bits in ("1085366897", "1084227584", "1132430055"):
            self.assertIn(bits, source)
        self.assertGreaterEqual(source.count('(bytes=?'), 2)
        self.assertIn("'invalid-state 'e3-evaluate-into!", source)
        self.assertIn('"prior report after failure"', source)
        self.assertIn("(token-dataset-open", source)
        self.assertIn("(diagnostic-model-create", source)
        self.assertIn("(tokenizer-byte", source)

    def test_installed_quota_witness_uses_exact_mixed_reservation_horizon(self) -> None:
        source = (ROOT / "tests/e3_diagnostic_public/quota_runtime.esk").read_text()
        self.assertIn("((remaining 8190)", source)
        self.assertIn('"failed reservation returns no public result"', source)
        self.assertIn('"failed reservation cursor restoration"', source)
        self.assertIn('"failed value is not report authority"', source)
        self.assertIn('"8193rd reservation rejection"', source)
        self.assertIn("'unsupported", source)
        self.assertIn("'diagnostic-evaluate-fixed!", source)
        self.assertIn('"oldest report survives cap rejection"', source)

        gate = (ROOT / "scripts/test-e3-diagnostic-public.sh").read_text()
        self.assertIn("E3_PUBLIC_QUOTA_TIMEOUT_SECONDS:-900", gate)
        self.assertIn("public_runtime quota_runtime", gate)


if __name__ == "__main__":
    unittest.main()
