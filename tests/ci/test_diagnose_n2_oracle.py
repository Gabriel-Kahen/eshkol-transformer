from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.ci import diagnose_n2_oracle as diagnostic
from tests.q0.oracle_format import encode_fixture


def _payload(word: str = "3f800000", *, seed: int = 1) -> dict:
    return {
        "cases": [{"expectation": {"error": None, "outputs": ["sample.output"]},
                   "inputs": [], "kind": "known_value", "name": "sample.case",
                   "operation": "sample", "tolerance": {"absolute": "0000000000000000",
                   "equal_nan": False, "relative": "0000000000000000"}}],
        "generator": {"dependency_lock_sha256": "0" * 64,
                      "framework": {"name": "pytorch", "version": "test"},
                      "name": "test.generator", "seed": seed,
                      "source_sha256": "1" * 64, "version": 1},
        "tensors": [{"data": [word], "device": "cpu", "dtype": "float32",
                     "encoding": "ieee754-hex-be", "layout": "row_major",
                     "name": "sample.output", "role": "expected", "shape": [1]}],
    }


class DiagnoseN2OracleTests(unittest.TestCase):
    def test_exact_fixture_matches(self) -> None:
        fixture = encode_fixture(_payload())
        report = diagnostic.compare_fixtures(fixture, fixture)
        self.assertTrue(report["exact_match"])
        self.assertEqual(report["byte_differences"]["count"], 0)
        self.assertEqual(report["tensor_word_differences"]["count"], 0)

    def test_reports_exact_word_value_and_metadata_delta(self) -> None:
        report = diagnostic.compare_fixtures(
            encode_fixture(_payload()), encode_fixture(_payload("40000000", seed=2)))
        self.assertFalse(report["exact_match"])
        difference = report["tensor_word_differences"]["first"][0]
        self.assertEqual((difference["frozen_word"], difference["generated_word"]),
                         ("3f800000", "40000000"))
        self.assertEqual(difference["generated_minus_frozen"]["repr"], "1.0")
        paths = {item["path"] for item in report["payload_metadata_differences"]["first"]}
        self.assertIn("$.generator.seed", paths)

    def test_byte_evidence_is_bounded(self) -> None:
        report = diagnostic.compare_fixtures(
            encode_fixture(_payload("00000000")), encode_fixture(_payload("ffffffff")))
        self.assertLessEqual(len(report["byte_differences"]["first"]), diagnostic.BYTE_DIFF_LIMIT)

    def test_cpu_info_reads_only_first_processor_and_allowlist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "cpuinfo"
            source.write_text("processor: 0\nmodel name: First\nflags: a b\nsecret: hidden\n\n"
                              "processor: 1\nmodel name: Second\n", encoding="utf-8")
            self.assertEqual(diagnostic._cpu_info(source),
                             {"model_name": "First", "flags": ["a", "b"]})


if __name__ == "__main__":
    unittest.main()
