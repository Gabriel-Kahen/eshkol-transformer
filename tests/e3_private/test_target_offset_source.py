"""Offline D1/D2 target-offset reachability and canonical-entry boundaries."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from tests.e3_reference.target_offset import CASES, materialize, validate_materialized


ROOT = Path(__file__).resolve().parents[2]


class TargetOffsetSourceTest(unittest.TestCase):
    def test_fresh_generated_resources_are_deterministic_and_checked(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e3-target-offset-") as temporary:
            roots = [Path(temporary) / label for label in ("a", "b")]
            payloads = [materialize(root) for root in roots]
            for root, payload in zip(roots, payloads):
                validate_materialized(root, payload)
                self.assertFalse((root / "reference.json").exists())
                with self.assertRaises(ValueError):
                    materialize(root)
            self.assertEqual(payloads[0], payloads[1])
            for name, _, _ in CASES:
                for filename in ("manifest.etm", "shard-0000000000000000.ets"):
                    self.assertEqual((roots[0] / name / filename).read_bytes(),
                                     (roots[1] / name / filename).read_bytes())
            rows = [case["batches"][0] for case in payloads[0]["cases"]]
            self.assertEqual([row["inputs"] for row in rows], [[3, 197]] * 2)
            self.assertEqual([row["mask"] for row in rows], [[True, True]] * 2)
            self.assertEqual([row["targets"] for row in rows],
                             [[197, 0], [197, 7]])

    def test_corrupt_shard_and_manifest_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e3-target-offset-") as temporary:
            root = Path(temporary) / "case"
            payload = materialize(root)
            changed = deepcopy(payload)
            changed["cases"][0]["batches"][0]["targets"][1] = 7
            with self.assertRaises(ValueError):
                validate_materialized(root, changed)
            shard = root / "target-seven" / "shard-0000000000000000.ets"
            data = bytearray(shard.read_bytes())
            data[-1] ^= 1
            shard.write_bytes(data)
            with self.assertRaises(ValueError):
                validate_materialized(root, payload)
            manifest_root = Path(temporary) / "manifest-case"
            manifest_payload = materialize(manifest_root)
            manifest = manifest_root / "target-zero" / "manifest.etm"
            data = bytearray(manifest.read_bytes())
            data[-1] ^= 1
            manifest.write_bytes(data)
            with self.assertRaises(ValueError):
                validate_materialized(manifest_root, manifest_payload)

    def test_runtime_keeps_the_canonical_two_field_scenario(self) -> None:
        runtime = (ROOT / "tests/e3_private/target_offset_runtime.esk").read_text()
        driver = (ROOT / "tests/e3_private/driver.esk").read_text()
        self.assertIn(":real et_e3_test_run_v1", runtime)
        self.assertIn("(e3-private-test-run (vector (vector #t 0))", runtime)
        self.assertIn('"target-zero"', runtime)
        self.assertIn('"target-seven"', runtime)
        self.assertNotIn("e3-zero-run", runtime)
        self.assertNotIn("e3-zero-control-run", runtime)
        self.assertIn('"expected #(packing? skipped-batches)"', driver)
        self.assertIn("(g3t-checked-m3t-initializer-state 1729)", driver)


if __name__ == "__main__":
    unittest.main()
