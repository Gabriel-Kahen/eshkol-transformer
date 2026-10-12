"""Generated-only finite attempt ordering and fail-closed cuts."""

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent))
import e3_n1_generated_attempt as attempt


class GeneratedAttemptTests(unittest.TestCase):
    def assert_hold(self, record):
        self.assertEqual(record["schema"], attempt.SCHEMA)
        self.assertEqual(record["status"], "HOLD")
        self.assertEqual(record["resource_proof"], "HOLD")
        self.assertEqual(record["protected_pre_read_custody"], "HOLD")
        self.assertFalse(record["source_opened"])
        self.assertGreater(record["elapsed_boottime_ns"], 0)
        self.assertGreater(record["setup_peak_rss_bytes"], 0)

    def test_generated_order_and_all_finite_role_peaks(self):
        record = attempt.probe()
        self.assert_hold(record)
        self.assertEqual(record["reason"], "generated-order-observed")
        self.assertEqual(record["phases"], ["HOLD", "RECEIPT", "SEAL"])
        self.assertEqual(set(record["roles"]), {"broker", "reader", "sealer"})
        self.assertEqual(record["unresolved"], [])
        self.assertEqual(record["uid_transition"], "STUB-SAME-UID")
        self.assertEqual(record["protected_channel"], "STUB-SAME-UID")
        self.assertEqual(record["private_cgroup"], "HOLD")
        self.assertEqual(record["v3_signatures"], "HOLD")
        for role in record["roles"].values():
            self.assertEqual(role["exit_code"], 0)
            self.assertGreater(role["peak_rss_bytes"], 0)
        self.assertEqual(record["observed_peak_sum_bytes"],
                         record["setup_peak_rss_bytes"] +
                         sum(item["peak_rss_bytes"] for item in record["roles"].values()))

    def test_fault_cuts_preserve_hold_and_reap_spawned_roles(self):
        expected = {
            "before_hold": [], "after_hold": ["HOLD"],
            "after_receipt": ["HOLD", "RECEIPT"],
            "after_seal": ["HOLD", "RECEIPT", "SEAL"],
            "hold_sync": [], "receipt_sync": ["HOLD"],
            "seal_sync": ["HOLD", "RECEIPT"],
            "timeout": ["HOLD"], "inherited_fd": [],
            "cgroup_migration": ["HOLD", "RECEIPT"],
        }
        for fault, phases in expected.items():
            with self.subTest(fault=fault):
                record = attempt.probe(fault)
                self.assert_hold(record)
                self.assertEqual(record["phases"], phases)
                self.assertNotEqual(record["reason"], "generated-order-observed")
                self.assertEqual(record["unresolved"], [])
                self.assertTrue(record["roles"])
                self.assertEqual(record["observed_peak_sum_bytes"],
                                 record["setup_peak_rss_bytes"] +
                                 sum(item["peak_rss_bytes"] for item in record["roles"].values()))
        for fault in ("before_hold", "hold_sync", "receipt_sync", "inherited_fd"):
            self.assertEqual(attempt.probe(fault)["roles"]["broker"]["exit_code"], 9)
        self.assertEqual(attempt.probe("seal_sync")["roles"]["sealer"]["exit_code"], 9)
        self.assertLess(attempt.probe("timeout")["elapsed_boottime_ns"], 3_000_000_000)

    def test_unknown_fault_or_malformed_setup_report_cannot_succeed(self):
        with self.assertRaisesRegex(ValueError, "unknown-generated-fault"):
            attempt.probe("real-source")
        with mock.patch.object(attempt, "_setup", return_value={"roles": {"broker": {"peak_rss_bytes": "bad"}}}):
            record = attempt.probe()
        self.assert_hold(record)
        self.assertEqual(record["reason"], "setup-report-unverified")
        self.assertNotIn("observed_peak_sum_bytes", record)


if __name__ == "__main__":
    unittest.main()
