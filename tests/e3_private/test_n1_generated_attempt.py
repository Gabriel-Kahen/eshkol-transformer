"""Generated-only finite attempt ordering and fail-closed cuts."""

import json
import os
import stat
import sys
import tempfile
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
        self.assertEqual(record["unexpected_actors"], [])
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
            "detached_actor": ["HOLD", "RECEIPT", "SEAL"],
        }
        for fault, phases in expected.items():
            with self.subTest(fault=fault):
                record = attempt.probe(fault)
                self.assert_hold(record)
                self.assertEqual(record["phases"], phases)
                self.assertNotEqual(record["reason"], "generated-order-observed")
                self.assertEqual(record["unresolved"], [])
                self.assertTrue(record["roles"])
                self.assertNotIn("observed_peak_sum_bytes", record)
        for fault in ("before_hold", "hold_sync", "receipt_sync", "inherited_fd"):
            self.assertEqual(attempt.probe(fault)["roles"]["broker"]["exit_code"], 9)
        self.assertEqual(attempt.probe("seal_sync")["roles"]["sealer"]["exit_code"], 9)
        self.assertLess(attempt.probe("timeout")["elapsed_boottime_ns"], 3_000_000_000)

    def test_detached_pipe_free_actor_is_reaped_and_blocks_peak_sum(self):
        record = attempt.probe("detached_actor")
        self.assert_hold(record)
        self.assertEqual(record["reason"], "unexpected-or-unmeasured-actor")
        self.assertEqual(record["phases"], ["HOLD", "RECEIPT", "SEAL"])
        self.assertEqual(record["unresolved"], [])
        self.assertEqual(len(record["unexpected_actors"]), 1)
        self.assertEqual(record["unexpected_actors"][0]["exit_code"], 0)
        self.assertGreater(record["unexpected_actors"][0]["peak_rss_bytes"], 0)
        self.assertNotIn("observed_peak_sum_bytes", record)
        outer = attempt._finite_observer_diagnostic("detached_actor")
        self.assertEqual(outer["reason"], "observer-report-unverified")
        self.assertNotIn("declared_plus_caller_peak_sum_bytes", outer)

    def test_subreaper_or_unlisted_child_uncertainty_blocks_peak_sum(self):
        with mock.patch.object(attempt, "_enable_subreaper",
                               side_effect=attempt.Hold("subreaper-unavailable")):
            unavailable = attempt.probe()
        self.assert_hold(unavailable)
        self.assertEqual(unavailable["reason"], "setup-or-report-closure-unverified")
        self.assertNotIn("observed_peak_sum_bytes", unavailable)

        with mock.patch.object(attempt, "_census_unlisted_children",
                               return_value=([], True)):
            live = attempt.probe()
        self.assert_hold(live)
        self.assertEqual(live["reason"], "actor-closure-unverified")
        self.assertEqual(live["unresolved"], ["unlisted-live-child"])
        self.assertNotIn("observed_peak_sum_bytes", live)

    def test_unknown_fault_or_malformed_setup_report_cannot_succeed(self):
        with self.assertRaisesRegex(ValueError, "unknown-generated-fault"):
            attempt.probe("real-source")
        with mock.patch.object(attempt, "_setup", return_value={"roles": {"broker": {"peak_rss_bytes": "bad"}}}):
            record = attempt.probe()
        self.assert_hold(record)
        self.assertEqual(record["reason"], "setup-report-unverified")
        self.assertNotIn("observed_peak_sum_bytes", record)

    def test_incomplete_or_failed_report_cannot_emit_peak_sum(self):
        complete = {"reason": "generated-order-observed",
                    "phases": ["HOLD", "RECEIPT", "SEAL"],
                    "roles": {role: {"exit_code": 0, "peak_rss_bytes": 1024}
                              for role in ("broker", "reader", "sealer")},
                    "unresolved": [], "uid_transition": "STUB-SAME-UID",
                    "unexpected_actors": [],
                    "protected_channel": "STUB-SAME-UID", "private_cgroup": "HOLD",
                    "v3_signatures": "HOLD"}
        mutants = {
            "missing-role": {**complete, "roles": {"broker": complete["roles"]["broker"],
                                                    "reader": complete["roles"]["reader"]}},
            "failed-role": {**complete, "roles": {**complete["roles"],
                                                   "sealer": {"exit_code": 9,
                                                              "peak_rss_bytes": 1024}}},
            "unresolved-role": {**complete, "unresolved": ["sealer"]},
            "missing-phase": {**complete, "phases": ["HOLD", "RECEIPT"]},
            "duplicate-phase": {**complete, "phases": ["HOLD", "RECEIPT", "SEAL", "SEAL"]},
            "unexpected-actor": {**complete, "unexpected_actors": [
                {"exit_code": 0, "peak_rss_bytes": 1024}]},
        }
        for name, body in mutants.items():
            with self.subTest(name=name), mock.patch.object(attempt, "_setup", return_value=body):
                record = attempt.probe()
            self.assert_hold(record)
            self.assertEqual(record["reason"], "finite-graph-unverified")
            self.assertNotIn("observed_peak_sum_bytes", record)

    def test_duplicate_raw_report_keys_cannot_emit_peak_sum(self):
        body = {"reason": "generated-order-observed",
                "phases": ["HOLD", "RECEIPT", "SEAL"],
                "roles": {role: {"exit_code": 0, "peak_rss_bytes": 1}
                          for role in ("broker", "reader", "sealer")},
                "unresolved": [], "uid_transition": "STUB-SAME-UID",
                "unexpected_actors": [],
                "protected_channel": "STUB-SAME-UID", "private_cgroup": "HOLD",
                "v3_signatures": "HOLD"}
        raw = json.dumps(body)
        broker = '"broker": {"exit_code": 0, "peak_rss_bytes": 1}'
        mutants = {
            "duplicate-broker": raw.replace(
                broker, '"broker": {"exit_code": 0, "peak_rss_bytes": 999999999}, '
                + broker, 1),
            "duplicate-peak": raw.replace(
                '"peak_rss_bytes": 1',
                '"peak_rss_bytes": 999999999, "peak_rss_bytes": 1', 1),
            "duplicate-top-level": raw.replace(
                '"reason": "generated-order-observed"',
                '"reason": "other", "reason": "generated-order-observed"', 1),
        }
        for name, report in mutants.items():
            self.assertNotEqual(report, raw)
            with self.subTest(name=name), mock.patch.object(attempt, "_setup", return_value={}), \
                    mock.patch.object(attempt.json, "dumps", return_value=report):
                record = attempt.probe()
            self.assert_hold(record)
            self.assertEqual(record["reason"], "setup-report-unverified")
            self.assertNotIn("observed_peak_sum_bytes", record)

    def test_elapsed_includes_observer_report_validation(self):
        real_now = attempt._now
        real_loads = json.loads
        for malformed in (False, True):
            with self.subTest(malformed=malformed):
                clock_advance = 0

                def clock():
                    return real_now() + clock_advance

                def parse_report(*args, **kwargs):
                    nonlocal clock_advance
                    clock_advance = 5_000_000_000
                    if malformed:
                        raise ValueError("injected-report-parse-failure")
                    return real_loads(*args, **kwargs)

                with mock.patch.object(attempt, "_now", side_effect=clock), \
                        mock.patch.object(attempt.json, "loads", side_effect=parse_report):
                    record = attempt.probe()
                self.assert_hold(record)
                self.assertGreaterEqual(record["elapsed_boottime_ns"], 5_000_000_000)
                self.assertEqual(record["reason"],
                                 "setup-report-unverified" if malformed else
                                 "generated-order-observed")

    def test_finite_observer_final_peak_includes_late_allocation(self):
        normal = attempt._finite_observer_diagnostic()
        self.assertEqual(normal["status"], "HOLD")
        self.assertEqual(normal["resource_proof"], "HOLD")
        self.assertEqual(normal["protected_pre_read_custody"], "HOLD")
        self.assertEqual(normal["reason"], "declared-peak-arithmetic-observed")
        self.assertGreater(normal["observer_peak_rss_bytes"], 0)
        self.assertGreaterEqual(normal["finite_observer_elapsed_boottime_ns"],
                                normal["inner_observation"]["elapsed_boottime_ns"])
        self.assertEqual(normal["declared_finite_peak_sum_bytes"],
                         normal["inner_observation"]["observed_peak_sum_bytes"] +
                         normal["observer_peak_rss_bytes"])
        self.assertEqual(normal["schema"], attempt.FINITE_SCHEMA)
        self.assertGreater(normal["caller_peak_rss_bytes_at_sample"], 0)
        self.assertEqual(normal["declared_plus_caller_peak_sum_bytes"],
                         normal["declared_finite_peak_sum_bytes"] +
                         normal["caller_peak_rss_bytes_at_sample"])

        real_emit = attempt._emit_observer_report

        def emit_then_allocate(fd, record):
            real_emit(fd, record)
            pages = bytearray(16 * 1024 * 1024)
            for offset in range(0, len(pages), 4096):
                pages[offset] = 1

        with mock.patch.object(attempt, "_emit_observer_report", side_effect=emit_then_allocate):
            late = attempt._finite_observer_diagnostic()
        self.assertEqual(late["reason"], "declared-peak-arithmetic-observed")
        self.assertGreater(late["observer_peak_rss_bytes"],
                           normal["observer_peak_rss_bytes"] + 8 * 1024 * 1024)
        self.assertEqual(late["declared_finite_peak_sum_bytes"],
                         late["inner_observation"]["observed_peak_sum_bytes"] +
                         late["observer_peak_rss_bytes"])

    def test_caller_peak_is_sampled_after_report_validation(self):
        validated = False
        real_complete = attempt._complete_inner_peak

        def validate(record):
            nonlocal validated
            subtotal = real_complete(record)
            validated = True
            return subtotal

        def sample():
            self.assertTrue(validated)
            return 12_345_678

        with mock.patch.object(attempt, "_complete_inner_peak", side_effect=validate), \
                mock.patch.object(attempt, "_caller_peak_rss_bytes", side_effect=sample):
            record = attempt._finite_observer_diagnostic()
        self.assertEqual(record["reason"], "declared-peak-arithmetic-observed")
        self.assertEqual(record["caller_peak_rss_bytes_at_sample"], 12_345_678)
        self.assertEqual(record["declared_plus_caller_peak_sum_bytes"],
                         record["declared_finite_peak_sum_bytes"] + 12_345_678)

    def test_unavailable_caller_peak_omits_combined_sum(self):
        for failure in (OSError("unavailable"), ValueError("bad sample")):
            with self.subTest(failure=failure), \
                    mock.patch.object(attempt, "_caller_peak_rss_bytes", side_effect=failure):
                record = attempt._finite_observer_diagnostic()
            self.assertEqual(record["reason"], "caller-peak-unverified")
            self.assertIn("declared_finite_peak_sum_bytes", record)
            self.assertNotIn("caller_peak_rss_bytes_at_sample", record)
            self.assertNotIn("declared_plus_caller_peak_sum_bytes", record)
            self.assertEqual(record["resource_proof"], "HOLD")
        for bad in (0, -1, 1.5, True):
            with self.subTest(bad=bad), \
                    mock.patch.object(attempt.resource, "getrusage",
                                      return_value=mock.Mock(ru_maxrss=bad)):
                with self.assertRaisesRegex(ValueError, "caller-peak-unverified"):
                    attempt._caller_peak_rss_bytes()

    def test_finite_observer_rejects_partial_and_malformed_reports(self):
        with mock.patch.object(attempt, "_caller_peak_rss_bytes") as sample:
            partial = attempt._finite_observer_diagnostic("after_hold")
        sample.assert_not_called()
        self.assertEqual(partial["reason"], "observer-report-unverified")
        self.assertNotIn("declared_finite_peak_sum_bytes", partial)
        self.assertNotIn("declared_plus_caller_peak_sum_bytes", partial)

        with mock.patch.object(attempt, "_emit_observer_report", return_value=None):
            empty = attempt._finite_observer_diagnostic()
        self.assertEqual(empty["reason"], "observer-report-unverified")
        self.assertNotIn("declared_finite_peak_sum_bytes", empty)

        real_emit = attempt._emit_observer_report

        def forge_elapsed(fd, record):
            record["elapsed_boottime_ns"] = 100_000_000_000
            real_emit(fd, record)

        with mock.patch.object(attempt, "_emit_observer_report", side_effect=forge_elapsed):
            inconsistent = attempt._finite_observer_diagnostic()
        self.assertEqual(inconsistent["reason"], "observer-report-unverified")
        self.assertNotIn("declared_finite_peak_sum_bytes", inconsistent)

        def forge_subtotal(fd, record):
            record["observed_peak_sum_bytes"] = 1
            real_emit(fd, record)

        with mock.patch.object(attempt, "_emit_observer_report", side_effect=forge_subtotal):
            inconsistent = attempt._finite_observer_diagnostic()
        self.assertEqual(inconsistent["reason"], "observer-report-unverified")
        self.assertNotIn("declared_finite_peak_sum_bytes", inconsistent)

        with mock.patch.object(attempt, "_emit_observer_report", side_effect=RuntimeError):
            failed = attempt._finite_observer_diagnostic()
        self.assertEqual(failed["reason"], "observer-or-report-closure-unverified")
        self.assertNotIn("declared_finite_peak_sum_bytes", failed)

    def test_finite_observer_report_requires_reaped_sender_identity(self):
        real_emit = attempt._emit_observer_report

        with self.assertRaisesRegex(ValueError, "observer-credentials-unverified"):
            attempt._observer_sender([])

        def descendant_emit(channel, record):
            pid = os.fork()
            if pid == 0:
                real_emit(channel, record)
                channel.close()
                os._exit(0)
            os.waitpid(pid, 0)

        with mock.patch.object(attempt, "_emit_observer_report", side_effect=descendant_emit):
            forged = attempt._finite_observer_diagnostic()
        self.assertEqual(forged["reason"], "observer-report-unverified")
        self.assertNotIn("declared_finite_peak_sum_bytes", forged)

        def duplicate_emit(channel, record):
            real_emit(channel, record)
            real_emit(channel, record)

        with mock.patch.object(attempt, "_emit_observer_report", side_effect=duplicate_emit):
            duplicate = attempt._finite_observer_diagnostic()
        self.assertEqual(duplicate["reason"], "observer-report-unverified")
        self.assertNotIn("declared_finite_peak_sum_bytes", duplicate)

        with mock.patch.object(attempt.socket, "socketpair", side_effect=OSError):
            unavailable = attempt._finite_observer_diagnostic()
        self.assertEqual(unavailable["reason"], "observer-credential-channel-unavailable")
        self.assertNotIn("declared_finite_peak_sum_bytes", unavailable)

    def test_generated_record_readback_rejects_same_bytes_path_replacement(self):
        body = {"schema": attempt.SCHEMA, "state": "HOLD"}
        raw = json.dumps(body, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        for kind, reason in (("replacement", "readback-identity-drift"),
                             ("symlink", "readback-open-unverified")):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory(
                    prefix="e3-n1-readback-test-") as directory:
                target = Path(directory) / "HOLD.json"
                replacement = Path(directory) / "replacement.json"
                replacement.write_bytes(raw)
                real_fsync = os.fsync
                swapped = False

                def swap_after_directory_sync(fd):
                    nonlocal swapped
                    real_fsync(fd)
                    if stat.S_ISDIR(os.fstat(fd).st_mode) and not swapped:
                        if kind == "replacement":
                            os.replace(replacement, target)
                        else:
                            target.unlink()
                            target.symlink_to(replacement)
                        swapped = True

                with mock.patch.object(attempt.os, "fsync", side_effect=swap_after_directory_sync):
                    with self.assertRaisesRegex(attempt.Hold, reason):
                        attempt._sync_json(target, body)
                self.assertTrue(swapped)
                self.assertEqual(target.read_bytes(), raw)

    def test_generated_record_readback_rejects_ancestor_swap_after_sync(self):
        body = {"schema": attempt.SCHEMA, "state": "HOLD"}
        for kind, reason in (("symlink", "parent-path-unverified"),
                             ("new-directory", "parent-identity-drift")):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory(
                    prefix="e3-n1-ancestor-test-") as directory:
                base = Path(directory)
                original = base / "root"
                moved = base / "moved"
                original.mkdir()
                target = original / "HOLD.json"
                real_fsync = os.fsync
                swapped = False

                def swap_after_directory_sync(fd):
                    nonlocal swapped
                    real_fsync(fd)
                    if stat.S_ISDIR(os.fstat(fd).st_mode) and not swapped:
                        original.rename(moved)
                        if kind == "symlink":
                            original.symlink_to(moved, target_is_directory=True)
                        else:
                            original.mkdir()
                        swapped = True

                with mock.patch.object(attempt.os, "fsync", side_effect=swap_after_directory_sync):
                    with self.assertRaisesRegex(attempt.Hold, reason):
                        attempt._sync_json(target, body)
                self.assertTrue(swapped)
                self.assertTrue((moved / "HOLD.json").is_file())

    def test_generated_record_rejects_preexisting_ancestor_symlink(self):
        body = {"schema": attempt.SCHEMA, "state": "HOLD"}
        with tempfile.TemporaryDirectory(prefix="e3-n1-ancestor-test-") as directory:
            base = Path(directory)
            actual = base / "actual"
            child = actual / "child"
            child.mkdir(parents=True)
            (base / "alias").symlink_to(actual, target_is_directory=True)
            with self.assertRaisesRegex(attempt.Hold, "parent-path-unverified"):
                attempt._sync_json(base / "alias" / "child" / "HOLD.json", body)
            self.assertFalse((child / "HOLD.json").exists())


if __name__ == "__main__":
    unittest.main()
