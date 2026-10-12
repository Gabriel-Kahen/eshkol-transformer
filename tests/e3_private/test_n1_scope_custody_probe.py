"""Generated role bytes only; no real OCI source or Docker access."""

import ctypes
import os
import sys
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent))
import e3_n1_scope_custody_probe as scope


class ScopeCustodyTests(unittest.TestCase):
    def test_direct_scope_placement_and_custody_hold(self):
        record = scope.probe()
        self.assertEqual(record["schema"], scope.SCHEMA)
        self.assertEqual(record["status"], "HOLD")
        self.assertEqual(record["resource_proof"], "HOLD")
        self.assertFalse(record["source_opened"])
        self.assertEqual(record["protected_pre_read_custody"], "HOLD")
        if record.get("scope_status") != "fixed-actor-direct-placement-observed":
            self.skipTest("delegated cgroup or clone3 unavailable: " + record["reason"])
        self.assertEqual(record["reason"], "no-independent-distinct-uid-authority")
        self.assertTrue(record["actor_scope_empty"])
        self.assertEqual(set(record["actors"]), set(scope.ROLES))
        self.assertFalse(record["broker_issued_receipt"])
        self.assertFalse(record["sealer_issued_seal"])
        self.assertEqual(len({v["direct_cgroup"] for v in record["actors"].values()}), 1)
        for actor in record["actors"].values():
            self.assertEqual(actor["exit_code"], 0)
            self.assertGreater(actor["peak_rss_bytes"], 0)
            self.assertTrue(actor["pipe_eof"])

    def test_missing_manager_fails_closed_before_any_role(self):
        with mock.patch.object(scope.subprocess, "run", side_effect=FileNotFoundError):
            record = scope.probe()
        self.assertEqual(record["status"], "HOLD")
        self.assertFalse(record["source_opened"])
        self.assertEqual(record["protected_pre_read_custody"], "HOLD")
        self.assertNotIn("actors", record)
        self.assertIn("delegated-scope-unavailable", record["reason"])

    def test_worker_rejects_unreviewed_scope_before_actor_clone(self):
        with mock.patch.object(scope, "_clone_role") as clone:
            with self.assertRaisesRegex(scope.Hold, "unexpected-scope"):
                scope._scope_worker("unreviewed.scope")
        clone.assert_not_called()

    def test_timeout_stops_only_its_unique_scope_and_reports_unverified_closure(self):
        calls = []
        def run(command, **kwargs):
            calls.append(command)
            if command[0] == "systemd-run":
                raise subprocess.TimeoutExpired(command, 4)
            return subprocess.CompletedProcess(command, 0)
        with mock.patch.object(scope.subprocess, "run", side_effect=run):
            record = scope.probe()
        self.assertEqual(record["reason"], "scope-timeout-closure-unverified")
        self.assertEqual(record["status"], "HOLD")
        self.assertFalse(record["source_opened"])
        self.assertNotIn("actors", record)
        self.assertEqual(calls[1], ["systemctl", "--user", "stop", record["unit"]])
        self.assertIn("--unit=" + record["unit"], calls[0])

    def test_cgroup_removal_failure_cannot_report_successful_actor_closure(self):
        with tempfile.TemporaryDirectory() as directory:
            current = Path(directory) / "test.scope"
            current.mkdir()
            with mock.patch.object(scope, "_cgroup_path", return_value=current), \
                 mock.patch.object(scope.closure, "_no_inherited_socket"), \
                 mock.patch.object(scope.closure, "_isolate"), \
                 mock.patch.object(scope, "_clone_role", return_value={}), \
                 mock.patch.object(scope, "_events_empty", return_value=True), \
                 mock.patch.object(Path, "rmdir", side_effect=PermissionError):
                with self.assertRaisesRegex(scope.Hold, "actor-cgroup-removal-uncertain"):
                    scope._scope_worker("test.scope")

    def test_post_clone_setup_error_kills_and_reaps_actor(self):
        spawned = []
        class FakeLibc:
            def __init__(self):
                self.syscall = lambda *args: self.invoke(*args)
            def invoke(self, _, args, __):
                pid = os.fork()
                if pid == 0:
                    time.sleep(3)
                    os._exit(0)
                spawned.append(pid)
                address = args._obj.pidfd
                ctypes.cast(address, ctypes.POINTER(ctypes.c_int)).contents.value = os.pidfd_open(pid)
                return pid
        with mock.patch.object(scope.ctypes, "CDLL", return_value=FakeLibc()), \
             mock.patch.object(scope.selectors, "DefaultSelector", side_effect=OSError("selector-fault")):
            with self.assertRaisesRegex(OSError, "selector-fault"):
                scope._clone_role(-1, Path("/unused"), "broker")
        self.assertEqual(len(spawned), 1)
        with self.assertRaises(ChildProcessError):
            os.waitpid(spawned[0], os.WNOHANG)

    def test_cgroup_open_failure_removes_created_child(self):
        with tempfile.TemporaryDirectory() as directory:
            current = Path(directory) / "test.scope"
            current.mkdir()
            with mock.patch.object(scope, "_cgroup_path", return_value=current), \
                 mock.patch.object(scope.closure, "_no_inherited_socket"), \
                 mock.patch.object(scope.closure, "_isolate"), \
                 mock.patch.object(scope.os, "open", side_effect=PermissionError("open-fault")):
                with self.assertRaisesRegex(PermissionError, "open-fault"):
                    scope._scope_worker("test.scope")
            self.assertFalse((current / "n1-generated-actors").exists())

if __name__ == "__main__":
    unittest.main()
