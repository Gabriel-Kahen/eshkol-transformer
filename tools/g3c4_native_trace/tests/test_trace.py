#!/usr/bin/env python3
"""Host-only canaries for the unsealed native trace prototype."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / "scripts"))
import g3c4_native_trace_receipt as receipt_trace
CC = ["cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-Wpedantic", "-O2"]
ENV = {"PATH": "/usr/bin:/bin", "G3C4_CANARY": "fixed"}


def read_events(path):
    lines = path.read_text(encoding="ascii").splitlines()
    if not lines or lines[0] != "g3c4-native-exec-v2":
        raise AssertionError("missing exact event log header")
    events = [line.split("\t") for line in lines[1:]]
    if any(not event or event[0] not in {"spawn", "exec", "exit", "merge", "complete"}
           for event in events):
        raise AssertionError("unknown native event")
    return events


class NativeTraceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build = tempfile.TemporaryDirectory(prefix="g3c4-native-trace-build-")
        cls.trace = Path(cls.build.name) / "trace"
        cls.canary = Path(cls.build.name) / "canary"
        subprocess.run(CC + [str(ROOT / "trace.c"), str(ROOT / "sha256.c"),
                             "-o", str(cls.trace)], check=True)
        subprocess.run(CC + ["-pthread", str(ROOT / "tests/canary.c"),
                             "-o", str(cls.canary)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.build.cleanup()

    def run_trace(self, command, *options):
        with tempfile.TemporaryDirectory(prefix="g3c4-native-trace-log-") as place:
            log = Path(place) / "events.log"
            result = subprocess.run([str(self.trace), "--log", str(log), *options,
                                     "--", *map(str, command)], env=ENV,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    timeout=30)
            events = read_events(log)
            return result, events

    def test_fast_fork_exec_children_are_all_observed(self):
        result, events = self.run_trace([self.canary, "fast"])
        self.assertEqual(result.returncode, 0, result.stderr)
        spawns = [event for event in events if event[0] == "spawn"]
        execs = [event for event in events if event[0] == "exec"]
        exits = [event for event in events if event[0] == "exit"]
        self.assertEqual(len(spawns), 32)
        self.assertEqual(len(execs), 33)
        self.assertEqual(len(exits), 33)
        spawned = {event[3] for event in spawns}
        self.assertEqual(spawned, {event[2] for event in execs[1:]})
        self.assertEqual(spawned, {event[2] for event in exits if event[2] in spawned})
        self.assertEqual({bytes.fromhex(event[7]) for event in execs[1:]},
                         {b"/bin/true\0"})
        expected_env = hashlib.sha256(b"PATH=/usr/bin:/bin\0G3C4_CANARY=fixed\0").hexdigest()
        self.assertEqual({event[10] for event in execs}, {expected_env})
        self.assertEqual({event[4] for event in exits}, {"0"})
        self.assertEqual([event[3:] for event in events if event[0] == "complete"],
                         [["0", "32", "33", "33", "0"]])
        self.assertEqual(events[-1][0], "complete")

    def test_spawn_and_system_descendants_are_observed(self):
        for mode in ("spawn", "system"):
            with self.subTest(mode=mode):
                result, events = self.run_trace([self.canary, mode])
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue(any(event[0] == "spawn" for event in events))
                self.assertTrue(any(event[0] == "exec" and
                                    bytes.fromhex(event[7]) == b"/bin/true\0"
                                    for event in events))
                self.assertEqual(events[-1][0], "complete")

    def test_thread_clone_is_observed_until_exit(self):
        result, events = self.run_trace([self.canary, "thread"])
        self.assertEqual(result.returncode, 0, result.stderr)
        spawns = [event for event in events if event[0] == "spawn"]
        self.assertEqual(len(spawns), 1)
        self.assertEqual(spawns[0][4], "clone")
        self.assertEqual({event[2] for event in events if event[0] == "exit"},
                         {spawns[0][2], spawns[0][3]})
        self.assertEqual(events[-1][0], "complete")

    def test_nonleader_thread_exec_reassigns_tid_without_leaking_task(self):
        result, events = self.run_trace([self.canary, "threadexec"])
        self.assertEqual(result.returncode, 0, result.stderr)
        merges = [event for event in events if event[0] == "merge"]
        spawns = [event for event in events if event[0] == "spawn"]
        self.assertEqual(len(merges), 1)
        self.assertEqual(len(spawns), 1)
        self.assertEqual(merges[0][2:], [spawns[0][3], spawns[0][2]])
        self.assertEqual(events[-1][3:], ["0", "1", "2", "1", "1"])

    def test_fault_kills_stopped_tracee_before_side_effect(self):
        with tempfile.TemporaryDirectory(prefix="g3c4-native-trace-fault-") as place:
            marker = Path(place) / "unwanted"
            result, events = self.run_trace(
                ["/bin/sh", "-c", f"touch {marker}"],
                "--fault-after-exec", "1")
            self.assertEqual(result.returncode, 86)
            self.assertIn(b"injected supervisor failure", result.stderr)
            self.assertFalse(marker.exists())
            self.assertEqual(len([event for event in events if event[0] == "exec"]), 1)
            self.assertFalse(any(event[0] == "complete" for event in events))

    def test_fault_kills_entire_traced_tree_before_child_side_effect(self):
        with tempfile.TemporaryDirectory(prefix="g3c4-native-trace-subtree-") as place:
            marker = Path(place) / "unwanted"
            result, events = self.run_trace(
                [self.canary, "faultchild", marker], "--fault-after-exec", "2")
            self.assertEqual(result.returncode, 86)
            self.assertIn(b"injected supervisor failure", result.stderr)
            self.assertFalse(marker.exists())
            self.assertEqual(len([event for event in events if event[0] == "exec"]), 2)
            self.assertEqual(len([event for event in events if event[0] == "spawn"]), 1)
            self.assertFalse(any(event[0] == "complete" for event in events))

    def test_existing_log_path_fails_before_target_starts(self):
        with tempfile.TemporaryDirectory(prefix="g3c4-native-trace-collision-") as place:
            marker = Path(place) / "unwanted"
            log = Path(place) / "events.log"
            log.write_text("existing", encoding="ascii")
            result = subprocess.run([str(self.trace), "--log", str(log), "--",
                                     "/bin/sh", "-c", f"touch {marker}"],
                                    env=ENV, capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 86)
            self.assertEqual(log.read_text(encoding="ascii"), "existing")
            self.assertFalse(marker.exists())

    def test_nonzero_root_has_no_success_footer(self):
        result, events = self.run_trace(["/bin/false"])
        self.assertEqual(result.returncode, 1)
        self.assertEqual(len([event for event in events if event[0] == "exec"]), 1)
        self.assertEqual(len([event for event in events if event[0] == "exit"]), 1)
        self.assertFalse(any(event[0] == "complete" for event in events))

    def test_untraced_clone_attempt_holds_even_if_target_ignores_it(self):
        with tempfile.TemporaryDirectory(prefix="g3c4-native-trace-untraced-") as place:
            marker = Path(place) / "unwanted"
            result, events = self.run_trace([self.canary, "untraced", marker])
            self.assertEqual(result.returncode, 86)
            self.assertIn(b"forbidden clone route", result.stderr)
            self.assertFalse(marker.exists())
            self.assertFalse(any(event[0] == "complete" for event in events))

    def test_real_command_receipt_matches_native_exec_and_tampering_holds(self):
        with tempfile.TemporaryDirectory(prefix="g3c4-native-receipt-") as place:
            root = Path(place)
            log = root / "events.log"
            helper = ROOT.parents[1] / "scripts/g3c4_proof_command.py"
            result = subprocess.run(
                [str(self.trace), "--log", str(log), "--", sys.executable,
                 str(helper), str(root), "--", "/bin/true"],
                env=ENV, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            command = next(root.glob("command.*/receipt.json"))
            meta = json.loads(command.read_text())
            rows = [(command.parent.joinpath("argv.nul").read_bytes(),
                     meta["start_monotonic_ns"], meta["end_monotonic_ns"])]
            summary = receipt_trace.verify(log, rows)
            self.assertEqual(summary["matched_receipts"], 1)
            self.assertEqual(summary["unmatched_native_execs"], 1)
            original = log.read_text()
            lines = original.splitlines()
            footer = lines[-1].split("\t")
            footer[5] = str(int(footer[5]) + 1)
            bad_count = "\n".join(lines[:-1] + ["\t".join(footer)]) + "\n"
            exec_line = next(i for i, line in enumerate(lines) if line.startswith("exec\t") and i > 1)
            bad_lineage = lines.copy()
            fields = bad_lineage[exec_line].split("\t")
            fields[3] = "99999999"
            bad_lineage[exec_line] = "\t".join(fields)
            for changed, reason in (
                (original.replace("complete\t", "missing\t"), "unknown event"),
                ("\n".join(lines[:-1]) + "\n", "completion"),
                (bad_count, "event count mismatch"),
                ("\n".join(bad_lineage) + "\n", "invalid exec lineage"),
            ):
                with self.subTest(reason=reason):
                    log.write_text(changed)
                    with self.assertRaisesRegex(ValueError, reason):
                        receipt_trace.verify(log, rows)
                    log.write_text(original)
            wrong = [(b"/bin/false\0", rows[0][1], rows[0][2])]
            with self.assertRaisesRegex(ValueError, "no traced exec"):
                receipt_trace.verify(log, wrong)
            late = [(rows[0][0], rows[0][2] + 1, rows[0][2] + 2)]
            with self.assertRaisesRegex(ValueError, "no traced exec"):
                receipt_trace.verify(log, late)


if __name__ == "__main__":
    unittest.main()
