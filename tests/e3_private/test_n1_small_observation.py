"""Only generated synthetic metadata; no OCI or Docker access."""

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent))
import e3_n1_small_observation as observation


class ObservationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.destination = Path(temporary.name) / "attempt"

    def test_completed_fixed_tree_is_diagnostic_only(self):
        result = observation.observe_synthetic(self.destination)
        self.assertEqual(result["status"], "HOLD")
        self.assertEqual(result["resource_proof"], "HOLD")
        self.assertTrue(result["nominal_expected_actor_caps_met"])
        self.assertEqual(result["expected_actors"], ["observer", "worker", "direct-child"])
        peaks = result["peak_rss_bytes"]
        self.assertTrue(all(type(value) is int and value > 0 for value in peaks.values()))
        self.assertEqual(result["expected_actor_peak_sum_bytes"], sum(peaks.values()))
        self.assertLessEqual(result["expected_actor_elapsed_ns"], result["elapsed_cap_ns"])
        self.assertLessEqual(result["expected_actor_peak_sum_bytes"], result["rss_cap_bytes"])
        self.assertEqual(result["preservation"]["status"], "HOLD")
        self.assertEqual((self.destination / "capture" / "stdout.bin").read_bytes()[:1], b"{")
        raw = (self.destination / "result.json").read_bytes()
        record = json.loads(raw)
        checksum = record.pop("checksum_sha256")
        self.assertEqual(checksum, observation.capture.sha(observation.capture.canonical(record)))
        self.assertEqual(raw, observation.capture.canonical({**record, "checksum_sha256": checksum}))

    def test_bad_child_outputs_and_exit_never_prove_caps(self):
        for mode in ("malformed", "failed", "overflow"):
            with self.subTest(mode=mode):
                self.destination = self.destination.parent / mode
                result = observation.observe_synthetic(self.destination, mode=mode)
                self.assertEqual(result["resource_proof"], "HOLD")
                self.assertFalse(result["nominal_expected_actor_caps_met"])
                self.assertIsNone(result["expected_actor_peak_sum_bytes"])
                self.assertEqual(result["preservation"]["status"], "HOLD")
                self.assertTrue((self.destination / "capture" / "stdout.bin").exists())

    def test_tiny_rss_cap_rejects_even_completed_tree(self):
        result = observation.observe_synthetic(self.destination, rss_cap=1)
        self.assertEqual(result["resource_proof"], "HOLD")
        self.assertFalse(result["nominal_expected_actor_caps_met"])
        self.assertIsNone(result["expected_actor_peak_sum_bytes"])

    def test_external_deadline_rejects_and_retains_intent(self):
        result = observation.observe_synthetic(self.destination,
                                               mode="observer-delay", seconds=0.3)
        self.assertEqual(result["resource_proof"], "HOLD")
        self.assertFalse(result["nominal_expected_actor_caps_met"])
        self.assertTrue((self.destination / "intent.json").exists())
        self.assertTrue((self.destination / "observer.stdout.bin").exists())
        self.assertFalse((self.destination / "worker.stdout.bin").exists())

    def test_child_timeout_retains_partial_capture(self):
        result = observation.observe_synthetic(self.destination, mode="timeout")
        self.assertEqual(result["resource_proof"], "HOLD")
        self.assertFalse(result["nominal_expected_actor_caps_met"])
        self.assertEqual((self.destination / "capture" / "stdout.bin").read_bytes(),
                         b"partial")

    def test_occupied_destination_never_launches(self):
        self.destination.mkdir()
        with self.assertRaises(FileExistsError):
            observation.observe_synthetic(self.destination)
        self.assertEqual(list(self.destination.iterdir()), [])

    def test_result_seal_failure_retains_partial_evidence(self):
        with mock.patch.object(observation.capture.os, "link", side_effect=OSError("seal-fault")):
            with self.assertRaisesRegex(OSError, "seal-fault"):
                observation.observe_synthetic(self.destination)
        self.assertTrue((self.destination / "intent.json").exists())
        self.assertTrue((self.destination / "capture" / "stdout.bin").exists())
        self.assertTrue((self.destination / "result.json.partial").exists())
        self.assertFalse((self.destination / "result.json").exists())

    def test_missing_rusage_cannot_be_a_positive_observation(self):
        real = observation._run

        def without_peak(*args, **kwargs):
            report = real(*args, **kwargs)
            report["peak_rss_bytes"] = None
            return report

        with mock.patch.object(observation, "_run", side_effect=without_peak):
            result = observation.observe_synthetic(self.destination)
        self.assertEqual(result["resource_proof"], "HOLD")
        self.assertFalse(result["nominal_expected_actor_caps_met"])
        self.assertIsNone(result["expected_actor_peak_sum_bytes"])

    def test_input_scope_rejects_unknown_modes_or_relaxed_caps(self):
        for options in ({"mode": "arbitrary"}, {"rss_cap": observation.RSS_CAP + 1},
                        {"seconds": 31}):
            with self.subTest(options=options), self.assertRaises(observation.Hold):
                observation.observe_synthetic(self.destination, **options)
        self.assertFalse(self.destination.exists())

    def test_synthetic_destination_requires_tmp_root(self):
        with self.assertRaisesRegex(observation.Hold, "synthetic-tmp-only"):
            observation.observe_synthetic(Path("/var/tmp/e3-n1-would-write"))

    def test_reaped_leader_with_escaped_pipe_holder_is_never_signaled(self):
        program = ("import os,sys,time\n"
                   "holder = os.fork()\n"
                   "if holder == 0:\n"
                   "    os.setsid()\n"
                   "    while not os.path.exists(sys.argv[1]):\n"
                   "        time.sleep(0.01)\n"
                   "    open(sys.argv[3], 'w').close()\n"
                   "    os._exit(0)\n"
                   "if sys.argv[2] == 'overflow':\n"
                   "    os.write(1, b'x' * 65)\n"
                   "os._exit(0)\n")
        self.destination.mkdir()
        directory = os.open(self.destination, os.O_RDONLY | os.O_DIRECTORY)
        self.addCleanup(os.close, directory)
        real_popen = observation.subprocess.Popen

        def after_leader_exit(*args, **kwargs):
            process = real_popen(*args, **kwargs)
            os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOWAIT)
            return process

        for case, expected in (("deadline", "deadline"), ("overflow", "stdout-cap")):
            with self.subTest(case=case):
                release = self.destination / (case + ".release")
                done = self.destination / (case + ".done")
                command = [sys.executable, "-c", program, str(release), case,
                           str(done)]
                deadline = time.monotonic_ns() + (200_000_000 if case == "deadline"
                                                  else 1_000_000_000)
                try:
                    with (mock.patch.object(observation.subprocess, "Popen",
                                            side_effect=after_leader_exit),
                          mock.patch.object(observation, "_kill") as kill,
                          mock.patch.object(observation, "PIPE_CAP", 64)):
                        result = observation._run(command, directory, case, deadline)
                    self.assertTrue(result["reaped"])
                    self.assertEqual(result["reason"], expected)
                    kill.assert_not_called()
                    if case == "overflow":
                        self.assertEqual(result["stdout"], b"x" * 64)
                finally:
                    release.touch()
                    stop = time.monotonic() + 1
                    while not done.exists() and time.monotonic() < stop:
                        time.sleep(0.01)
                    self.assertTrue(done.exists())


if __name__ == "__main__":
    unittest.main()
