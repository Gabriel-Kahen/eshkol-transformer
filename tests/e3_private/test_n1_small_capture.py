"""Synthetic child output only; no OCI content-store or Docker access."""

import base64
import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent))
import e3_n1_small_capture as capture


def sha(raw):
    return "sha256:" + hashlib.sha256(raw).hexdigest()


class CaptureTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.destination = self.root / "one-use-attempt"
        self.request_sha = sha(b"synthetic request")
        manifest = {"mediaType": capture.small.MANIFEST_TYPE,
                    "digest": sha(b"synthetic manifest"), "size": 18,
                    "platform": {"os": "linux", "architecture": "amd64"}}
        raw_index = capture.canonical({"schemaVersion": 2,
                        "mediaType": capture.small.INDEX_TYPE,
                        "manifests": [manifest]})
        self.index_sha = sha(raw_index)
        index_path = self.root / "synthetic-index"
        index_path.write_bytes(raw_index)
        fd, ancestors = capture.small.open_nofollow(str(index_path))
        try:
            observed = {"file": capture.small.identity(os.fstat(fd)),
                        "ancestors": ancestors}
        finally:
            os.close(fd)
        body = {"schema": "e3-n1-small-metadata-v1", "status": "HOLD",
                "stage": "index", "candidate": {"name": capture.small.F32_NAME,
                                                "index_digest": self.index_sha},
                "request_sha256": self.request_sha, "raw_sha256": self.index_sha,
                "raw_size": len(raw_index),
                "raw_base64": base64.b64encode(raw_index).decode(),
                "source_path": str(index_path),
                "read_time_identity": observed,
                "claims": {"selected_manifest": manifest},
                "limits": {"max_bytes": 4096, "max_seconds": 30,
                           "max_rss_bytes": capture.RSS_CAP},
                "scope": "small-json-read-only;no-layer-or-custody-authority"}
        body["checksum_sha256"] = sha(capture.canonical(body))
        self.valid = capture.canonical(body)

    def child(self, raw, exit_code=0, sleep=0):
        program = ("import sys,time;"
                   "sys.stdout.buffer.write(bytes.fromhex(%r));"
                   "sys.stdout.flush();time.sleep(%r);sys.exit(%r)" %
                   (raw.hex(), sleep, exit_code))
        return [sys.executable, "-c", program]

    def run_capture(self, child, **limits):
        return capture.capture(child, self.destination, self.request_sha,
                               self.index_sha, **limits)

    def record(self):
        raw = (self.destination / "result.json").read_bytes()
        value = json.loads(raw)
        checksum = value.pop("checksum_sha256")
        self.assertEqual(checksum, sha(capture.canonical(value)))
        self.assertEqual(raw, capture.canonical({**value,
                                                "checksum_sha256": checksum}))
        return value

    def test_public_wrapper_holds_before_request_or_destination(self):
        result = subprocess.run([sys.executable, str(Path(capture.__file__)),
                                 str(self.root / "missing-request"),
                                 str(self.destination)], capture_output=True,
                                check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, b"")
        self.assertIn(b"capture-procedure-not-admitted", result.stderr)
        self.assertFalse(self.destination.exists())

    def test_future_public_gate_rejects_relative_request_before_child(self):
        with (mock.patch.object(capture, "CAPTURE_RUN_ENABLED", True),
              mock.patch.object(capture, "capture", side_effect=AssertionError("child")),
              mock.patch.object(sys, "argv", [str(Path(capture.__file__)),
                                              "relative-request", str(self.destination)]),
              contextlib.redirect_stderr(io.StringIO()) as stderr):
            self.assertEqual(capture.main(), 1)
        self.assertIn("absolute-path", stderr.getvalue())
        self.assertFalse(self.destination.exists())

    def test_valid_synthetic_output_is_still_hold(self):
        result = self.run_capture(self.child(self.valid))
        self.assertEqual(result["status"], "HOLD")
        self.assertEqual(result["outcome"], "captured-hold")
        self.assertIsNone(result["reason"])
        self.assertGreater(result["child_peak_rss_bytes"], 0)
        self.assertEqual((self.destination / "stdout.bin").read_bytes(), self.valid)
        self.assertEqual(self.record()["streams"]["stdout"]["sha256"], sha(self.valid))
        self.assertTrue((self.destination / "intent.json").exists())

    def test_stdout_overflow_retains_prefix_and_hold(self):
        result = self.run_capture(self.child(b"x" * 1024), stdout_cap=64)
        self.assertEqual(result["reason"], "stdout-cap")
        self.assertEqual((self.destination / "stdout.bin").read_bytes(), b"x" * 64)
        self.assertTrue(self.record()["streams"]["stdout"]["truncated"])

    def test_stderr_overflow_retains_prefix_and_hold(self):
        child = [sys.executable, "-c", "import sys;sys.stderr.buffer.write(b'x'*1024)"]
        result = self.run_capture(child, stderr_cap=64)
        self.assertEqual(result["reason"], "stderr-cap")
        self.assertEqual((self.destination / "stderr.bin").read_bytes(), b"x" * 64)
        self.assertTrue(self.record()["streams"]["stderr"]["truncated"])

    def test_timeout_retains_partial_stdout(self):
        result = self.run_capture(self.child(b"partial", sleep=2), seconds=0.2)
        self.assertEqual(result["reason"], "deadline")
        self.assertEqual((self.destination / "stdout.bin").read_bytes(), b"partial")
        self.assertEqual(self.record()["status"], "HOLD")

    def test_nonzero_child_and_malformed_output_hold(self):
        for raw, exit_code, expected in ((b"partial", 7, "child-exit"),
                                         (b"{bad", 0, "malformed-child-evidence")):
            with self.subTest(expected=expected):
                self.destination = self.root / expected
                result = self.run_capture(self.child(raw, exit_code))
                self.assertEqual(result["reason"], expected)
                self.assertEqual((self.destination / "stdout.bin").read_bytes(), raw)
                self.assertEqual(self.record()["outcome"], "failed-hold")

    def test_occupied_destination_rejects_before_child(self):
        self.destination.mkdir()
        marker = self.root / "child-ran"
        child = [sys.executable, "-c", "from pathlib import Path; Path(%r).touch()" % str(marker)]
        with self.assertRaises(FileExistsError):
            self.run_capture(child)
        self.assertFalse(marker.exists())
        self.assertEqual(list(self.destination.iterdir()), [])

    def test_result_seal_failure_retains_raw_attempt(self):
        with mock.patch.object(capture.os, "link", side_effect=OSError("seal-fault")):
            with self.assertRaisesRegex(OSError, "seal-fault"):
                self.run_capture(self.child(self.valid))
        self.assertTrue((self.destination / "intent.json").exists())
        self.assertEqual((self.destination / "stdout.bin").read_bytes(), self.valid)
        self.assertTrue((self.destination / "result.json.partial").exists())
        self.assertFalse((self.destination / "result.json").exists())

    def test_child_rss_cap_is_measured_from_wait4(self):
        result = self.run_capture(self.child(self.valid), rss_cap=1)
        self.assertEqual(result["reason"], "child-rss-cap")
        self.assertGreater(result["child_peak_rss_bytes"], 1)


if __name__ == "__main__":
    unittest.main()
