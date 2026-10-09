"""Owned source-only controls for the inactive F loader outer launcher."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from tests.g3c4 import f_loader_outer as outer
from tests.g3c4 import check_f_loader_outer as source_check


class OuterTests(unittest.TestCase):
    def test_source_guard_denies_root_pin_and_profile_mutants(self) -> None:
        source = (source_check.ROOT / source_check.ADDED[1]).read_text()
        guest = (source_check.ROOT / source_check.ADDED[0]).read_text()
        original = source_check.OLD
        roadmap = source_check.NEW
        base = source_check.BASE.read_text().splitlines()
        closure = source_check.CLOSURE.read_text().splitlines()
        with self.assertRaises(AssertionError):
            source_check.validate(base, closure, source.replace(
                'external immutable ROOT wrapper required',
                'caller-provided pin is accepted'),
                guest, roadmap, original)
        with self.assertRaises(AssertionError):
            source_check.validate(base, closure,
                                  source.replace('"--network", "none"',
                                                 '"--network", "host"'),
                                  guest, roadmap, original)
        with self.assertRaises(AssertionError):
            source_check.validate(base, closure[:-1], source, guest,
                                  roadmap, original)

    def test_inactive_exact_root_pin(self) -> None:
        with self.assertRaisesRegex(ValueError, "exact external ROOT decision pin"):
            outer.selected(Path("/missing"), "caller-selected", fresh=True)

    def test_create_profile_and_guest_route(self) -> None:
        d = {"name": "g3c4-f-loader-owned", "compiler": "/one",
             "source": "/two", "f_output": "/three", "bundle": "/four",
             "candidate": "/five", "out": "/six",
             "source_admission_sha256": "a" * 64}
        argv = outer.create_argv(d)
        self.assertEqual(argv.count("--mount"), 6)
        self.assertEqual(argv.count("--read-only"), 1)
        self.assertEqual(argv[argv.index("--network") + 1], "none")
        self.assertEqual(argv[argv.index("--memory") + 1], "4g")
        self.assertEqual(argv[argv.index("--pids-limit") + 1], "128")
        self.assertEqual(argv[-2:], ["--source-admission-sha256", "a" * 64])
        self.assertIn("tests.g3c4.f_loader_guest", argv)
        self.assertEqual(argv[-4], "-m")

    def test_real_wait_and_raw_streams(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "normal"
            r = outer.capture([sys.executable, "-B", "-c",
                               "import sys;sys.stdout.write('ok')"], path, 5)
            self.assertTrue(r["quiescent"])
            self.assertEqual(r["returncode"], 0)
            self.assertEqual(r["wait_status"], 0)
            self.assertEqual((path / "stdout.raw").read_bytes(), b"ok")
            self.assertEqual((path / "stderr.raw").read_bytes(), b"")

    def test_owned_overflow_and_timeout_are_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.object(outer, "STREAM_LIMIT", 32):
            path = Path(tmp)
            overflow = outer.capture([sys.executable, "-B", "-c",
                                      "import sys;sys.stderr.write('x'*1000)"],
                                     path / "overflow", 5)
            self.assertIn("overflow", overflow["failure"])
            self.assertLessEqual(len((path / "overflow/stderr.raw").read_bytes()), 32)
            timed = outer.capture([sys.executable, "-B", "-c",
                                   "import time;time.sleep(10)"], path / "timeout", 1)
            self.assertIn("timeout", timed["failure"])
            self.assertIsNotNone(timed["wait_status"])
            self.assertTrue((path / "timeout/status.json").is_file())

    def test_inspect_requires_owned_exact_mounts(self) -> None:
        d = {"name": "g3c4-f-loader-owned", "compiler": "/one",
             "source": "/two", "f_output": "/three", "bundle": "/four",
             "candidate": "/five", "out": "/six"}
        mounts = [{"Type": "bind", "Source": d[key], "Destination": dest,
                   "RW": not ro} for key, dest, ro in
                  (("compiler", "/workspace", True), ("source", "/source", True),
                   ("f_output", "/f-output", True), ("bundle", "/bundle", True),
                   ("candidate", "/candidate", False), ("out", "/out", False))]
        row = {"Id": "b" * 64, "Name": "/" + d["name"],
               "Image": outer.IMAGE, "Config": {"Labels": {"eshkol.gate": d["name"]}},
               "State": {"Running": False}, "Mounts": mounts,
               "HostConfig": {"ReadonlyRootfs": True, "NetworkMode": "none",
                              "Tmpfs": {"/tmp": "rw,nosuid,nodev,size=268435456,mode=1777"},
                              "Memory": 4 * 1024**3, "MemorySwap": 4 * 1024**3,
                              "PidsLimit": 128}}
        self.assertEqual(outer.inspect_owned(json.dumps([row]).encode(), d,
                                             "b" * 64, False), row)
        row["Mounts"].append({"Type": "bind", "Source": "/extra",
                              "Destination": "/extra", "RW": False})
        with self.assertRaisesRegex(ValueError, "mounts differ"):
            outer.inspect_owned(json.dumps([row]).encode(), d, "b" * 64, False)
        row["Mounts"].pop()
        row["HostConfig"]["ReadonlyRootfs"] = False
        with self.assertRaisesRegex(ValueError, "inspect mismatch"):
            outer.inspect_owned(json.dumps([row]).encode(), d, "b" * 64, False)

    def test_guest_receipt_must_be_durable_and_bound_to_library_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            evidence = out / "evidence"
            (evidence / "loader").mkdir(parents=True)
            with patch.object(outer.attest, "copied_artifacts"):
                with self.assertRaisesRegex(ValueError, "durable guest receipts"):
                    outer.guest_output({"out": str(out)})
                (evidence / "F-copy-prepared.json").write_text("{}")
                (evidence / "loader/image-libraries.tsv").write_text("lib\n")
                (evidence / "loader/collector.json").write_text(json.dumps({
                    "format": "g3c4-F-image-loader-collector-v1",
                    "image_pins_sha256": "0" * 64}))
                with self.assertRaisesRegex(ValueError, "output identity"):
                    outer.guest_output({"out": str(out)})
                (evidence / "loader/collector.json").write_text(json.dumps({
                    "format": "g3c4-F-image-loader-collector-v1",
                    "image_pins_sha256": outer.sha(b"lib\n")}))
                outer.guest_output({"out": str(out)})


if __name__ == "__main__":
    unittest.main()
