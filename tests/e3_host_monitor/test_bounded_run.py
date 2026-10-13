"""Offline fail-closed tests for the successor E3 one-use host monitor."""

from __future__ import annotations

import errno
import importlib.util
from contextlib import nullcontext
from pathlib import Path
import os
import sys
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parent / "bounded_run.py"
SPEC = importlib.util.spec_from_file_location("e3_bounded_run", SCRIPT)
assert SPEC and SPEC.loader
monitor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(monitor)


def receipt(path: Path) -> dict[str, str]:
    return dict(line.split("\t", 1) for line in path.read_text().splitlines())


class BoundedRunTest(unittest.TestCase):
    def test_regular_tree_and_original_traversal_caps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "nested").mkdir()
            (root / "nested/data").write_bytes(b"abc")
            self.assertEqual(monitor.output_bytes(root), 3)
            with self.assertRaises(monitor.OutputScanFault) as caught:
                monitor.output_bytes(root, max_entries=1)
            self.assertEqual(caught.exception.code, "traversal-bound")
            with self.assertRaises(monitor.OutputScanFault) as caught:
                monitor.output_bytes(root, max_depth=0)
            self.assertEqual(caught.exception.code, "traversal-bound")

    def test_symlink_hardlink_and_special_entries_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "generated").mkdir()
            link = root / "generated/odd\tname\n"
            link.symlink_to("/host/private/credential")
            with self.assertRaises(monitor.OutputScanFault) as caught:
                monitor.output_bytes(root)
            self.assertEqual(caught.exception.code, "symlink-entry")
            detail = monitor.safe_scan_detail(caught.exception, "live")
            self.assertEqual(detail[3], "generated/odd%09name%0A")
            self.assertNotIn("/host/private", "\t".join(detail))
            link.unlink()
            first = root / "file"
            first.write_bytes(b"data")
            os.link(first, root / "other")
            with self.assertRaises(monitor.OutputScanFault) as caught:
                monitor.output_bytes(root)
            self.assertEqual(caught.exception.code, "hardlinked-entry")
            first.unlink()
            (root / "other").unlink()
            os.mkfifo(root / "pipe")
            with self.assertRaises(monitor.OutputScanFault) as caught:
                monitor.output_bytes(root)
            self.assertEqual(caught.exception.code, "nonregular-entry")

    def test_os_error_diagnostic_omits_raw_message_and_host_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            error = PermissionError(errno.EACCES, "TOP_SECRET", "/host/private")
            with mock.patch.object(monitor.os, "scandir", side_effect=error):
                with self.assertRaises(monitor.OutputScanFault) as caught:
                    monitor.output_bytes(root)
            detail = monitor.safe_scan_detail(caught.exception, "live")
            self.assertEqual(detail, (
                "live", "OSError", "scandir-error", ".",
                "guest output directory scan failed", str(errno.EACCES)))
            self.assertNotIn("TOP_SECRET", "\t".join(detail))
            self.assertNotIn("/host/private", "\t".join(detail))

            class BrokenEntry:
                name = "bad\tentry"

                def stat(self, *, follow_symlinks: bool):
                    if follow_symlinks:
                        raise AssertionError("scanner followed an entry")
                    raise OSError(errno.EIO, "SECRET_STAT", "/host/secret")

            with mock.patch.object(monitor.os, "scandir",
                                   return_value=nullcontext([BrokenEntry()])):
                with self.assertRaises(monitor.OutputScanFault) as caught:
                    monitor.output_bytes(root)
            detail = monitor.safe_scan_detail(caught.exception, "live")
            self.assertEqual(detail[1:4],
                             ("OSError", "stat-error", "bad%09entry"))
            self.assertEqual(detail[5], str(errno.EIO))
            self.assertNotIn("SECRET_STAT", "\t".join(detail))
            self.assertNotIn("/host/secret", "\t".join(detail))

    def test_diagnostic_path_is_relative_encoded_and_bounded(self) -> None:
        path = monitor.safe_guest_path(("generated", "secret\r\n\t\\%", "x" * 255))
        self.assertLessEqual(len(path), 256)
        self.assertTrue(path.startswith("generated/secret%0D%0A%09%5C%25/"))
        self.assertNotIn("\n", path)
        self.assertEqual(monitor.safe_guest_path(("..",)),
                         "<invalid-relative-path>")
        self.assertEqual(monitor.safe_guest_path(("\ud800",)),
                         "<unavailable-relative-path>")

    def test_transient_symlink_keeps_first_live_fault_after_clean_final_scan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            run = base / "run"
            output = base / "output"
            run.mkdir()
            (output / "generated").mkdir(parents=True)
            link = output / "generated/odd\tname\n"
            original = monitor.output_bytes
            calls = 0

            def transient(root: Path, *args, **kwargs) -> int:
                nonlocal calls
                calls += 1
                if calls == 1:
                    link.symlink_to("/host/private/credential")
                    try:
                        return original(root, *args, **kwargs)
                    finally:
                        link.unlink()
                return original(root, *args, **kwargs)

            with mock.patch.object(monitor, "output_bytes", side_effect=transient):
                status = monitor.run(
                    run, output, 4096, 4096, 10,
                    [sys.executable, "-c", "import time; time.sleep(30)"],
                )
            self.assertEqual(status, 74)
            row = receipt(run / "docker.monitor.tsv")
            self.assertEqual(row["reason"], "unsafe-guest-output")
            self.assertEqual(row["guest_output_bytes"], "0")
            self.assertEqual(row["first_output_error_phase"], "live")
            self.assertEqual(row["first_output_error_class"], "ValueError")
            self.assertEqual(row["first_output_error_code"], "symlink-entry")
            self.assertEqual(row["first_output_error_path"],
                             "generated/odd%09name%0A")
            self.assertEqual(row["host_group_absent"], "true")
            self.assertEqual((run / "docker.exit").read_text(), "74\n")
            self.assertGreaterEqual(calls, 2)
            self.assertNotIn("/host/private", (run / "docker.monitor.tsv").read_text())
            self.assertNotIn(str(base), (run / "docker.monitor.tsv").read_text())

    def test_success_receipt_has_no_fault_and_retains_output_cap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            run = base / "run"
            output = base / "output"
            run.mkdir()
            output.mkdir()
            (output / "data").write_bytes(b"abc")
            command = [sys.executable, "-c", "print('ok')"]
            self.assertEqual(monitor.run(run, output, 4096, 4096, 10, command), 0)
            row = receipt(run / "docker.monitor.tsv")
            self.assertEqual(row["reason"], "completed")
            self.assertEqual(row["guest_output_bytes"], "3")
            self.assertEqual(row["first_output_error_phase"], "none")
            self.assertEqual((run / "docker.stdout").read_bytes(), b"ok\n")
            second = base / "second"
            second.mkdir()
            self.assertEqual(monitor.run(second, output, 4096, 2, 10, command), 74)
            self.assertEqual(receipt(second / "docker.monitor.tsv")["reason"],
                             "guest-output-cap")

    def test_stream_and_wall_caps_still_terminate_client_group(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            output = base / "output"
            output.mkdir()
            stream = base / "stream"
            stream.mkdir()
            noisy = [sys.executable, "-c",
                     "import os,time; os.write(1,b'x'*100000); time.sleep(30)"]
            self.assertEqual(monitor.run(stream, output, 16, 4096, 10, noisy), 74)
            row = receipt(stream / "docker.monitor.tsv")
            self.assertEqual(row["reason"], "host-stream-cap")
            self.assertEqual(row["stdout_kept_bytes"], "16")
            self.assertEqual(row["host_group_absent"], "true")
            wall = base / "wall"
            wall.mkdir()
            sleepy = [sys.executable, "-c", "import time; time.sleep(30)"]
            self.assertEqual(monitor.run(wall, output, 4096, 4096, 1, sleepy), 74)
            row = receipt(wall / "docker.monitor.tsv")
            self.assertEqual(row["reason"], "wall-timeout")
            self.assertEqual(row["host_group_absent"], "true")


if __name__ == "__main__":
    unittest.main()
