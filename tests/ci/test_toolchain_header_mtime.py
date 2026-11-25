"""The bootstrap keeps Ninja's header dependency clean only for equal bytes."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


COMMON = Path(__file__).resolve().parents[2] / "scripts" / "common.sh"


class HeaderMtimeTest(unittest.TestCase):
    def test_equal_bytes_restore_original_mtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            before, after = root / "before", root / "after"
            before.write_bytes(b"same\n")
            after.write_bytes(b"same\n")
            os.utime(before, ns=(1_700_000_000_123_456_789,) * 2)
            os.utime(after, ns=(1_700_000_100_123_456_789,) * 2)
            subprocess.run(
                ["bash", "-c", 'source "$1"; restore_mtime_if_same_contents "$2" "$3"',
                 "bash", str(COMMON), str(before), str(after)],
                check=True,
            )
            self.assertEqual(after.stat().st_mtime_ns, before.stat().st_mtime_ns)

    def test_changed_bytes_keep_new_mtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            before, after = root / "before", root / "after"
            before.write_bytes(b"old\n")
            after.write_bytes(b"new\n")
            os.utime(before, ns=(1_700_000_000_123_456_789,) * 2)
            changed_mtime = 1_700_000_100_123_456_789
            os.utime(after, ns=(changed_mtime,) * 2)
            subprocess.run(
                ["bash", "-c", 'source "$1"; restore_mtime_if_same_contents "$2" "$3"',
                 "bash", str(COMMON), str(before), str(after)],
                check=True,
            )
            self.assertEqual(after.stat().st_mtime_ns, changed_mtime)


if __name__ == "__main__":
    unittest.main()
