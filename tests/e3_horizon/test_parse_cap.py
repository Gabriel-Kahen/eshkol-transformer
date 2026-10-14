from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.e3_horizon.parse_cap import verify

STDOUT = (
    "E3-CAP-PASS rows=8193 denials=2 cursor=unchanged "
    "first-shard=unread arena-delta=8976\n"
)
TIME = "elapsed_seconds=0.12 peak_rss_kib=160948\n"


class CapTranscriptTests(unittest.TestCase):
    def test_exact_record_and_negative_mutations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            shard = Path(temporary) / "shard-0000000000000000.ets"
            shard.write_bytes(b"fixture")
            verify(STDOUT, "", TIME, shard)
            bad = (
                (STDOUT.replace("8193", "8192"), "", TIME),
                (STDOUT.replace("denials=2", "denials=1"), "", TIME),
                (STDOUT.replace("unread", "read"), "", TIME),
                (STDOUT + STDOUT, "", TIME),
                (STDOUT, "warning\n", TIME),
                (STDOUT, "", TIME.replace("160948", "524289")),
                (STDOUT, "", TIME.replace("0.12", "120.01")),
                (STDOUT, "", TIME + TIME),
            )
            for stdout, stderr, timing in bad:
                with self.subTest(stdout=stdout, stderr=stderr, timing=timing):
                    with self.assertRaises(ValueError):
                        verify(stdout, stderr, timing, shard)
            shard.rename(shard.with_name(shard.name + ".held"))
            with self.assertRaisesRegex(ValueError, "not restored"):
                verify(STDOUT, "", TIME, shard)


if __name__ == "__main__":
    unittest.main()
