#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import tempfile
import unittest

from test_checkpoint_format import make_fixture


class CodecTests(unittest.TestCase):
    driver: Path

    def run_driver(self, source: Path, output: Path, mode: int) -> subprocess.CompletedProcess[str]:
        return subprocess.run([str(self.driver), str(source), str(output), str(mode)],
                              text=True, capture_output=True, check=False)

    def test_exact_fixture_and_failure_atomicity(self) -> None:
        data, _ = make_fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "input.c2"
            output = root / "output.c2"; repeated = root / "repeated.c2"
            source.write_bytes(data)
            run = self.run_driver(source, output, 0)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(output.read_bytes(), data)
            run = self.run_driver(source, repeated, 0)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(repeated.read_bytes(), output.read_bytes())
            for mode in range(1, 12):
                run = self.run_driver(source, output, mode)
                self.assertEqual(run.returncode, 0, (mode, run.stdout, run.stderr))
            for mode in (12, 13):
                run = self.run_driver(source, output, mode)
                self.assertEqual(run.returncode, 0, (mode, run.stdout, run.stderr))


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("driver", type=Path)
    args = parser.parse_args(); CodecTests.driver = args.driver.resolve()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CodecTests)
    return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
