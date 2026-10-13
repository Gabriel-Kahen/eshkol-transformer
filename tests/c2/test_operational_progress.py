#!/usr/bin/env python3
"""Offline checks for the watchdog-surviving C2 joint progress witness."""

import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "tests/c2/c2_checkpoint_operational_metrics.c"
DRIVER = r"""
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

extern void et_c2_operational_progress_v1(int64_t phase);

int main(int argc, char **argv) {
  if (argc == 2 && strcmp(argv[1], "hold") == 0) {
    et_c2_operational_progress_v1(1);
    pause();
    return 0;
  }
  for (int i = 1; i < argc; ++i)
    et_c2_operational_progress_v1((int64_t)strtoll(argv[i], NULL, 10));
  return 0;
}
"""


class OperationalProgressTest(unittest.TestCase):
    def test_progress_is_bounded_persistent_and_separate_from_stderr(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "progress-driver"
            subprocess.run(
                [os.environ.get("CC", "cc"), "-std=c11", "-Wall", "-Wextra",
                 "-Werror", "-x", "c", "-", str(METRICS), "-o", str(executable)],
                input=DRIVER, text=True, check=True, capture_output=True,
            )
            path = root / "joint.progress"
            env = os.environ.copy()
            env.pop("C2_OPERATIONAL_PROGRESS_FILE", None)
            cases = [
                (["1"], env),
                (["-1", "0", "42"],
                 {**env, "C2_OPERATIONAL_PROGRESS_FILE": str(path)}),
            ]
            for phases, selected_env in cases:
                result = subprocess.run([str(executable), *phases], env=selected_env,
                                        text=True, capture_output=True, check=True)
                self.assertEqual((result.stdout, result.stderr), ("", ""))
                self.assertFalse(path.exists())

            env["C2_OPERATIONAL_PROGRESS_FILE"] = str(path)
            subprocess.run([str(executable), "1", "2", "10", "11", "41"],
                           env=env, check=True, capture_output=True)
            subprocess.run([str(executable), "16"], env=env, check=True,
                           capture_output=True)
            lines = path.read_text().splitlines()
            self.assertEqual(len(lines), 6)
            self.assertEqual(
                [int(re.search(r"phase=(\d+)", line).group(1)) for line in lines],
                [1, 2, 10, 11, 41, 16],
            )
            self.assertTrue(all(re.fullmatch(
                r"C2 OPERATIONAL PROGRESS: phase=\d+ monotonic-ms=\d+", line
            ) for line in lines))

            env["C2_OPERATIONAL_PROGRESS_FILE"] = str(root / "missing" / "log")
            result = subprocess.run([str(executable), "1"], env=env,
                                    check=True, text=True, capture_output=True)
            self.assertEqual((result.stdout, result.stderr), ("", ""))

            watchdog_log = root / "watchdog.progress"
            env["C2_OPERATIONAL_PROGRESS_FILE"] = str(watchdog_log)
            result = subprocess.run(
                ["timeout", "--foreground", "--signal=TERM", "--kill-after=1s",
                 "0.1s", str(executable), "hold"],
                env=env, text=True, capture_output=True,
            )
            self.assertEqual(result.returncode, 124)
            self.assertRegex(watchdog_log.read_text(), r"phase=1 monotonic-ms=\d+")


if __name__ == "__main__":
    unittest.main()
