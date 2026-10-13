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
  if (argc == 2 && strcmp(argv[1], "repeat") == 0) {
    for (int i = 0; i < 1000; ++i) et_c2_operational_progress_v1(1);
    return 0;
  }
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


def build_driver(root):
    executable = root / "progress-driver"
    subprocess.run(
        [os.environ.get("CC", "cc"), "-std=c11", "-Wall", "-Wextra",
         "-Werror", "-x", "c", "-", str(METRICS), "-o", str(executable)],
        input=DRIVER, text=True, check=True, capture_output=True,
    )
    return executable


class OperationalProgressTest(unittest.TestCase):
    def test_progress_is_bounded_persistent_and_separate_from_stderr(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = build_driver(root)
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
            subprocess.run([str(executable), "1", "2", "10", "11", "41", "16"],
                           env=env, check=True, capture_output=True)
            lines = path.read_text().splitlines()
            self.assertEqual(len(lines), 6)
            self.assertEqual(
                [int(re.search(r"phase=(\d+)", line).group(1)) for line in lines],
                [1, 2, 10, 11, 41, 16],
            )
            self.assertTrue(all(re.fullmatch(
                r"C2 OPERATIONAL PROGRESS: phase=\d+ monotonic-ms=\d+", line
            ) for line in lines))
            subprocess.run([str(executable), "1"], env=env, check=True,
                           capture_output=True)
            self.assertEqual(path.read_text().splitlines(), lines)

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

    def test_rejects_existing_targets_and_caps_append(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = build_driver(root)
            target = root / "unrelated"
            target.write_bytes(b"do not change")
            paths = [root / "existing", root / "symlink", root / "hardlink",
                     root / "pipe"]
            paths[0].write_bytes(b"x" * 1048576)
            paths[1].symlink_to(target)
            os.link(target, paths[2])
            os.mkfifo(paths[3])
            for path in paths:
                env = {**os.environ, "C2_OPERATIONAL_PROGRESS_FILE": str(path)}
                result = subprocess.run([str(executable), "1"], env=env,
                                        capture_output=True, timeout=2)
                self.assertEqual(result.returncode, 0, path.name)
                self.assertEqual((result.stdout, result.stderr), (b"", b""))
            self.assertEqual(target.read_bytes(), b"do not change")
            self.assertEqual(paths[0].stat().st_size, 1048576)
            self.assertTrue(paths[3].is_fifo())

            capped = root / "capped.progress"
            env = {**os.environ, "C2_OPERATIONAL_PROGRESS_FILE": str(capped)}
            subprocess.run([str(executable), "repeat"], env=env,
                           check=True, capture_output=True)
            self.assertLessEqual(capped.stat().st_size, 2048)
            self.assertGreaterEqual(len(capped.read_text().splitlines()), 25)
            self.assertLessEqual(len(capped.read_text().splitlines()), 32)

    def test_failure_dump_rejects_unsafe_progress(self):
        source = (ROOT / "scripts/test-c2-checkpoint-operational.sh").read_text()
        functions = ("progress_file_valid() {" +
                     source.split("progress_file_valid() {", 1)[1]
                     .split("trap cleanup EXIT", 1)[0] + "trap cleanup EXIT\n")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outside = root / "outside"
            outside.write_text("C2 OPERATIONAL PROGRESS: phase=23 monotonic-ms=1\n")
            for kind in ("valid", "symlink", "hardlink", "fifo",
                         "malformed", "oversize"):
                tmp = root / kind
                progress = tmp / "clang-b/joint.progress"
                progress.parent.mkdir(parents=True)
                if kind == "valid":
                    progress.write_text(outside.read_text())
                elif kind == "symlink":
                    progress.symlink_to(outside)
                elif kind == "hardlink":
                    os.link(outside, progress)
                elif kind == "fifo":
                    os.mkfifo(progress)
                elif kind == "malformed":
                    progress.write_text("not a progress record\n")
                else:
                    progress.write_bytes(b"x" * 2049)
                env = {**os.environ, "tmp": str(tmp), "development": "0",
                       "active_runtime_label": "clang-b",
                       "active_runtime_mode": "joint"}
                result = subprocess.run(["bash", "-c", functions + "exit 124\n"],
                                        env=env, text=True, capture_output=True,
                                        timeout=2)
                self.assertEqual(result.returncode, 124)
                if kind == "valid":
                    self.assertIn("phase=23", result.stderr)
                    self.assertNotIn("UNAVAILABLE", result.stderr)
                else:
                    self.assertIn("PROGRESS UNAVAILABLE: clang-b/joint",
                                  result.stderr)
                    self.assertNotIn("phase=23", result.stderr)
            self.assertEqual(outside.read_text(),
                             "C2 OPERATIONAL PROGRESS: phase=23 monotonic-ms=1\n")


if __name__ == "__main__":
    unittest.main()
