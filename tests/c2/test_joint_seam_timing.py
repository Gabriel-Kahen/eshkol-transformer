"""Offline completeness and phase-attribution checks for the opt-in receipt."""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest

from c2_joint_seam_timing import SPECS, generate, validate


class JointSeamTimingTest(unittest.TestCase):
    def test_overlay_and_complete_attribution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generated = root / "timed.esk"
            generate(generated)
            source = generated.read_text()
            self.assertIn("(set! c2-save-model-encode-internal", source)
            self.assertIn("(set! c2-save-o2-stage-internal", source)
            self.assertIn("c2-timed-save-native-measure", (generated.parent / "overlay/c2_checkpoint_save_extension.esk").read_text())
            self.assertIn("c2-timed-load-native-stage", (generated.parent / "overlay/c2_checkpoint_load_extension.esk").read_text())

            phases = [1, 2, *range(10, 17), *range(20, 27), *range(30, 37), 40, 41]
            stamps = {phase: (1000 if phase == 1 else 2000 if phase == 2 else
                              3000 + (phase // 10 - 1) * 1000 + phase % 10 * 100
                              if phase < 40 else 7000 + phase)
                      for phase in phases}
            progress = root / "progress"
            progress.write_text("".join(
                f"C2 OPERATIONAL PROGRESS: phase={phase} monotonic-ms={stamps[phase]}\n"
                for phase in phases))
            observations = {}
            for name, _, _, count in SPECS:
                for index in range(count):
                    if name.startswith("load-"):
                        start = {"load-native-measure": 1100,
                                 "load-native-stage": 1300,
                                 "load-parse": 1500}[name]
                    else:
                        save = index // 2 if name in ("model", "o2") else index
                        parent = 12 if name == "save-native-measure" or (
                            name in ("model", "o2") and index % 2 == 0) else 13
                        start = stamps[save * 10 + parent] + (
                            10 if name == "model" else 40 if name == "o2" else 70)
                    observations[name, index] = (start, start + 10)
            stdout = root / "stdout"

            def write():
                stdout.write_text("C2 OPERATIONAL RUNTIME PASS: 26 checks mode=joint baseline-kib=1\n" + "".join(
                    f"C2 SEAM TIMING: name={name} index={index} start-ms={start} end-ms={end}\n"
                    for (name, index), (start, end) in observations.items()))

            write()
            with contextlib.redirect_stdout(io.StringIO()) as captured:
                validate(stdout, progress)
            self.assertIn("observations=21 progress=25", captured.getvalue())
            complete = stdout.read_text()
            stdout.write_text(complete.replace("26 checks", "25 checks"))
            with self.assertRaisesRegex(ValueError, "exact 26-check"):
                validate(stdout, progress)
            stdout.write_text(complete)
            stdout.write_text(complete + "C2 SEAM TIMING OVERFLOW: name=model calls=7\n")
            with self.assertRaisesRegex(ValueError, "overflow"):
                validate(stdout, progress)
            stdout.write_text(complete)
            del observations["model", 5]
            write()
            with self.assertRaisesRegex(ValueError, "incomplete observations"):
                validate(stdout, progress)
            observations["model", 5] = (stamps[33] - 1, stamps[33] + 1)
            write()
            with self.assertRaisesRegex(ValueError, "outside parent phase"):
                validate(stdout, progress)
            observations["model", 5] = (stamps[33] + 35, stamps[33] + 45)
            write()
            with self.assertRaisesRegex(ValueError, "save seam order changed"):
                validate(stdout, progress)


if __name__ == "__main__":
    unittest.main()
