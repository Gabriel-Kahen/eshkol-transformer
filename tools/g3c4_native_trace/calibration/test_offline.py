"""Source-only controls for the prepared v2 calibration bundle."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

BUNDLE = Path(__file__).resolve().parent
PLAN = json.loads((BUNDLE / "plan.json").read_text())
os.environ["G3C4_TRACE_SOURCE"] = PLAN["source_worktree"]


def load(name):
    spec = importlib.util.spec_from_file_location(name, BUNDLE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LAUNCHER = load("launcher")
POSTFLIGHT = load("postflight")


class PreparedCalibration(unittest.TestCase):
    def test_exact_source_and_disabled_template(self):
        self.assertFalse(PLAN["launch_enabled"])
        self.assertFalse((BUNDLE / "CLAIMED").exists())
        self.assertFalse((BUNDLE / "RESULT").exists())
        LAUNCHER.vessel().pristine(BUNDLE)
        LAUNCHER.verify_inputs(BUNDLE, PLAN)

    def test_enabled_copy_is_new_and_sealed(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "one-use"
            LAUNCHER.prepare_enabled(target)
            self.assertTrue(json.loads((target / "plan.json").read_text())["launch_enabled"])
            self.assertFalse((target / "CLAIMED").exists())
            LAUNCHER.vessel().pristine(target)
            with self.assertRaisesRegex(ValueError, "already exists"):
                LAUNCHER.prepare_enabled(target)

    def test_causal_cases_and_failure_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bin").mkdir()
            for binary in ("trace", "canary"):
                (root / "bin" / binary).write_bytes(b"ELF fixture")
            digest = "0" * 64

            def exe(when, pid, parent, ticks, argv=b"/out/bin/canary\0fast\0"):
                return (f"exec\t{when}\t{pid}\t{parent}\t{ticks}\t1\t1\t{argv.hex()}\t"
                        f"{b'/out/bin/canary'.hex()}\t{b'/workspace'.hex()}\t{digest}")

            def write_case(name, body, footer, status=0, stderr=""):
                (root / f"{name}.events").write_text(
                    "g3c4-native-exec-v2\n" + "\n".join(body + footer) + "\n")
                (root / f"{name}.exit").write_text(f"{status}\n")
                (root / f"{name}.stderr").write_text(stderr)

            root_exec = exe(1, 100, 0, 1)
            fast = [root_exec]
            for child in range(101, 133):
                fast.extend((f"spawn\t2\t100\t{child}\tfork\t{child}\t1",
                             exe(2, child, 100, 1, b"/bin/true\0"),
                             f"exit\t2\t{child}\t0\t0"))
            fast.append("exit\t3\t100\t0\t0")
            write_case("fast", fast, ["complete\t4\t100\t0\t32\t33\t33\t0"])
            child = [root_exec, "spawn\t2\t100\t101\tvfork\t101\t1",
                     exe(3, 101, 100, 1, b"/bin/true\0"),
                     "exit\t4\t101\t0\t0", "exit\t5\t100\t0\t0"]
            write_case("spawn", child, ["complete\t6\t100\t0\t1\t2\t2\t0"])
            thread = [root_exec, "spawn\t2\t100\t101\tclone\t100\t2",
                      "exit\t3\t101\t0\t0", "exit\t4\t100\t0\t0"]
            write_case("thread", thread, ["complete\t5\t100\t0\t1\t1\t2\t0"])
            merged = [root_exec, "spawn\t2\t100\t101\tclone\t100\t2",
                      "merge\t3\t101\t100", exe(4, 100, 0, 2, b"/bin/true\0"),
                      "exit\t5\t100\t0\t0"]
            write_case("threadexec", merged,
                       ["complete\t6\t100\t0\t1\t2\t1\t1"])
            write_case("fault", [root_exec, "spawn\t2\t100\t101\tfork\t101\t1",
                                 exe(3, 101, 100, 1, b"/bin/touch\0")], [], 86,
                       "native trace HOLD: injected supervisor failure")
            write_case("untraced", [root_exec], [], 86,
                       "native trace HOLD: forbidden clone route")
            self.assertEqual(POSTFLIGHT.audit(root)["status"],
                             "V2_CALIBRATION_ONLY_UNADMITTED")
            (root / "fault-unwanted").write_text("escaped")
            with self.assertRaisesRegex(ValueError, "child escaped"):
                POSTFLIGHT.audit(root)
            (root / "fault-unwanted").unlink()
            (root / "threadexec.events").write_text(
                (root / "threadexec.events").read_text().replace("clone\t100\t2",
                                                                 "clone\t101\t2"))
            with self.assertRaises(ValueError):
                POSTFLIGHT.audit(root)


if __name__ == "__main__":
    unittest.main()
