"""Exercise the package builder's exact cleanup helper without a compiler."""

from pathlib import Path
import signal
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / "scripts/e3-mode-graph-stage-cleanup.sh"
BUILDER = ROOT / "scripts/build-e3-mode-graph-test.sh"


class StageCleanupTest(unittest.TestCase):
    def test_private_generated_source_stage_is_stable_and_fresh(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e3-mode-graph-gate-") as temp:
            gate = Path(temp)
            stages = [gate / ".e3-mode-graph.ABC123",
                      gate / ".e3-mode-graph.XYZ789"]
            for stage in stages:
                stage.mkdir()

            def select(object_path: Path) -> subprocess.CompletedProcess[str]:
                return subprocess.run(
                    ["bash", "-c", 'source "$1"; e3_mode_graph_generated_stage "$2"',
                     "bash", str(HELPER), str(object_path)],
                    capture_output=True, text=True, check=False,
                )

            expected = str(gate / ".e3-mode-graph-generated") + "\n"
            for stage in stages:
                selected = select(stage / "e3_mode_graph_package.o")
                self.assertEqual((selected.returncode, selected.stdout),
                                 (0, expected), selected.stderr)
            self.assertNotEqual(select(stages[0] / "other.o").returncode, 0)
            alias = gate / ".e3-mode-graph.ALIAS1"
            alias.symlink_to(stages[0], target_is_directory=True)
            self.assertNotEqual(select(alias / "e3_mode_graph_package.o").returncode, 0)
            (gate / ".e3-mode-graph-generated").symlink_to(
                stages[0], target_is_directory=True)
            self.assertNotEqual(select(stages[0] / "e3_mode_graph_package.o").returncode, 0)

    def test_failed_and_signaled_builds_retain_actual_stage(self) -> None:
        self.assertIn('source "${PROJECT_ROOT}/scripts/e3-mode-graph-stage-cleanup.sh"',
                      BUILDER.read_text())
        self.assertIn('e3_mode_graph_install_stage_cleanup "${staging}"',
                      BUILDER.read_text())
        for label, signum, status in (("failure", None, 7),
                                      ("TERM", signal.SIGTERM, 143),
                                      ("INT", signal.SIGINT, 130)):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temp:
                stage = Path(temp) / "stage"
                stage.mkdir()
                (stage / "actual-manifest.txt").write_text("emitted inventory\n")
                script = ('source "$1"; e3_mode_graph_install_stage_cleanup "$2"; '
                          'printf "READY\\n"; '
                          + ('while :; do sleep 0.05; done' if signum else 'exit 7'))
                process = subprocess.Popen(
                    ["bash", "-c", script, "bash", str(HELPER), str(stage)],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                assert process.stdout is not None
                self.assertEqual(process.stdout.readline(), "READY\n")
                if signum:
                    process.send_signal(signum)
                _, stderr = process.communicate(timeout=5)
                self.assertEqual(process.returncode, status, stderr)
                self.assertEqual((stage / "exit.status").read_text(), f"{status}\n")
                self.assertEqual((stage / "actual-manifest.txt").read_text(),
                                 "emitted inventory\n")
                self.assertIn("failed staging retained", stderr)

    def test_success_cleans_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            stage = Path(temp) / "stage"
            stage.mkdir()
            (stage / "actual-manifest.txt").write_text("emitted inventory\n")
            result = subprocess.run(
                ["bash", "-c",
                 'source "$1"; e3_mode_graph_install_stage_cleanup "$2"; exit 0',
                 "bash", str(HELPER), str(stage)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(stage.exists())


if __name__ == "__main__":
    unittest.main()
