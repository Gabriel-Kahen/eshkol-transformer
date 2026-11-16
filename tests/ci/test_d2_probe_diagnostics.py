"""Exercise the D2 resource probe shell diagnostics without compiling D2."""

from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "test-d2.sh"


class D2ProbeDiagnosticsTests(unittest.TestCase):
    def run_probe(self, child_body, *, advance_to=None):
        source = SCRIPT.read_text()
        function = source.split("run_resource_probe() {\n", 1)[1].split(
            "\n}\nrun_resource_probe small", 1
        )[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "public-a-resource_instrumented_runtime" / "resource_instrumented_runtime"
            executable.parent.mkdir()
            executable.write_text("#!/usr/bin/bash\n" + child_body)
            executable.chmod(0o700)
            command = (
                "set -euo pipefail\n"
                "d2_tmp=$1\n"
                "d2_fixture=unused\n"
                'die() { printf "error: %s\\n" "$1" >&2; exit 1; }\n'
                f"run_resource_probe() {{\n{function}\n}}\n"
                + (
                    f"sleep() {{ SECONDS=$((started + {advance_to})); /bin/sleep 0.01; }}\n"
                    if advance_to is not None
                    else ""
                )
                + 'run_resource_probe large unused 8192\n'
            )
            return subprocess.run(
                ["bash", "-c", command, str(SCRIPT), str(root)],
                capture_output=True,
                text=True,
                timeout=5,
            )

    def test_success_reports_start_and_end_without_changing_stdout_assertions(self):
        child = (
            'echo "D2 RESOURCE CONSUME PASS: 8192 batches"\n'
            'echo "D2 RESOURCE NATIVE COUNTERS: baseline=0/0/0/0 open-delta=1/0/0/2 live-delta=1/1/0/4 released-delta=1/0/0/2 post-close=0/0/0/0"\n'
            'echo "D2 RESOURCE NATIVE FD: baseline=0 peak=1 post-close=0 baseline-to-peak=1 final-delta=0"\n'
        )
        result = self.run_probe(child)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertRegex(result.stdout, r"D2 RESOURCE PROBE START: large limit=300s\n")
        self.assertRegex(
            result.stdout,
            r"D2 RESOURCE PROBE END: large elapsed=\d+s exit=0 rss-max=\d+ KiB fd-max=\d+\n",
        )

    def test_failure_reports_exit_code(self):
        result = self.run_probe("exit 7\n")
        self.assertEqual(result.returncode, 1)
        self.assertRegex(result.stdout, r"D2 RESOURCE PROBE END: large elapsed=\d+s exit=7 ")
        self.assertIn("D2 large resource probe failed", result.stderr)

    def test_periodic_sample_reports_cpu_and_resource_observations(self):
        result = self.run_probe("exec /bin/sleep 0.1\n", advance_to=30)
        self.assertEqual(result.returncode, 1)
        self.assertRegex(
            result.stdout,
            r"D2 RESOURCE PROBE SAMPLE: large elapsed=30s rss=\d+ KiB "
            r"rss-max=\d+ KiB fd=\d+ fd-max=\d+ cpu-ticks=\d+\n",
        )
        self.assertNotIn("D2 RESOURCE PROBE TIMEOUT", result.stdout)

    def test_timeout_reports_last_observations_and_retains_300_second_cap(self):
        result = self.run_probe("exec /bin/sleep 30\n", advance_to=301)
        self.assertEqual(result.returncode, 1)
        self.assertRegex(
            result.stdout,
            r"D2 RESOURCE PROBE TIMEOUT: large elapsed=301s limit=300s "
            r"last-rss=\d+ KiB rss-max=\d+ KiB last-fd=\d+ "
            r"fd-max=\d+ last-cpu-ticks=unavailable\n",
        )
        self.assertNotIn("D2 RESOURCE PROBE SAMPLE", result.stdout)
        self.assertIn("D2 large resource probe timed out", result.stderr)


if __name__ == "__main__":
    unittest.main()
