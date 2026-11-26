"""Bounded source and synthetic receipt checks for the P1 diagnostic workflow."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/p1-pair-memory-diagnostic.yml"
CANDIDATE = "15f2522946b8ee13b9f166967a99f2665263d4d0"
TREE = "672e9da3d40061278e94ca0aa1954e98936afa7e"


def block(source, begin, end):
    return source.split(begin, 1)[1].split(end, 1)[0]


class PairMemoryDiagnosticTests(unittest.TestCase):
    def test_branch_only_exact_source_and_supported_lane(self):
        text = WORKFLOW.read_text()
        self.assertIn("  pull_request:\n    branches: [codex/p1-pair-memory-hosted-diagnostic-20261126]", text)
        self.assertNotIn("  push:\n", text)
        self.assertNotIn("  workflow_dispatch:\n", text)
        self.assertIn("startsWith(github.head_ref, 'codex/p1-pair-memory-trigger-')", text)
        self.assertIn("ref: " + CANDIDATE, text)
        self.assertIn("test \"$(git rev-parse HEAD)\" = " + CANDIDATE, text)
        self.assertIn("test \"$(git rev-parse 'HEAD^{tree}')\" = " + TREE, text)
        for setting in ("P1_TWO_COMPILERS", "P1_PAIR_MEMORY_TRACE", "P1_KEEP_TMP"):
            self.assertIn(setting + ": '1'", text)
        self.assertIn("scripts/ci-build-prerequisites.sh parameter-state", text)
        self.assertIn("make test-ci-parameters-after-build", text)
        self.assertNotIn("make test-acceptance", text)
        self.assertIn("if: always()", text)
        self.assertIn("sha256sum -c SHA256SUMS", text)
        self.assertNotIn(".deps/eshkol-transformer-p1.*\n          path:", text)

    def test_collector_accepts_only_validated_trace_files(self):
        text = WORKFLOW.read_text()
        section = block(text, "      - name: Collect validated pair-memory evidence\n",
                        "      - name: Seal diagnostic receipt\n")
        script = textwrap.dedent(section.split("        run: |\n", 1)[1])
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp)
            source = work / ".deps/eshkol-transformer-p1.abc123"
            source.mkdir(parents=True)
            receipt = work / "p1-pair-memory-diagnostic"
            receipt.mkdir()
            (receipt / "parameter-state.log").write_text(
                "P1 preserved temporary evidence: " + str(source) + "\n")
            (source / "compiler-timing.log").write_text(
                "elapsed_s=1.00 max_rss_kib=2000 exit=0 command=compiler\n")
            header = ("utc\tmonotonic_ns\tpgid_first\tpgid_second\t"
                      "rss_first_bytes\trss_second_bytes\trss_sum_bytes\t"
                      "processes_first\tprocesses_second\n")
            row = "2026-11-26T10:00:00.000+00:00\t1\t10\t11\t100\t200\t300\t1\t1\n"
            for index in range(1, 7):
                trace = source / f"pair-{index}-memory.tsv"
                trace.write_text(header + row)
                trace.with_suffix(".summary.json").write_text(json.dumps({
                    "scope": "sampled RSS of two P1 compiler process groups only",
                    "samples": 1, "both_active_samples": 1,
                    "max_sampled_pair_rss_bytes": 300, "interval_seconds": 0.25}))
            (source / "build-scratch.bin").write_bytes(b"not evidence")
            env = {**os.environ, "RUNNER_TEMP": temp, "SUITE_OUTCOME": "success"}
            command = ["bash", "-euo", "pipefail", "-c", script]
            result = subprocess.run(command, cwd=work, env=env,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            copied = receipt / "validated-p1"
            self.assertEqual(len(list(copied.iterdir())), 13)
            self.assertFalse((copied / "build-scratch.bin").exists())
            (source / "pair-1-memory.summary.json").write_text("{}")
            new_receipt = work / "bad/p1-pair-memory-diagnostic"
            new_receipt.mkdir(parents=True)
            (new_receipt / "parameter-state.log").write_text(
                "P1 preserved temporary evidence: " + str(source) + "\n")
            failed = subprocess.run(["bash", "-euo", "pipefail", "-c", script],
                                    cwd=work, env={**env, "RUNNER_TEMP": str(work / "bad")},
                                    capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)


if __name__ == "__main__":
    unittest.main()
