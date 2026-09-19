from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "ci-build-prerequisites.sh"


def run_script(*args: str, build_dir: Path | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    if build_dir is not None:
        env["BUILD_DIR"] = str(build_dir)
    return subprocess.run(
        ["/usr/bin/bash", str(SCRIPT), *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def producers(result: subprocess.CompletedProcess[str]) -> list[str]:
    return [
        line.split("\t")[1]
        for line in result.stdout.splitlines()
        if line.startswith("producer\t")
    ]


class BuildPrerequisitesTests(unittest.TestCase):
    def test_native_plan_is_exact_and_excludes_unused_outer_builds(self) -> None:
        result = run_script("--plan", "native-numerics")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            producers(result),
            [
                "k1",
                "a2",
                "l2",
                "i1",
                "i2",
                "k2",
                "n2",
                "n3k",
                "t2",
                "d2",
                "o2",
            ],
        )
        self.assertNotIn("p1", result.stdout)
        self.assertNotIn("c1", result.stdout)
        self.assertNotIn("x1", result.stdout)

    def test_native_optimizer_plan_contains_exact_o2_inputs(self) -> None:
        result = run_script("--plan", "native-optimizer")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(producers(result), ["k1", "i2", "t2", "d2", "o2"])

    def test_native_partition_union_is_deduplicated(self) -> None:
        native = run_script("--plan", "native-numerics")
        combined = run_script(
            "--plan", "native-numerics", "native-optimizer"
        )
        self.assertEqual(native.returncode, 0, native.stderr)
        self.assertEqual(combined.returncode, 0, combined.stderr)
        self.assertEqual(producers(combined), producers(native))
        self.assertEqual(len(producers(combined)), len(set(producers(combined))))

    def test_smoke_is_a_separate_post_suite_prerequisite(self) -> None:
        native = run_script("--plan", "native-numerics")
        with_post_suite = run_script(
            "--plan", "native-numerics", "smoke-benchmark"
        )
        self.assertEqual(native.returncode, 0, native.stderr)
        self.assertEqual(with_post_suite.returncode, 0, with_post_suite.stderr)
        self.assertNotIn("smoke", producers(native))
        self.assertEqual(producers(with_post_suite)[0], "smoke")

    def test_predecessor_plan_is_deduplicated_for_multiple_suites(self) -> None:
        result = run_script(
            "--plan", "acceptance-predecessors", "contracts-data", "bpe-boundary"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        planned = producers(result)
        self.assertEqual(
            planned,
            [
                "k1",
                "a2",
                "l2",
                "i1",
                "i2",
                "k2",
                "n2",
                "n3k",
                "t2",
                "d2",
                "o2",
                "d1",
            ],
        )
        self.assertEqual(len(planned), len(set(planned)))
        self.assertEqual(planned.count("d1"), 1)
        self.assertEqual(planned.count("d2"), 1)
        self.assertNotIn("c2", planned)

    def test_partition_plans_only_build_canonical_artifacts_they_read(self) -> None:
        cases = {
            "contracts-data": ["d1"],
            "checkpoint-io": [],
            "parameter-state": [],
            "byte-tokenizer": ["d2"],
            "bpe-tokenizer": [],
            "bpe-boundary": ["d2"],
            "shard-loader": ["k1", "i1", "d2"],
            "c2-format": [],
            "c2-state": [],
            "c2-save": [],
            "c2-load": [],
            "c2-public": ["c2"],
            "c2-operational": [],
        }
        for suite, expected in cases.items():
            with self.subTest(suite=suite):
                result = run_script("--plan", suite)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(producers(result), expected)

    def test_unknown_suite_is_rejected(self) -> None:
        result = run_script("--plan", "not-a-suite")
        self.assertEqual(result.returncode, 2)
        self.assertIn("unknown CI suite: not-a-suite", result.stderr)

    def test_build_mode_validates_every_suite_before_cleaning(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            build_dir = Path(temporary)
            sentinel = build_dir / "must-survive"
            sentinel.write_text("sentinel\n", encoding="utf-8")
            result = run_script(
                "native-numerics", "not-a-suite", build_dir=build_dir
            )
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "sentinel\n")
        self.assertEqual(result.returncode, 2)
        self.assertIn("unknown CI suite: not-a-suite", result.stderr)
        self.assertNotIn("refusing to clean", result.stderr)

    def test_verify_only_rejects_missing_canonical_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = run_script(
                "--verify-only", "c2-public", build_dir=Path(temporary)
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("CI prerequisite artifact is missing or empty", result.stderr)

    def test_verify_only_accepts_complete_canonical_artifact_guard(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            build_dir = Path(temporary)
            required = [
                build_dir / "toolchain-manifest.tsv",
                build_dir / "c2" / "c2_wave2.o",
                build_dir / "c2" / "libeshkol_transformer_wave2.a",
                build_dir / "c2" / "c2_wave2.o.evidence" / "global-defined.txt",
            ]
            for artifact in required:
                artifact.parent.mkdir(parents=True, exist_ok=True)
                artifact.write_bytes(b"guard-fixture\n")
            result = run_script("--verify-only", "c2-public", build_dir=build_dir)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("CI prerequisite artifacts verified: c2", result.stdout)


if __name__ == "__main__":
    unittest.main()
