"""Contract tests for the standalone Wave 4/5 hosted CI matrix workflow."""

from __future__ import annotations

from collections import Counter
from itertools import product
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

from tests.ci.topology import check as check_wave03_topology, field, jobs, recipes, run, steps


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = ROOT / ".github/workflows/wave45-suites.yml"
MAKEFILE_PATH = ROOT / "Makefile"

PINNED_ACTIONS = {
    "actions/checkout": "11bd71901bbe5b1630ceea73d27597364c9af683",
    "actions/cache": "0057852bfaa89a56745cba8c7296529d2fc39830",
    "actions/upload-artifact": "ea165f8d65b6e75b540449e92b4886f43607fa02",
}

LLVM_GPG_FINGERPRINT = "6084F3CF814B57C1CF12EFD515CF4D18AF4F7421"

# Minimal, explicitly justified exclusions from the Wave 4/5 Makefile block
# (between test-io5 and test-c2). Every other target in that block MUST appear
# in exactly one matrix shard of .github/workflows/wave45-suites.yml.
EXPLICITLY_EXCLUDED_TARGETS = {
    "test-io5-gguf-native": (
        "requires external IO5_GGUF_NATIVE_PREFIX ggml witness library; "
        "enforced in .github/workflows/full-coverage.yml io5-gguf job"
    ),
    "benchmark-wave45": (
        "standalone timing benchmark runner; validated end-to-end by test-b0-wave45"
    ),
    "benchmark-wave5": (
        "standalone timing benchmark runner; validated end-to-end by test-b0-wave5"
    ),
}


def extract_wave45_makefile_targets(makefile_text: str) -> list[str]:
    """Return every Make target declared between test-io5 and test-c2."""
    start = makefile_text.index(".PHONY: test-io5\n")
    end = makefile_text.index("\ntest-c2: configure\n")
    section = makefile_text[start:end]
    return re.findall(r"^([a-z0-9-]+):(?!=)", section, re.M)


def parse_matrix_shards(workflow_text: str) -> list[tuple[str, list[str]]]:
    """Parse (shard_name, [target, ...]) entries from wave45-matrix."""
    matrix_job = jobs(workflow_text)["wave45-matrix"]
    entries = re.findall(
        r"^          - shard: ([a-z0-9-]+)\n            targets: '([^']*)'$",
        matrix_job,
        re.M,
    )
    return [(shard, targets.split()) for shard, targets in entries]


def parse_dispatch_shard_options(workflow_text: str) -> list[str]:
    """Parse the workflow_dispatch shard choice options."""
    header = workflow_text.split("\njobs:\n", 1)[0]
    match = re.search(
        r"^        options:\n((?:^          - [a-z0-9-]+\n)+)",
        header,
        re.M,
    )
    if not match:
        return []
    return re.findall(r"^          - ([a-z0-9-]+)$", match.group(1), re.M)


class Wave45WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.workflow_text = WORKFLOW_PATH.read_text(encoding="utf-8")
        cls.makefile_text = MAKEFILE_PATH.read_text(encoding="utf-8")
        cls.workflow_jobs = jobs(cls.workflow_text)
        cls.make_recipes = recipes(cls.makefile_text)
        cls.wave45_targets = extract_wave45_makefile_targets(cls.makefile_text)
        cls.shards = parse_matrix_shards(cls.workflow_text)

    def test_yaml_structure_and_optional_pyyaml_parse(self) -> None:
        self.assertNotIn("\t", self.workflow_text, "YAML must not contain tabs")
        for lineno, line in enumerate(self.workflow_text.splitlines(), 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            indent = len(line) - len(line.lstrip(" "))
            self.assertEqual(
                indent % 2,
                0,
                f"Unexpected odd indentation at line {lineno}: {line!r}",
            )
        header = self.workflow_text.split("\njobs:\n", 1)[0]
        for key in ("name:", "on:", "permissions:", "concurrency:"):
            self.assertRegex(header, re.compile(rf"^{re.escape(key)}", re.M))
        self.assertEqual(
            set(self.workflow_jobs),
            {"static-guards", "wave45-matrix", "wave45-gate"},
        )
        # Job-level if: expressions cannot reference matrix.* in GitHub Actions.
        for job_name, body in self.workflow_jobs.items():
            job_header = body.split("\n    steps:\n", 1)[0]
            for line in job_header.splitlines():
                if re.match(r"^    if:", line):
                    self.assertNotIn(
                        "matrix.",
                        line,
                        f"Job-level if in {job_name} cannot reference matrix.*",
                    )
        try:
            import yaml  # type: ignore[import-not-found]
        except ImportError:
            return
        parsed = yaml.safe_load(self.workflow_text)
        self.assertIsInstance(parsed, dict)
        self.assertIn("jobs", parsed)
        self.assertEqual(
            set(parsed["jobs"]),
            {"static-guards", "wave45-matrix", "wave45-gate"},
        )

    def test_exact_wave45_target_partition_and_makefile_alignment(self) -> None:
        self.assertGreaterEqual(len(self.shards), 8)
        self.assertLessEqual(len(self.shards), 10)
        shard_names = [name for name, _ in self.shards]
        self.assertEqual(len(shard_names), len(set(shard_names)), "Duplicate shard names")

        covered: list[str] = []
        for shard_name, targets in self.shards:
            self.assertTrue(targets, f"Shard {shard_name} must not be empty")
            covered.extend(targets)

        counts = Counter(covered)
        duplicates = sorted(t for t, c in counts.items() if c > 1)
        self.assertEqual(duplicates, [], f"Duplicated targets across shards: {duplicates}")

        covered_set = set(covered)
        excluded_set = set(EXPLICITLY_EXCLUDED_TARGETS)
        wave45_set = set(self.wave45_targets)

        self.assertEqual(
            len(self.wave45_targets),
            len(wave45_set),
            "Duplicate Wave 4/5 targets in Makefile",
        )
        self.assertEqual(
            covered_set & excluded_set,
            set(),
            "Excluded targets must not also appear in matrix shards",
        )
        self.assertTrue(
            excluded_set <= wave45_set,
            f"Stale exclusions not in Makefile: {sorted(excluded_set - wave45_set)}",
        )
        missing = sorted(wave45_set - covered_set - excluded_set)
        extra = sorted(covered_set - wave45_set)
        self.assertEqual(missing, [], f"Uncovered Wave 4/5 Makefile targets: {missing}")
        self.assertEqual(extra, [], f"Workflow targets missing from Wave 4/5 Makefile section: {extra}")

        # Every target in the workflow must exist in Makefile with a real recipe
        # and every referenced script must exist on disk.
        for target in covered:
            self.assertIn(target, self.make_recipes, f"Missing Makefile recipe for {target}")
            recipe_lines = self.make_recipes[target]
            self.assertTrue(recipe_lines, f"Empty Makefile recipe for {target}")
            for line in recipe_lines:
                for script in re.findall(r"scripts/([a-zA-Z0-9_.-]+\.sh)", line):
                    self.assertTrue(
                        (ROOT / "scripts" / script).is_file(),
                        f"Missing script scripts/{script} referenced by {target}",
                    )

        # Every explicit exclusion must also be documented in the workflow header.
        header = self.workflow_text.split("\njobs:\n", 1)[0]
        for excluded in sorted(excluded_set):
            self.assertIn(
                excluded,
                header,
                f"Excluded target {excluded} must be documented in workflow comments",
            )

    def test_dispatch_options_and_static_validator_match_shards(self) -> None:
        shard_names = [name for name, _ in self.shards]
        options = parse_dispatch_shard_options(self.workflow_text)
        self.assertEqual(options, ["all"] + shard_names)

        validate_step = steps(self.workflow_jobs["static-guards"])[
            "Validate shard dispatch input"
        ]
        validate_lines = "\n".join(run(validate_step))
        expected_case = "|".join(["all"] + shard_names) + ")"
        self.assertIn(expected_case, validate_lines)

        for requested in ["all"] + shard_names:
            proc = subprocess.run(
                ["/usr/bin/bash", "-euo", "pipefail", "-c", validate_lines],
                env={
                    **os.environ,
                    "EVENT_NAME": "workflow_dispatch",
                    "REQUESTED_SHARD": requested,
                },
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
        for bad_shard in ("unknown-shard", "", "core-wave4;true"):
            proc = subprocess.run(
                ["/usr/bin/bash", "-euo", "pipefail", "-c", validate_lines],
                env={
                    **os.environ,
                    "EVENT_NAME": "workflow_dispatch",
                    "REQUESTED_SHARD": bad_shard,
                },
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(proc.returncode, 0, f"Invalid shard accepted: {bad_shard!r}")

    def test_triggers_permissions_and_concurrency(self) -> None:
        header = self.workflow_text.split("\njobs:\n", 1)[0]
        self.assertIn("  pull_request:\n", header)
        self.assertIn("  push:\n", header)
        self.assertIn("  workflow_dispatch:\n", header)
        self.assertIn("      - main\n", header)
        self.assertIn("      - 'codex/wave4-5-*'\n", header)
        self.assertNotIn("pull_request_target", self.workflow_text)
        self.assertNotIn("continue-on-error", self.workflow_text)

        perm_block = re.search(
            r"^permissions:\n((?:^  [a-z-]+: [a-z]+\n)+)",
            header,
            re.M,
        )
        self.assertIsNotNone(perm_block)
        assert perm_block is not None
        self.assertEqual(perm_block.group(1), "  contents: read\n")

        self.assertIn(
            "  cancel-in-progress: ${{ github.event_name == 'pull_request' }}\n",
            header,
        )
        self.assertIn(
            "  group: ${{ github.workflow }}-${{ github.event.pull_request.number || github.ref }}\n",
            header,
        )

    def test_actions_are_pinned_to_repository_shas(self) -> None:
        uses_refs = re.findall(r"^\s*-?\s*uses:\s*([^\s#]+)$", self.workflow_text, re.M)
        self.assertTrue(uses_refs, "Workflow must use pinned GitHub Actions")
        seen_actions: set[str] = set()
        for ref in uses_refs:
            match = re.fullmatch(r"([a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+)@([0-9a-f]{40})", ref)
            self.assertIsNotNone(match, f"Action not pinned to 40-hex SHA: {ref}")
            assert match is not None
            action, sha = match.group(1), match.group(2)
            self.assertIn(action, PINNED_ACTIONS, f"Unapproved action: {action}")
            self.assertEqual(
                sha,
                PINNED_ACTIONS[action],
                f"Mismatched SHA for {action}: got {sha}, expected {PINNED_ACTIONS[action]}",
            )
            seen_actions.add(action)
        self.assertEqual(seen_actions, set(PINNED_ACTIONS))

    def test_toolchain_environment_evidence_and_timeouts(self) -> None:
        for job_name, body in self.workflow_jobs.items():
            self.assertEqual(
                field(body, 4, "runs-on"),
                "ubuntu-22.04",
                f"Job {job_name} must run on ubuntu-22.04",
            )

        matrix_job = self.workflow_jobs["wave45-matrix"]
        self.assertEqual(field(matrix_job, 4, "needs"), "static-guards")
        self.assertEqual(field(matrix_job, 4, "timeout-minutes"), "60")
        self.assertEqual(field(matrix_job, 6, "fail-fast"), "false")

        for env_key, env_val in {
            "CC": "clang-21",
            "CXX": "clang++-21",
            "ESHKOL_CXX_COMPILER": "clang++-21",
            "LLVM_CONFIG_EXECUTABLE": "llvm-config-21",
            "ESHKOL_JOBS": "'2'",
        }.items():
            self.assertEqual(field(matrix_job, 6, env_key), env_val)

        matrix_steps = steps(matrix_job)
        deps_body = matrix_steps["Install supported host dependencies"]
        self.assertIn(f"fpr:::::::::{LLVM_GPG_FINGERPRINT}:", deps_body)
        self.assertIn("llvm-toolchain-jammy-21 main", deps_body)
        self.assertIn("test \"$(llvm-config-21 --version)\" = '21.1.8'", deps_body)

        cache_body = matrix_steps["Restore pinned Eshkol toolchain"]
        self.assertIn(
            "key: eshkol-v1-${{ runner.os }}-${{ runner.arch }}-llvm-21.1.8-"
            "${{ hashFiles('toolchain/eshkol.lock', 'scripts/bootstrap-eshkol.sh', 'scripts/common.sh') }}",
            cache_body,
        )
        self.assertIn("path: .deps", cache_body)

        toolchain_body = matrix_steps["Build pinned Eshkol toolchain"]
        self.assertIn("make toolchain", toolchain_body)

        identity_body = matrix_steps["Record commit and tree identity"]
        self.assertIn("git rev-parse HEAD", identity_body)
        self.assertIn("git rev-parse 'HEAD^{tree}'", identity_body)

        seal_body = matrix_steps["Seal shard evidence"]
        self.assertIn("always()", field(seal_body, 8, "if"))
        self.assertIn("SHA256SUMS", seal_body)
        self.assertIn("sha256sum -c SHA256SUMS", seal_body)

        upload_body = matrix_steps["Upload shard evidence"]
        self.assertIn("always()", field(upload_body, 8, "if"))
        self.assertEqual(field(upload_body, 10, "retention-days"), "14")
        self.assertEqual(field(upload_body, 10, "if-no-files-found"), "error")

        static_steps = steps(self.workflow_jobs["static-guards"])
        guard_cmds = run(static_steps["Run Python isolation and CI topology guards"])
        self.assertIn("make test-python-isolation", guard_cmds)
        self.assertIn("make test-ci-topology", guard_cmds)
        self.assertIn("python3 -m unittest -v tests.ci.test_wave45_workflow", guard_cmds)

    def test_all_shell_steps_pass_bash_syntax_check(self) -> None:
        for job_name, body in self.workflow_jobs.items():
            for step_name, step_body in steps(body).items():
                if "run:" not in step_body:
                    continue
                self.assertEqual(
                    field(step_body, 8, "shell"),
                    "/usr/bin/bash -euo pipefail {0}",
                    f"{job_name}/{step_name} must use strict bash shell",
                )
                script = "\n".join(run(step_body)) + "\n"
                proc = subprocess.run(
                    ["/usr/bin/bash", "-n"],
                    input=script,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(
                    proc.returncode,
                    0,
                    f"bash -n failed for {job_name}/{step_name}: {proc.stderr}",
                )

    def test_wave45_gate_fails_closed(self) -> None:
        gate = self.workflow_jobs["wave45-gate"]
        self.assertEqual(field(gate, 4, "needs"), "[static-guards, wave45-matrix]")
        self.assertEqual(field(gate, 4, "if"), "${{ always() }}")
        command = "\n".join(run(steps(gate)["Require static guards and matrix success"]))
        statuses = ("success", "failure", "cancelled", "skipped", "")
        with tempfile.TemporaryDirectory() as directory:
            summary = Path(directory) / "summary.md"
            for static_res, matrix_res in product(statuses, repeat=2):
                with self.subTest(static=static_res, matrix=matrix_res):
                    summary.write_text("", encoding="utf-8")
                    proc = subprocess.run(
                        ["/usr/bin/bash", "-euo", "pipefail", "-c", command],
                        env={
                            **os.environ,
                            "STATIC_RESULT": static_res,
                            "MATRIX_RESULT": matrix_res,
                            "EVENT_NAME": "pull_request",
                            "REQUESTED_SHARD": "all",
                            "GITHUB_STEP_SUMMARY": str(summary),
                        },
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    expected_ok = static_res == "success" and matrix_res == "success"
                    self.assertEqual(proc.returncode == 0, expected_ok)

    def test_shard_runner_records_logs_and_fails_on_target_error(self) -> None:
        matrix_steps = steps(self.workflow_jobs["wave45-matrix"])
        runner_script = "\n".join(run(matrix_steps["Run Wave 4/5 shard targets"]))
        seal_script = "\n".join(run(matrix_steps["Seal shard evidence"]))

        for should_fail in (False, True):
            with self.subTest(should_fail=should_fail), tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                fake_bin = tmp_path / "bin"
                fake_bin.mkdir()
                fake_make = fake_bin / "make"
                fake_make.write_text(
                    "#!/usr/bin/bash\n"
                    "echo \"running $1\"\n"
                    "if [[ \"$1\" == \"target-fail\" ]]; then\n"
                    "  echo \"boom\" >&2\n"
                    "  exit 42\n"
                    "fi\n"
                    "exit 0\n",
                    encoding="utf-8",
                )
                fake_make.chmod(0o755)
                evidence_root = tmp_path / "wave45-evidence" / "mock-shard"
                evidence_root.mkdir(parents=True)
                (evidence_root / "workflow-status.tsv").write_text(
                    "state\treserved\n", encoding="utf-8"
                )
                summary = tmp_path / "summary.md"
                summary.write_text("", encoding="utf-8")
                targets = "target-ok target-fail target-after" if should_fail else "target-ok target-after"
                env = {
                    **os.environ,
                    "PATH": f"{fake_bin}:{os.environ['PATH']}",
                    "RUNNER_TEMP": str(tmp_path),
                    "SHARD_NAME": "mock-shard",
                    "SHARD_TARGETS": targets,
                    "GITHUB_STEP_SUMMARY": str(summary),
                    "PREVIOUS_JOB_STATUS": "failure" if should_fail else "success",
                }
                proc = subprocess.run(
                    ["/usr/bin/bash", "-euo", "pipefail", "-c", runner_script],
                    env=env,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(proc.returncode != 0, should_fail)
                results = (evidence_root / "results.tsv").read_text(encoding="utf-8")
                self.assertIn("target-ok\t0\t", results)
                self.assertIn("target-after\t0\t", results)
                if should_fail:
                    self.assertIn("target-fail\t42\t", results)
                    self.assertIn("FAIL: target-fail", proc.stderr)
                else:
                    self.assertNotIn("FAIL:", proc.stderr)

                seal = subprocess.run(
                    ["/usr/bin/bash", "-euo", "pipefail", "-c", seal_script],
                    env=env,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(seal.returncode, 0, seal.stderr)
                sums = (evidence_root / "SHA256SUMS").read_text(encoding="utf-8")
                self.assertIn("./results.tsv", sums)
                self.assertIn("./target-ok.log", sums)

    def test_existing_wave03_ci_topology_untouched(self) -> None:
        self.assertEqual(check_wave03_topology(ROOT), 23)


if __name__ == "__main__":
    unittest.main()
