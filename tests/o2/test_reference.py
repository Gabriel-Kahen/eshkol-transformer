from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from tests.q0.oracle_format import load_fixture

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "fixtures" / "adamw_v1.json"
GENERATOR = HERE / "generate_adamw.py"
LOCK = ROOT / "tests" / "q0" / "requirements-oracle.lock"


class O2ReferenceTests(unittest.TestCase):
    def test_frozen_fixture_contract_and_identity(self) -> None:
        payload = load_fixture(FIXTURE)
        self.assertEqual(
            [case["name"] for case in payload["cases"]],
            [
                "adamw_bias_correction_decay_two_steps",
                "adamw_bias_correction_later_step",
                "adamw_multiple_parameter_groups",
                "adamw_present_zero_gradient_decay",
                "global_l2_clip_above",
                "global_l2_clip_boundary",
                "global_l2_clip_multiple_tensors",
                "gradient_explicit_zero_after_step",
                "optimizer_state_continuation_identical_next_update",
                "schedule_constant_successful_updates",
                "schedule_failed_attempt_does_not_advance",
                "schedule_linear_warmup_decay_clamp",
                "schedule_linear_zero_warmup",
                "weighted_microbatch_accumulation",
                "weighted_microbatch_contribution_count",
            ],
        )
        generator = payload["generator"]
        self.assertEqual(
            generator["framework"],
            {"name": "pytorch", "version": "2.13.0+cpu"},
        )
        self.assertEqual(
            generator["source_sha256"],
            hashlib.sha256(GENERATOR.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            generator["dependency_lock_sha256"],
            hashlib.sha256(LOCK.read_bytes()).hexdigest(),
        )

    def test_fixture_values_cover_required_boundaries(self) -> None:
        payload = load_fixture(FIXTURE)
        tensors = {tensor["name"]: tensor for tensor in payload["tensors"]}
        cases = {case["name"]: case for case in payload["cases"]}
        self.assertEqual(
            tensors["single.input.group_options"]["data"],
            ["3c23d70a", "3f666666", "3f7fbe77", "322bcc77", "3dcccccd"],
        )
        self.assertEqual(
            tensors["single.output.step2.exp_avg"]["data"],
            ["3c23d70c", "bca3d70c", "3dccccd0"],
        )
        self.assertEqual(
            tensors["later.input.update_indices"]["data"][-1],
            "0000000000000007",
        )
        self.assertIn(
            "later.output.step7.exp_avg",
            cases["adamw_bias_correction_later_step"]["expectation"]["outputs"],
        )
        self.assertEqual(
            tensors["clip.output.boundary.norm"]["data"], ["40a00000"]
        )
        self.assertEqual(
            tensors["clip.output.boundary.gradient"]["data"],
            tensors["clip.input.boundary.gradient"]["data"],
        )
        self.assertNotEqual(
            tensors["clip.output.above.gradient"]["data"],
            tensors["clip.input.above.gradient"]["data"],
        )
        self.assertEqual(tensors["clip_multi.output.norm"]["data"], ["41500000"])
        self.assertEqual(
            tensors["clip_multi.output.first_gradient"]["data"],
            ["3fc00000", "40000000"],
        )
        self.assertEqual(
            tensors["clip_multi.output.second_gradient"]["data"],
            ["00000000", "40c00000", "80000000"],
        )
        self.assertEqual(
            tensors["clip_multi.input.alias_to_unique"]["data"],
            [
                "0000000000000000",
                "0000000000000000",
                "0000000000000001",
            ],
        )
        self.assertEqual(
            tensors["accumulation.output.total_weight"]["data"], ["40400000"]
        )
        self.assertIn(
            "accumulation.input.numerator_gradients",
            cases["weighted_microbatch_accumulation"]["inputs"],
        )
        self.assertIn(
            "accumulation.output.next_parameter",
            cases["weighted_microbatch_accumulation"]["expectation"]["outputs"],
        )
        self.assertNotEqual(
            tensors["accumulation.input.naive_microbatch_mean_anti_oracle"]["data"],
            tensors["accumulation.output.effective_gradient"]["data"],
        )
        self.assertEqual(
            tensors["accumulation.input.contribution_ordinals"]["data"],
            ["0000000000000000", "0000000000000001"],
        )
        self.assertEqual(
            tensors["accumulation.output.contribution_count"]["data"],
            ["0000000000000002"],
        )
        self.assertEqual(
            tensors["schedule.input.linear_scalars"]["data"],
            ["3dcccccd", "3ca3d70a"],
        )
        self.assertEqual(
            tensors["schedule.output.constant_factors"]["data"],
            ["3f800000"] * 8,
        )
        self.assertEqual(
            tensors["schedule.output.linear_factors"]["data"][-2:],
            ["3dcccccd", "3dcccccd"],
        )
        self.assertEqual(
            tensors["schedule.zero_warmup.output.factors"]["data"],
            [
                "3f500000",
                "3f200000",
                "3ee00000",
                "3e800000",
                "3e800000",
                "3e800000",
            ],
        )
        self.assertEqual(
            tensors["schedule.input.attempted_update_indices"]["data"],
            [
                "0000000000000001",
                "0000000000000002",
                "0000000000000002",
                "0000000000000002",
                "0000000000000003",
            ],
        )
        self.assertEqual(
            tensors["schedule.output.attempted_factors"]["data"],
            ["3f000000", "3f800000", "3f800000", "3f800000", "3f466666"],
        )

    def test_zero_gradient_and_state_continuation_contracts(self) -> None:
        payload = load_fixture(FIXTURE)
        tensors = {tensor["name"]: tensor for tensor in payload["tensors"]}
        cases = {case["name"]: case for case in payload["cases"]}
        positive_zeros = ["00000000"] * 4
        self.assertEqual(tensors["zero.output.exp_avg"]["data"], positive_zeros)
        self.assertEqual(tensors["zero.output.exp_avg_sq"]["data"], positive_zeros)
        self.assertNotEqual(
            tensors["zero.input.parameter"]["data"],
            tensors["zero.output.parameter"]["data"],
        )
        self.assertEqual(
            tensors["zero.input.gradient_present"]["data"],
            ["0000000000000001"],
        )
        self.assertEqual(
            tensors["single.output.step2.retained_gradient"]["data"],
            ["be000000", "3e800000", "3f800000"],
        )
        self.assertEqual(
            tensors["zeroing.output.explicit_zero_gradient"]["data"],
            ["00000000", "00000000", "00000000"],
        )
        self.assertEqual(
            tensors["continuation.input.completed_updates"]["data"],
            ["0000000000000003"],
        )
        self.assertEqual(
            tensors["continuation.input.next_update_index"]["data"],
            ["0000000000000004"],
        )
        self.assertEqual(
            tensors["continuation.input.commit_count"]["data"],
            ["0000000000000001"],
        )
        continuation = cases[
            "optimizer_state_continuation_identical_next_update"
        ]
        self.assertIn(
            "continuation.input.exp_avg",
            continuation["inputs"],
        )
        self.assertIn(
            "continuation.input.exp_avg_sq",
            continuation["inputs"],
        )
        for field in ("exp_avg", "exp_avg_sq", "parameter"):
            self.assertEqual(
                tensors[f"continuation.output.source.{field}"]["data"],
                tensors[f"continuation.output.restored.{field}"]["data"],
            )
        self.assertEqual(
            tensors["continuation.output.effective_learning_rate"]["data"],
            ["3c449ba6"],
        )
        self.assertIn(
            "continuation.output.restored.parameter",
            continuation["expectation"]["outputs"],
        )

    def test_generation_is_byte_identical_in_fresh_processes(self) -> None:
        python = os.environ.get("O2_ORACLE_PYTHON")
        if python is None:
            self.skipTest("O2_ORACLE_PYTHON pinned PyTorch environment not supplied")
        self.assertTrue(Path(python).is_file())
        self.assertTrue(os.access(python, os.X_OK))
        version = subprocess.run(
            [python, "-c", "import torch; print(torch.__version__)"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(version.stdout.strip(), "2.13.0+cpu")
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first.json"
            second = Path(temporary) / "second.json"
            for output in (first, second):
                subprocess.run(
                    [python, "-m", "tests.o2.generate_adamw", "--output", str(output)],
                    cwd=ROOT,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(first.read_bytes(), FIXTURE.read_bytes())

    def test_fixture_contains_no_executable_payload(self) -> None:
        document = json.loads(FIXTURE.read_text(encoding="utf-8"))
        serialized = json.dumps(document, sort_keys=True).lower()
        for forbidden in ("pickle", "torch.load", "eval(", "exec(", "importlib"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, serialized)


if __name__ == "__main__":
    unittest.main()
