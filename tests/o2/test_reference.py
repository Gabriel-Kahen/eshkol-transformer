from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
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
                "adamw_multiple_parameter_groups",
                "global_l2_clip_above",
                "global_l2_clip_boundary",
                "schedule_constant_successful_updates",
                "schedule_linear_warmup_decay_clamp",
                "weighted_microbatch_accumulation",
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

    def test_generation_is_byte_identical_in_fresh_processes(self) -> None:
        python = os.environ.get("O2_ORACLE_PYTHON", sys.executable)
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
