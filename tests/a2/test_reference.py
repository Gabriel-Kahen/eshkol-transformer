from __future__ import annotations

import hashlib
import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

from tests.q0.oracle_format import decode_tensor, load_fixture, tensor_by_name

HERE = Path(__file__).resolve().parent
GENERATOR = HERE / "generate_reference.py"
FIXTURE = HERE / "fixtures" / "a2_attention_v1.json"
LOCK = HERE / "requirements-reference.lock"
HEADER = HERE / "reference_vectors.h"


class A2ReferenceTests(unittest.TestCase):
    def test_fixture_identity_and_roles(self) -> None:
        payload = load_fixture(FIXTURE)
        self.assertEqual(payload["generator"]["framework"],
                         {"name": "pytorch", "version": "2.13.0+cpu"})
        self.assertEqual(payload["generator"]["source_sha256"],
                         hashlib.sha256(GENERATOR.read_bytes()).hexdigest())
        self.assertEqual(payload["generator"]["dependency_lock_sha256"],
                         hashlib.sha256(LOCK.read_bytes()).hexdigest())
        self.assertEqual([case["name"] for case in payload["cases"]],
                         ["attention.gqa.forward", "attention.gqa.gradient",
                          "attention.n2.forward", "attention.n2.gradient",
                          "cache.n2.incremental", "rope.boundary.forward",
                          "rope.boundary.gradient", "rope.n2.forward",
                          "rope.n2.gradient"])

    def test_fresh_generation_is_byte_identical(self) -> None:
        oracle_python = os.environ.get("A2_ORACLE_PYTHON")
        if oracle_python is None:
            self.skipTest("set A2_ORACLE_PYTHON to verify pinned PyTorch regeneration")
        with tempfile.TemporaryDirectory() as temporary:
            outputs = [Path(temporary) / "first.json", Path(temporary) / "second.json"]
            for output in outputs:
                subprocess.run([oracle_python, str(GENERATOR), "--output", str(output)],
                               cwd=HERE.parent.parent, check=True,
                               capture_output=True, text=True)
            self.assertEqual(outputs[0].read_bytes(), outputs[1].read_bytes())
            self.assertEqual(outputs[0].read_bytes(), FIXTURE.read_bytes())

    def test_fully_masked_row_and_rope_zero_position(self) -> None:
        payload = load_fixture(FIXTURE)
        forward = decode_tensor(tensor_by_name(payload, "gqa.output.forward"))
        # The final query row is fully masked for every head.
        for head in range(4):
            base = (head * 3 + 2) * 2
            self.assertEqual(forward[base:base + 2], (0.0, 0.0))
        rope_input = decode_tensor(tensor_by_name(payload, "rope.input.x"))
        rope_output = decode_tensor(tensor_by_name(payload, "rope.output.forward"))
        self.assertEqual(rope_output[:4], rope_input[:4])

    def test_n2_fixture_exercises_distinct_batches(self) -> None:
        payload = load_fixture(FIXTURE)
        for name in ("n2_attention.input.q", "n2_attention.input.k",
                     "n2_attention.input.v", "n2_attention.input.qpos",
                     "n2_attention.input.kpos", "n2_attention.input.mask",
                     "n2_attention.input.upstream", "n2_rope.input.x",
                     "n2_rope.input.positions", "n2_rope.input.upstream",
                     "n2_cache.input.q", "n2_cache.input.k", "n2_cache.input.v",
                     "n2_cache.input.positions"):
            values = decode_tensor(tensor_by_name(payload, name))
            half = len(values) // 2
            self.assertNotEqual(values[:half], values[half:], name)
        expected_shapes = {
            "n2_attention.output.forward": [2, 4, 2, 4],
            "n2_attention.gradient.dq": [2, 4, 2, 4],
            "n2_attention.gradient.dk": [2, 2, 3, 4],
            "n2_attention.gradient.dv": [2, 2, 3, 4],
            "n2_rope.output.forward": [2, 2, 3, 4],
            "n2_rope.gradient.dx": [2, 2, 3, 4],
            "n2_cache.output.incremental": [2, 3, 4, 2],
        }
        for name, shape in expected_shapes.items():
            record = tensor_by_name(payload, name)
            self.assertEqual(record["shape"], shape, name)
            values = decode_tensor(record)
            half = len(values) // 2
            self.assertNotEqual(values[:half], values[half:], name)
        for name in ("n2_attention.gradient.dq", "n2_attention.gradient.dk",
                     "n2_attention.gradient.dv", "n2_rope.gradient.dx"):
            values = decode_tensor(tensor_by_name(payload, name))
            half = len(values) // 2
            self.assertTrue(any(value != 0.0 for value in values[:half]), name)
            self.assertTrue(any(value != 0.0 for value in values[half:]), name)

    def test_c_vectors_are_bound_to_frozen_fixture(self) -> None:
        payload = load_fixture(FIXTURE)
        source = HEADER.read_text(encoding="utf-8")
        self.assertIn(hashlib.sha256(FIXTURE.read_bytes()).hexdigest(), source)
        mappings = {
            "et_a2_ref_gqa_output": "gqa.output.forward",
            "et_a2_ref_gqa_dq": "gqa.gradient.dq",
            "et_a2_ref_gqa_dk": "gqa.gradient.dk",
            "et_a2_ref_gqa_dv": "gqa.gradient.dv",
            "et_a2_ref_rope_output": "rope.output.forward",
            "et_a2_ref_rope_dx": "rope.gradient.dx",
            "et_a2_ref_n2_attention_output": "n2_attention.output.forward",
            "et_a2_ref_n2_attention_dq": "n2_attention.gradient.dq",
            "et_a2_ref_n2_attention_dk": "n2_attention.gradient.dk",
            "et_a2_ref_n2_attention_dv": "n2_attention.gradient.dv",
            "et_a2_ref_n2_rope_output": "n2_rope.output.forward",
            "et_a2_ref_n2_rope_dx": "n2_rope.gradient.dx",
            "et_a2_ref_n2_cache_incremental": "n2_cache.output.incremental",
        }
        for symbol, tensor_name in mappings.items():
            match = re.search(
                rf"{symbol}\[\]\s*=\s*\{{(?P<body>.*?)\}};", source, re.S
            )
            self.assertIsNotNone(match, symbol)
            actual = [
                f"{int(bits, 0):08x}"
                for bits in re.findall(r"UINT32_C\((0x[0-9a-f]+|0)\)",
                                       match.group("body"))
            ]
            expected = tensor_by_name(payload, tensor_name)["data"]
            self.assertEqual(actual, expected, symbol)


if __name__ == "__main__":
    unittest.main()
