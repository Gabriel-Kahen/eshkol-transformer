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
                          "rope.boundary.forward", "rope.boundary.gradient"])

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
