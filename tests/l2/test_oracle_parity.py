from __future__ import annotations

import hashlib
import struct
import subprocess
import sys
import unittest
from pathlib import Path

from tests.q0.numerics import ComparisonPolicy, TensorMetadata, compare_values
from tests.q0.oracle_format import decode_tensor, load_fixture, tensor_by_name

if len(sys.argv) != 2:
    raise RuntimeError("pass the native oracle runner path")
NATIVE_RUNNER = Path(sys.argv.pop())


class L2OracleParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = NATIVE_RUNNER
        cls.payload = load_fixture(
            Path(__file__).parents[1] / "q0" / "fixtures" / "indexed_cross_entropy_v1.json"
        )

    def _native(self) -> dict[str, tuple[float, ...]]:
        completed = subprocess.run(
            [self.runner], check=True, capture_output=True, text=True, timeout=30
        )
        self.assertEqual(completed.stderr, "")
        result: dict[str, tuple[float, ...]] = {}
        for line in completed.stdout.splitlines():
            fields = line.split()
            result[fields[0]] = tuple(
                struct.unpack(">f", bytes.fromhex(value))[0] for value in fields[1:]
            )
        self.assertEqual(set(result), {"loss", "gradient"})
        return result

    def test_forward_and_direct_backward_match_frozen_pytorch(self) -> None:
        native = self._native()
        cases = {case["name"]: case for case in self.payload["cases"]}
        self.assertEqual(
            cases["indexed_cross_entropy_forward"]["operation"],
            "indexed-cross-entropy.forward",
        )
        self.assertEqual(
            cases["indexed_cross_entropy_backward"]["operation"],
            "indexed-cross-entropy.backward",
        )
        loss = tensor_by_name(self.payload, "output.loss")
        gradient = tensor_by_name(self.payload, "output.gradient")
        self.assertTrue(
            compare_values(
                native["loss"],
                decode_tensor(loss),
                actual=TensorMetadata("native.loss", (1, 2), "float32", "cpu"),
                expected=TensorMetadata("oracle.loss", (1, 2), "float32", "cpu"),
                policy=ComparisonPolicy(2.0e-6, 2.0e-5),
            ).passed
        )
        self.assertTrue(
            compare_values(
                native["gradient"],
                decode_tensor(gradient),
                actual=TensorMetadata("native.gradient", (1, 2, 3), "float32", "cpu"),
                expected=TensorMetadata("oracle.gradient", (1, 2, 3), "float32", "cpu"),
                policy=ComparisonPolicy(3.0e-6, 3.0e-5),
            ).passed
        )

    def test_native_output_is_bitwise_deterministic(self) -> None:
        first = subprocess.run([self.runner], check=True, capture_output=True, timeout=30)
        second = subprocess.run([self.runner], check=True, capture_output=True, timeout=30)
        self.assertEqual(first.stdout, second.stdout)
        self.assertEqual(first.stderr, second.stderr)

    def test_frozen_generator_and_lock_identities_match_sources(self) -> None:
        q0 = Path(__file__).parents[1] / "q0"
        generator = q0 / "generate_indexed_cross_entropy.py"
        lock = q0 / "requirements-oracle.lock"
        metadata = self.payload["generator"]
        self.assertEqual(metadata["source_sha256"], hashlib.sha256(generator.read_bytes()).hexdigest())
        self.assertEqual(metadata["dependency_lock_sha256"], hashlib.sha256(lock.read_bytes()).hexdigest())
        self.assertEqual(metadata["framework"], {"name": "pytorch", "version": "2.13.0+cpu"})


if __name__ == "__main__":
    unittest.main()
