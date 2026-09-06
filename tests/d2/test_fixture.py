from __future__ import annotations

import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests.q0.oracle_format import load_fixture


HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "fixtures" / "shifted_batch_v1.json"
GENERATOR = HERE / "generate_batch_fixture.py"
LOCK = HERE / "requirements-oracle.lock"


class D2FixtureTests(unittest.TestCase):
    def test_frozen_q0_fixture_identity_and_exact_values(self) -> None:
        payload = load_fixture(FIXTURE)
        generator = payload["generator"]
        self.assertEqual(generator["framework"], {"name": "python-stdlib", "version": "3"})
        self.assertEqual(generator["source_sha256"], hashlib.sha256(GENERATOR.read_bytes()).hexdigest())
        self.assertEqual(generator["dependency_lock_sha256"], hashlib.sha256(LOCK.read_bytes()).hexdigest())
        records = {record["name"]: record for record in payload["tensors"]}
        self.assertEqual(records["output.inputs"]["shape"], [2, 4])
        self.assertEqual(
            records["output.inputs"]["data"],
            ["000000000000000a", "000000000000000b", "000000000000000c", "000000000000000d",
             "000000000000000e", "0000000000000000", "0000000000000000", "0000000000000000"],
        )
        self.assertEqual(
            records["output.targets"]["data"],
            ["000000000000000b", "000000000000000c", "000000000000000d", "000000000000000e",
             "000000000000000f", "0000000000000000", "0000000000000000", "0000000000000000"],
        )
        self.assertEqual(records["output.loss_mask"]["data"], ["1", "1", "1", "1", "1", "0", "0", "0"])

    def test_fresh_generations_are_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            outputs = [Path(temporary) / "first.json", Path(temporary) / "second.json"]
            for output in outputs:
                subprocess.run(
                    [sys.executable, "-m", "tests.d2.generate_batch_fixture", "--output", str(output)],
                    cwd=HERE.parents[1],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            self.assertEqual(outputs[0].read_bytes(), outputs[1].read_bytes())
            self.assertEqual(outputs[0].read_bytes(), FIXTURE.read_bytes())


if __name__ == "__main__":
    unittest.main()
