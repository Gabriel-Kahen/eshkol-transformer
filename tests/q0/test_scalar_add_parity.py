from __future__ import annotations

import os
import re
import shutil
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

from tests.q0.numerics import (
    ComparisonPolicy,
    TensorMetadata,
    compare_values,
)
from tests.q0.oracle_format import decode_tensor, load_fixture, tensor_by_name

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "fixtures" / "scalar_add_v1.json"
PROBE = HERE / "probes" / "scalar_add.esk"
_SCALAR = re.compile(
    r"[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|inf(?:inity)?|nan)\Z",
    re.IGNORECASE,
)
_VERSION = re.compile(r"Eshkol Compiler v(\d+)\.(\d+)\.(\d+)(?:[-+][^\s]+)?")
_MINIMUM_VERSION = (1, 3, 4)


def _tolerance(bits: str) -> float:
    return struct.unpack(">d", bytes.fromhex(bits))[0]


def _parse_scalar_output(stdout: str) -> float:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if len(lines) != 1 or _SCALAR.fullmatch(lines[0]) is None:
        raise ValueError(f"expected exactly one scalar output, found {stdout!r}")
    return float(lines[0])


class CompiledEshkolParityTests(unittest.TestCase):
    def test_scalar_add_matches_frozen_pytorch_oracle(self) -> None:
        explicit = os.environ.get("ESHKOL_RUN")
        runner = explicit or shutil.which("eshkol-run")
        if runner is None:
            self.skipTest(
                "BLOCKED: compiled Eshkol runner unavailable; set ESHKOL_RUN to "
                "the real eshkol-run executable. No Python substitute was used."
            )
        if explicit and (
            not Path(explicit).is_file() or not os.access(explicit, os.X_OK)
        ):
            self.fail(f"BLOCKED: ESHKOL_RUN is not an executable file: {explicit}")

        identified = subprocess.run(
            [runner, "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
        version_output = identified.stdout + identified.stderr
        match = _VERSION.search(version_output)
        if match is None:
            self.fail(
                "BLOCKED: could not identify the Eshkol compiler version; "
                f"exit={identified.returncode}; output={version_output!r}"
            )
        version = tuple(int(part) for part in match.groups())
        if version < _MINIMUM_VERSION:
            self.fail(
                "BLOCKED: Eshkol compiler "
                f"{'.'.join(map(str, version))} is below required 1.3.4; "
                "no incompatible runner or Python substitute was used"
            )

        with tempfile.TemporaryDirectory() as temporary:
            binary = Path(temporary) / "scalar-add"
            compiled = subprocess.run(
                [runner, "-o", str(binary), str(PROBE)],
                cwd=temporary,
                capture_output=True,
                text=True,
                check=False,
            )
            if compiled.returncode != 0:
                self.fail(
                    "BLOCKED: compatible Eshkol compilation failed with "
                    f"exit={compiled.returncode}; stdout={compiled.stdout!r}; "
                    f"stderr={compiled.stderr!r}"
                )
            if not binary.is_file() or not os.access(binary, os.X_OK):
                self.fail(
                    "BLOCKED: compatible Eshkol compilation returned success but "
                    f"did not emit an executable at {binary}"
                )
            executed = subprocess.run(
                [str(binary)],
                cwd=temporary,
                capture_output=True,
                text=True,
                check=False,
            )
            if executed.returncode != 0:
                self.fail(
                    "BLOCKED: compiled Eshkol executable failed with "
                    f"exit={executed.returncode}; stdout={executed.stdout!r}; "
                    f"stderr={executed.stderr!r}"
                )
            try:
                actual_value = _parse_scalar_output(executed.stdout)
            except ValueError as error:
                self.fail(str(error))
                raise AssertionError from error

        payload = load_fixture(FIXTURE)
        expected_tensor = tensor_by_name(payload, "output.sum")
        expected_values = decode_tensor(expected_tensor)
        tolerance = payload["cases"][0]["tolerance"]
        metadata = TensorMetadata(
            expected_tensor["name"],
            tuple(expected_tensor["shape"]),
            expected_tensor["dtype"],
            expected_tensor["device"],
        )
        result = compare_values(
            [actual_value],
            expected_values,
            actual=metadata,
            expected=metadata,
            policy=ComparisonPolicy(
                _tolerance(tolerance["absolute"]),
                _tolerance(tolerance["relative"]),
                tolerance["equal_nan"],
            ),
        )
        self.assertTrue(result.passed, result.error)

    def test_scalar_output_parser_rejects_empty_and_non_scalar_output(self) -> None:
        for output in ("", "\n", "3.75 4.0\n", "3.75\n4.0\n", "value=3.75\n"):
            with self.subTest(output=output):
                with self.assertRaisesRegex(ValueError, "exactly one scalar"):
                    _parse_scalar_output(output)
        self.assertEqual(_parse_scalar_output(" 3.75\n"), 3.75)


if __name__ == "__main__":
    unittest.main()
