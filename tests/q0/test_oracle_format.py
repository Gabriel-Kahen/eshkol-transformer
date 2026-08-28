from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests.q0.oracle_format import (
    FixtureError,
    canonical_json,
    encode_fixture,
    load_fixture,
)

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "fixtures" / "scalar_add_v1.json"
GENERATOR = HERE / "generate_scalar_add.py"
LOCK = HERE / "requirements-oracle.lock"


def _document() -> dict[str, object]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _resign(document: dict[str, object]) -> bytes:
    integrity = {
        "format": document["format"],
        "payload": document["payload"],
        "version": document["version"],
    }
    document["checksum"] = {
        "algorithm": "sha256",
        "digest": hashlib.sha256(canonical_json(integrity)).hexdigest(),
    }
    return canonical_json(document) + b"\n"


class OracleFormatTests(unittest.TestCase):
    def test_frozen_fixture_loads_and_generator_identity_matches(self) -> None:
        payload = load_fixture(FIXTURE)
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

    def test_generation_is_byte_identical_in_fresh_processes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first.json"
            second = Path(temporary) / "second.json"
            for output in (first, second):
                subprocess.run(
                    [sys.executable, str(GENERATOR), "--output", str(output)],
                    cwd=HERE,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(first.read_bytes(), FIXTURE.read_bytes())

    def test_payload_and_checksum_corruption_are_rejected(self) -> None:
        document = _document()
        document["payload"]["tensors"][2]["data"][0] = "400c000000000000"
        with self.assertRaisesRegex(FixtureError, "checksum mismatch"):
            load_fixture(canonical_json(document) + b"\n")

        document = _document()
        document["checksum"]["digest"] = "0" * 64
        with self.assertRaisesRegex(FixtureError, "checksum mismatch"):
            load_fixture(canonical_json(document) + b"\n")

    def test_valid_checksum_with_unsupported_version_is_rejected(self) -> None:
        document = _document()
        document["version"] = 2
        with self.assertRaisesRegex(FixtureError, "unsupported fixture version 2"):
            load_fixture(_resign(document))

        document = _document()
        document["version"] = True
        with self.assertRaisesRegex(FixtureError, r"\$\.version: expected integer"):
            load_fixture(_resign(document))

    def test_dtype_shape_device_and_unknown_key_are_rejected(self) -> None:
        mutations = (
            ("dtype", "float16", "unsupported dtype"),
            ("shape", [2], "does not match shape"),
            ("device", "cuda", "only explicit cpu"),
        )
        for field, value, message in mutations:
            with self.subTest(field=field):
                document = _document()
                document["payload"]["tensors"][0][field] = value
                with self.assertRaisesRegex(FixtureError, message):
                    load_fixture(_resign(document))
        document = _document()
        document["payload"]["unknown"] = True
        with self.assertRaisesRegex(FixtureError, "unknown=.*unknown"):
            load_fixture(_resign(document))

    def test_duplicate_keys_and_noncanonical_bytes_are_rejected(self) -> None:
        raw = FIXTURE.read_bytes()
        duplicate = raw.replace(
            b'{"checksum":', b'{"format":"eshkol-oracle","checksum":', 1
        )
        with self.assertRaisesRegex(FixtureError, "duplicate object key"):
            load_fixture(duplicate)
        with self.assertRaisesRegex(FixtureError, "not canonical"):
            load_fixture(raw.replace(b",", b", ", 1))
        with self.assertRaisesRegex(FixtureError, "not canonical"):
            load_fixture(raw + b"\n")
        with self.assertRaisesRegex(FixtureError, "BOM"):
            load_fixture(b"\xef\xbb\xbf" + raw)

    def test_lone_surrogates_are_reported_as_fixture_errors(self) -> None:
        with self.assertRaisesRegex(FixtureError, "Unicode scalar text"):
            canonical_json({"text": "\ud800"})
        with self.assertRaisesRegex(FixtureError, "Unicode scalar text"):
            load_fixture(b'{"text":"\\ud800"}\n')

    def test_case_tensor_roles_are_enforced(self) -> None:
        document = _document()
        document["payload"]["cases"][0]["inputs"] = ["output.sum"]
        with self.assertRaisesRegex(FixtureError, "expected 'input'"):
            load_fixture(_resign(document))

        document = _document()
        document["payload"]["cases"][0]["expectation"]["outputs"] = ["input.lhs"]
        with self.assertRaisesRegex(FixtureError, "expected 'expected'"):
            load_fixture(_resign(document))

        document = _document()
        document["payload"]["cases"][0]["kind"] = "gradient"
        with self.assertRaisesRegex(FixtureError, "expected 'analytic_gradient'"):
            load_fixture(_resign(document))

    def test_unhashable_tensor_references_are_fixture_errors(self) -> None:
        for field in ("inputs", "outputs"):
            with self.subTest(field=field):
                document = _document()
                if field == "inputs":
                    document["payload"]["cases"][0]["inputs"] = [[]]
                    message = r"inputs\[0\]: expected string"
                else:
                    document["payload"]["cases"][0]["expectation"]["outputs"] = [
                        []
                    ]
                    message = r"outputs\[0\]: expected string"
                with self.assertRaisesRegex(FixtureError, message):
                    load_fixture(_resign(document))

    def test_integer_and_boolean_outputs_require_exact_tolerance(self) -> None:
        document = _document()
        output = document["payload"]["tensors"][2]
        output.update(
            {
                "data": ["0000000000000003"],
                "dtype": "int64",
                "encoding": "twos-complement-hex-be",
            }
        )
        document["payload"]["cases"][0]["tolerance"]["absolute"] = (
            "3ff0000000000000"
        )
        with self.assertRaisesRegex(FixtureError, "requires zero tolerance"):
            load_fixture(_resign(document))

        document["payload"]["cases"][0]["tolerance"]["absolute"] = (
            "0000000000000000"
        )
        document["payload"]["cases"][0]["tolerance"]["equal_nan"] = True
        with self.assertRaisesRegex(FixtureError, "equal_nan=false"):
            load_fixture(_resign(document))

    def test_all_case_kinds_and_empty_tensor_are_expressible(self) -> None:
        source = load_fixture(FIXTURE)
        for kind in (
            "known_value",
            "parity",
            "gradient",
            "repeated_input",
            "special_value",
            "boundary",
            "malformed",
        ):
            with self.subTest(kind=kind):
                payload = copy.deepcopy(source)
                payload["cases"][0]["kind"] = kind
                if kind == "malformed":
                    payload["cases"][0]["expectation"] = {
                        "error": "invalid test input",
                        "outputs": [],
                    }
                elif kind == "gradient":
                    payload["tensors"][2]["role"] = "analytic_gradient"
                elif kind == "repeated_input":
                    payload["tensors"][1]["data"] = payload["tensors"][0]["data"]
                elif kind == "special_value":
                    payload["tensors"][0]["data"] = ["7ff8000000000000"]
                    payload["tensors"][1]["data"] = ["7ff0000000000000"]
                    payload["tensors"][2]["data"] = ["7ff8000000000000"]
                    payload["cases"][0]["tolerance"]["equal_nan"] = True
                elif kind == "boundary":
                    payload["tensors"][0]["data"] = ["7fefffffffffffff"]
                    payload["tensors"][1]["data"] = ["0000000000000001"]
                self.assertEqual(load_fixture(encode_fixture(payload)), payload)

        payload = copy.deepcopy(source)
        empty = payload["tensors"][2]
        empty["shape"] = [0]
        empty["data"] = []
        self.assertEqual(load_fixture(encode_fixture(payload)), payload)


if __name__ == "__main__":
    unittest.main()
