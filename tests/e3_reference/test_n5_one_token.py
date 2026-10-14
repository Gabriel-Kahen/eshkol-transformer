"""Offline N5 final-token causality, resource integrity, and result binding."""

from __future__ import annotations

import copy
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

from tests.c2.test_handler_order import parse as parse_source
from tests.e3_private.parse_n5_one_token import parse_one_token
from tests.e3_reference.n5_one_token import (
    CONFIG, NAME, ROW, build_reference, materialize, validate_manifest,
    validate_materialized, validate_observed,
)
from tests.e3_reference.transcript import canonical, decode, f32_bits


def synthetic_frame(case: dict[str, object], cursor: str) -> bytes:
    """Fixture for parser negatives; never labeled a compiled observation."""
    batch = case["synthetic"]["batches"][0]
    mathematical = case["mathematical"][0]
    fields = lambda values: ",".join(str(int(value)) for value in values)
    lines = ["E3ROW\t0\t" + "\t".join((fields(batch["inputs"]),
                                             fields(batch["targets"]),
                                             fields(batch["mask"])))]
    for role in mathematical["roles"]:
        words = [f32_bits(struct.unpack(">d", bytes.fromhex(word))[0])
                 for word in role["values_f64"]]
        lines.append("E3ROLE\t0\t" + role["name"] + "\t" +
                     fields(role["shape"]) + "\t" + ",".join(words))
    lines.extend((
        "E3CE\t0\t" + ",".join(batch["ce_f32"]),
        "E3BATCH\t0\t" + ",".join(batch["reduction_f32"] +
                                    [batch["correct_f32"]]) + "\t1",
        "E3STATE\t0\t" + ",".join(batch["state_f32"]) + "\t" +
        fields(batch["counters"]),
        "E3FINAL\t" + ",".join(case["synthetic"]["final"]["metrics_f32"]) +
        "\t" + fields(case["synthetic"]["final"]["counters"]),
    ))
    words = [int(word, 16) for word in case["synthetic"]["final"]["metrics_f32"]]
    words.extend(case["synthetic"]["final"]["counters"])
    lines.extend(("E3RESULT\t" + fields(words), "E3CURSOR\t" + cursor +
                  "\t" + cursor, "E3-NUMERICAL-RUNTIME-PASS"))
    return ("\n".join(lines) + "\n").encode("ascii")


class OneFinalTokenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(prefix="e3-n5-one-token-")
        cls.root = Path(cls.temporary.name)
        cls.manifest = materialize(cls.root / "corpus")
        cls.reference = build_reference(cls.manifest)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_authentic_one_row_and_independent_oracle(self) -> None:
        self.assertEqual(self.manifest["source_bytes"], "0708")
        self.assertEqual(self.manifest["total_rows"], 1)
        self.assertEqual(self.manifest["batches"], [ROW])
        self.assertEqual(self.reference["cases"][0]["name"], NAME)
        synthetic = self.reference["cases"][0]["synthetic"]
        self.assertEqual(synthetic["final"]["counters"], [1, 1])
        self.assertEqual(synthetic["final"]["metrics_f32"][1], "3f800000")
        validate_observed(self.manifest, synthetic)
        driver = (Path(__file__).resolve().parents[2] /
                  "tests/e3_private/driver.esk").read_text()
        self.assertIn("':maximum-batch-bytes 34", driver)
        self.assertEqual(CONFIG.maximum_batch_bytes, 34)

    def test_resource_reproducibility_and_mutations(self) -> None:
        second = self.root / "second"
        self.assertEqual(canonical(self.manifest), canonical(materialize(second)))
        validate_materialized(second, self.manifest)
        with self.assertRaises(FileExistsError):
            materialize(second)
        for mutate in (
            lambda m: m["batches"][0]["mask"].__setitem__(1, True),
            lambda m: m["batches"][0]["targets"].__setitem__(0, 7),
            lambda m: m.__setitem__("total_rows", 2),
            lambda m: m.__setitem__("source_bytes", "0709"),
        ):
            changed = copy.deepcopy(self.manifest)
            mutate(changed)
            with self.assertRaises(ValueError):
                validate_manifest(changed)
        shard = second / NAME / "shard-0000000000000000.ets"
        shard.write_bytes(shard.read_bytes() + b"x")
        with self.assertRaisesRegex(ValueError, "file identity"):
            validate_materialized(second, self.manifest)

    def test_result_and_cursor_binding_negative_controls(self) -> None:
        case = self.reference["cases"][0]
        raw = synthetic_frame(case, self.manifest["start_cursor"])
        observed, roles, result = parse_one_token(raw, self.manifest)
        synthetic = copy.deepcopy(case["synthetic"])
        synthetic["kind"] = "observed-f32"
        self.assertEqual(observed, synthetic)
        self.assertEqual(len(roles.splitlines()), 21)
        self.assertEqual(result["outputs_u32"][4:], [1, 1])
        mutations = (
            (raw.replace(b"E3ROW\t0\t7,0\t8,0\t1,0", b"E3ROW\t0\t7,0\t8,0\t1,1"),
             "active count"),
            (raw.replace(b"E3ROW\t0\t7,0\t8,0\t1,0", b"E3ROW\t0\t7,0\t7,0\t1,0"),
             "D2"),
            (raw.replace(b"E3RESULT\t" + str(result["outputs_u32"][0]).encode(),
                         b"E3RESULT\t0"), "published metrics"),
            (raw.replace(self.manifest["start_cursor"].encode() + b"\nE3-NUMERICAL",
                         self.manifest["end_cursor"].encode() + b"\nE3-NUMERICAL"),
             "cursor was not restored"),
            (raw.replace(b"E3STATE\t0\t", b"E3STATE\t1\t"), "ordinal"),
        )
        for changed, message in mutations:
            with self.subTest(message=message):
                self.assertNotEqual(changed, raw)
                with self.assertRaisesRegex(ValueError, message):
                    parse_one_token(changed, self.manifest)

    def test_runtime_entry_and_cli_reference_are_bounded(self) -> None:
        root = Path(__file__).resolve().parents[2]
        runtime = (root / "tests/e3_private/n5_one_token_runtime.esk").read_text()
        forms = parse_source(runtime)
        self.assertEqual(forms[-2:], [["display", '"E3-NUMERICAL-RUNTIME-PASS"'],
                                     ["newline"]])
        self.assertIn("et_e3_test_run_v1", runtime)
        self.assertIn("(vector (vector #t 0))", runtime)
        self.assertNotIn("(load ", runtime)
        reference = self.root / "reference.json"
        process = subprocess.run(
            [sys.executable, "-B", "-m", "tests.e3_reference.n5_one_token",
             "--corpus", str(self.root / "corpus"), "--reference", str(reference)],
            cwd=root, capture_output=True, text=True, check=False,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(decode(reference.read_bytes()), self.reference)
        repeat = subprocess.run(
            [sys.executable, "-B", "-m", "tests.e3_reference.n5_one_token",
             "--corpus", str(self.root / "corpus"), "--reference", str(reference)],
            cwd=root, capture_output=True, text=True, check=False,
        )
        self.assertNotEqual(repeat.returncode, 0)
        self.assertIn("must be fresh", repeat.stderr)


if __name__ == "__main__":
    unittest.main()
