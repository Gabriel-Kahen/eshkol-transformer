"""Strict one-row target-offset frame/result parsing with negative mutations."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tests.e3_private.parse_target_offset_observation import parse_target
from tests.e3_private.test_numerical_tuple import frame_lines
from tests.e3_private.verify_target_offset import _paired_witness
from tests.e3_reference.target_offset import materialize
from tests.e3_reference.transcript import canonical


ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = "1065353216\t1073741824\t1077936128\t0\t2\t1"


def sample(case: str, cursor: str) -> bytes:
    records = frame_lines(1)
    if case == "target-seven":
        records[0] = records[0].replace("197,0", "197,7")
    records[-1:] = [f"E3-TARGET-OFFSET\t{case}\t{OUTPUTS}\t{cursor}\t{cursor}",
                    "E3-TARGET-OFFSET-RUNTIME-PASS"]
    return ("\n".join(records) + "\n").encode("ascii")


class TargetOffsetObservationTests(unittest.TestCase):
    def test_pair_comparison_rejects_wrong_target_and_same_ce(self) -> None:
        pair = [{"batches": [{"inputs": [3, 197], "targets": [197, target],
                              "mask": [True, True], "logits_f32": ["3f800000"] * 512,
                              "ce_f32": ["40b28797", ce]}]}
                for target, ce in ((0, "40b0d5ab"), (7, "40b105ab"))]
        self.assertEqual(_paired_witness(pair), ["40b0d5ab", "40b105ab"])
        for path, replacement in (("targets", [197, 0]),
                                  ("ce_f32", ["40b28797", "40b0d5ab"]),
                                  ("logits_f32", ["00000000"] * 512),
                                  ("inputs", [3, 0]),
                                  ("mask", [True, False])):
            changed = copy.deepcopy(pair)
            changed[1]["batches"][0][path] = replacement
            with self.subTest(path=path), self.assertRaises(ValueError):
                _paired_witness(changed)

    def test_both_genuine_rows_bind_full_frame_result_and_cursor(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e3-target-observation-") as temporary:
            manifest = materialize(Path(temporary) / "corpus")
            digest = hashlib.sha256(canonical(manifest)).hexdigest()
            for index, name in enumerate(("target-zero", "target-seven")):
                cursor = manifest["cases"][index]["start_cursor"]
                observed, roles, result = parse_target(
                    sample(name, cursor), manifest, name, digest,
                )
                self.assertEqual(observed["batches"][0]["targets"],
                                 [197, 0 if index == 0 else 7])
                self.assertEqual(len(roles.splitlines()), 21)
                self.assertEqual(len(observed["batches"][0]["logits_f32"]), 512)
                self.assertEqual(result["kind"], "e3-target-offset-result-v1")
                self.assertEqual(result["outputs_u32"][4:], [2, 1])
                self.assertEqual(result["cursor_before"], cursor)
                self.assertEqual(result["provenance"]["corpora_sha256"], digest)

    def test_malformed_duplicate_missing_and_wrong_target_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e3-target-observation-") as temporary:
            manifest = materialize(Path(temporary) / "corpus")
            cursor = manifest["cases"][0]["start_cursor"]
            raw = sample("target-zero", cursor)
            row = b"E3ROW\t0\t3,197\t197,0\t1,1\n"
            role = next(line + b"\n" for line in raw.splitlines()
                        if line.startswith(b"E3ROLE\t0\tET\t"))
            variants = (
                raw.replace(row, b"", 1),
                raw.replace(row, row + row, 1),
                raw.replace(role, b"", 1),
                raw.replace(role, role + role, 1),
                raw.replace(row, row.replace(b"197,0", b"197,7"), 1),
                raw.replace(b"E3CE\t0", b"E3CE\t1", 1),
                raw.replace(b"E3FINAL\t", b"E3FINAL\t0\t", 1),
                raw.replace(b"\t1065353216\t", b"\t0\t", 1),
                raw.replace(b"\t2\t1\t", b"\t2\t2\t", 1),
                raw.replace(cursor.encode(), b"aa", 1),
                raw.replace(b"target-zero\t", b"target-seven\t", 1),
                raw.replace(b"E3-TARGET-OFFSET-RUNTIME-PASS\n", b"", 1),
                raw + b"E3-TARGET-OFFSET-RUNTIME-PASS\n",
            )
            for index, variant in enumerate(variants):
                with self.subTest(index=index), self.assertRaises(ValueError):
                    parse_target(variant, manifest, "target-zero", "0" * 64)
            changed = copy.deepcopy(manifest)
            changed["cases"][0]["batches"][0]["targets"][1] = 7
            with self.assertRaises(ValueError):
                parse_target(raw, changed, "target-zero", "0" * 64)

    def test_cli_validates_manifest_and_writes_exclusive_records(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e3-target-observation-") as temporary:
            base = Path(temporary)
            corpus = base / "corpus"
            manifest = materialize(corpus)
            stdout = base / "stdout"
            stdout.write_bytes(sample("target-zero", manifest["cases"][0]["start_cursor"]))
            command = [sys.executable, "-B", "-m",
                       "tests.e3_private.parse_target_offset_observation",
                       "--stdout", str(stdout), "--corpus", str(corpus),
                       "--case", "target-zero", "--observed", str(base / "observed"),
                       "--roles", str(base / "roles"), "--result", str(base / "result")]
            accepted = subprocess.run(command, cwd=ROOT, capture_output=True, check=False)
            self.assertEqual(accepted.returncode, 0, accepted.stderr)
            self.assertEqual(json.loads((base / "result").read_bytes())["outputs_u32"][4:],
                             [2, 1])
            self.assertNotEqual(subprocess.run(command, cwd=ROOT, capture_output=True).returncode,
                                0)
            raw = (corpus / "corpora.json").read_bytes()
            (corpus / "corpora.json").write_bytes(
                raw.replace(b'"cases":', b'"cases":[],"cases":', 1))
            command[-1] = str(base / "unused-result")
            rejected = subprocess.run(command, cwd=ROOT, capture_output=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn(b"duplicate target manifest field", rejected.stderr)


if __name__ == "__main__":
    unittest.main()
