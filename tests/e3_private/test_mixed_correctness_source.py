"""Source and parser guards for the one-row mixed-correctness N3 leaf."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tests.e3_private.parse_mixed_correctness_observation import parse_mixed
from tests.e3_private.test_numerical_tuple import frame_lines
from tests.e3_private.verify_mixed_correctness import _paired_witness
from tests.e3_reference.mixed_correctness import (
    build_reference, materialize, validate_materialized,
)
from tests.e3_reference.transcript import canonical, validate_observed_case


OUTPUTS = "1065353216\t1073741824\t1077936128\t0\t2\t1"


def sample(case: str, cursor: str) -> bytes:
    records = frame_lines(1)
    records[0] = f"E3ROW\t0\t3,14\t14,{31 if case == 'matched-correct' else 0}\t1,1"
    records[-1:] = [f"E3-MIXED-CORRECTNESS\t{case}\t{OUTPUTS}\t{cursor}\t{cursor}",
                    "E3-MIXED-CORRECTNESS-RUNTIME-PASS"]
    return ("\n".join(records) + "\n").encode("ascii")


class MixedCorrectnessSourceTests(unittest.TestCase):
    def test_genuine_d1_d2_pair_and_seeded_independent_oracle(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e3-mixed-source-") as temporary:
            root = Path(temporary) / "corpus"
            manifest = materialize(root)
            validate_materialized(root, manifest)
            self.assertEqual([case["source_bytes"] for case in manifest["cases"]],
                             ["030e1f", "030e00"])
            self.assertEqual([case["batches"][0] for case in manifest["cases"]], [
                {"inputs": [3, 14], "targets": [14, 31], "mask": [True, True]},
                {"inputs": [3, 14], "targets": [14, 0], "mask": [True, True]},
            ])
            reference = build_reference(manifest)
            observations = [case["synthetic"] for case in reference["cases"]]
            self.assertEqual(_paired_witness(observations), ["40aeb52c", "40b0b72f"])
            self.assertEqual([item["final"]["metrics_f32"][3] for item in observations],
                             ["3f800000", "3f000000"])
            # The accepted exact-44c compiled all-correct frame emitted this
            # 1-ULP control CE2. It leaves its exact row reduction/state intact.
            compiled_control = copy.deepcopy(observations)
            compiled_control[0]["batches"][0]["ce_f32"][1] = "40aeb52d"
            validate_observed_case(compiled_control[0],
                                   tuple(manifest["cases"][0]["batches"]))
            self.assertEqual(_paired_witness(compiled_control),
                             ["40aeb52d", "40b0b72f"])
            for field, value in (("targets", [14, 31]),
                                 ("logits_f32", ["00000000"] * 512),
                                 ("ce_f32", ["40af07af", "40aeb52d"]),
                                 ("correct_f32", "40000000")):
                changed = copy.deepcopy(observations)
                changed[1]["batches"][0][field] = value
                with self.subTest(field=field), self.assertRaises(ValueError):
                    _paired_witness(changed)
            changed = copy.deepcopy(observations)
            changed[1]["batches"][0]["correct_f32"] = "40000000"
            with self.assertRaises(ValueError):
                _paired_witness(changed)

    def test_physical_identity_and_single_row_parser_rejections(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e3-mixed-parser-") as temporary:
            root = Path(temporary) / "corpus"
            manifest = materialize(root)
            digest = hashlib.sha256(canonical(manifest)).hexdigest()
            for index, case in enumerate(("matched-correct", "mixed")):
                cursor = manifest["cases"][index]["start_cursor"]
                raw = sample(case, cursor)
                observed, roles, result = parse_mixed(raw, manifest, case, digest)
                self.assertEqual(observed["batches"][0]["targets"],
                                 [14, 31 if index == 0 else 0])
                self.assertEqual(len(roles.splitlines()), 21)
                self.assertEqual(result["outputs_u32"][4:], [2, 1])
                row = raw.splitlines(keepends=True)[0]
                role = next(line for line in raw.splitlines(keepends=True)
                            if line.startswith(b"E3ROLE\t0\tET\t"))
                for mutation in (raw.replace(row, b"", 1),
                                 raw.replace(row, row + row, 1),
                                 raw.replace(role, role + role, 1),
                                 raw.replace(b"\t2\t1\t", b"\t2\t2\t", 1),
                                 raw.replace(cursor.encode(), b"aa", 1),
                                 raw.replace(b"-RUNTIME-PASS\n", b"-RUNTIME-FAIL\n", 1),
                                 raw + b"E3-MIXED-CORRECTNESS-RUNTIME-PASS\n"):
                    with self.subTest(case=case, mutation=mutation[:32]), self.assertRaises(ValueError):
                        parse_mixed(mutation, manifest, case, digest)
            shard = root / "mixed" / "shard-0000000000000000.ets"
            shard.write_bytes(shard.read_bytes() + b"x")
            with self.assertRaises(ValueError):
                validate_materialized(root, manifest)

    def test_cli_rejects_duplicate_manifest_and_existing_results(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e3-mixed-cli-") as temporary:
            base = Path(temporary)
            corpus = base / "corpus"
            manifest = materialize(corpus)
            stdout = base / "stdout"
            stdout.write_bytes(sample("mixed", manifest["cases"][1]["start_cursor"]))
            command = [sys.executable, "-B", "-m",
                       "tests.e3_private.parse_mixed_correctness_observation",
                       "--stdout", str(stdout), "--corpus", str(corpus),
                       "--case", "mixed", "--observed", str(base / "observed"),
                       "--roles", str(base / "roles"), "--result", str(base / "result")]
            accepted = subprocess.run(command, capture_output=True, check=False)
            self.assertEqual(accepted.returncode, 0, accepted.stderr)
            self.assertEqual(json.loads((base / "result").read_bytes())["outputs_u32"][4:],
                             [2, 1])
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            source = corpus / "corpora.json"
            source.write_bytes(source.read_bytes().replace(b'"cases":',
                                                          b'"cases":[],"cases":', 1))
            command[-1] = str(base / "unused-result")
            duplicate = subprocess.run(command, capture_output=True)
            self.assertNotEqual(duplicate.returncode, 0)
            self.assertIn(b"duplicate mixed-correctness manifest field", duplicate.stderr)


if __name__ == "__main__":
    unittest.main()
