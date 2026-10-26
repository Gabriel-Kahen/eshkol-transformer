"""Source and resource contract for the bounded nonzero private M3 control."""

from __future__ import annotations

import copy
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import unittest

from tests.e3_reference.nonzero_equivalence import build_reference, materialize, validate
from tests.d2.reference import (ReferenceConfig, decode_reference_cursor,
                                encode_reference_cursor)


ROOT = Path(__file__).resolve().parents[2]


class NonzeroEquivalenceTest(unittest.TestCase):
    @unittest.skipUnless(platform.python_version() == "3.14.6", "requires pinned Q0")
    def test_independent_zero_m3_reference_has_nonzero_correct(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e3-nonzero-q0-") as temporary:
            reference = build_reference(materialize(Path(temporary) / "fixture"))
            case = reference["cases"][0]["synthetic"]
            self.assertEqual([row["correct_f32"] for row in case["batches"]],
                             ["3f800000", "3f800000"])
            self.assertEqual(case["final"], {
                "counters": [4, 2],
                "metrics_f32": ["40b17218", "40800000", "43800000", "3f000000"],
            })

    def test_authentic_two_full_rows_and_rejections(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e3-nonzero-") as temporary:
            root = Path(temporary) / "fixture"
            payload = materialize(root)
            rows = payload["case"]["batches"]
            self.assertEqual(rows, [
                {"inputs": [0, 0], "targets": [0, 1], "mask": [True, True]},
                {"inputs": [1, 1], "targets": [1, 0], "mask": [True, True]},
            ])
            self.assertEqual(payload["case"]["source_bytes"], "0000010100")
            self.assertEqual(payload["case"]["total_rows"], 2)
            cursor = decode_reference_cursor(bytes.fromhex(payload["case"]["start_cursor"]))
            self.assertEqual(cursor.config.maximum_batch_bytes, 34)
            self.assertEqual(cursor.next_ordinal, 0)
            self.assertEqual(cursor.total_rows, 2)
            changed = copy.deepcopy(payload)
            changed["case"]["start_cursor"] = encode_reference_cursor(
                cursor.manifest_digest, cursor.tokenizer_fingerprint,
                cursor.vocab_size, ReferenceConfig(1, 2, None, 1, True),
                cursor.total_rows, cursor.next_ordinal).hex()
            with self.assertRaisesRegex(ValueError, "D2 full rows or cursor"):
                validate(root, changed)
            changed = copy.deepcopy(payload)
            changed["case"]["batches"][1]["targets"][1] = 1
            with self.assertRaisesRegex(ValueError, "D2 full rows"):
                validate(root, changed)
            changed = copy.deepcopy(payload)
            changed["checkpoint"]["alias_indices"] = [6, 8]
            with self.assertRaisesRegex(ValueError, "C1 bytes"):
                validate(root, changed)
            original = (root / "nonzero-equivalence" / "shard-0000000000000000.ets").read_bytes()
            shard = root / "nonzero-equivalence" / "shard-0000000000000000.ets"
            shard.write_bytes(original[:-1] + bytes((original[-1] ^ 1,)))
            with self.assertRaises((ValueError, AssertionError)):
                validate(root, payload)

    def test_four_controls_are_distinct_private_modes(self) -> None:
        builder = (ROOT / "scripts/build-e1b-consumer.sh").read_text()
        gate = (ROOT / "scripts/test-e3-nonzero-equivalence.sh").read_text()
        provider = (ROOT / "native/e3_evaluation_metrics_provider.c").read_text()
        modes = {
            "two-term-reversal": "ET_E3_NUMERICAL_REVERSE_CORRECT",
            "bool-fma": "ET_E3_EQUIV_BOOL_FMA",
            "weight-from-tokens": "ET_E3_EQUIV_W_FROM_TOKENS",
            "f64-add-cast": "ET_E3_EQUIV_F64_ADD_CAST",
        }
        for mode, macro in modes.items():
            self.assertIn(f'{mode}) package_native_cflags+=(-D{macro})', builder)
            self.assertIn(mode, gate)
            self.assertIn(f"#ifdef {macro}", provider)
        self.assertIn('"${e3_tuple_kind}" == uniform', builder)
        self.assertIn('cmp "${out}/observed-a.json" "${out}/observed-${variant}.json"', gate)
        self.assertIn('cmp "${out}/roles-a.tsv" "${out}/roles-${variant}.tsv"', gate)
        self.assertIn('cmp "${out}/transcript-a.json" "${out}/transcript-${variant}.json"', gate)
        self.assertIn('cmp "${out}/run-a.stdout" "${out}/run-${variant}.stdout"', gate)
        base = (ROOT / "tests/e3_private/uniform_package_undefined_symbols.txt").read_text().splitlines()
        fma = (ROOT / "tests/e3_private/uniform_bool_fma_undefined_symbols.txt").read_text().splitlines()
        self.assertEqual(fma, sorted(base + ["fmaf"]))
        if shutil.which("clang") is None:
            self.skipTest("Clang unavailable")
        source = ROOT / "native/e3_evaluation_metrics_provider.c"
        compile = ["clang", "-std=c11", "-fsyntax-only", "-I", str(ROOT / "include"),
                   "-I", str(ROOT / "src")]
        for macro in modes.values():
            denied = subprocess.run(compile + [f"-D{macro}", str(source)],
                                    capture_output=True, text=True, check=False)
            self.assertNotEqual(denied.returncode, 0, macro)
            admitted = subprocess.run(compile + ["-DET_E3_NUMERICAL_TESTING",
                                               f"-D{macro}", str(source)],
                                      capture_output=True, text=True, check=False)
            self.assertEqual(admitted.returncode, 0, admitted.stderr)

    def test_c1_paths_are_set_inside_private_scenario(self) -> None:
        runtime = (ROOT / "tests/e3_private/uniform_runtime.esk").read_text()
        driver = (ROOT / "tests/e3_private/uniform_driver.esk").read_text()
        self.assertNotIn("(e3-uniform-set-files!", runtime)
        self.assertIn("(vector (vector #t 0\n                 (vector", runtime)
        self.assertIn("(files-ready (e3-uniform-set-files! (vector-ref checked 2)))", driver)
        self.assertLess(driver.index("(files-ready (e3-uniform-set-files!"),
                        driver.index("(fixture (e3-test-fixture))"))


if __name__ == "__main__":
    unittest.main()
