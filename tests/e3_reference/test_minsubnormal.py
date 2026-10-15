"""Host checks for the development-only finite subnormal state and D2 oracle."""

from __future__ import annotations

import hashlib
from pathlib import Path
import struct
import tempfile
import unittest

from tests.c1.test_checkpoint_format import FormatError, parse_golden
from tests.d2.reference import decode_reference_cursor
from tests.e3_reference.minsubnormal import (
    ALIAS, NAME, PROVIDER, SHAPES, checkpoint, materialize,
    require_first_logit,
)


class MinSubnormalTest(unittest.TestCase):
    def test_fresh_resources_and_independent_m3_reference_are_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            first, reference, mutant = materialize(Path(temporary) / "a")
            second, second_reference, second_mutant = materialize(Path(temporary) / "b")
            self.assertEqual(first, second)
            self.assertEqual(reference, second_reference)
            self.assertEqual(mutant, second_mutant)
            for name in ("tokenizer.etok", "finite-state.etcp", "bias-clear.etcp",
                         "fixture.json", "reference.json", "bias-clear-reference.json"):
                self.assertEqual((Path(temporary) / "a" / name).read_bytes(),
                                 (Path(temporary) / "b" / name).read_bytes())
            for name, size, digest in first["d1_files"]:
                raw = (Path(temporary) / "a" / NAME / name).read_bytes()
                self.assertEqual((len(raw), hashlib.sha256(raw).hexdigest()),
                                 (size, digest))
                self.assertEqual(raw, (Path(temporary) / "b" / NAME / name).read_bytes())
            self.assertEqual([row["mask"] for row in first["batches"]],
                             [[True, True], [True, True], [True, False]])
            self.assertEqual(decode_reference_cursor(bytes.fromhex(
                first["start_cursor"])).next_ordinal, 0)
            self.assertEqual(decode_reference_cursor(bytes.fromhex(
                first["end_cursor"])).next_ordinal, 3)
            case = reference["cases"][0]
            self.assertEqual(case["corpus"]["start_cursor"], first["start_cursor"])
            self.assertEqual(case["corpus"]["end_cursor"], first["end_cursor"])
            self.assertEqual(len(case["mathematical"]), 3)
            for roles, observed in zip(case["mathematical"],
                                       case["synthetic"]["batches"], strict=True):
                self.assertEqual(len(roles["roles"]), 21)
                self.assertEqual(roles["roles"][-1]["name"], "Z")
                self.assertEqual(observed["logits_f32"][0], "00000001")
                self.assertEqual(observed["logits_f32"][256], "00000001")
                self.assertEqual(observed["mask"], first["batches"][observed["ordinal"]]["mask"])
                self.assertEqual(len(observed["ce_f32"]), 2)
                self.assertEqual(len(observed["state_f32"]), 3)
            self.assertEqual(case["synthetic"]["final"]["counters"], [5, 3])
            self.assertEqual(len(case["synthetic"]["final"]["metrics_f32"]), 4)

    def test_c1_exact_f32_bits_tie_and_bias_clear_first_logit(self) -> None:
        ordinary = checkpoint()
        cleared = checkpoint(clear_bias=True)
        self.assertNotEqual(ordinary, cleared)
        parsed = parse_golden(ordinary, PROVIDER)
        self.assertEqual(len(parsed.records), 15)
        self.assertEqual(parsed.aliases[0].members, ALIAS)
        self.assertEqual(sum(record.payload_bytes for record in parsed.records[:14]), 4736)
        self.assertEqual(tuple((tuple(part.decode() for part in record.path),
                                record.rank, record.dtype, record.device, record.layout)
                               for record in parsed.records),
                         tuple((tuple(path.split("/")), len(shape), 3, 1, 1)
                               for path, shape in SHAPES))
        values = {path: ordinary[row.payload_start:row.payload_start + row.payload_bytes]
                  for (path, _), row in zip(SHAPES, parsed.records, strict=True)}
        self.assertEqual(values["norm_final/bias"][:4], struct.pack("<I", 1))
        self.assertEqual(values["head/weight"][:4], struct.pack("<f", 1.0))
        self.assertEqual(values["head/weight"], values["token_embedding/weight"])
        self.assertEqual(parse_golden(cleared, PROVIDER).records[11].payload_bytes, 16)
        corrupt = bytearray(ordinary)
        corrupt[parsed.records[11].payload_start] ^= 1
        with self.assertRaises(FormatError):
            parse_golden(bytes(corrupt), PROVIDER)
        with tempfile.TemporaryDirectory() as temporary:
            _, reference, mutant = materialize(Path(temporary) / "fixture")
        baseline = reference["cases"][0]["synthetic"]["batches"][0]["logits_f32"]
        changed = mutant["cases"][0]["synthetic"]["batches"][0]["logits_f32"]
        self.assertEqual((baseline[0], changed[0]), ("00000001", "00000000"))
        self.assertEqual([index for index, (left, right) in
                          enumerate(zip(baseline, changed, strict=True)) if left != right],
                         [0, 256])
        with self.assertRaisesRegex(AssertionError, "logit0 differs"):
            require_first_logit(reference, mutant)


if __name__ == "__main__":
    unittest.main()
