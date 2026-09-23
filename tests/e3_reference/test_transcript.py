from __future__ import annotations

import math
import unittest

from tests.e3_metrics.reference_probe import divide, reduce_bool
from tests.e3_reference.transcript import (
    TranscriptError, decode, encode, f32_bits, validate_observed_case,
)


class TranscriptFormatTests(unittest.TestCase):
    def test_canonical_round_trip(self) -> None:
        payload = {"case": [3, 197, 0, 255, 7, 7], "kind": "corpus-reference"}
        self.assertEqual(decode(encode(payload)), payload)

    def test_checksum_unknown_fields_duplicates_and_noncanonical_json_reject(self) -> None:
        raw = encode({"kind": "corpus-reference"})
        mutations = (
            raw.replace(b'"kind":"corpus-reference"', b'"kind":"corpus-referencf"'),
            raw[:-2] + b" \n",
            raw.replace(b'"version":1', b'"extra":0,"version":1'),
            b'{"format":"x","format":"x","payload":{},"payload_sha256":"' + b"0" * 64 + b'","version":1}\n',
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation[:40]), self.assertRaises(TranscriptError):
                decode(mutation)

    def test_encoder_rejects_nonfinite_json(self) -> None:
        with self.assertRaises(ValueError):
            encode({"bad": float("nan")})

    def test_unknown_payload_kind_and_oversize_encoding_reject(self) -> None:
        with self.assertRaisesRegex(TranscriptError, "kind"):
            decode(encode({"kind": "corpus-reference"}).replace(
                b'"kind":"corpus-reference"', b'"kind":"not-a-record"'
            ))
        with self.assertRaisesRegex(TranscriptError, "byte limit"):
            encode({"kind": "corpus-reference", "padding": "x" * (8 * 1024 * 1024)})

    def test_success_transcript_cannot_encode_exponential_overflow_as_max_finite(self) -> None:
        loss_word = "42b17218"
        loss = float.fromhex("0x1.62e430p+6")
        target_logit = math.log(255.0) - loss
        row0 = [f32_bits(target_logit)] + [f32_bits(0.0)] * 255
        row1 = [f32_bits(0.0)] * 256
        ce = [loss_word, f32_bits(math.log(256.0))]
        numerator, weight = reduce_bool(tuple(int(word, 16) for word in ce), (True, False))
        observed = {
            "batches": [{
                "ce_f32": ce, "correct_f32": "00000000", "counters": [1, 1],
                "inputs": [1, 0], "logits_f32": row0 + row1,
                "mask": [True, False], "ordinal": 0,
                "reduction_f32": [f"{numerator:08x}", f"{weight:08x}",
                                  f"{divide(numerator, weight):08x}"],
                "state_f32": [f"{numerator:08x}", f"{weight:08x}", "00000000"],
                "targets": [0, 0],
            }],
            "final": {"counters": [1, 1],
                      "metrics_f32": [loss_word, "3f800000", "7f7fffff", "00000000"]},
            "kind": "synthetic-observation", "profile": "e3-private-n1-t2-v256-d4-v1",
        }
        with self.assertRaisesRegex(TranscriptError, "expected overflow"):
            validate_observed_case(
                observed,
                ({"inputs": [1, 0], "targets": [0, 0], "mask": [True, False]},),
            )


if __name__ == "__main__":
    unittest.main()
