"""Small, build-free checks for the fixed MOD5 EVAL5 reference oracle."""

from dataclasses import replace
import math
import struct
import unittest

from tests.eval5 import mod5_reference_oracle as oracle


def fixture_state():
    tensors = []
    for p, (name, shape) in enumerate(zip(oracle.PATHS, oracle.SHAPES, strict=True)):
        count = math.prod(shape)
        base = 1.0 if len(shape) == 1 else 0.0
        data = [base + ((29 * p + 11 * i + i // 5) % 37 - 18) / 128
                for i in range(count)]
        tensors.append(oracle.safe.Tensor(tuple(name.split(".")), "parameter", shape,
                                          struct.pack("<" + "f" * count, *data)))
    return oracle.safe.State("i2-dense-cpu-f32-v1", tuple(tensors))


TRAIN = ((1, 4, 1, 5, 9, 2, 6, 5, 3), (2, 7, 1, 8, 2, 8, 1, 8, 2))
CLEAN = ((0, 3, 10, 11, 12, 13, 14, 15, 0),
         (15, 14, 13, 12, 11, 10, 3, 0, 15))


class Mod5ReferenceOracleTests(unittest.TestCase):
    def test_c1_decode_and_clean_score(self):
        weights = oracle.weights_from_c1(oracle.safe.encode_c1(fixture_state()))
        result = oracle.evaluate(weights, TRAIN, CLEAN)
        self.assertEqual(result["status"], "clean")
        self.assertEqual(result["audit"]["matched_eval_ngrams"], 0)
        self.assertEqual(result["lm"]["token_count"], 16)
        self.assertEqual(result["lm"]["byte_count"], 32)
        self.assertTrue(0 < result["lm"]["total_nll"] < 100)
        self.assertEqual(result["mc"]["total_count"], 4)
        self.assertEqual(oracle.evaluate(weights, TRAIN, TRAIN)["status"], "contaminated")

    def test_rejects_wrong_model_and_token(self):
        state = fixture_state()
        with self.assertRaises(ValueError):
            oracle.weights_from_c1(oracle.safe.encode_c1(replace(state, provider="other")))
        weights = oracle.weights_from_c1(oracle.safe.encode_c1(state))
        with self.assertRaises(ValueError):
            oracle.logits(weights, (16,) * 8)
        with self.assertRaises(oracle.safe.InteropError):
            oracle.weights_from_c1(oracle.safe.encode_c1(state)[:-1] + b"x")
        with self.assertRaises(ValueError):
            oracle.documents({"fingerprint": oracle.FINGERPRINT, "vocab_size": 16,
                              "shard_count": 1, "tokens": [16] * 18})

    def test_uniform_logits_and_audit_policy(self):
        zero = [tuple(0.0 for _ in range(math.prod(shape))) for shape in oracle.SHAPES]
        result = oracle.evaluate(zero, TRAIN, CLEAN)
        self.assertAlmostEqual(result["lm"]["total_nll"], 16 * math.log(16))
        self.assertAlmostEqual(result["lm"]["bits_per_byte"], 2.0)
        self.assertEqual(result["mc"], {"correct_count": 2, "total_count": 4,
                                        "accuracy": 0.5})
        exact = (TRAIN[0], CLEAN[1])
        self.assertEqual(oracle.audit(TRAIN, exact)["exact_doc_matches"], 1)
        self.assertEqual(oracle.evaluate(zero, TRAIN, exact)["status"], "contaminated")


if __name__ == "__main__":
    unittest.main()
