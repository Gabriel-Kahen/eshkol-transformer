"""Source-only checks for the N4 C1/D2/model candidate."""

import unittest
import hashlib
import struct
from pathlib import Path
from tempfile import TemporaryDirectory

from tests.c1.test_checkpoint_format import parse_golden
from tests.e3_reference import n4_sparse_bits as candidate
from tests.e3_reference.finite_overflow import PROFILE, RAW
from tests.e3_reference.corpus import FINGERPRINT, VOCAB_SIZE
from tests.d2.reference import ReferenceDataset
from tests.d2.test_resources import load_d1_resource, write_d1_resource


class SparseBitsTest(unittest.TestCase):
    def test_checkpoint_bits_and_tie(self):
        encoded = candidate.checkpoint()
        self.assertEqual(hashlib.sha256(encoded).hexdigest(),
                         "c4f00bea005b36c7e015af3ea35738656fab48038aacf8a947a6546c67696f04")
        parsed = parse_golden(encoded, candidate.PROVIDER)
        self.assertEqual(tuple(parsed.aliases[0].members), (10, 14))
        for index in (10, 14):
            row = parsed.records[index]
            payload = encoded[row.payload_start:row.payload_start + row.payload_bytes]
            self.assertEqual(payload[4 * (4 * 4):4 * (4 * 4 + 1)],
                             b"\x01\x00\x00\x00")
            self.assertEqual(payload[4 * (3 * 4):4 * (4 * 4)],
                             b"\x00\x00\x80?\x00\x00\x80\xbf" * 2)

    def test_independent_forward_logit_bits(self):
        logits = candidate.forward_bits()
        self.assertEqual(len(logits), 512)
        self.assertEqual((logits[3], logits[259]), (0x407FFFAC,) * 2)
        self.assertEqual(logits[4], 1)
        self.assertEqual(logits[256 + 4], 1)
        self.assertEqual(hashlib.sha256(struct.pack("<512I", *logits)).hexdigest(),
                         "1da1daa97726623f058c16662b69c5bd98c4e0420a9237841a383e5bb8a2e3de")
        self.assertTrue(all(word == 0 for i, word in enumerate(logits)
                            if i not in (3, 4, 259, 260)))

    def test_authentic_padded_d2_row(self):
        with TemporaryDirectory(prefix="e3-n4-sparse-") as directory:
            corpus = Path(directory)
            write_d1_resource(corpus, (tuple(RAW),),
                              fingerprint=FINGERPRINT, vocab=VOCAB_SIZE)
            loaded, digest = load_d1_resource(
                corpus, expected_fingerprint=FINGERPRINT,
                expected_vocab=VOCAB_SIZE, maximum_total_tokens=2)
            dataset = ReferenceDataset(loaded, PROFILE, manifest_digest=digest,
                                       tokenizer_fingerprint=FINGERPRINT,
                                       vocab_size=VOCAB_SIZE)
            row = dataset.next_batch()
            self.assertIsNotNone(row)
            self.assertEqual(list(row.inputs[0]), [3, 0])
            self.assertEqual(list(row.targets[0]), [4, 0])
            self.assertEqual(list(row.loss_mask[0]), [True, False])
            self.assertIsNone(dataset.next_batch())


if __name__ == "__main__":
    unittest.main()
