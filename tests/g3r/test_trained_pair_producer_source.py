"""Source custody and independent byte-contract checks; no runtime admission."""
import hashlib
from pathlib import Path
import struct
import unittest

from tests.c2.test_handler_order import parse
from tests.g3r.validate_trained_pair_artifacts import FINGERPRINT, MAGIC, verify


ROOT = Path(__file__).resolve().parents[2]


def fixture(prompt=65):
    c2 = bytearray(512)
    c2[:16] = MAGIC
    struct.pack_into("<I", c2, 20, 256)
    struct.pack_into("<Q", c2, 48, 256)
    struct.pack_into("<I", c2, 136, 96)
    struct.pack_into("<QQ", c2, 184, 4, 1)
    c2[256:352] = FINGERPRINT
    record = bytearray(288)
    record[:8] = b"G3RCV1\0\0"
    struct.pack_into("<HHI", record, 8, 1, 0, 288)
    struct.pack_into("<6I", record, 16, 1, 1, 1, 1, 0, 0)
    struct.pack_into("<6q", record, 40, 1, 1065353216, 256,
                     1065353216, 0, -1)
    struct.pack_into("<4q", record, 88, 1, 1729, 0, 0)
    record[120] = prompt
    record[128:224] = FINGERPRINT
    record[224:256] = hashlib.sha256(c2).digest()
    record[256:] = hashlib.sha256(record[:256]).digest()
    return bytes(c2), bytes(record)


class TrainedPairProducerSource(unittest.TestCase):
    def test_single_registry_and_order(self):
        root = (ROOT / "native/g3r_trained_pair_producer_root.esk").read_text()
        extension = (ROOT / "native/g3r_trained_pair_producer_extension.esk").read_text()
        producer = extension[extension.index('(define (g3r-trained-pair-produce-internal!'):]
        self.assertEqual(len(parse(extension)), 2)
        self.assertIn('(load "cli3_generate_root.esk")', root)
        self.assertNotIn('(load "t1_wave1_root.esk")', root)
        bridge = (ROOT / "native/cli3_generate_t2_bridge.esk").read_text()
        self.assertIn('(define t1-private-entry t2-private-entry)', bridge)
        ordered = [
            '(cli3-b-restore! trainer trained-c2)',
            '(g3r-trained-pair-file-image trained-c2 16777216)',
            '(set! state (trainer-state trainer))',
            '(c2-public-checkpoint-save!',
            '(trainer-release! trainer)',
            '(module-eval! model)',
            '(g3t-generator-create model tokenizer',
            '(g3m-prefill-p1! generator input)',
            '(g3r-committed-save-snapshot!',
            '(g3r-trained-pair-file-image saved-c2 16777216)',
            '(g3r-record-encode!',
            '(t1-native-atomic-write',
        ]
        positions = []
        after = 0
        for part in ordered:
            after = producer.index(part, after) + len(part)
            positions.append(after)
        self.assertEqual(positions, sorted(positions))
        self.assertIn("companion-path record 288 0", producer)
        self.assertIn("state saved-c2 (cli3-b-policy) '()", producer)
        self.assertIn("(t2-wave2-tokenizer-fingerprint tokenizer)", producer)
        self.assertIn("(g3r-bytes=? trained-image 0 saved-image 0", producer)
        self.assertIn("(g3t-generator-create model tokenizer", producer)
        self.assertNotIn("g3r-c2-pair-file-verify-c2-tokenizer-internal!", extension)
        caller = (ROOT / "tests/g3r/trained_pair_producer.esk").read_text()
        self.assertIn('((= (vector-length arguments) 7)\n     (g3r-trained-pair-produce-internal!', root)
        self.assertIn('(cli3-parse-i64 (vector-ref arguments 6) "PROMPT" 255)', root)
        self.assertIn('et_g3r_private_test_trained_pair_produce_v1', caller)
        self.assertIn('(vector (vector (argument 0) (argument 1) (argument 2)\n'
                      '                  (argument 3) (argument 4) (argument 5)\n'
                      '                  (argument 6)))', caller)
        self.assertNotIn('(load "g3r_trained_pair_producer_root.esk")', caller)
        self.assertNotIn('(list-ref args', caller)

    def test_single_private_package_entry(self):
        root = (ROOT / "native/g3r_trained_pair_producer_root.esk").read_text()
        bridge = (ROOT / "native/g3r_trained_pair_package_bridge.c").read_text()
        renames = (ROOT / "native/g3r_trained_pair_private_renames.txt").read_text()
        self.assertIn('(g3r-trained-pair-produce-internal!', root)
        self.assertEqual(renames.strip(),
                         'g3r-trained-pair-private-produce '
                         'et_e1b_private_g3r_trained_pair_produce_cabi_v1')
        self.assertIn('et_e1b_ensure_private_initialized_v1();', bridge)
        self.assertIn('et_e1b_box_value_v1(arguments)', bridge)
        self.assertIn('et_e1b_box_value_v1(output)', bridge)
        self.assertIn('et_e1b_private_g3r_trained_pair_produce_cabi_v1(', bridge)

    def test_source_closure(self):
        closure = (ROOT / "native/g3r_trained_pair_producer_source_closure.txt").read_text().splitlines()
        prior = (ROOT / "native/cli3_generate_source_closure.txt").read_text().splitlines()
        self.assertEqual(closure[0], "native/g3r_trained_pair_producer_root.esk")
        self.assertEqual(closure[1:1 + len(prior)], prior)
        self.assertEqual(len(closure), len(set(closure)))
        for item in closure:
            self.assertTrue((ROOT / item).is_file(), item)
        self.assertEqual(closure[-2:], [
            "native/g3r_trained_pair_producer_extension.esk",
            "native/g3r_trained_c2_eos_reference_extension.esk",
        ])
        self.assertEqual(len(closure), 93)

    def test_byte_oracle_and_negatives(self):
        c2, record = fixture()
        verify(c2, c2, record, 65)
        cases = []
        changed = bytearray(c2)
        changed[-1] ^= 1
        cases.append((c2, bytes(changed), record, 65))
        changed = bytearray(c2)
        changed[256:352] = FINGERPRINT[:32] + b"0" * 64
        cases.append((bytes(changed), bytes(changed), record, 65))
        changed = bytearray(c2)
        struct.pack_into("<Q", changed, 192, 0)
        cases.append((bytes(changed), bytes(changed), record, 65))
        changed = bytearray(record)
        changed[224] ^= 1
        changed[256:] = hashlib.sha256(changed[:256]).digest()
        cases.append((c2, c2, bytes(changed), 65))
        changed = bytearray(record)
        changed[128] ^= 1
        changed[256:] = hashlib.sha256(changed[:256]).digest()
        cases.append((c2, c2, bytes(changed), 65))
        changed = bytearray(record)
        changed[120] ^= 1
        changed[256:] = hashlib.sha256(changed[:256]).digest()
        cases.append((c2, c2, bytes(changed), 65))
        changed = bytearray(record)
        changed[-1] ^= 1
        cases.append((c2, c2, bytes(changed), 65))
        for case in cases:
            with self.subTest(case=cases.index(case)), self.assertRaises(ValueError):
                verify(*case)


if __name__ == "__main__":
    unittest.main()
