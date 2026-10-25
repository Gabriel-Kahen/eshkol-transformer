"""Focused V1 layout, rejection atomicity and source-owner guards."""
import hashlib
from pathlib import Path
import struct
import unittest

from tests.g3r.test_trained_pair_producer_source import fixture
from tests.g3r.validate_trained_transcript import SHAPES, SIZE, decode


ROOT = Path(__file__).resolve().parents[2]


def synthetic():
    c2, original = fixture()
    companion = bytearray(original)
    struct.pack_into("<q", companion, 72, 1)
    companion[256:] = hashlib.sha256(companion[:256]).digest()
    record = bytearray(SIZE)
    record[:8] = b"G3RTV1\0\0"
    struct.pack_into("<II", record, 8, 1, SIZE)
    record[16:48] = hashlib.sha256(c2).digest()
    record[48:80] = hashlib.sha256(companion).digest()
    struct.pack_into("<4I", record, 80, 65, 2, 14, 0)
    offset = 96
    for index, (dim0, dim1) in enumerate(SHAPES):
        struct.pack_into("<4I", record, offset, index, 1 if dim1 == 0 else 2,
                         dim0, dim1)
        offset += 16 + 4 * dim0 * (dim1 or 1)
    assert offset == 5056
    struct.pack_into("<4q", record, 6112, *struct.unpack_from("<4q", companion, 88))
    struct.pack_into("<2I", record, 6144, 1, 1)
    struct.pack_into("<I", record, 6152, 42)
    struct.pack_into("<2I", record, 6252, 1, 2)
    record[6260:] = hashlib.sha256(record[:6260]).digest()
    return c2, bytes(companion), bytes(record)


class TranscriptV1(unittest.TestCase):
    def test_layout_and_mutations_preserve_destination(self):
        c2, companion, record = synthetic()
        destination = ["untouched"]
        decode(record, c2, companion, destination)
        self.assertEqual(destination, [record])
        structural = (0, 8, 12, 16, 48, 80, 84, 88, 92, 96, 100,
                      104, 108, 6144, 6148, 6252, 6256)
        checksum = (112, 5056, 6080, 6096, 6112, 6152, 6156, 6188,
                    6220, 6260)
        for at in structural + checksum:
            with self.subTest(offset=at):
                changed = bytearray(record)
                changed[at] ^= 1
                if at in structural:
                    changed[6260:] = hashlib.sha256(changed[:6260]).digest()
                destination = ["untouched"]
                with self.assertRaises(ValueError):
                    decode(bytes(changed), c2, companion, destination)
                self.assertEqual(destination, ["untouched"])
        destination = ["untouched"]
        with self.assertRaises(ValueError):
            decode(record + b"\0", c2, companion, destination)
        self.assertEqual(destination, ["untouched"])

    def test_source_owner_and_live_carrier_order(self):
        root = (ROOT / "native/g3r_trained_transcript_root.esk").read_text()
        producer = (ROOT / "native/g3r_trained_replay_producer_extension.esk").read_text()
        compare = (ROOT / "native/g3r_trained_replay_compare_extension.esk").read_text()
        capture = (ROOT / "native/g3r_trained_transcript_extension.esk").read_text()
        self.assertIn('(load "g3r_trained_replay_root.esk")', root)
        self.assertIn('(load "g3r_trained_transcript_extension.esk")', root)
        self.assertLess(compare.index('(g3m-prefill-p1! replayed replay-input)'),
                        compare.index("(observer 'p1 replay-native"))
        self.assertLess(compare.index('(g3r-private-replayed-p1-g1! replayed)'),
                        compare.index("(observer 'g1 replay-native"))
        self.assertIn('(i2-native-copy-bytes', capture)
        self.assertIn('(parameter-handle-tensor-internal handle)', capture)
        self.assertIn('(g3rt-fixed-path? path index)', capture)
        self.assertIn('(g3rt-fixed-shape? shape index)', capture)
        self.assertNotIn('(tr3-fixed-path? path index)', capture)
        self.assertNotIn('(tr3-fixed-shape? shape index)', capture)
        self.assertNotIn('state-dict-tensor', capture)
        self.assertLess(producer.index('(cli3-b-restore! trainer trained-c2)'),
                        producer.index('(observer-builder tree digest record prompt)'))
        self.assertLess(producer.index('(observer-builder tree digest record prompt)'),
                        producer.index("(observer 'finish #f #f #f)"))
        closure = (ROOT / "native/g3r_trained_transcript_source_closure.txt").read_text().splitlines()
        replay = (ROOT / "native/g3r_trained_replay_source_closure.txt").read_text().splitlines()
        self.assertEqual(closure, ["native/g3r_trained_transcript_root.esk",
                                   *replay, "native/g3r_trained_transcript_extension.esk"])
        self.assertEqual(len(closure), len(set(closure)))


if __name__ == "__main__":
    unittest.main()
