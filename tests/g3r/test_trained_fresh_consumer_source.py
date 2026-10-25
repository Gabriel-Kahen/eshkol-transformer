"""Source and data-only negatives for the proposed fresh C2 consumer.

This does not execute Eshkol or authenticate a future producer transcript.
"""
from pathlib import Path
import hashlib
import struct
import unittest

from tests.c2.test_handler_order import parse


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "native/g3r_trained_fresh_consumer_root.esk"
CLOSURE = ROOT / "native/g3r_trained_fresh_consumer_source_closure.txt"
SEAM = ROOT / "native/g3r_c2_receipt_tr3_restore_source_closure.txt"
BRIDGE = ROOT / "native/g3r_trained_fresh_consumer_bridge.c"
RENAMES = ROOT / "native/g3r_trained_fresh_consumer_private_renames.txt"
CALLER = ROOT / "tests/g3r/trained_fresh_consumer.esk"
NEGATIVE = ROOT / "tests/g3r/trained_fresh_consumer_negative.esk"
C2_SHA = bytes.fromhex(
    "9ec2a1d40ebd43e57d1e8b75fa8f6d96cd4d17abc1002f70da04d20144ee78df"
)
PAIR_SHA = bytes.fromhex(
    "75fec71697cf50556228481b0a2a30f634ce8991a7f6db4ad7fbdc700f1cfbf2"
)
SHAPES = ((4, 4),) * 4 + ((4, 8), (8, 4)) + ((4, 0),) * 4 + (
    (256, 4),) + ((4, 0),) * 2 + ((2, 4),)


def verify_source(source: str, closure: str, seam: str) -> None:
    forms = parse(source)
    expected_loads = [
        ["load", '"g3r_trained_pair_reader_root.esk"'],
        ["load", '"g3r_c2_receipt_tr3_restore_extension.esk"'],
        ["load", '"g3r_rng_import_extension.esk"'],
        ["load", '"g3r_replayed_p1_g1_extension.esk"'],
    ]
    if forms[:4] != expected_loads:
        raise ValueError("one-root load order changed")
    seam_paths = seam.splitlines()
    expected = ["native/g3r_trained_fresh_consumer_root.esk",
                *seam_paths[1:],
                "native/g3r_rng_import_extension.esk",
                "native/g3r_replayed_p1_g1_extension.esk"]
    paths = closure.splitlines()
    if (len(seam_paths) != 101 or paths != expected or
            len(paths) != len(set(paths)) != 103 or
            "native/g3r_trained_replay_root.esk" in paths or
            "native/cli3_generate_root.esk" not in paths):
        raise ValueError("103-source union changed")
    required = (
        "(g3r-c2-receipt-borrow-begin-internal",
        "(g3r-c2-receipt-borrow-sha-internal",
        "(g3rf-decode! transcript digest companion destination)",
        "(g3r-c2-receipt-tr3-restore-internal! receipt trainer)",
        "(g3rf-receipt-model-bits! receipt model)",
        "(g3rf-model-bits! model (vector-ref (vector-ref destination 0) 0))",
        "(tr3-lease-unenroll-internal! trainer)",
        "(g3r-c2-receipt-release-internal! receipt)",
        "(module-eval! model)",
        "(g3r-rng-owner-from-checked-words",
        "(g3m-prefill-p1! generator input)",
        "(g3rf-p1-trace! native bytes)",
        "(g3r-private-replayed-p1-g1! generator)",
        "(g3rf-g1-trace!",
        "(g3r-native-replay-preflight native)",
        "(state-dict-c2-owned-tensor-borrow-end-internal",
        "(g3r-c2-receipt-borrow-end-internal!",
        "(i2-value-equal?",
        "(i2-checkpoint-encode",
    )
    entry = source[source.index(
        "(define (g3r-trained-fresh-consume-internal!"):]
    positions = [entry.find(part) for part in required[:14]]
    if -1 in positions or positions != sorted(positions):
        raise ValueError("fresh restore/trace order changed")
    if any(part not in source for part in required[14:]):
        raise ValueError("bit, borrow, or full-context guard removed")
    forbidden = ("(load \"g3r_trained_replay_root.esk\")",
                 "(load \"cli3_root.esk\")",
                 "(c2-public-checkpoint-load", "g3r-trained-replay-compare!")
    if any(part in source for part in forbidden):
        raise ValueError("second registry or bypass entered consumer")


def fixture() -> bytes:
    data = bytearray(6292)
    data[:8] = b"G3RTV1\0\0"
    struct.pack_into("<II", data, 8, 1, len(data))
    data[16:48], data[48:80] = C2_SHA, PAIR_SHA
    struct.pack_into("<4I", data, 80, 65, 2, 14, 0)
    offset = 96
    for index, (d0, d1) in enumerate(SHAPES):
        struct.pack_into("<4I", data, offset, index, 1 if d1 == 0 else 2,
                         d0, d1)
        offset += 16 + 4 * d0 * (d1 or 1)
    assert offset == 5056
    struct.pack_into("<4q", data, 6112, 1, 1729, 0, 0)
    struct.pack_into("<2I", data, 6144, 1, 1)
    struct.pack_into("<I", data, 6152, 7)
    struct.pack_into("<2I", data, 6252, 1, 2)
    data[6260:] = hashlib.sha256(data[:6260]).digest()
    return bytes(data)


def decode(data: bytes, destination: list) -> None:
    if len(destination) != 1 or len(data) != 6292:
        raise ValueError("extent")
    if (data[:8] != b"G3RTV1\0\0" or
            struct.unpack_from("<II", data, 8) != (1, 6292) or
            data[16:48] != C2_SHA or data[48:80] != PAIR_SHA or
            struct.unpack_from("<4I", data, 80) != (65, 2, 14, 0) or
            data[6260:] != hashlib.sha256(data[:6260]).digest()):
        raise ValueError("header or checksum")
    offset = 96
    for index, (d0, d1) in enumerate(SHAPES):
        if struct.unpack_from("<4I", data, offset) != (
                index, 1 if d1 == 0 else 2, d0, d1):
            raise ValueError("roster")
        offset += 16 + 4 * d0 * (d1 or 1)
    if (offset != 5056 or struct.unpack_from("<2I", data, 6144) != (1, 1) or
            struct.unpack_from("<I", data, 6152)[0] > 255 or
            struct.unpack_from("<2I", data, 6252) != (1, 2) or
            struct.unpack_from("<4q", data, 6112) != (1, 1729, 0, 0)):
        raise ValueError("trace")
    destination[0] = bytes(data)


class FreshConsumerSourceTest(unittest.TestCase):
    def test_one_root_and_authority_order(self):
        verify_source(SOURCE.read_text(), CLOSURE.read_text(), SEAM.read_text())

    def test_distinct_process_file_harness(self):
        source, bridge, caller = (SOURCE.read_text(), BRIDGE.read_text(),
                                  CALLER.read_text())
        wrapper = source[source.index("(define (g3rf-private-test-run"):]
        sequence = (
            "(cli3-b-resolved", "(cli3-b-tokenizer",
            "(t1-native-read-exact companion-path companion",
            "(t1-native-read-exact transcript-path transcript",
            "(cli3-b-dataset", "(cli3-b-create-trainer",
            "(g3r-c2-load-with-sha-internal",
            "(g3r-c2-pair-decode-internal!",
            "(g3r-trained-fresh-consume-internal!",
        )
        positions = [wrapper.find(part) for part in sequence]
        self.assertNotIn(-1, positions)
        self.assertEqual(positions, sorted(positions))
        self.assertIn("(cli3-b-policy)", wrapper)
        self.assertIn("(k2-public-capability-discover)", wrapper)
        self.assertIn("(g3r-c2-receipt-release-internal! receipt)", wrapper)
        self.assertIn("(trainer-release! trainer)", wrapper)
        self.assertEqual(RENAMES.read_text().strip(),
                         "g3rf-private-test-run et_e1b_private_g3r_fresh_test_run_cabi_v1")
        self.assertIn("et_e1b_private_g3r_fresh_test_run_cabi_v1", bridge)
        self.assertIn("et_g3r_private_test_fresh_consume_v1", bridge)
        self.assertIn("et_g3r_private_test_fresh_consume_v1", caller)
        self.assertIn("G3-R FRESH TRAINED C2 TRANSCRIPT PASS", caller)
        negative = NEGATIVE.read_text()
        self.assertIn("(require transformer.error_consumer)", negative)
        self.assertIn("(eq? (vector-ref output 0) 'untouched)", negative)
        self.assertIn("(eq? (transformer-error-category caught) 'invalid-state)",
                      negative)
        self.assertIn("'g3r-trained-fresh-consume!", negative)
        self.assertIn("(eq? (vector-ref record 1) 'dead)", wrapper)

    def test_source_mutations(self):
        source, closure, seam = (SOURCE.read_text(), CLOSURE.read_text(),
                                 SEAM.read_text())
        for old in (
            "(g3r-c2-receipt-tr3-restore-internal! receipt trainer)",
            "(g3rf-receipt-model-bits! receipt model)",
            "(g3rf-model-bits! model (vector-ref (vector-ref destination 0) 0))",
            "(g3rf-p1-trace! native bytes)",
            "(g3r-private-replayed-p1-g1! generator)",
            "(g3rf-g1-trace!",
            "(g3r-native-replay-preflight native)",
            "(i2-value-equal?", "(i2-checkpoint-encode",
        ):
            with self.subTest(old=old), self.assertRaises(ValueError):
                verify_source(source.replace(old, "(removed"), closure, seam)
        with self.assertRaises(ValueError):
            verify_source(source, closure +
                          "native/g3r_trained_replay_root.esk\n", seam)

    def test_synthetic_layout_and_mutations(self):
        baseline = fixture()
        destination = [None]
        decode(baseline, destination)
        self.assertEqual(destination[0], baseline)
        for at, reseal in ((0, False), (8, True), (16, True), (48, True),
                           (84, True), (88, True), (96, True), (100, True),
                           (6144, True), (6148, True), (6153, True),
                           (6112, True), (6252, True), (6256, True),
                           (5056, False)):
            changed = bytearray(baseline)
            changed[at] ^= 1
            if reseal:
                changed[6260:] = hashlib.sha256(changed[:6260]).digest()
            destination = ["untouched"]
            with self.subTest(at=at), self.assertRaises(ValueError):
                decode(bytes(changed), destination)
            self.assertEqual(destination, ["untouched"])
        for at in (112, 5056, 6080, 6152, 6156, 6220):
            changed = bytearray(baseline)
            changed[at] ^= 1
            changed[6260:] = hashlib.sha256(changed[:6260]).digest()
            destination = [None]
            decode(bytes(changed), destination)
            with self.subTest(at=at), self.assertRaises(ValueError):
                compare_trace_and_parameters(baseline, destination[0])


def compare_trace_and_parameters(expected: bytes, observed: bytes) -> None:
    """Exact fixed schema payload comparison; header fields were decoded first."""
    for start, end in ((112, 5056), (5056, 6080), (6080, 6144),
                       (6152, 6156), (6156, 6220), (6220, 6252)):
        if expected[start:end] != observed[start:end]:
            raise ValueError("fresh parameter or P1/G1 trace mismatch")


if __name__ == "__main__":
    unittest.main()
