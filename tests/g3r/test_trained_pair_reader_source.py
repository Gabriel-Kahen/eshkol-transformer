"""One-registry reader contract; compiled pair and continuation need separate receipts."""
from pathlib import Path
import unittest

from tests.c2.test_handler_order import parse

ROOT = Path(__file__).resolve().parents[2]


class TrainedPairReaderSource(unittest.TestCase):
    def test_one_package_closure(self):
        source = (ROOT / "native/g3r_trained_pair_reader_root.esk").read_text()
        producer = (ROOT / "native/g3r_trained_pair_producer_source_closure.txt").read_text().splitlines()
        closure = (ROOT / "native/g3r_trained_pair_reader_source_closure.txt").read_text().splitlines()
        self.assertEqual(closure[0], "native/g3r_trained_pair_reader_root.esk")
        self.assertEqual(closure[1:1 + len(producer)], producer)
        self.assertEqual(len(closure), len(set(closure)))
        self.assertEqual(sum("cli3_generate_root.esk" in item for item in closure), 1)
        self.assertEqual(sum("t2_wave2_root.esk" in item for item in closure), 1)
        self.assertEqual(sum("g3r_c2_sha_receipt_extension.esk" in item for item in closure), 1)
        self.assertEqual(closure[-1], "native/g3r_trained_pair_reader_extension.esk")
        for item in closure:
            self.assertTrue((ROOT / item).is_file(), item)
        loads = [form[1] for form in parse(source) if form[0] == "load"]
        self.assertEqual(loads, [
            '"g3r_trained_pair_producer_root.esk"',
            '"g3r_c2_sha_stage_extension.esk"',
            '"g3r_c2_sha_receipt_extension.esk"',
            '"g3r_c2_pair_decode_extension.esk"',
            '"g3r_c2_pair_file_read_extension.esk"',
            '"g3r_c2_pair_tokenizer_bind_extension.esk"',
            '"g3r_trained_pair_reader_extension.esk"',
        ])
        self.assertNotIn('(load "t1_wave1_root.esk")', source)

    def test_authentication_and_retirement_before_return(self):
        source = (ROOT / "native/g3r_trained_pair_reader_extension.esk").read_text()
        self.assertEqual(len(parse(source)), 1)
        ordered = [
            "(vector-set! t2-wave2-policy-projector 0 c2-policy-t2-entry-internal)",
            "(cli3-b-resolved config-path 'generate)",
            "(cli3-b-tokenizer tokenizer-path resolved)",
            "(t2-wave2-tokenizer-fingerprint tokenizer)",
            "(g3r-c2-load-with-sha-internal",
            "(g3r-c2-pair-file-verify-c2-tokenizer-internal!",
            "(string=? live-fingerprint",
            "(g3r-c2-receipt-release-internal! (vector-ref receipt-cell 0))",
            "(vector-set! receipt-cell 0 #f)",
            "decoded))))",
        ]
        positions = [source.rindex(part) for part in ordered]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("(if (vector-ref receipt-cell 0)", source)
        self.assertIn("(g3r-c2-receipt-release-internal!", source)
        self.assertNotIn("trainer-load-state!", source)
        self.assertNotIn("g3r-replayed-p1-g1", source)

    def test_private_bridge_and_thin_caller(self):
        root = (ROOT / "native/g3r_trained_pair_reader_root.esk").read_text()
        bridge = (ROOT / "native/g3r_trained_pair_reader_package_bridge.c").read_text()
        renames = (ROOT / "native/g3r_trained_pair_reader_private_renames.txt").read_text()
        caller = (ROOT / "tests/g3r/trained_pair_reader.esk").read_text()
        self.assertIn("(= (vector-length arguments) 4)", root)
        self.assertEqual(renames.strip(),
            "g3r-trained-pair-private-read "
            "et_e1b_private_g3r_trained_pair_read_cabi_v1")
        self.assertIn("et_e1b_ensure_private_initialized_v1();", bridge)
        self.assertEqual(bridge.count("et_e1b_box_value_v1("), 2)
        self.assertIn("et_g3r_private_test_trained_pair_read_v1", caller)
        self.assertNotIn("(load", caller)
        self.assertIn("(= (vector-ref (vector-ref record 0) 4) 0)", caller)
        self.assertEqual(len(parse(caller)), 12)


if __name__ == "__main__":
    unittest.main()
