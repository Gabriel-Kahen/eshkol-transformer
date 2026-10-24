"""Private C2/FILE fingerprint binder source and gate contract."""
from pathlib import Path
import unittest

from tests.c2.test_handler_order import definition, parse

ROOT = Path(__file__).resolve().parents[2]


class PairTokenizerBindSourceTest(unittest.TestCase):
    def test_borrow_and_publication_contract(self):
        source = (ROOT / "native/g3r_c2_pair_tokenizer_bind_extension.esk").read_text()
        forms = parse(source)
        self.assertEqual(len(forms), 1)
        entry = definition({"source": forms},
                           "g3r-c2-pair-file-verify-c2-tokenizer-internal!")
        self.assertEqual(entry[1],
                         ["g3r-c2-pair-file-verify-c2-tokenizer-internal!",
                          "receipt", "path", "destination"])
        ordered = (
            "(vector? destination) (= (vector-length destination) 8)",
            "(g3r-c2-pair-file-decode-internal! receipt path staged)",
            "(g3r-c2-receipt-borrow-begin-internal receipt operation)",
            "(g3r-c2-receipt-borrow-c2-internal",
            "(c2-training-state-borrow-projection-internal c2-borrow)",
            "(vector-ref projection 2)", "(vector-ref controls 1)",
            "(= vocabulary 256)", "(utf8->string record-fingerprint)",
            "(g3r-c2-receipt-borrow-end-internal! (vector-ref borrow-cell 0))",
            "(vector-copy! destination 0 staged 0 8)",
        )
        locations = [source.index(part) for part in ordered[:-2]]
        locations.extend(source.rindex(part) for part in ordered[-2:])
        self.assertEqual(locations, sorted(locations))
        self.assertIn("(if (vector-ref borrow-cell 0)", source)
        self.assertNotIn("generator-", source)

    def test_root_and_compiled_gate(self):
        root = (ROOT / "native/g3r_c2_pair_tokenizer_bind_root.esk").read_text()
        self.assertEqual(parse(root), [
            ["load", '"g3r_c2_pair_file_read_root.esk"'],
            ["load", '"g3r_c2_pair_tokenizer_bind_extension.esk"'],
        ])
        predecessor = (ROOT / "native/g3r_c2_pair_file_read_source_closure.txt").read_text().splitlines()
        closure = (ROOT / "native/g3r_c2_pair_tokenizer_bind_source_closure.txt").read_text().splitlines()
        self.assertEqual(closure, ["native/g3r_c2_pair_tokenizer_bind_root.esk",
                                   *predecessor,
                                   "native/g3r_c2_pair_tokenizer_bind_extension.esk"])
        gate = (ROOT / "scripts/test-g3r-c2-authentic-save.sh").read_text()
        self.assertIn("build_tokenizer_bind normal", gate)
        self.assertIn("build_tokenizer_bind sanitize", gate)
        self.assertIn("tokenizer-normal/repeat.stdout", gate)
        self.assertIn("g3r_c2_pair_tokenizer_bind_source_closure.txt", gate)


if __name__ == "__main__":
    unittest.main()
