"""Private G3RCV1 file-reader source contract and deterministic fixture tests."""
import hashlib
from pathlib import Path
import stat
import tempfile
import unittest

from tests.c2.test_handler_order import definition, parse
from tests.c2.test_checkpoint_format import make_fixture
from tests.g3r.prepare_pair_file_inputs import prepare

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "native/g3r_c2_pair_file_read_extension.esk"
ROOT_FILE = ROOT / "native/g3r_c2_pair_file_read_root.esk"
CLOSURE = ROOT / "native/g3r_c2_pair_file_read_source_closure.txt"
PREDECESSOR = ROOT / "native/g3r_c2_sha_receipt_source_closure.txt"
FIXTURE = ROOT / "tests/g3r/pair_file_read_runtime.esk"


def verify(source: str, root: str, closure: str, predecessor: str,
           fixture: str) -> None:
    forms = parse(source)
    if len(forms) != 1:
        raise ValueError("file reader must add one private entry only")
    entry = definition({"source": forms}, "g3r-c2-pair-file-decode-internal!")
    if entry[1] != ["g3r-c2-pair-file-decode-internal!",
                    "receipt", "path", "destination"]:
        raise ValueError("private file reader signature changed")
    required = (
        "(t2-wave2-valid-file-path? path)",
        "(c2-load-fail 'invalid-argument operation",
        "(make-bytevector 288 0)",
        "(t1-native-read-exact path record 288 288)",
        "(if (not (= status 0))",
        "(t2-wave2-io-fail operation",
        "(g3r-c2-pair-decode-internal! receipt record destination)",
    )
    positions = [source.find(part) for part in required]
    if -1 in positions or positions != sorted(positions):
        raise ValueError("path, exact C1 read, error and pair decode order changed")
    if (source.count("t1-native-read-exact") != 1 or
            source.count("g3r-c2-pair-decode-internal!") != 1 or
            "checkpoint-save" in source or "generator-" in source):
        raise ValueError("reader widened the private route")
    expected_root = [
        ["load", '"g3r_c2_sha_receipt_root.esk"'],
        ["load", '"g3r_c2_pair_file_read_extension.esk"'],
    ]
    if parse(root) != expected_root:
        raise ValueError("private file root load order changed")
    expected = ["native/g3r_c2_pair_file_read_root.esk",
                *predecessor.splitlines(),
                "native/g3r_c2_pair_file_read_extension.esk"]
    if closure.splitlines() != expected or len(expected) != len(set(expected)):
        raise ValueError("source closure differs from exact predecessor")
    parse(fixture)
    for text in (
            '"valid.g3r"', '"wrong-digest.g3r"', '"torn.g3r"',
            '"short.g3r"', '"long.g3r"', '"link.g3r"', '"fifo.g3r"',
            '(string #\\nul)',
            '(make-string 4097 #\\a)',
            '(string->utf8 oversized-utf8-path)',
            '"embedded-NUL path"', '"oversized UTF-8 path"',
            '(g3r-pair-file-io-fail 6 1 5)',
            '(g3r-pair-file-io-fail 7 1 5)',
            '(g3r-pair-file-io-fail 16 1 5)',
            '(vector \'forged)', 'g3r-c2-receipt-borrow-begin-internal',
            'g3r-c2-receipt-release-internal!',
            '(untouched? pending)', '(receipt-idle? receipt)'):
        if text not in fixture:
            raise ValueError(f"runtime fixture lost {text}")


class FileReadSourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SOURCE.read_text()
        cls.root = ROOT_FILE.read_text()
        cls.closure = CLOSURE.read_text()
        cls.predecessor = PREDECESSOR.read_text()
        cls.fixture = FIXTURE.read_text()

    def test_source_contract(self):
        verify(self.source, self.root, self.closure,
               self.predecessor, self.fixture)

    def test_causal_mutations(self):
        for old, new in (
                ("(make-bytevector 288 0)", "(make-bytevector 287 0)"),
                ("(t1-native-read-exact path record 288 288)",
                 "(t1-native-read-exact path record 288 1024)"),
                ("(t2-wave2-valid-file-path? path)", "(string? path)"),
                ("(if (not (= status 0))", "(if (= status 0)"),
                ("(g3r-c2-pair-decode-internal! receipt record destination)",
                 "(g3r-record-decode! record receipt destination)")):
            with self.subTest(old=old):
                mutated = self.source.replace(old, new, 1)
                with self.assertRaises(ValueError):
                    verify(mutated, self.root, self.closure,
                           self.predecessor, self.fixture)
        with self.assertRaises(ValueError):
            verify(self.source, self.root,
                   self.closure.replace(
                       "native/g3r_c2_pair_file_read_extension.esk", ""),
                   self.predecessor, self.fixture)
        with self.assertRaises(ValueError):
            verify(self.source, self.root, self.closure,
                   self.predecessor,
                   self.fixture.replace('(g3r-pair-file-io-fail 7 1 5)', ''))
        for marker in ('(string #\\nul)',
                       '(make-string 4097 #\\a)',
                       '(string->utf8 oversized-utf8-path)',
                       '"embedded-NUL path"', '"oversized UTF-8 path"'):
            with self.subTest(marker=marker), self.assertRaises(ValueError):
                verify(self.source, self.root, self.closure,
                       self.predecessor, self.fixture.replace(marker, '', 1))

    def test_independent_files(self):
        with tempfile.TemporaryDirectory() as scratch:
            home = Path(scratch)
            c2 = home / "image.c2"
            c2.write_bytes(make_fixture()[0])
            output = home / "files"
            prepare(c2, output)
            valid = (output / "valid.g3r").read_bytes()
            wrong = (output / "wrong-digest.g3r").read_bytes()
            matched = (output / "matched.g3r").read_bytes()
            self.assertEqual(len(valid), 288)
            self.assertEqual(valid[:8], b"G3RCV1\0\0")
            self.assertEqual(valid[224:256], hashlib.sha256(c2.read_bytes()).digest())
            self.assertEqual(valid[256:], hashlib.sha256(valid[:256]).digest())
            self.assertNotEqual(valid[224:256], wrong[224:256])
            self.assertEqual(matched[128:224], c2.read_bytes()[256:352])
            self.assertNotEqual(matched[128:224], valid[128:224])
            self.assertEqual(wrong[256:], hashlib.sha256(wrong[:256]).digest())
            self.assertEqual(len((output / "short.g3r").read_bytes()), 287)
            self.assertEqual(len((output / "long.g3r").read_bytes()), 289)
            self.assertTrue((output / "link.g3r").is_symlink())
            self.assertTrue(stat.S_ISFIFO((output / "fifo.g3r").stat().st_mode))
            with self.assertRaises(FileExistsError):
                prepare(c2, output)


if __name__ == "__main__":
    unittest.main()
