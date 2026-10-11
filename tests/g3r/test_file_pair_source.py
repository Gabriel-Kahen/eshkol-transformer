"""Source contract and native exact-read checks for the private G3-R pair leaf."""
import ctypes
from hashlib import sha256
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

from tests.c2.test_handler_order import definition, parse, render
from tests.g3r.prepare_file_pair_fixtures import record

ROOT = Path(__file__).resolve().parents[2]
EXTENSION = ROOT / "native/g3r_file_pair_extension.esk"
ROOT_FILE = ROOT / "native/g3r_file_pair_root.esk"
FIXTURE = ROOT / "tests/g3r/file_pair_test.esk"
INCLUDE_DIRS = (
    "internal/p1/lib", "internal/c1/lib", "internal/t2/lib",
    "internal/t1/lib", "internal/d2/lib", "src", "lib", "native",
)
LOAD = re.compile(r'^\s*\(load\s+"([^"]+)"\)', re.MULTILINE)


def transitive_loads(path):
    """Resolve the first matching include path, as the private receipt gate does."""
    resolved = []

    def visit(current, parents):
        if current in parents:
            raise ValueError(f"cyclic source load: {current}")
        for name in LOAD.findall(current.read_text()):
            dependency = next((ROOT / directory / name for directory in INCLUDE_DIRS
                               if (ROOT / directory / name).is_file()), None)
            if dependency is None:
                raise ValueError(f"missing source load: {name}")
            visit(dependency, parents + (current,))
        resolved.append(current.relative_to(ROOT).as_posix())

    visit(path, ())
    return resolved


class FilePairSourceTest(unittest.TestCase):
    def test_transitive_roots_load_codec_once(self):
        for name in ("g3r_file_pair_root.esk", "g3r_pair_rng_bridge_root.esk"):
            with self.subTest(root=name):
                loads = transitive_loads(ROOT / "native" / name)
                self.assertEqual(len(loads), len(set(loads)))
                self.assertEqual(loads.count("native/g3r_record_codec_extension.esk"), 1)
                self.assertLess(loads.index("native/g3r_record_codec_extension.esk"),
                                loads.index("native/g3r_file_pair_extension.esk"))
                self.assertIn("native/g3r_c2_pair_decode_extension.esk", loads)

    def test_composition_and_syntax(self):
        loads = parse(ROOT_FILE.read_text())
        self.assertEqual(loads, [
            ["load", '"g3r_c2_sha_receipt_root.esk"'],
            ["load", '"g3r_file_pair_extension.esk"'],
        ])
        parse(EXTENSION.read_text())
        parse(FIXTURE.read_text())

    def test_authority_cleanup_and_atomic_publication(self):
        source = EXTENSION.read_text()
        admit = render(definition({"pair": parse(source)},
                                  "g3r-file-pair-admit-internal"))
        ordered = [
            "(g3r-c2-load-with-sha-internal c2-path policy report)",
            "(g3r-c2-receipt-borrow-begin-internal receipt (quote checkpoint-load))",
            "(g3r-c2-receipt-borrow-sha-internal borrow (quote checkpoint-load))",
            "(g3r-file-pair-read-record-internal companion-path)",
            "(g3r-record-decode! record digest fields)",
            "(c2-training-state-borrow-projection-internal c2-borrow)",
            "(g3r-c2-receipt-borrow-end-internal! borrow)",
            "(vector-copy! destination 0 staged 0 9)",
        ]
        positions = [admit.index(text) for text in ordered]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("(g3r-file-pair-cleanup! receipt borrow)", admit)
        self.assertIn("(g3r-c2-sha-fail-stop 134)", admit)
        self.assertEqual(admit.count("(vector-copy! destination"), 1)
        self.assertNotIn("c2-checkpoint-load-stage-internal", source)
        self.assertNotIn("c1-sha256", source)

    def test_independent_record_negative_vectors(self):
        digest = sha256(b"authoritative second C2 image").digest()
        good = record(digest)
        self.assertEqual(len(good), 288)
        self.assertEqual(good[224:256], digest)
        self.assertEqual(good[256:288], sha256(good[:256]).digest())
        self.assertEqual(len(good[:-1]), 287)
        self.assertEqual(len(good + b"\0"), 289)
        torn = bytearray(good)
        torn[120] ^= 1
        self.assertNotEqual(torn[256:288], sha256(torn[:256]).digest())
        other = record(bytes(32))
        self.assertNotEqual(other[224:256], digest)
        wrong_tokenizer = record(digest, b"a")
        self.assertEqual(wrong_tokenizer[224:256], digest)
        self.assertNotEqual(wrong_tokenizer[128:224], good[128:224])
        fixture = FIXTURE.read_text()
        for label in ("orphan C2 image", "orphan companion", "short companion",
                      "extended companion", "torn companion checksum",
                      "other C2 digest", "record tokenizer differs from loaded C2",
                      "post-receipt fault releases owner",
                      "prepublication fault releases owner",
                      "failed whole-vector promotion"):
            self.assertIn(label, fixture)

    def test_existing_native_exact_reader_preserves_failed_destination(self):
        cc = shutil.which("cc")
        if not cc:
            self.skipTest("C compiler unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            library = root / "checkpoint_io.so"
            subprocess.run([cc, "-std=c11", "-Wall", "-Wextra", "-Werror",
                            "-fPIC", "-shared", "-I", str(ROOT / "native"),
                            str(ROOT / "native/checkpoint_io.c"), "-o", str(library)],
                           check=True, capture_output=True, text=True)
            read = ctypes.CDLL(str(library)).et_checkpoint_io_read_exact_v1
            read.argtypes = [ctypes.c_char_p, ctypes.c_void_p,
                             ctypes.c_uint64, ctypes.c_uint64]
            read.restype = ctypes.c_int64

            class Buffer(ctypes.Structure):
                _fields_ = [("length", ctypes.c_int64),
                            ("bytes", ctypes.c_ubyte * 288)]

            dest = Buffer(288, (ctypes.c_ubyte * 288)(*([0x5a] * 288)))
            good = root / "good.g3r"
            good.write_bytes(record(bytes(32)))
            for name, payload in (("short", good.read_bytes()[:-1]),
                                  ("long", good.read_bytes() + b"\0")):
                path = root / f"{name}.g3r"
                path.write_bytes(payload)
                self.assertNotEqual(read(str(path).encode(), ctypes.byref(dest),
                                         288, 288), 0)
                self.assertEqual(bytes(dest.bytes), b"\x5a" * 288)
            missing = root / "missing.g3r"
            self.assertNotEqual(read(str(missing).encode(), ctypes.byref(dest),
                                     288, 288), 0)
            self.assertEqual(bytes(dest.bytes), b"\x5a" * 288)
            link = root / "link.g3r"
            link.symlink_to(good)
            self.assertNotEqual(read(str(link).encode(), ctypes.byref(dest),
                                     288, 288), 0)
            self.assertEqual(bytes(dest.bytes), b"\x5a" * 288)
            self.assertEqual(read(str(good).encode(), ctypes.byref(dest),
                                  288, 288), 0)
            self.assertEqual(bytes(dest.bytes), good.read_bytes())


if __name__ == "__main__":
    unittest.main()
