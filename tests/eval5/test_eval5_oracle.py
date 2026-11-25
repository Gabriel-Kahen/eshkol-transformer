"""Independent stdlib oracle for the EVAL5 C driver."""
import hashlib
import math
import struct
import subprocess
import sys
import unittest
from pathlib import Path

DRIVER, DATA = (Path(arg).resolve() for arg in sys.argv[1:3])
BYTE_TRAIN = [b"abcabc", b"xy", b"abcabc", b"pqr", b"stu", b""]
BYTE_EVAL = [b"zzabcabcq", b"xy", b"abcabc", b"pqrstu", b"xyab", b"", b"xxx", b"stuabc"]
A, B = (1, 257), (2, 0)
C = (2**32 - 1, 0, 3, 5, 7, 8, 9, 7, 8, 9, 7, 8, 9, 4, 6)
D = (7, 8, 9, 7, 8, 9, 7, 8, 9, 4, 6, 1, 257, 0, 1)
E = C[:-1] + (5,)
TOKEN_TRAIN = [(b"collider-a", A), (b"long-train", C), (b"duplicate-ids", A)]
TOKEN_EVAL = [
    (b"collider-b", B),
    (b"collider-a-different-bytes", A),
    (b"near", E),
    (b"partial", D),
    (b"collider-a", B),
    (b"long-train", ()),
    (b"other", ()),
]


def f32(value):
    return struct.unpack(">f", struct.pack(">f", value))[0]


def bits(value):
    return struct.pack(">d", value).hex()


def nll(rows, targets):
    def row_loss(row, target):
        row = list(map(f32, row))
        largest = max(row)
        max_index = row.index(largest)
        tail = math.fsum(math.exp(value - largest)
                         for index, value in enumerate(row) if index != max_index)
        return largest - row[target] + math.log1p(tail)
    return math.fsum(row_loss(row, target) for row, target in zip(rows, targets, strict=True))


def audit(train, evaluation, order, min_overlap):
    def windows(units):
        return [units[i:i + order] for i in range(len(units) - order + 1)]

    identities = {raw for raw, _ in train}
    train_grams = {gram for _, units in train for gram in windows(units)}
    exact, overlaps = set(), set()
    total_windows = matched_windows = 0
    for index, (raw, units) in enumerate(evaluation):
        if raw in identities:
            exact.add(index)
        grams = windows(units)
        matches = sum(gram in train_grams for gram in grams)
        if matches and matches / len(grams) >= min_overlap:
            overlaps.add(index)
        total_windows += len(grams)
        matched_windows += matches
    contaminated = exact | overlaps
    return (
        len(evaluation), len(contaminated), len(exact), len(overlaps),
        total_windows, matched_windows, len(contaminated) / len(evaluation),
        sorted(contaminated),
    )


def seal(payload):
    return payload + b"manifest_sha256=" + hashlib.sha256(payload).hexdigest().encode("ascii") + b"\n"


class Eval5Oracle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = [line.split() for line in (DATA / "results").read_text().splitlines()]
        cls.manifest = (DATA / "benchmark.manifest").read_bytes()

    def read_status(self, path):
        return int(subprocess.check_output([DRIVER, "--read-status", path], text=True))

    def test_metric_parity(self):
        rows = [
            [0.1, -0.2, 0.3, -0.4],
            [-100, 100, 99, -100],
            [-40, 0, -50, -60],
            [-1, -1, -1, -1],
        ]
        targets = [2, 2, 1, 0]
        total = nll(rows, targets)
        mean = total / len(targets)
        lm, = [row[1:] for row in self.rows if row[0] == "lm"]
        self.assertEqual(tuple(map(int, lm[:2])), (4, 31))
        for actual, expected in zip(
            lm[2:], [total, mean, math.exp(mean), total / (math.log(2) * 31)], strict=True
        ):
            self.assertTrue(math.isclose(float.fromhex(actual), expected, rel_tol=2e-15),
                            (actual, expected))

        short = ([[0, 0]], [0])
        long = ([[0, 0.2], [0, 0.2]], [1, 1])
        choices = [([short, long], 1), ([short, short], 0), ([short, short], 1)]
        mc_rows = [row[1:] for row in self.rows if row[0] == "mc"]
        self.assertEqual([int(row[0]) for row in mc_rows], [0, 1])
        for mode, correct_count, total_count, accuracy in mc_rows:
            normalized = int(mode)
            winners = [
                min(range(len(options)), key=lambda i: (nll(*options[i]) /
                    (len(options[i][1]) if normalized else 1), i))
                for options, _ in choices
            ]
            self.assertEqual(winners, [normalized, 0, 0])
            correct = sum(winner == answer for winner, (_, answer) in zip(winners, choices, strict=True))
            self.assertEqual((int(correct_count), int(total_count)), (correct, len(choices)))
            self.assertEqual(float.fromhex(accuracy), correct / len(choices))

    def test_contamination_parity(self):
        audits = [row for row in self.rows if row[0] in {"byte", "token"}]
        self.assertEqual([(row[0], int(row[1])) for row in audits],
                         [("byte", 3)] + [("token", n) for n in (1, 2, 5, 8, 13, 2**32 - 1)])
        for mode, order, status, *report in audits:
            if mode == "byte":
                expected = audit([(raw, raw) for raw in BYTE_TRAIN],
                                 [(raw, raw) for raw in BYTE_EVAL], int(order), 0.5)
            else:
                expected = audit(TOKEN_TRAIN, TOKEN_EVAL, int(order), 0)
            self.assertEqual(int(status), 0)
            actual = (*map(int, report[:6]), float.fromhex(report[6]), list(map(int, report[7:])))
            self.assertEqual(actual, expected, (mode, order))
        self.assertEqual(A[0] * 257 + A[1], B[0] * 257 + B[1])
        self.assertNotEqual(A, B)

    def test_canonical_manifest(self):
        lm, = [row[1:] for row in self.rows if row[0] == "lm"]
        mc, = [row[2:] for row in self.rows if row[:2] == ["mc", "1"]]
        expected = [
            ("version", "1.0"),
            ("suite", "fixture-suite"),
            ("task", "lm-and-mc"),
            ("split", "validation.v1"),
            ("train_dataset_sha256", bytes(range(32)).hex()),
            ("eval_dataset_sha256", bytes(range(0x20, 0x40)).hex()),
            ("tokenizer_sha256", bytes(range(0x40, 0x60)).hex()),
            ("checkpoint_sha256", bytes(range(0x60, 0x80)).hex()),
            ("context_length", 2048),
            ("max_samples", 100),
            ("contamination.ngram_mode", 1),
            ("contamination.ngram_order", 3),
            ("contamination.forbid_exact_match", 0),
            ("contamination.min_ngram_overlap", bits(0.5)),
            ("contamination.max_allowed_rate", bits(0.75)),
            ("metric_mask", 3),
            ("mc_normalize_by_length", 1),
            ("lm.token_count", 4),
            ("lm.byte_count", 31),
            *[(f"lm.{name}", bits(float.fromhex(value))) for name, value in zip(
                ("total_nll", "mean_nll", "perplexity", "bits_per_byte"), lm[2:], strict=True)],
            ("mc.correct_count", int(mc[0])),
            ("mc.total_count", int(mc[1])),
            ("mc.accuracy", bits(float.fromhex(mc[2]))),
            ("audit.eval_documents", 8),
            ("audit.contaminated_documents", 6),
            ("audit.exact_doc_matches", 3),
            ("audit.ngram_contaminated_docs", 4),
            ("audit.total_eval_ngrams", 22),
            ("audit.matched_eval_ngrams", 12),
            ("audit.contamination_rate", bits(0.75)),
        ]
        payload = "EVAL5_MANIFEST_V1\n" + "".join(f"{key}={value}\n" for key, value in expected)
        self.assertEqual(self.manifest, seal(payload.encode("ascii")))
        self.assertEqual(self.read_status(DATA / "benchmark.manifest"), 0)

    def test_resealed_invalid_manifests(self):
        payload, _, _ = self.manifest.rpartition(b"manifest_sha256=")
        cases = [
            (b"version=1.0", b"version=1.1", -10),
            (b"task=lm-and-mc\n", b"", -8),
            (b"task=lm-and-mc\n", b"task=lm-and-mc\ntask=lm-and-mc\n", -8),
            (b"task=lm-and-mc\n", b"unknown=lm-and-mc\n", -8),
            (b"task=lm-and-mc\nsplit=validation.v1\n",
             b"split=validation.v1\ntask=lm-and-mc\n", -8),
            (b"split=validation.v1", b"split=x\x00y", -8),
            (b"split=validation.v1", b"split=" + b"a" * 128, -8),
            (b"split=validation.v1", b"split=validation.v1\r", -8),
            (b"split=validation.v1", b"split=x/y", -8),
            (b"context_length=2048", b"context_length=02048", -8),
            (b"context_length=2048", b"context_length=+2048", -8),
            (b"context_length=2048", b"context_length=18446744073709551616", -8),
            (b"contamination.ngram_order=3", b"contamination.ngram_order=4294967296", -8),
            (b"train_dataset_sha256=000102", b"train_dataset_sha256=00102", -8),
            (b"train_dataset_sha256=000102", b"train_dataset_sha256=F00102", -8),
            (b"min_ngram_overlap=3fe0000000000000", b"min_ngram_overlap=7ff8000000000000", -8),
            (b"min_ngram_overlap=3fe0000000000000", b"min_ngram_overlap=7ff0000000000000", -8),
            (b"min_ngram_overlap=3fe0000000000000", b"min_ngram_overlap=8000000000000000", -8),
            (b"audit.matched_eval_ngrams=12", b"audit.matched_eval_ngrams=23", -8),
            (b"audit.contamination_rate=3fe8000000000000", b"audit.contamination_rate=0000000000000000", -8),
        ]
        path = DATA / "mutant.manifest"
        try:
            for before, after, expected in cases:
                with self.subTest(mutation=after):
                    self.assertEqual(payload.count(before), 1)
                    path.write_bytes(seal(payload.replace(before, after)))
                    self.assertEqual(self.read_status(path), expected)
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)
