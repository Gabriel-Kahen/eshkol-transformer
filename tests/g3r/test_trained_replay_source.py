"""Focused source guard for the exact codec rejection in replay setup."""
from pathlib import Path
import unittest


SOURCE = Path(__file__).resolve().parents[2] / "native/g3r_trained_replay_compare_extension.esk"


def exact_negative_guard(source: str) -> None:
    body = source.split("(define (g3r-trained-replay-wrong-digest-rejects?", 1)[1]
    body = body.split("(define (g3r-trained-replay-compare!", 1)[0]
    required = (
        '(string? caught)',
        '(string=? caught',
        '"G3-R record decode: header, pair or checksum"',
        '(g3r-record-decode! readback wrong destination)',
        "(eq? (vector-ref destination i) 'untouched)",
    )
    if any(item not in body for item in required):
        raise AssertionError("wrong-digest guard lost exact rejection or unchanged destination")


class TrainedReplaySource(unittest.TestCase):
    def test_wrong_digest_exact_rejection(self):
        source = SOURCE.read_text()
        exact_negative_guard(source)
        for before, after in (
            ('(string? caught)', '#t'),
            ('(string=? caught', '(eq? caught'),
            ('"G3-R record decode: header, pair or checksum"', '"wrong error"'),
            ("(eq? (vector-ref destination i) 'untouched)", '#t'),
        ):
            with self.subTest(mutation=before), self.assertRaises(AssertionError):
                exact_negative_guard(source.replace(before, after, 1))


if __name__ == "__main__":
    unittest.main()
